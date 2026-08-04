// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 static model — processing methods.
//!
//! ## Block Size Contract
//!
//! Any input size ≤ `max_buffer_size` is safe: processing is internally chunked
//! into sub-blocks of ≤ `WAVENET_MAX_NUM_FRAMES` (64), matching the kernel scratch
//! buffer capacity. Exceeding `max_buffer_size` causes silent truncation: only the
//! first `max_buffer_size` frames are processed and the remaining are left as zeros.
//! This matches the audio host contract which guarantees
//! `block_size <= max_block_size` negotiated at activation.
//!
//! The `dispatch_simd!` macro evaluates `M::ISA` (compile-time constant
//! through the `SimdMath` trait) to monomorphize the full forward pass,
//! eliminating per-frame `is_x86_feature_detected` branches and the dynamic
//! read of `SIMD_MATH` in the inner loop.

use crate::math::common::SimdMath;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

use super::super::super::params::{A2_HEAD_KERNEL_SIZE, A2_NUM_LAYERS};
use super::WaveNetA2;

impl<const CH: usize> WaveNetA2<CH> {
    /// Full forward pass through the A2 model.
    ///
    /// Processes `input` samples and writes to `output`.
    /// Requires layers to be populated via `set_weights`.
    /// Outputs silence until weights are loaded.
    ///
    /// ## Ring buffer architecture
    ///
    /// Layer history uses `MirroredBuffer<f32>` with power-of-2 sizes.
    /// Writes go to unmasked positions in the 2× virtual mapping; reads are
    /// branchless because the mirror maps `[S, 2S)` → `[0, S)`. When
    /// `buffer_start` approaches the 2× boundary, it rewinds by subtracting
    /// `ring_size`. No `copy_within` / memmove on the hot path.
    ///
    /// Head accumulator uses a plain `AlignedVec` with pow2 mask (`& ring_mask`).
    /// A pre-write memmove preserves `K-1` tail samples when the ring is about
    /// to overflow, keeping the write-positions unmasked for vectorized stores.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::common::dispatch_simd!(self, process_internal, input, output);
        }
    }

    /// Monomorphized inner loop — see [`process`](Self::process) for contract.
    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total = input.len();
        if total == 0 {
            return;
        }

        output[..total].fill(0.0);

        if self.layers.is_empty() {
            self.head_write_pos = (self.head_write_pos + total) & self.head_ring_mask;
            return;
        }

        debug_assert!(
            total <= self.max_buffer_size,
            "process: input ({total}) > max_buffer_size ({}) — host violated block-size contract",
            self.max_buffer_size
        );
        let nf_total = total.min(self.max_buffer_size);

        let mut pos = 0;
        while pos < nf_total {
            let nf = (nf_total - pos).min(WAVENET_MAX_NUM_FRAMES);

            self.rechannel_prescale(input, pos, nf);
            let head_wp = self.advance_head_ring(nf);

            for li in 0..A2_NUM_LAYERS {
                unsafe {
                    self.layer_forward_dispatch::<M>(li, nf, input, pos, head_wp);
                }
            }

            self.head_finalize(head_wp, nf, &mut output[pos..pos + nf]);
            pos += nf;
        }
    }

    /// Phase 0: input rechannel pre-scaling — `input × rechannel_w_f32 → layer_in`.
    #[inline(always)]
    fn rechannel_prescale(&mut self, input: &[f32], pos: usize, nf: usize) {
        if CH == 8 {
            use core::arch::x86_64::{
                _mm256_load_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_store_ps,
            };
            unsafe {
                let rw_vec = _mm256_load_ps(self.rechannel_w_f32.as_ptr());
                for (f, &x) in input[pos..pos + nf].iter().enumerate() {
                    let x_vec = _mm256_set1_ps(x);
                    let res = _mm256_mul_ps(rw_vec, x_vec);
                    _mm256_store_ps(self.layer_in.as_mut_ptr().add(f * 8), res);
                }
            }
        } else {
            for (f, &x) in input[pos..pos + nf].iter().enumerate() {
                let base = f * CH;
                for c in 0..CH {
                    self.layer_in[base + c] = self.rechannel_w_f32[c] * x;
                }
            }
        }
    }

    /// Advances the head accumulator ring buffer.
    ///
    /// When the write cursor plus `nf` would overflow the ring capacity,
    /// the tail `K-1` samples are memmove'd to the start and the write
    /// position wraps around. Returns the (possibly wrapped) write position
    /// for use by the layer loop.
    #[inline(always)]
    fn advance_head_ring(&mut self, nf: usize) -> usize {
        let head_keep = A2_HEAD_KERNEL_SIZE - 1;
        let head_cap = self.head_ring_mask + 1;
        if self.head_write_pos + nf > head_cap {
            let keep_start = self.head_write_pos - head_keep;
            let keep_bytes = head_keep * CH;
            let src = keep_start * CH;
            self.head_accum.copy_within(src..src + keep_bytes, 0);
            self.head_write_pos = head_keep;
        }
        self.head_write_pos
    }

    /// Per-layer forward dispatch for a single layer index.
    ///
    /// # Safety
    ///
    /// Caller must ensure `li < A2_NUM_LAYERS` and that `nf` frames of valid
    /// data are available at `input[pos..pos+nf]`. Internal conv/film/head
    /// accesses assume caller-verified buffer capacities.
    #[inline(always)]
    unsafe fn layer_forward_dispatch<M: SimdMath>(
        &mut self,
        li: usize,
        nf: usize,
        input: &[f32],
        pos: usize,
        head_wp: usize,
    ) {
        let is_first = li == 0;
        let is_last = li == A2_NUM_LAYERS - 1;
        let ch = CH;
        let ring_size = self.layer_ring_sizes[li];
        let lookback = self.layer_lookbacks[li];
        let max_lookback_cols = lookback / ch;
        let bs = self.layer_buffer_starts[li];

        debug_assert!(bs >= lookback);
        debug_assert!(bs + nf * ch <= ring_size * 2);

        // Copy layer_in → history buffer, then apply conv_pre_film.
        {
            let buf = &mut self.layer_buffers[li];
            buf[bs..bs + nf * ch].copy_from_slice(&self.layer_in[..nf * ch]);
            for f in 0..nf {
                if let Some(ref mut film) = self.layers[li].conv_pre_film {
                    unsafe {
                        film.process(
                            &mut buf[bs + f * ch..bs + (f + 1) * ch],
                            &input[pos + f..pos + f + 1],
                        );
                    }
                }
            }
        }

        if bs + nf * ch + self.max_buffer_size * ch > ring_size * 2 {
            self.layer_buffer_starts[li] = bs + nf * ch - ring_size;
        } else {
            self.layer_buffer_starts[li] = bs + nf * ch;
        }

        {
            let history = &self.layer_buffers[li][bs - lookback..bs + nf * ch];
            let layer = &mut self.layers[li];

            let conv_ch = layer.conv_ch.as_ref();
            let mixin_w = &layer.mixin_w;
            let l1x1_w = &layer.l1x1_w;
            let l1x1_b = &layer.l1x1_b;

            let mut film_block = super::super::super::film::FilmBlock {
                conv_pre_film: layer.conv_pre_film.as_mut(),
                conv_post_film: layer.conv_post_film.as_mut(),
                input_mixin_pre_film: layer.input_mixin_pre_film.as_mut(),
                input_mixin_post_film: layer.input_mixin_post_film.as_mut(),
                activation_pre_film: layer.activation_pre_film.as_mut(),
                activation_post_film: layer.activation_post_film.as_mut(),
                layer1x1_post_film: layer.layer1x1_post_film.as_mut(),
                head1x1_post_film: layer.head1x1_post_film.as_mut(),
            };

            if let Some(conv_ch) = conv_ch {
                match conv_ch {
                    super::super::super::layer::A2ConvCh::Ch3(ch3_conv) => unsafe {
                        super::super::super::conv1d_ch3::layer_forward_ch3_block(
                            ch3_conv,
                            mixin_w,
                            l1x1_w,
                            l1x1_b,
                            &mut film_block,
                            false,
                            history,
                            max_lookback_cols,
                            nf,
                            &input[pos..pos + nf],
                            &mut self.head_accum,
                            head_wp,
                            &mut self.layer_in,
                            is_first,
                            is_last,
                        );
                    },
                    super::super::super::layer::A2ConvCh::Ch8(ch8_conv) => unsafe {
                        match M::ISA {
                            crate::math::common::InstructionSet::Avx512
                            | crate::math::common::InstructionSet::Avx512VnniBf16 => {
                                super::super::super::conv1d_ch8::layer_forward_ch8_block_simdmath::<
                                    M,
                                >(
                                    ch8_conv,
                                    mixin_w,
                                    l1x1_w,
                                    l1x1_b,
                                    &mut film_block,
                                    false,
                                    history,
                                    max_lookback_cols,
                                    nf,
                                    &input[pos..pos + nf],
                                    &mut self.head_accum,
                                    head_wp,
                                    &mut self.layer_in,
                                    is_first,
                                    is_last,
                                );
                            }
                            _ => {
                                super::super::super::conv1d_ch8::layer_forward_ch8_block(
                                    ch8_conv,
                                    mixin_w,
                                    l1x1_w,
                                    l1x1_b,
                                    &mut film_block,
                                    false,
                                    history,
                                    max_lookback_cols,
                                    nf,
                                    &input[pos..pos + nf],
                                    &mut self.head_accum,
                                    head_wp,
                                    &mut self.layer_in,
                                    is_first,
                                    is_last,
                                );
                            }
                        }
                    },
                }
                return;
            }
            // Fallback: per-frame path using the generic A2Conv1d enum.
            #[cfg(any(test, feature = "dynamic-engine"))]
            {
                use super::super::super::params::A2_LEAKY_SLOPE;

                debug_assert!(self.z_scratch.len() >= ch);
                for f in 0..nf {
                    let frame_idx = max_lookback_cols + f;

                    // 1. Dilated conv → z_buf.
                    unsafe {
                        layer.conv.process_single_frame::<M>(
                            history,
                            &mut self.z_scratch[..ch],
                            frame_idx,
                            None,
                        );
                    }

                    // 1b. FiLM post-conv.
                    let cond = &input[pos + f..pos + f + 1];
                    if let Some(ref mut film) = film_block.conv_post_film {
                        unsafe {
                            film.process(&mut self.z_scratch[..ch], cond);
                        }
                    }

                    // 2. Input mixin — input_mixin_pre_film applied to condition (self-modulation).
                    let cond_val = input[pos + f];
                    let cond_for_mixin = if let Some(ref mut film) = film_block.input_mixin_pre_film
                    {
                        let mut modulated = cond_val;
                        let orig = cond_val;
                        unsafe {
                            film.process(
                                core::slice::from_mut(&mut modulated),
                                core::slice::from_ref(&orig),
                            );
                        }
                        modulated
                    } else {
                        cond_val
                    };
                    for c in 0..ch {
                        self.z_scratch[c] += mixin_w[c] * cond_for_mixin;
                    }

                    // 2b. FiLM post-mixin.
                    if let Some(ref mut film) = film_block.input_mixin_post_film {
                        unsafe {
                            film.process(&mut self.z_scratch[..ch], cond);
                        }
                    }
                    if let Some(ref mut film) = film_block.activation_pre_film {
                        unsafe {
                            film.process(&mut self.z_scratch[..ch], cond);
                        }
                    }

                    // 3. LeakyReLU.
                    for z in self.z_scratch.iter_mut().take(ch) {
                        if *z < 0.0 {
                            *z *= A2_LEAKY_SLOPE;
                        }
                    }

                    // 3b. FiLM post-activation.
                    if let Some(ref mut film) = film_block.activation_post_film {
                        unsafe {
                            film.process(&mut self.z_scratch[..ch], cond);
                        }
                    }

                    // 4. Head accumulator.
                    let head_off = (head_wp + f) * ch;
                    if is_first {
                        self.head_accum[head_off..head_off + ch]
                            .copy_from_slice(&self.z_scratch[..ch]);
                    } else {
                        for (c, z) in self.z_scratch.iter().enumerate().take(ch) {
                            self.head_accum[head_off + c] += *z;
                        }
                    }

                    // 5. L1x1 residual.
                    if !is_last {
                        let base = f * ch;
                        for c in 0..ch {
                            let mut sum = l1x1_b[c];
                            for u in 0..ch {
                                sum += l1x1_w[u * ch + c] * self.z_scratch[u];
                            }
                            self.layer_in[base + c] += sum;
                        }
                        if let Some(ref mut film) = film_block.layer1x1_post_film {
                            unsafe {
                                film.process(&mut self.layer_in[base..base + ch], cond);
                            }
                        }
                    }
                }
            }
            #[cfg(not(any(test, feature = "dynamic-engine")))]
            {
                // RT-safe fallback: this branch is unreachable per set_weights
                // invariant (A2 layers always have CH=3 or CH=8 conv). Retained
                // as a panic-free guard for future model format changes — never
                // panics on the audio thread.
                if let Some(ref rt) = self.rt_status {
                    rt.set_flag(crate::common::spsc::RT_STATUS_A2_FALLBACK_TRIGGERED);
                }
                debug_assert!(
                    false,
                    "A2 layers always have ch3 or ch8 conv; \
                     scalar fallback triggered — silencing layer output"
                );
                self.z_scratch[..ch].fill(0.0);
                for f in 0..nf {
                    let head_off = (head_wp + f) * ch;
                    if is_first {
                        self.head_accum[head_off..head_off + ch]
                            .copy_from_slice(&self.z_scratch[..ch]);
                    }
                    if !is_last {
                        let base = f * ch;
                        for c in 0..ch {
                            let mut sum = l1x1_b[c];
                            for u in 0..ch {
                                sum += l1x1_w[u * ch + c] * self.z_scratch[u];
                            }
                            self.layer_in[base + c] += sum;
                        }
                        if let Some(ref mut film) = film_block.layer1x1_post_film {
                            unsafe {
                                film.process(
                                    &mut self.layer_in[base..base + ch],
                                    &input[pos + f..pos + f + 1],
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Finalizes the head convolution and advances the head write position.
    #[inline(always)]
    fn head_finalize(&mut self, head_wp: usize, nf: usize, output: &mut [f32]) {
        self.head_write_pos = (head_wp + nf) & self.head_ring_mask;

        if let Some(ref head) = self.head_conv {
            head.process(
                &self.head_accum,
                self.head_write_pos,
                self.head_ring_mask,
                nf,
                output,
            );
        }
    }
}
