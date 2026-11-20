// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 Dynamic model — processing methods.
//!
//! ## Architecture
//!
//! 1. Input rechannel: `Conv1x1(1 → channels)` (bias, no activation)
//! 2. Per-layer (per-frame):
//!    - Dilated causal conv: `channels → bottleneck` (or `2*bottleneck` if gating/blending)
//!    - FiLM post-conv (optional)
//!    - Input mixin: `+ mixin_w[c] * input_cond`
//!    - FiLM post-mixin (optional)
//!    - Activation (heterogeneous) or Gating/Blending
//!    - FiLM post-activation (optional)
//!    - Head accumulator: direct or via head1x1 projection `bottleneck → channels`
//!    - L1x1 residual: `bottleneck → channels` added to `layer_in` (skip last layer)
//!    - FiLM post-l1x1 (optional)
//! 3. Head conv: `Conv1D(channels → 1, K=16, bias)` × head_scale
//!
//! ## Ring buffer architecture
//!
//! Same MirroredBuffer + pow2 head ring as `WaveNetA2<CH>`. Per-layer history
//! stores `channels`-wide data. The dilated conv reads `channels`-wide history
//! and produces `bottleneck` (or `2*bottleneck`) outputs.
//!
//! ## RT-Safety
//!
//! All scratch buffers (z_scratch, gating_scratch) and gating/blending configs
//! are pre-allocated at construction time. Zero heap alloc on the hot-path.

use crate::math::common::SimdMath;
use crate::models::NamModel;
use crate::models::a2::gating::GatingMode;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

use super::process_frame::process_frame_dyn;

use crate::arch::simd::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps};

#[cfg(any(test, feature = "testing"))]
use crate::math::common::AlignedVec;
#[cfg(any(test, feature = "testing"))]
use crate::testing::diagnostics::{ConditionDspSnapshot, HeadPerLayerSnapshot};

use super::WaveNetA2Dyn;

impl WaveNetA2Dyn {
    /// Full forward pass through the dynamic A2 model.
    ///
    /// Uses per-frame processing with the polymorphic `A2Conv1d::process_single_frame`
    /// for maximum flexibility. Each layer applies activation or gating/blending
    /// according to its per-layer config.
    ///
    /// # Block Size Contract
    /// Any input size is processed in full: processing is internally chunked
    /// into sub-blocks of ≤ `WAVENET_MAX_NUM_FRAMES` (64).
    ///
    /// **SIMD Dispatch:** The `dispatch_simd!` macro evaluates the hardware once
    /// and monomorphizes `process_internal` to the detected ISA (AVX2/AVX-512),
    /// eliminating per-frame `is_x86_feature_detected` branches.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // SAFETY: `dispatch_simd!` dispatches on runtime CPUID feature checks to a
        // matching `#[target_feature]` backend; `input`/`output` are the same valid
        // slices passed to this safe wrapper.
        unsafe {
            crate::math::common::dispatch_simd!(self, process_internal, input, output);
        }
    }

    /// Monomorphized inner loop — see [`process`](Self::process) for contract.
    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        // `head_size` interleaved channels are written per frame; the frame
        // count is bounded by both the input length and the output capacity in
        // frames.
        let in_ch = self.input_channels.max(1);
        let out_ch = self.head_size.max(1);
        let total = (input.len() / in_ch).min(output.len() / out_ch);
        if total == 0 {
            return;
        }

        output[..total * out_ch].fill(0.0);

        if self.layers.is_empty() {
            self.head_write_pos = (self.head_write_pos + total) & self.head_ring_mask;
            return;
        }

        #[cfg(any(test, feature = "testing"))]
        {
            if let Some(ref mut dump) = self.diag.dump {
                dump.total_frames = total;
            }
        }

        let cond_size = self.condition_size;

        let mut pos = 0;
        while pos < total {
            let nf = (total - pos).min(WAVENET_MAX_NUM_FRAMES);

            // Pre-process input through condition_dsp if present.
            // The condition_dsp output replaces the raw input as the parameter
            // for per-layer mixin and FiLM (C++ _process_condition pattern).
            if let Some(cond_dsp) = self.condition_dsp.as_mut() {
                cond_dsp.process(
                    &input[pos * in_ch..(pos + nf) * in_ch],
                    &mut self.condition_dsp_output[0..nf * cond_size],
                );
                let dsp_ch = cond_dsp.num_output_channels();
                if dsp_ch > 0 && dsp_ch < cond_size {
                    let buf = &mut self.condition_dsp_output[0..nf * cond_size];
                    if dsp_ch == 1 {
                        for f in (0..nf).rev() {
                            let val = buf[f];
                            for c in 0..cond_size {
                                buf[f * cond_size + c] = val;
                            }
                        }
                    } else {
                        for f in (0..nf).rev() {
                            for c in (0..dsp_ch).rev() {
                                buf[f * cond_size + c] = buf[f * dsp_ch + c];
                            }
                            for c in dsp_ch..cond_size {
                                buf[f * cond_size + c] = buf[f * cond_size + (c % dsp_ch)];
                            }
                        }
                    }
                }
            }
            let use_cond_dsp = self.condition_dsp.is_some();

            #[cfg(any(test, feature = "testing"))]
            {
                if let Some(ref mut dump) = self.diag.dump
                    && use_cond_dsp
                    && dump.total_frames > 0
                    && self.diag.config.capture_condition_dsp
                {
                    let d = &self.condition_dsp_output[..nf * cond_size];
                    let mut snap = AlignedVec::new(d.len(), 0.0f32).expect("diagnostic alloc");
                    snap.copy_from_slice(d);
                    dump.condition_dsp_snapshots.push(ConditionDspSnapshot {
                        channels: cond_size,
                        num_frames: nf,
                        data: snap,
                    });
                }
            }

            self.rechannel_prescale(input, pos, nf);
            let head_wp = self.advance_head_ring(nf);

            for li in 0..self.num_layers {
                self.layer_forward_dispatch::<M>(
                    li,
                    nf,
                    input,
                    pos,
                    head_wp,
                    use_cond_dsp,
                    cond_size,
                    true,
                    true,
                );

                #[cfg(any(test, feature = "testing"))]
                {
                    if let Some(ref mut dump) = self.diag.dump
                        && dump.total_frames > 0
                        && self.diag.config.capture_head_per_layer
                    {
                        let accum_size = self.head_accum_size;
                        let region_len = nf * accum_size;
                        let start = head_wp * accum_size;
                        let mut data =
                            AlignedVec::new(region_len, 0.0f32).expect("diagnostic alloc");
                        let end = start + region_len;
                        if end <= self.head_accum.len() {
                            data.copy_from_slice(&self.head_accum[start..end]);
                        } else {
                            let head_cap = self.head_ring_mask + 1;
                            let first_part = head_cap * accum_size - start;
                            data[..first_part].copy_from_slice(&self.head_accum[start..]);
                            data[first_part..]
                                .copy_from_slice(&self.head_accum[..region_len - first_part]);
                        }
                        dump.head_per_layer_snapshots.push(HeadPerLayerSnapshot {
                            layer: li,
                            accum_size,
                            num_frames: nf,
                            head_wp,
                            data,
                        });
                    }
                }
            }

            self.head_finalize(head_wp, nf, &mut output[pos * out_ch..(pos + nf) * out_ch]);
            pos += nf;
        }

        #[cfg(any(test, feature = "testing"))]
        {
            if let Some(ref mut dump) = self.diag.dump
                && dump.total_frames > 0
                && self.diag.config.capture_final_output
            {
                let mut out = AlignedVec::new(total, 0.0f32).expect("diagnostic alloc");
                out.copy_from_slice(&output[..total]);
                dump.final_output = Some(out);
            }
        }
    }

    /// Phase 0: rechannel pre-scaling — `input × rechannel_w_f32 → layer_in`.
    /// For mono input (input_channels == 1): `layer_in[c] = rechannel_w_f32[c] * x`.
    /// For multi-channel input (input_channels > 1): matrix multiply per frame.
    #[inline(always)]
    pub(crate) fn rechannel_prescale(&mut self, input: &[f32], pos: usize, nf: usize) {
        let channels = self.channels;
        let in_ch = self.input_channels;
        if in_ch == 1 {
            if channels >= 8 {
                let channels_aligned = channels & !7;
                for (f, &x) in input[pos..pos + nf].iter().enumerate() {
                    let base = f * channels;
                    // SAFETY: `channels_aligned = channels & !7` is a multiple of 8, so
                    // at each `while c < channels_aligned` iteration `c + 8 <= channels_aligned`;
                    // the 8-lane `loadu`/`storeu` at offsets `c` and `base + c` stay within
                    // `rechannel_w_f32` (len `channels`) and `layer_in` (`base = f*channels`,
                    // capacity ≥ `nf*channels`); `loadu`/`storeu` need no alignment.
                    unsafe {
                        let x_vec = _mm256_set1_ps(x);
                        let mut c = 0;
                        while c < channels_aligned {
                            let rw = _mm256_loadu_ps(self.rechannel_w_f32.as_ptr().add(c));
                            _mm256_storeu_ps(
                                self.layer_in.as_mut_ptr().add(base + c),
                                _mm256_mul_ps(rw, x_vec),
                            );
                            c += 8;
                        }
                        for c in channels_aligned..channels {
                            self.layer_in[base + c] = self.rechannel_w_f32[c] * x;
                        }
                    }
                }
            } else {
                for (f, &x) in input[pos..pos + nf].iter().enumerate() {
                    let base = f * channels;
                    for c in 0..channels {
                        self.layer_in[base + c] = self.rechannel_w_f32[c] * x;
                    }
                }
            }
        } else {
            for f in 0..nf {
                let base = f * channels;
                let in_base = (pos + f) * in_ch;
                for c in 0..channels {
                    let mut sum = 0.0f32;
                    for ic in 0..in_ch {
                        sum += input[in_base + ic] * self.rechannel_w_f32[c * in_ch + ic];
                    }
                    self.layer_in[base + c] = sum;
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
    pub(crate) fn advance_head_ring(&mut self, nf: usize) -> usize {
        let head_keep = self.head_kernel_size.saturating_sub(1);
        let head_cap = self.head_ring_mask + 1;
        if self.head_write_pos + nf > head_cap {
            let keep_start = self.head_write_pos - head_keep;
            let keep_bytes = head_keep * self.head_accum_size;
            let src = keep_start * self.head_accum_size;
            self.head_accum.copy_within(src..src + keep_bytes, 0);
            self.head_write_pos = head_keep;
        }
        self.head_write_pos
    }

    /// Per-layer forward dispatch for a single layer index.
    ///
    /// # Safety
    ///
    /// Caller must ensure `li < self.num_layers` and that `nf` frames of valid
    /// data are available at `input[pos..pos+nf]`. Internal conv/film/head
    /// accesses assume caller-verified buffer capacities.
    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "A2 dynamic model process function requiring many buffer/stride parameters for real-time audio inference"
    )]
    pub(crate) fn layer_forward_dispatch<M: SimdMath>(
        &mut self,
        li: usize,
        nf: usize,
        input: &[f32],
        pos: usize,
        head_wp: usize,
        use_cond_dsp: bool,
        cond_size: usize,
        is_first_array: bool,
        skip_last_residual: bool,
    ) {
        let channels = self.channels;
        let bottleneck = self.bottleneck;
        let is_first = is_first_array && li == 0;
        let is_last = skip_last_residual && li == self.num_layers - 1;
        let ring_size = self.layer_ring_sizes[li];
        let lookback = self.layer_lookbacks[li];
        let max_lookback_cols = lookback / channels;
        let bs = self.layer_buffer_starts[li];
        let use_gating = self.gating_modes[li] == GatingMode::Gated;
        let use_blending = self.gating_modes[li] == GatingMode::Blended;
        let z_out_ch = if use_gating || use_blending {
            bottleneck * 2
        } else {
            bottleneck
        };

        // Copy layer_in → history buffer.
        {
            let buf = self.layer_buffers[li].window_mut(bs, nf * channels);
            buf.copy_from_slice(&self.layer_in[..nf * channels]);
            // Apply conv_pre_film on new frames.
            // With condition_dsp, the condition signal is multi-channel (cond_size > 1).
            // Without condition_dsp, the conditioning signal is the parent input (stride `cond_size`,
            // which equals `parent_in_channels`, NOT the previous cascade array's `input_channels`).
            let cond_ch = cond_size.max(1);
            let cond_buf: &[f32] = if use_cond_dsp {
                &self.condition_dsp_output[..nf * cond_size]
            } else {
                &input[pos * cond_ch..(pos + nf) * cond_ch]
            };
            if let Some(ref mut film) = self.layers[li].conv_pre_film {
                for f in 0..nf {
                    let cond_slice = &cond_buf[f * cond_size..(f + 1) * cond_size];
                    // SAFETY: `cond_slice` has length exactly `cond_size`, matching this
                    // FiLM layer's `cond_size`, and `&mut buf[f * channels..(f + 1) * channels]`
                    // is an in-bounds sub-slice of length `channels`; both satisfy `film.process`'s
                    // documented preconditions.
                    unsafe {
                        film.process(&mut buf[f * channels..(f + 1) * channels], cond_slice);
                    }
                }
            }
        }

        // Advance buffer start with wrap.
        if bs + nf * channels + WAVENET_MAX_NUM_FRAMES * channels > ring_size * 2 {
            self.layer_buffer_starts[li] = bs + nf * channels - ring_size;
        } else {
            self.layer_buffer_starts[li] = bs + nf * channels;
        }

        debug_assert!(bs >= lookback);
        debug_assert!(bs + nf * channels <= ring_size * 2);

        if (use_gating || use_blending)
            && z_out_ch <= 16
            && matches!(
                self.layers[li].conv,
                crate::models::a2::conv1d::A2Conv1d::Standard(ref conv_std)
                    if conv_std.interleave_width == 4
                        && conv_std.do_bias
                        && conv_std.kernel >= 6
                        && max_lookback_cols >= (conv_std.kernel - 1) * conv_std.dilation
            )
        {
            let cond_input: &[f32] = if use_cond_dsp {
                &[]
            } else {
                let cond_ch = cond_size.max(1);
                &input[pos * cond_ch..(pos + nf) * cond_ch]
            };
            let flags = (is_first as u8)
                | ((is_last as u8) << 1)
                | ((use_gating as u8) << 2)
                | ((use_blending as u8) << 3)
                | ((use_cond_dsp as u8) << 4);
            // SAFETY: `self.layers[li].conv` satisfies `interleave_width == 4`, `do_bias`,
            // `kernel >= 6`, `max_lookback_cols >= (kernel - 1) * dilation`, and `z_out_ch <= 16`.
            unsafe {
                self.layer_forward_dyn_block_conv::<M>(li, nf, bs, head_wp, cond_input, flags);
            }
            return;
        }

        {
            let history = &self.layer_buffers[li][bs - lookback..bs + nf * channels];
            let layer = &mut self.layers[li];

            let cond_ch = cond_size.max(1);
            let cond_buf: &[f32] = if use_cond_dsp {
                &self.condition_dsp_output[..nf * cond_size]
            } else {
                &input[pos * cond_ch..(pos + nf) * cond_ch]
            };

            if !use_gating
                && !use_blending
                && channels == bottleneck
                && cond_size == 1
                && !layer.head1x1_active
                && self.head_accum_size == channels
                && layer.mixin_groups <= 1
                && layer.l1x1_groups <= 1
                && matches!(
                    self.activations[li],
                    crate::models::a2::activations::ActivationType::LeakyReLU { negative_slope }
                        if negative_slope == crate::models::a2::params::A2_LEAKY_SLOPE
                )
                && let Some(conv_ch) = layer.conv_ch.as_ref()
            {
                let mixin_w = &layer.mixin_w;
                let l1x1_w = &layer.l1x1_w;
                let l1x1_b = &layer.l1x1_b;
                let mut film_block = crate::models::a2::film::FilmBlock {
                    conv_pre_film: layer.conv_pre_film.as_mut(),
                    conv_post_film: layer.conv_post_film.as_mut(),
                    input_mixin_pre_film: layer.input_mixin_pre_film.as_mut(),
                    input_mixin_post_film: layer.input_mixin_post_film.as_mut(),
                    activation_pre_film: layer.activation_pre_film.as_mut(),
                    activation_post_film: layer.activation_post_film.as_mut(),
                    layer1x1_post_film: layer.layer1x1_post_film.as_mut(),
                    head1x1_post_film: layer.head1x1_post_film.as_mut(),
                };
                let input_cond = &cond_buf[..nf];
                match conv_ch {
                    crate::models::a2::layer::A2ConvCh::Ch3(ch3_conv) => {
                        // SAFETY: `layer_forward_ch3_block` requires `nf <= MAX_KERNEL_FRAMES`,
                        // `history` covering `bs - lookback .. bs + nf * 3` (asserted above),
                        // `input_cond.len() >= nf`, and `head_accum`/`layer_in` sized for CH=3
                        // (`head_accum_size == channels == 3`, `advance_head_ring` keeps `head_wp + nf`
                        // in bounds).
                        unsafe {
                            crate::models::a2::conv1d_ch3::layer_forward_ch3_block(
                                ch3_conv,
                                mixin_w,
                                l1x1_w,
                                l1x1_b,
                                &mut film_block,
                                false,
                                history,
                                max_lookback_cols,
                                nf,
                                input_cond,
                                &mut self.head_accum,
                                head_wp,
                                &mut self.layer_in,
                                is_first,
                                is_last,
                            );
                        }
                    }
                    crate::models::a2::layer::A2ConvCh::Ch8(ch8_conv) => {
                        // SAFETY: `layer_forward_ch8_block*` require `nf <= MAX_KERNEL_FRAMES`,
                        // `history` covering `bs - lookback .. bs + nf * 8` (asserted above),
                        // `input_cond.len() >= nf`, and `head_accum`/`layer_in` sized for CH=8
                        // (`head_accum_size == channels == 8`, `advance_head_ring` keeps `head_wp + nf`
                        // in bounds).
                        unsafe {
                            #[expect(deprecated)]
                            match M::ISA {
                                crate::math::common::InstructionSet::Avx512
                                | crate::math::common::InstructionSet::Avx512VnniBf16 => {
                                    crate::models::a2::conv1d_ch8::layer_forward_ch8_block_simdmath::<
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
                                        input_cond,
                                        &mut self.head_accum,
                                        head_wp,
                                        &mut self.layer_in,
                                        is_first,
                                        is_last,
                                    );
                                }
                                _ => {
                                    crate::models::a2::conv1d_ch8::layer_forward_ch8_block(
                                        ch8_conv,
                                        mixin_w,
                                        l1x1_w,
                                        l1x1_b,
                                        &mut film_block,
                                        false,
                                        history,
                                        max_lookback_cols,
                                        nf,
                                        input_cond,
                                        &mut self.head_accum,
                                        head_wp,
                                        &mut self.layer_in,
                                        is_first,
                                        is_last,
                                    );
                                }
                            }
                        }
                    }
                }
                return;
            }

            let z_scratch = &mut self.z_scratch;
            let mixin_scratch = &mut self.mixin_scratch;
            let l1x1_scratch = &mut self.l1x1_scratch;
            let head_accum = &mut self.head_accum;
            let layer_in = &mut self.layer_in;
            let head1x1_scratch = &mut self.head1x1_scratch;
            let cond_scratch = &mut self.cond_scratch;
            let gating_config = self.gating_configs[li].as_ref();
            let mut blending_config = self.blending_configs[li].as_mut();
            let activation = &self.activations[li];

            // activation_precision() lida uma vez por bloco para evitar leitura de TLS por frame — padrão adotado do caminho LSTM (layer_kernels.rs:36-42).
            let is_hf = crate::math::activations::activation_precision()
                == crate::math::activations::ActivationPrecision::Standard;

            for f in 0..nf {
                let bc = blending_config.as_deref_mut();
                // SAFETY: `process_frame_dyn` is an `unsafe fn`; the caller has verified
                // its documented preconditions: `li < self.num_layers`, `nf` frames of
                // valid data, all scratch/ring buffers sized for the model's topology,
                // and `M` matches the CPU ISA (top-level `dispatch_simd!`).
                unsafe {
                    process_frame_dyn::<M, true>(
                        layer,
                        history,
                        f,
                        max_lookback_cols,
                        head_wp,
                        z_out_ch,
                        use_gating,
                        use_blending,
                        is_first,
                        is_last,
                        self.channels,
                        self.head_accum_size,
                        self.bottleneck,
                        z_scratch,
                        mixin_scratch,
                        l1x1_scratch,
                        head_accum,
                        layer_in,
                        head1x1_scratch,
                        cond_scratch,
                        gating_config,
                        bc,
                        activation,
                        cond_buf,
                        cond_size,
                        is_hf,
                    );
                }
            }
        }
    }

    /// Out-of-line block-tiled conv1d + post-conv forward pass for non-`conv_ch` layers
    /// with `kernel >= 6` and `z_out_ch <= 16` (e.g., gated/blended A2 dynamic layers).
    #[inline(never)]
    unsafe fn layer_forward_dyn_block_conv<M: SimdMath>(
        &mut self,
        li: usize,
        nf: usize,
        bs: usize,
        head_wp: usize,
        cond_input: &[f32],
        flags: u8,
    ) {
        const Z_BLOCK_MAX_CH: usize = 16;
        let is_first = (flags & 1) != 0;
        let is_last = (flags & 2) != 0;
        let use_gating = (flags & 4) != 0;
        let use_blending = (flags & 8) != 0;
        let use_cond_dsp = (flags & 16) != 0;
        let channels = self.channels;
        let bottleneck = self.bottleneck;
        let cond_size = self.condition_size;
        let cond_buf: &[f32] = if use_cond_dsp {
            &self.condition_dsp_output[..nf * cond_size]
        } else {
            cond_input
        };
        let z_out_ch = if use_gating || use_blending {
            bottleneck * 2
        } else {
            bottleneck
        };
        let lookback = self.layer_lookbacks[li];
        let max_lookback_cols = lookback / channels;
        let history = &self.layer_buffers[li][bs - lookback..bs + nf * channels];
        let layer = &mut self.layers[li];

        let mut z_block_uninit =
            [core::mem::MaybeUninit::<f32>::uninit(); WAVENET_MAX_NUM_FRAMES * Z_BLOCK_MAX_CH];
        let z_block_ptr = z_block_uninit.as_mut_ptr() as *mut f32;
        if let crate::models::a2::conv1d::A2Conv1d::Standard(ref conv_std) = layer.conv {
            // SAFETY: `z_block_uninit` holds `WAVENET_MAX_NUM_FRAMES * Z_BLOCK_MAX_CH` elements,
            // `nf <= WAVENET_MAX_NUM_FRAMES`, `z_out_ch <= Z_BLOCK_MAX_CH`, and
            // `process_block_no_mixin_4wide` initializes every element of `0..nf * z_out_ch`.
            unsafe {
                let z_block = core::slice::from_raw_parts_mut(z_block_ptr, nf * z_out_ch);
                conv_std.process_block_no_mixin_4wide(history, z_block, max_lookback_cols, nf);
            }
        }

        let mixin_scratch = &mut self.mixin_scratch;
        let l1x1_scratch = &mut self.l1x1_scratch;
        let head_accum = &mut self.head_accum;
        let layer_in = &mut self.layer_in;
        let head1x1_scratch = &mut self.head1x1_scratch;
        let cond_scratch = &mut self.cond_scratch;
        let gating_config = self.gating_configs[li].as_ref();
        let mut blending_config = self.blending_configs[li].as_mut();
        let activation = &self.activations[li];
        let is_hf = crate::math::activations::activation_precision()
            == crate::math::activations::ActivationPrecision::Standard;

        for f in 0..nf {
            // SAFETY: `process_block_no_mixin_4wide` initialized `0..nf * z_out_ch`.
            let z_frame =
                unsafe { core::slice::from_raw_parts_mut(z_block_ptr.add(f * z_out_ch), z_out_ch) };
            let bc = blending_config.as_deref_mut();
            // SAFETY: `z_frame` holds the precomputed conv output for frame `f`; remaining
            // preconditions match `process_frame_dyn::<M, false>`.
            unsafe {
                process_frame_dyn::<M, false>(
                    layer,
                    &[],
                    f,
                    0,
                    head_wp,
                    z_out_ch,
                    use_gating,
                    use_blending,
                    is_first,
                    is_last,
                    channels,
                    self.head_accum_size,
                    bottleneck,
                    z_frame,
                    mixin_scratch,
                    l1x1_scratch,
                    head_accum,
                    layer_in,
                    head1x1_scratch,
                    cond_scratch,
                    gating_config,
                    bc,
                    activation,
                    cond_buf,
                    cond_size,
                    is_hf,
                );
            }
        }
    }

    /// Finalizes the head convolution and advances the head write position.
    #[inline(always)]
    fn head_finalize(&mut self, head_wp: usize, nf: usize, output: &mut [f32]) {
        self.head_write_pos = (head_wp + nf) & self.head_ring_mask;

        let head_size = self.head_size;
        if head_size == 1 {
            if let Some(ref head) = self.head_conv {
                head.process(
                    &self.head_accum,
                    self.head_write_pos,
                    self.head_ring_mask,
                    nf,
                    output,
                );
            }
        } else {
            // Multi-output head: same projection as `cascade_head_finalize`
            // via the shared `project_heads` helper (`head_size` interleaved
            // channels per frame, `head_rechannel_w` Conv1D over
            // `head_accum_size`, bias, then per-channel scale).
            super::process_cascade::project_heads(
                &self.head_accum,
                &self.head_rechannel_w,
                &self.head_rechannel_b,
                &self.head_rechannel_scale,
                self.head_write_pos,
                self.head_ring_mask,
                head_size,
                self.head_accum_size,
                self.head_kernel_size,
                nf,
                output,
            );
        }
    }
}
