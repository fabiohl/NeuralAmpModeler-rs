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
use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::{BlendingActivationConfig, GatingActivationConfig, GatingMode};
use crate::models::a2::layer::A2Layer;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

use core::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};

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
    /// Any input size ≤ `max_buffer_size` is safe: processing is internally chunked
    /// into sub-blocks of ≤ `WAVENET_MAX_NUM_FRAMES` (64).
    ///
    /// **SIMD Dispatch:** The `dispatch_simd!` macro evaluates the hardware once
    /// and monomorphizes `process_internal` to the detected ISA (AVX2/AVX-512),
    /// eliminating per-frame `is_x86_feature_detected` branches.
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
            "process: input ({total}) > max_buffer_size ({})",
            self.max_buffer_size
        );
        let nf_total = total.min(self.max_buffer_size);

        #[cfg(any(test, feature = "testing"))]
        {
            if let Some(ref mut dump) = self.diag.dump {
                dump.total_frames = nf_total;
            }
        }

        let cond_size = self.condition_size;

        let mut pos = 0;
        while pos < nf_total {
            let nf = (nf_total - pos).min(WAVENET_MAX_NUM_FRAMES);

            // Pre-process input through condition_dsp if present.
            // The condition_dsp output replaces the raw input as the parameter
            // for per-layer mixin and FiLM (C++ _process_condition pattern).
            if let Some(cond_dsp) = self.condition_dsp.as_mut() {
                cond_dsp.process(
                    &input[pos..pos + nf],
                    &mut self.condition_dsp_output[0..nf * cond_size],
                );
                let dsp_ch = cond_dsp.num_output_channels();
                if dsp_ch > 0 && dsp_ch < cond_size {
                    let buf = &mut self.condition_dsp_output[0..nf * cond_size];
                    if dsp_ch == 1 {
                        for f in (0..nf).rev() {
                            let val = buf[f];
                            for c in 1..cond_size {
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

            self.head_finalize(head_wp, nf, &mut output[pos..pos + nf]);
            pos += nf;
        }

        #[cfg(any(test, feature = "testing"))]
        {
            if let Some(ref mut dump) = self.diag.dump
                && dump.total_frames > 0
                && self.diag.config.capture_final_output
            {
                let mut out = AlignedVec::new(nf_total, 0.0f32).expect("diagnostic alloc");
                out.copy_from_slice(&output[..nf_total]);
                dump.final_output = Some(out);
            }
        }
    }

    /// Phase 0: rechannel pre-scaling — `input × rechannel_w_f32 → layer_in`.
    /// For mono input (input_channels == 1): `layer_in[c] = rechannel_w_f32[c] * x`.
    /// For multi-channel input (input_channels > 1): matrix multiply per frame.
    #[inline(always)]
    fn rechannel_prescale(&mut self, input: &[f32], pos: usize, nf: usize) {
        let channels = self.channels;
        let in_ch = self.input_channels;
        if in_ch == 1 {
            for (f, &x) in input[pos..pos + nf].iter().enumerate() {
                let base = f * channels;
                for c in 0..channels {
                    self.layer_in[base + c] = self.rechannel_w_f32[c] * x;
                }
            }
        } else {
            for f in 0..nf {
                let base = f * channels;
                let in_base = pos + f * in_ch;
                for c in 0..channels {
                    let mut sum = 0.0f32;
                    for ic in 0..in_ch {
                        sum += input[in_base + ic] * self.rechannel_w_f32[ic * channels + c];
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
    ) {
        let channels = self.channels;
        let bottleneck = self.bottleneck;
        let is_first = is_first_array && li == 0;
        let is_last = li == self.num_layers - 1;
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
            let buf = &mut self.layer_buffers[li];
            buf[bs..bs + nf * channels].copy_from_slice(&self.layer_in[..nf * channels]);
            // Apply conv_pre_film on new frames.
            // With condition_dsp, the condition signal is multi-channel (cond_size > 1).
            let cond_buf: &[f32] = if use_cond_dsp {
                &self.condition_dsp_output[..nf * cond_size]
            } else {
                &input[pos..pos + nf]
            };
            for f in 0..nf {
                if let Some(ref mut film) = self.layers[li].conv_pre_film {
                    let cond_slice = &cond_buf[f * cond_size..(f + 1) * cond_size];
                    unsafe {
                        film.process(
                            &mut buf[bs + f * channels..bs + (f + 1) * channels],
                            cond_slice,
                        );
                    }
                }
            }
        }

        // Advance buffer start with wrap.
        if bs + nf * channels + self.max_buffer_size * channels > ring_size * 2 {
            self.layer_buffer_starts[li] = bs + nf * channels - ring_size;
        } else {
            self.layer_buffer_starts[li] = bs + nf * channels;
        }

        {
            let history = &self.layer_buffers[li][bs - lookback..bs + nf * channels];
            let layer = &mut self.layers[li];

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

            let cond_buf: &[f32] = if use_cond_dsp {
                &self.condition_dsp_output[..nf * cond_size]
            } else {
                &input[pos..pos + nf]
            };

            for f in 0..nf {
                let bc = blending_config.as_deref_mut();
                unsafe {
                    process_frame_dyn::<M>(
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
                    );
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

// ── Private helpers ────────────────────────────────────────────

/// Per-frame inner core: conv, FiLM, mixin, activation/gating/blending,
/// head accumulation, and l1x1 residual for a single frame in one layer.
///
/// `M` is the ISA monomorphization type propagated from the top-level
/// `dispatch_simd!` in [`WaveNetA2Dyn::process`].
#[expect(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "Audio DSP kernel with many dimension parameters and explicit SIMD indexing — struct consolidation would add indirection overhead in the hot path"
)]
#[inline(always)]
unsafe fn process_frame_dyn<M: SimdMath>(
    layer: &mut A2Layer,
    history: &[f32],
    f: usize,
    max_lookback_cols: usize,
    head_wp: usize,
    z_out_ch: usize,
    use_gating: bool,
    use_blending: bool,
    is_first: bool,
    is_last: bool,
    channels: usize,
    head_accum_size: usize,
    bottleneck: usize,
    z_scratch: &mut [f32],
    mixin_scratch: &mut [f32],
    l1x1_scratch: &mut [f32],
    head_accum: &mut [f32],
    layer_in: &mut [f32],
    head1x1_scratch: &mut [f32],
    cond_scratch: &mut [f32],
    gating_config: Option<&GatingActivationConfig>,
    blending_config: Option<&mut BlendingActivationConfig>,
    activation: &ActivationType,
    cond_buf: &[f32],
    cond_size: usize,
) {
    let frame_idx = max_lookback_cols + f;
    let cond_slice = &cond_buf[f * cond_size..(f + 1) * cond_size];

    #[expect(
        unused_assignments,
        reason = "Variable assigned for clarity but value consumed by debug_assert only in release builds"
    )]
    let mut z_len = z_out_ch;

    // 1. Dilated conv → z_scratch.
    unsafe {
        layer
            .conv
            .process_single_frame::<M>(history, &mut z_scratch[..z_out_ch], frame_idx, None);
    }

    // FiLM post-conv + pre-mixin.
    if let Some(ref mut film) = layer.conv_post_film {
        unsafe {
            film.process(&mut z_scratch[..z_out_ch], cond_slice);
        }
    }

    // 2. Input mixin — matrix-vector multiply.
    // When input_mixin_pre_film is active, the condition vector is first
    // modulated by FiLM (self-modulation, C++ model.cpp:188-197), then the
    // modulated condition feeds the mixin instead of the raw condition.
    //
    // Weights are stored col-major [in_pg][out_per_g] per group (T4.3
    // transposition in builder.rs). Each condition channel broadcasts
    // into 8-wide SIMD FMA over contiguous output weights.
    {
        let mut cond_is_modulated = false;
        if let Some(ref mut film) = layer.input_mixin_pre_film {
            debug_assert!(
                cond_size <= cond_scratch.len(),
                "cond_size ({cond_size}) exceeds cond_scratch capacity ({})",
                cond_scratch.len()
            );
            cond_scratch[..cond_size].copy_from_slice(cond_slice);
            unsafe {
                film.process(&mut cond_scratch[..cond_size], cond_slice);
            }
            cond_is_modulated = true;
        }
        let cond_for_mixin: &[f32] = if cond_is_modulated {
            &cond_scratch[..cond_size]
        } else {
            cond_slice
        };

        let in_pg = if layer.mixin_groups <= 1 {
            cond_size
        } else {
            cond_size / layer.mixin_groups as usize
        };
        let out_per_g = if layer.mixin_groups <= 1 {
            z_out_ch
        } else {
            z_out_ch / layer.mixin_groups as usize
        };
        let num_groups = layer.mixin_groups.max(1) as usize;

        if out_per_g >= 8 {
            for g in 0..num_groups {
                let group_base = g * out_per_g * in_pg;
                let in_start = g * in_pg;
                let out_start = g * out_per_g;
                unsafe {
                    let mut oc = 0;
                    while oc + 8 <= out_per_g {
                        let mut acc = _mm256_setzero_ps();
                        for ic in 0..in_pg {
                            let cond = _mm256_set1_ps(cond_for_mixin[in_start + ic]);
                            let w = _mm256_loadu_ps(
                                layer.mixin_w.as_ptr().add(group_base + ic * out_per_g + oc),
                            );
                            acc = _mm256_fmadd_ps(cond, w, acc);
                        }
                        _mm256_storeu_ps(mixin_scratch.as_mut_ptr().add(out_start + oc), acc);
                        oc += 8;
                    }
                    // Scalar tail for remaining output channels in this group.
                    for oc in oc..out_per_g {
                        let mut sum = 0.0;
                        for ic in 0..in_pg {
                            sum += layer.mixin_w[group_base + ic * out_per_g + oc]
                                * cond_for_mixin[in_start + ic];
                        }
                        mixin_scratch[out_start + oc] = sum;
                    }
                }
            }
        } else {
            // Scalar fallback for small groups (out_per_g < 8).
            for g in 0..num_groups {
                let group_base = g * out_per_g * in_pg;
                let in_start = g * in_pg;
                let out_start = g * out_per_g;
                for oc in 0..out_per_g {
                    let mut sum = 0.0;
                    for ic in 0..in_pg {
                        sum += layer.mixin_w[group_base + ic * out_per_g + oc]
                            * cond_for_mixin[in_start + ic];
                    }
                    mixin_scratch[out_start + oc] = sum;
                }
            }
        }
    }

    // FiLM post-mixin + pre-activation.
    // Apply FiLM on the isolated mixin buffer before summing.
    if let Some(ref mut film) = layer.input_mixin_post_film {
        unsafe {
            film.process(&mut mixin_scratch[..z_out_ch], cond_slice);
        }
    }

    // Sum mixin output to z_scratch (vectorized 8-wide).
    if z_out_ch >= 8 {
        unsafe {
            let mut c = 0;
            while c + 8 <= z_out_ch {
                let src = _mm256_loadu_ps(mixin_scratch.as_ptr().add(c));
                let dst = _mm256_loadu_ps(z_scratch.as_ptr().add(c));
                _mm256_storeu_ps(z_scratch.as_mut_ptr().add(c), _mm256_add_ps(dst, src));
                c += 8;
            }
            for c in c..z_out_ch {
                z_scratch[c] += mixin_scratch[c];
            }
        }
    } else {
        for c in 0..z_out_ch {
            z_scratch[c] += mixin_scratch[c];
        }
    }

    if let Some(ref mut film) = layer.activation_pre_film {
        unsafe {
            film.process(&mut z_scratch[..z_out_ch], cond_slice);
        }
    }

    // 3. Activation or Gating/Blending.
    if use_gating {
        if let Some(gc) = gating_config {
            unsafe {
                gc.apply_gating_simd::<M>(&mut z_scratch[..z_out_ch]);
            }
        }
        z_len = bottleneck;
    } else if use_blending {
        if let Some(bc) = blending_config {
            unsafe {
                bc.apply_blending_simd::<M>(&mut z_scratch[..z_out_ch]);
            }
        }
        z_len = bottleneck;
    } else {
        unsafe {
            activation.apply_simd::<M>(&mut z_scratch[..bottleneck]);
        }
        z_len = bottleneck;
    }

    // FiLM post-activation.
    if let Some(ref mut film) = layer.activation_post_film {
        unsafe {
            film.process(&mut z_scratch[..z_len], cond_slice);
        }
    }

    let head1x1_active = layer.head1x1_active;
    let head1x1_w = &layer.head1x1_w;
    let head1x1_b = &layer.head1x1_b;
    let head_off = (head_wp + f) * head_accum_size;
    if head1x1_active {
        // head1x1_w is [head_accum_size][h1_in] row-major (transposed in build.rs).
        let h1_in = if head1x1_w.is_empty() {
            0
        } else {
            head1x1_w.len() / head_accum_size
        };
        let h1_groups = bottleneck.checked_div(h1_in).unwrap_or(1);
        let ch_per_group = head_accum_size / h1_groups;
        for grp in 0..h1_groups {
            let z_off = grp * h1_in;
            let ch_start = grp * ch_per_group;
            let ch_end = (grp + 1) * ch_per_group;
            // Vectorized inner dot product for each output channel.
            // Processes h1_in in 8-wide SIMD steps, extracting lanes
            // sequentially to preserve exact left-to-right accumulation.
            if h1_in >= 8 {
                for oc in ch_start..ch_end {
                    unsafe {
                        let mut acc = _mm256_setzero_ps();
                        let mut ic = 0;
                        while ic + 8 <= h1_in {
                            let inputs = _mm256_loadu_ps(z_scratch.as_ptr().add(z_off + ic));
                            let weights = _mm256_loadu_ps(head1x1_w.as_ptr().add(oc * h1_in + ic));
                            acc = _mm256_fmadd_ps(inputs, weights, acc);
                            ic += 8;
                        }
                        // Extract lanes preserving left-to-right accumulation order.
                        let mut sum = head1x1_b[oc];
                        {
                            let mut lane_buf = [0.0f32; 8];
                            _mm256_storeu_ps(lane_buf.as_mut_ptr(), acc);
                            for v in &lane_buf {
                                sum += *v;
                            }
                        }
                        // Scalar tail for remaining h1_in.
                        for ic in ic..h1_in {
                            sum += head1x1_w[oc * h1_in + ic] * z_scratch[z_off + ic];
                        }
                        head1x1_scratch[oc] = sum;
                    }
                }
            } else {
                for oc in ch_start..ch_end {
                    let mut sum = head1x1_b[oc];
                    let b_start = oc * h1_in;
                    for ic in 0..h1_in {
                        sum += head1x1_w[b_start + ic] * z_scratch[z_off + ic];
                    }
                    head1x1_scratch[oc] = sum;
                }
            }
        }
        // FiLM after head1x1 projection (C++ model.cpp:283-287).
        if let Some(ref mut film) = layer.head1x1_post_film {
            unsafe {
                film.process(&mut head1x1_scratch[..head_accum_size], cond_slice);
            }
        }
        if is_first {
            head_accum[head_off..head_off + head_accum_size]
                .copy_from_slice(&head1x1_scratch[..head_accum_size]);
        } else {
            // Vectorized accumulation into head ring buffer.
            unsafe {
                let mut c = 0;
                while c + 8 <= head_accum_size {
                    let src = _mm256_loadu_ps(head1x1_scratch.as_ptr().add(c));
                    let dst = _mm256_loadu_ps(head_accum.as_ptr().add(head_off + c));
                    _mm256_storeu_ps(
                        head_accum.as_mut_ptr().add(head_off + c),
                        _mm256_add_ps(dst, src),
                    );
                    c += 8;
                }
                for c in c..head_accum_size {
                    head_accum[head_off + c] += head1x1_scratch[c];
                }
            }
        }
    } else {
        debug_assert_eq!(
            bottleneck, head_accum_size,
            "head1x1 must be active when bottleneck != head_accum_size"
        );
        if is_first {
            head_accum[head_off..head_off + bottleneck].copy_from_slice(&z_scratch[..bottleneck]);
        } else {
            unsafe {
                let mut c = 0;
                while c + 8 <= bottleneck {
                    let src = _mm256_loadu_ps(z_scratch.as_ptr().add(c));
                    let dst = _mm256_loadu_ps(head_accum.as_ptr().add(head_off + c));
                    _mm256_storeu_ps(
                        head_accum.as_mut_ptr().add(head_off + c),
                        _mm256_add_ps(dst, src),
                    );
                    c += 8;
                }
                for c in c..bottleneck {
                    head_accum[head_off + c] += z_scratch[c];
                }
            }
        }
    }

    // 5. L1x1 residual (skip on last layer).
    if !is_last {
        let base = f * channels;
        let l1x1_w = &layer.l1x1_w;
        let l1x1_b = &layer.l1x1_b;
        if layer.l1x1_groups <= 1 {
            // Dense L1x1: weights are col-major [bottleneck][channels].
            // Each ic row has `channels` contiguous weights, enabling
            // 8-wide SIMD across output channels with broadcast input.
            if channels >= 8 {
                let channels_aligned = channels & !7;
                unsafe {
                    for oc in (0..channels_aligned).step_by(8) {
                        let mut acc = _mm256_loadu_ps(l1x1_b.as_ptr().add(oc));
                        for ic in 0..bottleneck {
                            let z = _mm256_set1_ps(z_scratch[ic]);
                            let w = _mm256_loadu_ps(l1x1_w.as_ptr().add(ic * channels + oc));
                            acc = _mm256_fmadd_ps(z, w, acc);
                        }
                        _mm256_storeu_ps(l1x1_scratch.as_mut_ptr().add(oc), acc);
                    }
                }
                // Scalar tail.
                for oc in channels_aligned..channels {
                    let mut sum = l1x1_b[oc];
                    for ic in 0..bottleneck {
                        sum += l1x1_w[ic * channels + oc] * z_scratch[ic];
                    }
                    l1x1_scratch[oc] = sum;
                }
            } else {
                for oc in 0..channels {
                    let mut sum = l1x1_b[oc];
                    for ic in 0..bottleneck {
                        sum += l1x1_w[ic * channels + oc] * z_scratch[ic];
                    }
                    l1x1_scratch[oc] = sum;
                }
            }
        } else {
            let in_pg = bottleneck / layer.l1x1_groups as usize;
            let out_per_g = channels / layer.l1x1_groups as usize;
            // Grouped L1x1: weights are row-major [channels][in_pg].
            // Vectorize inner dot product over in_pg dimension.
            if in_pg >= 8 {
                for g in 0..layer.l1x1_groups as usize {
                    let in_start = g * in_pg;
                    let out_start = g * out_per_g;
                    for oc in out_start..out_start + out_per_g {
                        unsafe {
                            let mut acc = _mm256_setzero_ps();
                            let mut ic = 0;
                            while ic + 8 <= in_pg {
                                let inputs = _mm256_loadu_ps(z_scratch.as_ptr().add(in_start + ic));
                                let weights = _mm256_loadu_ps(l1x1_w.as_ptr().add(oc * in_pg + ic));
                                acc = _mm256_fmadd_ps(inputs, weights, acc);
                                ic += 8;
                            }
                            let mut sum = l1x1_b[oc];
                            {
                                let mut lane_buf = [0.0f32; 8];
                                _mm256_storeu_ps(lane_buf.as_mut_ptr(), acc);
                                for v in &lane_buf {
                                    sum += *v;
                                }
                            }
                            for ic in ic..in_pg {
                                sum += l1x1_w[oc * in_pg + ic] * z_scratch[in_start + ic];
                            }
                            l1x1_scratch[oc] = sum;
                        }
                    }
                }
            } else {
                for g in 0..layer.l1x1_groups as usize {
                    let in_start = g * in_pg;
                    let out_start = g * out_per_g;
                    for oc in out_start..out_start + out_per_g {
                        let mut sum = l1x1_b[oc];
                        let w_base = oc * in_pg;
                        for ic in 0..in_pg {
                            sum += l1x1_w[w_base + ic] * z_scratch[in_start + ic];
                        }
                        l1x1_scratch[oc] = sum;
                    }
                }
            }
        }
        if let Some(ref mut film) = layer.layer1x1_post_film.as_mut().filter(|_| use_blending) {
            unsafe {
                film.process(&mut l1x1_scratch[..channels], cond_slice);
            }
        }
        // Vectorized accumulation into layer_in.
        if channels >= 8 {
            unsafe {
                let mut oc = 0;
                while oc + 8 <= channels {
                    let src = _mm256_loadu_ps(l1x1_scratch.as_ptr().add(oc));
                    let dst = _mm256_loadu_ps(layer_in.as_ptr().add(base + oc));
                    _mm256_storeu_ps(
                        layer_in.as_mut_ptr().add(base + oc),
                        _mm256_add_ps(dst, src),
                    );
                    oc += 8;
                }
                for oc in oc..channels {
                    layer_in[base + oc] += l1x1_scratch[oc];
                }
            }
        } else {
            for oc in 0..channels {
                layer_in[base + oc] += l1x1_scratch[oc];
            }
        }
    }
}
