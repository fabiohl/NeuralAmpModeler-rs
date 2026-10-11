// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 Dynamic model — cascade pipeline methods.
//!
//! These methods are shared with the `WaveNetA2Cascade` orchestrator. They allow the cascade to drive individual sub-arrays
//! (`[WaveNetA2Dyn]`) in sequence: propagating condition outputs, residuals,
//! and head accumulators from one array to the next.

use crate::math::common::SimdMath;

use super::WaveNetA2Dyn;

/// Projects the head accumulator ring into `head_size` interleaved channels
/// per frame (`output[f * head_size + oc]`), applying the `head_rechannel`
/// Conv1D over `channels`, the per-channel bias and the per-channel scale.
///
/// Shared by the single-array [`WaveNetA2Dyn::head_finalize`](super::process::WaveNetA2Dyn)
/// path and the cascade [`WaveNetA2Dyn::cascade_head_finalize`] path so both
/// geometries scale identically (single source of truth for the multi-output
/// head projection).
#[expect(
    clippy::too_many_arguments,
    reason = "Head projection needs the full ring geometry (accum, weights, bias, scale, ring, dims); callers pass model fields directly"
)]
#[inline(always)]
pub(crate) fn project_heads(
    head_accum: &[f32],
    head_rechannel_w: &[f32],
    head_rechannel_b: &[f32],
    head_rechannel_scale: &[f32],
    write_pos: usize,
    ring_mask: usize,
    head_size: usize,
    channels: usize,
    kernel: usize,
    nf: usize,
    output: &mut [f32],
) {
    let base = write_pos.wrapping_sub(nf);
    for f in 0..nf {
        let col_base = base.wrapping_add(f);
        let out_off = f * head_size;
        for oc in 0..head_size {
            let w_base = oc * kernel * channels;
            let mut y = head_rechannel_b[oc];
            for t in 0..kernel {
                let col = col_base.wrapping_sub(kernel - 1 - t) & ring_mask;
                let ha_off = col * channels;
                let w_off = w_base + t * channels;
                for ic in 0..channels {
                    y += head_rechannel_w[w_off + ic] * head_accum[ha_off + ic];
                }
            }
            output[out_off + oc] = y * head_rechannel_scale[oc];
        }
    }
}

impl WaveNetA2Dyn {
    /// Runs the per-layer loop for the cascade pipeline.
    /// Caller must pre-fill `self.layer_in` and initialize `self.head_accum`
    /// before calling this method. After return, `self.layer_in` contains
    /// the residual output (for cascading to next array) and
    /// `self.head_accum` contains the accumulated head (for head_conv or
    /// cascading to next array).
    #[inline(always)]
    pub(crate) fn cascade_layer_loop<M: SimdMath>(
        &mut self,
        nf: usize,
        input: &[f32],
        pos: usize,
        use_cond_dsp: bool,
        cond_size: usize,
        is_first_array: bool,
    ) {
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
                is_first_array,
                false, // The next array consumes even the last layer's residual.
            );
        }

        self.head_write_pos = (head_wp + nf) & self.head_ring_mask;
    }

    /// Finalizes the head convolution for cascade — uses the current
    /// head_write_pos (set by cascade_layer_loop).
    /// When head_size > 1, applies full Conv1D(CH→1, K=16, bias, head_scale)
    /// per output channel matching the C++ reference.
    #[inline(always)]
    pub(crate) fn cascade_head_finalize(&mut self, nf: usize, output: &mut [f32]) {
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
            project_heads(
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

    /// Writes mono input into `layer_in` with rechannel scaling.
    /// For the cascade: Array 0 receives mono input directly.
    #[inline(always)]
    pub(crate) fn cascade_write_mono_input(&mut self, input: &[f32], pos: usize, nf: usize) {
        let channels = self.channels;
        for f in 0..nf {
            let base = f * channels;
            let x = input[pos + f];
            for c in 0..channels {
                self.layer_in[base + c] = self.rechannel_w_f32[c] * x;
            }
        }
    }

    /// Writes multi-channel residual from a previous array into `layer_in`
    /// with rechannel scaling. `residual` is `nf * src_channels` elements.
    #[inline(always)]
    pub(crate) fn cascade_write_residual_input(
        &mut self,
        residual: &[f32],
        nf: usize,
        src_channels: usize,
    ) {
        let channels = self.channels;
        let in_ch = self.input_channels;
        assert_eq!(in_ch, src_channels, "input_channels mismatch in cascade");
        for f in 0..nf {
            let base = f * channels;
            let rbase = f * src_channels;
            for c in 0..channels {
                let mut sum = 0.0f32;
                for ic in 0..src_channels {
                    sum += residual[rbase + ic] * self.rechannel_w_f32[c * src_channels + ic];
                }
                self.layer_in[base + c] = sum;
            }
        }
    }

    /// Copies condition output into this array's `condition_dsp_output` buffer
    /// so that `layer_forward_dispatch` can use it.
    ///
    /// Fail-closed (F-10/H-05): this runs on the RT path and must never
    /// allocate or panic. The destination is pre-allocated to
    /// `condition_size × max_buffer_size`; the copied frame count is clamped to
    /// both the destination and source capacities so a hostile/host-provided
    /// `cond_buf` or oversized `nf` degrades gracefully instead of panicking.
    #[inline(always)]
    pub(crate) fn cascade_set_condition(&mut self, cond_buf: &[f32], nf: usize, cond_size: usize) {
        let dest = &mut self.condition_dsp_output;
        let cond_size = cond_size.max(1);
        let dest_frames = dest.len() / cond_size;
        let src_frames = cond_buf.len() / cond_size;
        let nf_copied = nf.min(dest_frames).min(src_frames);
        if nf_copied > 0 {
            let n = nf_copied * cond_size;
            // When `condition_dsp` is present the caller passes the engine's own
            // `condition_dsp_output` as `cond_buf` — a self-copy. Skip it to
            // avoid overlapping `copy_from_slice` (UB contract violation).
            if !core::ptr::eq(cond_buf.as_ptr(), dest.as_ptr()) {
                dest[0..n].copy_from_slice(&cond_buf[0..n]);
            }
        }
    }

    /// Seeds this array's head_accum from the previous array's post-rechannel
    /// head output (computed by `cascade_head_finalize`).
    ///
    /// Copies `min(prev_head_size, self.head_accum_size)` channels per frame,
    /// zero-filling the remainder.
    #[inline(always)]
    pub(crate) fn cascade_seed_head_from_output(
        &mut self,
        prev_head_output: &[f32],
        nf: usize,
        prev_head_size: usize,
    ) {
        let copy_ch = prev_head_size.min(self.head_accum_size);
        // Match the layer loop's write position before seeding: wrapping after
        // this copy would leave the incoming head at the old ring position.
        let curr_wp = self.advance_head_ring(nf);
        let curr_head = &mut self.head_accum;
        for f in 0..nf {
            let prev_off = f * prev_head_size;
            let curr_off = ((curr_wp + f) & self.head_ring_mask) * self.head_accum_size;
            for c in 0..copy_ch {
                curr_head[curr_off + c] = prev_head_output[prev_off + c];
            }
            for c in copy_ch..self.head_accum_size {
                curr_head[curr_off + c] = 0.0;
            }
        }
    }
}
