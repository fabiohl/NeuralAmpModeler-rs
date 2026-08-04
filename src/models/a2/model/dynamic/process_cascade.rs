// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 Dynamic model — cascade pipeline methods.
//!
//! These methods are shared with the `WaveNetA2Cascade` orchestrator. They allow the cascade to drive individual sub-arrays
//! (`[WaveNetA2Dyn]`) in sequence: propagating condition outputs, residuals,
//! and head accumulators from one array to the next.

use crate::math::common::{AlignedVec, SimdMath};

use super::WaveNetA2Dyn;

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
            let channels = self.head_accum_size;
            let k = crate::models::a2::params::A2_HEAD_KERNEL_SIZE;
            let hw = &self.head_rechannel_w;
            let hb = &self.head_rechannel_b;
            let hs = &self.head_rechannel_scale;
            let ha = &self.head_accum;
            let wp = self.head_write_pos;
            let mask = self.head_ring_mask;
            let base = wp.wrapping_sub(nf);
            for f in 0..nf {
                let col_base = base.wrapping_add(f);
                let out_off = f * head_size;
                for oc in 0..head_size {
                    let w_base = oc * k * channels;
                    let mut y = hb[oc];
                    for t in 0..k {
                        let col = col_base.wrapping_sub(k - 1 - t) & mask;
                        let ha_off = col * channels;
                        let w_off = w_base + t * channels;
                        for ic in 0..channels {
                            y += hw[w_off + ic] * ha[ha_off + ic];
                        }
                    }
                    output[out_off + oc] = y * hs[oc];
                }
            }
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
                    sum += residual[rbase + ic] * self.rechannel_w_f32[ic * channels + c];
                }
                self.layer_in[base + c] = sum;
            }
        }
    }

    /// Copies condition output into this array's `condition_dsp_output` buffer
    /// so that `layer_forward_dispatch` can use it.
    #[inline(always)]
    pub(crate) fn cascade_set_condition(&mut self, cond_buf: &[f32], nf: usize, cond_size: usize) {
        let dest = &mut self.condition_dsp_output;
        if dest.len() < nf * cond_size {
            *dest = AlignedVec::new(nf * cond_size, 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
        }
        dest[0..nf * cond_size].copy_from_slice(&cond_buf[0..nf * cond_size]);
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
        let curr_wp = self.head_write_pos;
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
