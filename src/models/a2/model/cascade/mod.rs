// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 Cascade — Multi-array chain of A2 Dynamic engines.
//!
//! ## Architecture
//!
//! `WaveNetA2Cascade` composes N `WaveNetA2Dyn` instances into a serial
//! pipeline matching the C++ multi-array A2 pattern:
//!
//! 1. Array 0 processes raw mono input → produces residual output
//!    (`layer_in`) and head output (`head_conv`).
//! 2. Arrays 1..N-1 receive the previous array's residual output as
//!    input and previous head_accum as seed for their head accumulator.
//! 3. The final array's head_conv output is the cascade output.
//!
//! When `condition_dsp` is `Some`, the raw audio input is first
//! pre-processed by the nested DSP before reaching the arrays.

use crate::math::common::{AlignedVec, SimdMath};
use crate::models::NamModel;
use crate::models::StaticModel;
use crate::models::a2::model::dynamic::WaveNetA2Dyn;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

/// Multi-array A2 cascade — serial chain of `WaveNetA2Dyn` engines.
pub struct WaveNetA2Cascade {
    /// Serial array engines (array 0 → array 1 → … → array N-1).
    pub arrays: Vec<WaveNetA2Dyn>,
    /// Combined receptive field (max per-array RF).
    pub receptive_field_size: usize,
    /// Optional condition DSP sub-model.
    pub condition_dsp: Option<Box<StaticModel>>,
    /// Pre-allocated output buffer for condition_dsp.
    pub condition_dsp_output: AlignedVec<f32>,
    /// Condition vector size.
    pub condition_size: usize,
    /// Whether to prewarm on reset.
    pub prewarm_on_reset: bool,
    /// Cascade residual buffer: stores array N's layer_in output for array N+1.
    /// Size: `max_channels * WAVENET_MAX_NUM_FRAMES`.
    cascade_residual: AlignedVec<f32>,
    /// Intermediate head output buffer: stores array N's post-rechannel head
    /// output to seed array N+1's head_accum. Size: `max_head_size * max_buffer_size`.
    intermediate_head_output: AlignedVec<f32>,
    /// Largest per-array channel count (for cascade scratch sizing).
    max_channels: usize,
    /// Largest per-array head_size (for intermediate head output sizing).
    max_head_size: usize,
    /// Maximum frames per processing block.
    max_buffer_size: usize,
    /// Model-level head scale multiplier applied to final output (C++ NAMCore `WaveNet::_head_scale`).
    pub head_scale: f32,
    /// Zeroed-sample stabilization work still pending for the deferred split
    /// pass armed by [`Self::prewarm_reset`](super::super::NamModel::prewarm_reset).
    /// Always `0` for freshly built models; the integral
    /// [`Self::prewarm`](super::super::NamModel::prewarm) /
    /// [`Self::reset`](super::super::NamModel::reset) paths complete it by
    /// construction and never consult or alter it otherwise.
    pub prewarm_pending: usize,
}

impl WaveNetA2Cascade {
    /// Creates a new cascade from individually-built A2 dynamic engines.
    ///
    /// The arrays must be ordered sequentially (array 0 first).
    ///
    /// Fallible (H-05): buffer allocation failures propagate to the caller as
    /// `Err` instead of panicking on the loader path.
    pub fn try_new(
        arrays: Vec<WaveNetA2Dyn>,
        condition_dsp: Option<Box<StaticModel>>,
        condition_size: usize,
    ) -> anyhow::Result<Self> {
        let rf = arrays
            .iter()
            .map(|a| a.receptive_field_size)
            .max()
            .unwrap_or(0);
        let max_ch = arrays.iter().map(|a| a.channels).max().unwrap_or(1);
        let max_hs = arrays.iter().map(|a| a.head_size).max().unwrap_or(1);
        let cond_buf_size = if condition_dsp.is_some() {
            condition_size
        } else {
            0
        } * WAVENET_MAX_NUM_FRAMES;

        // H-05/F-10: when the cascade feeds a processed condition, each array's
        // `condition_dsp_output` must be pre-allocated off-RT — the processing
        // path must never allocate (previously `cascade_set_condition` lazily
        // allocated on the RT thread and the hardcoded `use_cond_dsp=true`
        // would then read a length-0 buffer).
        let mut arrays = arrays;
        if condition_dsp.is_some() {
            for arr in arrays.iter_mut() {
                arr.condition_dsp_output =
                    AlignedVec::new(arr.condition_size.max(1) * WAVENET_MAX_NUM_FRAMES, 0.0f32)?;
            }
        }

        Ok(Self {
            arrays,
            receptive_field_size: rf,
            condition_dsp,
            condition_dsp_output: AlignedVec::new(cond_buf_size, 0.0f32)?,
            condition_size,
            prewarm_on_reset: true,
            cascade_residual: AlignedVec::new(max_ch * WAVENET_MAX_NUM_FRAMES, 0.0f32)?,
            intermediate_head_output: AlignedVec::new(max_hs * WAVENET_MAX_NUM_FRAMES, 0.0f32)?,
            max_channels: max_ch,
            max_head_size: max_hs,
            max_buffer_size: WAVENET_MAX_NUM_FRAMES,
            head_scale: 1.0f32,
            prewarm_pending: 0,
        })
    }

    /// Returns the number of channels of the first array.
    pub fn channels(&self) -> usize {
        self.arrays.first().map(|a| a.channels).unwrap_or(0)
    }

    /// Full forward pass through the cascade.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // SAFETY: `dispatch_simd!` dispatches on runtime CPUID feature checks to a
        // matching `#[target_feature]` backend; `input`/`output` are the same valid
        // slices passed to this safe wrapper.
        unsafe {
            crate::math::common::dispatch_simd!(self, process_internal, input, output);
        }
    }

    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        // Each frame writes `last.head_size` output samples; clamp the frame
        // count so the output is never indexed beyond its actual length.
        let out_per_frame = self.arrays.last().map_or(1, |a| a.head_size.max(1));
        let total = input.len().min(output.len() / out_per_frame);
        if total == 0 {
            return;
        }

        let num_arrays = self.arrays.len();
        if num_arrays == 0 {
            return;
        }

        output[..total * out_per_frame].fill(0.0);
        debug_assert!(
            total <= self.max_buffer_size,
            "process: input ({total}) > max_buffer_size ({})",
            self.max_buffer_size
        );
        let nf_total = total.min(self.max_buffer_size);

        let cond_size = self.condition_size;

        let mut pos = 0;
        while pos < nf_total {
            let nf = (nf_total - pos).min(WAVENET_MAX_NUM_FRAMES);

            // Pre-process condition_dsp at cascade level.
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

            // Determine the condition slice for all arrays.
            let cond_slice: &[f32] = if self.condition_dsp.is_some() {
                &self.condition_dsp_output[0..nf * cond_size]
            } else {
                &input[pos..pos + nf]
            };

            // Array 0: mono input, no head seed.
            {
                let arr0 = &mut self.arrays[0];
                arr0.cascade_write_mono_input(input, pos, nf);
                arr0.cascade_set_condition(cond_slice, nf, arr0.condition_size);
                arr0.cascade_layer_loop::<M>(
                    nf,
                    input,
                    pos,
                    self.condition_dsp.is_some(),
                    arr0.condition_size,
                    true,
                );

                // Save residual and compute head output for next array.
                let ch0 = arr0.channels;
                self.cascade_residual[0..nf * ch0].copy_from_slice(&arr0.layer_in[0..nf * ch0]);
                if num_arrays > 1 {
                    let hs = arr0.head_size;
                    arr0.cascade_head_finalize(nf, &mut self.intermediate_head_output[0..nf * hs]);
                }
            }

            // Arrays 1..N-1: residual from previous, head seed from
            // previous array's post-rechannel head output.
            for ai in 1..num_arrays {
                let prev_ch = self.arrays[ai - 1].channels;
                let prev_hs = self.arrays[ai - 1].head_size;

                let (_left, right) = self.arrays.split_at_mut(ai);
                let curr = &mut right[0];

                // Seed head_accum from previous array's post-rechannel head output.
                curr.cascade_seed_head_from_output(
                    &self.intermediate_head_output[0..nf * prev_hs],
                    nf,
                    prev_hs,
                );

                // Write residual input.
                curr.cascade_write_residual_input(&self.cascade_residual, nf, prev_ch);
                curr.cascade_set_condition(cond_slice, nf, curr.condition_size);
                curr.cascade_layer_loop::<M>(
                    nf,
                    input,
                    pos,
                    self.condition_dsp.is_some(),
                    curr.condition_size,
                    false,
                );

                // Save residual and compute head output for next array (if not last).
                let curr_ch = curr.channels;
                self.cascade_residual[0..nf * curr_ch]
                    .copy_from_slice(&curr.layer_in[0..nf * curr_ch]);
                if ai < num_arrays - 1 {
                    let curr_hs = curr.head_size;
                    curr.cascade_head_finalize(
                        nf,
                        &mut self.intermediate_head_output[0..nf * curr_hs],
                    );
                }
            }

            // Finalize head on the last array.
            let last_idx = num_arrays - 1;
            let last = &mut self.arrays[last_idx];
            let out_start = pos * last.head_size;
            let out_end = out_start + nf * last.head_size;
            last.cascade_head_finalize(nf, &mut output[out_start..out_end]);
            if self.head_scale != 1.0f32 {
                for s in &mut output[out_start..out_end] {
                    *s *= self.head_scale;
                }
            }

            pos += nf;
        }
    }

    /// Reallocates internal buffers.
    pub fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        if max_buf <= self.max_buffer_size {
            for arr in &mut self.arrays {
                arr.set_max_buffer_size(max_buf)?;
            }
            return Ok(());
        }
        self.max_buffer_size = max_buf;
        for arr in &mut self.arrays {
            arr.set_max_buffer_size(max_buf)?;
        }
        let cond_output_size = self.condition_size * max_buf;
        self.condition_dsp_output = AlignedVec::new(cond_output_size, 0.0f32)?;
        self.cascade_residual = AlignedVec::new(self.max_channels * max_buf, 0.0f32)?;
        self.intermediate_head_output = AlignedVec::new(self.max_head_size * max_buf, 0.0f32)?;
        Ok(())
    }

    /// Resets internal state.
    pub fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.set_max_buffer_size(max_buffer_size)?;
        for arr in &mut self.arrays {
            arr.reset(sample_rate, max_buffer_size)?;
        }
        if self.prewarm_on_reset {
            self.prewarm();
        }
        Ok(())
    }

    /// Zero phase of the deferred split stabilization: clears the cascade
    /// scratch buffers in place (via the shrink-equivariant
    /// `set_max_buffer_size`, which zeroes the subarray states in place), arms
    /// the per-subarray split passes (without integral prewarm), and records
    /// the cascade-level zero-feed units. Allocation- and lock-free: a
    /// real-time consumer may call it on its audio thread and amortize
    /// [`WaveNetA2Cascade::prewarm_step`] within per-callback budgets.
    pub fn prewarm_reset(&mut self) {
        // Order matters (A2-Max class): the subarray zero-feeds must run
        // BEFORE the cascade-level scratch is touched, because the subarray
        // `prewarm_reset()` calls `set_max_buffer_size(max)` on models whose
        // nested condition path is only valid at their build size — the same
        // sizing the old integral cascade prewarm relied on (it fed
        // `max_buffer_size`-sized blocks from a fresh build without
        // re-sizing). Re-sizing the cascade level first (which recreates
        // level scratch at a larger `max_buffer_size`) would leave the
        // subarrays' condition paths undersized for the blocks the armed
        // feeds later process.
        for arr in &mut self.arrays {
            // NOTE: `arr.reset()` would invoke the subarray's own integral
            // `prewarm()` (a full RF feed through the subarray), which is
            // exactly what the split flow must NOT do — the subarray feeds
            // stay armed and drain through `prewarm_step`. `reset()` is also
            // wrong for a second reason: the integral `prewarm()` panics on
            // models whose nested condition path is not yet fully sized (the
            // A2-Max class: `cond_dsp` cascade whose subarrays were built at
            // `max_buf=64` while the parent processes larger blocks; the old
            // integral cascade prewarm only ever fed `max_buffer_size`-sized
            // blocks and never hit this). The split flow therefore reproduces
            // only the *zeroing* half of the integral reset here (`arr` was
            // already sized by the `set_max_buffer_size` above, which takes
            // the in-place zeroing path when the size is unchanged) and arms
            // the subarray feed via `prewarm_reset()`.
            if arr.set_max_buffer_size(self.max_buffer_size).is_err() {
                return;
            }
            arr.set_prewarm_on_reset(false);
            arr.prewarm_reset();
            arr.set_prewarm_on_reset(true);
        }

        self.prewarm_pending = self.receptive_field_size.max(2048);
    }

    /// Advances the armed stabilization by at most `samples` zeroed samples
    /// (counted across subarrays and level) and returns the work still pending
    /// (0 = converged). The integral order is preserved: subarray feeds
    /// complete first — driven through the chained [`Self::process`] path
    /// (never by direct subarray `process`, which would feed mono zeros into
    /// a multi-channel residual input and index out of bounds) — then the
    /// cascade-level feed continues through the same chained path.
    /// Allocation- and lock-free.
    pub fn prewarm_step(&mut self, samples: usize) -> usize {
        // Subarray zero-feeds run *through the chain*: each chained `process`
        // advances every subarray's feed by `nf` (each subarray's
        // `prewarm_step` is driven by the chained call below, not here —
        // calling `arr.prewarm_step` directly would feed mono zeros into
        // residual inputs of width `input_channels > 1`). Drive the chain in
        // `WAVENET_MAX_NUM_FRAMES`-capped blocks until either the caller's
        // `samples` budget or all pending work (subarrays + level) drains.
        let mut budget = samples;
        loop {
            let sub_pending: usize = self
                .arrays
                .iter()
                .map(|a| a.prewarm_pending_stabilization())
                .max()
                .unwrap_or(0);
            let total = sub_pending + self.prewarm_pending;
            if total == 0 || budget == 0 {
                return total;
            }
            let nf = budget.min(total).min(WAVENET_MAX_NUM_FRAMES);
            let zeros = [0.0f32; WAVENET_MAX_NUM_FRAMES];
            let mut discard = [0.0f32; WAVENET_MAX_NUM_FRAMES];
            self.process(&zeros[..nf], &mut discard[..nf]);
            // The chained pass advances each subarray feed by exactly `nf`.
            for arr in &mut self.arrays {
                arr.prewarm_advance(nf);
            }
            self.prewarm_pending = self.prewarm_pending.saturating_sub(nf);
            budget -= nf;
        }
    }

    /// Deferred pass pending? (`true` when nothing is armed/left.)
    pub fn prewarm_complete(&self) -> bool {
        self.prewarm_pending == 0 && self.arrays.iter().all(|a| a.prewarm_complete())
    }

    /// Pre-warms all arrays.
    #[cold]
    pub fn prewarm(&mut self) {
        self.prewarm_reset();
        while !self.prewarm_complete() {
            let take = self.prewarm_pending;
            self.prewarm_step(take);
        }
    }
}

#[cfg(test)]
#[path = "../cascade_test.rs"]
mod tests;
