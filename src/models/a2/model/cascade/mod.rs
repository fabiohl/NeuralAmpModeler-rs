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
        })
    }

    /// Compatibility wrapper over [`try_new`](Self::try_new) for external
    /// callers that pre-date the fallible constructor (H-05). The loader path
    /// uses [`try_new`](Self::try_new) so allocation failures propagate as
    /// errors instead of aborting the process.
    ///
    /// # Panics
    /// Panics if a cascade scratch allocation fails.
    pub fn new(
        arrays: Vec<WaveNetA2Dyn>,
        condition_dsp: Option<Box<StaticModel>>,
        condition_size: usize,
    ) -> Self {
        Self::try_new(arrays, condition_dsp, condition_size)
            .expect("WaveNetA2Cascade allocation should succeed for fixed-size scratch buffers")
    }

    /// Returns the number of channels of the first array.
    pub fn channels(&self) -> usize {
        self.arrays.first().map(|a| a.channels).unwrap_or(0)
    }

    /// Full forward pass through the cascade.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
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

        output[..total].fill(0.0);
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
            last.cascade_head_finalize(nf, &mut output[pos..pos + nf * last.head_size]);

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

    /// Pre-warms all arrays.
    #[cold]
    pub fn prewarm(&mut self) {
        let zeros = vec![0.0f32; self.receptive_field_size.max(2048)];
        let mut dummy = vec![0.0f32; zeros.len()];
        self.process(&zeros, &mut dummy);
    }
}
