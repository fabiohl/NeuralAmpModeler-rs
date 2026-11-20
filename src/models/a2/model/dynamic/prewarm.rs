// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Pre-warm implementation for the dynamic A2 model.
//!
//! Zeroes internal buffers and feeds `receptive_field_size` frames
//! of silence through `process()` to reach the proper steady state.
//! When a condition_dsp sub-model is present, it is also pre-warmed.
//!
//! The integral pass is implemented over the split primitives:
//! `prewarm_reset()` runs the zero phase (buffer clearing in place plus the
//! condition sub-model's own integral pre-configuration) and the zero-feed
//! is consumed through `prewarm_step` in caller-chosen chunks until
//! `prewarm_pending == 0`. Any chunking reproduces the integral result
//! bit-exactly (the feed is an exact sequence of per-frame `process()` calls).

use crate::models::NamModel;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

use super::super::a2_prewarm_common;
use super::WaveNetA2Dyn;

impl WaveNetA2Dyn {
    /// Zero phase of the deferred split stabilization: clears every buffer in
    /// place and resets the condition sub-model via `prewarm_reset()` (never
    /// running synchronous integral prewarm on the RT thread), arming the
    /// pending RF zero-feed. Zero-allocation, zero-log, and lock-free;
    /// real-time consumers amortize the feed via [`WaveNetA2Dyn::prewarm_step`].
    pub fn prewarm_reset(&mut self) {
        a2_prewarm_common(
            self.num_layers,
            self.receptive_field_size,
            &mut self.layer_buffers,
            &self.layer_ring_sizes,
            &mut self.layer_buffer_starts,
            &mut self.layer_in,
            &mut self.head_accum,
            &mut self.head_write_pos,
        );

        if self.has_weights()
            && let Some(ref mut cond_dsp) = self.condition_dsp
        {
            cond_dsp.prewarm_reset();
        }

        self.prewarm_pending = if self.has_weights() {
            self.receptive_field_size
        } else {
            0
        };
    }

    /// Clears every layer ring, the head accumulator and the nested
    /// condition-DSP state to the freshly-built condition, in place and
    /// without allocating.
    ///
    /// This is the state-clearing half of [`WaveNetA2Dyn::prewarm_reset`]; it is
    /// invoked by `reset` so that a reset without prewarm still removes all
    /// FIR/recurrent memory (C++ `DSP::Reset → SetMaxBufferSize`).
    pub fn clear_state(&mut self) {
        a2_prewarm_common(
            self.num_layers,
            self.receptive_field_size,
            &mut self.layer_buffers,
            &self.layer_ring_sizes,
            &mut self.layer_buffer_starts,
            &mut self.layer_in,
            &mut self.head_accum,
            &mut self.head_write_pos,
        );
        if let Some(ref mut cond_dsp) = self.condition_dsp {
            cond_dsp.clear_state();
        }
    }

    /// Advances the armed stabilization by at most `samples` zeroed samples
    /// and returns the work still pending (0 = converged). Zero-allocation,
    /// zero-log, and lock-free, sharing the same per-frame kernels as
    /// [`Self::process`].
    pub fn prewarm_step(&mut self, samples: usize) -> usize {
        if samples == 0 {
            let cond_rem = self
                .condition_dsp
                .as_mut()
                .filter(|c| !c.prewarm_complete())
                .map_or(0, |c| c.prewarm_step(0).max(1));
            return self.prewarm_pending.saturating_add(cond_rem);
        }

        if self.has_weights()
            && let Some(ref mut cond_dsp) = self.condition_dsp
            && !cond_dsp.prewarm_complete()
        {
            let cond_rem = cond_dsp.prewarm_step(samples);
            if !cond_dsp.prewarm_complete() {
                return self.prewarm_pending.saturating_add(cond_rem.max(1));
            }
            return self.prewarm_pending;
        }

        let n = samples.min(self.prewarm_pending);
        if n > 0 {
            const STACK_FLOATS: usize = 512;
            let in_ch = self.input_channels.max(1);
            let out_ch = self.head_size.max(1);
            let width = in_ch.max(out_ch);
            if width <= STACK_FLOATS {
                let max_chunk_frames = (STACK_FLOATS / width).clamp(1, WAVENET_MAX_NUM_FRAMES);
                let zeros = [0.0f32; STACK_FLOATS];
                let mut discard = [0.0f32; STACK_FLOATS];
                let mut fed = 0usize;
                while fed < n {
                    let nf = (n - fed).min(max_chunk_frames);
                    self.process(&zeros[..nf * in_ch], &mut discard[..nf * out_ch]);
                    fed += nf;
                }
            }
            self.prewarm_pending -= n;
        }
        self.prewarm_pending
    }

    /// Returns `true` when both the outer array and any nested `condition_dsp`
    /// have finished their armed prewarm pass.
    pub fn prewarm_complete(&self) -> bool {
        self.prewarm_pending == 0
            && self
                .condition_dsp
                .as_ref()
                .is_none_or(|c| c.prewarm_complete())
    }

    /// Pending stabilization units (for cascade-chained accounting).
    ///
    /// Exposed `pub(crate)` so the cascade's chained [`prewarm_step`](super::super::WaveNetA2Cascade::prewarm_step)
    /// can account subarray feeds advanced through the chain without driving
    /// a subarray `process` directly (which would feed mono zeros into a
    /// multi-channel residual input).
    pub(crate) fn prewarm_pending_stabilization(&self) -> usize {
        self.prewarm_pending
    }

    /// Advances the armed feed accounting by `n` samples already fed through
    /// the cascade chain (saturating; the chained `process` call advanced the
    /// real state, this only retires the accounting).
    pub(crate) fn prewarm_advance(&mut self, n: usize) {
        self.prewarm_pending = self.prewarm_pending.saturating_sub(n);
    }

    /// Pre-warms the model by filling the receptive field with silence.
    #[cold]
    pub fn prewarm(&mut self) {
        self.prewarm_reset();
        while !self.prewarm_complete() {
            let take = self.prewarm_pending.max(WAVENET_MAX_NUM_FRAMES);
            self.prewarm_step(take);
        }
    }
}
