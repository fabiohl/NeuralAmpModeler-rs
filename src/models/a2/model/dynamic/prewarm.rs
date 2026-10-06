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

use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

use super::super::a2_prewarm_common;
use super::WaveNetA2Dyn;

impl WaveNetA2Dyn {
    /// Zero phase of the deferred split stabilization: clears every buffer in
    /// place and primes the condition sub-model (the integral pass's
    /// preamble), arming the pending RF zero-feed without running it.
    /// Zero-allocation and lock-free; real-time consumers amortize the feed
    /// via [`WaveNetA2Dyn::prewarm_step`].
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
            crate::models::NamModel::prewarm(&mut **cond_dsp, 0);
        }

        self.prewarm_pending = if self.has_weights() {
            self.receptive_field_size
        } else {
            0
        };
    }

    /// Advances the armed stabilization by at most `samples` zeroed samples
    /// and returns the work still pending (0 = converged). Zero-allocation
    /// and lock-free, sharing the same per-frame kernels as [`Self::process`].
    pub fn prewarm_step(&mut self, samples: usize) -> usize {
        let n = samples.min(self.prewarm_pending);
        if n > 0 {
            let zeros = [0.0f32; WAVENET_MAX_NUM_FRAMES];
            let mut discard = [0.0f32; WAVENET_MAX_NUM_FRAMES];
            let mut fed = 0usize;
            while fed < n {
                let nf = (n - fed).min(WAVENET_MAX_NUM_FRAMES);
                self.process(&zeros[..nf], &mut discard[..nf]);
                fed += nf;
            }
            self.prewarm_pending -= n;
        }
        self.prewarm_pending
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
        while self.prewarm_pending > 0 {
            let take = self.prewarm_pending;
            self.prewarm_step(take);
        }
    }
}
