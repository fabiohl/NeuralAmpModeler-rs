// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Pre-warm implementation for the static A2 model.
//!
//! Mirrors `A2FastModel::prewarm()` in `a2_fast.cpp`: zeroes all buffers,
//! then feeds `receptive_field_size` frames of zero input through `process()`
//! so that layer biases populate the head accumulator ring — matching the
//! C++ steady-state initial condition.
//!
//! ## Root cause of initial A2 Rust×C++ divergence
//!
//! The initial Rust port diverged from C++ because `prewarm()` only zeroed buffers
//! while the C++ `A2FastModel::prewarm()` feeds `_prewarm_samples` frames of silence
//! through `process()`. Even with zero input, A2 layers produce non-zero activations
//! (conv bias + mixin × 0 + LeakyReLU), populating the head accumulator ring.
//! The Rust zero-fill vs C++ silent-process mismatch meant zero initial state in Rust
//! vs bias-driven steady state in C++ — causing frame-level divergence from the first
//! output sample. Fixed by making Rust `prewarm()` run `process()` with zeros for
//! `receptive_field_size` frames after the zero-fill, mirroring the C++ DSP::Reset
//! → prewarm() flow exactly.
//!
//! The zero-feed is an exact sequence of per-frame `process()` calls, so the
//! integral pass is implemented over the split primitives
//! (`prewarm_reset` + `prewarm_step` until `prewarm_pending == 0`); any
//! chunking of the zeroed frames reproduces the integral result bit-exactly.

use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

use super::super::super::params::A2_NUM_LAYERS;
use super::super::a2_prewarm_common;
use super::WaveNetA2;

impl<const CH: usize> WaveNetA2<CH> {
    /// Zero phase of the deferred split stabilization: clears every buffer in
    /// place (the integral prewarm's preamble) and arms the pending
    /// zeroed-sample feed without running it. Zero-allocation and lock-free:
    /// a real-time consumer may call it on its audio thread and amortize
    /// [`WaveNetA2::prewarm_step`] within per-callback budgets.
    pub fn prewarm_reset(&mut self) {
        a2_prewarm_common(
            A2_NUM_LAYERS,
            self.receptive_field_size,
            &mut self.layer_buffers,
            &self.layer_ring_sizes,
            &mut self.layer_buffer_starts,
            &mut self.layer_in,
            &mut self.head_accum,
            &mut self.head_write_pos,
        );

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
