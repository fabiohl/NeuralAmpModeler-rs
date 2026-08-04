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

use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

use super::super::super::params::A2_NUM_LAYERS;
use super::super::a2_prewarm_common;
use super::WaveNetA2;

impl<const CH: usize> WaveNetA2<CH> {
    /// Pre-warms the model by filling the receptive field with silence.
    #[cold]
    pub fn prewarm(&mut self) {
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

        if self.has_weights() {
            let prewarm_samples = self.receptive_field_size;
            let block = WAVENET_MAX_NUM_FRAMES;
            let zeros = [0.0f32; WAVENET_MAX_NUM_FRAMES];
            let mut discard = [0.0f32; WAVENET_MAX_NUM_FRAMES];
            let mut remaining = prewarm_samples;
            while remaining > 0 {
                let nf = remaining.min(block);
                self.process(&zeros[..nf], &mut discard[..nf]);
                remaining -= nf;
            }
        }
    }
}
