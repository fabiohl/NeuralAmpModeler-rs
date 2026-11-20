// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! ConvNet feed-forward architecture (F4).
//!
//! Layers: Conv1D → BatchNorm → Activation, chained sequentially
//! with an optional post-stack head.

pub mod batch_norm;
pub mod block;
pub mod model;

pub use batch_norm::BatchNorm1D;
pub use block::ConvNetBlock;
pub use model::{ConvNetModel, LinearHead};

use super::NamModel;
use super::sealed;

impl sealed::Sealed for ConvNetModel {}

impl NamModel for ConvNetModel {
    fn num_input_channels(&self) -> usize {
        self.in_channels()
    }

    fn num_output_channels(&self) -> usize {
        self.out_channels()
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }

    /// Zero phase of the deferred split flow: clears every block's delay-line
    /// state and arms `receptive_field_size + 1` zero-feed samples.
    fn prewarm_reset(&mut self) {
        self.prewarm_reset();
    }

    /// Advances the armed stabilization by up to `samples` zeroed samples
    /// without heap allocation; returns the remaining sample count.
    fn prewarm_step(&mut self, samples: usize) -> usize {
        self.prewarm_step(samples)
    }

    /// Deferred pass pending? (`true` when nothing is armed/left.)
    fn prewarm_complete(&self) -> bool {
        self.prewarm_complete()
    }

    fn prewarm_samples(&self) -> usize {
        self.receptive_field_size + 1
    }

    fn set_max_buffer_size(&mut self, _max_buf: usize) -> anyhow::Result<()> {
        Ok(())
    }

    /// Clears every block's causal delay-line state (see
    /// [`ConvNetModel::clear_state`]); called by the default [`reset`](NamModel::reset)
    /// before the optional stabilization pass.
    fn clear_state(&mut self) {
        self.clear_state();
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    fn set_prewarm_on_reset(&mut self, val: bool) {
        self.prewarm_on_reset = val;
    }
}
