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
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }

    fn prewarm_samples(&self) -> usize {
        self.receptive_field_size + 1
    }

    fn set_max_buffer_size(&mut self, _max_buf: usize) -> anyhow::Result<()> {
        Ok(())
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    fn set_prewarm_on_reset(&mut self, val: bool) {
        self.prewarm_on_reset = val;
    }
}
