// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Architecture.
//!
//! This module isolates A2 architecture components (v0.6+), including
//! activations, FiLM, gating, parameters, and A2 models.
//!
//! ## Status
//!
//! The existing scaffolding includes complete parameter definitions and
//! architectural constants (mirroring `a2_fast.h`). The A2-Full/Lite
//! fast-path uses a subset of these structs; FiLM, gating (`Gated`/`Blended`),
//! `head1x1`, `bottleneck ≠ channels`, and heterogeneous activation fields
//! are reserved for the future general A2 engine.

pub mod activations;
pub mod conv1d;
pub mod conv1d_ch;
pub mod conv1d_ch3;
pub mod conv1d_ch8;
pub mod conv1d_fallback;
pub mod film;
pub mod gating;
pub mod grouped_conv1d;
pub mod head;
pub mod layer;
pub mod model;
pub mod params;
pub mod weights_layout;
/// Public re-exports for easy access.
pub use activations::{ActivationFn, ActivationType};
pub use conv1d::A2Conv1d;
pub use film::{FiLMConfig, FiLMLayer, FilmBlock};
pub use gating::GatingMode;
pub use grouped_conv1d::{
    A2GroupedConv1d, grouped_conv1d_block_ref, grouped_conv1d_single_frame_ref,
};
pub use head::{
    A2HeadConv, a2_head_block_scalar_ref, a2_head_single_frame_scalar_ref, head_process_ch3_sse,
    head_process_ch8_avx2,
};
pub use layer::{A2ConvCh, A2Layer, a2_layer_single_frame_scalar_ref};
pub use model::WaveNetA2;
pub use model::cascade::WaveNetA2Cascade;
pub use model::dynamic::WaveNetA2Dyn;
pub use params::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS,
    A2_VALID_CHANNELS, HeadParams, LayerArrayParamsA2, LayerParamsA2,
};

/// Total weight count for `WaveNetA2<CH>`.
///
/// Formula: `rechannel(ch) + Σ(kernel sizes: ch²·k + ch + ch + ch² + ch) + head(16·ch + 2)`.
/// This is the single canonical source — topology changes require updates here only.
pub const fn a2_weight_count<const CH: usize>() -> usize {
    let mut total = CH; // rechannel_w
    let mut i = 0;
    while i < A2_NUM_LAYERS {
        let k = params::A2_KERNEL_SIZES[i];
        total += CH * CH * k; // conv_w
        total += CH; // conv_b
        total += CH; // mixin_w
        total += CH * CH; // l1x1_w
        total += CH; // l1x1_b
        i += 1;
    }
    total += A2_HEAD_KERNEL_SIZE * CH; // head_w
    total += 1; // head_b
    total += 1; // head_scale
    total
}

use crate::models::NamModel;
use crate::models::sealed;

impl<const CH: usize> sealed::Sealed for model::WaveNetA2<CH> {}

impl<const CH: usize> NamModel for model::WaveNetA2<CH> {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }

    fn prewarm_samples(&self) -> usize {
        self.receptive_field_size
    }

    fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        self.set_max_buffer_size(max_buf)
    }

    fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.reset(_sample_rate, max_buffer_size)
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    fn set_prewarm_on_reset(&mut self, val: bool) {
        self.prewarm_on_reset = val;
    }
}

impl sealed::Sealed for model::dynamic::WaveNetA2Dyn {}

impl NamModel for model::dynamic::WaveNetA2Dyn {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }

    fn prewarm_samples(&self) -> usize {
        self.receptive_field_size
    }

    fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        self.set_max_buffer_size(max_buf)
    }

    fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.reset(_sample_rate, max_buffer_size)
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    fn set_prewarm_on_reset(&mut self, val: bool) {
        self.prewarm_on_reset = val;
    }
}

impl sealed::Sealed for model::cascade::WaveNetA2Cascade {}

impl NamModel for model::cascade::WaveNetA2Cascade {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }

    fn prewarm_samples(&self) -> usize {
        self.receptive_field_size
    }

    fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        self.set_max_buffer_size(max_buf)
    }

    fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.reset(_sample_rate, max_buffer_size)
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    fn set_prewarm_on_reset(&mut self, val: bool) {
        self.prewarm_on_reset = val;
    }
}

#[cfg(test)]
mod a2_test;
