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

    /// Zero phase of the deferred split flow (buffer clearing in place; the
    /// RF zero-feed pends). Real-time safe.
    fn prewarm_reset(&mut self) {
        self.prewarm_reset();
    }

    /// Advances the armed RF zero-feed by up to `samples` zeroed samples;
    /// returns the remaining count. Real-time safe.
    fn prewarm_step(&mut self, samples: usize) -> usize {
        self.prewarm_step(samples)
    }

    /// Deferred pass pending? (`true` when nothing is armed/left.)
    fn prewarm_complete(&self) -> bool {
        self.prewarm_pending == 0
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

    /// Clears all A2 layer rings and the head accumulator (see
    /// [`WaveNetA2::clear_state`](model::WaveNetA2::clear_state)).
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

impl sealed::Sealed for model::dynamic::WaveNetA2Dyn {}

impl NamModel for model::dynamic::WaveNetA2Dyn {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    /// Input width consumed per frame by [`process`](Self::process)
    /// (mono for standalone models; wider for cascade sub-arrays fed by a
    /// multi-channel residual).
    fn num_input_channels(&self) -> usize {
        self.input_channels.max(1)
    }

    /// Output width written per frame by [`process`](Self::process):
    /// the `head_size` interleaved channels (mirrors the `out_ch` clamp in
    /// `process_internal`).
    fn num_output_channels(&self) -> usize {
        self.head_size.max(1)
    }

    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }

    /// Zero phase of the deferred split flow (buffer clearing + condition
    /// sub-model pre-configuration; the RF zero-feed pends). Real-time safe.
    fn prewarm_reset(&mut self) {
        self.prewarm_reset();
    }

    /// Advances the armed RF zero-feed by up to `samples` zeroed samples;
    /// returns the remaining count. Real-time safe.
    fn prewarm_step(&mut self, samples: usize) -> usize {
        self.prewarm_step(samples)
    }

    /// Deferred pass pending? (`true` when nothing is armed/left.)
    fn prewarm_complete(&self) -> bool {
        self.prewarm_pending == 0
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

    /// Clears all A2 layer rings, the head accumulator and the nested
    /// condition-DSP state (see
    /// [`WaveNetA2Dyn::clear_state`](model::dynamic::WaveNetA2Dyn::clear_state)).
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

impl sealed::Sealed for model::cascade::WaveNetA2Cascade {}

impl NamModel for model::cascade::WaveNetA2Cascade {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    /// Mono input; the cascade always consumes one sample per frame.
    fn num_input_channels(&self) -> usize {
        1
    }

    /// Output width written per frame by [`process`](Self::process):
    /// the last array's `head_size` (mirrors the `out_per_frame` clamp in
    /// `process_internal`).
    fn num_output_channels(&self) -> usize {
        self.arrays.last().map(|a| a.head_size).unwrap_or(1).max(1)
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

    /// Clears every subarray's state and the cascade scratch buffers (see
    /// [`WaveNetA2Cascade::clear_state`](model::cascade::WaveNetA2Cascade::clear_state)).
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

#[cfg(test)]
mod a2_test;
