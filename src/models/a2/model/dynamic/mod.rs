// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 Dynamic model (`WaveNetA2Dyn`).
//!
//! Runtime-dimensioned A2 engine supporting `bottleneck != channels`,
//! heterogeneous activations, gating/blending, head1x1, grouped layer1x1,
//! and the full A2 topology spectrum.

use crate::dsp::mirror_buf::MirroredBuffer;
use crate::math::common::AlignedVec;
use crate::models::StaticModel;
use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::{BlendingActivationConfig, GatingActivationConfig, GatingMode};
use crate::models::a2::head::A2HeadConv;
use crate::models::a2::layer::A2Layer;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;
use serde_json::Value;

#[cfg(any(test, feature = "testing"))]
use crate::testing::diagnostics::{DiagnosticConfig, DiagnosticDump};

/// Off-RT diagnostic state (zero-cost when compiled out).
///
/// In release builds, this is a zero-sized struct; in debug/test builds,
/// it holds the capture config and the accumulated dump for the current
/// `process()` call.
#[derive(Default)]
pub struct DiagnosticState {
    #[cfg(any(test, feature = "testing"))]
    /// Diagnostic capture config (off-RT only).
    pub config: DiagnosticConfig,
    #[cfg(any(test, feature = "testing"))]
    /// Accumulated diagnostic dump for the current `process()` call.
    pub dump: Option<DiagnosticDump>,
}

/// Weight-loading routines (`set_weights` and helpers).
pub mod build;
/// Pre-warm implementation (fills receptive field with silence).
pub mod prewarm;
/// Forward-pass methods (`process`, `process_internal`, per-frame loops).
pub mod process;
/// Cascade pipeline methods (shared with `WaveNetA2Cascade` orchestrator).
pub mod process_cascade;

/// Runtime-dimensioned WaveNet A2 model.
///
/// Supports the full A2 topology spectrum: arbitrary channel counts,
/// bottleneck ≠ channels, heterogeneous activations, gating/blending,
/// grouped convolutions, head1x1, and FiLM.
///
/// Uses per-frame processing for maximum flexibility. Each frame's conv
/// dispatch (standard, grouped, depthwise) is handled by the polymorphic
/// `A2Conv1d::process_single_frame`.
pub struct WaveNetA2Dyn {
    /// Input channel count (1 for standalone mono models;
    /// may be > 1 for cascade sub-arrays receiving multi-channel input).
    pub input_channels: usize,
    /// Output head size (1 for mono A2, may be > 1 for cascade sub-arrays).
    pub head_size: usize,
    /// Inter-layer channel count.
    pub channels: usize,
    /// Internal bottleneck channel count (conv output size).
    pub bottleneck: usize,
    /// Number of layers.
    pub num_layers: usize,

    /// A2 layers (conv, mixin, l1x1, FiLM). Conv is `channels → bottleneck`
    /// (or `channels → 2*bottleneck` if gating/blending).
    /// mixin_w has `bottleneck` elements.
    /// l1x1_w has `bottleneck × channels` elements, l1x1_b has `channels`.
    pub layers: Vec<A2Layer>,

    /// Input rechannel weights (pre-decoded f32).
    pub rechannel_w_f32: AlignedVec<f32>,

    /// Head convolution (K=16 over head accumulator, bias, head_scale).
    /// Used when head_size == 1 (mono output).
    pub head_conv: Option<A2HeadConv>,
    /// Head convolution kernel size (= 16 for A2 canonical, = 1 for legacy models).
    /// Controls weight count, receptive field, and ring buffer keep size.
    pub head_kernel_size: usize,
    /// Head rechannel weights for multi-channel output (head_size > 1).
    /// Layout: per output channel `[K][head_accum_size]` column-major per tap,
    /// concatenated as `head_size * K * head_accum_size` f32, matching
    /// the Conv1D(head_accum_size → 1, K=16, bias, head_scale) per output channel.
    pub head_rechannel_w: AlignedVec<f32>,
    /// Head rechannel per-output-channel bias. Size: `head_size`.
    pub head_rechannel_b: AlignedVec<f32>,
    /// Head rechannel per-output-channel scale. Size: `head_size`.
    pub head_rechannel_scale: AlignedVec<f32>,

    /// Head accumulator ring buffer (channels-wide, pow2 size).
    pub head_accum: AlignedVec<f32>,
    /// Write position in head_accum (in columns, wraps via head_ring_mask).
    pub head_write_pos: usize,
    /// Ring mask for head_accum (pow2 ring, mask = capacity - 1).
    pub head_ring_mask: usize,

    /// Per-layer history buffers: one MirroredBuffer per layer.
    /// Each stores `channels`-wide frames (the layer_in data).
    pub layer_buffers: Vec<MirroredBuffer<f32>>,
    /// Per-layer ring sizes in elements (pow2 page-aligned).
    pub layer_ring_sizes: Vec<usize>,
    /// Per-layer maximum dilation lookback = (kernel-1) * dilation * channels.
    pub layer_lookbacks: Vec<usize>,
    /// Per-layer buffer starts (advanced with each written frame).
    pub layer_buffer_starts: Vec<usize>,

    /// Inter-layer data buffer: `channels × max_buffer_size` f32.
    pub layer_in: AlignedVec<f32>,

    /// Per-layer kernel sizes.
    pub kernel_sizes: Vec<usize>,
    /// Per-layer dilation factors.
    pub dilations: Vec<usize>,

    /// Primary activation per layer.
    pub activations: Vec<ActivationType>,
    /// Gating mode per layer.
    pub gating_modes: Vec<GatingMode>,
    /// Secondary activation per layer (for gating/blending).
    pub secondary_activations: Vec<Option<ActivationType>>,

    /// Pre-allocated gating configs per layer (None when gating_mode != Gated).
    pub gating_configs: Vec<Option<GatingActivationConfig>>,
    /// Pre-allocated blending configs per layer (None when gating_mode != Blended).
    pub blending_configs: Vec<Option<BlendingActivationConfig>>,

    /// Whether head1x1 projection is active for this layer array.
    /// Config flag derived from JSON `head1x1.active`. The weights
    /// themselves are stored per-layer in `A2Layer.head1x1_w` / `.head1x1_b`.
    pub head1x1_active: bool,
    /// Input channel count to each head1x1 layer (= bottleneck / groups).
    /// Per-array config; used during weight loading to size each layer's
    /// head1x1 weight matrices.
    pub head1x1_h1_in: usize,
    /// Number of groups for the input mixin projection across all layers
    /// (C++ `groups_input_mixin`). Per-array config, default 1.
    pub mixin_groups: u32,
    /// Number of groups for the layer1x1 projection across all layers
    /// (C++ `layer1x1.groups`). Per-array config, default 1.
    pub l1x1_groups: u32,
    /// Dimension size of the head accumulator (output size of head1x1 projection).
    pub head_accum_size: usize,
    /// Scratch buffer for head1x1 projection output (channels elements).
    pub head1x1_scratch: AlignedVec<f32>,

    /// Total receptive field: sum of `(kernel-1)*dilation` + head kernel - 1.
    pub receptive_field_size: usize,
    /// Maximum frames per processing block.
    pub max_buffer_size: usize,

    /// Raw JSON for the single layer array.
    pub layer_raw: Option<Value>,

    /// Condition vector size (default 1 for A2 fast-path equivalence).
    pub condition_size: usize,

    /// Per-frame scratch buffer for conv output (2*bottleneck elements).
    /// When gating/blending is active, the conv output is 2× wide.
    pub z_scratch: AlignedVec<f32>,

    /// Scratch buffer for isolated mixin output before FiLM (2*bottleneck elements).
    pub mixin_scratch: AlignedVec<f32>,

    /// Scratch buffer for layer1x1 projection output (channels elements).
    pub l1x1_scratch: AlignedVec<f32>,
    /// Scratch buffer for input_mixin_pre_film condition modulation (condition_size elements).
    pub cond_scratch: AlignedVec<f32>,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,

    /// Optional nested condition DSP sub-model (C++ `_condition_dsp`).
    ///
    /// Built eagerly during model construction from the `condition_dsp` JSON
    /// object. Its `process()` is called with mono audio input; its multi-channel
    /// output replaces the raw input as the `condition` parameter passed to
    /// per-layer FiLM/mixin. When `None`, the raw input is used as `condition`
    /// (passthrough, behavior equivalent to A2 fast-path).
    pub condition_dsp: Option<Box<StaticModel>>,
    /// Pre-allocated output buffer for condition_dsp processing.
    ///
    /// Size: `condition_size × WAVENET_MAX_NUM_FRAMES`.
    pub condition_dsp_output: AlignedVec<f32>,

    /// Off-RT diagnostic capture state (zero-cost in release builds).
    pub diag: DiagnosticState,
}

impl WaveNetA2Dyn {
    /// Creates a new uninitialized dynamic A2 model.
    ///
    /// Allocates ring buffers (MirroredBuffer per layer, pow2 head accumulator)
    /// sized for the architecture and computes the receptive field.
    ///
    /// `kernel_sizes` and `dilations` must have length `num_layers`.
    /// The activation config vectors must also have length `num_layers`.
    #[expect(
        clippy::too_many_arguments,
        reason = "A2 dynamic model constructor requiring many topology parameters for runtime-adaptive neural network initialization"
    )]
    pub fn new(
        input_channels: usize,
        channels: usize,
        bottleneck: usize,
        head_size: usize,
        head_accum_size: usize,
        head1x1_h1_in: usize,
        head_kernel_size: usize,
        kernel_sizes: &[usize],
        dilations: &[usize],
        activations: Vec<ActivationType>,
        gating_modes: Vec<GatingMode>,
        secondary_activations: Vec<Option<ActivationType>>,
    ) -> anyhow::Result<Self> {
        let num_layers = kernel_sizes.len();
        assert_eq!(dilations.len(), num_layers);
        assert_eq!(activations.len(), num_layers);
        assert_eq!(gating_modes.len(), num_layers);
        assert_eq!(secondary_activations.len(), num_layers);

        let max_buf = WAVENET_MAX_NUM_FRAMES;

        let mut rf = 0usize;
        for i in 0..num_layers {
            rf += (kernel_sizes[i] - 1) * dilations[i];
        }
        rf += head_kernel_size.saturating_sub(1);

        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        let head_ring_mask = head_ring_size - 1;

        let mut layer_buffers = Vec::with_capacity(num_layers);
        let mut layer_ring_sizes = Vec::with_capacity(num_layers);
        let mut layer_lookbacks = Vec::with_capacity(num_layers);
        let mut layer_buffer_starts = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let max_lookback = (kernel_sizes[i] - 1) * dilations[i];
            let cap = max_lookback + max_buf + 1;
            let mb = MirroredBuffer::<f32>::new(cap * channels)?;
            let ring_size = mb.size();
            layer_buffers.push(mb);
            layer_ring_sizes.push(ring_size);
            layer_lookbacks.push(max_lookback * channels);
            layer_buffer_starts.push(ring_size);
        }

        // Pre-allocate gating/blending configs.
        let mut gating_configs = Vec::with_capacity(num_layers);
        let mut blending_configs = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let gc = match gating_modes[i] {
                GatingMode::Gated => {
                    let sec = secondary_activations[i]
                        .clone()
                        .unwrap_or(ActivationType::Sigmoid);
                    Some(GatingActivationConfig::new(activations[i].clone(), sec))
                }
                _ => None,
            };
            let bc = match gating_modes[i] {
                GatingMode::Blended => {
                    let sec = secondary_activations[i]
                        .clone()
                        .unwrap_or(ActivationType::Sigmoid);
                    Some(BlendingActivationConfig::new(
                        activations[i].clone(),
                        sec,
                        bottleneck,
                    )?)
                }
                _ => None,
            };
            gating_configs.push(gc);
            blending_configs.push(bc);
        }

        let head1x1_scratch = AlignedVec::new(head_accum_size, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");

        Ok(Self {
            input_channels,
            head_size,
            head_accum_size,
            channels,
            bottleneck,
            num_layers,
            layers: Vec::with_capacity(num_layers),
            rechannel_w_f32: AlignedVec::new(input_channels * channels, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            head_conv: None,
            head_kernel_size,
            head_rechannel_w: AlignedVec::new(
                head_size.max(1) * head_kernel_size * head_accum_size,
                0.0f32,
            )
            .expect("allocation should succeed for test-sized buffers"),
            head_rechannel_b: AlignedVec::new(head_size.max(1), 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            head_rechannel_scale: AlignedVec::new(head_size.max(1), 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            head_accum: AlignedVec::new(head_ring_size * head_accum_size, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            head_write_pos: rf,
            head_ring_mask,
            layer_buffers,
            layer_ring_sizes,
            layer_lookbacks,
            layer_buffer_starts,
            layer_in: AlignedVec::new(channels * max_buf, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            kernel_sizes: kernel_sizes.to_vec(),
            dilations: dilations.to_vec(),
            activations,
            gating_modes,
            secondary_activations,
            gating_configs,
            blending_configs,
            head1x1_active: false,
            head1x1_h1_in,
            mixin_groups: 1,
            l1x1_groups: 1,
            receptive_field_size: rf,
            max_buffer_size: max_buf,
            layer_raw: None,
            condition_size: 1,
            z_scratch: AlignedVec::new(bottleneck * 2, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            mixin_scratch: AlignedVec::new(bottleneck * 2, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            l1x1_scratch: AlignedVec::new(channels, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            cond_scratch: AlignedVec::new(1, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            head1x1_scratch,
            prewarm_on_reset: true,
            condition_dsp: None,
            condition_dsp_output: AlignedVec::new(0, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            diag: DiagnosticState::default(),
        })
    }

    /// Returns the channel count.
    #[inline(always)]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the total receptive field size.
    #[inline(always)]
    pub fn receptive_field(&self) -> usize {
        self.receptive_field_size
    }

    /// Stores the raw layer JSON for FiLM config parsing during weight loading.
    pub fn set_layer_raw(&mut self, raw: Option<Value>) {
        self.layer_raw = raw;
    }

    /// Sets the optional condition DSP sub-model and allocates its output buffer.
    ///
    /// When `condition_dsp` is `Some`, the sub-model pre-processes the audio input
    /// before the main layer loop. Its output channels replace the raw input as
    /// the `condition` parameter for per-layer mixin and FiLM.
    ///
    /// `max_buf` is the maximum number of frames per processing block
    /// (typically [`WAVENET_MAX_NUM_FRAMES`]).
    pub fn set_condition_dsp(&mut self, cond_dsp: Option<Box<StaticModel>>, max_buf: usize) {
        let cond_size = if cond_dsp.is_some() {
            self.condition_size
        } else {
            0
        };
        self.condition_dsp_output = AlignedVec::new(cond_size * max_buf, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        self.condition_dsp = cond_dsp;
    }

    /// Returns whether weights have been loaded.
    #[inline(always)]
    pub fn has_weights(&self) -> bool {
        !self.layers.is_empty()
    }

    /// Enables diagnostic capture for the next `process()` call.
    ///
    /// Allocates a new `DiagnosticDump`. Not compiled in release builds.
    #[cfg(any(test, feature = "testing"))]
    pub fn enable_diagnostics(&mut self, config: DiagnosticConfig) {
        self.diag.config = config;
        self.diag.dump = Some(DiagnosticDump::new(0));
    }

    /// Takes the accumulated diagnostic dump, resetting internal state.
    #[cfg(any(test, feature = "testing"))]
    pub fn take_diagnostics(&mut self) -> Option<DiagnosticDump> {
        self.diag.dump.take()
    }

    /// Reallocates internal buffers to support the given maximum block size.
    ///
    /// If `max_buf` is smaller than the current capacity, this is a no-op.
    /// If `max_buf` equals the current capacity, state variables are reset
    /// and all buffers are zero-filled in-place — avoiding heap allocation
    /// on the RT thread (F9 / RT-Safety).
    pub fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        if max_buf < self.max_buffer_size {
            return Ok(());
        }
        if max_buf == self.max_buffer_size {
            let rf = self.receptive_field_size;
            let ha_len = self.head_accum.len();
            self.head_accum[..ha_len].fill(0.0);
            self.head_write_pos = rf;
            for buf in self.layer_buffers.iter_mut() {
                let len = buf.size();
                buf[..len].fill(0.0);
            }
            self.layer_buffer_starts
                .copy_from_slice(&self.layer_ring_sizes);
            let li_len = self.layer_in.len();
            self.layer_in[..li_len].fill(0.0);
            let cd_len = self.condition_dsp_output.len();
            self.condition_dsp_output[..cd_len].fill(0.0);
            return Ok(());
        }
        self.max_buffer_size = max_buf;
        let rf = self.receptive_field_size;
        let channels = self.channels;

        self.layer_buffers.clear();
        self.layer_ring_sizes.clear();
        self.layer_lookbacks.clear();
        self.layer_buffer_starts.clear();

        for i in 0..self.num_layers {
            let max_lookback = (self.kernel_sizes[i] - 1) * self.dilations[i];
            let cap = max_lookback + max_buf + 1;
            let mb = MirroredBuffer::<f32>::new(cap * channels)?;
            let ring_size = mb.size();
            self.layer_buffers.push(mb);
            self.layer_ring_sizes.push(ring_size);
            self.layer_lookbacks.push(max_lookback * channels);
            self.layer_buffer_starts.push(ring_size);
        }

        self.layer_in = AlignedVec::new(channels * max_buf, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");

        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        self.head_ring_mask = head_ring_size - 1;
        self.head_accum = AlignedVec::new(head_ring_size * self.head_accum_size, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        self.head_write_pos = rf;

        let cond_output_size = self.condition_size * max_buf;
        self.condition_dsp_output = AlignedVec::new(cond_output_size, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");

        Ok(())
    }

    /// Resets internal state for a new sample rate and max buffer size.
    pub fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.set_max_buffer_size(max_buffer_size)?;
        if self.prewarm_on_reset {
            self.prewarm();
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../dynamic_test.rs"]
mod tests;
