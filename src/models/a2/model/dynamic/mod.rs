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
    #[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
    /// Diagnostic capture config (off-RT only).
    pub config: DiagnosticConfig,
    #[cfg(any(test, feature = "testing"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
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
/// Per-frame dynamic processing kernel (`process_frame_dyn`).
pub mod process_frame;
mod update_weights;

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
    // Prepared by load_weights_inner. Reloads invalidate it before changing storage.
    pub(super) runtime_weight_layout: Result<usize, super::A2WeightUpdateError>,
    /// Input channel count (1 for standalone mono models;
    /// may be > 1 for cascade sub-arrays receiving multi-channel input).
    pub(crate) input_channels: usize,
    /// Output head size (1 for mono A2, may be > 1 for cascade sub-arrays).
    pub(crate) head_size: usize,
    /// Inter-layer channel count.
    pub(crate) channels: usize,
    /// Internal bottleneck channel count (conv output size).
    pub(crate) bottleneck: usize,
    /// Number of layers.
    pub(crate) num_layers: usize,

    /// A2 layers (conv, mixin, l1x1, FiLM). Conv is `channels → bottleneck`
    /// (or `channels → 2*bottleneck` if gating/blending).
    /// mixin_w has `bottleneck` elements.
    /// l1x1_w has `bottleneck × channels` elements, l1x1_b has `channels`.
    pub(crate) layers: Vec<A2Layer>,

    /// Input rechannel weights (pre-decoded f32).
    pub(crate) rechannel_w_f32: AlignedVec<f32>,

    /// Head convolution (K=16 over head accumulator, bias, head_scale).
    /// Used when head_size == 1 (mono output).
    pub(crate) head_conv: Option<A2HeadConv>,
    /// Head convolution kernel size (= 16 for A2 canonical, = 1 for legacy models).
    /// Controls weight count, receptive field, and ring buffer keep size.
    pub(crate) head_kernel_size: usize,
    /// Head rechannel weights for multi-channel output (head_size > 1).
    /// Layout: per output channel `[K][head_accum_size]` column-major per tap,
    /// concatenated as `head_size * K * head_accum_size` f32, matching
    /// the Conv1D(head_accum_size → 1, K=16, bias, head_scale) per output channel.
    pub(crate) head_rechannel_w: AlignedVec<f32>,
    /// Head rechannel per-output-channel bias. Size: `head_size`.
    pub(crate) head_rechannel_b: AlignedVec<f32>,
    /// Head rechannel per-output-channel scale. Size: `head_size`.
    pub(crate) head_rechannel_scale: AlignedVec<f32>,
    /// Whether head rechannel convolution has a bias term.
    pub(crate) head_bias: bool,

    /// Head accumulator ring buffer (channels-wide, pow2 size).
    pub(crate) head_accum: AlignedVec<f32>,
    /// Write position in head_accum (in columns, wraps via head_ring_mask).
    pub(crate) head_write_pos: usize,
    /// Ring mask for head_accum (pow2 ring, mask = capacity - 1).
    pub(crate) head_ring_mask: usize,

    /// Per-layer history buffers: one MirroredBuffer per layer.
    /// Each stores `channels`-wide frames (the layer_in data).
    pub(crate) layer_buffers: Vec<MirroredBuffer<f32>>,
    /// Per-layer ring sizes in elements (pow2 page-aligned).
    pub(crate) layer_ring_sizes: Vec<usize>,
    /// Per-layer maximum dilation lookback = (kernel-1) * dilation * channels.
    pub(crate) layer_lookbacks: Vec<usize>,
    /// Per-layer buffer starts (advanced with each written frame).
    pub(crate) layer_buffer_starts: Vec<usize>,

    /// Inter-layer data buffer: `channels × max_buffer_size` f32.
    pub(crate) layer_in: AlignedVec<f32>,

    /// Per-layer kernel sizes.
    pub(crate) kernel_sizes: Vec<usize>,
    /// Per-layer dilation factors.
    pub(crate) dilations: Vec<usize>,

    /// Primary activation per layer.
    pub(crate) activations: Vec<ActivationType>,
    /// Gating mode per layer.
    pub(crate) gating_modes: Vec<GatingMode>,
    /// Secondary activation per layer (for gating/blending).
    pub(crate) secondary_activations: Vec<Option<ActivationType>>,

    /// Pre-allocated gating configs per layer (None when gating_mode != Gated).
    pub(crate) gating_configs: Vec<Option<GatingActivationConfig>>,
    /// Pre-allocated blending configs per layer (None when gating_mode != Blended).
    pub(crate) blending_configs: Vec<Option<BlendingActivationConfig>>,

    /// Whether head1x1 projection is active for this layer array.
    /// Config flag derived from JSON `head1x1.active`. The weights
    /// themselves are stored per-layer in `A2Layer.head1x1_w` / `.head1x1_b`.
    pub(crate) head1x1_active: bool,
    /// Input channel count to each head1x1 layer (= bottleneck / groups).
    /// Per-array config; used during weight loading to size each layer's
    /// head1x1 weight matrices.
    pub(crate) head1x1_h1_in: usize,
    /// Number of groups for the input mixin projection across all layers
    /// (C++ `groups_input_mixin`). Per-array config, default 1.
    pub(crate) mixin_groups: u32,
    /// Number of groups for the layer1x1 projection across all layers
    /// (C++ `layer1x1.groups`). Per-array config, default 1.
    pub(crate) l1x1_groups: u32,
    /// Number of groups for the input convolution across all layers
    /// (C++ `groups_input`). Per-array config, default 1.
    pub(crate) groups_input: u32,
    /// Dimension size of the head accumulator (output size of head1x1 projection).
    pub(crate) head_accum_size: usize,
    /// Scratch buffer for head1x1 projection output (channels elements).
    pub(crate) head1x1_scratch: AlignedVec<f32>,

    /// Total receptive field: sum of `(kernel-1)*dilation` + head kernel - 1.
    pub(crate) receptive_field_size: usize,
    /// Maximum frames per processing block.
    pub(crate) max_buffer_size: usize,
    /// Largest block size this instance has been sized for. Distinct from
    /// [`Self::max_buffer_size`], which stays capped at
    /// `WAVENET_MAX_NUM_FRAMES` because the internal rings never inflate.
    /// Tracking the negotiated maximum makes a later smaller request (e.g.
    /// the host block after the loader's `MAX_RESAMP_BUF` sizing) a true
    /// no-op, so it cannot silently wipe stabilized state (including the
    /// nested `condition_dsp`).
    pub(crate) negotiated_max_buffer_size: usize,

    /// Raw JSON for the single layer array.
    pub(crate) layer_raw: Option<Value>,

    /// Condition vector size (default 1 for A2 fast-path equivalence).
    pub(crate) condition_size: usize,

    /// Per-frame scratch buffer for conv output (2*bottleneck elements).
    /// When gating/blending is active, the conv output is 2× wide.
    pub(crate) z_scratch: AlignedVec<f32>,

    /// Scratch buffer for isolated mixin output before FiLM (2*bottleneck elements).
    pub(crate) mixin_scratch: AlignedVec<f32>,

    /// Scratch buffer for layer1x1 projection output (channels elements).
    pub(crate) l1x1_scratch: AlignedVec<f32>,
    /// Scratch buffer for input_mixin_pre_film condition modulation (condition_size elements).
    pub(crate) cond_scratch: AlignedVec<f32>,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub(crate) prewarm_on_reset: bool,

    /// Optional nested condition DSP sub-model (C++ `_condition_dsp`).
    ///
    /// Built eagerly during model construction from the `condition_dsp` JSON
    /// object. Its `process()` is called with mono audio input; its multi-channel
    /// output replaces the raw input as the `condition` parameter passed to
    /// per-layer FiLM/mixin. When `None`, the raw input is used as `condition`
    /// (passthrough, behavior equivalent to A2 fast-path).
    pub(crate) condition_dsp: Option<Box<StaticModel>>,
    /// Pre-allocated output buffer for condition_dsp processing.
    ///
    /// Size: `condition_size × WAVENET_MAX_NUM_FRAMES`.
    pub(crate) condition_dsp_output: AlignedVec<f32>,

    /// Off-RT diagnostic capture state (zero-cost in release builds).
    #[cfg_attr(
        not(any(test, feature = "testing")),
        expect(
            dead_code,
            reason = "DiagnosticState is read only when test/testing is enabled"
        )
    )]
    pub(crate) diag: DiagnosticState,
    /// Zeroed-sample stabilization work still pending for the deferred split
    /// pass armed by [`Self::prewarm_reset`](super::super::NamModel::prewarm_reset).
    /// Always `0` for freshly built models; the integral
    /// [`Self::prewarm`](super::super::NamModel::prewarm) /
    /// [`Self::reset`](super::super::NamModel::reset) paths complete it by
    /// construction and never consult or alter it otherwise.
    pub(crate) prewarm_pending: usize,
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
        // F-02: diverging topology vector lengths are pre-condition errors
        // (previously `assert_eq!` — panicked across the loader API).
        if dilations.len() != num_layers {
            anyhow::bail!(
                "WaveNetA2Dyn::new: dilations length ({}) != kernel_sizes length ({num_layers})",
                dilations.len()
            );
        }
        if activations.len() != num_layers {
            anyhow::bail!(
                "WaveNetA2Dyn::new: activations length ({}) != kernel_sizes length ({num_layers})",
                activations.len()
            );
        }
        if gating_modes.len() != num_layers {
            anyhow::bail!(
                "WaveNetA2Dyn::new: gating_modes length ({}) != kernel_sizes length ({num_layers})",
                gating_modes.len()
            );
        }
        if secondary_activations.len() != num_layers {
            anyhow::bail!(
                "WaveNetA2Dyn::new: secondary_activations length ({}) != kernel_sizes length ({num_layers})",
                secondary_activations.len()
            );
        }

        let max_buf = WAVENET_MAX_NUM_FRAMES;

        // F-03/F-04: checked arithmetic and layer caps before any allocation.
        let mut rf = 0usize;
        let mut per_layer_lookbacks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let kernel = kernel_sizes[i];
            let dilation = dilations[i];
            let lookback = kernel
                .checked_sub(1)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "WaveNetA2Dyn::new: kernel_sizes[{i}] must be >= 1, got {kernel}"
                    )
                })?
                .checked_mul(dilation)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "WaveNetA2Dyn::new: (kernel_sizes[{i}] - 1) * dilations[{i}] overflows usize"
                    )
                })?;
            rf = rf.checked_add(lookback).ok_or_else(|| {
                anyhow::anyhow!("WaveNetA2Dyn::new: receptive field sum overflows usize")
            })?;
            per_layer_lookbacks.push(lookback);
        }
        rf += head_kernel_size.saturating_sub(1);

        let head_ring_cap = rf.checked_add(max_buf + 1).ok_or_else(|| {
            anyhow::anyhow!("WaveNetA2Dyn::new: head ring capacity overflows usize")
        })?;
        let head_ring_size = head_ring_cap.next_power_of_two();
        let head_ring_mask = head_ring_size - 1;

        let mut layer_buffers = Vec::with_capacity(num_layers);
        let mut layer_ring_sizes = Vec::with_capacity(num_layers);
        let mut layer_lookbacks = Vec::with_capacity(num_layers);
        let mut layer_buffer_starts = Vec::with_capacity(num_layers);

        for (i, &max_lookback) in per_layer_lookbacks.iter().enumerate() {
            let cap = max_lookback.checked_add(max_buf + 1).ok_or_else(|| {
                anyhow::anyhow!("WaveNetA2Dyn::new: layer[{i}] buffer capacity overflows usize")
            })?;
            let elem_count = cap.checked_mul(channels).ok_or_else(|| {
                anyhow::anyhow!("WaveNetA2Dyn::new: layer[{i}] buffer size overflows usize")
            })?;
            let mb = MirroredBuffer::<f32>::new(elem_count)?;
            let ring_size = mb.size();
            layer_buffers.push(mb);
            layer_ring_sizes.push(ring_size);
            layer_lookbacks.push(max_lookback * channels);
            layer_buffer_starts.push(ring_size);
        }

        // Validate PReLU slope dimensions against bottleneck channels before
        // allocation (C++ ActivationPReLU applies 1 slope per channel, or a
        // scalar broadcast of length 1; empty or mismatched slope vectors must
        // fail closed rather than silently identity/cycling).
        let validate_prelu = |act: &ActivationType,
                              role: &str,
                              layer_idx: usize|
         -> anyhow::Result<()> {
            if let ActivationType::PReLU { negative_slopes } = act
                && (negative_slopes.is_empty()
                    || (negative_slopes.len() != 1 && negative_slopes.len() != bottleneck))
            {
                return Err(crate::common::diagnostics::NamErrorCode::InvalidModelTopology.into())
                    .map_err(|e: anyhow::Error| {
                        e.context(format!(
                            "WaveNetA2Dyn::new layer[{layer_idx}] {role} PReLU negative_slopes length ({}) must be 1 or bottleneck ({bottleneck})",
                            negative_slopes.len()
                        ))
                    });
            }
            Ok(())
        };

        // Pre-allocate gating/blending configs.
        let mut gating_configs = Vec::with_capacity(num_layers);
        let mut blending_configs = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            validate_prelu(&activations[i], "primary", i)?;
            let gc = match gating_modes[i] {
                GatingMode::Gated => {
                    let sec = secondary_activations[i].clone().ok_or_else(|| {
                        anyhow::Error::from(
                            crate::common::diagnostics::NamErrorCode::NamJsonUnknownActivation,
                        )
                        .context(format!(
                            "WaveNetA2Dyn::new layer[{i}] gating_mode=gated missing secondary activation"
                        ))
                    })?;
                    validate_prelu(&sec, "secondary", i)?;
                    Some(GatingActivationConfig::new(activations[i].clone(), sec))
                }
                _ => None,
            };
            let bc = match gating_modes[i] {
                GatingMode::Blended => {
                    let sec = secondary_activations[i].clone().ok_or_else(|| {
                        anyhow::Error::from(
                            crate::common::diagnostics::NamErrorCode::NamJsonUnknownActivation,
                        )
                        .context(format!(
                            "WaveNetA2Dyn::new layer[{i}] gating_mode=blended missing secondary activation"
                        ))
                    })?;
                    validate_prelu(&sec, "secondary", i)?;
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

        let head1x1_scratch = AlignedVec::new(head_accum_size, 0.0f32)?;

        Ok(Self {
            runtime_weight_layout: Err(super::A2WeightUpdateError::NotLoaded),
            input_channels,
            head_size,
            head_accum_size,
            channels,
            bottleneck,
            num_layers,
            layers: Vec::with_capacity(num_layers),
            rechannel_w_f32: AlignedVec::new(
                input_channels
                    .checked_mul(channels)
                    .ok_or_else(|| anyhow::anyhow!("Overflow calculating rechannel_w size"))?,
                0.0f32,
            )?,
            head_conv: None,
            head_kernel_size,
            head_rechannel_w: AlignedVec::new(
                head_size
                    .max(1)
                    .checked_mul(head_kernel_size)
                    .and_then(|s| s.checked_mul(head_accum_size))
                    .ok_or_else(|| anyhow::anyhow!("Overflow calculating head_rechannel_w size"))?,
                0.0f32,
            )?,
            head_rechannel_b: AlignedVec::new(head_size.max(1), 0.0f32)?,
            head_rechannel_scale: AlignedVec::new(head_size.max(1), 0.0f32)?,
            head_bias: true,
            head_accum: AlignedVec::new(
                head_ring_size
                    .checked_mul(head_accum_size)
                    .ok_or_else(|| anyhow::anyhow!("Overflow calculating head_accum size"))?,
                0.0f32,
            )?,
            head_write_pos: rf,
            head_ring_mask,
            layer_buffers,
            layer_ring_sizes,
            layer_lookbacks,
            layer_buffer_starts,
            layer_in: AlignedVec::new(
                channels
                    .checked_mul(max_buf)
                    .ok_or_else(|| anyhow::anyhow!("Overflow calculating layer_in size"))?,
                0.0f32,
            )?,
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
            groups_input: 1,
            receptive_field_size: rf,
            max_buffer_size: max_buf,
            negotiated_max_buffer_size: max_buf,
            layer_raw: None,
            condition_size: 1,
            z_scratch: AlignedVec::new(bottleneck * 2, 0.0f32)?,
            mixin_scratch: AlignedVec::new(bottleneck * 2, 0.0f32)?,
            l1x1_scratch: AlignedVec::new(channels, 0.0f32)?,
            cond_scratch: AlignedVec::new(1, 0.0f32)?,
            head1x1_scratch,
            prewarm_on_reset: true,
            prewarm_pending: 0,
            condition_dsp: None,
            condition_dsp_output: AlignedVec::new(0, 0.0f32)?,
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

    /// Returns the total receptive field size in samples.
    #[inline(always)]
    pub fn receptive_field_size(&self) -> usize {
        self.receptive_field_size
    }

    /// Returns the input channel count.
    #[inline(always)]
    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    /// Returns the output head size.
    #[inline(always)]
    pub fn head_size(&self) -> usize {
        self.head_size
    }

    /// Returns the internal bottleneck channel count.
    #[inline(always)]
    pub fn bottleneck(&self) -> usize {
        self.bottleneck
    }

    /// Returns the number of layers.
    #[inline(always)]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Returns the A2 layers.
    #[inline(always)]
    pub fn layers(&self) -> &[A2Layer] {
        &self.layers
    }

    /// Returns the input rechannel weight slice.
    #[inline(always)]
    pub fn rechannel_w_f32(&self) -> &[f32] {
        &self.rechannel_w_f32
    }

    /// Returns the optional mono head convolution.
    #[inline(always)]
    pub fn head_conv(&self) -> Option<&A2HeadConv> {
        self.head_conv.as_ref()
    }

    /// Returns the head convolution kernel size.
    #[inline(always)]
    pub fn head_kernel_size(&self) -> usize {
        self.head_kernel_size
    }

    /// Returns the multi-channel head rechannel weight slice.
    #[inline(always)]
    pub fn head_rechannel_w(&self) -> &[f32] {
        &self.head_rechannel_w
    }

    /// Returns the multi-channel head rechannel bias slice.
    #[inline(always)]
    pub fn head_rechannel_b(&self) -> &[f32] {
        &self.head_rechannel_b
    }

    /// Returns the multi-channel head rechannel scale slice.
    #[inline(always)]
    pub fn head_rechannel_scale(&self) -> &[f32] {
        &self.head_rechannel_scale
    }

    /// Returns whether head rechannel bias is enabled.
    #[inline(always)]
    pub fn head_bias(&self) -> bool {
        self.head_bias
    }

    /// Returns the head accumulator ring buffer.
    #[inline(always)]
    pub fn head_accum(&self) -> &[f32] {
        &self.head_accum
    }

    /// Returns the current head write position in columns.
    #[inline(always)]
    pub fn head_write_pos(&self) -> usize {
        self.head_write_pos
    }

    /// Returns the power-of-2 ring mask for `head_accum`.
    #[inline(always)]
    pub fn head_ring_mask(&self) -> usize {
        self.head_ring_mask
    }

    /// Returns the per-layer history mirrored buffers.
    #[inline(always)]
    pub fn layer_buffers(&self) -> &[MirroredBuffer<f32>] {
        &self.layer_buffers
    }

    /// Returns the per-layer ring sizes in elements.
    #[inline(always)]
    pub fn layer_ring_sizes(&self) -> &[usize] {
        &self.layer_ring_sizes
    }

    /// Returns the per-layer lookbacks in elements.
    #[inline(always)]
    pub fn layer_lookbacks(&self) -> &[usize] {
        &self.layer_lookbacks
    }

    /// Returns the per-layer buffer start cursors.
    #[inline(always)]
    pub fn layer_buffer_starts(&self) -> &[usize] {
        &self.layer_buffer_starts
    }

    /// Returns the inter-layer working buffer.
    #[inline(always)]
    pub fn layer_in(&self) -> &[f32] {
        &self.layer_in
    }

    /// Returns the per-layer kernel sizes.
    #[inline(always)]
    pub fn kernel_sizes(&self) -> &[usize] {
        &self.kernel_sizes
    }

    /// Returns the per-layer dilation factors.
    #[inline(always)]
    pub fn dilations(&self) -> &[usize] {
        &self.dilations
    }

    /// Returns the per-layer primary activation types.
    #[inline(always)]
    pub fn activations(&self) -> &[ActivationType] {
        &self.activations
    }

    /// Returns the per-layer gating modes.
    #[inline(always)]
    pub fn gating_modes(&self) -> &[GatingMode] {
        &self.gating_modes
    }

    /// Returns the per-layer secondary activation types.
    #[inline(always)]
    pub fn secondary_activations(&self) -> &[Option<ActivationType>] {
        &self.secondary_activations
    }

    /// Returns whether head1x1 projection is active.
    #[inline(always)]
    pub fn head1x1_active(&self) -> bool {
        self.head1x1_active
    }

    /// Returns the input channel count for each head1x1 layer.
    #[inline(always)]
    pub fn head1x1_h1_in(&self) -> usize {
        self.head1x1_h1_in
    }

    /// Returns the number of groups for the input mixin projection.
    #[inline(always)]
    pub fn mixin_groups(&self) -> u32 {
        self.mixin_groups
    }

    /// Returns the number of groups for the layer1x1 projection.
    #[inline(always)]
    pub fn l1x1_groups(&self) -> u32 {
        self.l1x1_groups
    }

    /// Returns the number of groups for the input convolution.
    #[inline(always)]
    pub fn groups_input(&self) -> u32 {
        self.groups_input
    }

    /// Returns the dimension size of the head accumulator.
    #[inline(always)]
    pub fn head_accum_size(&self) -> usize {
        self.head_accum_size
    }

    /// Returns the configured maximum block size in frames.
    #[inline(always)]
    pub fn max_buffer_size(&self) -> usize {
        self.max_buffer_size
    }

    /// Returns the optional raw JSON layer metadata.
    #[inline(always)]
    pub fn layer_raw(&self) -> Option<&Value> {
        self.layer_raw.as_ref()
    }

    /// Returns the condition vector size.
    #[inline(always)]
    pub fn condition_size(&self) -> usize {
        self.condition_size
    }

    /// Returns whether prewarm runs during `reset()`.
    #[inline(always)]
    pub fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    /// Returns the optional nested condition DSP sub-model.
    #[inline(always)]
    pub fn condition_dsp(&self) -> Option<&StaticModel> {
        self.condition_dsp.as_deref()
    }

    /// Returns the remaining deferred prewarm work units.
    #[inline(always)]
    pub fn prewarm_pending(&self) -> usize {
        self.prewarm_pending
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
    ///
    /// Fallible (H-05): an allocation failure propagates as `Err` instead of
    /// panicking on the loader path.
    pub fn set_condition_dsp(
        &mut self,
        cond_dsp: Option<Box<StaticModel>>,
        max_buf: usize,
    ) -> anyhow::Result<()> {
        let cond_size = if cond_dsp.is_some() {
            self.condition_size
        } else {
            0
        };
        self.condition_dsp_output = AlignedVec::new(cond_size * max_buf, 0.0f32)?;
        self.condition_dsp = cond_dsp;
        Ok(())
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
    #[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
    pub fn enable_diagnostics(&mut self, config: DiagnosticConfig) {
        self.diag.config = config;
        self.diag.dump = Some(DiagnosticDump::new(0));
    }

    /// Takes the accumulated diagnostic dump, resetting internal state.
    #[cfg(any(test, feature = "testing"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
    pub fn take_diagnostics(&mut self) -> Option<DiagnosticDump> {
        self.diag.dump.take()
    }

    /// Resets internal state without inflating the ring buffers beyond
    /// `WAVENET_MAX_NUM_FRAMES` (64), propagating the capped block size to
    /// `condition_dsp` if present.
    ///
    /// Any block size is accepted by [`process`]: processing is internally
    /// chunked into sub-blocks of ≤ `WAVENET_MAX_NUM_FRAMES` (64), so the
    /// per-layer `MirroredBuffer`s and scratch buffers never need capacity above
    /// `max_lookback + WAVENET_MAX_NUM_FRAMES + 1`.
    ///
    /// A request no larger than the largest size this instance was already
    /// sized for is a no-op, including equality: the negotiated size is
    /// idempotent so a host quantum that repeats the loader's `MAX_RESAMP_BUF`
    /// sizing cannot wipe stabilized/prewarmed state. Callers that need a
    /// deliberate reset must call [`Self::reset`] or `clear_state`. Only a
    /// genuine growth re-zeroes state in-place — avoiding heap allocation on
    /// the RT thread (F9 / RT-Safety).
    pub fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        if max_buf <= self.negotiated_max_buffer_size {
            return Ok(());
        }
        if let Some(ref mut cond_dsp) = self.condition_dsp {
            crate::models::NamModel::set_max_buffer_size(
                &mut **cond_dsp,
                max_buf.min(WAVENET_MAX_NUM_FRAMES),
            )?;
            // A genuine growth invalidates temporal state. The capped child may
            // already sit at the same negotiated size (idempotent no-op), so
            // clear it explicitly for the same coherent reset semantics.
            crate::models::NamModel::clear_state(&mut **cond_dsp);
        }
        let rf = self.receptive_field_size;
        let ha_len = self.head_accum.len();
        self.head_accum[..ha_len].fill(0.0);
        self.head_write_pos = rf;
        for buf in self.layer_buffers.iter_mut() {
            buf.fill(0.0);
        }
        self.layer_buffer_starts
            .copy_from_slice(&self.layer_ring_sizes);
        let li_len = self.layer_in.len();
        self.layer_in[..li_len].fill(0.0);
        let cd_len = self.condition_dsp_output.len();
        self.condition_dsp_output[..cd_len].fill(0.0);
        self.negotiated_max_buffer_size = max_buf;
        Ok(())
    }

    /// Resets internal state for a new sample rate and max buffer size.
    ///
    /// Ring and scratch capacities stay capped at `WAVENET_MAX_NUM_FRAMES` (64),
    /// so `reset()` propagates the capped size to `condition_dsp` (if present)
    /// and performs a single buffer clear: via `prewarm()` (which starts with
    /// `prewarm_reset()` -> `a2_prewarm_common`) when `prewarm_on_reset` is
    /// enabled, or `clear_state()` otherwise.
    pub fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        if let Some(ref mut cond_dsp) = self.condition_dsp {
            crate::models::NamModel::set_max_buffer_size(
                &mut **cond_dsp,
                max_buffer_size.min(WAVENET_MAX_NUM_FRAMES),
            )?;
        }
        if self.prewarm_on_reset {
            self.prewarm();
        } else {
            self.clear_state();
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../dynamic_test.rs"]
mod tests;
