// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Layer — per-layer forward pass for the A2 WaveNet fast-path.
//!
//! Implements the sequence from `a2_fast.cpp:514`:
//! 1. Dilated causal Conv1D → 2. Input mixin (condition × mixin_w, no bias) →
//! 3. LeakyReLU(0.01) → 4. Head accumulator (assign layer 0, accumulate layers 1-22) →
//! 5. Layer1x1 residual `layer_in += l1x1_b + l1x1_w * z` (skipped on last layer).
//!
//! ## Source of truth
//! - `a2_fast.cpp:417-690` (`_layer_forward_k`)
//! - `NAM/wavenet/detail.h` (`Layer`)

use super::conv1d::A2Conv1d;
use super::conv1d_ch::A2Conv1dCh;
use super::film::FiLMLayer;
use super::params::A2_LEAKY_SLOPE;
use crate::math::common::{AlignedVec, InstructionSet, SimdMath, effective_instruction_set};
use crate::models::wavenet::common::MAX_KERNEL;

/// Rejection reasons for [`A2Layer::try_process_single_frame`].
///
/// Every variant names one precondition that must hold before the layer is
/// allowed to touch `z_buf`, `head_accum`, `layer_in_out` or the `unsafe`
/// convolution kernel. The error carries no owned data, so building it never
/// allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum A2LayerError {
    /// `in_ch` or `out_ch` is zero.
    #[error("A2 layer channels must be positive (in_ch={in_ch}, out_ch={out_ch})")]
    EmptyChannels {
        /// Number of input channels.
        in_ch: usize,
        /// Number of output channels.
        out_ch: usize,
    },
    /// The kernel size is zero or above the tap-pointer ceiling.
    #[error("A2 layer kernel size {kernel} is unsupported (must be in 1..={max})")]
    UnsupportedKernel {
        /// Requested kernel size.
        kernel: usize,
        /// Largest kernel the SIMD tap-pointer array supports.
        max: usize,
    },
    /// `frame_idx` is smaller than the causal receptive field of the layer.
    #[error(
        "A2 layer frame index {frame_idx} is below the dilated lookback {lookback} \
         ((kernel-1)*dilation)"
    )]
    FrameBeforeReceptiveField {
        /// Absolute frame index handed to the dilated conv.
        frame_idx: usize,
        /// Required causal lookback.
        lookback: usize,
    },
    /// `layer_history` cannot cover the taps and the current frame.
    #[error(
        "A2 layer history length {len} is too short for frame {frame_idx} (needs {required} f32s)"
    )]
    HistoryTooShort {
        /// Actual history length.
        len: usize,
        /// Required length: `(frame_idx+1) * in_ch`.
        required: usize,
        /// Absolute frame index handed to the dilated conv.
        frame_idx: usize,
    },
    /// `z_buf` is narrower than the layer output.
    #[error("A2 layer output scratch length {len} is smaller than out_ch {out_ch}")]
    OutputScratchTooSmall {
        /// Actual scratch length.
        len: usize,
        /// Required length.
        out_ch: usize,
    },
    /// `layer_in_out` is narrower than the layer output.
    #[error("A2 layer residual buffer length {len} is smaller than out_ch {out_ch}")]
    ResidualBufferTooSmall {
        /// Actual residual buffer length.
        len: usize,
        /// Required length.
        out_ch: usize,
    },
    /// `head_accum` does not cover the requested head column.
    #[error(
        "A2 layer head accumulator length {len} is too short for column {head_col} \
         (needs {required} f32s)"
    )]
    HeadAccumTooShort {
        /// Actual head accumulator length.
        len: usize,
        /// Head column index.
        head_col: usize,
        /// Required length: `(head_col+1) * out_ch`.
        required: usize,
    },
    /// The input mixin weights do not cover every output channel.
    #[error("A2 layer mixin weights length {len} is smaller than out_ch {out_ch}")]
    MixinWeightsTooShort {
        /// Actual mixin weight length.
        len: usize,
        /// Required length.
        out_ch: usize,
    },
    /// The input mixin width differs from `out_ch`, which means the layer was
    /// built for a condition width the scalar-`input_cond` helper cannot supply.
    #[error(
        "A2 layer mixin width {len} does not match out_ch {out_ch} \
         (condition width > 1 is unsupported by the single-frame helper)"
    )]
    MixinWidthMismatch {
        /// Actual mixin weight length.
        len: usize,
        /// Convolution output width.
        out_ch: usize,
    },
    /// The standard conv weights are inconsistent with the declared geometry.
    #[error("A2 conv weights length {len} is smaller than the SIMD-padded geometry {required}")]
    ConvWeightsTooShort {
        /// Actual weight length.
        len: usize,
        /// Required SIMD-padded length.
        required: usize,
    },
    /// The standard conv weight layout uses an interleave width other than 4/8/16.
    #[error("A2 conv interleave width {width} is unsupported (must be 4, 8 or 16)")]
    UnsupportedInterleaveWidth {
        /// Declared interleave width.
        width: usize,
    },
    /// The standard conv failed the canonical `Conv1dDyn::is_sound` gate.
    #[error("A2 conv failed the canonical Conv1dDyn soundness check")]
    ConvUnsound,
    /// The grouped conv failed its canonical soundness gate, including the
    /// cross-check of its stored derived layout fields.
    #[error("A2 grouped conv failed its canonical soundness check")]
    GroupedUnsound,
    /// The conv bias is narrower than the layer output.
    #[error("A2 conv bias length {len} is smaller than out_ch {out_ch}")]
    ConvBiasTooShort {
        /// Actual bias length.
        len: usize,
        /// Required length.
        out_ch: usize,
    },
    /// The grouped conv geometry is not a valid group partition.
    #[error(
        "A2 grouped conv topology is invalid (groups={groups}, in_ch={in_ch}, out_ch={out_ch})"
    )]
    InvalidGroupedTopology {
        /// Declared group count.
        groups: usize,
        /// Number of input channels.
        in_ch: usize,
        /// Number of output channels.
        out_ch: usize,
    },
    /// The grouped conv weights do not cover the declared grouped layout (and
    /// depthwise SoA tail, when applicable).
    #[error(
        "A2 grouped conv weights length {len} is smaller than the grouped-padded geometry {required}"
    )]
    GroupedWeightsTooShort {
        /// Actual weight length.
        len: usize,
        /// Required grouped-padded length.
        required: usize,
    },
    /// A checked size or index computation overflowed `usize`.
    #[error("A2 layer size or index arithmetic overflowed usize")]
    SizeOverflow,
    /// The residual (layer1x1) weights do not cover the dense `out_ch × out_ch` layout.
    #[error("A2 layer1x1 weights length {len} is smaller than out_ch² {required}")]
    ResidualWeightsTooShort {
        /// Actual residual weight length.
        len: usize,
        /// Required length.
        required: usize,
    },
    /// The residual (layer1x1) bias width differs from the layer output width.
    #[error("A2 layer1x1 bias width {residual} does not match out_ch {out_ch}")]
    ResidualWidthMismatch {
        /// Declared residual width (`l1x1_out_ch`).
        residual: usize,
        /// Convolution output width.
        out_ch: usize,
    },
    /// The per-layer head1x1 projection is active; the helper does not implement
    /// projected head widths (D-05 option A).
    #[error("A2 layer has an active head1x1 projection, unsupported by the single-frame helper")]
    HeadProjectionActive,
    /// The input mixin is grouped; the helper only implements the dense layout.
    #[error("A2 layer input mixin groups {groups} are unsupported (only 1 is implemented)")]
    UnsupportedMixinGroups {
        /// Declared mixin group count.
        groups: u32,
    },
    /// The layer1x1 residual is grouped; the helper only implements the dense layout.
    #[error("A2 layer1x1 groups {groups} are unsupported (only 1 is implemented)")]
    UnsupportedResidualGroups {
        /// Declared layer1x1 group count.
        groups: u32,
    },
    /// The requested SIMD backend is not available on the host.
    #[error(
        "A2 layer cannot run on {requested:?}: host exposes {available:?} \
         (ISA must be proven before the kernel)"
    )]
    UnsupportedInstructionSet {
        /// Backend selected by the generic `M` parameter.
        requested: InstructionSet,
        /// Backend guaranteed by the crate's dispatch layer for this host.
        available: InstructionSet,
    },
}

/// CH-optimized conv wrapper (enum dispatch).
///
/// Both variants hold the same underlying `A2Conv1dCh<CH>` type; the enum
/// preserves the monomorphized type so the hot-path can call the correct
/// SIMD kernel without indirect dispatch.
pub enum A2ConvCh {
    /// CH=3 variant (SSE 128-bit, stride-16).
    Ch3(A2Conv1dCh<3>),
    /// CH=8 variant (AVX2 256-bit, stride-64).
    Ch8(A2Conv1dCh<8>),
}

impl A2ConvCh {
    /// Constructs a CH-specialized convolution wrapper from a standard `A2Conv1d`
    /// when the geometry matches the CH=3 or CH=8 fast-path (`groups == 1`,
    /// `in_ch == out_ch ∈ {3, 8}`, `kernel ∈ {6, 15}`, `do_bias`, `interleave_width == 4`).
    pub fn try_from_conv(conv: &A2Conv1d) -> Option<Self> {
        let A2Conv1d::Standard(c) = conv else {
            return None;
        };
        if !c.do_bias || c.interleave_width != 4 || c.in_ch != c.out_ch {
            return None;
        }
        if c.kernel != 6 && c.kernel != 15 {
            return None;
        }
        match c.out_ch {
            3 => Some(Self::Ch3(deinterleave_4wide_to_ch::<3>(c)?)),
            8 => Some(Self::Ch8(deinterleave_4wide_to_ch::<8>(c)?)),
            _ => None,
        }
    }
}

fn deinterleave_4wide_to_ch<const CH: usize>(
    c: &crate::models::wavenet::conv1d_dyn::Conv1dDyn,
) -> Option<A2Conv1dCh<CH>> {
    let ch_pad = super::conv1d_ch::ch_pad::<CH>();
    let num_blocks = CH.div_ceil(4);
    let expected_w = num_blocks
        .checked_mul(4)?
        .checked_mul(CH)?
        .checked_mul(c.kernel)?;
    if c.weights.len() < expected_w || c.bias.len() < CH {
        return None;
    }
    let stride = ch_pad * ch_pad;
    let mut weights = AlignedVec::new(c.kernel * stride, 0.0f32).ok()?;
    for b in 0..num_blocks {
        for k in 0..c.kernel {
            for in_c in 0..CH {
                for lane in 0..4 {
                    let out_c = b * 4 + lane;
                    if out_c < CH {
                        let src = b * (c.kernel * CH * 4) + k * (CH * 4) + in_c * 4 + lane;
                        let dst = k * stride + in_c * ch_pad + out_c;
                        weights[dst] = c.weights[src];
                    }
                }
            }
        }
    }
    let mut bias = AlignedVec::new(ch_pad, 0.0f32).ok()?;
    bias[..CH].copy_from_slice(&c.bias[..CH]);
    Some(A2Conv1dCh {
        weights,
        bias,
        dilation: c.dilation,
        kernel: c.kernel,
    })
}

/// Single A2 WaveNet layer.
///
/// Holds the weights for: dilated conv (via `A2Conv1d`), input mixin (`Conv1x1 condition→CH`, no bias),
/// and layer1x1 (`Conv1x1 CH→CH`, with bias, col-major).
///
/// When `conv_ch` is `Some`, it holds f32 col-major-per-tap weights for the CH=3 or CH=8
/// optimized path. When `None`, the standard `A2Conv1d` (f32 interleaved) is used (fallback).
///
/// FiLM layers (8 insertion points) are `Some` only when the JSON config marks them `active: true`.
pub struct A2Layer {
    /// Dilated causal Conv1D (kernel ∈ {6, 15}). Standard (groups=1) or Grouped (groups>1).
    pub(crate) conv: A2Conv1d,
    /// CH-optimized conv (col-major-per-tap f32). `Some` when CH ∈ {3, 8}.
    pub(crate) conv_ch: Option<A2ConvCh>,
    /// Input mixin weights (cond_size=1: `CH` elements; cond_size>1: `CH × cond_size` row-major).
    /// For grouped mixin (groups>1): compact `[out_ch × in_per_group]` row-major; group
    /// membership is determined by output channel index.
    pub(crate) mixin_w: AlignedVec<f32>,
    /// Number of groups for the input mixin projection (C++ `groups_input_mixin`).
    /// Default: 1 (dense).
    pub(crate) mixin_groups: u32,
    /// Layer1x1 weights (groups=1: `bottleneck × channels` col-major `[bottleneck][out]`);
    /// groups>1: compact `[channels × in_per_group]` row-major per output channel.
    pub(crate) l1x1_w: AlignedVec<f32>,
    /// Layer1x1 bias (`CH` elements, f32). Bias is always dense (no grouping).
    pub(crate) l1x1_b: AlignedVec<f32>,
    /// Number of groups for the layer1x1 projection (C++ `layer1x1.groups`).
    /// Default: 1 (dense).
    pub(crate) l1x1_groups: u32,
    /// FiLM before dilated convolution — modulates `layer_in` new frames in history buffer.
    pub(crate) conv_pre_film: Option<FiLMLayer>,
    /// FiLM after dilated convolution — modulates `z_buf` before mixin.
    pub(crate) conv_post_film: Option<FiLMLayer>,
    /// FiLM before input mixin (same insertion point as `conv_post_film`).
    pub(crate) input_mixin_pre_film: Option<FiLMLayer>,
    /// FiLM after input mixin — modulates `z_buf` after mixin, before activation.
    pub(crate) input_mixin_post_film: Option<FiLMLayer>,
    /// FiLM before activation (same insertion point as `input_mixin_post_film`).
    pub(crate) activation_pre_film: Option<FiLMLayer>,
    /// FiLM after activation — modulates `z_buf` after LeakyReLU.
    pub(crate) activation_post_film: Option<FiLMLayer>,
    /// FiLM after layer 1x1 residual — modulates `layer_in` after l1x1 accumulation.
    pub(crate) layer1x1_post_film: Option<FiLMLayer>,
    /// FiLM after head 1x1 (reserved for future general A2 engine).
    pub(crate) head1x1_post_film: Option<FiLMLayer>,
    /// Whether the per-layer head1x1 projection (`bottleneck → channels`) is active.
    /// Mirrors C++ `Layer::_head1x1` (model.cpp:273-297).
    pub(crate) head1x1_active: bool,
    /// Head1x1 projection weights — `[head_accum_size][bottleneck]` row-major
    /// after transposition. Empty when `head1x1_active` is false.
    pub(crate) head1x1_w: AlignedVec<f32>,
    /// Head1x1 projection bias — `head_accum_size` elements.
    /// Empty when `head1x1_active` is false.
    pub(crate) head1x1_b: AlignedVec<f32>,
}

impl A2Layer {
    fn new_base(
        conv: A2Conv1d,
        conv_ch: Option<A2ConvCh>,
        mixin_w: AlignedVec<f32>,
        mixin_groups: u32,
        l1x1_w: AlignedVec<f32>,
        l1x1_b: AlignedVec<f32>,
        l1x1_groups: u32,
    ) -> Self {
        Self {
            conv,
            conv_ch,
            mixin_w,
            mixin_groups,
            l1x1_w,
            l1x1_b,
            l1x1_groups,
            conv_pre_film: None,
            conv_post_film: None,
            input_mixin_pre_film: None,
            input_mixin_post_film: None,
            activation_pre_film: None,
            activation_post_film: None,
            layer1x1_post_film: None,
            head1x1_post_film: None,
            head1x1_active: false,
            // Pre-allocated empty buffers for the optional 1x1 head projection.
            // Using `empty()` is infallible (zero heap allocation, zero capacity, dangling pointer)
            // and avoids panicking `.expect()` in model construction paths, adhering to the
            // resilience standard of MirroredBuffer. Head weights are populated when configured.
            head1x1_w: AlignedVec::empty(),
            head1x1_b: AlignedVec::empty(),
        }
    }

    /// The mixin has no bias in the A2 fast-path (cond_size=1, `Conv1x1 condition→CH`).
    pub fn new(
        conv: A2Conv1d,
        mixin_w: AlignedVec<f32>,
        l1x1_w: AlignedVec<f32>,
        l1x1_b: AlignedVec<f32>,
    ) -> Self {
        let ch = conv.out_ch();
        debug_assert_eq!(mixin_w.len(), ch);
        debug_assert_eq!(l1x1_w.len(), ch * ch);
        debug_assert_eq!(l1x1_b.len(), ch);
        Self::new_base(conv, None, mixin_w, 1, l1x1_w, l1x1_b, 1)
    }

    /// Creates a CH=3 layer with f32-native col-major-per-tap weights.
    pub fn new_with_ch3(
        conv: A2Conv1d,
        ch3_conv: A2Conv1dCh<3>,
        mixin_w: AlignedVec<f32>,
        l1x1_w: AlignedVec<f32>,
        l1x1_b: AlignedVec<f32>,
    ) -> Self {
        debug_assert_eq!(conv.out_ch(), 3);
        debug_assert_eq!(mixin_w.len(), 3);
        debug_assert_eq!(l1x1_w.len(), 9);
        debug_assert_eq!(l1x1_b.len(), 3);
        Self::new_base(
            conv,
            Some(A2ConvCh::Ch3(ch3_conv)),
            mixin_w,
            1,
            l1x1_w,
            l1x1_b,
            1,
        )
    }

    /// Creates a layer with CH=8 optimized weights.
    pub fn new_with_ch8(
        conv: A2Conv1d,
        ch8_conv: A2Conv1dCh<8>,
        mixin_w: AlignedVec<f32>,
        l1x1_w: AlignedVec<f32>,
        l1x1_b: AlignedVec<f32>,
    ) -> Self {
        debug_assert_eq!(conv.out_ch(), 8);
        debug_assert_eq!(mixin_w.len(), 8);
        debug_assert_eq!(l1x1_w.len(), 64);
        debug_assert_eq!(l1x1_b.len(), 8);
        Self::new_base(
            conv,
            Some(A2ConvCh::Ch8(ch8_conv)),
            mixin_w,
            1,
            l1x1_w,
            l1x1_b,
            1,
        )
    }

    /// Creates a layer with arbitrary l1x1 dimensions (bottleneck → channels).
    ///
    /// Used by the dynamic A2 engine where `l1x1_out_ch` (channels) may differ
    /// from `conv.out_ch()` (bottleneck). When gating/blending is active,
    /// `conv.out_ch()` is `2*bottleneck` but l1x1 operates on the post-gating
    /// bottleneck-wide output.
    ///
    /// `_condition_size` controls the mixin weight layout:
    /// `mixin_w` has `conv.out_ch() * (condition_size / mixin_groups)` elements
    /// for grouped mixin, or `conv.out_ch() * condition_size` when ungrouped,
    /// laid out as row-major `[output_channel][condition_index]`.
    pub fn new_dyn(
        conv: A2Conv1d,
        mixin_w: AlignedVec<f32>,
        l1x1_w: AlignedVec<f32>,
        l1x1_b: AlignedVec<f32>,
        l1x1_out_ch: usize,
        bottleneck: usize,
        condition_size: usize,
    ) -> Self {
        debug_assert_eq!(l1x1_b.len(), l1x1_out_ch);
        let conv_ch = if l1x1_out_ch == bottleneck && condition_size == 1 {
            A2ConvCh::try_from_conv(&conv)
        } else {
            None
        };
        Self::new_base(conv, conv_ch, mixin_w, 1, l1x1_w, l1x1_b, 1)
    }

    /// Channel count (bottleneck == channels in A2 fast-path).
    #[inline(always)]
    pub fn channels(&self) -> usize {
        self.conv.out_ch()
    }

    /// Number of groups for the dilated conv (1 = standard, >1 = grouped/depthwise).
    #[inline(always)]
    pub fn groups(&self) -> usize {
        self.conv.groups()
    }

    /// Returns true if the dilated conv is depthwise (groups == channels).
    #[inline(always)]
    pub fn is_depthwise(&self) -> bool {
        self.conv.is_depthwise()
    }

    /// Kernel size of this layer's dilated conv.
    #[inline(always)]
    pub fn kernel_size(&self) -> usize {
        self.conv.kernel_size()
    }

    /// Dilation factor of this layer.
    #[inline(always)]
    pub fn dilation(&self) -> usize {
        self.conv.dilation()
    }

    /// Returns the dilated causal Conv1D of this layer.
    #[inline(always)]
    pub fn conv(&self) -> &A2Conv1d {
        &self.conv
    }

    /// Returns the optional CH-optimized convolution wrapper (`CH ∈ {3, 8}`).
    #[inline(always)]
    pub fn conv_ch(&self) -> Option<&A2ConvCh> {
        self.conv_ch.as_ref()
    }

    /// Returns the input mixin weight slice.
    #[inline(always)]
    pub fn mixin_w(&self) -> &[f32] {
        &self.mixin_w
    }

    /// Returns the number of groups for the input mixin projection.
    #[inline(always)]
    pub fn mixin_groups(&self) -> u32 {
        self.mixin_groups
    }

    /// Returns the layer1x1 weight slice.
    #[inline(always)]
    pub fn l1x1_w(&self) -> &[f32] {
        &self.l1x1_w
    }

    /// Returns the layer1x1 bias slice.
    #[inline(always)]
    pub fn l1x1_b(&self) -> &[f32] {
        &self.l1x1_b
    }

    /// Returns the number of groups for the layer1x1 projection.
    #[inline(always)]
    pub fn l1x1_groups(&self) -> u32 {
        self.l1x1_groups
    }

    /// Returns the optional FiLM layer applied before dilated convolution.
    #[inline(always)]
    pub fn conv_pre_film(&self) -> Option<&FiLMLayer> {
        self.conv_pre_film.as_ref()
    }

    /// Returns the optional FiLM layer applied after dilated convolution.
    #[inline(always)]
    pub fn conv_post_film(&self) -> Option<&FiLMLayer> {
        self.conv_post_film.as_ref()
    }

    /// Returns the optional FiLM layer applied before input mixin.
    #[inline(always)]
    pub fn input_mixin_pre_film(&self) -> Option<&FiLMLayer> {
        self.input_mixin_pre_film.as_ref()
    }

    /// Returns the optional FiLM layer applied after input mixin.
    #[inline(always)]
    pub fn input_mixin_post_film(&self) -> Option<&FiLMLayer> {
        self.input_mixin_post_film.as_ref()
    }

    /// Returns the optional FiLM layer applied before activation.
    #[inline(always)]
    pub fn activation_pre_film(&self) -> Option<&FiLMLayer> {
        self.activation_pre_film.as_ref()
    }

    /// Returns the optional FiLM layer applied after activation.
    #[inline(always)]
    pub fn activation_post_film(&self) -> Option<&FiLMLayer> {
        self.activation_post_film.as_ref()
    }

    /// Returns the optional FiLM layer applied after layer1x1 residual.
    #[inline(always)]
    pub fn layer1x1_post_film(&self) -> Option<&FiLMLayer> {
        self.layer1x1_post_film.as_ref()
    }

    /// Returns the optional FiLM layer applied after head1x1 projection.
    #[inline(always)]
    pub fn head1x1_post_film(&self) -> Option<&FiLMLayer> {
        self.head1x1_post_film.as_ref()
    }

    /// Returns whether per-layer head1x1 projection is active.
    #[inline(always)]
    pub fn head1x1_active(&self) -> bool {
        self.head1x1_active
    }

    /// Returns the per-layer head1x1 weight slice.
    #[inline(always)]
    pub fn head1x1_w(&self) -> &[f32] {
        &self.head1x1_w
    }

    /// Returns the per-layer head1x1 bias slice.
    #[inline(always)]
    pub fn head1x1_b(&self) -> &[f32] {
        &self.head1x1_b
    }

    /// Processes a single frame through this layer, validating every
    /// precondition before touching any buffer or reaching the convolution
    /// kernel.
    ///
    /// This is the fallible variant of [`process_single_frame`](Self::process_single_frame).
    /// It is transactional: on `Err`, `z_buf`, `head_accum` and `layer_in_out`
    /// are left untouched, and the `unsafe` convolution kernel is not called.
    ///
    /// ## Accepted subset (D-05 option A)
    ///
    /// The helper implements the dense A2 fast-path layout only:
    /// `mixin_groups == 1`, `l1x1_groups == 1`, `!head1x1_active`, a dense
    /// mixin width equal to `channels()` and a residual width (`l1x1_b.len()`)
    /// equal to `channels()`. Configurations produced by
    /// [`new_dyn`](Self::new_dyn) with a different residual width
    /// (gating/blending), with `condition_size > 1` (wider mixin) or by an
    /// active head1x1 projection are rejected with
    /// [`A2LayerError::ResidualWidthMismatch`],
    /// [`A2LayerError::MixinWidthMismatch`],
    /// [`A2LayerError::HeadProjectionActive`],
    /// [`A2LayerError::UnsupportedMixinGroups`] or
    /// [`A2LayerError::UnsupportedResidualGroups`]. The dynamic engine has its
    /// own projection path and does not call this helper.
    ///
    /// ## Data flow (matches `_layer_forward_k` in `a2_fast.cpp:514`)
    ///
    /// 1. Dilated conv over `layer_history` → `z_buf[..CH]`.
    /// 2. Mixin: `z_buf[c] += mixin_w[c] * input_cond`.
    /// 3. LeakyReLU(0.01) in-place.
    /// 4. Head accumulator: assign (layer 0) or add (layer > 0) into `head_accum`.
    /// 5. L1x1 residual: `layer_in[c] += l1x1_b[c] + sum_u(l1x1_w[u*CH+c] * z_buf[u])`.
    ///    Skipped on the last layer (output of last layer is dead — only head matters).
    ///
    /// # Parameters
    /// * `layer_history` — per-layer ring buffer (column-major: CH rows × N cols).
    /// * `frame_idx` — absolute column index in `layer_history` for the dilated conv (already ring-masked or linear).
    /// * `input_cond` — scalar input condition for the mixin (original input signal at this frame).
    /// * `head_accum` — head accumulator ring buffer (column-major).
    /// * `head_col` — column index in `head_accum` for this frame's output.
    /// * `z_buf` — scratch buffer for conv output (length ≥ `channels()`).
    /// * `layer_in_out` — mutable reference to this frame's layer_in (will be updated).
    /// * `is_first` — layer 0 writes to head, layers 1-22 accumulate.
    /// * `is_last` — layer 22 skips l1x1 residual.
    ///
    /// # Errors
    /// Returns an [`A2LayerError`] variant naming the violated precondition:
    /// empty channels, unsupported kernel, `frame_idx` below the dilated
    /// lookback, undersized history/scratch/head/residual buffers, inconsistent
    /// conv or residual weights/bias, unsupported grouped layouts, unsupported
    /// projected topologies, or a SIMD backend the host cannot execute.
    /// No buffer is modified on error.
    #[expect(
        clippy::too_many_arguments,
        reason = "A2 neural network layer requiring many shape/stride/buffer parameters for dynamic topology construction"
    )]
    #[inline(always)]
    pub fn try_process_single_frame<M: SimdMath>(
        &self,
        layer_history: &[f32],
        frame_idx: usize,
        input_cond: f32,
        head_accum: &mut [f32],
        head_col: usize,
        z_buf: &mut [f32],
        layer_in_out: &mut [f32],
        is_first: bool,
        is_last: bool,
    ) -> Result<(), A2LayerError> {
        self.validate_frame(
            layer_history,
            frame_idx,
            head_accum,
            head_col,
            z_buf,
            layer_in_out,
            is_last,
        )?;
        let available = effective_instruction_set();
        if !isa_supported_by_host(M::ISA, available) {
            return Err(A2LayerError::UnsupportedInstructionSet {
                requested: M::ISA,
                available,
            });
        }

        let ch = self.channels();
        // SAFETY: `validate_frame` proved `z_buf.len() >= ch`, the history
        // covers `(frame_idx+1) * in_ch` and `frame_idx >= (kernel-1)*dilation`,
        // the standard/grouped weight geometry is sound, and
        // `isa_supported_by_host` proved the selected backend is available;
        // these are exactly `A2Conv1d::process_single_frame`'s preconditions.
        unsafe {
            self.conv
                .process_single_frame::<M>(layer_history, z_buf, frame_idx, None);
        }

        let mixin = &self.mixin_w;
        for c in 0..ch {
            z_buf[c] += mixin[c] * input_cond;
        }

        for z in z_buf.iter_mut().take(ch) {
            if *z < 0.0 {
                *z *= A2_LEAKY_SLOPE;
            }
        }

        let head_off = head_col * ch;
        if is_first {
            head_accum[head_off..head_off + ch].copy_from_slice(&z_buf[..ch]);
        } else {
            for (c, z_val) in z_buf.iter().enumerate().take(ch) {
                head_accum[head_off + c] += *z_val;
            }
        }

        if !is_last {
            let l1x1 = &self.l1x1_w;
            let bias = &self.l1x1_b;
            for c in 0..ch {
                let mut sum = bias[c];
                for u in 0..ch {
                    // Col-major: l1x1_w[u * ch + c] = weight from bottleneck u to output c.
                    sum += l1x1[u * ch + c] * z_buf[u];
                }
                layer_in_out[c] += sum;
            }
        }
        Ok(())
    }

    /// Processes a single frame through this layer.
    ///
    /// This is the infallible legacy wrapper around
    /// [`try_process_single_frame`](Self::try_process_single_frame). For valid
    /// inputs it produces bit-identical results.
    ///
    /// # Invalid input (D-04 option B)
    ///
    /// The preconditions are validated in every build. In debug builds a
    /// violation triggers `debug_assert!` (diagnostics only). In release builds
    /// the call returns without modifying `z_buf`, `head_accum` or
    /// `layer_in_out`, and never reaches the `unsafe` convolution kernel.
    /// Callers that need the rejection reason should use
    /// [`try_process_single_frame`](Self::try_process_single_frame).
    ///
    /// # Parameters
    /// See [`try_process_single_frame`](Self::try_process_single_frame).
    #[expect(
        clippy::too_many_arguments,
        reason = "A2 neural network layer requiring many shape/stride/buffer parameters for dynamic topology construction"
    )]
    #[inline(always)]
    pub fn process_single_frame<M: SimdMath>(
        &self,
        layer_history: &[f32],
        frame_idx: usize,
        input_cond: f32,
        head_accum: &mut [f32],
        head_col: usize,
        z_buf: &mut [f32],
        layer_in_out: &mut [f32],
        is_first: bool,
        is_last: bool,
    ) {
        let result = self.try_process_single_frame::<M>(
            layer_history,
            frame_idx,
            input_cond,
            head_accum,
            head_col,
            z_buf,
            layer_in_out,
            is_first,
            is_last,
        );
        debug_assert!(
            result.is_ok(),
            "A2Layer::process_single_frame rejected invalid input: {:?}",
            result.err()
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "pure precondition check mirroring the frame parameters of process_single_frame"
    )]
    fn validate_frame(
        &self,
        layer_history: &[f32],
        frame_idx: usize,
        head_accum: &[f32],
        head_col: usize,
        z_buf: &[f32],
        layer_in_out: &[f32],
        is_last: bool,
    ) -> Result<(), A2LayerError> {
        if self.head1x1_active {
            return Err(A2LayerError::HeadProjectionActive);
        }

        let in_ch = self.conv.in_ch();
        let out_ch = self.conv.out_ch();
        if in_ch == 0 || out_ch == 0 {
            return Err(A2LayerError::EmptyChannels { in_ch, out_ch });
        }

        let kernel = self.conv.kernel_size();
        if kernel == 0 || kernel > MAX_KERNEL {
            return Err(A2LayerError::UnsupportedKernel {
                kernel,
                max: MAX_KERNEL,
            });
        }

        let dilation = self.conv.dilation();
        if dilation > isize::MAX as usize {
            return Err(A2LayerError::SizeOverflow);
        }
        let lookback = kernel
            .checked_sub(1)
            .and_then(|taps| taps.checked_mul(dilation))
            .ok_or(A2LayerError::SizeOverflow)?;
        if frame_idx < lookback {
            return Err(A2LayerError::FrameBeforeReceptiveField {
                frame_idx,
                lookback,
            });
        }

        let history_required = frame_idx
            .checked_add(1)
            .and_then(|frames| frames.checked_mul(in_ch))
            .ok_or(A2LayerError::SizeOverflow)?;
        if layer_history.len() < history_required {
            return Err(A2LayerError::HistoryTooShort {
                len: layer_history.len(),
                required: history_required,
                frame_idx,
            });
        }

        if z_buf.len() < out_ch {
            return Err(A2LayerError::OutputScratchTooSmall {
                len: z_buf.len(),
                out_ch,
            });
        }
        if layer_in_out.len() < out_ch {
            return Err(A2LayerError::ResidualBufferTooSmall {
                len: layer_in_out.len(),
                out_ch,
            });
        }

        let head_required = head_col
            .checked_mul(out_ch)
            .and_then(|offset| offset.checked_add(out_ch))
            .ok_or(A2LayerError::SizeOverflow)?;
        if head_accum.len() < head_required {
            return Err(A2LayerError::HeadAccumTooShort {
                len: head_accum.len(),
                head_col,
                required: head_required,
            });
        }

        if self.mixin_w.len() < out_ch {
            return Err(A2LayerError::MixinWeightsTooShort {
                len: self.mixin_w.len(),
                out_ch,
            });
        }
        if self.mixin_w.len() != out_ch {
            return Err(A2LayerError::MixinWidthMismatch {
                len: self.mixin_w.len(),
                out_ch,
            });
        }
        if self.mixin_groups != 1 {
            return Err(A2LayerError::UnsupportedMixinGroups {
                groups: self.mixin_groups,
            });
        }
        if self.l1x1_groups != 1 {
            return Err(A2LayerError::UnsupportedResidualGroups {
                groups: self.l1x1_groups,
            });
        }
        if self.l1x1_b.len() != out_ch {
            return Err(A2LayerError::ResidualWidthMismatch {
                residual: self.l1x1_b.len(),
                out_ch,
            });
        }

        if !is_last {
            let residual_required = out_ch
                .checked_mul(out_ch)
                .ok_or(A2LayerError::SizeOverflow)?;
            if self.l1x1_w.len() < residual_required {
                return Err(A2LayerError::ResidualWeightsTooShort {
                    len: self.l1x1_w.len(),
                    required: residual_required,
                });
            }
        }

        match &self.conv {
            A2Conv1d::Standard(conv) => {
                let width = conv.interleave_width;
                if width != 4 && width != 8 && width != 16 {
                    return Err(A2LayerError::UnsupportedInterleaveWidth { width });
                }
                if conv.dilation > isize::MAX as usize
                    || conv.dilation.checked_mul(conv.in_ch).is_none()
                {
                    return Err(A2LayerError::SizeOverflow);
                }
                let required = conv
                    .out_ch
                    .div_ceil(width)
                    .checked_mul(width)
                    .and_then(|total| total.checked_mul(conv.in_ch))
                    .and_then(|total| total.checked_mul(conv.kernel))
                    .ok_or(A2LayerError::SizeOverflow)?;
                if conv.weights.len() < required {
                    return Err(A2LayerError::ConvWeightsTooShort {
                        len: conv.weights.len(),
                        required,
                    });
                }
                if conv.do_bias && conv.bias.len() < out_ch {
                    return Err(A2LayerError::ConvBiasTooShort {
                        len: conv.bias.len(),
                        out_ch,
                    });
                }
                if !conv.is_sound() {
                    return Err(A2LayerError::ConvUnsound);
                }
            }
            A2Conv1d::Grouped(conv) => {
                let groups = conv.groups();
                if groups == 0 || !in_ch.is_multiple_of(groups) || !out_ch.is_multiple_of(groups) {
                    return Err(A2LayerError::InvalidGroupedTopology {
                        groups,
                        in_ch,
                        out_ch,
                    });
                }
                let in_per_group = in_ch / groups;
                let blocks_per_group = (out_ch / groups).div_ceil(4);
                let interleaved = groups
                    .checked_mul(blocks_per_group)
                    .and_then(|blocks| blocks.checked_mul(4))
                    .and_then(|total| total.checked_mul(in_per_group))
                    .and_then(|total| total.checked_mul(kernel))
                    .ok_or(A2LayerError::SizeOverflow)?;
                let required = if groups == in_ch && groups == out_ch {
                    kernel
                        .checked_mul(in_ch)
                        .and_then(|soa| interleaved.checked_add(soa))
                        .ok_or(A2LayerError::SizeOverflow)?
                } else {
                    interleaved
                };
                if conv.weights().len() < required {
                    return Err(A2LayerError::GroupedWeightsTooShort {
                        len: conv.weights().len(),
                        required,
                    });
                }
                if conv.do_bias() && conv.bias().len() < out_ch {
                    return Err(A2LayerError::ConvBiasTooShort {
                        len: conv.bias().len(),
                        out_ch,
                    });
                }
                if !conv.is_sound() {
                    return Err(A2LayerError::GroupedUnsound);
                }
            }
        }

        Ok(())
    }
}

/// Decides whether the dispatch layer can prove `requested` is executable on
/// this host. `Avx2` is the compile-time baseline (x86-64-v3 / NEON) and is
/// always available; AVX-512 backends are usable only when the host detection
/// (cached once in `SIMD_MATH`) selected an AVX-512 instruction set.
#[inline]
#[expect(
    deprecated,
    reason = "the legacy Avx512VnniBf16 variant remains part of the public InstructionSet surface"
)]
fn isa_supported_by_host(requested: InstructionSet, available: InstructionSet) -> bool {
    match requested {
        InstructionSet::Avx2 => true,
        InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
            matches!(
                available,
                InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16
            )
        }
    }
}

// =============================================================================
// Scalar reference (oracle) for parity testing
// =============================================================================

/// Scalar reference for the A2Layer forward pass on a single frame.
///
/// Replicates the exact computation using the scalar conv fallback, then
/// applies mixin, LeakyReLU, head accumulation, and l1x1 residual.
/// Used as oracle in parity tests.
#[expect(
    clippy::too_many_arguments,
    reason = "A2 neural network layer requiring many shape/stride/buffer parameters for dynamic topology construction"
)]
pub fn a2_layer_single_frame_scalar_ref(
    conv_weights: &[f32],
    conv_bias: &[f32],
    conv_do_bias: bool,
    dilation: usize,
    kernel_size: usize,
    layer_history: &[f32],
    frame_idx: usize,
    mixin_w: &[f32],
    input_cond: f32,
    l1x1_w: &[f32],
    l1x1_b: &[f32],
    head_accum: &mut [f32],
    head_col: usize,
    layer_in_out: &mut [f32],
    is_first: bool,
    is_last: bool,
) {
    let ch = mixin_w.len();

    // 1. Dilated conv (scalar fallback).
    let mut z_buf = vec![0.0f32; ch];
    super::conv1d_fallback::a2_conv1d_single_frame_fallback(
        conv_weights,
        conv_bias,
        conv_do_bias,
        dilation,
        ch, // in_ch == out_ch for A2 fast-path
        ch,
        kernel_size,
        layer_history,
        frame_idx,
        None,
        &mut z_buf,
    );

    // 2. Mixin.
    for c in 0..ch {
        z_buf[c] += mixin_w[c] * input_cond;
    }

    // 3. LeakyReLU(0.01).
    for z in z_buf.iter_mut().take(ch) {
        if *z < 0.0 {
            *z *= A2_LEAKY_SLOPE;
        }
    }

    // 4. Head accumulator.
    let head_off = head_col * ch;
    if is_first {
        head_accum[head_off..head_off + ch].copy_from_slice(&z_buf[..ch]);
    } else {
        for c in 0..ch {
            head_accum[head_off + c] += z_buf[c];
        }
    }

    // 5. L1x1 residual.
    if !is_last {
        for c in 0..ch {
            let mut sum = l1x1_b[c];
            for u in 0..ch {
                sum += l1x1_w[u * ch + c] * z_buf[u];
            }
            layer_in_out[c] += sum;
        }
    }
}

#[cfg(test)]
mod isa_precondition_tests {
    use super::{InstructionSet, isa_supported_by_host};

    #[test]
    fn baseline_backend_is_always_supported() {
        assert!(isa_supported_by_host(
            InstructionSet::Avx2,
            InstructionSet::Avx2
        ));
        assert!(isa_supported_by_host(
            InstructionSet::Avx2,
            InstructionSet::Avx512
        ));
    }

    #[test]
    fn avx512_is_rejected_when_host_detects_only_baseline() {
        assert!(!isa_supported_by_host(
            InstructionSet::Avx512,
            InstructionSet::Avx2
        ));
        assert!(isa_supported_by_host(
            InstructionSet::Avx512,
            InstructionSet::Avx512
        ));
    }
}

#[cfg(test)]
#[path = "layer_test.rs"]
mod tests;
