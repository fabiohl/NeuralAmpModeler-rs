// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Weight extraction and channel slicing infrastructure for `SlimmableModel`.
//!
//! These functions run at construction time (main thread, SPSC GC pipeline) —
//! never on the real-time audio thread.

use crate::common::diagnostics::NamErrorCode;
use crate::common::spsc::GcItem;
use crate::loader::dispatcher::wavenet::layout::select_interleave_width;
use crate::math::common::AlignedVec;
use crate::models::wavenet::{
    Conv1dDyn, DenseLayerDyn, WAVENET_MAX_NUM_FRAMES, WaveNetLayerArrayDyn, WaveNetLayerDyn,
    WaveNetLayerState, WaveNetModelDyn,
};
use crate::models::{NamModel, StaticModel};

/// Typed error representing slicing failures in `SlimmableModel`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlicingError {
    /// Requested input channels exceed original layer input channels.
    InvalidInputChannels {
        /// Requested input channel count.
        requested: usize,
        /// Available input channel count in layer.
        available: usize,
    },
    /// Requested output channels exceed original layer output channels.
    InvalidOutputChannels {
        /// Requested output channel count.
        requested: usize,
        /// Available output channel count in layer.
        available: usize,
    },
    /// Target channel count must be greater than zero.
    ZeroChannelCount,
    /// Requested channel count exceeds original model channel count.
    RequestedChannelExceedsModel {
        /// Requested channel count.
        requested: usize,
        /// Available model channel count.
        available: usize,
    },
    /// Requested channel count exceeds minimum channel count across arrays.
    RequestedChannelExceedsMinArray {
        /// Requested channel count.
        requested: usize,
        /// Minimum channel count across all layer arrays.
        min_array_ch: usize,
    },
    /// Allocation error during slicing.
    Allocation(NamErrorCode),
}

impl std::fmt::Display for SlicingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInputChannels {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Slicing invalid input channels: requested {requested} > available {available}"
                )
            }
            Self::InvalidOutputChannels {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Slicing invalid output channels: requested {requested} > available {available}"
                )
            }
            Self::ZeroChannelCount => {
                write!(f, "Slicing target channel count must be > 0")
            }
            Self::RequestedChannelExceedsModel {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Slicing target channel count ({requested}) exceeds model channel count ({available})"
                )
            }
            Self::RequestedChannelExceedsMinArray {
                requested,
                min_array_ch,
            } => {
                write!(
                    f,
                    "Slicing target channel count ({requested}) exceeds minimum array channel count ({min_array_ch})"
                )
            }
            Self::Allocation(code) => {
                write!(f, "Slicing allocation failure: {code:?}")
            }
        }
    }
}

impl std::error::Error for SlicingError {}

impl From<NamErrorCode> for SlicingError {
    fn from(code: NamErrorCode) -> Self {
        SlicingError::Allocation(code)
    }
}

impl From<SlicingError> for std::io::Error {
    fn from(err: SlicingError) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, err)
    }
}

/// Slices a `Conv1dDyn` to a reduced channel configuration.
///
/// The convolution weight layout is `[block][kernel][in_ch][4]`
/// (4-wide lane-interleaved). This function extracts only the first
/// `new_in_ch` input channels and first `new_out_ch` output channels,
/// keeping the same `kernel` size and `dilation`.
///
/// # Errors
/// Returns [`SlicingError`] if `new_in_ch > conv.in_ch` or `new_out_ch > conv.out_ch`,
/// or if weight allocation fails.
pub fn slice_conv1d(
    conv: &Conv1dDyn,
    new_in_ch: usize,
    new_out_ch: usize,
) -> Result<Conv1dDyn, SlicingError> {
    if new_in_ch > conv.in_ch {
        return Err(SlicingError::InvalidInputChannels {
            requested: new_in_ch,
            available: conv.in_ch,
        });
    }
    if new_out_ch > conv.out_ch {
        return Err(SlicingError::InvalidOutputChannels {
            requested: new_out_ch,
            available: conv.out_ch,
        });
    }

    let dst_width = select_interleave_width(new_out_ch);
    let src_width = conv.interleave_width;
    let new_num_blocks = new_out_ch.div_ceil(dst_width);
    let kernel = conv.kernel;
    let new_weights_len = new_num_blocks * dst_width * new_in_ch * kernel;
    let mut new_weights = AlignedVec::new(new_weights_len, 0.0f32)?;

    for src_b in 0..new_out_ch.div_ceil(src_width) {
        for k in 0..kernel {
            for in_c in 0..new_in_ch {
                for lane in 0..src_width {
                    let out_c = src_b * src_width + lane;
                    if out_c >= new_out_ch {
                        break;
                    }
                    let dst_b = out_c / dst_width;
                    let dst_lane = out_c % dst_width;
                    let src_idx =
                        (src_b * kernel + k) * conv.in_ch * src_width + in_c * src_width + lane;
                    let dst_idx =
                        (dst_b * kernel + k) * new_in_ch * dst_width + in_c * dst_width + dst_lane;
                    new_weights[dst_idx] = conv.weights[src_idx];
                }
            }
        }
    }

    let mut new_bias = AlignedVec::new(new_out_ch, 0.0f32)?;
    new_bias.copy_from_slice(&conv.bias[..new_out_ch]);

    Ok(Conv1dDyn {
        weights: new_weights,
        bias: new_bias,
        do_bias: conv.do_bias,
        dilation: conv.dilation,
        in_ch: new_in_ch,
        out_ch: new_out_ch,
        num_blocks: new_num_blocks,
        interleave_width: dst_width,
        kernel,
    })
}

/// Slices a `DenseLayerDyn` to a reduced channel configuration.
///
/// Dense weights are stored in column-major layout:
/// `weights[in_c * out_ch + out_c]`.
/// This extracts the first `new_in_ch` input channels and first `new_out_ch`
/// output channels.
///
/// # Errors
/// Returns [`SlicingError`] if `new_in_ch > dense.in_ch` or `new_out_ch > dense.out_ch`,
/// or if weight allocation fails.
pub fn slice_dense(
    dense: &DenseLayerDyn,
    new_in_ch: usize,
    new_out_ch: usize,
) -> Result<DenseLayerDyn, SlicingError> {
    if new_in_ch > dense.in_ch {
        return Err(SlicingError::InvalidInputChannels {
            requested: new_in_ch,
            available: dense.in_ch,
        });
    }
    if new_out_ch > dense.out_ch {
        return Err(SlicingError::InvalidOutputChannels {
            requested: new_out_ch,
            available: dense.out_ch,
        });
    }

    let mut new_weights = AlignedVec::new(new_in_ch * new_out_ch, 0.0f32)?;

    for in_c in 0..new_in_ch {
        let src_start = in_c * dense.out_ch;
        let dst_start = in_c * new_out_ch;
        new_weights[dst_start..dst_start + new_out_ch]
            .copy_from_slice(&dense.weights[src_start..src_start + new_out_ch]);
    }

    let mut new_bias = AlignedVec::new(new_out_ch, 0.0f32)?;
    new_bias.copy_from_slice(&dense.bias[..new_out_ch]);

    Ok(DenseLayerDyn {
        in_ch: new_in_ch,
        out_ch: new_out_ch,
        weights: new_weights,
        bias: new_bias,
        do_bias: dense.do_bias,
    })
}

/// Creates a new `WaveNetLayerDyn` with reduced internal channel count.
///
/// Slices all three internal tensors:
/// - `conv1d`: `(ch, ch)` → `(new_ch, new_ch)`
/// - `input_mixin`: `(cond, ch)` → `(cond, new_ch)`
/// - `one_by_one`: `(ch, ch)` → `(new_ch, new_ch)`
pub fn slice_wavenet_layer(
    layer: &WaveNetLayerDyn,
    new_ch: usize,
) -> Result<WaveNetLayerDyn, SlicingError> {
    let conv1d = slice_conv1d(&layer.conv1d, new_ch, new_ch)?;
    let input_mixin = slice_dense(&layer.input_mixin, layer.input_mixin.in_ch, new_ch)?;
    let one_by_one = slice_dense(&layer.one_by_one, new_ch, new_ch)?;
    WaveNetLayerDyn::new(new_ch, conv1d, input_mixin, one_by_one).map_err(SlicingError::Allocation)
}

/// Creates a new `WaveNetLayerArrayDyn` with reduced internal channel count.
///
/// Rebuilds all sub-components (rechannel, layers, states, head_rechannel)
/// with the new channel dimensions. States are freshly allocated via
/// `WaveNetLayerState::new` — prewarm will stabilize them later.
///
/// `new_in_ch`: input channels for this array (1 for the first array,
///              `new_ch` for subsequent arrays).
/// `alloc_num`: allocation counter for state jitter (pass a `&mut usize`
///              that increments across all arrays in the model).
pub fn slice_wavenet_array(
    array: &WaveNetLayerArrayDyn,
    new_in_ch: usize,
    new_ch: usize,
    alloc_num: &mut usize,
) -> Result<WaveNetLayerArrayDyn, SlicingError> {
    let rechannel = slice_dense(&array.rechannel, new_in_ch, new_ch)?;

    let mut layers = Vec::with_capacity(array.layers.len());
    let mut states = Vec::with_capacity(array.layers.len());

    for layer in &array.layers {
        layers.push(slice_wavenet_layer(layer, new_ch)?);
        let rf = (layer.conv1d.kernel - 1) * layer.conv1d.dilation;
        states.push(
            WaveNetLayerState::new(new_ch, rf, *alloc_num)
                .map_err(|_| SlicingError::Allocation(NamErrorCode::OutOfMemory))?,
        );
        *alloc_num += 1;
    }

    let head_rechannel = slice_dense(&array.head_rechannel, new_ch, array.head)?;

    let receptive_field_size: usize = array
        .layers
        .iter()
        .map(|l| (l.conv1d.kernel - 1) * l.conv1d.dilation)
        .sum();

    let block_size = new_ch;
    let num_layers = layers.len();

    Ok(WaveNetLayerArrayDyn {
        in_ch: new_in_ch,
        cond: array.cond,
        ch: new_ch,
        k: array.k,
        head: array.head,
        layers,
        states,
        rechannel,
        head_rechannel,
        array_outputs: AlignedVec::new(new_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        head_accum: AlignedVec::new(new_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        head_outputs: AlignedVec::new(array.head * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        receptive_field_size,
        block_size,
        block_buffer: AlignedVec::new(block_size * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        effective_layers: num_layers,
    })
}

/// Creates a new `WaveNetModelDyn` with all internal channels reduced to
/// `new_ch`. This is the primary entry point for the SPSC GC swap pipeline.
///
/// Each layer array's internal `ch` is reduced to `new_ch`. The head
/// projection and condition dimensions remain unchanged.
///
/// ## Limitations
///
/// - **`condition_dsp`**: Cloned from the original model. Only `WavenetDyn`
///   sub-models are supported for deep cloning; other variants fall back to
///   `None`. The `post_stack_head` and
///   `condition_dsp_output` buffers are allocated fresh.
/// - **`post_stack_head`**: Cloned as-is (not affected by channel slicing).
///
/// # Errors
/// Returns [`SlicingError`] if `new_ch == 0`, `new_ch > model.ch`, or `new_ch` exceeds
/// the minimum channel count across arrays.
pub fn slice_wavenet_model(
    model: &WaveNetModelDyn,
    new_ch: usize,
) -> Result<WaveNetModelDyn, SlicingError> {
    if new_ch == 0 {
        return Err(SlicingError::ZeroChannelCount);
    }
    if new_ch > model.ch {
        return Err(SlicingError::RequestedChannelExceedsModel {
            requested: new_ch,
            available: model.ch,
        });
    }

    let min_array_ch = model.arrays.iter().map(|a| a.ch).min().unwrap_or(model.ch);
    if new_ch > min_array_ch {
        return Err(SlicingError::RequestedChannelExceedsMinArray {
            requested: new_ch,
            min_array_ch,
        });
    }

    let mut alloc_num = 0usize;
    let mut arrays = Vec::with_capacity(model.arrays.len());

    for (i, array) in model.arrays.iter().enumerate() {
        let in_ch = if i == 0 { 1 } else { new_ch };
        arrays.push(slice_wavenet_array(array, in_ch, new_ch, &mut alloc_num)?);
    }

    let cond = model.arrays[0].cond;
    let cond_dsp_output_size = cond * WAVENET_MAX_NUM_FRAMES;

    let head_out_ch = model
        .post_stack_head
        .as_ref()
        .map(|h| h.out_channels())
        .unwrap_or(1);
    let head_output_scratch = AlignedVec::new(head_out_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?;

    let mut rf = arrays
        .iter()
        .map(|a| a.receptive_field_size)
        .max()
        .unwrap_or(0);
    if let Some(ref head_proc) = model.post_stack_head {
        rf += head_proc.receptive_field() - 1;
    }

    Ok(WaveNetModelDyn {
        ch: new_ch,
        k: model.k,
        head: model.head,
        arrays,
        head_scale: model.head_scale,
        receptive_field_size: rf,
        condition_dsp: crate::models::clone_condition_dsp(&model.condition_dsp),
        condition_dsp_output: AlignedVec::new(cond_dsp_output_size, 0.0)?,
        post_stack_head: model.post_stack_head.clone(),
        head_output_scratch,
        prewarm_on_reset: model.prewarm_on_reset,
        slimmable_capable: model.slimmable_capable,
        allowed_channels: model.allowed_channels.clone(),
        pending_slim_channel: None,
    })
}

/// Usage:
/// Creates a full exact clone of a WaveNet model for storage, enabling main-thread
/// slimmable rebuilds. Uses `clone_exact()` so that immutable weights and topology
/// are duplicated with absolute fidelity for both homogeneous and heterogeneous models
/// without invoking channel slicing.
pub fn clone_wavenet_for_slimmable_storage(
    model: &WaveNetModelDyn,
) -> std::io::Result<Box<StaticModel>> {
    let full_copy = model.clone_exact();
    Ok(Box::new(StaticModel::WavenetDyn(Box::new(full_copy))))
}

/// Centralized helper for slimmable WaveNet channel rebuild.
///
/// Checks if a model slot needs a WaveNet channel count change
/// and performs the allocation-intensive `slice_channels` + GC swap.
///
/// Must be called **before** DSP to keep the hot-path zero-alloc.
/// Callers should obtain `target_ch` via `AdaptiveCompute::take_slimmable_rebuild()`
/// before invoking this function.
///
/// `max_buffer_size`: if `Some(n)`, calls `set_max_buffer_size(n)` after prewarm
/// (needed by host plugins). Pass `None` for standalone mode.
/// `on_gc`: callback to dispose the old model (e.g., `gc_cascade` or `push_to_gc`).
/// `on_slice_error`: callback invoked when `slice_channels` fails.
#[inline(always)]
pub fn try_slimmable_rebuild_single(
    model: &mut Option<Box<StaticModel>>,
    target_ch: usize,
    max_buffer_size: Option<usize>,
    on_gc: &mut impl FnMut(GcItem),
    on_slice_error: &mut impl FnMut(),
) {
    if let Some(model_inner) = model.as_ref()
        && let StaticModel::WavenetDyn(w) = model_inner.as_ref()
        && w.ch != target_ch
    {
        match w.slice_channels(target_ch) {
            Ok(mut new_model) => {
                new_model.prewarm();
                if let Some(max) = max_buffer_size
                    && new_model.set_max_buffer_size(max).is_err()
                {
                    return;
                }
                let old = model.replace(Box::new(StaticModel::WavenetDyn(Box::new(new_model))));
                if let Some(old) = old {
                    on_gc(GcItem::Model(old));
                }
            }
            Err(_) => {
                on_slice_error();
            }
        }
    }
}
