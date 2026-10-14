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
    Conv1dDyn, DenseLayerDyn, HeadRechannelDyn, WAVENET_MAX_NUM_FRAMES, WaveNetLayerArrayDyn,
    WaveNetLayerDyn, WaveNetLayerState, WaveNetModelDyn,
};
use crate::models::{NamModel, StaticModel};

/// Typed error representing slicing failures in `SlimmableModel`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
    /// Target channels slice length does not match model layer array count.
    ArrayCountMismatch {
        /// Requested number of target channels.
        requested: usize,
        /// Number of layer arrays available in model.
        available: usize,
    },
    /// Target channel count for an array exceeds loaded array channel count.
    TargetExceedsArrayChannels {
        /// Layer array index (0-based).
        array_idx: usize,
        /// Requested channel count.
        requested: usize,
        /// Loaded channel count of this array.
        available: usize,
    },
    /// Target channel count is not in the array's allowed_channels tier list.
    TargetNotInAllowedList {
        /// Layer array index (0-based).
        array_idx: usize,
        /// Requested channel count.
        requested: usize,
        /// Allowed channels configured for this array.
        allowed: Vec<usize>,
    },
    /// Target channel count on a fixed array must equal loaded channels.
    FixedArrayTargetMismatch {
        /// Layer array index (0-based).
        array_idx: usize,
        /// Requested channel count.
        requested: usize,
        /// Loaded channel count of this fixed array.
        fixed: usize,
    },
    /// Head rechannel convolution kernel size must be 1.
    UnsupportedHeadKernelSize {
        /// Head rechannel kernel size encountered.
        kernel: usize,
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
            Self::ArrayCountMismatch {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Slicing array count mismatch: requested {requested} targets for {available} arrays"
                )
            }
            Self::TargetExceedsArrayChannels {
                array_idx,
                requested,
                available,
            } => {
                write!(
                    f,
                    "Slicing target channel count ({requested}) for array {array_idx} exceeds loaded channels ({available})"
                )
            }
            Self::TargetNotInAllowedList {
                array_idx,
                requested,
                allowed,
            } => {
                write!(
                    f,
                    "Slicing target channel count ({requested}) for array {array_idx} is not in allowed list {allowed:?}"
                )
            }
            Self::FixedArrayTargetMismatch {
                array_idx,
                requested,
                fixed,
            } => {
                write!(
                    f,
                    "Slicing fixed array {array_idx} requires target {fixed}, got {requested}"
                )
            }
            Self::UnsupportedHeadKernelSize { kernel } => {
                write!(
                    f,
                    "Slicing head rechannel kernel size must be 1, found {kernel}"
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

    // Release-stable padding/bounds proof (R-2): `new_weights` is allocated to
    // exactly the SIMD-padded total for `dst_width`, so the validated
    // constructor re-check below cannot fail for the sizes computed above
    // (a panic here would indicate a slicing-layout regression, not an OOB).
    Ok(Conv1dDyn::try_from_parts(
        new_weights,
        new_bias,
        conv.do_bias,
        conv.dilation,
        new_in_ch,
        new_out_ch,
        kernel,
        dst_width,
    )
    .expect("slice_conv1d output buffer is exactly SIMD-padded by construction"))
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

    Ok(
        DenseLayerDyn::try_from_parts(new_weights, new_bias, dense.do_bias, new_in_ch, new_out_ch)
            .expect("slice_dense output buffer is correctly sized by construction"),
    )
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
///              target channels of the previous array for subsequent arrays).
/// `new_ch`: internal layer channels for this array.
/// `new_head`: head rechannel output channels (target channels of the next array,
///             or final model head channels for the last array).
/// `alloc_num`: allocation counter for state jitter (pass a `&mut usize`
///              that increments across all arrays in the model).
pub fn slice_wavenet_array(
    array: &WaveNetLayerArrayDyn,
    new_in_ch: usize,
    new_ch: usize,
    new_head: usize,
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

    let (head_rechannel, head_rf) = match &array.head_rechannel {
        HeadRechannelDyn::Dense(dense) => (
            HeadRechannelDyn::Dense(slice_dense(dense, new_ch, new_head)?),
            0,
        ),
        HeadRechannelDyn::Conv1d { conv, state } => {
            if conv.kernel != 1 {
                return Err(SlicingError::UnsupportedHeadKernelSize {
                    kernel: conv.kernel,
                });
            }
            let sliced_conv = slice_conv1d(conv, new_ch, new_head)?;
            let rf = state.receptive_field_size;
            let new_state = WaveNetLayerState::new(new_ch, rf, *alloc_num)
                .map_err(|_| SlicingError::Allocation(NamErrorCode::OutOfMemory))?;
            *alloc_num += 1;
            (
                HeadRechannelDyn::Conv1d {
                    conv: sliced_conv,
                    state: new_state,
                },
                rf,
            )
        }
    };

    let receptive_field_size: usize = array
        .layers
        .iter()
        .map(|l| (l.conv1d.kernel - 1) * l.conv1d.dilation)
        .sum::<usize>()
        + head_rf;

    let block_size = new_ch;
    let num_layers = layers.len();

    Ok(WaveNetLayerArrayDyn {
        in_ch: new_in_ch,
        cond: array.cond,
        ch: new_ch,
        k: array.k,
        head: new_head,
        layers,
        states,
        rechannel,
        head_rechannel,
        array_outputs: AlignedVec::new(new_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        head_accum: AlignedVec::new(new_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        head_outputs: AlignedVec::new(new_head * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        receptive_field_size,
        block_size,
        block_buffer: AlignedVec::new(block_size * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        effective_layers: num_layers,
    })
}

/// Creates a new `WaveNetModelDyn` with per-array target channel counts.
///
/// This is the primary heterogeneous slicing entry point.
///
/// # Slicing Rules & Topology Chain
///
/// For array `i` in `0..n`:
/// - `slim_ch = targets[i]`
/// - `slim_input_size = if i == 0 { array.in_ch } else { targets[i - 1] }`
/// - `slim_head_size = if i < n - 1 { targets[i + 1] } else { array.head }`
///
/// # Validation & Invariants
///
/// - `targets.len() == model.arrays.len()`
/// - Each target `t ∈ [1, array.ch]` (slicing can only reduce channels; expanding
///   beyond loaded width is an error — store the full-width model via
///   [`clone_wavenet_for_slimmable_storage`] for future re-slicing).
/// - If `allowed_channels` exists for array `i`:
///   - If `allowed.is_empty()` (fixed array), `targets[i] == array.ch` is required.
///   - If `!allowed.is_empty()`, `targets[i]` must be in `allowed`.
///
/// # Allocation & RT Safety
///
/// Allocation is strictly off-RT (main thread / background worker).
pub fn slice_wavenet_model_per_array(
    model: &WaveNetModelDyn,
    targets: &[usize],
) -> Result<WaveNetModelDyn, SlicingError> {
    if targets.len() != model.arrays.len() {
        return Err(SlicingError::ArrayCountMismatch {
            requested: targets.len(),
            available: model.arrays.len(),
        });
    }

    for (i, (&t, array)) in targets.iter().zip(model.arrays.iter()).enumerate() {
        if t == 0 {
            return Err(SlicingError::ZeroChannelCount);
        }
        if t > array.ch {
            return Err(SlicingError::TargetExceedsArrayChannels {
                array_idx: i,
                requested: t,
                available: array.ch,
            });
        }
        if let Some(allowed) = model.allowed_channels.as_ref().and_then(|p| p.get(i)) {
            if allowed.is_empty() {
                if t != array.ch {
                    return Err(SlicingError::FixedArrayTargetMismatch {
                        array_idx: i,
                        requested: t,
                        fixed: array.ch,
                    });
                }
            } else if !allowed.contains(&t) {
                return Err(SlicingError::TargetNotInAllowedList {
                    array_idx: i,
                    requested: t,
                    allowed: allowed.clone(),
                });
            }
        }
    }

    let num_arrays = model.arrays.len();
    let mut alloc_num = 0usize;
    let mut arrays = Vec::with_capacity(num_arrays);

    for (i, array) in model.arrays.iter().enumerate() {
        let in_ch = if i == 0 { array.in_ch } else { targets[i - 1] };
        let ch = targets[i];
        let head = if i < num_arrays - 1 {
            targets[i + 1]
        } else {
            array.head
        };
        arrays.push(slice_wavenet_array(array, in_ch, ch, head, &mut alloc_num)?);
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

    let final_head = arrays.last().map(|a| a.head).unwrap_or(model.head);

    Ok(WaveNetModelDyn {
        ch: arrays[0].ch,
        k: model.k,
        head: final_head,
        arrays,
        head_scale: model.head_scale,
        receptive_field_size: rf,
        condition_dsp: crate::models::clone_condition_dsp(&model.condition_dsp),
        condition_dsp_output: AlignedVec::new(cond_dsp_output_size, 0.0)?,
        post_stack_head: model.post_stack_head.clone(),
        head_output_scratch,
        prewarm_on_reset: model.prewarm_on_reset,
        prewarm_pending: false,
        slimmable_capable: model.slimmable_capable,
        allowed_channels: model.allowed_channels.clone(),
        pending_slim_channel: None,
    })
}

/// Creates a new `WaveNetModelDyn` with uniform channel reduction on all slimmable arrays.
///
/// Arrays marked fixed (empty `allowed_channels`) retain their loaded channel count.
/// Slicing delegates to [`slice_wavenet_model_per_array`].
///
/// # Errors
/// Returns [`SlicingError`] if `new_ch == 0` or target exceeds array channel counts.
pub fn slice_wavenet_model(
    model: &WaveNetModelDyn,
    new_ch: usize,
) -> Result<WaveNetModelDyn, SlicingError> {
    if new_ch == 0 {
        return Err(SlicingError::ZeroChannelCount);
    }
    let mut targets = Vec::with_capacity(model.arrays.len());
    for (i, array) in model.arrays.iter().enumerate() {
        let is_fixed = model
            .allowed_channels
            .as_ref()
            .and_then(|ac| ac.get(i))
            .map(|a| a.is_empty())
            .unwrap_or(false);
        if is_fixed {
            targets.push(array.ch);
        } else {
            targets.push(new_ch);
        }
    }
    slice_wavenet_model_per_array(model, &targets)
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
/// Checks if a model slot needs a WaveNet channel count change
/// and performs the allocation-intensive `slice_channels_per_array` + GC swap.
///
/// Must be called **off-RT** (before DSP or in background thread) to keep the hot-path zero-alloc.
/// Callers can obtain `targets` via [`crate::dsp::adaptive::AdaptiveCompute::take_slimmable_rebuild`]
/// (or the transactional `peek_slimmable_rebuild` +
/// `confirm_slimmable_rebuild` pair) before invoking this function.
///
/// Returns `true` only when the model was actually rebuilt and swapped.
/// A `false` return means no rebuild happened (already at targets, wrong
/// variant, or rebuild failed) — the caller's pending request must be kept
/// so it retries instead of being silently dropped.
///
/// `max_buffer_size`: if `Some(n)`, calls `set_max_buffer_size(n)` after prewarm
/// (needed by host plugins). Pass `None` for standalone mode.
/// `on_gc`: callback to dispose the old model (e.g., `gc_cascade` or `push_to_gc`).
/// `on_slice_error`: callback invoked when slicing fails.
pub fn try_slimmable_rebuild_single(
    model: &mut Option<Box<StaticModel>>,
    targets: &[usize],
    max_buffer_size: Option<usize>,
    on_gc: &mut impl FnMut(GcItem),
    on_slice_error: &mut impl FnMut(),
) -> bool {
    if let Some(model_inner) = model.as_ref()
        && let StaticModel::WavenetDyn(w) = model_inner.as_ref()
    {
        let needs_rebuild = w.arrays.len() != targets.len()
            || w.arrays.iter().zip(targets.iter()).any(|(a, &t)| a.ch != t);
        if needs_rebuild {
            match w.slice_channels_per_array(targets) {
                Ok(mut new_model) => {
                    new_model.pending_slim_channel = None;
                    new_model.prewarm();
                    if let Some(max) = max_buffer_size
                        && new_model.set_max_buffer_size(max).is_err()
                    {
                        on_slice_error();
                        return false;
                    }
                    let old = model.replace(Box::new(StaticModel::WavenetDyn(Box::new(new_model))));
                    if let Some(old) = old {
                        on_gc(GcItem::Model(old));
                    }
                    return true;
                }
                Err(_) => {
                    on_slice_error();
                    return false;
                }
            }
        }
    }
    false
}

/// Checks if a model slot has a pending slimmable channel request staged via
/// `set_slimmable_size`, and if so, performs the off-RT rebuild and GC swap.
///
/// Transactional: the pending request is only consumed (cleared) when the
/// rebuild actually succeeds. A failed rebuild keeps `pending_slim_channel`
/// intact so the next call retries.
///
/// Returns `true` if a rebuild occurred.
pub fn try_slimmable_rebuild_from_pending(
    model: &mut Option<Box<StaticModel>>,
    max_buffer_size: Option<usize>,
    on_gc: &mut impl FnMut(GcItem),
    on_slice_error: &mut impl FnMut(),
) -> bool {
    let targets = if let Some(model_inner) = model.as_ref()
        && let StaticModel::WavenetDyn(w) = model_inner.as_ref()
    {
        w.pending_slim_channel.clone()
    } else {
        None
    };

    if let Some(targets) = targets {
        if try_slimmable_rebuild_single(model, &targets, max_buffer_size, on_gc, on_slice_error) {
            // Success: the replacement model already carries
            // `pending_slim_channel = None` (cleared in `_single`); nothing
            // left to consume.
            true
        } else {
            // Failure (or no-op): keep the pending request for retry.
            false
        }
    } else {
        false
    }
}
