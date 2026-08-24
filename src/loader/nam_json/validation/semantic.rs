// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Semantic (cross-field / topology) validation constants for `.nam` format.
//!
//! Defines architecture-specific bounds and topology constraints used by
//! the dispatcher and topology detection modules after the structural
//! (per-field) parse-time guards have passed.

// ── Topology bounds (DoS/OOM prevention) ──

/// Maximum number of LSTM layers accepted from model config.
pub const MAX_LSTM_LAYERS: usize = 16;

/// Maximum LSTM hidden size accepted from model config.
pub const MAX_LSTM_HIDDEN_SIZE: usize = 1024;

/// Maximum channels per layer-array for WaveNet A1 free-geometry (non-catalog SKU).
pub const MAX_WAVENET_FREE_CHANNELS: usize = 512;

/// Maximum channels for A2-Dynamic (A2 that doesn't fit the fast-path).
pub const MAX_A2_DYN_CHANNELS: usize = 256;

/// Maximum bottleneck size for A2-Dynamic.
pub const MAX_A2_DYN_BOTTLENECK: usize = 256;

// ── Topology bounds (DoS/OOM prevention — F2) ──

/// Maximum kernel size accepted from model config.
/// Larger values cause O(n³) all-pair computations in the hot-path.
///
/// Aligned with [`crate::models::wavenet::common::MAX_KERNEL`]: the dynamic
/// WaveNet convolution kernels (`Conv1dDyn` / dual-frame path) read taps through
/// a fixed `[null; MAX_KERNEL]` array, so a kernel above this cap would either
/// silently drop taps or compute out-of-bounds negative offsets (F-01).
pub const MAX_KERNEL_SIZE: usize = crate::models::wavenet::common::MAX_KERNEL;

/// Maximum dilation factor accepted from model config.
/// Unbounded dilations create oversized receptive fields and kernel striding.
pub const MAX_DILATION: usize = 4096;

/// Maximum number of dilations per layer-array.
/// Each dilation adds a full Conv1D+activation stack.
pub const MAX_DILATIONS_PER_ARRAY: usize = 64;

/// Maximum number of WaveNet layer-arrays.
pub const MAX_WAVENET_ARRAYS: usize = 8;

/// Maximum head_size (head projection dimension) accepted from model config.
pub const MAX_HEAD_SIZE: usize = 512;

/// Maximum channels per block for ConvNet.
pub const MAX_CONVNET_CHANNELS: usize = 512;

/// Maximum kernel size per block for ConvNet.
pub const MAX_CONVNET_KERNEL_SIZE: usize = 64;

/// Maximum receptive field (in samples) for the Linear architecture.
/// Limited by the weight array cap (MAX_WEIGHTS) plus a generous margin.
pub const MAX_RECEPTIVE_FIELD: usize = 65536;

/// Aggregate cap for all WaveNet layer state frames (pre-allocated mirrored buffers).
/// Prevents DoS via receptive-field amplification. Default: 64 Mi frames ≈ 256 MB @ f32.
/// Each "frame" represents one sample per channel across all layer delay-line buffers.
pub const MAX_TOTAL_STATE_FRAMES: usize = 1 << 26;

/// Semantic cap on `condition_size` accepted from `.nam` JSON files (H-03).
///
/// Real conditioning sizes are ≤ 64 channels. This cap bounds the
/// `cond_scratch` allocation (proportional to `condition_size`) so a hostile
/// or corrupted file cannot trigger OOM-class allocations on the loader path,
/// for both the A2-Dynamic builder and the WaveNet A1 free-geometry route.
pub const MAX_CONDITION_SIZE: usize = 4096;
