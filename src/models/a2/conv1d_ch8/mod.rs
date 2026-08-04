// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! CH=8 T=8 frame-tiled tap-major convolution with broadcast-FMA.
//!
//! When `out_ch == 8`, the generic 4-wide interleaved scheme processes frames
//! one at a time. This module implements a **block-level** kernel that processes
//! 8 consecutive frames per tile, amortizing weight loads across all 8 frames
//! via SIMD broadcast-FMA instructions. Weights are stored in **col-major-per-tap**
//! layout (`w[k * 64 + in * 8 + out]`) so that the 8 output-channel weights for
//! a single (tap, input_channel) pair are contiguous — one `_mm256_loadu_ps` loads
//! them all.
//!
//! For each tile of T=8 frames, the inner loop is:
//!
//! ```text
//! a[f][o] += Wcol[o] * h[f]   (o vectorized, h[f] scalar broadcast)
//! ```
//!
//! On x86-64-v3 this emits `vfmadd231ps` (broadcast-FMA).
//!
//! ## Source of truth
//! - `a2_fast.cpp:617-681` (strategy `Channels >= 8`, T=4 tap-major).
//!   Elevated to T=8 to saturate FMA ports.

use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

/// Maximum frames per kernel invocation.
/// Guaranteed by `process()` internal chunking.
const MAX_KERNEL_FRAMES: usize = WAVENET_MAX_NUM_FRAMES;

/// CH=8 dilated causal Conv1D alias — see `super::conv1d_ch::A2Conv1dCh` for docs.
pub type A2Conv1dCh8 = super::conv1d_ch::A2Conv1dCh<8>;

mod scalar;
mod simd;
pub use scalar::*;
pub use simd::*;

#[cfg(test)]
#[path = "conv1d_ch8_test.rs"]
mod tests;
