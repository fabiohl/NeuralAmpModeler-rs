// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! SIMD kernels for WaveNet architecture (Accumulate, Gated Activation).
//!
//! This module contains highly specialized kernels for the WaveNet layer cascade,
//! focusing on minimizing memory accesses and maximizing arithmetic density.
//!
//! # Critical Operations
//! - **Head Accumulate**: Sum of skip-connection vectors for the final output.
//! - **Gated Activation**: Fusion of `tanh` and `sigmoid` in a single SIMD kernel.
//! - **Dilated Conv Fetches**: Data fetch optimization for delay lines.
//!

pub mod accumulate;

pub use accumulate::*;
