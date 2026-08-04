// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! SIMD kernels for LSTM (Long Short-Term Memory) networks.
//!
//! This module focuses on state update efficiency for LSTM cells,
//! using operation fusion to reduce inference latency.
//!
//! # Highlights
//! - **Fused Gate Processing**: The 4 gates (input, forget, cell, output) are
//!   computed in a single pass over SIMD registers.
//! - **State Persistence**: Optimized management of `cell_state` and `hidden_state`.
//! - **Parallelism**: Simultaneous processing of 8 (AVX2) or 16 (AVX-512) cells at once.
//!
//! Extracted from `activations/fused.rs` and `simd/avx2.rs`/`simd/avx512.rs`.

pub mod gates;

pub use gates::*;
