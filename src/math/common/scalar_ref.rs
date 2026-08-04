// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Scalar Reference Implementations for DSP math kernels.
//!
//! # Purpose
//!
//! The role of this module is to be the **scalar reference implementation**: simple,
//! correct, unoptimized algorithms that serve as an oracle in parity tests
//! for AVX2 and AVX-512 kernels. Every vector kernel must produce the same
//! numerical result (within floating-point tolerance) as its scalar counterpart here.
//!
//! # Source of Truth
//!
//! The definitive mathematical specification of each operation is NeuralAmpModelerCore (C++).
//! The implementations here are faithful scalar translations of that reference code,
//! used exclusively for internal validation.

// Re-exports of Wavenet fallbacks (Test Only).
#[cfg(test)]
pub use crate::math::wavenet::accumulate::{
    accumulate_head_fallback, gated_activation_and_accumulate_block_fallback,
    gated_activation_and_overwrite_block_fallback, tanh_and_accumulate_block_fallback,
    tanh_and_accumulate_with_seed_fallback, tanh_and_overwrite_block_fallback,
};

/// Convolution fallback implementations.
pub mod convolution;
/// Dot product fallback implementations.
pub mod dot;
/// GEMM and GEMV fallback implementations.
pub mod gemm;
/// LSTM gate fallback implementations.
pub mod lstm;
/// Miscellaneous utility and helper fallback implementations.
pub mod utility;

pub use convolution::*;
pub use dot::*;
pub use gemm::*;
pub use lstm::*;
pub use utility::*;
