// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! AVX2 implementations of the `SimdMath` trait.
//!
//! This module contains the `Avx2Math` struct.
//! that implement the `SimdMath` trait using AVX2/FMA instructions.
//! Methods delegate to kernel functions in `math::gemm`, `math::wavenet`,
//! `math::lstm`, `math::dsp`, and `math::common::utility`.

use crate::math::common::InstructionSet;
use crate::math::common::traits::SimdMath;
use core::arch::x86_64::*;

/// Concrete implementation of the SimdMath trait for processors with AVX2 and FMA support.
///
/// This is where we "connect the wires": we connect the abstract mathematical operations of the system
/// to the ultra-fast functions documented above. This struct ensures that NeuralAmpModeler-rs
/// takes full advantage of modern hardware to process audio in real time.
pub struct Avx2Math;

mod activations;
mod bf16;
mod dsp;
mod gemv;
mod reduce;

impl SimdMath for Avx2Math {
    type V = __m256;
    const ISA: InstructionSet = InstructionSet::Avx2;

    gemv::impl_avx2_gemv!();
    activations::impl_avx2_activations!();
    bf16::impl_avx2_bf16!();
    reduce::impl_avx2_reduce!();
    dsp::impl_avx2_dsp!();
}
