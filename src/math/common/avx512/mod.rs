// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    // SAFETY: All unsafe blocks in this module are guarded by CPU feature detection
    // (AVX-512F or AVX-512 VNNI+BF16) at dispatch time via SimdMathConfig::current().
    // Each function's doc comment documents specific slice/pointer invariants.
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! AVX-512 implementations of the `SimdMath` trait.
//!
//! Contains `Avx512Math` and `Avx512VnniBf16Math`.
//! `Avx512VnniBf16Math` has real implementations (native BF16 dot product via `_mm512_dpbf16_ps`).
//! Methods delegate to kernel functions in `math::gemm`, `math::wavenet`,
//! `math::lstm`, `math::dsp`, `math::common::ops`, and `math::common::utility`.
//!
//! # Submodules
//! - `gemv`: GEMV/GEMM kernels, dot products, and 4-gate LSTM.
//! - `activations`: Activation functions (tanh/sigmoid), accumulation, and LSTM fusions.
//! - `bf16`: FP32↔BF16 conversions and `store_bf16` wrappers.
//! - `reduce`: Horizontal sum, energy, and max-diff.
//! - `dsp`: Convolution, gain, ramp, and WaveNet head-sum.

mod activations;
mod bf16;
mod dsp;
mod gemv;
mod reduce;

use crate::math::common::InstructionSet;
use crate::math::common::scalar_ref::*;
use crate::math::common::traits::SimdMath;
use core::arch::x86_64::*;

/// SIMD implementation via AVX-512.
/// This struct groups all mathematical functions optimized for processors that support AVX-512.
pub struct Avx512Math;

/// Static implementation for AVX-512 with VNNI and BF16 (Brain Float 16) support.
/// This is the "Ferrari" of audio processing, available on very recent Intel CPUs (e.g.: Sapphire Rapids).
/// The BF16 format allows the chip to process twice the numbers with almost the same precision as the original f32.
pub struct Avx512VnniBf16Math;

// ── Avx512Math ──

impl SimdMath for Avx512Math {
    type V = __m512;

    const ISA: InstructionSet = InstructionSet::Avx512;

    gemv::impl_avx512_gemv!();
    activations::impl_avx512_activations!();
    bf16::impl_avx512_bf16!();
    reduce::impl_avx512_reduce!();
    dsp::impl_avx512_dsp!();
}

// ── Avx512VnniBf16Math ──

impl SimdMath for Avx512VnniBf16Math {
    type V = __m512;

    const ISA: InstructionSet = InstructionSet::Avx512VnniBf16;

    gemv::impl_avx512vnni_bf16_gemv!();
    activations::impl_avx512vnni_bf16_activations!();
    bf16::impl_avx512vnni_bf16_bf16!();
    reduce::impl_avx512vnni_bf16_reduce!();
    dsp::impl_avx512vnni_bf16_dsp!();
}
