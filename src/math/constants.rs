// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(clippy::excessive_precision)]

//! Mathematical constants, Minimax polynomial coefficients, and LUT parameters.
//! Centralized to ensure consistency across scalar, AVX2, and AVX-512 implementations.

// --- Gain LUT Parameters ---

/// Size of the gain LUT table. 4096 points provide sub-0.001 dB precision.
pub const GAIN_LUT_SIZE: usize = 4096;
/// Lower dB limit for the LUT (practical silence floor -96dB).
pub const GAIN_MIN_DB: f32 = -96.0;
/// Upper dB limit for the LUT (+30dB is an extreme boost).
pub const GAIN_MAX_DB: f32 = 30.0;
/// Total gain range in dB supported by the LUT.
pub const GAIN_DB_RANGE: f32 = GAIN_MAX_DB - GAIN_MIN_DB;
/// dB step between each LUT entry.
pub const GAIN_DB_STEP: f32 = GAIN_DB_RANGE / (GAIN_LUT_SIZE as f32 - 1.0);
/// Inverse of the gain step (optimization to avoid divisions in the hot-path).
pub const INV_GAIN_DB_STEP: f32 = 1.0 / GAIN_DB_STEP;

// --- Padé Coefficients for Tanh (Rational [5,4]) ---

/// Numerator constant: x⁴ + 105·x² + 945.
pub const PADE_TANH_NUM_A: f32 = 105.0;
/// Numerator constant.
pub const PADE_TANH_NUM_B: f32 = 945.0;
/// Denominator x⁴ coefficient: 15.
pub const PADE_TANH_DEN_C4: f32 = 15.0;
/// Denominator x² coefficient: 420.
pub const PADE_TANH_DEN_C2: f32 = 420.0;
/// Denominator constant.
pub const PADE_TANH_DEN_A: f32 = 945.0;
/// Clamp limit for the Padé tanh domain (|x| < 4).
pub const PADE_TANH_CLAMP: f32 = 4.0;

// --- Minimax Polynomial Coefficients for Direct Sigmoid (Degree 17, 9 odd terms) ---
// Optimized for the interval [-8, 8] via Lawson's weighted minimax algorithm.
// Max absolute error: ~4.09e-4 vs f32::exp native sigmoid.
/// Input clamp for the direct sigmoid polynomial domain (|x| < 8).
pub const SIGMOID_MINIMAX_CLAMP: f32 = 8.0;
/// Minimax coefficient for x^1 term.
pub const SIGMOID_MINIMAX_C0: f32 = 2.4885319190e-01;
/// Minimax coefficient for x^3 term.
pub const SIGMOID_MINIMAX_C1: f32 = -1.9318685012e-02;
/// Minimax coefficient for x^5 term.
pub const SIGMOID_MINIMAX_C2: f32 = 1.4623214305e-03;
/// Minimax coefficient for x^7 term.
pub const SIGMOID_MINIMAX_C3: f32 = -7.9953400187e-05;
/// Minimax coefficient for x^9 term.
pub const SIGMOID_MINIMAX_C4: f32 = 2.9140652422e-06;
/// Minimax coefficient for x^11 term.
pub const SIGMOID_MINIMAX_C5: f32 = -6.8000246432e-08;
/// Minimax coefficient for x^13 term.
pub const SIGMOID_MINIMAX_C6: f32 = 9.6897239158e-10;
/// Minimax coefficient for x^15 term.
pub const SIGMOID_MINIMAX_C7: f32 = -7.6498626314e-12;
/// Minimax coefficient for x^17 term.
pub const SIGMOID_MINIMAX_C8: f32 = 2.5585471676e-14;

// --- Polynomial exp/activation constants (degree-6 Taylor, range reduction) ---
// Used by `simd_exp_poly_*` and `simd_tanh_poly_*`/`simd_sigmoid_poly_*`.

/// ln(2) for range reduction in `simd_exp_poly`.
#[expect(
    clippy::approx_constant,
    reason = "Constants derived from mathematical formulas, not accidental approximations"
)]
pub const POLY_LN2: f32 = 0.6931471805599453;
/// 1 / ln(2) for range reduction in `simd_exp_poly`.
#[expect(
    clippy::approx_constant,
    reason = "Constants derived from mathematical formulas, not accidental approximations"
)]
pub const POLY_LOG2_E: f32 = 1.4426950408889634;
/// Taylor coefficient for x^6/720 in exp polynomial (degree 6).
pub const POLY_EXP_C6: f32 = 1.3888889220953e-03;
/// Taylor coefficient for x^5/120 in exp polynomial.
pub const POLY_EXP_C5: f32 = 8.3333337679505e-03;
/// Taylor coefficient for x^4/24 in exp polynomial.
pub const POLY_EXP_C4: f32 = 4.1666667908430e-02;
/// Taylor coefficient for x^3/6 in exp polynomial.
pub const POLY_EXP_C3: f32 = 1.6666667163372e-01;
/// Taylor coefficient for x^2/2 in exp polynomial.
pub const POLY_EXP_C2: f32 = 0.5;
/// Domain clamp for polynomial activation inputs (prevents exp overflow).
pub const POLY_ACTIVATION_CLAMP: f32 = 20.0;

// --- Polynomial Tanh Rational [11,10] (exp-based high precision) ---
// tanh(x) = (e^x − e^{-x}) / (e^x + e^{-x}) evaluated via the polynomial exp kernel.
// Max absolute error: ≤ 2.4e-7 vs f32::tanh on [-20, 20].  Throughput: ~19 SIMD ops.
