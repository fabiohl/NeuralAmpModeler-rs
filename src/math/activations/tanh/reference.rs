// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

// ══════════════════════════════════════════════════════════════════════════════
// Reference Variants — Padé NR1 / NR2
// ══════════════════════════════════════════════════════════════════════════════
//
// Padé NR2 (E8.T04): reference showing double Newton-Raphson saturates
// f32 mantissa; error ratio NR2/Div = 1.000×. Retained for documentation.
//
// Padé NR1 (A8): experimental single Newton-Raphson.  Evaluates whether
// rcp_ps + 1×NR (~23-bit precision) is sufficient given the Padé [5,4]
// intrinsic error of ~2.32e-3.  Higher throughput potential than div_ps.
//
//   Benchmarked 2026-06-22: NR1 passes precision (−144 dB vs −80 dB gate)
//   but div_ps is 1.77× faster (62 ns vs 110 ns on 256-elem AVX2).
//   Decision: keep div_ps in production.  NR1 retained for documentation.
//
// NOTE: The production path (simd_tanh_avx2 / simd_tanh_avx512) now uses
// the Padé [5,4] + hardware-div approach. The NR1/NR2 variants below are
// kept for reference / benchmarking.
//
// Padé [5,4]: tanh(x) ≈ x * P(x²) / Q(x²)
//   P(t) = t² + 105t + 945
//   Q(t) = 15t² + 420t + 945
//
// Variants:
//   - _nr1: single Newton-Raphson on rcp → ~23 bits (experimental)
//   - _nr2: double Newton-Raphson on rcp → saturates f32 mantissa (24 bits)
//   - _div: hardware division → IEEE 754 full precision oracle

use crate::math::constants::*;
use core::arch::x86_64::*;

/// Padé \[5,4\] rational tanh with single Newton-Raphson (AVX2).
///
/// Experimental variant (A8). Single NR yields ~23-bit reciprocal precision;
/// the residual ~1 ulp of reciprocal error must be weighed against the
/// Padé approximation's intrinsic error of ~2.32e-3.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_pade_nr1_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));

    let x_sq = _mm256_mul_ps(x, x);

    let num_a = _mm256_set1_ps(PADE_TANH_NUM_A); // 105.0
    let num_b = _mm256_set1_ps(PADE_TANH_NUM_B); // 945.0
    let num = _mm256_add_ps(x_sq, num_a);
    let num = _mm256_fmadd_ps(num, x_sq, num_b);
    let num = _mm256_mul_ps(x, num);

    let den_c4 = _mm256_set1_ps(PADE_TANH_DEN_C4); // 15.0
    let den_c2 = _mm256_set1_ps(PADE_TANH_DEN_C2); // 420.0
    let den_a = _mm256_set1_ps(PADE_TANH_DEN_A); // 945.0
    let den = _mm256_fmadd_ps(x_sq, den_c4, den_c2);
    let den = _mm256_fmadd_ps(den, x_sq, den_a);

    // Single Newton-Raphson (~23 bits)
    let mut r = _mm256_rcp_ps(den);
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(den, r, two));

    let result = _mm256_mul_ps(num, r);
    _mm256_max_ps(neg_one, _mm256_min_ps(one, result))
}

/// Padé \[5,4\] rational tanh with single Newton-Raphson — dual 16-float path (AVX2).
///
/// Experimental variant (A8). Evaluates two independent `__m256` registers.
/// Coefficients are broadcast once and shared between both lanes.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_pade_nr1_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x1 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x1));
    let x2 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x2));

    let sq1 = _mm256_mul_ps(x1, x1);
    let sq2 = _mm256_mul_ps(x2, x2);

    let num_a = _mm256_set1_ps(PADE_TANH_NUM_A); // 105.0
    let num_b = _mm256_set1_ps(PADE_TANH_NUM_B); // 945.0
    let den_c4 = _mm256_set1_ps(PADE_TANH_DEN_C4); // 15.0
    let den_c2 = _mm256_set1_ps(PADE_TANH_DEN_C2); // 420.0
    let den_a = _mm256_set1_ps(PADE_TANH_DEN_A); // 945.0

    // Lane 1 — numerator
    let num1 = _mm256_fmadd_ps(_mm256_add_ps(sq1, num_a), sq1, num_b);
    let num1 = _mm256_mul_ps(x1, num1);
    // Lane 1 — denominator
    let den1 = _mm256_fmadd_ps(_mm256_fmadd_ps(sq1, den_c4, den_c2), sq1, den_a);

    // Lane 2 — numerator
    let num2 = _mm256_fmadd_ps(_mm256_add_ps(sq2, num_a), sq2, num_b);
    let num2 = _mm256_mul_ps(x2, num2);
    // Lane 2 — denominator
    let den2 = _mm256_fmadd_ps(_mm256_fmadd_ps(sq2, den_c4, den_c2), sq2, den_a);

    // Single Newton-Raphson (~23 bits)
    let mut r1 = _mm256_rcp_ps(den1);
    r1 = _mm256_mul_ps(r1, _mm256_fnmadd_ps(den1, r1, two));
    let mut r2 = _mm256_rcp_ps(den2);
    r2 = _mm256_mul_ps(r2, _mm256_fnmadd_ps(den2, r2, two));

    let res1 = _mm256_mul_ps(num1, r1);
    let res2 = _mm256_mul_ps(num2, r2);

    (
        _mm256_max_ps(neg_one, _mm256_min_ps(one, res1)),
        _mm256_max_ps(neg_one, _mm256_min_ps(one, res2)),
    )
}

/// Padé \[5,4\] rational tanh with single Newton-Raphson (AVX-512).
///
/// Experimental variant (A8). Single NR yields ~23-bit reciprocal precision.
///
/// # Safety
/// The caller must guarantee AVX-512F, AVX-512VL, and AVX-512DQ support.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512dq")]
pub unsafe fn simd_tanh_pade_nr1_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm512_set1_ps(PADE_TANH_CLAMP);
    let two = _mm512_set1_ps(2.0);
    let one = _mm512_set1_ps(1.0);
    let neg_one = _mm512_set1_ps(-1.0);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));

    let x_sq = _mm512_mul_ps(x, x);

    let num_a = _mm512_set1_ps(PADE_TANH_NUM_A);
    let num_b = _mm512_set1_ps(PADE_TANH_NUM_B);
    let num = _mm512_add_ps(x_sq, num_a);
    let num = _mm512_fmadd_ps(num, x_sq, num_b);
    let num = _mm512_mul_ps(x, num);

    let den_c4 = _mm512_set1_ps(PADE_TANH_DEN_C4);
    let den_c2 = _mm512_set1_ps(PADE_TANH_DEN_C2);
    let den_a = _mm512_set1_ps(PADE_TANH_DEN_A);
    let den = _mm512_fmadd_ps(x_sq, den_c4, den_c2);
    let den = _mm512_fmadd_ps(den, x_sq, den_a);

    // Single Newton-Raphson (~23 bits)
    let mut r = _mm512_rcp14_ps(den);
    r = _mm512_mul_ps(r, _mm512_fnmadd_ps(den, r, two));

    let result = _mm512_mul_ps(num, r);
    _mm512_max_ps(neg_one, _mm512_min_ps(one, result))
}

/// Padé \[5,4\] rational tanh with double Newton-Raphson (AVX2).
///
/// Reference variant (E8.T04). Double NR saturates f32 mantissa (24 bits);
/// error ratio vs hardware-div = 1.000× — reciprocal contributes zero drift.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_pade_nr2_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));

    let x_sq = _mm256_mul_ps(x, x);

    // Numerator: x * ((x² + 105) * x² + 945)
    // Horner: P(t) = (t + 105) * t + 945  where t = x²
    let num_a = _mm256_set1_ps(PADE_TANH_NUM_A); // 105.0
    let num_b = _mm256_set1_ps(PADE_TANH_NUM_B); // 945.0
    let num = _mm256_add_ps(x_sq, num_a); // x² + 105
    let num = _mm256_fmadd_ps(num, x_sq, num_b); // (x² + 105) * x² + 945
    let num = _mm256_mul_ps(x, num);

    // Denominator: (15 * x² + 420) * x² + 945
    // Horner: Q(t) = (15*t + 420) * t + 945  where t = x²
    let den_c4 = _mm256_set1_ps(PADE_TANH_DEN_C4); // 15.0
    let den_c2 = _mm256_set1_ps(PADE_TANH_DEN_C2); // 420.0
    let den_a = _mm256_set1_ps(PADE_TANH_DEN_A); // 945.0
    let den = _mm256_fmadd_ps(x_sq, den_c4, den_c2); // 15*x² + 420
    let den = _mm256_fmadd_ps(den, x_sq, den_a); // (15*x² + 420) * x² + 945

    // Reciprocal with double Newton-Raphson (saturates f32 mantissa)
    let mut r = _mm256_rcp_ps(den);
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(den, r, two));
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(den, r, two));

    let result = _mm256_mul_ps(num, r);
    _mm256_max_ps(neg_one, _mm256_min_ps(one, result))
}

/// Padé \[5,4\] rational tanh with double Newton-Raphson (AVX-512).
///
/// Reference variant (E8.T04). Double NR saturates f32 mantissa (24 bits);
/// error ratio vs hardware-div = 1.000× — reciprocal contributes zero drift.
///
/// # Safety
/// The caller must guarantee AVX-512F, AVX-512VL, and AVX-512DQ support.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512dq")]
pub unsafe fn simd_tanh_pade_nr2_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm512_set1_ps(PADE_TANH_CLAMP);
    let two = _mm512_set1_ps(2.0);
    let one = _mm512_set1_ps(1.0);
    let neg_one = _mm512_set1_ps(-1.0);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));

    let x_sq = _mm512_mul_ps(x, x);

    // Numerator: x * ((x² + 105) * x² + 945)
    let num_a = _mm512_set1_ps(PADE_TANH_NUM_A);
    let num_b = _mm512_set1_ps(PADE_TANH_NUM_B);
    let num = _mm512_add_ps(x_sq, num_a);
    let num = _mm512_fmadd_ps(num, x_sq, num_b);
    let num = _mm512_mul_ps(x, num);

    // Denominator: (15 * x² + 420) * x² + 945
    let den_c4 = _mm512_set1_ps(PADE_TANH_DEN_C4);
    let den_c2 = _mm512_set1_ps(PADE_TANH_DEN_C2);
    let den_a = _mm512_set1_ps(PADE_TANH_DEN_A);
    let den = _mm512_fmadd_ps(x_sq, den_c4, den_c2);
    let den = _mm512_fmadd_ps(den, x_sq, den_a);

    let mut r = _mm512_rcp14_ps(den);
    r = _mm512_mul_ps(r, _mm512_fnmadd_ps(den, r, two));
    r = _mm512_mul_ps(r, _mm512_fnmadd_ps(den, r, two));

    let result = _mm512_mul_ps(num, r);
    _mm512_max_ps(neg_one, _mm512_min_ps(one, result))
}

#[cfg(test)]
#[path = "reference_test.rs"]
mod reference_test;
