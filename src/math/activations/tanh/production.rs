// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized Tanh (Hyperbolic Tangent) activation kernels — production path.
//!
//! **Production path:** Padé \[5,4\] rational approximant with hardware `_mm256_div_ps`.
//!
//! ```text
//! tanh(x) ≈ x * (x² + 105) * (x² + 945) / ((15x² + 420) * x² + 945)
//! ```
//!
//! - Max absolute error: ~2.32e-3 vs `f32::tanh` on [-4, 4].
//! - Throughput: ~9 SIMD ops.
//!
//! Coefficients in `crate::math::constants` (`PADE_TANH_*`).

use crate::activation_simd_avx2;
#[cfg(feature = "avx512")]
use crate::activation_simd_avx512;
use crate::math::constants::*;
use core::arch::x86_64::*;

/// Padé \[5,4\] rational approximant for `tanh(x)` with hardware division (AVX2).
///
/// Production path. ~9 SIMD ops, max absolute error ~2.32e-3.
///
/// Formula: `tanh(x) ≈ x·(x²+105)·(x²+945) / ((15x²+420)·x²+945)`
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));

    let x_sq = _mm256_mul_ps(x, x);

    // Numerator: x * ((x² + 105) * x² + 945)
    // Horner: P(t) = (t + 105) * t + 945  where t = x²
    let num_a = _mm256_set1_ps(PADE_TANH_NUM_A); // 105.0
    let num_b = _mm256_set1_ps(PADE_TANH_NUM_B); // 945.0
    let num = _mm256_add_ps(x_sq, num_a);
    let num = _mm256_fmadd_ps(num, x_sq, num_b);
    let num = _mm256_mul_ps(x, num);

    // Denominator: (15 * x² + 420) * x² + 945
    // Horner: Q(t) = (15*t + 420) * t + 945  where t = x²
    let den_c4 = _mm256_set1_ps(PADE_TANH_DEN_C4); // 15.0
    let den_c2 = _mm256_set1_ps(PADE_TANH_DEN_C2); // 420.0
    let den_a = _mm256_set1_ps(PADE_TANH_DEN_A); // 945.0
    let den = _mm256_fmadd_ps(x_sq, den_c4, den_c2);
    let den = _mm256_fmadd_ps(den, x_sq, den_a);

    let result = _mm256_div_ps(num, den);
    _mm256_max_ps(neg_one, _mm256_min_ps(one, result))
}

/// Padé \[5,4\] rational approximant for `tanh(x)` — dual 16-float path (AVX2).
///
/// Evaluates two independent `__m256` registers. Coefficients are broadcast
/// once and shared between both lanes, amortising setup cost.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x1 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x1));
    let x2 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x2));

    let sq1 = _mm256_mul_ps(x1, x1);
    let sq2 = _mm256_mul_ps(x2, x2);

    // Broadcast Padé coefficients once for both lanes.
    let num_a = _mm256_set1_ps(PADE_TANH_NUM_A); // 105.0
    let num_b = _mm256_set1_ps(PADE_TANH_NUM_B); // 945.0
    let den_c4 = _mm256_set1_ps(PADE_TANH_DEN_C4); // 15.0
    let den_c2 = _mm256_set1_ps(PADE_TANH_DEN_C2); // 420.0
    let den_a = _mm256_set1_ps(PADE_TANH_DEN_A); // 945.0

    // Lane 1 — Numerator: x1 * ((x1² + 105) * x1² + 945)
    let num1 = _mm256_fmadd_ps(_mm256_add_ps(sq1, num_a), sq1, num_b);
    let num1 = _mm256_mul_ps(x1, num1);
    // Lane 1 — Denominator: (15 * x1² + 420) * x1² + 945
    let den1 = _mm256_fmadd_ps(_mm256_fmadd_ps(sq1, den_c4, den_c2), sq1, den_a);

    // Lane 2 — Numerator: x2 * ((x2² + 105) * x2² + 945)
    let num2 = _mm256_fmadd_ps(_mm256_add_ps(sq2, num_a), sq2, num_b);
    let num2 = _mm256_mul_ps(x2, num2);
    // Lane 2 — Denominator: (15 * x2² + 420) * x2² + 945
    let den2 = _mm256_fmadd_ps(_mm256_fmadd_ps(sq2, den_c4, den_c2), sq2, den_a);

    let res1 = _mm256_div_ps(num1, den1);
    let res2 = _mm256_div_ps(num2, den2);

    (
        _mm256_max_ps(neg_one, _mm256_min_ps(one, res1)),
        _mm256_max_ps(neg_one, _mm256_min_ps(one, res2)),
    )
}

/// Padé \[5,4\] rational tanh — single 16-float path (AVX-512).
///
/// # Safety
/// The caller must guarantee AVX-512F, AVX-512VL, and AVX-512DQ support.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512dq")]
pub unsafe fn simd_tanh_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm512_set1_ps(PADE_TANH_CLAMP);
    let one = _mm512_set1_ps(1.0);
    let neg_one = _mm512_set1_ps(-1.0);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));

    let x_sq = _mm512_mul_ps(x, x);

    // Numerator: x * ((x² + 105) * x² + 945)
    let num_a = _mm512_set1_ps(PADE_TANH_NUM_A); // 105.0
    let num_b = _mm512_set1_ps(PADE_TANH_NUM_B); // 945.0
    let num = _mm512_add_ps(x_sq, num_a);
    let num = _mm512_fmadd_ps(num, x_sq, num_b);
    let num = _mm512_mul_ps(x, num);

    // Denominator: (15 * x² + 420) * x² + 945
    let den_c4 = _mm512_set1_ps(PADE_TANH_DEN_C4); // 15.0
    let den_c2 = _mm512_set1_ps(PADE_TANH_DEN_C2); // 420.0
    let den_a = _mm512_set1_ps(PADE_TANH_DEN_A); // 945.0
    let den = _mm512_fmadd_ps(x_sq, den_c4, den_c2);
    let den = _mm512_fmadd_ps(den, x_sq, den_a);

    let result = _mm512_div_ps(num, den);
    _mm512_max_ps(neg_one, _mm512_min_ps(one, result))
}

/// Applies Tanh activation to a slice of f32 using AVX2 dual-path.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        activation_simd_avx2!(
            i,
            len,
            {
                let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                let (y1, y2) = simd_tanh_dual_avx2(x1, x2);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), y1);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), y2);
            },
            {
                let x = _mm256_loadu_ps(slice.as_ptr().add(i));
                let y = simd_tanh_avx2(x);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), y);
            }
        );
    }

    for item in slice.iter_mut().skip(i) {
        *item = scalar_pade_tanh(*item);
        if item.abs() < f32::MIN_POSITIVE {
            *item = 0.0;
        }
    }
}

/// Applies Tanh activation to a slice of f32 using AVX-512.
///
/// # Safety
/// Requires AVX-512F, AVX-512VL, and AVX-512DQ support.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512dq")]
pub unsafe fn tanh_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            let y = simd_tanh_avx512(x);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), y);
        });
    }

    for item in slice.iter_mut().skip(i) {
        *item = scalar_pade_tanh(*item);
        if item.abs() < f32::MIN_POSITIVE {
            *item = 0.0;
        }
    }
}

/// Scalar Padé \[5,4\] rational approximation for `tanh(x)`.
///
/// Formula: `x·(x²+105)·(x²+945) / ((15x²+420)·x²+945)`
/// Domain: [-4, 4], output clamped to [-1, 1].
/// Max absolute error: ~2.32e-3 vs `f32::tanh`.
#[inline]
pub fn scalar_pade_tanh(x: f32) -> f32 {
    let x = x.clamp(-PADE_TANH_CLAMP, PADE_TANH_CLAMP);
    let x2 = x * x;

    let num = x * (x2 + PADE_TANH_NUM_A).mul_add(x2, PADE_TANH_NUM_B);
    let den = (PADE_TANH_DEN_C4.mul_add(x2, PADE_TANH_DEN_C2)).mul_add(x2, PADE_TANH_DEN_A);

    (num / den).clamp(-1.0, 1.0)
}

/// Scalar version of `tanh` — delegates to the Padé rational approximation.
#[inline]
pub fn tanh(x: f32) -> f32 {
    scalar_pade_tanh(x)
}
