// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized Sigmoid (Logistic) activation kernels.
//!
//! Uses a direct degree-17 minimax polynomial (9 odd terms) optimised
//! for the interval [-8, 8], computed via Lawson's weighted minimax
//! algorithm.  Max absolute error: ~4.09e-4 vs `f32::exp` reference.
//!
//! This eliminates the previous dependence on the `tanh(x/2)` identity
//! and its associated rescaling and error propagation.

use crate::activation_simd_avx2;
#[cfg(feature = "avx512")]
use crate::activation_simd_avx512;
use crate::math::constants::*;
use core::arch::x86_64::*;

/// Direct minimax polynomial approximation of `sigmoid(x)` (AVX2).
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_sigmoid_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-SIGMOID_MINIMAX_CLAMP);
    let clamp_hi = _mm256_set1_ps(SIGMOID_MINIMAX_CLAMP);
    let c0 = _mm256_set1_ps(SIGMOID_MINIMAX_C0);
    let c1 = _mm256_set1_ps(SIGMOID_MINIMAX_C1);
    let c2 = _mm256_set1_ps(SIGMOID_MINIMAX_C2);
    let c3 = _mm256_set1_ps(SIGMOID_MINIMAX_C3);
    let c4 = _mm256_set1_ps(SIGMOID_MINIMAX_C4);
    let c5 = _mm256_set1_ps(SIGMOID_MINIMAX_C5);
    let c6 = _mm256_set1_ps(SIGMOID_MINIMAX_C6);
    let c7 = _mm256_set1_ps(SIGMOID_MINIMAX_C7);
    let c8 = _mm256_set1_ps(SIGMOID_MINIMAX_C8);
    let half = _mm256_set1_ps(0.5);
    let zero = _mm256_set1_ps(0.0);
    let one = _mm256_set1_ps(1.0);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));
    let x2 = _mm256_mul_ps(x, x);

    let p = _mm256_fmadd_ps(c8, x2, c7);
    let p = _mm256_fmadd_ps(p, x2, c6);
    let p = _mm256_fmadd_ps(p, x2, c5);
    let p = _mm256_fmadd_ps(p, x2, c4);
    let p = _mm256_fmadd_ps(p, x2, c3);
    let p = _mm256_fmadd_ps(p, x2, c2);
    let p = _mm256_fmadd_ps(p, x2, c1);
    let p = _mm256_fmadd_ps(p, x2, c0);

    let result = _mm256_fmadd_ps(x, p, half);
    _mm256_max_ps(zero, _mm256_min_ps(one, result))
}

/// Direct minimax polynomial approximation of `sigmoid(x)` — dual 16-float path (AVX2).
///
/// Evaluates two independent `__m256` registers. Coefficients are broadcast
/// once and shared between both lanes, amortising setup cost.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_sigmoid_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let clamp_lo = _mm256_set1_ps(-SIGMOID_MINIMAX_CLAMP);
    let clamp_hi = _mm256_set1_ps(SIGMOID_MINIMAX_CLAMP);
    let c0 = _mm256_set1_ps(SIGMOID_MINIMAX_C0);
    let c1 = _mm256_set1_ps(SIGMOID_MINIMAX_C1);
    let c2 = _mm256_set1_ps(SIGMOID_MINIMAX_C2);
    let c3 = _mm256_set1_ps(SIGMOID_MINIMAX_C3);
    let c4 = _mm256_set1_ps(SIGMOID_MINIMAX_C4);
    let c5 = _mm256_set1_ps(SIGMOID_MINIMAX_C5);
    let c6 = _mm256_set1_ps(SIGMOID_MINIMAX_C6);
    let c7 = _mm256_set1_ps(SIGMOID_MINIMAX_C7);
    let c8 = _mm256_set1_ps(SIGMOID_MINIMAX_C8);
    let half = _mm256_set1_ps(0.5);
    let zero = _mm256_set1_ps(0.0);
    let one = _mm256_set1_ps(1.0);

    let x1 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x1));
    let x2 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x2));

    let x2_1 = _mm256_mul_ps(x1, x1);
    let x2_2 = _mm256_mul_ps(x2, x2);

    let p1 = _mm256_fmadd_ps(c8, x2_1, c7);
    let p1 = _mm256_fmadd_ps(p1, x2_1, c6);
    let p1 = _mm256_fmadd_ps(p1, x2_1, c5);
    let p1 = _mm256_fmadd_ps(p1, x2_1, c4);
    let p1 = _mm256_fmadd_ps(p1, x2_1, c3);
    let p1 = _mm256_fmadd_ps(p1, x2_1, c2);
    let p1 = _mm256_fmadd_ps(p1, x2_1, c1);
    let p1 = _mm256_fmadd_ps(p1, x2_1, c0);

    let p2 = _mm256_fmadd_ps(c8, x2_2, c7);
    let p2 = _mm256_fmadd_ps(p2, x2_2, c6);
    let p2 = _mm256_fmadd_ps(p2, x2_2, c5);
    let p2 = _mm256_fmadd_ps(p2, x2_2, c4);
    let p2 = _mm256_fmadd_ps(p2, x2_2, c3);
    let p2 = _mm256_fmadd_ps(p2, x2_2, c2);
    let p2 = _mm256_fmadd_ps(p2, x2_2, c1);
    let p2 = _mm256_fmadd_ps(p2, x2_2, c0);

    let res1 = _mm256_fmadd_ps(x1, p1, half);
    let res2 = _mm256_fmadd_ps(x2, p2, half);

    (
        _mm256_max_ps(zero, _mm256_min_ps(one, res1)),
        _mm256_max_ps(zero, _mm256_min_ps(one, res2)),
    )
}

/// Direct minimax polynomial approximation of `sigmoid(x)` (AVX-512).
///
/// # Safety
/// The caller must guarantee AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_sigmoid_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-SIGMOID_MINIMAX_CLAMP);
    let clamp_hi = _mm512_set1_ps(SIGMOID_MINIMAX_CLAMP);
    let c0 = _mm512_set1_ps(SIGMOID_MINIMAX_C0);
    let c1 = _mm512_set1_ps(SIGMOID_MINIMAX_C1);
    let c2 = _mm512_set1_ps(SIGMOID_MINIMAX_C2);
    let c3 = _mm512_set1_ps(SIGMOID_MINIMAX_C3);
    let c4 = _mm512_set1_ps(SIGMOID_MINIMAX_C4);
    let c5 = _mm512_set1_ps(SIGMOID_MINIMAX_C5);
    let c6 = _mm512_set1_ps(SIGMOID_MINIMAX_C6);
    let c7 = _mm512_set1_ps(SIGMOID_MINIMAX_C7);
    let c8 = _mm512_set1_ps(SIGMOID_MINIMAX_C8);
    let half = _mm512_set1_ps(0.5);
    let zero = _mm512_set1_ps(0.0);
    let one = _mm512_set1_ps(1.0);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));
    let x2 = _mm512_mul_ps(x, x);

    let p = _mm512_fmadd_ps(c8, x2, c7);
    let p = _mm512_fmadd_ps(p, x2, c6);
    let p = _mm512_fmadd_ps(p, x2, c5);
    let p = _mm512_fmadd_ps(p, x2, c4);
    let p = _mm512_fmadd_ps(p, x2, c3);
    let p = _mm512_fmadd_ps(p, x2, c2);
    let p = _mm512_fmadd_ps(p, x2, c1);
    let p = _mm512_fmadd_ps(p, x2, c0);

    let result = _mm512_fmadd_ps(x, p, half);
    _mm512_max_ps(zero, _mm512_min_ps(one, result))
}

/// Applies Sigmoid activation to a slice of f32 using AVX2 optimization.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sigmoid_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        activation_simd_avx2!(
            i,
            len,
            {
                let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                let (y1, y2) = simd_sigmoid_dual_avx2(x1, x2);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), y1);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), y2);
            },
            {
                let x = _mm256_loadu_ps(slice.as_ptr().add(i));
                let y = simd_sigmoid_avx2(x);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), y);
            }
        );
    }

    for item in slice.iter_mut().skip(i) {
        *item = scalar_minimax_sigmoid(*item);
        if item.abs() < f32::MIN_POSITIVE {
            *item = 0.0;
        }
    }
}

/// Applies Sigmoid activation to a slice of f32 using AVX-512 optimization.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn sigmoid_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            let y = simd_sigmoid_avx512(x);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), y);
        });
    }

    for item in slice.iter_mut().skip(i) {
        *item = scalar_minimax_sigmoid(*item);
        if item.abs() < f32::MIN_POSITIVE {
            *item = 0.0;
        }
    }
}

/// Scalar version of `sigmoid` — uses the direct minimax rational approximation.
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    scalar_minimax_sigmoid(x)
}

/// Scalar direct minimax sigmoid (degree 17, 9 odd terms).
///
/// Scalar reference implementation for testing and benchmarks.
/// Mirrors the SIMD kernel.
#[inline]
#[expect(
    clippy::excessive_precision,
    reason = "High-precision constants required for bit-exact numerical validation against reference"
)]
pub fn scalar_minimax_sigmoid(x: f32) -> f32 {
    let c0 = 2.4885319190e-01_f32;
    let c1 = -1.9318685012e-02_f32;
    let c2 = 1.4623214305e-03_f32;
    let c3 = -7.9953400187e-05_f32;
    let c4 = 2.9140652422e-06_f32;
    let c5 = -6.8000246432e-08_f32;
    let c6 = 9.6897239158e-10_f32;
    let c7 = -7.6498626314e-12_f32;
    let c8 = 2.5585471676e-14_f32;

    let x_clamped = x.clamp(-8.0, 8.0);
    let x2 = x_clamped * x_clamped;

    let p = c8.mul_add(x2, c7);
    let p = p.mul_add(x2, c6);
    let p = p.mul_add(x2, c5);
    let p = p.mul_add(x2, c4);
    let p = p.mul_add(x2, c3);
    let p = p.mul_add(x2, c2);
    let p = p.mul_add(x2, c1);
    let p = p.mul_add(x2, c0);

    let result = x_clamped.mul_add(p, 0.5);
    result.clamp(0.0, 1.0)
}
