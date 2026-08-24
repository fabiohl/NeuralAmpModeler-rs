// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized ReLU (Rectified Linear Unit) activation kernels.

use crate::activation_simd_avx2;
#[cfg(feature = "avx512")]
use crate::activation_simd_avx512;
use core::arch::x86_64::*;

/// Vector approximation of `ReLU(x) = max(0, x)` using AVX2.
///
/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
pub unsafe fn simd_relu_avx2(x: __m256) -> __m256 {
    _mm256_max_ps(_mm256_setzero_ps(), x)
}

/// Vector approximation of `ReLU(x)` (Dual, 16 floats).
///
/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
pub unsafe fn simd_relu_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let zero = _mm256_setzero_ps();
    (_mm256_max_ps(zero, x1), _mm256_max_ps(zero, x2))
}

/// Vector approximation of `ReLU(x) = max(0, x)` using AVX-512.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_relu_avx512(x: __m512) -> __m512 {
    _mm512_max_ps(_mm512_setzero_ps(), x)
}

/// Applies ReLU activation to a slice of f32 using AVX2 optimization.
///
/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
pub unsafe fn relu_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();
    let zero = _mm256_setzero_ps();

    // SAFETY: `activation_simd_avx2!` runs the dual body while `i + 16 <= len`
    // (offsets `i`/`i + 8`) and the single body while `i + 8 <= len`, so every
    // 8-lane load/store stays within `slice`; `loadu`/`storeu` need no alignment.
    unsafe {
        activation_simd_avx2!(
            i,
            len,
            {
                let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), _mm256_max_ps(zero, x1));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), _mm256_max_ps(zero, x2));
            },
            {
                let x = _mm256_loadu_ps(slice.as_ptr().add(i));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), _mm256_max_ps(zero, x));
            }
        );
    }

    for item in slice.iter_mut().skip(i) {
        if *item < 0.0 {
            *item = 0.0;
        }
    }
}

/// Applies ReLU activation to a slice of f32 using AVX-512 optimization.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn relu_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();
    let zero = _mm512_setzero_ps();

    // SAFETY: `activation_simd_avx512!` runs its body while `i + 16 <= len` (it
    // loads/stores only at offset `i`), so the 16-lane access stays within
    // `slice`; `loadu`/`storeu` need no alignment.
    unsafe {
        activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), _mm512_max_ps(zero, x));
        });
    }

    for item in slice.iter_mut().skip(i) {
        if *item < 0.0 {
            *item = 0.0;
        }
    }
}

/// Scalar version of `relu` (max(0, x)).
#[inline(always)]
pub fn relu(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}
