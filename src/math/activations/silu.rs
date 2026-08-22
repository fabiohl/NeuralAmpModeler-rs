// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized SiLU (Sigmoid Linear Unit / Swish) activation kernels.

use super::sigmoid::high_fidelity::simd_sigmoid_poly_avx2;
#[cfg(feature = "avx512")]
use super::sigmoid::high_fidelity_avx512::simd_sigmoid_poly_avx512;
#[cfg(feature = "avx512")]
use super::sigmoid::simd_sigmoid_avx512;
use super::sigmoid::{simd_sigmoid_avx2, simd_sigmoid_dual_avx2};
use crate::activation_simd_avx2;
#[cfg(feature = "avx512")]
use crate::activation_simd_avx512;
use core::arch::x86_64::*;

/// Vector approximation of `SiLU(x) = x * sigmoid(x)` using AVX2.
///
/// Reuses the `simd_sigmoid_avx2` kernel (Minimax D6).
///
/// # Safety
/// Requires AVX2 and FMA support.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_silu_avx2(x: __m256) -> __m256 {
    unsafe {
        let s = simd_sigmoid_avx2(x);
        _mm256_mul_ps(x, s)
    }
}

/// Vector approximation of `SiLU(x)` (Dual, 16 floats).
///
/// # Safety
/// Requires AVX2 and FMA support.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_silu_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    unsafe {
        let (s1, s2) = simd_sigmoid_dual_avx2(x1, x2);
        (_mm256_mul_ps(x1, s1), _mm256_mul_ps(x2, s2))
    }
}

/// Vector approximation of `SiLU(x) = x * sigmoid(x)` using AVX-512.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_silu_avx512(x: __m512) -> __m512 {
    unsafe {
        let s = simd_sigmoid_avx512(x);
        _mm512_mul_ps(x, s)
    }
}

/// Applies SiLU activation to a slice of f32 using AVX2 optimization.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn silu_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        activation_simd_avx2!(
            i,
            len,
            {
                let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                let (y1, y2) = simd_silu_dual_avx2(x1, x2);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), y1);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), y2);
            },
            {
                let x = _mm256_loadu_ps(slice.as_ptr().add(i));
                let y = simd_silu_avx2(x);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), y);
            }
        );
    }

    for item in slice.iter_mut().skip(i) {
        *item = *item * super::sigmoid::scalar_minimax_sigmoid(*item);
        if item.abs() < f32::MIN_POSITIVE {
            *item = 0.0;
        }
    }
}

/// Applies SiLU activation to a slice of f32 using AVX-512 optimization.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn silu_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            let y = simd_silu_avx512(x);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), y);
        });
    }

    for item in slice.iter_mut().skip(i) {
        *item = *item * super::sigmoid::scalar_minimax_sigmoid(*item);
        if item.abs() < f32::MIN_POSITIVE {
            *item = 0.0;
        }
    }
}

/// Scalar version of `silu` (x * sigmoid(x)) — uses rational minimax approximation.
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x * super::sigmoid::scalar_minimax_sigmoid(x)
}

/// Scalar version of `silu` using high-fidelity polynomial sigmoid.
#[inline(always)]
pub fn silu_hf(x: f32) -> f32 {
    x * super::sigmoid::high_fidelity::scalar_sigmoid_poly(x)
}

/// High-fidelity SiLU using polynomial exp-based sigmoid (AVX2).
///
/// # Safety
/// Requires AVX2 and FMA support.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_silu_poly_avx2(x: __m256) -> __m256 {
    unsafe {
        let s = simd_sigmoid_poly_avx2(x);
        _mm256_mul_ps(x, s)
    }
}

/// High-fidelity SiLU using polynomial exp-based sigmoid (AVX-512).
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_silu_poly_avx512(x: __m512) -> __m512 {
    unsafe {
        let s = simd_sigmoid_poly_avx512(x);
        _mm512_mul_ps(x, s)
    }
}

/// Applies high-fidelity SiLU to a slice using AVX2 polynomial sigmoid.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn silu_poly_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        activation_simd_avx2!(
            i,
            len,
            {
                let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_silu_poly_avx2(x1));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), simd_silu_poly_avx2(x2));
            },
            {
                let x = _mm256_loadu_ps(slice.as_ptr().add(i));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_silu_poly_avx2(x));
            }
        );
    }

    for item in slice.iter_mut().skip(i) {
        *item = silu_hf(*item);
        if item.abs() < f32::MIN_POSITIVE {
            *item = 0.0;
        }
    }
}

/// Applies high-fidelity SiLU to a slice using AVX-512 polynomial sigmoid.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn silu_poly_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), simd_silu_poly_avx512(x));
        });
    }

    for item in slice.iter_mut().skip(i) {
        *item = silu_hf(*item);
        if item.abs() < f32::MIN_POSITIVE {
            *item = 0.0;
        }
    }
}
