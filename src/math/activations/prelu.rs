// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized PReLU (Parametric ReLU) activation kernels.

use crate::activation_simd_avx2;
use crate::activation_simd_avx512;
use core::arch::x86_64::*;

/// Vector approximation of `PReLU(x) = x > 0 ? x : alpha * x` using AVX2.
///
/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
pub unsafe fn simd_prelu_avx2(x: __m256, alpha: __m256) -> __m256 {
    // Mask of positive values (x > 0)
    let mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OQ);
    // alpha * x for the negative region
    let neg_part = _mm256_mul_ps(alpha, x);
    // Selects x if mask is true, otherwise neg_part
    _mm256_blendv_ps(neg_part, x, mask)
}

/// Vector approximation of `PReLU(x) = x > 0 ? x : alpha * x` using AVX-512.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_prelu_avx512(x: __m512, alpha: __m512) -> __m512 {
    let zero = _mm512_setzero_ps();
    // Mask of positive values (x > 0)
    let mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
    // alpha * x for the negative region
    let neg_part = _mm512_mul_ps(alpha, x);
    // Selects x if mask is true, otherwise neg_part (using AVX-512 masking)
    _mm512_mask_blend_ps(mask, neg_part, x)
}

/// Applies PReLU activation to a slice of f32 using AVX2 optimization.
///
/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
pub unsafe fn prelu_slice_avx2(slice: &mut [f32], slopes: &[f32]) {
    let mut i = 0;
    let len = slice.len();
    let s_len = slopes.len();

    if s_len == 0 {
        return;
    }

    // Optimized case: single slope (LeakyReLU)
    if s_len == 1 {
        let alpha = _mm256_set1_ps(slopes[0]);
        unsafe {
            activation_simd_avx2!(
                i,
                len,
                {
                    let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
                    let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                    _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx2(x1, alpha));
                    _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), simd_prelu_avx2(x2, alpha));
                },
                {
                    let x = _mm256_loadu_ps(slice.as_ptr().add(i));
                    _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx2(x, alpha));
                }
            );
        }
    } else if s_len == len {
        // Optimized case: per-element slopes
        unsafe {
            activation_simd_avx2!(
                i,
                len,
                {
                    let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
                    let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                    let a1 = _mm256_loadu_ps(slopes.as_ptr().add(i));
                    let a2 = _mm256_loadu_ps(slopes.as_ptr().add(i + 8));
                    _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx2(x1, a1));
                    _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), simd_prelu_avx2(x2, a2));
                },
                {
                    let x = _mm256_loadu_ps(slice.as_ptr().add(i));
                    let a = _mm256_loadu_ps(slopes.as_ptr().add(i));
                    _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx2(x, a));
                }
            );
        }
    }

    // Scalar fallback (also handles remainder for optimized cases and cycling)
    for idx in i..len {
        let x = slice[idx];
        if x < 0.0 {
            slice[idx] = x * slopes[idx % s_len];
        }
    }
}

/// Applies PReLU activation to a slice of f32 using AVX-512 optimization.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn prelu_slice_avx512(slice: &mut [f32], slopes: &[f32]) {
    let mut i = 0;
    let len = slice.len();
    let s_len = slopes.len();

    if s_len == 0 {
        return;
    }

    if s_len == 1 {
        let alpha = _mm512_set1_ps(slopes[0]);
        unsafe {
            activation_simd_avx512!(i, len, {
                let x = _mm512_loadu_ps(slice.as_ptr().add(i));
                _mm512_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx512(x, alpha));
            });
        }
    } else if s_len == len {
        unsafe {
            activation_simd_avx512!(i, len, {
                let x = _mm512_loadu_ps(slice.as_ptr().add(i));
                let a = _mm512_loadu_ps(slopes.as_ptr().add(i));
                _mm512_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx512(x, a));
            });
        }
    }

    for idx in i..len {
        let x = slice[idx];
        if x < 0.0 {
            slice[idx] = x * slopes[idx % s_len];
        }
    }
}

/// Scalar version of `prelu`.
#[inline(always)]
pub fn prelu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 { x } else { x * alpha }
}
