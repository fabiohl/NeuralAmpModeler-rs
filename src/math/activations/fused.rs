// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Fused activation kernels for performance (e.g.: Sigmoid + ReLU, LSTM Gates).
//!
//! Sigmoid now uses a direct degree-17 minimax polynomial independent of tanh.

#[cfg(feature = "avx512")]
use super::relu::simd_relu_avx512;
use super::relu::{simd_relu_avx2, simd_relu_dual_avx2};
#[cfg(feature = "avx512")]
use super::sigmoid::simd_sigmoid_avx512;
use super::sigmoid::{simd_sigmoid_avx2, simd_sigmoid_dual_avx2};
use super::tanh::simd_tanh_avx2;
#[cfg(feature = "avx512")]
use super::tanh::simd_tanh_avx512;
use crate::activation_simd_avx2;
#[cfg(feature = "avx512")]
use crate::activation_simd_avx512;
use core::arch::x86_64::*;

/// Applies Sigmoid followed by ReLU (fused).
///
/// # Safety
/// Requires AVX2 and FMA support.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_fused_sigmoid_relu_avx2(x: __m256) -> __m256 {
    let s = unsafe { simd_sigmoid_avx2(x) };
    unsafe { simd_relu_avx2(s) }
}

/// Applies Sigmoid followed by ReLU (Dual, 16 floats).
///
/// # Safety
/// Requires AVX2 and FMA support.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_fused_sigmoid_relu_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let (s1, s2) = unsafe { simd_sigmoid_dual_avx2(x1, x2) };
    unsafe { simd_relu_dual_avx2(s1, s2) }
}

/// Applies Tanh on x1 and Sigmoid on x2 (Dual).
/// Used in Gated Activation blocks (e.g.: Wavenet).
/// Sigmoid is computed via direct minimax polynomial, independent of tanh.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_sigmoid_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let t1 = unsafe { simd_tanh_avx2(x1) };
    let s2 = unsafe { simd_sigmoid_avx2(x2) };
    (t1, s2)
}

/// Applies Tanh on x1 and Sigmoid on x2 (Dual, AVX-512).
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_tanh_sigmoid_dual_avx512(x1: __m512, x2: __m512) -> (__m512, __m512) {
    let t1 = unsafe { simd_tanh_avx512(x1) };
    let s2 = unsafe { simd_sigmoid_avx512(x2) };
    (t1, s2)
}

/// Applies Sigmoid followed by ReLU (AVX-512).
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_fused_sigmoid_relu_avx512(x: __m512) -> __m512 {
    let s = unsafe { simd_sigmoid_avx512(x) };
    unsafe { simd_relu_avx512(s) }
}

/// Applies fused Sigmoid+ReLU activation to a slice using AVX2.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_sigmoid_relu_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        activation_simd_avx2!(
            i,
            len,
            {
                let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                let (y1, y2) = simd_fused_sigmoid_relu_dual_avx2(x1, x2);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), y1);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), y2);
            },
            {
                let x = _mm256_loadu_ps(slice.as_ptr().add(i));
                let y = simd_fused_sigmoid_relu_avx2(x);
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), y);
            }
        );
    }

    for item in slice.iter_mut().skip(i) {
        let s = super::sigmoid::scalar_minimax_sigmoid(*item);
        *item = if s < 0.0 { 0.0 } else { s };
        if item.abs() < f32::MIN_POSITIVE {
            *item = 0.0;
        }
    }
}

/// Applies fused Sigmoid+ReLU activation to a slice using AVX-512.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_sigmoid_relu_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            let y = simd_fused_sigmoid_relu_avx512(x);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), y);
        });
    }

    for item in slice.iter_mut().skip(i) {
        let s = super::sigmoid::scalar_minimax_sigmoid(*item);
        *item = if s < 0.0 { 0.0 } else { s };
        if item.abs() < f32::MIN_POSITIVE {
            *item = 0.0;
        }
    }
}
