// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized HardTanh activation kernels.

use crate::activation_simd_avx2;
#[cfg(feature = "avx512")]
use crate::activation_simd_avx512;
use core::arch::x86_64::*;

/// Applies HardTanh (`clamp(x, -1.0, 1.0)`) to a slice using AVX2.
///
/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
pub unsafe fn hard_tanh_slice_avx2(data: &mut [f32]) {
    let neg_one = _mm256_set1_ps(-1.0_f32);
    let pos_one = _mm256_set1_ps(1.0_f32);
    let mut i = 0;
    let len = data.len();
    unsafe {
        activation_simd_avx2!(
            i,
            len,
            {
                let x1 = _mm256_loadu_ps(data.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
                _mm256_storeu_ps(
                    data.as_mut_ptr().add(i),
                    _mm256_min_ps(pos_one, _mm256_max_ps(neg_one, x1)),
                );
                _mm256_storeu_ps(
                    data.as_mut_ptr().add(i + 8),
                    _mm256_min_ps(pos_one, _mm256_max_ps(neg_one, x2)),
                );
            },
            {
                let x = _mm256_loadu_ps(data.as_ptr().add(i));
                _mm256_storeu_ps(
                    data.as_mut_ptr().add(i),
                    _mm256_min_ps(pos_one, _mm256_max_ps(neg_one, x)),
                );
            }
        );
    }
    for x in data.iter_mut().skip(i) {
        *x = x.clamp(-1.0, 1.0);
    }
}

/// Applies HardTanh (`clamp(x, -1.0, 1.0)`) to a slice using AVX-512.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn hard_tanh_slice_avx512(data: &mut [f32]) {
    let neg_one = _mm512_set1_ps(-1.0_f32);
    let pos_one = _mm512_set1_ps(1.0_f32);
    let mut i = 0;
    let len = data.len();
    unsafe {
        activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(data.as_ptr().add(i));
            _mm512_storeu_ps(
                data.as_mut_ptr().add(i),
                _mm512_min_ps(pos_one, _mm512_max_ps(neg_one, x)),
            );
        });
    }
    for x in data.iter_mut().skip(i) {
        *x = x.clamp(-1.0, 1.0);
    }
}

/// Scalar HardTanh: `clamp(x, -1.0, 1.0)`.
#[inline(always)]
pub fn hard_tanh(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}
