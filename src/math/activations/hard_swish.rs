// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized HardSwish activation kernels.

use crate::activation_simd_avx2;
use crate::activation_simd_avx512;
use core::arch::x86_64::*;

/// Applies HardSwish (`x * clamp(x+3, 0, 6) / 6`) to a slice using AVX2.
///
/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
pub unsafe fn hard_swish_slice_avx2(data: &mut [f32]) {
    let three = _mm256_set1_ps(3.0_f32);
    let six = _mm256_set1_ps(6.0_f32);
    let inv6 = _mm256_set1_ps(1.0_f32 / 6.0_f32);
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    let len = data.len();
    unsafe {
        activation_simd_avx2!(
            i,
            len,
            {
                let x1 = _mm256_loadu_ps(data.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
                let t1 = _mm256_add_ps(x1, three);
                let t2 = _mm256_add_ps(x2, three);
                let c1 = _mm256_min_ps(six, _mm256_max_ps(zero, t1));
                let c2 = _mm256_min_ps(six, _mm256_max_ps(zero, t2));
                _mm256_storeu_ps(
                    data.as_mut_ptr().add(i),
                    _mm256_mul_ps(_mm256_mul_ps(x1, c1), inv6),
                );
                _mm256_storeu_ps(
                    data.as_mut_ptr().add(i + 8),
                    _mm256_mul_ps(_mm256_mul_ps(x2, c2), inv6),
                );
            },
            {
                let x = _mm256_loadu_ps(data.as_ptr().add(i));
                let t = _mm256_add_ps(x, three);
                let c = _mm256_min_ps(six, _mm256_max_ps(zero, t));
                _mm256_storeu_ps(
                    data.as_mut_ptr().add(i),
                    _mm256_mul_ps(_mm256_mul_ps(x, c), inv6),
                );
            }
        );
    }
    for x in data.iter_mut().skip(i) {
        let t = *x + 3.0;
        *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
    }
}

/// Applies HardSwish (`x * clamp(x+3, 0, 6) / 6`) to a slice using AVX-512.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn hard_swish_slice_avx512(data: &mut [f32]) {
    let three = _mm512_set1_ps(3.0_f32);
    let six = _mm512_set1_ps(6.0_f32);
    let inv6 = _mm512_set1_ps(1.0_f32 / 6.0_f32);
    let zero = _mm512_setzero_ps();
    let mut i = 0;
    let len = data.len();
    unsafe {
        activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(data.as_ptr().add(i));
            let t = _mm512_add_ps(x, three);
            let c = _mm512_min_ps(six, _mm512_max_ps(zero, t));
            _mm512_storeu_ps(
                data.as_mut_ptr().add(i),
                _mm512_mul_ps(_mm512_mul_ps(x, c), inv6),
            );
        });
    }
    for x in data.iter_mut().skip(i) {
        let t = *x + 3.0;
        *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
    }
}

/// Scalar HardSwish: `x * clamp(x+3, 0, 6) / 6`.
#[inline(always)]
pub fn hard_swish(x: f32) -> f32 {
    let t = x + 3.0;
    x * t.clamp(0.0, 6.0) * (1.0 / 6.0)
}
