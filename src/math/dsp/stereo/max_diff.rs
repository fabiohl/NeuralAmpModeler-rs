// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use core::arch::x86_64::*;

/// Computes the maximum absolute difference between two blocks via AVX2.
/// $\max(|L_i - R_i|)$
///
/// # Safety
/// The slices `a` and `b` must have the same length.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_max_diff_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    if len == 0 {
        return 0.0;
    }

    let mut i = 0;
    let mut max_v = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0f32);

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);
        max_v = _mm256_max_ps(max_v, abs_diff);
        i += 8;
    }

    let hi = _mm256_extractf128_ps(max_v, 1);
    let lo = _mm256_castps256_ps128(max_v);
    let m128 = _mm_max_ps(lo, hi);

    let shuf = _mm_shuffle_ps(m128, m128, 0xEE);
    let m64 = _mm_max_ps(m128, shuf);
    let shuf2 = _mm_shuffle_ps(m64, m64, 0x55);
    let m32 = _mm_max_ps(m64, shuf2);

    let mut max_diff = 0.0f32;
    _mm_store_ss(&mut max_diff, m32);

    while i < len {
        let d = (a[i] - b[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
        i += 1;
    }

    max_diff
}

/// Computes the maximum absolute difference between two blocks via AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_max_diff_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    let mut max_v = _mm512_setzero_ps();
    let sign_mask = _mm512_set1_ps(-0.0f32);

    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        let abs_diff = _mm512_andnot_ps(sign_mask, diff);
        max_v = _mm512_max_ps(max_v, abs_diff);
        i += 16;
    }

    let mut max_diff = _mm512_reduce_max_ps(max_v);

    while i < len {
        let d = (a[i] - b[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
        i += 1;
    }

    max_diff
}
