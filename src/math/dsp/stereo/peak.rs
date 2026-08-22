// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use core::arch::x86_64::*;

/// Computes the peak absolute value of both stereo channels via AVX2.
/// Returns `(max(|L_i|), max(|R_i|))`
///
/// # Safety
/// Slices must have the same length.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_peak_abs_stereo_avx2(left: &[f32], right: &[f32]) -> (f32, f32) {
    let len = core::cmp::min(left.len(), right.len());
    if len == 0 {
        return (0.0, 0.0);
    }

    let mut i = 0;
    let mut max_l = _mm256_setzero_ps();
    let mut max_r = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0f32);

    while i + 8 <= len {
        let vl = _mm256_loadu_ps(left.as_ptr().add(i));
        let vr = _mm256_loadu_ps(right.as_ptr().add(i));
        let abs_l = _mm256_andnot_ps(sign_mask, vl);
        let abs_r = _mm256_andnot_ps(sign_mask, vr);
        max_l = _mm256_max_ps(max_l, abs_l);
        max_r = _mm256_max_ps(max_r, abs_r);
        i += 8;
    }

    // Horizontal max for left channel
    let hi_l = _mm256_extractf128_ps(max_l, 1);
    let lo_l = _mm256_castps256_ps128(max_l);
    let m128_l = _mm_max_ps(lo_l, hi_l);
    let shuf_l = _mm_shuffle_ps(m128_l, m128_l, 0xEE);
    let m64_l = _mm_max_ps(m128_l, shuf_l);
    let shuf2_l = _mm_shuffle_ps(m64_l, m64_l, 0x55);
    let m32_l = _mm_max_ps(m64_l, shuf2_l);
    let mut peak_l = 0.0f32;
    _mm_store_ss(&mut peak_l, m32_l);

    // Horizontal max for right channel
    let hi_r = _mm256_extractf128_ps(max_r, 1);
    let lo_r = _mm256_castps256_ps128(max_r);
    let m128_r = _mm_max_ps(lo_r, hi_r);
    let shuf_r = _mm_shuffle_ps(m128_r, m128_r, 0xEE);
    let m64_r = _mm_max_ps(m128_r, shuf_r);
    let shuf2_r = _mm_shuffle_ps(m64_r, m64_r, 0x55);
    let m32_r = _mm_max_ps(m64_r, shuf2_r);
    let mut peak_r = 0.0f32;
    _mm_store_ss(&mut peak_r, m32_r);

    while i < len {
        let al = left[i].abs();
        let ar = right[i].abs();
        if al > peak_l {
            peak_l = al;
        }
        if ar > peak_r {
            peak_r = ar;
        }
        i += 1;
    }

    (peak_l, peak_r)
}

/// Computes the peak absolute value of both stereo channels via AVX-512.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_peak_abs_stereo_avx512(left: &[f32], right: &[f32]) -> (f32, f32) {
    let len = core::cmp::min(left.len(), right.len());
    if len == 0 {
        return (0.0, 0.0);
    }

    let mut i = 0;
    let mut max_l = _mm512_setzero_ps();
    let mut max_r = _mm512_setzero_ps();
    let sign_mask = _mm512_set1_ps(-0.0f32);

    while i + 16 <= len {
        let vl = _mm512_loadu_ps(left.as_ptr().add(i));
        let vr = _mm512_loadu_ps(right.as_ptr().add(i));
        let abs_l = _mm512_andnot_ps(sign_mask, vl);
        let abs_r = _mm512_andnot_ps(sign_mask, vr);
        max_l = _mm512_max_ps(max_l, abs_l);
        max_r = _mm512_max_ps(max_r, abs_r);
        i += 16;
    }

    let mut peak_l = _mm512_reduce_max_ps(max_l);
    let mut peak_r = _mm512_reduce_max_ps(max_r);

    while i < len {
        let al = left[i].abs();
        let ar = right[i].abs();
        if al > peak_l {
            peak_l = al;
        }
        if ar > peak_r {
            peak_r = ar;
        }
        i += 1;
    }

    (peak_l, peak_r)
}

/// Mono peak absolute value via AVX2.
/// Returns `max(|x_i|)`
///
/// # Safety
/// The `data` slice must be valid.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_peak_abs_mono_avx2(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    let mut i = 0;
    let mut max_v = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0f32);

    while i + 8 <= len {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        let abs_v = _mm256_andnot_ps(sign_mask, v);
        max_v = _mm256_max_ps(max_v, abs_v);
        i += 8;
    }

    let hi = _mm256_extractf128_ps(max_v, 1);
    let lo = _mm256_castps256_ps128(max_v);
    let m128 = _mm_max_ps(lo, hi);
    let shuf = _mm_shuffle_ps(m128, m128, 0xEE);
    let m64 = _mm_max_ps(m128, shuf);
    let shuf2 = _mm_shuffle_ps(m64, m64, 0x55);
    let m32 = _mm_max_ps(m64, shuf2);
    let mut peak = 0.0f32;
    _mm_store_ss(&mut peak, m32);

    while i < len {
        let a = data[i].abs();
        if a > peak {
            peak = a;
        }
        i += 1;
    }

    peak
}

/// Mono peak absolute value via AVX-512.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_peak_abs_mono_avx512(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    let mut i = 0;
    let mut max_v = _mm512_setzero_ps();
    let sign_mask = _mm512_set1_ps(-0.0f32);

    while i + 16 <= len {
        let v = _mm512_loadu_ps(data.as_ptr().add(i));
        let abs_v = _mm512_andnot_ps(sign_mask, v);
        max_v = _mm512_max_ps(max_v, abs_v);
        i += 16;
    }

    let mut peak = _mm512_reduce_max_ps(max_v);

    while i < len {
        let a = data[i].abs();
        if a > peak {
            peak = a;
        }
        i += 1;
    }

    peak
}
