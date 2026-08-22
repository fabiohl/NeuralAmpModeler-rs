// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use core::arch::x86_64::*;

/// Computes the energy (Mean Square) of a block via AVX2.
/// $E = \frac{1}{N} \sum x_i^2$
///
/// # Safety
/// The `data` slice must be valid.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_energy_avx2(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    let mut i = 0;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();

    while i + 16 <= len {
        let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
        sum0 = _mm256_fmadd_ps(v0, v0, sum0);
        sum1 = _mm256_fmadd_ps(v1, v1, sum1);
        i += 16;
    }

    while i + 8 <= len {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(v, v, sum0);
        i += 8;
    }

    let sum = _mm256_add_ps(sum0, sum1);
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let s128 = _mm_add_ps(lo, hi);

    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let r = _mm_add_ss(sums, shuf2);

    let mut total_sum = 0.0f32;
    _mm_store_ss(&mut total_sum, r);

    while i < len {
        total_sum += data[i] * data[i];
        i += 1;
    }

    total_sum / (len as f32)
}

/// Computes the maximum energy between two channels (Mean Square) via AVX2.
/// Fuses both passes into one to save memory bandwidth.
#[target_feature(enable = "avx2")]
pub unsafe fn compute_energy_stereo_avx2(l: &[f32], r: &[f32]) -> f32 {
    let len = core::cmp::min(l.len(), r.len());
    if len == 0 {
        return 0.0;
    }

    let mut i = 0;
    let mut sum_l0 = _mm256_setzero_ps();
    let mut sum_l1 = _mm256_setzero_ps();
    let mut sum_r0 = _mm256_setzero_ps();
    let mut sum_r1 = _mm256_setzero_ps();

    while i + 16 <= len {
        let vl0 = _mm256_loadu_ps(l.as_ptr().add(i));
        let vl1 = _mm256_loadu_ps(l.as_ptr().add(i + 8));
        let vr0 = _mm256_loadu_ps(r.as_ptr().add(i));
        let vr1 = _mm256_loadu_ps(r.as_ptr().add(i + 8));

        sum_l0 = _mm256_fmadd_ps(vl0, vl0, sum_l0);
        sum_l1 = _mm256_fmadd_ps(vl1, vl1, sum_l1);
        sum_r0 = _mm256_fmadd_ps(vr0, vr0, sum_r0);
        sum_r1 = _mm256_fmadd_ps(vr1, vr1, sum_r1);
        i += 16;
    }

    while i + 8 <= len {
        let vl = _mm256_loadu_ps(l.as_ptr().add(i));
        let vr = _mm256_loadu_ps(r.as_ptr().add(i));
        sum_l0 = _mm256_fmadd_ps(vl, vl, sum_l0);
        sum_r0 = _mm256_fmadd_ps(vr, vr, sum_r0);
        i += 8;
    }

    // Horizontal sum for L
    let sum_l = _mm256_add_ps(sum_l0, sum_l1);
    let hi_l = _mm256_extractf128_ps(sum_l, 1);
    let lo_l = _mm256_castps256_ps128(sum_l);
    let s128_l = _mm_add_ps(lo_l, hi_l);
    let shuf_l = _mm_movehdup_ps(s128_l);
    let sums_l = _mm_add_ps(s128_l, shuf_l);
    let shuf2_l = _mm_movehl_ps(sums_l, sums_l);
    let r_l = _mm_add_ss(sums_l, shuf2_l);
    let mut total_sum_l = 0.0f32;
    _mm_store_ss(&mut total_sum_l, r_l);

    // Horizontal sum for R
    let sum_r = _mm256_add_ps(sum_r0, sum_r1);
    let hi_r = _mm256_extractf128_ps(sum_r, 1);
    let lo_r = _mm256_castps256_ps128(sum_r);
    let s128_r = _mm_add_ps(lo_r, hi_r);
    let shuf_r = _mm_movehdup_ps(s128_r);
    let sums_r = _mm_add_ps(s128_r, shuf_r);
    let shuf2_r = _mm_movehl_ps(sums_r, sums_r);
    let r_r = _mm_add_ss(sums_r, shuf2_r);
    let mut total_sum_r = 0.0f32;
    _mm_store_ss(&mut total_sum_r, r_r);

    while i < len {
        total_sum_l += l[i] * l[i];
        total_sum_r += r[i] * r[i];
        i += 1;
    }

    let energy_l = total_sum_l / (len as f32);
    let energy_r = total_sum_r / (len as f32);
    energy_l.max(energy_r)
}

// AVX-512 Kernels
// ═══════════════════════════════════════════════════════════════

/// Computes the maximum energy between two channels (Mean Square) via AVX-512.
/// Fuses both passes into one to save memory bandwidth.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_energy_stereo_avx512(l: &[f32], r: &[f32]) -> f32 {
    let len = core::cmp::min(l.len(), r.len());
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    let mut sum_lv = _mm512_setzero_ps();
    let mut sum_rv = _mm512_setzero_ps();

    while i + 16 <= len {
        let lv = _mm512_loadu_ps(l.as_ptr().add(i));
        let rv = _mm512_loadu_ps(r.as_ptr().add(i));
        sum_lv = _mm512_fmadd_ps(lv, lv, sum_lv);
        sum_rv = _mm512_fmadd_ps(rv, rv, sum_rv);
        i += 16;
    }

    let mut sum_l = crate::math::common::utility::hsum_avx512(sum_lv);
    let mut sum_r = crate::math::common::utility::hsum_avx512(sum_rv);

    while i < len {
        sum_l += l[i] * l[i];
        sum_r += r[i] * r[i];
        i += 1;
    }

    let energy_l = sum_l / (len as f32);
    let energy_r = sum_r / (len as f32);
    energy_l.max(energy_r)
}

/// Computes the energy (Mean Square) of a block via AVX-512.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_energy_avx512(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();

    while i + 16 <= len {
        let v = _mm512_loadu_ps(data.as_ptr().add(i));
        sum_v = _mm512_fmadd_ps(v, v, sum_v);
        i += 16;
    }

    let mut total_sum = crate::math::common::utility::hsum_avx512(sum_v);

    while i < len {
        total_sum += data[i] * data[i];
        i += 1;
    }

    total_sum / (len as f32)
}
