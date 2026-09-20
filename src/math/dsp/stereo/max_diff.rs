// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use core::arch::x86_64::*;

/// Computes the maximum absolute difference between two blocks via AVX2,
/// returning (max_diff, has_non_finite).
/// $\max(|L_i - R_i|)$
///
/// # Safety
/// The slices `a` and `b` must have the same length.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_max_diff_avx2(a: &[f32], b: &[f32]) -> (f32, bool) {
    let len = core::cmp::min(a.len(), b.len());
    if len == 0 {
        return (0.0, false);
    }

    let mut i = 0;
    let mut max_diff = 0.0f32;
    let mut non_finite_simd = false;

    // SAFETY: every raw-pointer access below is in-bounds of valid slices `a` and `b`
    // (guarded by `len = min(a.len(), b.len())` and the loop bound `i + 8 <= len`).
    // `_mm256_loadu_ps` performs unaligned 32-byte loads, so 32-byte alignment is not required.
    // `a` and `b` are immutable borrows ensuring no aliasing violations.
    // AVX2 feature support is guaranteed by `#[target_feature(enable = "avx2")]` and caller contract.
    // `_mm_store_ss` writes into valid local variable address `&mut max_diff`.
    unsafe {
        let mut max_v = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0f32);
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let max_val = _mm256_set1_ps(f32::MAX);
        let mut valid_mask = _mm256_set1_ps(f32::from_bits(0xFFFF_FFFF));

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(va, vb);
            let abs_diff = _mm256_andnot_ps(sign_mask, diff);
            max_v = _mm256_max_ps(max_v, abs_diff);

            let ord_a = _mm256_cmp_ps(va, va, _CMP_ORD_Q);
            let abs_a = _mm256_and_ps(va, abs_mask);
            let le_a = _mm256_cmp_ps(abs_a, max_val, _CMP_LE_OQ);
            let fin_a = _mm256_and_ps(ord_a, le_a);

            let ord_b = _mm256_cmp_ps(vb, vb, _CMP_ORD_Q);
            let abs_b = _mm256_and_ps(vb, abs_mask);
            let le_b = _mm256_cmp_ps(abs_b, max_val, _CMP_LE_OQ);
            let fin_b = _mm256_and_ps(ord_b, le_b);

            valid_mask = _mm256_and_ps(valid_mask, _mm256_and_ps(fin_a, fin_b));
            i += 8;
        }

        if _mm256_movemask_ps(valid_mask) as u32 != 0xFF {
            non_finite_simd = true;
        }

        let hi = _mm256_extractf128_ps(max_v, 1);
        let lo = _mm256_castps256_ps128(max_v);
        let m128 = _mm_max_ps(lo, hi);

        let shuf = _mm_shuffle_ps(m128, m128, 0xEE);
        let m64 = _mm_max_ps(m128, shuf);
        let shuf2 = _mm_shuffle_ps(m64, m64, 0x55);
        let m32 = _mm_max_ps(m64, shuf2);

        _mm_store_ss(&mut max_diff, m32);
    }

    let mut has_non_finite = non_finite_simd;
    while i < len {
        let va = a[i];
        let vb = b[i];
        if !va.is_finite() || !vb.is_finite() {
            has_non_finite = true;
        }
        let d = (va - vb).abs();
        if d > max_diff {
            max_diff = d;
        }
        i += 1;
    }

    let res = if has_non_finite { 0.0 } else { max_diff };
    (res, has_non_finite)
}

/// Computes the maximum absolute difference between two blocks via AVX-512,
/// returning (max_diff, has_non_finite).
///
/// # Safety
/// The slices `a` and `b` must have the same length.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_max_diff_avx512(a: &[f32], b: &[f32]) -> (f32, bool) {
    let len = core::cmp::min(a.len(), b.len());
    if len == 0 {
        return (0.0, false);
    }
    let mut i = 0;
    let mut max_diff;
    let mut non_finite_simd = false;

    // SAFETY: every raw-pointer access below is in-bounds of valid slices `a` and `b`
    // (guarded by `len = min(a.len(), b.len())` and the loop bound `i + 16 <= len`).
    // `_mm512_loadu_ps` performs unaligned 64-byte loads, so 64-byte alignment is not required.
    // `a` and `b` are immutable borrows ensuring no aliasing violations.
    // AVX-512F feature support is guaranteed by `#[target_feature(enable = "avx512f")]` and caller contract.
    unsafe {
        let mut max_v = _mm512_setzero_ps();
        let sign_mask = _mm512_set1_ps(-0.0f32);
        let abs_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let max_val = _mm512_set1_ps(f32::MAX);
        let mut valid_mask: u16 = 0xFFFF;

        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let diff = _mm512_sub_ps(va, vb);
            let abs_diff = _mm512_andnot_ps(sign_mask, diff);
            max_v = _mm512_max_ps(max_v, abs_diff);

            let ord_a = _mm512_cmp_ps_mask(va, va, _CMP_ORD_Q);
            let abs_a = _mm512_and_ps(va, abs_mask);
            let le_a = _mm512_cmp_ps_mask(abs_a, max_val, _CMP_LE_OQ);

            let ord_b = _mm512_cmp_ps_mask(vb, vb, _CMP_ORD_Q);
            let abs_b = _mm512_and_ps(vb, abs_mask);
            let le_b = _mm512_cmp_ps_mask(abs_b, max_val, _CMP_LE_OQ);

            valid_mask &= ord_a & le_a & ord_b & le_b;
            i += 16;
        }

        if valid_mask != 0xFFFF {
            non_finite_simd = true;
        }

        max_diff = _mm512_reduce_max_ps(max_v);
    }

    let mut has_non_finite = non_finite_simd;
    while i < len {
        let va = a[i];
        let vb = b[i];
        if !va.is_finite() || !vb.is_finite() {
            has_non_finite = true;
        }
        let d = (va - vb).abs();
        if d > max_diff {
            max_diff = d;
        }
        i += 1;
    }

    let res = if has_non_finite { 0.0 } else { max_diff };
    (res, has_non_finite)
}
