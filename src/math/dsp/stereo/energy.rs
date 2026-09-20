// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use core::arch::x86_64::*;

/// Computes the energy (Mean Square) of a block via AVX2, returning (energy, has_non_finite).
/// $E = \frac{1}{N} \sum x_i^2$
///
/// The function body is safe; the `&[f32]` reference guarantees slice validity
/// and the `_mm256_*` intrinsics are valid because AVX2/FMA are always
/// available at the `x86-64-v3` baseline.
///
/// # Safety
///
/// Calling this function from code whose own codegen does not enable the
/// `avx2`/`fma` target features (via its own `#[target_feature]` attribute or
/// inlining) is an unsafe operation and requires an `unsafe` block.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub fn compute_energy_avx2(data: &[f32]) -> (f32, bool) {
    let len = data.len();
    if len == 0 {
        return (0.0, false);
    }

    let mut i = 0;
    let mut total_sum = 0.0f32;
    let mut non_finite_simd = false;

    // SAFETY: every raw-pointer access below is in-bounds of the valid `data`
    // slice (guarded by the `while i + N <= len` bounds); the AVX2/FMA
    // instructions are enabled by this function's `#[target_feature]`
    // attribute, which the `x86-64-v3` baseline guarantees at runtime.
    unsafe {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let max_val = _mm256_set1_ps(f32::MAX);
        let mut valid_mask = _mm256_set1_ps(f32::from_bits(0xFFFF_FFFF));

        while i + 16 <= len {
            let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
            sum0 = _mm256_fmadd_ps(v0, v0, sum0);
            sum1 = _mm256_fmadd_ps(v1, v1, sum1);

            let ord0 = _mm256_cmp_ps(v0, v0, _CMP_ORD_Q);
            let abs0 = _mm256_and_ps(v0, abs_mask);
            let le0 = _mm256_cmp_ps(abs0, max_val, _CMP_LE_OQ);
            let fin0 = _mm256_and_ps(ord0, le0);

            let ord1 = _mm256_cmp_ps(v1, v1, _CMP_ORD_Q);
            let abs1 = _mm256_and_ps(v1, abs_mask);
            let le1 = _mm256_cmp_ps(abs1, max_val, _CMP_LE_OQ);
            let fin1 = _mm256_and_ps(ord1, le1);

            valid_mask = _mm256_and_ps(valid_mask, _mm256_and_ps(fin0, fin1));
            i += 16;
        }

        while i + 8 <= len {
            let v = _mm256_loadu_ps(data.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(v, v, sum0);

            let ord = _mm256_cmp_ps(v, v, _CMP_ORD_Q);
            let abs = _mm256_and_ps(v, abs_mask);
            let le = _mm256_cmp_ps(abs, max_val, _CMP_LE_OQ);
            let fin = _mm256_and_ps(ord, le);

            valid_mask = _mm256_and_ps(valid_mask, fin);
            i += 8;
        }

        if _mm256_movemask_ps(valid_mask) as u32 != 0xFF {
            non_finite_simd = true;
        }

        let sum = _mm256_add_ps(sum0, sum1);
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let s128 = _mm_add_ps(lo, hi);

        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let r = _mm_add_ss(sums, shuf2);

        _mm_store_ss(&mut total_sum, r);
    }

    let mut has_non_finite = non_finite_simd;
    while i < len {
        let val = data[i];
        if !val.is_finite() {
            has_non_finite = true;
        }
        total_sum += val * val;
        i += 1;
    }

    let energy = if has_non_finite {
        0.0
    } else {
        total_sum / (len as f32)
    };
    (energy, has_non_finite)
}

/// Computes the maximum energy between two channels (Mean Square) via AVX2,
/// returning (energy, has_non_finite).
/// Fuses both passes and non-finite containment scanning into one to save memory bandwidth.
///
/// The function body is safe; the `&[f32]` references guarantee slice validity
/// and the `_mm256_*` intrinsics are valid because AVX2/FMA are always
/// available at the `x86-64-v3` baseline.
///
/// # Safety
///
/// Calling this function from code whose own codegen does not enable the
/// `avx2`/`fma` target features (via its own `#[target_feature]` attribute or
/// inlining) is an unsafe operation and requires an `unsafe` block.
#[target_feature(enable = "avx2,fma")]
pub fn compute_energy_stereo_avx2(l: &[f32], r: &[f32]) -> (f32, bool) {
    let len = core::cmp::min(l.len(), r.len());
    if len == 0 {
        return (0.0, false);
    }

    let mut i = 0;
    let mut total_sum_l = 0.0f32;
    let mut total_sum_r = 0.0f32;
    let mut non_finite_simd = false;

    // SAFETY: every raw-pointer access below is in-bounds of the valid `l`/`r`
    // slices (guarded by the `while i + N <= len` bounds); the AVX2/FMA
    // instructions are enabled by this function's `#[target_feature]`
    // attribute, which the `x86-64-v3` baseline guarantees at runtime.
    unsafe {
        let mut sum_l0 = _mm256_setzero_ps();
        let mut sum_l1 = _mm256_setzero_ps();
        let mut sum_r0 = _mm256_setzero_ps();
        let mut sum_r1 = _mm256_setzero_ps();
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let max_val = _mm256_set1_ps(f32::MAX);
        let mut valid_mask = _mm256_set1_ps(f32::from_bits(0xFFFF_FFFF));

        while i + 16 <= len {
            let vl0 = _mm256_loadu_ps(l.as_ptr().add(i));
            let vl1 = _mm256_loadu_ps(l.as_ptr().add(i + 8));
            let vr0 = _mm256_loadu_ps(r.as_ptr().add(i));
            let vr1 = _mm256_loadu_ps(r.as_ptr().add(i + 8));

            sum_l0 = _mm256_fmadd_ps(vl0, vl0, sum_l0);
            sum_l1 = _mm256_fmadd_ps(vl1, vl1, sum_l1);
            sum_r0 = _mm256_fmadd_ps(vr0, vr0, sum_r0);
            sum_r1 = _mm256_fmadd_ps(vr1, vr1, sum_r1);

            let ord_l0 = _mm256_cmp_ps(vl0, vl0, _CMP_ORD_Q);
            let abs_l0 = _mm256_and_ps(vl0, abs_mask);
            let le_l0 = _mm256_cmp_ps(abs_l0, max_val, _CMP_LE_OQ);
            let fin_l0 = _mm256_and_ps(ord_l0, le_l0);

            let ord_l1 = _mm256_cmp_ps(vl1, vl1, _CMP_ORD_Q);
            let abs_l1 = _mm256_and_ps(vl1, abs_mask);
            let le_l1 = _mm256_cmp_ps(abs_l1, max_val, _CMP_LE_OQ);
            let fin_l1 = _mm256_and_ps(ord_l1, le_l1);

            let ord_r0 = _mm256_cmp_ps(vr0, vr0, _CMP_ORD_Q);
            let abs_r0 = _mm256_and_ps(vr0, abs_mask);
            let le_r0 = _mm256_cmp_ps(abs_r0, max_val, _CMP_LE_OQ);
            let fin_r0 = _mm256_and_ps(ord_r0, le_r0);

            let ord_r1 = _mm256_cmp_ps(vr1, vr1, _CMP_ORD_Q);
            let abs_r1 = _mm256_and_ps(vr1, abs_mask);
            let le_r1 = _mm256_cmp_ps(abs_r1, max_val, _CMP_LE_OQ);
            let fin_r1 = _mm256_and_ps(ord_r1, le_r1);

            let fin_l = _mm256_and_ps(fin_l0, fin_l1);
            let fin_r = _mm256_and_ps(fin_r0, fin_r1);
            valid_mask = _mm256_and_ps(valid_mask, _mm256_and_ps(fin_l, fin_r));
            i += 16;
        }

        while i + 8 <= len {
            let vl = _mm256_loadu_ps(l.as_ptr().add(i));
            let vr = _mm256_loadu_ps(r.as_ptr().add(i));
            sum_l0 = _mm256_fmadd_ps(vl, vl, sum_l0);
            sum_r0 = _mm256_fmadd_ps(vr, vr, sum_r0);

            let ord_l = _mm256_cmp_ps(vl, vl, _CMP_ORD_Q);
            let abs_l = _mm256_and_ps(vl, abs_mask);
            let le_l = _mm256_cmp_ps(abs_l, max_val, _CMP_LE_OQ);
            let fin_l = _mm256_and_ps(ord_l, le_l);

            let ord_r = _mm256_cmp_ps(vr, vr, _CMP_ORD_Q);
            let abs_r = _mm256_and_ps(vr, abs_mask);
            let le_r = _mm256_cmp_ps(abs_r, max_val, _CMP_LE_OQ);
            let fin_r = _mm256_and_ps(ord_r, le_r);

            valid_mask = _mm256_and_ps(valid_mask, _mm256_and_ps(fin_l, fin_r));
            i += 8;
        }

        if _mm256_movemask_ps(valid_mask) as u32 != 0xFF {
            non_finite_simd = true;
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
        _mm_store_ss(&mut total_sum_r, r_r);
    }

    let mut has_non_finite = non_finite_simd;
    while i < len {
        let sl = l[i];
        let sr = r[i];
        if !sl.is_finite() || !sr.is_finite() {
            has_non_finite = true;
        }
        total_sum_l += sl * sl;
        total_sum_r += sr * sr;
        i += 1;
    }

    let energy = if has_non_finite {
        0.0
    } else {
        let energy_l = total_sum_l / (len as f32);
        let energy_r = total_sum_r / (len as f32);
        energy_l.max(energy_r)
    };
    (energy, has_non_finite)
}

// AVX-512 Kernels
// ═══════════════════════════════════════════════════════════════

/// Computes the maximum energy between two channels (Mean Square) via AVX-512,
/// returning (energy, has_non_finite).
/// Fuses both passes and non-finite containment scanning into one to save memory bandwidth.
///
/// The function body is safe (and only compiled with the opt-in `avx512`
/// feature); the `&[f32]` references guarantee slice validity, and the
/// `_mm512_*` intrinsics are only valid because AVX-512F is verified by the
/// SIMD dispatch before this path is reachable.
///
/// # Safety
///
/// Calling this function from code whose own codegen does not enable the
/// `avx512f` target feature (via its own `#[target_feature]` attribute or
/// inlining) is an unsafe operation and requires an `unsafe` block; the caller
/// must guarantee the CPU supports AVX-512F.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f")]
pub fn compute_energy_stereo_avx512(l: &[f32], r: &[f32]) -> (f32, bool) {
    let len = core::cmp::min(l.len(), r.len());
    if len == 0 {
        return (0.0, false);
    }
    let mut i = 0;
    let mut non_finite_simd = false;

    // SAFETY: every raw-pointer access below is in-bounds of the valid `l`/`r`
    // slices (guarded by the `while i + 16 <= len` bounds); AVX-512F is enabled
    // by this function's `#[target_feature]` and verified at dispatch time.
    let (mut sum_l, mut sum_r) = unsafe {
        let mut sum_lv = _mm512_setzero_ps();
        let mut sum_rv = _mm512_setzero_ps();
        let abs_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let max_val = _mm512_set1_ps(f32::MAX);
        let mut valid_mask: u16 = 0xFFFF;

        while i + 16 <= len {
            let lv = _mm512_loadu_ps(l.as_ptr().add(i));
            let rv = _mm512_loadu_ps(r.as_ptr().add(i));
            sum_lv = _mm512_fmadd_ps(lv, lv, sum_lv);
            sum_rv = _mm512_fmadd_ps(rv, rv, sum_rv);

            let ord_l = _mm512_cmp_ps_mask(lv, lv, _CMP_ORD_Q);
            let abs_l = _mm512_and_ps(lv, abs_mask);
            let le_l = _mm512_cmp_ps_mask(abs_l, max_val, _CMP_LE_OQ);

            let ord_r = _mm512_cmp_ps_mask(rv, rv, _CMP_ORD_Q);
            let abs_r = _mm512_and_ps(rv, abs_mask);
            let le_r = _mm512_cmp_ps_mask(abs_r, max_val, _CMP_LE_OQ);

            valid_mask &= ord_l & le_l & ord_r & le_r;
            i += 16;
        }

        if valid_mask != 0xFFFF {
            non_finite_simd = true;
        }

        (
            crate::math::common::utility::hsum_avx512(sum_lv),
            crate::math::common::utility::hsum_avx512(sum_rv),
        )
    };

    let mut has_non_finite = non_finite_simd;
    while i < len {
        let sl = l[i];
        let sr = r[i];
        if !sl.is_finite() || !sr.is_finite() {
            has_non_finite = true;
        }
        sum_l += sl * sl;
        sum_r += sr * sr;
        i += 1;
    }

    let energy = if has_non_finite {
        0.0
    } else {
        let energy_l = sum_l / (len as f32);
        let energy_r = sum_r / (len as f32);
        energy_l.max(energy_r)
    };
    (energy, has_non_finite)
}

/// Computes the energy (Mean Square) of a block via AVX-512, returning (energy, has_non_finite).
///
/// The function body is safe (and only compiled with the opt-in `avx512`
/// feature); the `&[f32]` reference guarantees slice validity, and the
/// `_mm512_*` intrinsics are only valid because AVX-512F is verified by the
/// SIMD dispatch before this path is reachable.
///
/// # Safety
///
/// Calling this function from code whose own codegen does not enable the
/// `avx512f` target feature (via its own `#[target_feature]` attribute or
/// inlining) is an unsafe operation and requires an `unsafe` block; the caller
/// must guarantee the CPU supports AVX-512F.
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f")]
pub fn compute_energy_avx512(data: &[f32]) -> (f32, bool) {
    let len = data.len();
    if len == 0 {
        return (0.0, false);
    }
    let mut i = 0;
    let mut non_finite_simd = false;

    // SAFETY: every raw-pointer access below is in-bounds of the valid `data`
    // slice (guarded by the `while i + 16 <= len` bounds); AVX-512F is enabled
    // by this function's `#[target_feature]` and verified at dispatch time.
    let mut total_sum = unsafe {
        let mut sum_v = _mm512_setzero_ps();
        let abs_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let max_val = _mm512_set1_ps(f32::MAX);
        let mut valid_mask: u16 = 0xFFFF;

        while i + 16 <= len {
            let v = _mm512_loadu_ps(data.as_ptr().add(i));
            sum_v = _mm512_fmadd_ps(v, v, sum_v);

            let ord = _mm512_cmp_ps_mask(v, v, _CMP_ORD_Q);
            let abs = _mm512_and_ps(v, abs_mask);
            let le = _mm512_cmp_ps_mask(abs, max_val, _CMP_LE_OQ);

            valid_mask &= ord & le;
            i += 16;
        }

        if valid_mask != 0xFFFF {
            non_finite_simd = true;
        }

        crate::math::common::utility::hsum_avx512(sum_v)
    };

    let mut has_non_finite = non_finite_simd;
    while i < len {
        let val = data[i];
        if !val.is_finite() {
            has_non_finite = true;
        }
        total_sum += val * val;
        i += 1;
    }

    let energy = if has_non_finite {
        0.0
    } else {
        total_sum / (len as f32)
    };
    (energy, has_non_finite)
}
