// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Specialized `gemv_overwrite_avx2_*` kernels for fixed (in_len, out_len) dimensions.
//!
//! Coverage: 1×4, 4×4, 4×6, 8×4, 8×6, 8×8.
//! Uses safe partial-YMM load/store via `f16_avx2_specialized`.

use super::f16_avx2_specialized::{load_partial_ymm, store_partial_ymm};
use core::arch::x86_64::*;

// ═════════════════════════════════════════════════════════════════════════════════
// gemv_overwrite specialized kernels
// ═════════════════════════════════════════════════════════════════════════════════

/// Specialized overwrite GEMV for: 1 input × 4 outputs.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_overwrite_avx2_1x4(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    // SAFETY: dispatched only for (in_len, out_len) == (1, 4) by
    // `gemv_overwrite_avx2`, so `in_frame` has >= 1 element, `weights` >= 4,
    // `out_frame` >= 4, and `bias` >= 4 when `do_bias`. AVX2+FMA are guaranteed
    // by the `#[target_feature]` attribute on this `unsafe fn`.
    unsafe {
        let v_in = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let vw = load_partial_ymm(weights, 4);
        let mut acc = _mm256_mul_ps(v_in, vw);
        if do_bias {
            let vb = load_partial_ymm(bias, 4);
            acc = _mm256_add_ps(acc, vb);
        }
        store_partial_ymm(acc, out_frame, 4);
    }
}

/// Specialized overwrite GEMV for: 4 inputs × 4 outputs.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_overwrite_avx2_4x4(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    // SAFETY: dispatched only for (in_len, out_len) == (4, 4), so `in_frame` has
    // >= 4 elements, `weights` >= 16, `out_frame` >= 4, and `bias` >= 4 when
    // `do_bias`. AVX2+FMA guaranteed by `#[target_feature]`.
    unsafe {
        let mut acc = if do_bias {
            load_partial_ymm(bias, 4)
        } else {
            _mm256_setzero_ps()
        };

        let v0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let w0 = load_partial_ymm(&weights[0..], 4);
        acc = _mm256_fmadd_ps(v0, w0, acc);

        let v1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let w1 = load_partial_ymm(&weights[4..], 4);
        acc = _mm256_fmadd_ps(v1, w1, acc);

        let v2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let w2 = load_partial_ymm(&weights[8..], 4);
        acc = _mm256_fmadd_ps(v2, w2, acc);

        let v3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let w3 = load_partial_ymm(&weights[12..], 4);
        acc = _mm256_fmadd_ps(v3, w3, acc);

        store_partial_ymm(acc, out_frame, 4);
    }
}

/// Specialized overwrite GEMV for: 4 inputs × 6 outputs.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_overwrite_avx2_4x6(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    // SAFETY: dispatched only for (in_len, out_len) == (4, 6), so `in_frame` has
    // >= 4 elements, `weights` >= 24, `bias` >= 6 (read when `do_bias`),
    // `out_frame` >= 6. AVX2+FMA guaranteed by `#[target_feature]`.
    unsafe {
        let mut acc = if do_bias {
            load_partial_ymm(bias, 4)
        } else {
            _mm256_setzero_ps()
        };

        let v0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let w0 = load_partial_ymm(&weights[0..], 4);
        acc = _mm256_fmadd_ps(v0, w0, acc);

        let v1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let w1 = load_partial_ymm(&weights[6..], 4);
        acc = _mm256_fmadd_ps(v1, w1, acc);

        let v2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let w2 = load_partial_ymm(&weights[12..], 4);
        acc = _mm256_fmadd_ps(v2, w2, acc);

        let v3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let w3 = load_partial_ymm(&weights[18..], 4);
        acc = _mm256_fmadd_ps(v3, w3, acc);

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);

        for oc in 4..6 {
            let w = *weights.get_unchecked(oc);
            let mut sum = *in_frame.get_unchecked(0) * w;
            sum += *in_frame.get_unchecked(1) * *weights.get_unchecked(6 + oc);
            sum += *in_frame.get_unchecked(2) * *weights.get_unchecked(12 + oc);
            sum += *in_frame.get_unchecked(3) * *weights.get_unchecked(18 + oc);
            if do_bias {
                sum += bias[oc];
            }
            tmp[oc] = sum;
        }

        for (i, &val) in tmp.iter().enumerate().take(6) {
            *out_frame.get_unchecked_mut(i) = val;
        }
    }
}

/// Specialized overwrite GEMV for: 8 inputs × 4 outputs.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_overwrite_avx2_8x4(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    // SAFETY: dispatched only for (in_len, out_len) == (8, 4), so `in_frame` has
    // >= 8 elements, `weights` >= 32, `out_frame` >= 4, and `bias` >= 4 when
    // `do_bias`. AVX2+FMA guaranteed by `#[target_feature]`.
    unsafe {
        let mut acc = if do_bias {
            load_partial_ymm(bias, 4)
        } else {
            _mm256_setzero_ps()
        };

        let v0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let w0 = load_partial_ymm(&weights[0..], 4);
        acc = _mm256_fmadd_ps(v0, w0, acc);

        let v1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let w1 = load_partial_ymm(&weights[4..], 4);
        acc = _mm256_fmadd_ps(v1, w1, acc);

        let v2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let w2 = load_partial_ymm(&weights[8..], 4);
        acc = _mm256_fmadd_ps(v2, w2, acc);

        let v3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let w3 = load_partial_ymm(&weights[12..], 4);
        acc = _mm256_fmadd_ps(v3, w3, acc);

        let v4 = _mm256_set1_ps(*in_frame.get_unchecked(4));
        let w4 = load_partial_ymm(&weights[16..], 4);
        acc = _mm256_fmadd_ps(v4, w4, acc);

        let v5 = _mm256_set1_ps(*in_frame.get_unchecked(5));
        let w5 = load_partial_ymm(&weights[20..], 4);
        acc = _mm256_fmadd_ps(v5, w5, acc);

        let v6 = _mm256_set1_ps(*in_frame.get_unchecked(6));
        let w6 = load_partial_ymm(&weights[24..], 4);
        acc = _mm256_fmadd_ps(v6, w6, acc);

        let v7 = _mm256_set1_ps(*in_frame.get_unchecked(7));
        let w7 = load_partial_ymm(&weights[28..], 4);
        acc = _mm256_fmadd_ps(v7, w7, acc);

        store_partial_ymm(acc, out_frame, 4);
    }
}

/// Specialized overwrite GEMV for: 8 inputs × 6 outputs.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_overwrite_avx2_8x6(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    // SAFETY: dispatched only for (in_len, out_len) == (8, 6), so `in_frame` has
    // >= 8 elements, `weights` >= 48, `bias` >= 6 (read when `do_bias`),
    // `out_frame` >= 6. AVX2+FMA guaranteed by `#[target_feature]`.
    unsafe {
        let out_len = 6usize;
        let w_ptr = weights.as_ptr();

        let mut acc_simd = if do_bias {
            load_partial_ymm(bias, 4)
        } else {
            _mm256_setzero_ps()
        };

        let mut sum4 = if do_bias { bias[4] } else { 0.0 };
        let mut sum5 = if do_bias { bias[5] } else { 0.0 };

        for in_c in 0..8 {
            let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
            let w_row = w_ptr.add(in_c * out_len);
            let w_simd = load_partial_ymm(core::slice::from_raw_parts(w_row, 4), 4);
            acc_simd = _mm256_fmadd_ps(vs, w_simd, acc_simd);
            sum4 += *in_frame.get_unchecked(in_c) * *weights.get_unchecked(in_c * out_len + 4);
            sum5 += *in_frame.get_unchecked(in_c) * *weights.get_unchecked(in_c * out_len + 5);
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc_simd);
        tmp[4] = sum4;
        tmp[5] = sum5;

        for (i, &val) in tmp.iter().enumerate().take(6) {
            *out_frame.get_unchecked_mut(i) = val;
        }
    }
}

/// Specialized overwrite GEMV for: 8 inputs × 8 outputs.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_overwrite_avx2_8x8(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    // SAFETY: dispatched only for (in_len, out_len) == (8, 8), so `in_frame` has
    // >= 8 elements, `weights` >= 64, `bias` >= 8 (read when `do_bias`),
    // `out_frame` >= 8. AVX2+FMA guaranteed by `#[target_feature]`.
    unsafe {
        let w_ptr = weights.as_ptr();

        let mut acc0 = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut acc4 = _mm256_setzero_ps();
        let mut acc5 = _mm256_setzero_ps();
        let mut acc6 = _mm256_setzero_ps();
        let mut acc7 = _mm256_setzero_ps();

        let vs0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let vs1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let vs2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let vs3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let vs4 = _mm256_set1_ps(*in_frame.get_unchecked(4));
        let vs5 = _mm256_set1_ps(*in_frame.get_unchecked(5));
        let vs6 = _mm256_set1_ps(*in_frame.get_unchecked(6));
        let vs7 = _mm256_set1_ps(*in_frame.get_unchecked(7));

        let w0 = _mm256_loadu_ps(w_ptr);
        acc0 = _mm256_fmadd_ps(vs0, w0, acc0);
        let w1 = _mm256_loadu_ps(w_ptr.add(8));
        acc1 = _mm256_fmadd_ps(vs1, w1, acc1);
        let w2 = _mm256_loadu_ps(w_ptr.add(16));
        acc2 = _mm256_fmadd_ps(vs2, w2, acc2);
        let w3 = _mm256_loadu_ps(w_ptr.add(24));
        acc3 = _mm256_fmadd_ps(vs3, w3, acc3);
        let w4 = _mm256_loadu_ps(w_ptr.add(32));
        acc4 = _mm256_fmadd_ps(vs4, w4, acc4);
        let w5 = _mm256_loadu_ps(w_ptr.add(40));
        acc5 = _mm256_fmadd_ps(vs5, w5, acc5);
        let w6 = _mm256_loadu_ps(w_ptr.add(48));
        acc6 = _mm256_fmadd_ps(vs6, w6, acc6);
        let w7 = _mm256_loadu_ps(w_ptr.add(56));
        acc7 = _mm256_fmadd_ps(vs7, w7, acc7);

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc4 = _mm256_add_ps(acc4, acc5);
        acc6 = _mm256_add_ps(acc6, acc7);
        acc0 = _mm256_add_ps(acc0, acc2);
        acc4 = _mm256_add_ps(acc4, acc6);
        acc0 = _mm256_add_ps(acc0, acc4);

        _mm256_storeu_ps(out_frame.as_mut_ptr(), acc0);
    }
}
