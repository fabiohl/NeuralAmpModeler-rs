// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::gemv_kernel;
use core::arch::x86_64::*;

// ── AVX-512 Small (CH=16 specialized) ──────────────────────────────────────

/// GEMV kernel AVX-512 specialized for Standard WaveNet (CH=16).
///
/// Uses 8 independent ZMM accumulators (8×16 = 128 lanes) with an inner loop
/// step of 8, breaking the FMA dependency chain and saturating the AVX-512
/// execution ports.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_avx512_small(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let in_len = in_frame.len();

    let mut acc0 = if do_bias {
        _mm512_loadu_ps(bias.as_ptr())
    } else {
        _mm512_setzero_ps()
    };
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();

    let mut in_c = 0;
    while in_c + 8 <= in_len {
        _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().wrapping_add(in_c + 64) as *const i8);

        let v_in0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        let v_in1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
        let v_in2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
        let v_in3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));
        let v_in4 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 4));
        let v_in5 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 5));
        let v_in6 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 6));
        let v_in7 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 7));

        let w_ptr = weights.as_ptr().add(in_c * 16);

        acc0 = _mm512_fmadd_ps(v_in0, _mm512_loadu_ps(w_ptr), acc0);
        acc1 = _mm512_fmadd_ps(v_in1, _mm512_loadu_ps(w_ptr.add(16)), acc1);
        acc2 = _mm512_fmadd_ps(v_in2, _mm512_loadu_ps(w_ptr.add(32)), acc2);
        acc3 = _mm512_fmadd_ps(v_in3, _mm512_loadu_ps(w_ptr.add(48)), acc3);
        acc4 = _mm512_fmadd_ps(v_in4, _mm512_loadu_ps(w_ptr.add(64)), acc4);
        acc5 = _mm512_fmadd_ps(v_in5, _mm512_loadu_ps(w_ptr.add(80)), acc5);
        acc6 = _mm512_fmadd_ps(v_in6, _mm512_loadu_ps(w_ptr.add(96)), acc6);
        acc7 = _mm512_fmadd_ps(v_in7, _mm512_loadu_ps(w_ptr.add(112)), acc7);
        in_c += 8;
    }

    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc4 = _mm512_add_ps(acc4, acc5);
    acc6 = _mm512_add_ps(acc6, acc7);
    acc0 = _mm512_add_ps(acc0, acc2);
    acc4 = _mm512_add_ps(acc4, acc6);
    acc0 = _mm512_add_ps(acc0, acc4);

    while in_c < in_len {
        let v_in = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        acc0 = _mm512_fmadd_ps(v_in, _mm512_loadu_ps(weights.as_ptr().add(in_c * 16)), acc0);
        in_c += 1;
    }

    _mm512_storeu_ps(out_frame.as_mut_ptr(), acc0);
}

/// Fused-Add-GEMV kernel AVX-512 specialized for Standard WaveNet (CH=16).
///
/// 8 independent ZMM accumulators with step 8 in the inner loop.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemv_avx512_small(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        gemv_kernel!(
            8,
            true,
            0,
            16,
            in_frame,
            weights,
            bias,
            out_frame,
            do_bias,
            _mm512_setzero_ps,
            |oc| _mm512_loadu_ps(out_frame.as_ptr().add(oc)),
            |oc| _mm512_loadu_ps(bias.as_ptr().add(oc)),
            _mm512_add_ps,
            |ptr| _mm512_loadu_ps(ptr),
            _mm512_fmadd_ps,
            |oc, val| _mm512_storeu_ps(out_frame.as_mut_ptr().add(oc), val)
        );
    }
}

// ── AVX-512 General ────────────────────────────────────────────────────────────

/// Performs the linear projection Y = Bias + W * Z (GEMV) replacing the contents of out_frame via AVX-512.
///
/// Uses 8 independent ZMM accumulators (8×16 = 128 lanes) with an inner loop
/// step of 8 to break the FMA dependency chain.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_avx512(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    if out_len == 16 {
        gemv_overwrite_avx512_small(in_frame, weights, bias, out_frame, do_bias);
        return;
    }

    unsafe {
        let mut out_c = 0;
        while out_c + 16 <= out_len {
            gemv_kernel!(
                8,
                false,
                out_c,
                out_len,
                in_frame,
                weights,
                bias,
                out_frame,
                do_bias,
                _mm512_setzero_ps,
                |oc| _mm512_loadu_ps(out_frame.as_ptr().add(oc)),
                |oc| _mm512_loadu_ps(bias.as_ptr().add(oc)),
                _mm512_add_ps,
                |ptr| _mm512_loadu_ps(ptr),
                _mm512_fmadd_ps,
                |oc, val| _mm512_storeu_ps(out_frame.as_mut_ptr().add(oc), val)
            );
            out_c += 16;
        }

        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            for in_c in 0..in_len {
                let w = *weights.get_unchecked(in_c * out_len + out_c);
                sum += *in_frame.get_unchecked(in_c) * w;
            }
            *out_frame.get_unchecked_mut(out_c) = sum;
            out_c += 1;
        }
    }
}

/// Batch version of gemv_overwrite via AVX-512.
///
/// # Safety
/// - `num_frames` must be > 0.
/// - `in_frames.len()` must be a multiple of `num_frames`.
/// - `out_frames.len()` must be a multiple of `num_frames`.
/// - `weights.len()` must be >= `in_len * out_len` where `in_len = in_frames.len() / num_frames`
///   and `out_len = out_frames.len() / num_frames`.
/// - If `do_bias` is true, `bias.len()` must be >= `out_len`.
///
/// All slices must be valid and accessible for reading/writing.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_batch_avx512(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    debug_assert_eq!(in_frames.len() % num_frames, 0);
    debug_assert_eq!(out_frames.len() % num_frames, 0);
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;
    debug_assert!(weights.len() >= in_len * out_len);
    if do_bias {
        debug_assert!(bias.len() >= out_len);
    }
    for i in 0..num_frames {
        let in_slice = &in_frames[i * in_len..(i + 1) * in_len];
        let out_slice = &mut out_frames[i * out_len..(i + 1) * out_len];
        gemv_overwrite_avx512(in_slice, weights, bias, out_slice, do_bias);
    }
}

/// Performs the fused operation Y = X_res + Bias + W * Z (Broadcast GEMV) via AVX-512.
///
/// 8 independent ZMM accumulators with step 8 in the inner loop.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemv_avx512(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    if out_len == 16 {
        fused_add_gemv_avx512_small(in_frame, weights, bias, out_frame, do_bias);
        return;
    }

    unsafe {
        let mut out_c = 0;
        while out_c + 16 <= out_len {
            gemv_kernel!(
                8,
                true,
                out_c,
                out_len,
                in_frame,
                weights,
                bias,
                out_frame,
                do_bias,
                _mm512_setzero_ps,
                |oc| _mm512_loadu_ps(out_frame.as_ptr().add(oc)),
                |oc| _mm512_loadu_ps(bias.as_ptr().add(oc)),
                _mm512_add_ps,
                |ptr| _mm512_loadu_ps(ptr),
                _mm512_fmadd_ps,
                |oc, val| _mm512_storeu_ps(out_frame.as_mut_ptr().add(oc), val)
            );
            out_c += 16;
        }

        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            for in_c in 0..in_len {
                let w = *weights.get_unchecked(in_c * out_len + out_c);
                sum += *in_frame.get_unchecked(in_c) * w;
            }
            *out_frame.get_unchecked_mut(out_c) += sum;
            out_c += 1;
        }
    }
}
