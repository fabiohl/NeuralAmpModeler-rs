// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::gemv_f32_inner_loop_avx2;
use core::arch::x86_64::*;

const SMALL_IN_LEN_THRESHOLD_AVX2: usize = 4;

// ── Batched f32 GEMV ──────────────────────────────────────────────────────────

/// Batch GEMV overwrite with bias using native f32 weights via AVX2.
///
/// Unified kernel using broadcast-input / accumulator-output pattern
/// with masked tail via `_mm256_maskstore_ps`. Covers all shapes without
/// scalar fallback.
///
/// Strategy:
/// - `in_len == 1`: broadcast the single input, multiply-add weights and bias
///   in blocks of 8 output channels, maskstore tail.
/// - `out_len == 1`: batch 8 frames per YMM (8-wide dot product deferred
///   horizontal reduction), with per-frame fallback for remainder.
/// - General: 8-way unrolled broadcast-input over output-channel blocks
///   of 8, with `_mm256_maskload_ps` + `_mm256_maskstore_ps` for the
///   final partial block when `out_len % 8 != 0`.
///
/// # Safety
/// - `num_frames` must be > 0.
/// - `in_frames.len()` must be a multiple of `num_frames`.
/// - `out_frames.len()` must be a multiple of `num_frames`.
/// - `weights.len()` must be >= `in_len * out_len` where `in_len = in_frames.len() / num_frames`
///   and `out_len = out_frames.len() / num_frames`.
/// - `bias.len()` must be >= `out_len`.
///
/// All slices must be valid and accessible for reading/writing.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_with_bias_f32_avx2(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
) {
    if num_frames == 0 {
        return;
    }
    debug_assert_eq!(in_frames.len() % num_frames, 0);
    debug_assert_eq!(out_frames.len() % num_frames, 0);
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;
    debug_assert!(weights.len() >= in_len * out_len);
    debug_assert!(bias.len() >= out_len);
    assert!(in_frames.len() == num_frames * in_len);
    assert!(out_frames.len() == num_frames * out_len);
    assert!(weights.len() >= in_len * out_len);
    assert!(bias.len() >= out_len);

    // ── in_len == 1: out[j] = bias[j] + in[0] * weights[j] ──────────────
    if in_len == 1 {
        for n in 0..num_frames {
            let v_in = _mm256_set1_ps(*in_frames.get_unchecked(n));
            let mut oc = 0;
            while oc + 8 <= out_len {
                let v_w = _mm256_loadu_ps(weights.as_ptr().add(oc));
                let v_b = _mm256_loadu_ps(bias.as_ptr().add(oc));
                let v_out = _mm256_fmadd_ps(v_in, v_w, v_b);
                _mm256_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), v_out);
                oc += 8;
            }
            if oc < out_len {
                let rem = out_len - oc;
                let mut mask_buf = [0i32; 8];
                let mut w_buf = [0.0f32; 8];
                let mut b_buf = [0.0f32; 8];
                for i in 0..rem {
                    mask_buf[i] = -1;
                    w_buf[i] = *weights.get_unchecked(oc + i);
                    b_buf[i] = *bias.get_unchecked(oc + i);
                }
                let mask = _mm256_loadu_si256(mask_buf.as_ptr() as *const __m256i);
                let v_w = _mm256_loadu_ps(w_buf.as_ptr());
                let v_b = _mm256_loadu_ps(b_buf.as_ptr());
                let v_out = _mm256_fmadd_ps(v_in, v_w, v_b);
                _mm256_maskstore_ps(out_frames.as_mut_ptr().add(n * out_len + oc), mask, v_out);
            }
        }
        return;
    }

    // ── out_len == 1: batch 8 frames per YMM ────────────────────────────
    if out_len == 1 {
        let mut n = 0;
        if in_len > SMALL_IN_LEN_THRESHOLD_AVX2 {
            while n + 8 <= num_frames {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();
                let mut acc4 = _mm256_setzero_ps();
                let mut acc5 = _mm256_setzero_ps();
                let mut acc6 = _mm256_setzero_ps();
                let mut acc7 = _mm256_setzero_ps();
                let mut ic = 0;
                while ic + 8 <= in_len {
                    let v_w0 = _mm256_set1_ps(*weights.get_unchecked(ic));
                    let v_w1 = _mm256_set1_ps(*weights.get_unchecked(ic + 1));
                    let v_w2 = _mm256_set1_ps(*weights.get_unchecked(ic + 2));
                    let v_w3 = _mm256_set1_ps(*weights.get_unchecked(ic + 3));
                    let v_w4 = _mm256_set1_ps(*weights.get_unchecked(ic + 4));
                    let v_w5 = _mm256_set1_ps(*weights.get_unchecked(ic + 5));
                    let v_w6 = _mm256_set1_ps(*weights.get_unchecked(ic + 6));
                    let v_w7 = _mm256_set1_ps(*weights.get_unchecked(ic + 7));
                    let row0 = _mm256_loadu_ps(in_frames.as_ptr().add((n) * in_len + ic));
                    let row1 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 1) * in_len + ic));
                    let row2 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 2) * in_len + ic));
                    let row3 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 3) * in_len + ic));
                    let row4 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 4) * in_len + ic));
                    let row5 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 5) * in_len + ic));
                    let row6 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 6) * in_len + ic));
                    let row7 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 7) * in_len + ic));
                    // 8×8 matrix transpose: row_k holds frame (n+k)'s 8 input channels
                    // at offset ic. unpack → shuffle → permute2f128 converts from
                    // row-major [frame][channel] to column-major [channel][frame]
                    // layout, so v_in_k broadcasts channel k across all 8 frames.
                    let t0 = _mm256_unpacklo_ps(row0, row1);
                    let t1 = _mm256_unpackhi_ps(row0, row1);
                    let t2 = _mm256_unpacklo_ps(row2, row3);
                    let t3 = _mm256_unpackhi_ps(row2, row3);
                    let t4 = _mm256_unpacklo_ps(row4, row5);
                    let t5 = _mm256_unpackhi_ps(row4, row5);
                    let t6 = _mm256_unpacklo_ps(row6, row7);
                    let t7 = _mm256_unpackhi_ps(row6, row7);
                    let s0 = _mm256_shuffle_ps(t0, t2, 0x44);
                    let s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
                    let s2 = _mm256_shuffle_ps(t1, t3, 0x44);
                    let s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
                    let s4 = _mm256_shuffle_ps(t4, t6, 0x44);
                    let s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
                    let s6 = _mm256_shuffle_ps(t5, t7, 0x44);
                    let s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
                    let v_in0 = _mm256_permute2f128_ps(s0, s4, 0x20);
                    let v_in1 = _mm256_permute2f128_ps(s1, s5, 0x20);
                    let v_in2 = _mm256_permute2f128_ps(s2, s6, 0x20);
                    let v_in3 = _mm256_permute2f128_ps(s3, s7, 0x20);
                    let v_in4 = _mm256_permute2f128_ps(s0, s4, 0x31);
                    let v_in5 = _mm256_permute2f128_ps(s1, s5, 0x31);
                    let v_in6 = _mm256_permute2f128_ps(s2, s6, 0x31);
                    let v_in7 = _mm256_permute2f128_ps(s3, s7, 0x31);
                    acc0 = _mm256_fmadd_ps(v_in0, v_w0, acc0);
                    acc1 = _mm256_fmadd_ps(v_in1, v_w1, acc1);
                    acc2 = _mm256_fmadd_ps(v_in2, v_w2, acc2);
                    acc3 = _mm256_fmadd_ps(v_in3, v_w3, acc3);
                    acc4 = _mm256_fmadd_ps(v_in4, v_w4, acc4);
                    acc5 = _mm256_fmadd_ps(v_in5, v_w5, acc5);
                    acc6 = _mm256_fmadd_ps(v_in6, v_w6, acc6);
                    acc7 = _mm256_fmadd_ps(v_in7, v_w7, acc7);
                    ic += 8;
                }
                acc0 = _mm256_add_ps(acc0, acc1);
                acc2 = _mm256_add_ps(acc2, acc3);
                acc4 = _mm256_add_ps(acc4, acc5);
                acc6 = _mm256_add_ps(acc6, acc7);
                acc0 = _mm256_add_ps(acc0, acc2);
                acc4 = _mm256_add_ps(acc4, acc6);
                acc0 = _mm256_add_ps(acc0, acc4);
                while ic < in_len {
                    let v_w = _mm256_set1_ps(*weights.get_unchecked(ic));
                    let mut buf = [0.0f32; 8];
                    #[expect(
                        clippy::needless_range_loop,
                        reason = "Range loop required for explicit SIMD lane indexing not expressible via iterator"
                    )]
                    for j in 0..8 {
                        buf[j] = *in_frames.get_unchecked((n + j) * in_len + ic);
                    }
                    let v_in = _mm256_loadu_ps(buf.as_ptr());
                    acc0 = _mm256_fmadd_ps(v_in, v_w, acc0);
                    ic += 1;
                }
                let v_b = _mm256_set1_ps(*bias.get_unchecked(0));
                acc0 = _mm256_add_ps(acc0, v_b);
                _mm256_storeu_ps(out_frames.as_mut_ptr().add(n), acc0);
                n += 8;
            }
        }
        for n in n..num_frames {
            let mut acc = _mm256_setzero_ps();
            let mut ic = 0;
            while ic + 8 <= in_len {
                let v_in = _mm256_loadu_ps(in_frames.as_ptr().add(n * in_len + ic));
                let v_w = _mm256_loadu_ps(weights.as_ptr().add(ic));
                acc = _mm256_fmadd_ps(v_in, v_w, acc);
                ic += 8;
            }
            if ic < in_len {
                let rem = in_len - ic;
                let mut buf_in = [0.0f32; 8];
                let mut buf_w = [0.0f32; 8];
                for i in 0..rem {
                    buf_in[i] = *in_frames.get_unchecked(n * in_len + ic + i);
                    buf_w[i] = *weights.get_unchecked(ic + i);
                }
                let v_in = _mm256_loadu_ps(buf_in.as_ptr());
                let v_w = _mm256_loadu_ps(buf_w.as_ptr());
                acc = _mm256_fmadd_ps(v_in, v_w, acc);
            }
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
            let sum: f32 = tmp.iter().sum();
            *out_frames.get_unchecked_mut(n) = sum + *bias.get_unchecked(0);
        }
        return;
    }

    // ── General unified path: all out_len >= 1 ───────────────────────────
    for n in 0..num_frames {
        let mut oc = 0;
        while oc + 8 <= out_len {
            let mut acc0 = _mm256_loadu_ps(bias.as_ptr().add(oc));
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut acc4 = _mm256_setzero_ps();
            let mut acc5 = _mm256_setzero_ps();
            let mut acc6 = _mm256_setzero_ps();
            let mut acc7 = _mm256_setzero_ps();
            let mut ic = 0;
            let frame_in = &in_frames[n * in_len..(n + 1) * in_len];
            gemv_f32_inner_loop_avx2!(
                ic, in_len, out_len, oc, frame_in, weights, acc0, acc1, acc2, acc3, acc4, acc5,
                acc6, acc7
            );
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc4 = _mm256_add_ps(acc4, acc5);
            acc6 = _mm256_add_ps(acc6, acc7);
            acc0 = _mm256_add_ps(acc0, acc2);
            acc4 = _mm256_add_ps(acc4, acc6);
            acc0 = _mm256_add_ps(acc0, acc4);
            let mut tail = [0.0f32; 8];
            while ic < in_len {
                let inp = *in_frames.get_unchecked(n * in_len + ic);
                let base_idx = ic * out_len + oc;
                for (j, t) in tail.iter_mut().enumerate() {
                    *t = f32::mul_add(inp, *weights.get_unchecked(base_idx + j), *t);
                }
                ic += 1;
            }
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(tail.as_ptr()));
            _mm256_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), acc0);
            oc += 8;
        }
        if oc < out_len {
            let rem = out_len - oc;
            let mut mask_buf = [0i32; 8];
            mask_buf.iter_mut().take(rem).for_each(|m| *m = -1);
            let mask = _mm256_loadu_si256(mask_buf.as_ptr() as *const __m256i);
            let mut bias_buf = [0.0f32; 8];
            for (i, b) in bias_buf.iter_mut().enumerate().take(rem) {
                *b = *bias.get_unchecked(oc + i);
            }
            let mut acc0 = _mm256_loadu_ps(bias_buf.as_ptr());
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut acc4 = _mm256_setzero_ps();
            let mut acc5 = _mm256_setzero_ps();
            let mut acc6 = _mm256_setzero_ps();
            let mut acc7 = _mm256_setzero_ps();
            let mut ic = 0;
            while ic + 8 <= in_len {
                let vs0 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vs1 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 1));
                let vs2 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 2));
                let vs3 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 3));
                let vs4 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 4));
                let vs5 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 5));
                let vs6 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 6));
                let vs7 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 7));
                let w_ptr = weights.as_ptr().add(ic * out_len + oc);
                let w0 = _mm256_maskload_ps(w_ptr, mask);
                acc0 = _mm256_fmadd_ps(vs0, w0, acc0);
                let w1 = _mm256_maskload_ps(w_ptr.add(out_len), mask);
                acc1 = _mm256_fmadd_ps(vs1, w1, acc1);
                let w2 = _mm256_maskload_ps(w_ptr.add(2 * out_len), mask);
                acc2 = _mm256_fmadd_ps(vs2, w2, acc2);
                let w3 = _mm256_maskload_ps(w_ptr.add(3 * out_len), mask);
                acc3 = _mm256_fmadd_ps(vs3, w3, acc3);
                let w4 = _mm256_maskload_ps(w_ptr.add(4 * out_len), mask);
                acc4 = _mm256_fmadd_ps(vs4, w4, acc4);
                let w5 = _mm256_maskload_ps(w_ptr.add(5 * out_len), mask);
                acc5 = _mm256_fmadd_ps(vs5, w5, acc5);
                let w6 = _mm256_maskload_ps(w_ptr.add(6 * out_len), mask);
                acc6 = _mm256_fmadd_ps(vs6, w6, acc6);
                let w7 = _mm256_maskload_ps(w_ptr.add(7 * out_len), mask);
                acc7 = _mm256_fmadd_ps(vs7, w7, acc7);
                ic += 8;
            }
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc4 = _mm256_add_ps(acc4, acc5);
            acc6 = _mm256_add_ps(acc6, acc7);
            acc0 = _mm256_add_ps(acc0, acc2);
            acc4 = _mm256_add_ps(acc4, acc6);
            acc0 = _mm256_add_ps(acc0, acc4);
            while ic < in_len {
                let vs = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vw = _mm256_maskload_ps(weights.as_ptr().add(ic * out_len + oc), mask);
                acc0 = _mm256_fmadd_ps(vs, vw, acc0);
                ic += 1;
            }
            _mm256_maskstore_ps(out_frames.as_mut_ptr().add(n * out_len + oc), mask, acc0);
        }
    }
}

/// Batch GEMV overwrite without bias using native f32 weights via AVX2.
///
/// Unified kernel using broadcast-input / accumulator-output pattern
/// with masked tail via `_mm256_maskstore_ps`. Covers all shapes without
/// scalar fallback.
///
/// Strategy:
/// - `in_len == 1`: broadcast the single input, multiply-add weights
///   in blocks of 8 output channels, maskstore tail.
/// - `out_len == 1`: batch 8 frames per YMM (8-wide dot product deferred
///   horizontal reduction), with per-frame fallback for remainder.
/// - General: 8-way unrolled broadcast-input over output-channel blocks
///   of 8, with `_mm256_maskload_ps` + `_mm256_maskstore_ps` for the
///   final partial block when `out_len % 8 != 0`.
///
/// # Safety
/// - `num_frames` must be > 0.
/// - `in_frames.len()` must be a multiple of `num_frames`.
/// - `out_frames.len()` must be a multiple of `num_frames`.
/// - `weights.len()` must be >= `in_len * out_len` where `in_len = in_frames.len() / num_frames`
///   and `out_len = out_frames.len() / num_frames`.
///
/// All slices must be valid and accessible for reading/writing.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_no_bias_f32_avx2(
    in_frames: &[f32],
    weights: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
) {
    if num_frames == 0 {
        return;
    }
    debug_assert_eq!(in_frames.len() % num_frames, 0);
    debug_assert_eq!(out_frames.len() % num_frames, 0);
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;
    debug_assert!(weights.len() >= in_len * out_len);
    assert!(in_frames.len() == num_frames * in_len);
    assert!(out_frames.len() == num_frames * out_len);
    assert!(weights.len() >= in_len * out_len);

    // ── in_len == 1: out[j] = in[0] * weights[j] ────────────────────────
    if in_len == 1 {
        for n in 0..num_frames {
            let v_in = _mm256_set1_ps(*in_frames.get_unchecked(n));
            let mut oc = 0;
            while oc + 8 <= out_len {
                let v_w = _mm256_loadu_ps(weights.as_ptr().add(oc));
                let v_out = _mm256_mul_ps(v_in, v_w);
                _mm256_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), v_out);
                oc += 8;
            }
            if oc < out_len {
                let rem = out_len - oc;
                let mut mask_buf = [0i32; 8];
                let mut w_buf = [0.0f32; 8];
                for i in 0..rem {
                    mask_buf[i] = -1;
                    w_buf[i] = *weights.get_unchecked(oc + i);
                }
                let mask = _mm256_loadu_si256(mask_buf.as_ptr() as *const __m256i);
                let v_w = _mm256_loadu_ps(w_buf.as_ptr());
                let v_out = _mm256_mul_ps(v_in, v_w);
                _mm256_maskstore_ps(out_frames.as_mut_ptr().add(n * out_len + oc), mask, v_out);
            }
        }
        return;
    }

    // ── out_len == 1: batch 8 frames per YMM ────────────────────────────
    if out_len == 1 {
        let mut n = 0;
        if in_len > SMALL_IN_LEN_THRESHOLD_AVX2 {
            while n + 8 <= num_frames {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();
                let mut acc4 = _mm256_setzero_ps();
                let mut acc5 = _mm256_setzero_ps();
                let mut acc6 = _mm256_setzero_ps();
                let mut acc7 = _mm256_setzero_ps();
                let mut ic = 0;
                while ic + 8 <= in_len {
                    let v_w0 = _mm256_set1_ps(*weights.get_unchecked(ic));
                    let v_w1 = _mm256_set1_ps(*weights.get_unchecked(ic + 1));
                    let v_w2 = _mm256_set1_ps(*weights.get_unchecked(ic + 2));
                    let v_w3 = _mm256_set1_ps(*weights.get_unchecked(ic + 3));
                    let v_w4 = _mm256_set1_ps(*weights.get_unchecked(ic + 4));
                    let v_w5 = _mm256_set1_ps(*weights.get_unchecked(ic + 5));
                    let v_w6 = _mm256_set1_ps(*weights.get_unchecked(ic + 6));
                    let v_w7 = _mm256_set1_ps(*weights.get_unchecked(ic + 7));
                    let row0 = _mm256_loadu_ps(in_frames.as_ptr().add((n) * in_len + ic));
                    let row1 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 1) * in_len + ic));
                    let row2 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 2) * in_len + ic));
                    let row3 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 3) * in_len + ic));
                    let row4 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 4) * in_len + ic));
                    let row5 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 5) * in_len + ic));
                    let row6 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 6) * in_len + ic));
                    let row7 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 7) * in_len + ic));
                    // 8×8 matrix transpose: row_k holds frame (n+k)'s 8 input channels
                    // at offset ic. unpack → shuffle → permute2f128 converts from
                    // row-major [frame][channel] to column-major [channel][frame]
                    // layout, so v_in_k broadcasts channel k across all 8 frames.
                    let t0 = _mm256_unpacklo_ps(row0, row1);
                    let t1 = _mm256_unpackhi_ps(row0, row1);
                    let t2 = _mm256_unpacklo_ps(row2, row3);
                    let t3 = _mm256_unpackhi_ps(row2, row3);
                    let t4 = _mm256_unpacklo_ps(row4, row5);
                    let t5 = _mm256_unpackhi_ps(row4, row5);
                    let t6 = _mm256_unpacklo_ps(row6, row7);
                    let t7 = _mm256_unpackhi_ps(row6, row7);
                    let s0 = _mm256_shuffle_ps(t0, t2, 0x44);
                    let s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
                    let s2 = _mm256_shuffle_ps(t1, t3, 0x44);
                    let s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
                    let s4 = _mm256_shuffle_ps(t4, t6, 0x44);
                    let s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
                    let s6 = _mm256_shuffle_ps(t5, t7, 0x44);
                    let s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
                    let v_in0 = _mm256_permute2f128_ps(s0, s4, 0x20);
                    let v_in1 = _mm256_permute2f128_ps(s1, s5, 0x20);
                    let v_in2 = _mm256_permute2f128_ps(s2, s6, 0x20);
                    let v_in3 = _mm256_permute2f128_ps(s3, s7, 0x20);
                    let v_in4 = _mm256_permute2f128_ps(s0, s4, 0x31);
                    let v_in5 = _mm256_permute2f128_ps(s1, s5, 0x31);
                    let v_in6 = _mm256_permute2f128_ps(s2, s6, 0x31);
                    let v_in7 = _mm256_permute2f128_ps(s3, s7, 0x31);
                    acc0 = _mm256_fmadd_ps(v_in0, v_w0, acc0);
                    acc1 = _mm256_fmadd_ps(v_in1, v_w1, acc1);
                    acc2 = _mm256_fmadd_ps(v_in2, v_w2, acc2);
                    acc3 = _mm256_fmadd_ps(v_in3, v_w3, acc3);
                    acc4 = _mm256_fmadd_ps(v_in4, v_w4, acc4);
                    acc5 = _mm256_fmadd_ps(v_in5, v_w5, acc5);
                    acc6 = _mm256_fmadd_ps(v_in6, v_w6, acc6);
                    acc7 = _mm256_fmadd_ps(v_in7, v_w7, acc7);
                    ic += 8;
                }
                acc0 = _mm256_add_ps(acc0, acc1);
                acc2 = _mm256_add_ps(acc2, acc3);
                acc4 = _mm256_add_ps(acc4, acc5);
                acc6 = _mm256_add_ps(acc6, acc7);
                acc0 = _mm256_add_ps(acc0, acc2);
                acc4 = _mm256_add_ps(acc4, acc6);
                acc0 = _mm256_add_ps(acc0, acc4);
                while ic < in_len {
                    let v_w = _mm256_set1_ps(*weights.get_unchecked(ic));
                    let mut buf = [0.0f32; 8];
                    #[expect(
                        clippy::needless_range_loop,
                        reason = "Range loop required for explicit SIMD lane indexing not expressible via iterator"
                    )]
                    for j in 0..8 {
                        buf[j] = *in_frames.get_unchecked((n + j) * in_len + ic);
                    }
                    let v_in = _mm256_loadu_ps(buf.as_ptr());
                    acc0 = _mm256_fmadd_ps(v_in, v_w, acc0);
                    ic += 1;
                }
                _mm256_storeu_ps(out_frames.as_mut_ptr().add(n), acc0);
                n += 8;
            }
        }
        for n in n..num_frames {
            let mut acc = _mm256_setzero_ps();
            let mut ic = 0;
            while ic + 8 <= in_len {
                let v_in = _mm256_loadu_ps(in_frames.as_ptr().add(n * in_len + ic));
                let v_w = _mm256_loadu_ps(weights.as_ptr().add(ic));
                acc = _mm256_fmadd_ps(v_in, v_w, acc);
                ic += 8;
            }
            if ic < in_len {
                let mut buf_in = [0.0f32; 8];
                let mut buf_w = [0.0f32; 8];
                let rem = in_len - ic;
                for i in 0..rem {
                    buf_in[i] = *in_frames.get_unchecked(n * in_len + ic + i);
                    buf_w[i] = *weights.get_unchecked(ic + i);
                }
                let v_in = _mm256_loadu_ps(buf_in.as_ptr());
                let v_w = _mm256_loadu_ps(buf_w.as_ptr());
                acc = _mm256_fmadd_ps(v_in, v_w, acc);
            }
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
            let sum: f32 = tmp.iter().sum();
            *out_frames.get_unchecked_mut(n) = sum;
        }
        return;
    }

    // ── General unified path: all out_len >= 1 ───────────────────────────
    for n in 0..num_frames {
        let mut oc = 0;
        while oc + 8 <= out_len {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut acc4 = _mm256_setzero_ps();
            let mut acc5 = _mm256_setzero_ps();
            let mut acc6 = _mm256_setzero_ps();
            let mut acc7 = _mm256_setzero_ps();
            let mut ic = 0;
            let frame_in = &in_frames[n * in_len..(n + 1) * in_len];
            gemv_f32_inner_loop_avx2!(
                ic, in_len, out_len, oc, frame_in, weights, acc0, acc1, acc2, acc3, acc4, acc5,
                acc6, acc7
            );
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc4 = _mm256_add_ps(acc4, acc5);
            acc6 = _mm256_add_ps(acc6, acc7);
            acc0 = _mm256_add_ps(acc0, acc2);
            acc4 = _mm256_add_ps(acc4, acc6);
            acc0 = _mm256_add_ps(acc0, acc4);
            let mut tail = [0.0f32; 8];
            while ic < in_len {
                let inp = *in_frames.get_unchecked(n * in_len + ic);
                let base_idx = ic * out_len + oc;
                for (j, t) in tail.iter_mut().enumerate() {
                    *t = f32::mul_add(inp, *weights.get_unchecked(base_idx + j), *t);
                }
                ic += 1;
            }
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(tail.as_ptr()));
            _mm256_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), acc0);
            oc += 8;
        }
        if oc < out_len {
            let rem = out_len - oc;
            let mut mask_buf = [0i32; 8];
            mask_buf.iter_mut().take(rem).for_each(|m| *m = -1);
            let mask = _mm256_loadu_si256(mask_buf.as_ptr() as *const __m256i);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut acc4 = _mm256_setzero_ps();
            let mut acc5 = _mm256_setzero_ps();
            let mut acc6 = _mm256_setzero_ps();
            let mut acc7 = _mm256_setzero_ps();
            let mut ic = 0;
            while ic + 8 <= in_len {
                let vs0 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vs1 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 1));
                let vs2 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 2));
                let vs3 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 3));
                let vs4 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 4));
                let vs5 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 5));
                let vs6 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 6));
                let vs7 = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 7));
                let w_ptr = weights.as_ptr().add(ic * out_len + oc);
                let w0 = _mm256_maskload_ps(w_ptr, mask);
                acc0 = _mm256_fmadd_ps(vs0, w0, acc0);
                let w1 = _mm256_maskload_ps(w_ptr.add(out_len), mask);
                acc1 = _mm256_fmadd_ps(vs1, w1, acc1);
                let w2 = _mm256_maskload_ps(w_ptr.add(2 * out_len), mask);
                acc2 = _mm256_fmadd_ps(vs2, w2, acc2);
                let w3 = _mm256_maskload_ps(w_ptr.add(3 * out_len), mask);
                acc3 = _mm256_fmadd_ps(vs3, w3, acc3);
                let w4 = _mm256_maskload_ps(w_ptr.add(4 * out_len), mask);
                acc4 = _mm256_fmadd_ps(vs4, w4, acc4);
                let w5 = _mm256_maskload_ps(w_ptr.add(5 * out_len), mask);
                acc5 = _mm256_fmadd_ps(vs5, w5, acc5);
                let w6 = _mm256_maskload_ps(w_ptr.add(6 * out_len), mask);
                acc6 = _mm256_fmadd_ps(vs6, w6, acc6);
                let w7 = _mm256_maskload_ps(w_ptr.add(7 * out_len), mask);
                acc7 = _mm256_fmadd_ps(vs7, w7, acc7);
                ic += 8;
            }
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc4 = _mm256_add_ps(acc4, acc5);
            acc6 = _mm256_add_ps(acc6, acc7);
            acc0 = _mm256_add_ps(acc0, acc2);
            acc4 = _mm256_add_ps(acc4, acc6);
            acc0 = _mm256_add_ps(acc0, acc4);
            while ic < in_len {
                let vs = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vw = _mm256_maskload_ps(weights.as_ptr().add(ic * out_len + oc), mask);
                acc0 = _mm256_fmadd_ps(vs, vw, acc0);
                ic += 1;
            }
            _mm256_maskstore_ps(out_frames.as_mut_ptr().add(n * out_len + oc), mask, acc0);
        }
    }
}

/// Produto externo escalar x vetor de pesos, sem bias, para `in_len == 1`.
///
/// Para cada frame `n`: `out[n*OUT + oc] = in[n] * weights[oc]`.
///
/// Const-generic em `OUT` permite endereçamento imediato no loop interno,
/// eliminando chains de `leaq` e a divisão `in_len = in_frames.len()/num_frames`.
///
/// # Safety
/// `in_frames.len() * OUT == out_frames.len()`
/// `weights.len() >= OUT`
/// AVX2+FMA ISA must be available (x86-64-v3).
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn broadcast_scale_f32_avx2<const OUT: usize>(
    in_frames: &[f32],
    weights: &[f32],
    out_frames: &mut [f32],
) {
    let num_frames = in_frames.len();
    debug_assert_eq!(out_frames.len(), num_frames * OUT);
    debug_assert!(weights.len() >= OUT);

    for n in 0..num_frames {
        let v_in = _mm256_set1_ps(*in_frames.get_unchecked(n));
        let mut oc = 0;
        while oc + 8 <= OUT {
            let v_w = _mm256_loadu_ps(weights.as_ptr().add(oc));
            let v_out = _mm256_mul_ps(v_in, v_w);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add(n * OUT + oc), v_out);
            oc += 8;
        }
        if oc < OUT {
            let rem = OUT - oc;
            let mut mask_buf = [0i32; 8];
            let mut w_buf = [0.0f32; 8];
            for i in 0..rem {
                mask_buf[i] = -1;
                w_buf[i] = *weights.get_unchecked(oc + i);
            }
            let mask = _mm256_loadu_si256(mask_buf.as_ptr() as *const __m256i);
            let v_w = _mm256_loadu_ps(w_buf.as_ptr());
            let v_out = _mm256_mul_ps(v_in, v_w);
            _mm256_maskstore_ps(out_frames.as_mut_ptr().add(n * OUT + oc), mask, v_out);
        }
    }
}

/// Produto externo escalar x vetor de pesos, com bias, para `in_len == 1`.
///
/// Para cada frame `n`: `out[n*OUT + oc] = bias[oc] + in[n] * weights[oc]`.
///
/// # Safety
/// `in_frames.len() * OUT == out_frames.len()`
/// `weights.len() >= OUT`, `bias.len() >= OUT`
/// AVX2+FMA ISA must be available (x86-64-v3).
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn broadcast_scale_with_bias_f32_avx2<const OUT: usize>(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frames: &mut [f32],
) {
    let num_frames = in_frames.len();
    debug_assert_eq!(out_frames.len(), num_frames * OUT);
    debug_assert!(weights.len() >= OUT);
    debug_assert!(bias.len() >= OUT);

    for n in 0..num_frames {
        let v_in = _mm256_set1_ps(*in_frames.get_unchecked(n));
        let mut oc = 0;
        while oc + 8 <= OUT {
            let v_w = _mm256_loadu_ps(weights.as_ptr().add(oc));
            let v_b = _mm256_loadu_ps(bias.as_ptr().add(oc));
            let v_out = _mm256_fmadd_ps(v_in, v_w, v_b);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add(n * OUT + oc), v_out);
            oc += 8;
        }
        if oc < OUT {
            let rem = OUT - oc;
            let mut mask_buf = [0i32; 8];
            let mut w_buf = [0.0f32; 8];
            let mut b_buf = [0.0f32; 8];
            for i in 0..rem {
                mask_buf[i] = -1;
                w_buf[i] = *weights.get_unchecked(oc + i);
                b_buf[i] = *bias.get_unchecked(oc + i);
            }
            let mask = _mm256_loadu_si256(mask_buf.as_ptr() as *const __m256i);
            let v_w = _mm256_loadu_ps(w_buf.as_ptr());
            let v_b = _mm256_loadu_ps(b_buf.as_ptr());
            let v_out = _mm256_fmadd_ps(v_in, v_w, v_b);
            _mm256_maskstore_ps(out_frames.as_mut_ptr().add(n * OUT + oc), mask, v_out);
        }
    }
}
