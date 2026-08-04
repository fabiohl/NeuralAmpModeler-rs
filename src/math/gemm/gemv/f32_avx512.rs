// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use core::arch::x86_64::*;

const SMALL_IN_LEN_THRESHOLD_AVX512: usize = 4;

// ── Batched f32 GEMV ──────────────────────────────────────────────────────────

/// Batch GEMV overwrite with bias using native f32 weights via AVX-512.
///
/// Unified kernel using broadcast-input / accumulator-output pattern
/// with masked tail via `_mm512_mask_storeu_ps`. Covers all shapes without
/// scalar fallback.
///
/// Strategy:
/// - `in_len == 1`: broadcast the single input, multiply-add weights and bias
///   in blocks of 16 output channels, maskstore tail.
/// - `out_len == 1`: batch 16 frames per ZMM (8-accumulator deferred
///   horizontal reduction), with per-frame fallback for remainder.
/// - General: 8-way unrolled broadcast-input over output-channel blocks
///   of 16, with `_mm512_maskz_loadu_ps` + `_mm512_mask_storeu_ps` for the
///   final partial block when `out_len % 16 != 0`.
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
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_with_bias_f32_avx512(
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

    // ── in_len == 1: out[j] = bias[j] + in[0] * weights[j] ──────────────
    if in_len == 1 {
        for n in 0..num_frames {
            let v_in = _mm512_set1_ps(*in_frames.get_unchecked(n));
            let mut oc = 0;
            while oc + 16 <= out_len {
                let v_w = _mm512_loadu_ps(weights.as_ptr().add(oc));
                let v_b = _mm512_loadu_ps(bias.as_ptr().add(oc));
                let v_out = _mm512_fmadd_ps(v_in, v_w, v_b);
                _mm512_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), v_out);
                oc += 16;
            }
            if oc < out_len {
                let rem = out_len - oc;
                let mask: u16 = (1 << rem) - 1;
                let v_w = _mm512_maskz_loadu_ps(mask, weights.as_ptr().add(oc));
                let v_b = _mm512_maskz_loadu_ps(mask, bias.as_ptr().add(oc));
                let v_out = _mm512_fmadd_ps(v_in, v_w, v_b);
                _mm512_mask_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), mask, v_out);
            }
        }
        return;
    }

    // ── out_len == 1: batch 16 frames per ZMM ────────────────────────────
    if out_len == 1 {
        let mut n = 0;
        if in_len > SMALL_IN_LEN_THRESHOLD_AVX512 {
            while n + 16 <= num_frames {
                let mut acc0 = _mm512_setzero_ps();
                let mut acc1 = _mm512_setzero_ps();
                let mut acc2 = _mm512_setzero_ps();
                let mut acc3 = _mm512_setzero_ps();
                let mut acc4 = _mm512_setzero_ps();
                let mut acc5 = _mm512_setzero_ps();
                let mut acc6 = _mm512_setzero_ps();
                let mut acc7 = _mm512_setzero_ps();
                let mut ic = 0;
                while ic + 8 <= in_len {
                    let v_w0 = _mm512_set1_ps(*weights.get_unchecked(ic));
                    let v_w1 = _mm512_set1_ps(*weights.get_unchecked(ic + 1));
                    let v_w2 = _mm512_set1_ps(*weights.get_unchecked(ic + 2));
                    let v_w3 = _mm512_set1_ps(*weights.get_unchecked(ic + 3));
                    let v_w4 = _mm512_set1_ps(*weights.get_unchecked(ic + 4));
                    let v_w5 = _mm512_set1_ps(*weights.get_unchecked(ic + 5));
                    let v_w6 = _mm512_set1_ps(*weights.get_unchecked(ic + 6));
                    let v_w7 = _mm512_set1_ps(*weights.get_unchecked(ic + 7));
                    let ra0 = _mm256_loadu_ps(in_frames.as_ptr().add((n) * in_len + ic));
                    let ra1 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 1) * in_len + ic));
                    let ra2 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 2) * in_len + ic));
                    let ra3 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 3) * in_len + ic));
                    let ra4 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 4) * in_len + ic));
                    let ra5 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 5) * in_len + ic));
                    let ra6 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 6) * in_len + ic));
                    let ra7 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 7) * in_len + ic));
                    let rb0 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 8) * in_len + ic));
                    let rb1 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 9) * in_len + ic));
                    let rb2 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 10) * in_len + ic));
                    let rb3 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 11) * in_len + ic));
                    let rb4 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 12) * in_len + ic));
                    let rb5 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 13) * in_len + ic));
                    let rb6 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 14) * in_len + ic));
                    let rb7 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 15) * in_len + ic));
                    // 16×8 matrix transpose: 16 frames (ra* = frames 0..7,
                    // rb* = frames 8..15) × 8 input channels at offset ic.
                    // Two independent shuffle trees (group a, group b) each
                    // transpose 8×8 via unpack → shuffle → permute2f128, then
                    // _mm512_insertf32x8 merges the transposed YMM halves into
                    // ZMM. Each v_in_k broadcasts channel k across all 16 frames.
                    let ta0 = _mm256_unpacklo_ps(ra0, ra1);
                    let ta1 = _mm256_unpackhi_ps(ra0, ra1);
                    let ta2 = _mm256_unpacklo_ps(ra2, ra3);
                    let ta3 = _mm256_unpackhi_ps(ra2, ra3);
                    let ta4 = _mm256_unpacklo_ps(ra4, ra5);
                    let ta5 = _mm256_unpackhi_ps(ra4, ra5);
                    let ta6 = _mm256_unpacklo_ps(ra6, ra7);
                    let ta7 = _mm256_unpackhi_ps(ra6, ra7);
                    let tb0 = _mm256_unpacklo_ps(rb0, rb1);
                    let tb1 = _mm256_unpackhi_ps(rb0, rb1);
                    let tb2 = _mm256_unpacklo_ps(rb2, rb3);
                    let tb3 = _mm256_unpackhi_ps(rb2, rb3);
                    let tb4 = _mm256_unpacklo_ps(rb4, rb5);
                    let tb5 = _mm256_unpackhi_ps(rb4, rb5);
                    let tb6 = _mm256_unpacklo_ps(rb6, rb7);
                    let tb7 = _mm256_unpackhi_ps(rb6, rb7);
                    let sa0 = _mm256_shuffle_ps(ta0, ta2, 0x44);
                    let sa1 = _mm256_shuffle_ps(ta0, ta2, 0xEE);
                    let sa2 = _mm256_shuffle_ps(ta1, ta3, 0x44);
                    let sa3 = _mm256_shuffle_ps(ta1, ta3, 0xEE);
                    let sa4 = _mm256_shuffle_ps(ta4, ta6, 0x44);
                    let sa5 = _mm256_shuffle_ps(ta4, ta6, 0xEE);
                    let sa6 = _mm256_shuffle_ps(ta5, ta7, 0x44);
                    let sa7 = _mm256_shuffle_ps(ta5, ta7, 0xEE);
                    let sb0 = _mm256_shuffle_ps(tb0, tb2, 0x44);
                    let sb1 = _mm256_shuffle_ps(tb0, tb2, 0xEE);
                    let sb2 = _mm256_shuffle_ps(tb1, tb3, 0x44);
                    let sb3 = _mm256_shuffle_ps(tb1, tb3, 0xEE);
                    let sb4 = _mm256_shuffle_ps(tb4, tb6, 0x44);
                    let sb5 = _mm256_shuffle_ps(tb4, tb6, 0xEE);
                    let sb6 = _mm256_shuffle_ps(tb5, tb7, 0x44);
                    let sb7 = _mm256_shuffle_ps(tb5, tb7, 0xEE);
                    let ca0 = _mm256_permute2f128_ps(sa0, sa4, 0x20);
                    let ca1 = _mm256_permute2f128_ps(sa1, sa5, 0x20);
                    let ca2 = _mm256_permute2f128_ps(sa2, sa6, 0x20);
                    let ca3 = _mm256_permute2f128_ps(sa3, sa7, 0x20);
                    let ca4 = _mm256_permute2f128_ps(sa0, sa4, 0x31);
                    let ca5 = _mm256_permute2f128_ps(sa1, sa5, 0x31);
                    let ca6 = _mm256_permute2f128_ps(sa2, sa6, 0x31);
                    let ca7 = _mm256_permute2f128_ps(sa3, sa7, 0x31);
                    let cb0 = _mm256_permute2f128_ps(sb0, sb4, 0x20);
                    let cb1 = _mm256_permute2f128_ps(sb1, sb5, 0x20);
                    let cb2 = _mm256_permute2f128_ps(sb2, sb6, 0x20);
                    let cb3 = _mm256_permute2f128_ps(sb3, sb7, 0x20);
                    let cb4 = _mm256_permute2f128_ps(sb0, sb4, 0x31);
                    let cb5 = _mm256_permute2f128_ps(sb1, sb5, 0x31);
                    let cb6 = _mm256_permute2f128_ps(sb2, sb6, 0x31);
                    let cb7 = _mm256_permute2f128_ps(sb3, sb7, 0x31);
                    let v_in0 = _mm512_insertf32x8(_mm512_castps256_ps512(ca0), cb0, 1);
                    let v_in1 = _mm512_insertf32x8(_mm512_castps256_ps512(ca1), cb1, 1);
                    let v_in2 = _mm512_insertf32x8(_mm512_castps256_ps512(ca2), cb2, 1);
                    let v_in3 = _mm512_insertf32x8(_mm512_castps256_ps512(ca3), cb3, 1);
                    let v_in4 = _mm512_insertf32x8(_mm512_castps256_ps512(ca4), cb4, 1);
                    let v_in5 = _mm512_insertf32x8(_mm512_castps256_ps512(ca5), cb5, 1);
                    let v_in6 = _mm512_insertf32x8(_mm512_castps256_ps512(ca6), cb6, 1);
                    let v_in7 = _mm512_insertf32x8(_mm512_castps256_ps512(ca7), cb7, 1);
                    acc0 = _mm512_fmadd_ps(v_in0, v_w0, acc0);
                    acc1 = _mm512_fmadd_ps(v_in1, v_w1, acc1);
                    acc2 = _mm512_fmadd_ps(v_in2, v_w2, acc2);
                    acc3 = _mm512_fmadd_ps(v_in3, v_w3, acc3);
                    acc4 = _mm512_fmadd_ps(v_in4, v_w4, acc4);
                    acc5 = _mm512_fmadd_ps(v_in5, v_w5, acc5);
                    acc6 = _mm512_fmadd_ps(v_in6, v_w6, acc6);
                    acc7 = _mm512_fmadd_ps(v_in7, v_w7, acc7);
                    ic += 8;
                }
                acc0 = _mm512_add_ps(acc0, acc1);
                acc2 = _mm512_add_ps(acc2, acc3);
                acc4 = _mm512_add_ps(acc4, acc5);
                acc6 = _mm512_add_ps(acc6, acc7);
                acc0 = _mm512_add_ps(acc0, acc2);
                acc4 = _mm512_add_ps(acc4, acc6);
                acc0 = _mm512_add_ps(acc0, acc4);
                while ic < in_len {
                    let v_w = _mm512_set1_ps(*weights.get_unchecked(ic));
                    let mut buf = [0.0f32; 16];
                    #[expect(
                        clippy::needless_range_loop,
                        reason = "Range loop required for explicit SIMD lane indexing not expressible via iterator"
                    )]
                    for j in 0..16 {
                        buf[j] = *in_frames.get_unchecked((n + j) * in_len + ic);
                    }
                    let v_in = _mm512_loadu_ps(buf.as_ptr());
                    acc0 = _mm512_fmadd_ps(v_in, v_w, acc0);
                    ic += 1;
                }
                let v_b = _mm512_set1_ps(*bias.get_unchecked(0));
                acc0 = _mm512_add_ps(acc0, v_b);
                _mm512_storeu_ps(out_frames.as_mut_ptr().add(n), acc0);
                n += 16;
            }
        }
        for n in n..num_frames {
            let mut acc = _mm512_setzero_ps();
            let mut ic = 0;
            while ic + 16 <= in_len {
                let v_in = _mm512_loadu_ps(in_frames.as_ptr().add(n * in_len + ic));
                let v_w = _mm512_loadu_ps(weights.as_ptr().add(ic));
                acc = _mm512_fmadd_ps(v_in, v_w, acc);
                ic += 16;
            }
            if ic < in_len {
                let rem = in_len - ic;
                let mask: u16 = (1 << rem) - 1;
                let v_in = _mm512_maskz_loadu_ps(mask, in_frames.as_ptr().add(n * in_len + ic));
                let v_w = _mm512_maskz_loadu_ps(mask, weights.as_ptr().add(ic));
                acc = _mm512_fmadd_ps(v_in, v_w, acc);
            }
            let sum = _mm512_reduce_add_ps(acc);
            *out_frames.get_unchecked_mut(n) = sum + *bias.get_unchecked(0);
        }
        return;
    }

    // ── General unified path: all out_len >= 1 ───────────────────────────
    for n in 0..num_frames {
        let mut oc = 0;
        while oc + 16 <= out_len {
            let mut acc0 = _mm512_loadu_ps(bias.as_ptr().add(oc));
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            let mut acc4 = _mm512_setzero_ps();
            let mut acc5 = _mm512_setzero_ps();
            let mut acc6 = _mm512_setzero_ps();
            let mut acc7 = _mm512_setzero_ps();
            let mut ic = 0;
            while ic + 8 <= in_len {
                let vs0 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vs1 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 1));
                let vs2 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 2));
                let vs3 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 3));
                let vs4 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 4));
                let vs5 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 5));
                let vs6 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 6));
                let vs7 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 7));
                let w_ptr = weights.as_ptr().add(ic * out_len + oc);
                let w0 = _mm512_loadu_ps(w_ptr);
                acc0 = _mm512_fmadd_ps(vs0, w0, acc0);
                let w1 = _mm512_loadu_ps(w_ptr.add(out_len));
                acc1 = _mm512_fmadd_ps(vs1, w1, acc1);
                let w2 = _mm512_loadu_ps(w_ptr.add(2 * out_len));
                acc2 = _mm512_fmadd_ps(vs2, w2, acc2);
                let w3 = _mm512_loadu_ps(w_ptr.add(3 * out_len));
                acc3 = _mm512_fmadd_ps(vs3, w3, acc3);
                let w4 = _mm512_loadu_ps(w_ptr.add(4 * out_len));
                acc4 = _mm512_fmadd_ps(vs4, w4, acc4);
                let w5 = _mm512_loadu_ps(w_ptr.add(5 * out_len));
                acc5 = _mm512_fmadd_ps(vs5, w5, acc5);
                let w6 = _mm512_loadu_ps(w_ptr.add(6 * out_len));
                acc6 = _mm512_fmadd_ps(vs6, w6, acc6);
                let w7 = _mm512_loadu_ps(w_ptr.add(7 * out_len));
                acc7 = _mm512_fmadd_ps(vs7, w7, acc7);
                ic += 8;
            }
            acc0 = _mm512_add_ps(acc0, acc1);
            acc2 = _mm512_add_ps(acc2, acc3);
            acc4 = _mm512_add_ps(acc4, acc5);
            acc6 = _mm512_add_ps(acc6, acc7);
            acc0 = _mm512_add_ps(acc0, acc2);
            acc4 = _mm512_add_ps(acc4, acc6);
            acc0 = _mm512_add_ps(acc0, acc4);
            let mut tail = [0.0f32; 16];
            while ic < in_len {
                let inp = *in_frames.get_unchecked(n * in_len + ic);
                let base_idx = ic * out_len + oc;
                for (j, t) in tail.iter_mut().enumerate() {
                    *t = f32::mul_add(inp, *weights.get_unchecked(base_idx + j), *t);
                }
                ic += 1;
            }
            acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(tail.as_ptr()));
            _mm512_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), acc0);
            oc += 16;
        }
        if oc < out_len {
            let rem = out_len - oc;
            let mask: u16 = (1 << rem) - 1;
            let v_b = _mm512_maskz_loadu_ps(mask, bias.as_ptr().add(oc));
            let mut acc0 = _mm512_add_ps(_mm512_setzero_ps(), v_b);
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            let mut acc4 = _mm512_setzero_ps();
            let mut acc5 = _mm512_setzero_ps();
            let mut acc6 = _mm512_setzero_ps();
            let mut acc7 = _mm512_setzero_ps();
            let mut ic = 0;
            while ic + 8 <= in_len {
                let vs0 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vs1 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 1));
                let vs2 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 2));
                let vs3 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 3));
                let vs4 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 4));
                let vs5 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 5));
                let vs6 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 6));
                let vs7 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 7));
                let w_ptr = weights.as_ptr().add(ic * out_len + oc);
                let w0 = _mm512_maskz_loadu_ps(mask, w_ptr);
                acc0 = _mm512_fmadd_ps(vs0, w0, acc0);
                let w1 = _mm512_maskz_loadu_ps(mask, w_ptr.add(out_len));
                acc1 = _mm512_fmadd_ps(vs1, w1, acc1);
                let w2 = _mm512_maskz_loadu_ps(mask, w_ptr.add(2 * out_len));
                acc2 = _mm512_fmadd_ps(vs2, w2, acc2);
                let w3 = _mm512_maskz_loadu_ps(mask, w_ptr.add(3 * out_len));
                acc3 = _mm512_fmadd_ps(vs3, w3, acc3);
                let w4 = _mm512_maskz_loadu_ps(mask, w_ptr.add(4 * out_len));
                acc4 = _mm512_fmadd_ps(vs4, w4, acc4);
                let w5 = _mm512_maskz_loadu_ps(mask, w_ptr.add(5 * out_len));
                acc5 = _mm512_fmadd_ps(vs5, w5, acc5);
                let w6 = _mm512_maskz_loadu_ps(mask, w_ptr.add(6 * out_len));
                acc6 = _mm512_fmadd_ps(vs6, w6, acc6);
                let w7 = _mm512_maskz_loadu_ps(mask, w_ptr.add(7 * out_len));
                acc7 = _mm512_fmadd_ps(vs7, w7, acc7);
                ic += 8;
            }
            acc0 = _mm512_add_ps(acc0, acc1);
            acc2 = _mm512_add_ps(acc2, acc3);
            acc4 = _mm512_add_ps(acc4, acc5);
            acc6 = _mm512_add_ps(acc6, acc7);
            acc0 = _mm512_add_ps(acc0, acc2);
            acc4 = _mm512_add_ps(acc4, acc6);
            acc0 = _mm512_add_ps(acc0, acc4);
            while ic < in_len {
                let vs = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vw = _mm512_maskz_loadu_ps(mask, weights.as_ptr().add(ic * out_len + oc));
                acc0 = _mm512_fmadd_ps(vs, vw, acc0);
                ic += 1;
            }
            _mm512_mask_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), mask, acc0);
        }
    }
}

/// Batch GEMV overwrite without bias using native f32 weights via AVX-512.
///
/// Unified kernel using broadcast-input / accumulator-output pattern
/// with masked tail via `_mm512_mask_storeu_ps`. Covers all shapes without
/// scalar fallback.
///
/// Strategy:
/// - `in_len == 1`: broadcast the single input, multiply weights
///   in blocks of 16 output channels, maskstore tail.
/// - `out_len == 1`: batch 16 frames per ZMM (8-accumulator deferred
///   horizontal reduction), with per-frame fallback for remainder.
/// - General: 8-way unrolled broadcast-input over output-channel blocks
///   of 16, with `_mm512_maskz_loadu_ps` + `_mm512_mask_storeu_ps` for the
///   final partial block when `out_len % 16 != 0`.
///
/// # Safety
/// - `num_frames` must be > 0.
/// - `in_frames.len()` must be a multiple of `num_frames`.
/// - `out_frames.len()` must be a multiple of `num_frames`.
/// - `weights.len()` must be >= `in_len * out_len` where `in_len = in_frames.len() / num_frames`
///   and `out_len = out_frames.len() / num_frames`.
///
/// All slices must be valid and accessible for reading/writing.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_no_bias_f32_avx512(
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

    // ── in_len == 1: out[j] = in[0] * weights[j] ────────────────────────
    if in_len == 1 {
        for n in 0..num_frames {
            let v_in = _mm512_set1_ps(*in_frames.get_unchecked(n));
            let mut oc = 0;
            while oc + 16 <= out_len {
                let v_w = _mm512_loadu_ps(weights.as_ptr().add(oc));
                let v_out = _mm512_mul_ps(v_in, v_w);
                _mm512_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), v_out);
                oc += 16;
            }
            if oc < out_len {
                let rem = out_len - oc;
                let mask: u16 = (1 << rem) - 1;
                let v_w = _mm512_maskz_loadu_ps(mask, weights.as_ptr().add(oc));
                let v_out = _mm512_mul_ps(v_in, v_w);
                _mm512_mask_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), mask, v_out);
            }
        }
        return;
    }

    // ── out_len == 1: batch 16 frames per ZMM ────────────────────────────
    if out_len == 1 {
        let mut n = 0;
        if in_len > SMALL_IN_LEN_THRESHOLD_AVX512 {
            while n + 16 <= num_frames {
                let mut acc0 = _mm512_setzero_ps();
                let mut acc1 = _mm512_setzero_ps();
                let mut acc2 = _mm512_setzero_ps();
                let mut acc3 = _mm512_setzero_ps();
                let mut acc4 = _mm512_setzero_ps();
                let mut acc5 = _mm512_setzero_ps();
                let mut acc6 = _mm512_setzero_ps();
                let mut acc7 = _mm512_setzero_ps();
                let mut ic = 0;
                while ic + 8 <= in_len {
                    let v_w0 = _mm512_set1_ps(*weights.get_unchecked(ic));
                    let v_w1 = _mm512_set1_ps(*weights.get_unchecked(ic + 1));
                    let v_w2 = _mm512_set1_ps(*weights.get_unchecked(ic + 2));
                    let v_w3 = _mm512_set1_ps(*weights.get_unchecked(ic + 3));
                    let v_w4 = _mm512_set1_ps(*weights.get_unchecked(ic + 4));
                    let v_w5 = _mm512_set1_ps(*weights.get_unchecked(ic + 5));
                    let v_w6 = _mm512_set1_ps(*weights.get_unchecked(ic + 6));
                    let v_w7 = _mm512_set1_ps(*weights.get_unchecked(ic + 7));
                    let ra0 = _mm256_loadu_ps(in_frames.as_ptr().add((n) * in_len + ic));
                    let ra1 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 1) * in_len + ic));
                    let ra2 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 2) * in_len + ic));
                    let ra3 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 3) * in_len + ic));
                    let ra4 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 4) * in_len + ic));
                    let ra5 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 5) * in_len + ic));
                    let ra6 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 6) * in_len + ic));
                    let ra7 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 7) * in_len + ic));
                    let rb0 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 8) * in_len + ic));
                    let rb1 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 9) * in_len + ic));
                    let rb2 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 10) * in_len + ic));
                    let rb3 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 11) * in_len + ic));
                    let rb4 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 12) * in_len + ic));
                    let rb5 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 13) * in_len + ic));
                    let rb6 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 14) * in_len + ic));
                    let rb7 = _mm256_loadu_ps(in_frames.as_ptr().add((n + 15) * in_len + ic));
                    // 16×8 matrix transpose: 16 frames (ra* = frames 0..7,
                    // rb* = frames 8..15) × 8 input channels at offset ic.
                    // Two independent shuffle trees (group a, group b) each
                    // transpose 8×8 via unpack → shuffle → permute2f128, then
                    // _mm512_insertf32x8 merges the transposed YMM halves into
                    // ZMM. Each v_in_k broadcasts channel k across all 16 frames.
                    let ta0 = _mm256_unpacklo_ps(ra0, ra1);
                    let ta1 = _mm256_unpackhi_ps(ra0, ra1);
                    let ta2 = _mm256_unpacklo_ps(ra2, ra3);
                    let ta3 = _mm256_unpackhi_ps(ra2, ra3);
                    let ta4 = _mm256_unpacklo_ps(ra4, ra5);
                    let ta5 = _mm256_unpackhi_ps(ra4, ra5);
                    let ta6 = _mm256_unpacklo_ps(ra6, ra7);
                    let ta7 = _mm256_unpackhi_ps(ra6, ra7);
                    let tb0 = _mm256_unpacklo_ps(rb0, rb1);
                    let tb1 = _mm256_unpackhi_ps(rb0, rb1);
                    let tb2 = _mm256_unpacklo_ps(rb2, rb3);
                    let tb3 = _mm256_unpackhi_ps(rb2, rb3);
                    let tb4 = _mm256_unpacklo_ps(rb4, rb5);
                    let tb5 = _mm256_unpackhi_ps(rb4, rb5);
                    let tb6 = _mm256_unpacklo_ps(rb6, rb7);
                    let tb7 = _mm256_unpackhi_ps(rb6, rb7);
                    let sa0 = _mm256_shuffle_ps(ta0, ta2, 0x44);
                    let sa1 = _mm256_shuffle_ps(ta0, ta2, 0xEE);
                    let sa2 = _mm256_shuffle_ps(ta1, ta3, 0x44);
                    let sa3 = _mm256_shuffle_ps(ta1, ta3, 0xEE);
                    let sa4 = _mm256_shuffle_ps(ta4, ta6, 0x44);
                    let sa5 = _mm256_shuffle_ps(ta4, ta6, 0xEE);
                    let sa6 = _mm256_shuffle_ps(ta5, ta7, 0x44);
                    let sa7 = _mm256_shuffle_ps(ta5, ta7, 0xEE);
                    let sb0 = _mm256_shuffle_ps(tb0, tb2, 0x44);
                    let sb1 = _mm256_shuffle_ps(tb0, tb2, 0xEE);
                    let sb2 = _mm256_shuffle_ps(tb1, tb3, 0x44);
                    let sb3 = _mm256_shuffle_ps(tb1, tb3, 0xEE);
                    let sb4 = _mm256_shuffle_ps(tb4, tb6, 0x44);
                    let sb5 = _mm256_shuffle_ps(tb4, tb6, 0xEE);
                    let sb6 = _mm256_shuffle_ps(tb5, tb7, 0x44);
                    let sb7 = _mm256_shuffle_ps(tb5, tb7, 0xEE);
                    let ca0 = _mm256_permute2f128_ps(sa0, sa4, 0x20);
                    let ca1 = _mm256_permute2f128_ps(sa1, sa5, 0x20);
                    let ca2 = _mm256_permute2f128_ps(sa2, sa6, 0x20);
                    let ca3 = _mm256_permute2f128_ps(sa3, sa7, 0x20);
                    let ca4 = _mm256_permute2f128_ps(sa0, sa4, 0x31);
                    let ca5 = _mm256_permute2f128_ps(sa1, sa5, 0x31);
                    let ca6 = _mm256_permute2f128_ps(sa2, sa6, 0x31);
                    let ca7 = _mm256_permute2f128_ps(sa3, sa7, 0x31);
                    let cb0 = _mm256_permute2f128_ps(sb0, sb4, 0x20);
                    let cb1 = _mm256_permute2f128_ps(sb1, sb5, 0x20);
                    let cb2 = _mm256_permute2f128_ps(sb2, sb6, 0x20);
                    let cb3 = _mm256_permute2f128_ps(sb3, sb7, 0x20);
                    let cb4 = _mm256_permute2f128_ps(sb0, sb4, 0x31);
                    let cb5 = _mm256_permute2f128_ps(sb1, sb5, 0x31);
                    let cb6 = _mm256_permute2f128_ps(sb2, sb6, 0x31);
                    let cb7 = _mm256_permute2f128_ps(sb3, sb7, 0x31);
                    let v_in0 = _mm512_insertf32x8(_mm512_castps256_ps512(ca0), cb0, 1);
                    let v_in1 = _mm512_insertf32x8(_mm512_castps256_ps512(ca1), cb1, 1);
                    let v_in2 = _mm512_insertf32x8(_mm512_castps256_ps512(ca2), cb2, 1);
                    let v_in3 = _mm512_insertf32x8(_mm512_castps256_ps512(ca3), cb3, 1);
                    let v_in4 = _mm512_insertf32x8(_mm512_castps256_ps512(ca4), cb4, 1);
                    let v_in5 = _mm512_insertf32x8(_mm512_castps256_ps512(ca5), cb5, 1);
                    let v_in6 = _mm512_insertf32x8(_mm512_castps256_ps512(ca6), cb6, 1);
                    let v_in7 = _mm512_insertf32x8(_mm512_castps256_ps512(ca7), cb7, 1);
                    acc0 = _mm512_fmadd_ps(v_in0, v_w0, acc0);
                    acc1 = _mm512_fmadd_ps(v_in1, v_w1, acc1);
                    acc2 = _mm512_fmadd_ps(v_in2, v_w2, acc2);
                    acc3 = _mm512_fmadd_ps(v_in3, v_w3, acc3);
                    acc4 = _mm512_fmadd_ps(v_in4, v_w4, acc4);
                    acc5 = _mm512_fmadd_ps(v_in5, v_w5, acc5);
                    acc6 = _mm512_fmadd_ps(v_in6, v_w6, acc6);
                    acc7 = _mm512_fmadd_ps(v_in7, v_w7, acc7);
                    ic += 8;
                }
                acc0 = _mm512_add_ps(acc0, acc1);
                acc2 = _mm512_add_ps(acc2, acc3);
                acc4 = _mm512_add_ps(acc4, acc5);
                acc6 = _mm512_add_ps(acc6, acc7);
                acc0 = _mm512_add_ps(acc0, acc2);
                acc4 = _mm512_add_ps(acc4, acc6);
                acc0 = _mm512_add_ps(acc0, acc4);
                while ic < in_len {
                    let v_w = _mm512_set1_ps(*weights.get_unchecked(ic));
                    let mut buf = [0.0f32; 16];
                    #[expect(
                        clippy::needless_range_loop,
                        reason = "Range loop required for explicit SIMD lane indexing not expressible via iterator"
                    )]
                    for j in 0..16 {
                        buf[j] = *in_frames.get_unchecked((n + j) * in_len + ic);
                    }
                    let v_in = _mm512_loadu_ps(buf.as_ptr());
                    acc0 = _mm512_fmadd_ps(v_in, v_w, acc0);
                    ic += 1;
                }
                _mm512_storeu_ps(out_frames.as_mut_ptr().add(n), acc0);
                n += 16;
            }
        }
        for n in n..num_frames {
            let mut acc = _mm512_setzero_ps();
            let mut ic = 0;
            while ic + 16 <= in_len {
                let v_in = _mm512_loadu_ps(in_frames.as_ptr().add(n * in_len + ic));
                let v_w = _mm512_loadu_ps(weights.as_ptr().add(ic));
                acc = _mm512_fmadd_ps(v_in, v_w, acc);
                ic += 16;
            }
            if ic < in_len {
                let rem = in_len - ic;
                let mask: u16 = (1 << rem) - 1;
                let v_in = _mm512_maskz_loadu_ps(mask, in_frames.as_ptr().add(n * in_len + ic));
                let v_w = _mm512_maskz_loadu_ps(mask, weights.as_ptr().add(ic));
                acc = _mm512_fmadd_ps(v_in, v_w, acc);
            }
            let mut tmp = [0.0f32; 16];
            _mm512_storeu_ps(tmp.as_mut_ptr(), acc);
            let sum: f32 = tmp.iter().sum();
            *out_frames.get_unchecked_mut(n) = sum;
        }
        return;
    }

    // ── General unified path: all out_len >= 1 ───────────────────────────
    for n in 0..num_frames {
        let mut oc = 0;
        while oc + 16 <= out_len {
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            let mut acc4 = _mm512_setzero_ps();
            let mut acc5 = _mm512_setzero_ps();
            let mut acc6 = _mm512_setzero_ps();
            let mut acc7 = _mm512_setzero_ps();
            let mut ic = 0;
            while ic + 8 <= in_len {
                let vs0 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vs1 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 1));
                let vs2 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 2));
                let vs3 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 3));
                let vs4 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 4));
                let vs5 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 5));
                let vs6 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 6));
                let vs7 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 7));
                let w_ptr = weights.as_ptr().add(ic * out_len + oc);
                let w0 = _mm512_loadu_ps(w_ptr);
                acc0 = _mm512_fmadd_ps(vs0, w0, acc0);
                let w1 = _mm512_loadu_ps(w_ptr.add(out_len));
                acc1 = _mm512_fmadd_ps(vs1, w1, acc1);
                let w2 = _mm512_loadu_ps(w_ptr.add(2 * out_len));
                acc2 = _mm512_fmadd_ps(vs2, w2, acc2);
                let w3 = _mm512_loadu_ps(w_ptr.add(3 * out_len));
                acc3 = _mm512_fmadd_ps(vs3, w3, acc3);
                let w4 = _mm512_loadu_ps(w_ptr.add(4 * out_len));
                acc4 = _mm512_fmadd_ps(vs4, w4, acc4);
                let w5 = _mm512_loadu_ps(w_ptr.add(5 * out_len));
                acc5 = _mm512_fmadd_ps(vs5, w5, acc5);
                let w6 = _mm512_loadu_ps(w_ptr.add(6 * out_len));
                acc6 = _mm512_fmadd_ps(vs6, w6, acc6);
                let w7 = _mm512_loadu_ps(w_ptr.add(7 * out_len));
                acc7 = _mm512_fmadd_ps(vs7, w7, acc7);
                ic += 8;
            }
            acc0 = _mm512_add_ps(acc0, acc1);
            acc2 = _mm512_add_ps(acc2, acc3);
            acc4 = _mm512_add_ps(acc4, acc5);
            acc6 = _mm512_add_ps(acc6, acc7);
            acc0 = _mm512_add_ps(acc0, acc2);
            acc4 = _mm512_add_ps(acc4, acc6);
            acc0 = _mm512_add_ps(acc0, acc4);
            let mut tail = [0.0f32; 16];
            while ic < in_len {
                let inp = *in_frames.get_unchecked(n * in_len + ic);
                let base_idx = ic * out_len + oc;
                for (j, t) in tail.iter_mut().enumerate() {
                    *t = f32::mul_add(inp, *weights.get_unchecked(base_idx + j), *t);
                }
                ic += 1;
            }
            acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(tail.as_ptr()));
            _mm512_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), acc0);
            oc += 16;
        }
        if oc < out_len {
            let rem = out_len - oc;
            let mask: u16 = (1 << rem) - 1;
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            let mut acc4 = _mm512_setzero_ps();
            let mut acc5 = _mm512_setzero_ps();
            let mut acc6 = _mm512_setzero_ps();
            let mut acc7 = _mm512_setzero_ps();
            let mut ic = 0;
            while ic + 8 <= in_len {
                let vs0 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vs1 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 1));
                let vs2 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 2));
                let vs3 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 3));
                let vs4 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 4));
                let vs5 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 5));
                let vs6 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 6));
                let vs7 = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic + 7));
                let w_ptr = weights.as_ptr().add(ic * out_len + oc);
                let w0 = _mm512_maskz_loadu_ps(mask, w_ptr);
                acc0 = _mm512_fmadd_ps(vs0, w0, acc0);
                let w1 = _mm512_maskz_loadu_ps(mask, w_ptr.add(out_len));
                acc1 = _mm512_fmadd_ps(vs1, w1, acc1);
                let w2 = _mm512_maskz_loadu_ps(mask, w_ptr.add(2 * out_len));
                acc2 = _mm512_fmadd_ps(vs2, w2, acc2);
                let w3 = _mm512_maskz_loadu_ps(mask, w_ptr.add(3 * out_len));
                acc3 = _mm512_fmadd_ps(vs3, w3, acc3);
                let w4 = _mm512_maskz_loadu_ps(mask, w_ptr.add(4 * out_len));
                acc4 = _mm512_fmadd_ps(vs4, w4, acc4);
                let w5 = _mm512_maskz_loadu_ps(mask, w_ptr.add(5 * out_len));
                acc5 = _mm512_fmadd_ps(vs5, w5, acc5);
                let w6 = _mm512_maskz_loadu_ps(mask, w_ptr.add(6 * out_len));
                acc6 = _mm512_fmadd_ps(vs6, w6, acc6);
                let w7 = _mm512_maskz_loadu_ps(mask, w_ptr.add(7 * out_len));
                acc7 = _mm512_fmadd_ps(vs7, w7, acc7);
                ic += 8;
            }
            acc0 = _mm512_add_ps(acc0, acc1);
            acc2 = _mm512_add_ps(acc2, acc3);
            acc4 = _mm512_add_ps(acc4, acc5);
            acc6 = _mm512_add_ps(acc6, acc7);
            acc0 = _mm512_add_ps(acc0, acc2);
            acc4 = _mm512_add_ps(acc4, acc6);
            acc0 = _mm512_add_ps(acc0, acc4);
            while ic < in_len {
                let vs = _mm512_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vw = _mm512_maskz_loadu_ps(mask, weights.as_ptr().add(ic * out_len + oc));
                acc0 = _mm512_fmadd_ps(vs, vw, acc0);
                ic += 1;
            }
            _mm512_mask_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), mask, acc0);
        }
    }
}
