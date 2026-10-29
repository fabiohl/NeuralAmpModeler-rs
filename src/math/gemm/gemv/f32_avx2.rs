// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::arch::simd::*;
use crate::gemv_f32_inner_loop_avx2;

const SMALL_IN_LEN_THRESHOLD_AVX2: usize = 4;

// Lane 0..7 summed in memory order, matching the previous `iter().sum()`
// reduction without materializing the vector on the stack.
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn sequential_lane_sum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let mut sum = _mm_cvtss_f32(lo);
    sum += _mm_cvtss_f32(_mm_shuffle_ps(lo, lo, 0x55));
    sum += _mm_cvtss_f32(_mm_shuffle_ps(lo, lo, 0xAA));
    sum += _mm_cvtss_f32(_mm_shuffle_ps(lo, lo, 0xFF));
    sum += _mm_cvtss_f32(hi);
    sum += _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, 0x55));
    sum += _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, 0xAA));
    sum += _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, 0xFF));
    sum
}

// ── Partial output-channel tail ───────────────────────────────────────────────
//
// The last `out_len % 8` lanes are accumulated with four 128-bit lanes plus
// scalar lanes for the remainder. Both paths keep the same per-lane summation
// order as the 8-accumulator AVX2 loop, so the result is bit-identical to the
// previous masked implementation while avoiding `_mm256_maskload_ps`,
// `_mm256_maskstore_ps` and the scratch-mask `memset`/`memcpy` they induced.
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn gemv_tail_f32<const WITH_BIAS: bool, const BROADCAST: bool>(
    in_frame: *const f32,
    weights: *const f32,
    bias: *const f32,
    out: *mut f32,
    in_len: usize,
    out_len: usize,
    oc: usize,
) {
    let rem = out_len - oc;

    // `in_len == 1`: every output lane is a single fused multiply-add.
    if BROADCAST {
        let v_in = _mm_set1_ps(*in_frame);
        if rem >= 4 {
            let v_w = _mm_loadu_ps(weights.add(oc));
            let v_out = if WITH_BIAS {
                _mm_fmadd_ps(v_in, v_w, _mm_loadu_ps(bias.add(oc)))
            } else {
                _mm_mul_ps(v_in, v_w)
            };
            _mm_storeu_ps(out.add(oc), v_out);
        }
        let mut lane = if rem >= 4 { 4 } else { 0 };
        while lane < rem {
            let w = *weights.add(oc + lane);
            *out.add(oc + lane) = if WITH_BIAS {
                f32::mul_add(*in_frame, w, *bias.add(oc + lane))
            } else {
                *in_frame * w
            };
            lane += 1;
        }
        return;
    }

    let mut lane = 0;
    if rem >= 4 {
        let mut acc0 = if WITH_BIAS {
            _mm_loadu_ps(bias.add(oc))
        } else {
            _mm_setzero_ps()
        };
        let mut acc1 = _mm_setzero_ps();
        let mut acc2 = _mm_setzero_ps();
        let mut acc3 = _mm_setzero_ps();
        let mut acc4 = _mm_setzero_ps();
        let mut acc5 = _mm_setzero_ps();
        let mut acc6 = _mm_setzero_ps();
        let mut acc7 = _mm_setzero_ps();
        let mut ic = 0;
        while ic + 8 <= in_len {
            let w_ptr = weights.add(ic * out_len + oc);
            acc0 = _mm_fmadd_ps(_mm_set1_ps(*in_frame.add(ic)), _mm_loadu_ps(w_ptr), acc0);
            acc1 = _mm_fmadd_ps(
                _mm_set1_ps(*in_frame.add(ic + 1)),
                _mm_loadu_ps(w_ptr.add(out_len)),
                acc1,
            );
            acc2 = _mm_fmadd_ps(
                _mm_set1_ps(*in_frame.add(ic + 2)),
                _mm_loadu_ps(w_ptr.add(2 * out_len)),
                acc2,
            );
            acc3 = _mm_fmadd_ps(
                _mm_set1_ps(*in_frame.add(ic + 3)),
                _mm_loadu_ps(w_ptr.add(3 * out_len)),
                acc3,
            );
            acc4 = _mm_fmadd_ps(
                _mm_set1_ps(*in_frame.add(ic + 4)),
                _mm_loadu_ps(w_ptr.add(4 * out_len)),
                acc4,
            );
            acc5 = _mm_fmadd_ps(
                _mm_set1_ps(*in_frame.add(ic + 5)),
                _mm_loadu_ps(w_ptr.add(5 * out_len)),
                acc5,
            );
            acc6 = _mm_fmadd_ps(
                _mm_set1_ps(*in_frame.add(ic + 6)),
                _mm_loadu_ps(w_ptr.add(6 * out_len)),
                acc6,
            );
            acc7 = _mm_fmadd_ps(
                _mm_set1_ps(*in_frame.add(ic + 7)),
                _mm_loadu_ps(w_ptr.add(7 * out_len)),
                acc7,
            );
            ic += 8;
        }
        acc0 = _mm_add_ps(acc0, acc1);
        acc2 = _mm_add_ps(acc2, acc3);
        acc4 = _mm_add_ps(acc4, acc5);
        acc6 = _mm_add_ps(acc6, acc7);
        acc0 = _mm_add_ps(acc0, acc2);
        acc4 = _mm_add_ps(acc4, acc6);
        acc0 = _mm_add_ps(acc0, acc4);
        while ic < in_len {
            acc0 = _mm_fmadd_ps(
                _mm_set1_ps(*in_frame.add(ic)),
                _mm_loadu_ps(weights.add(ic * out_len + oc)),
                acc0,
            );
            ic += 1;
        }
        _mm_storeu_ps(out.add(oc), acc0);
        lane = 4;
    }

    // Lanes past the 4-wide block, with the same 8-chain order per lane.
    while lane < rem {
        let mut acc0 = if WITH_BIAS { *bias.add(oc + lane) } else { 0.0 };
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;
        let mut acc4 = 0.0f32;
        let mut acc5 = 0.0f32;
        let mut acc6 = 0.0f32;
        let mut acc7 = 0.0f32;
        let mut ic = 0;
        while ic + 8 <= in_len {
            let w = weights.add(ic * out_len + oc + lane);
            acc0 = f32::mul_add(*in_frame.add(ic), *w, acc0);
            acc1 = f32::mul_add(*in_frame.add(ic + 1), *w.add(out_len), acc1);
            acc2 = f32::mul_add(*in_frame.add(ic + 2), *w.add(2 * out_len), acc2);
            acc3 = f32::mul_add(*in_frame.add(ic + 3), *w.add(3 * out_len), acc3);
            acc4 = f32::mul_add(*in_frame.add(ic + 4), *w.add(4 * out_len), acc4);
            acc5 = f32::mul_add(*in_frame.add(ic + 5), *w.add(5 * out_len), acc5);
            acc6 = f32::mul_add(*in_frame.add(ic + 6), *w.add(6 * out_len), acc6);
            acc7 = f32::mul_add(*in_frame.add(ic + 7), *w.add(7 * out_len), acc7);
            ic += 8;
        }
        acc0 += acc1;
        acc2 += acc3;
        acc4 += acc5;
        acc6 += acc7;
        acc0 += acc2;
        acc4 += acc6;
        acc0 += acc4;
        while ic < in_len {
            acc0 = f32::mul_add(
                *in_frame.add(ic),
                *weights.add(ic * out_len + oc + lane),
                acc0,
            );
            ic += 1;
        }
        *out.add(oc + lane) = acc0;
        lane += 1;
    }
}

// ── Single-output dot product, four frames per tile ──────────────────────────
//
// `out_len == 1` transposes four frames by eight channels into four YMM
// accumulators, each packing two channel chains (`k` in the low half, `k + 4`
// in the high half). The pairwise reduction below reproduces the 8-chain
// balanced tree exactly, so each frame keeps its previous summation order.
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn gemv_out1_tile4<const WITH_BIAS: bool>(
    in_frames: *const f32,
    weights: *const f32,
    bias: f32,
    out: *mut f32,
    in_len: usize,
    n: usize,
) {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut ic = 0;
    while ic + 8 <= in_len {
        let w0 = *weights.add(ic);
        let w1 = *weights.add(ic + 1);
        let w2 = *weights.add(ic + 2);
        let w3 = *weights.add(ic + 3);
        let w4 = *weights.add(ic + 4);
        let w5 = *weights.add(ic + 5);
        let w6 = *weights.add(ic + 6);
        let w7 = *weights.add(ic + 7);

        let row0 = _mm256_loadu_ps(in_frames.add(n * in_len + ic));
        let row1 = _mm256_loadu_ps(in_frames.add((n + 1) * in_len + ic));
        let row2 = _mm256_loadu_ps(in_frames.add((n + 2) * in_len + ic));
        let row3 = _mm256_loadu_ps(in_frames.add((n + 3) * in_len + ic));

        let t0 = _mm256_unpacklo_ps(row0, row1);
        let t1 = _mm256_unpackhi_ps(row0, row1);
        let t2 = _mm256_unpacklo_ps(row2, row3);
        let t3 = _mm256_unpackhi_ps(row2, row3);

        // Low 128 bits: channel k across the four frames. High 128 bits:
        // channel k + 4. Each register is consumed by the FMA that follows.
        let s0 = _mm256_shuffle_ps(t0, t2, 0x44);
        let s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        let s2 = _mm256_shuffle_ps(t1, t3, 0x44);
        let s3 = _mm256_shuffle_ps(t1, t3, 0xEE);

        acc0 = _mm256_fmadd_ps(s0, _mm256_set_m128(_mm_set1_ps(w4), _mm_set1_ps(w0)), acc0);
        acc1 = _mm256_fmadd_ps(s1, _mm256_set_m128(_mm_set1_ps(w5), _mm_set1_ps(w1)), acc1);
        acc2 = _mm256_fmadd_ps(s2, _mm256_set_m128(_mm_set1_ps(w6), _mm_set1_ps(w2)), acc2);
        acc3 = _mm256_fmadd_ps(s3, _mm256_set_m128(_mm_set1_ps(w7), _mm_set1_ps(w3)), acc3);
        ic += 8;
    }

    // P0 = [a0 | a4], P1 = [a1 | a5], P2 = [a2 | a6], P3 = [a3 | a7].
    // (P0 + P1) + (P2 + P3) yields [((a0+a1)+(a2+a3)) | ((a4+a5)+(a6+a7))];
    // the final half sum matches the original tree.
    let p01 = _mm256_add_ps(acc0, acc1);
    let p23 = _mm256_add_ps(acc2, acc3);
    let packed = _mm256_add_ps(p01, p23);
    let mut res = _mm_add_ps(
        _mm256_castps256_ps128(packed),
        _mm256_extractf128_ps(packed, 1),
    );
    while ic < in_len {
        let v_in = _mm_setr_ps(
            *in_frames.add(n * in_len + ic),
            *in_frames.add((n + 1) * in_len + ic),
            *in_frames.add((n + 2) * in_len + ic),
            *in_frames.add((n + 3) * in_len + ic),
        );
        res = _mm_fmadd_ps(v_in, _mm_set1_ps(*weights.add(ic)), res);
        ic += 1;
    }
    if WITH_BIAS {
        res = _mm_add_ps(res, _mm_set1_ps(bias));
    }
    _mm_storeu_ps(out.add(n), res);
}

// ── Batched f32 GEMV ──────────────────────────────────────────────────────────

/// Batch GEMV overwrite with bias using native f32 weights via AVX2.
///
/// Unified kernel using broadcast-input / accumulator-output pattern with a
/// mask-free tail (4-wide SIMD plus scalar lanes). Covers all shapes without
/// scalar fallback.
///
/// Strategy:
/// - `in_len == 1`: broadcast the single input, multiply-add weights and bias
///   in blocks of 8 output channels, mask-free tail.
/// - `out_len == 1`: two 4-frame tiles per 8-frame block (8-wide dot product
///   deferred horizontal reduction), with per-frame fallback for remainder.
/// - General: 8-way unrolled broadcast-input over output-channel blocks
///   of 8, with the mask-free tail for the final partial block when
///   `out_len % 8 != 0`.
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
    debug_assert_eq!(in_frames.len(), num_frames * in_len);
    debug_assert_eq!(out_frames.len(), num_frames * out_len);
    debug_assert!(weights.len() >= in_len * out_len);
    debug_assert!(bias.len() >= out_len);

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
                gemv_tail_f32::<true, true>(
                    in_frames.as_ptr().add(n),
                    weights.as_ptr(),
                    bias.as_ptr(),
                    out_frames.as_mut_ptr().add(n * out_len),
                    1,
                    out_len,
                    oc,
                );
            }
        }
        return;
    }

    // ── out_len == 1: two 4-frame tiles per 8-frame block ───────────────
    // When projecting down to a single output channel (e.g. WaveNet head arrays or
    // linear projections), processing one frame at a time would underutilize the 8-lane
    // 256-bit AVX2 registers. We therefore batch 8 consecutive audio frames across the SIMD lanes.
    if out_len == 1 {
        let mut n = 0;
        if in_len > SMALL_IN_LEN_THRESHOLD_AVX2 {
            // Preserve the previous 8-frame block partition: each block is two
            // independent 4-frame tiles, so trailing frames keep the legacy
            // per-frame fallback below.
            while n + 8 <= num_frames {
                gemv_out1_tile4::<true>(
                    in_frames.as_ptr(),
                    weights.as_ptr(),
                    *bias.get_unchecked(0),
                    out_frames.as_mut_ptr(),
                    in_len,
                    n,
                );
                gemv_out1_tile4::<true>(
                    in_frames.as_ptr(),
                    weights.as_ptr(),
                    *bias.get_unchecked(0),
                    out_frames.as_mut_ptr(),
                    in_len,
                    n + 4,
                );
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
                let in_at = |i: usize| {
                    if i < rem {
                        *in_frames.get_unchecked(n * in_len + ic + i)
                    } else {
                        0.0
                    }
                };
                let w_at = |i: usize| {
                    if i < rem {
                        *weights.get_unchecked(ic + i)
                    } else {
                        0.0
                    }
                };
                let v_in = _mm256_setr_ps(
                    in_at(0),
                    in_at(1),
                    in_at(2),
                    in_at(3),
                    in_at(4),
                    in_at(5),
                    in_at(6),
                    in_at(7),
                );
                let v_w = _mm256_setr_ps(
                    w_at(0),
                    w_at(1),
                    w_at(2),
                    w_at(3),
                    w_at(4),
                    w_at(5),
                    w_at(6),
                    w_at(7),
                );
                acc = _mm256_fmadd_ps(v_in, v_w, acc);
            }
            let sum = sequential_lane_sum(acc);
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
            let mut tail = _mm256_setzero_ps();
            while ic < in_len {
                let inp = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vw = _mm256_loadu_ps(weights.as_ptr().add(ic * out_len + oc));
                tail = _mm256_fmadd_ps(inp, vw, tail);
                ic += 1;
            }
            acc0 = _mm256_add_ps(acc0, tail);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), acc0);
            oc += 8;
        }
        if oc < out_len {
            gemv_tail_f32::<true, false>(
                in_frames.as_ptr().add(n * in_len),
                weights.as_ptr(),
                bias.as_ptr(),
                out_frames.as_mut_ptr().add(n * out_len),
                in_len,
                out_len,
                oc,
            );
        }
    }
}

/// Batch GEMV overwrite without bias using native f32 weights via AVX2.
///
/// Unified kernel using broadcast-input / accumulator-output pattern with a
/// mask-free tail (4-wide SIMD plus scalar lanes). Covers all shapes without
/// scalar fallback.
///
/// Strategy:
/// - `in_len == 1`: broadcast the single input, multiply-add weights
///   in blocks of 8 output channels, mask-free tail.
/// - `out_len == 1`: two 4-frame tiles per 8-frame block (8-wide dot product
///   deferred horizontal reduction), with per-frame fallback for remainder.
/// - General: 8-way unrolled broadcast-input over output-channel blocks
///   of 8, with the mask-free tail for the final partial block when
///   `out_len % 8 != 0`.
///
/// # Safety
/// - `num_frames` must be > 0.
/// - `in_frames.len()` must be a multiple of `num_frames`.
/// - `out_frames.len()` must be a multiple of `num_frames`.
/// - `weights.len()` must be >= `in_len * out_len` where `in_len = in_frames.len() / num_frames`
///   and `out_len = out_frames.len() / num_frames`.
///
/// All slices must be valid and accessible for reading/writing.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
    debug_assert_eq!(in_frames.len(), num_frames * in_len);
    debug_assert_eq!(out_frames.len(), num_frames * out_len);
    debug_assert!(weights.len() >= in_len * out_len);

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
                gemv_tail_f32::<false, true>(
                    in_frames.as_ptr().add(n),
                    weights.as_ptr(),
                    core::ptr::null(),
                    out_frames.as_mut_ptr().add(n * out_len),
                    1,
                    out_len,
                    oc,
                );
            }
        }
        return;
    }

    // ── out_len == 1: two 4-frame tiles per 8-frame block ───────────────
    if out_len == 1 {
        let mut n = 0;
        if in_len > SMALL_IN_LEN_THRESHOLD_AVX2 {
            // Preserve the previous 8-frame block partition: each block is two
            // independent 4-frame tiles, so trailing frames keep the legacy
            // per-frame fallback below.
            while n + 8 <= num_frames {
                gemv_out1_tile4::<false>(
                    in_frames.as_ptr(),
                    weights.as_ptr(),
                    0.0,
                    out_frames.as_mut_ptr(),
                    in_len,
                    n,
                );
                gemv_out1_tile4::<false>(
                    in_frames.as_ptr(),
                    weights.as_ptr(),
                    0.0,
                    out_frames.as_mut_ptr(),
                    in_len,
                    n + 4,
                );
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
                let in_at = |i: usize| {
                    if i < rem {
                        *in_frames.get_unchecked(n * in_len + ic + i)
                    } else {
                        0.0
                    }
                };
                let w_at = |i: usize| {
                    if i < rem {
                        *weights.get_unchecked(ic + i)
                    } else {
                        0.0
                    }
                };
                let v_in = _mm256_setr_ps(
                    in_at(0),
                    in_at(1),
                    in_at(2),
                    in_at(3),
                    in_at(4),
                    in_at(5),
                    in_at(6),
                    in_at(7),
                );
                let v_w = _mm256_setr_ps(
                    w_at(0),
                    w_at(1),
                    w_at(2),
                    w_at(3),
                    w_at(4),
                    w_at(5),
                    w_at(6),
                    w_at(7),
                );
                acc = _mm256_fmadd_ps(v_in, v_w, acc);
            }
            let sum = sequential_lane_sum(acc);
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
            let mut tail = _mm256_setzero_ps();
            while ic < in_len {
                let inp = _mm256_set1_ps(*in_frames.get_unchecked(n * in_len + ic));
                let vw = _mm256_loadu_ps(weights.as_ptr().add(ic * out_len + oc));
                tail = _mm256_fmadd_ps(inp, vw, tail);
                ic += 1;
            }
            acc0 = _mm256_add_ps(acc0, tail);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add(n * out_len + oc), acc0);
            oc += 8;
        }
        if oc < out_len {
            gemv_tail_f32::<false, false>(
                in_frames.as_ptr().add(n * in_len),
                weights.as_ptr(),
                core::ptr::null(),
                out_frames.as_mut_ptr().add(n * out_len),
                in_len,
                out_len,
                oc,
            );
        }
    }
}

/// Outer product: scalar x weight vector, without bias, for `in_len == 1`.
///
/// For each frame `n`: `out[n*OUT + oc] = in[n] * weights[oc]`.
///
/// Const-generic on `OUT` enables immediate addressing in the inner loop,
/// eliminating `leaq` dependency chains and the `in_len = in_frames.len()/num_frames` division.
///
/// # Safety
/// `in_frames.len() * OUT == out_frames.len()`
/// `weights.len() >= OUT`
/// AVX2+FMA ISA must be available (x86-64-v3).
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
            gemv_tail_f32::<false, true>(
                in_frames.as_ptr().add(n),
                weights.as_ptr(),
                core::ptr::null(),
                out_frames.as_mut_ptr().add(n * OUT),
                1,
                OUT,
                oc,
            );
        }
    }
}

/// Outer product: scalar x weight vector, with bias, for `in_len == 1`.
///
/// For each frame `n`: `out[n*OUT + oc] = bias[oc] + in[n] * weights[oc]`.
///
/// # Safety
/// `in_frames.len() * OUT == out_frames.len()`
/// `weights.len() >= OUT`, `bias.len() >= OUT`
/// AVX2+FMA ISA must be available (x86-64-v3).
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
            gemv_tail_f32::<true, true>(
                in_frames.as_ptr().add(n),
                weights.as_ptr(),
                bias.as_ptr(),
                out_frames.as_mut_ptr().add(n * OUT),
                1,
                OUT,
                oc,
            );
        }
    }
}
