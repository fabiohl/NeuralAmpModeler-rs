// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// Specialized GEMM micro-kernels utilize unsafe SIMD intrinsics and parameter lists.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

//! Batch GEMM kernels — AVX-512.
//!
//! Process multiple audio frames simultaneously for efficient weight reuse.

use super::super::gemv::fused_add_gemv_avx512;
use core::arch::x86_64::*;

/// Batch version of the fused operation Y = X_res + Bias + W * Z via AVX-512.
///
/// # Safety
/// - `num_frames` must be > 0.
/// - `in_frames.len()` must be a multiple of `num_frames`.
/// - `out_frames.len()` must be a multiple of `num_frames`.
/// - `weights.len()` must be >= `in_len * out_len` where `in_len = in_frames.len() / num_frames`
///   and `out_len = out_frames.len() / num_frames`.
/// - If `do_bias` is true, `bias.len()` must be >= `out_len`.
/// - The CPU must support the AVX-512F+VL instructions used here. Reachable
///   AVX-512 dispatch checks the full F+VL+BW+DQ matrix or uses a validated test
///   override; `#[target_feature]` is not a runtime CPU check.
///
/// All slices must be valid and accessible for reading/writing.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemm_batch_avx512(
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

    let mut f = 0;
    while f + 8 <= num_frames {
        let mut out_c = 0;
        while out_c + 16 <= out_len {
            // SAFETY: `out_c + 16 <= out_len` and `f + 8 <= num_frames`, so each
            // row base `frame * out_len + out_c` addresses 16 in-bounds
            // initialized f32 values of `out_frames`; `loadu` needs no alignment,
            // and the input/output slice references are exclusive as required.
            // The `# Safety` contract requires `out_frames.len()` to be an exact multiple of
            // `num_frames` (hence `num_frames * out_len == out_frames.len()`),
            // and the loop guards bound `frame`/`out_c`. ISA: AVX-512F+VL,
            // reachable only after `has_full_avx512()` / a validated override;
            // `#[target_feature]` is not a runtime check.
            let (mut acc0, mut acc1, mut acc2, mut acc3, mut acc4, mut acc5, mut acc6, mut acc7) = unsafe {
                (
                    _mm512_loadu_ps(out_frames.as_ptr().add(f * out_len + out_c)),
                    _mm512_loadu_ps(out_frames.as_ptr().add((f + 1) * out_len + out_c)),
                    _mm512_loadu_ps(out_frames.as_ptr().add((f + 2) * out_len + out_c)),
                    _mm512_loadu_ps(out_frames.as_ptr().add((f + 3) * out_len + out_c)),
                    _mm512_loadu_ps(out_frames.as_ptr().add((f + 4) * out_len + out_c)),
                    _mm512_loadu_ps(out_frames.as_ptr().add((f + 5) * out_len + out_c)),
                    _mm512_loadu_ps(out_frames.as_ptr().add((f + 6) * out_len + out_c)),
                    _mm512_loadu_ps(out_frames.as_ptr().add((f + 7) * out_len + out_c)),
                )
            };

            if do_bias {
                // SAFETY: `do_bias` and the `# Safety` contract imply
                // `bias.len() >= out_len`, and `out_c + 16 <= out_len`. `bias`
                // contains initialized f32 values and `loadu` accepts unaligned
                // pointers. ISA: AVX-512F+VL, gated by full AVX-512 detection or
                // validated override; `#[target_feature]` does not check CPU.
                let b = unsafe { _mm512_loadu_ps(bias.as_ptr().add(out_c)) };
                acc0 = _mm512_add_ps(acc0, b);
                acc1 = _mm512_add_ps(acc1, b);
                acc2 = _mm512_add_ps(acc2, b);
                acc3 = _mm512_add_ps(acc3, b);
                acc4 = _mm512_add_ps(acc4, b);
                acc5 = _mm512_add_ps(acc5, b);
                acc6 = _mm512_add_ps(acc6, b);
                acc7 = _mm512_add_ps(acc7, b);
            }

            for in_c in 0..in_len {
                // SAFETY: `in_c < in_len` and `out_c + 16 <= out_len`, so the
                // weight row addresses 16 in-bounds values (`weights.len() >=
                // in_len * out_len` per the `# Safety` contract). The `# Safety`
                // contract also requires `in_frames.len()` to be an exact
                // multiple of `num_frames`, so `num_frames * in_len ==
                // in_frames.len()` and each `(f + k) * in_len + in_c`
                // (`k <= 7`, guarded by `f + 8 <= num_frames`) is in bounds.
                // ISA: AVX-512F+VL; dispatch is gated by `has_full_avx512()` /
                // a validated override, not by `#[target_feature]` alone.
                let (vw, vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7) = unsafe {
                    (
                        _mm512_loadu_ps(weights.as_ptr().add(in_c * out_len + out_c)),
                        _mm512_set1_ps(*in_frames.get_unchecked(f * in_len + in_c)),
                        _mm512_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c)),
                        _mm512_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c)),
                        _mm512_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c)),
                        _mm512_set1_ps(*in_frames.get_unchecked((f + 4) * in_len + in_c)),
                        _mm512_set1_ps(*in_frames.get_unchecked((f + 5) * in_len + in_c)),
                        _mm512_set1_ps(*in_frames.get_unchecked((f + 6) * in_len + in_c)),
                        _mm512_set1_ps(*in_frames.get_unchecked((f + 7) * in_len + in_c)),
                    )
                };

                acc0 = _mm512_fmadd_ps(vs0, vw, acc0);
                acc1 = _mm512_fmadd_ps(vs1, vw, acc1);
                acc2 = _mm512_fmadd_ps(vs2, vw, acc2);
                acc3 = _mm512_fmadd_ps(vs3, vw, acc3);
                acc4 = _mm512_fmadd_ps(vs4, vw, acc4);
                acc5 = _mm512_fmadd_ps(vs5, vw, acc5);
                acc6 = _mm512_fmadd_ps(vs6, vw, acc6);
                acc7 = _mm512_fmadd_ps(vs7, vw, acc7);
            }

            // SAFETY: same extents as the loads above (caller-contract exact
            // multiples + loop guards); each row base addresses 16 in-bounds
            // mutable f32 values of `out_frames`. ISA: AVX-512F+VL as above.
            unsafe {
                _mm512_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
                _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
                _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
                _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
                _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 4) * out_len + out_c), acc4);
                _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 5) * out_len + out_c), acc5);
                _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 6) * out_len + out_c), acc6);
                _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 7) * out_len + out_c), acc7);
            }
            out_c += 16;
        }

        while out_c < out_len {
            for i in 0..8 {
                let frame_idx = f + i;
                // SAFETY: `frame_idx < f + 8 <= num_frames` and `out_c < out_len`;
                // the `# Safety` exact-multiple precondition + loop guards bound
                // `out_frames`.
                let mut sum = unsafe { *out_frames.get_unchecked(frame_idx * out_len + out_c) };
                if do_bias {
                    // SAFETY: `do_bias` and the `# Safety` contract imply
                    // `bias.len() >= out_len`, and `out_c < out_len`.
                    sum += unsafe { *bias.get_unchecked(out_c) };
                }
                for in_c in 0..in_len {
                    // SAFETY: `in_c < in_len`, `out_c < out_len`, and
                    // `weights.len() >= in_len * out_len` per the `# Safety`
                    // contract; ISA: AVX-512F+VL as above.
                    let w = unsafe { *weights.get_unchecked(in_c * out_len + out_c) };
                    // SAFETY: `frame_idx < num_frames`, `in_c < in_len`, and the
                    // documented exact-multiple precondition makes
                    // `in_frames.len() == num_frames * in_len`.
                    sum += unsafe { *in_frames.get_unchecked(frame_idx * in_len + in_c) } * w;
                }
                // SAFETY: `frame_idx < num_frames` and `out_c < out_len`, so the
                // destination is in bounds of `out_frames`.
                unsafe { *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum };
            }
            out_c += 1;
        }
        f += 8;
    }

    while f < num_frames {
        // SAFETY: `f < num_frames`, so the frame slices are in bounds of
        // `in_frames`/`out_frames` under the documented exact-multiple
        // precondition; the caller's `# Safety` contract supplies the
        // `weights`/`bias`/`do_bias` preconditions for the callee. ISA:
        // AVX-512F+VL dispatch guarantee as above; no runtime check here.
        unsafe {
            fused_add_gemv_avx512(
                in_frames.get_unchecked(f * in_len..(f + 1) * in_len),
                weights,
                bias,
                out_frames.get_unchecked_mut(f * out_len..(f + 1) * out_len),
                do_bias,
            );
        }
        f += 1;
    }
}
