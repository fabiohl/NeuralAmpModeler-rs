// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Fused Residual GEMM Batch kernels — AVX-512.
//!
//! Process multiple audio frames simultaneously for efficient weight reuse using 512-bit vector registers.

use core::arch::x86_64::*;

/// Fused residual GEMM kernel AVX-512.
///
/// # Safety
/// - `num_frames` must be > 0.
/// - `in_frames.len()` must be a multiple of `num_frames`.
/// - `out_frames.len()` must be a multiple of `num_frames`.
/// - `residual.len()` must be a multiple of `num_frames`.
/// - `weights.len()` must be >= `in_len * out_len` where `in_len = in_frames.len() / num_frames`
///   and `out_len = out_frames.len() / num_frames`.
/// - `residual.len()` must be >= `out_frames.len()`.
/// - If `do_bias` is true, `bias.len()` must be >= `out_len`.
///
/// All slices must be valid and accessible for reading/writing.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_gemm_residual_batch_avx512(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    residual: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    debug_assert_eq!(in_frames.len() % num_frames, 0);
    debug_assert_eq!(out_frames.len() % num_frames, 0);
    debug_assert_eq!(residual.len() % num_frames, 0);
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;
    debug_assert!(weights.len() >= in_len * out_len);
    debug_assert!(residual.len() >= out_frames.len());
    if do_bias {
        debug_assert!(bias.len() >= out_len);
    }

    let mut f = 0;
    while f + 8 <= num_frames {
        let mut out_c = 0;
        while out_c + 16 <= out_len {
            let mut acc0 = _mm512_loadu_ps(residual.as_ptr().add(f * out_len + out_c));
            let mut acc1 = _mm512_loadu_ps(residual.as_ptr().add((f + 1) * out_len + out_c));
            let mut acc2 = _mm512_loadu_ps(residual.as_ptr().add((f + 2) * out_len + out_c));
            let mut acc3 = _mm512_loadu_ps(residual.as_ptr().add((f + 3) * out_len + out_c));
            let mut acc4 = _mm512_loadu_ps(residual.as_ptr().add((f + 4) * out_len + out_c));
            let mut acc5 = _mm512_loadu_ps(residual.as_ptr().add((f + 5) * out_len + out_c));
            let mut acc6 = _mm512_loadu_ps(residual.as_ptr().add((f + 6) * out_len + out_c));
            let mut acc7 = _mm512_loadu_ps(residual.as_ptr().add((f + 7) * out_len + out_c));

            if do_bias {
                let b = _mm512_loadu_ps(bias.as_ptr().add(out_c));
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
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm512_loadu_ps(weight_ptr);

                acc0 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked(f * in_len + in_c)),
                    vw,
                    acc0,
                );
                acc1 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c)),
                    vw,
                    acc1,
                );
                acc2 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c)),
                    vw,
                    acc2,
                );
                acc3 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c)),
                    vw,
                    acc3,
                );
                acc4 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 4) * in_len + in_c)),
                    vw,
                    acc4,
                );
                acc5 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 5) * in_len + in_c)),
                    vw,
                    acc5,
                );
                acc6 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 6) * in_len + in_c)),
                    vw,
                    acc6,
                );
                acc7 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 7) * in_len + in_c)),
                    vw,
                    acc7,
                );
            }

            _mm512_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 4) * out_len + out_c), acc4);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 5) * out_len + out_c), acc5);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 6) * out_len + out_c), acc6);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 7) * out_len + out_c), acc7);
            out_c += 16;
        }

        while out_c < out_len {
            for i in 0..8 {
                let frame_idx = f + i;
                let mut sum = *residual.get_unchecked(frame_idx * out_len + out_c);
                if do_bias {
                    sum += *bias.get_unchecked(out_c);
                }
                for in_c in 0..in_len {
                    let w = *weights.get_unchecked(in_c * out_len + out_c);
                    sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                }
                *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum;
            }
            out_c += 1;
        }
        f += 8;
    }

    while f < num_frames {
        let in_frame = &in_frames[f * in_len..(f + 1) * in_len];
        let out_frame = &mut out_frames[f * out_len..(f + 1) * out_len];
        let res_frame = &residual[f * out_len..(f + 1) * out_len];

        let mut out_c = 0;
        while out_c + 16 <= out_len {
            let mut accum = _mm512_loadu_ps(res_frame.as_ptr().add(out_c));
            if do_bias {
                accum = _mm512_add_ps(accum, _mm512_loadu_ps(bias.as_ptr().add(out_c)));
            }
            for in_c in 0..in_len {
                let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm512_loadu_ps(weight_ptr);
                accum = _mm512_fmadd_ps(vs, vw, accum);
            }
            _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
            out_c += 16;
        }
        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            sum += res_frame[out_c];
            for in_c in 0..in_len {
                sum +=
                    *in_frame.get_unchecked(in_c) * *weights.get_unchecked(in_c * out_len + out_c);
            }
            *out_frame.get_unchecked_mut(out_c) = sum;
            out_c += 1;
        }
        f += 1;
    }
}

/// Fused residual GEMM kernel via AVX-512 with native f32 weights.
///
/// # Safety
/// Preconditions are identical to [`fused_gemm_residual_batch_avx512`].
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_gemm_residual_batch_f32_avx512(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    residual: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    debug_assert_eq!(in_frames.len() % num_frames, 0);
    debug_assert_eq!(out_frames.len() % num_frames, 0);
    debug_assert_eq!(residual.len() % num_frames, 0);
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;
    debug_assert!(weights.len() >= in_len * out_len);
    debug_assert!(residual.len() >= out_frames.len());
    if do_bias {
        debug_assert!(bias.len() >= out_len);
    }

    let mut f = 0;
    while f + 8 <= num_frames {
        let mut out_c = 0;
        while out_c + 16 <= out_len {
            let mut acc0 = _mm512_loadu_ps(residual.as_ptr().add(f * out_len + out_c));
            let mut acc1 = _mm512_loadu_ps(residual.as_ptr().add((f + 1) * out_len + out_c));
            let mut acc2 = _mm512_loadu_ps(residual.as_ptr().add((f + 2) * out_len + out_c));
            let mut acc3 = _mm512_loadu_ps(residual.as_ptr().add((f + 3) * out_len + out_c));
            let mut acc4 = _mm512_loadu_ps(residual.as_ptr().add((f + 4) * out_len + out_c));
            let mut acc5 = _mm512_loadu_ps(residual.as_ptr().add((f + 5) * out_len + out_c));
            let mut acc6 = _mm512_loadu_ps(residual.as_ptr().add((f + 6) * out_len + out_c));
            let mut acc7 = _mm512_loadu_ps(residual.as_ptr().add((f + 7) * out_len + out_c));

            if do_bias {
                let b = _mm512_loadu_ps(bias.as_ptr().add(out_c));
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
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm512_loadu_ps(weight_ptr);

                acc0 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked(f * in_len + in_c)),
                    vw,
                    acc0,
                );
                acc1 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c)),
                    vw,
                    acc1,
                );
                acc2 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c)),
                    vw,
                    acc2,
                );
                acc3 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c)),
                    vw,
                    acc3,
                );
                acc4 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 4) * in_len + in_c)),
                    vw,
                    acc4,
                );
                acc5 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 5) * in_len + in_c)),
                    vw,
                    acc5,
                );
                acc6 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 6) * in_len + in_c)),
                    vw,
                    acc6,
                );
                acc7 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 7) * in_len + in_c)),
                    vw,
                    acc7,
                );
            }

            _mm512_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 4) * out_len + out_c), acc4);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 5) * out_len + out_c), acc5);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 6) * out_len + out_c), acc6);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 7) * out_len + out_c), acc7);
            out_c += 16;
        }

        while out_c < out_len {
            for i in 0..8 {
                let frame_idx = f + i;
                let mut sum = *residual.get_unchecked(frame_idx * out_len + out_c);
                if do_bias {
                    sum += *bias.get_unchecked(out_c);
                }
                for in_c in 0..in_len {
                    let w = *weights.get_unchecked(in_c * out_len + out_c);
                    sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                }
                *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum;
            }
            out_c += 1;
        }
        f += 8;
    }

    while f < num_frames {
        let in_frame = &in_frames[f * in_len..(f + 1) * in_len];
        let out_frame = &mut out_frames[f * out_len..(f + 1) * out_len];
        let res_frame = &residual[f * out_len..(f + 1) * out_len];

        let mut out_c = 0;
        while out_c + 16 <= out_len {
            let mut accum = _mm512_loadu_ps(res_frame.as_ptr().add(out_c));
            if do_bias {
                accum = _mm512_add_ps(accum, _mm512_loadu_ps(bias.as_ptr().add(out_c)));
            }
            for in_c in 0..in_len {
                let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm512_loadu_ps(weight_ptr);
                accum = _mm512_fmadd_ps(vs, vw, accum);
            }
            _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
            out_c += 16;
        }
        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            sum += res_frame[out_c];
            for in_c in 0..in_len {
                sum +=
                    *in_frame.get_unchecked(in_c) * *weights.get_unchecked(in_c * out_len + out_c);
            }
            *out_frame.get_unchecked_mut(out_c) = sum;
            out_c += 1;
        }
        f += 1;
    }
}
