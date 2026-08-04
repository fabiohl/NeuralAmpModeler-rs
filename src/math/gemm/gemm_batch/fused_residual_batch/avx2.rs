// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Fused Residual GEMM Batch kernels — AVX2.
//!
//! Process multiple audio frames simultaneously for efficient weight reuse.

use crate::gemm_batch_frame_loop_avx2;
use crate::gemm_batch_inner_dual_avx2;
use crate::gemm_batch_outc_loop_avx2;
use core::arch::x86_64::*;

/// Fused residual GEMM kernel via AVX2.
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
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_gemm_residual_batch_avx2(
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
    assert!(in_frames.len() == num_frames * in_len);
    assert!(out_frames.len() == num_frames * out_len);
    assert!(weights.len() >= in_len * out_len);
    assert!(residual.len() >= num_frames * out_len);
    if do_bias {
        assert!(bias.len() >= out_len);
    }

    let mut f = 0;
    gemm_batch_frame_loop_avx2!(
        f,
        num_frames,
        {
            let mut out_c = 0;
            gemm_batch_outc_loop_avx2!(
                out_c,
                out_len,
                {
                    let res0 = _mm256_loadu_ps(residual.as_ptr().add(f * out_len + out_c));
                    let res1 = _mm256_loadu_ps(residual.as_ptr().add((f + 1) * out_len + out_c));
                    let res2 = _mm256_loadu_ps(residual.as_ptr().add((f + 2) * out_len + out_c));
                    let res3 = _mm256_loadu_ps(residual.as_ptr().add((f + 3) * out_len + out_c));

                    let b = if do_bias {
                        _mm256_loadu_ps(bias.as_ptr().add(out_c))
                    } else {
                        _mm256_setzero_ps()
                    };

                    let mut acc0_lo = _mm256_add_ps(res0, b);
                    let mut acc0_hi = _mm256_setzero_ps();
                    let mut acc1_lo = _mm256_add_ps(res1, b);
                    let mut acc1_hi = _mm256_setzero_ps();
                    let mut acc2_lo = _mm256_add_ps(res2, b);
                    let mut acc2_hi = _mm256_setzero_ps();
                    let mut acc3_lo = _mm256_add_ps(res3, b);
                    let mut acc3_hi = _mm256_setzero_ps();

                    let mut in_c = 0;
                    gemm_batch_inner_dual_avx2!(
                        in_c,
                        in_len,
                        {
                            let wp_lo = weights.as_ptr().add(in_c * out_len + out_c);
                            let vw_lo = _mm256_loadu_ps(wp_lo);
                            let wp_hi = weights.as_ptr().add((in_c + 1) * out_len + out_c);
                            let vw_hi = _mm256_loadu_ps(wp_hi);

                            let vs0_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                            let vs0_hi =
                                _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 1));
                            acc0_lo = _mm256_fmadd_ps(vs0_lo, vw_lo, acc0_lo);
                            acc0_hi = _mm256_fmadd_ps(vs0_hi, vw_hi, acc0_hi);

                            let vs1_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                            let vs1_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 1) * in_len + in_c + 1),
                            );
                            acc1_lo = _mm256_fmadd_ps(vs1_lo, vw_lo, acc1_lo);
                            acc1_hi = _mm256_fmadd_ps(vs1_hi, vw_hi, acc1_hi);

                            let vs2_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                            let vs2_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 2) * in_len + in_c + 1),
                            );
                            acc2_lo = _mm256_fmadd_ps(vs2_lo, vw_lo, acc2_lo);
                            acc2_hi = _mm256_fmadd_ps(vs2_hi, vw_hi, acc2_hi);

                            let vs3_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));
                            let vs3_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 3) * in_len + in_c + 1),
                            );
                            acc3_lo = _mm256_fmadd_ps(vs3_lo, vw_lo, acc3_lo);
                            acc3_hi = _mm256_fmadd_ps(vs3_hi, vw_hi, acc3_hi);
                        },
                        {
                            let wp = weights.as_ptr().add(in_c * out_len + out_c);
                            let vw = _mm256_loadu_ps(wp);
                            let vs0 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                            let vs1 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                            let vs2 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                            let vs3 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));

                            acc0_lo = _mm256_fmadd_ps(vs0, vw, acc0_lo);
                            acc1_lo = _mm256_fmadd_ps(vs1, vw, acc1_lo);
                            acc2_lo = _mm256_fmadd_ps(vs2, vw, acc2_lo);
                            acc3_lo = _mm256_fmadd_ps(vs3, vw, acc3_lo);
                        }
                    );

                    let acc0 = _mm256_add_ps(acc0_lo, acc0_hi);
                    let acc1 = _mm256_add_ps(acc1_lo, acc1_hi);
                    let acc2 = _mm256_add_ps(acc2_lo, acc2_hi);
                    let acc3 = _mm256_add_ps(acc3_lo, acc3_hi);

                    _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
                },
                {
                    for i in 0..4 {
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
                }
            );
        },
        {
            let in_frame = &in_frames[f * in_len..(f + 1) * in_len];
            let out_frame = &mut out_frames[f * out_len..(f + 1) * out_len];
            let res_frame = &residual[f * out_len..(f + 1) * out_len];

            let mut out_c = 0;
            while out_c + 8 <= out_len {
                let mut accum = _mm256_loadu_ps(res_frame.as_ptr().add(out_c));
                if do_bias {
                    accum = _mm256_add_ps(accum, _mm256_loadu_ps(bias.as_ptr().add(out_c)));
                }
                for in_c in 0..in_len {
                    let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                    let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                    let vw = _mm256_loadu_ps(weight_ptr);
                    accum = _mm256_fmadd_ps(vs, vw, accum);
                }
                _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
                out_c += 8;
            }
            while out_c < out_len {
                let mut sum = if do_bias {
                    *bias.get_unchecked(out_c)
                } else {
                    0.0
                };
                sum += *res_frame.get_unchecked(out_c);
                for in_c in 0..in_len {
                    sum += *in_frame.get_unchecked(in_c)
                        * *weights.get_unchecked(in_c * out_len + out_c);
                }
                *out_frame.get_unchecked_mut(out_c) = sum;
                out_c += 1;
            }
        }
    );
}

/// Fused residual GEMM kernel via AVX2 with native f32 weights.
///
/// # Safety
/// Preconditions are identical to [`fused_gemm_residual_batch_avx2`].
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_gemm_residual_batch_f32_avx2(
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
    assert!(in_frames.len() == num_frames * in_len);
    assert!(out_frames.len() == num_frames * out_len);
    assert!(weights.len() >= in_len * out_len);
    assert!(residual.len() >= num_frames * out_len);
    if do_bias {
        assert!(bias.len() >= out_len);
    }

    let mut f = 0;
    gemm_batch_frame_loop_avx2!(
        f,
        num_frames,
        {
            let mut out_c = 0;
            gemm_batch_outc_loop_avx2!(
                out_c,
                out_len,
                {
                    let res0 = _mm256_loadu_ps(residual.as_ptr().add(f * out_len + out_c));
                    let res1 = _mm256_loadu_ps(residual.as_ptr().add((f + 1) * out_len + out_c));
                    let res2 = _mm256_loadu_ps(residual.as_ptr().add((f + 2) * out_len + out_c));
                    let res3 = _mm256_loadu_ps(residual.as_ptr().add((f + 3) * out_len + out_c));

                    let b = if do_bias {
                        _mm256_loadu_ps(bias.as_ptr().add(out_c))
                    } else {
                        _mm256_setzero_ps()
                    };

                    let mut acc0_lo = _mm256_add_ps(res0, b);
                    let mut acc0_hi = _mm256_setzero_ps();
                    let mut acc1_lo = _mm256_add_ps(res1, b);
                    let mut acc1_hi = _mm256_setzero_ps();
                    let mut acc2_lo = _mm256_add_ps(res2, b);
                    let mut acc2_hi = _mm256_setzero_ps();
                    let mut acc3_lo = _mm256_add_ps(res3, b);
                    let mut acc3_hi = _mm256_setzero_ps();

                    let mut in_c = 0;
                    let mut wp_base = weights.as_ptr().add(out_c);
                    gemm_batch_inner_dual_avx2!(
                        in_c,
                        in_len,
                        {
                            let vw_lo = _mm256_loadu_ps(wp_base);
                            let vw_hi = _mm256_loadu_ps(wp_base.add(out_len));

                            let vs0_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                            let vs0_hi =
                                _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 1));
                            acc0_lo = _mm256_fmadd_ps(vs0_lo, vw_lo, acc0_lo);
                            acc0_hi = _mm256_fmadd_ps(vs0_hi, vw_hi, acc0_hi);

                            let vs1_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                            let vs1_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 1) * in_len + in_c + 1),
                            );
                            acc1_lo = _mm256_fmadd_ps(vs1_lo, vw_lo, acc1_lo);
                            acc1_hi = _mm256_fmadd_ps(vs1_hi, vw_hi, acc1_hi);

                            let vs2_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                            let vs2_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 2) * in_len + in_c + 1),
                            );
                            acc2_lo = _mm256_fmadd_ps(vs2_lo, vw_lo, acc2_lo);
                            acc2_hi = _mm256_fmadd_ps(vs2_hi, vw_hi, acc2_hi);

                            let vs3_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));
                            let vs3_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 3) * in_len + in_c + 1),
                            );
                            acc3_lo = _mm256_fmadd_ps(vs3_lo, vw_lo, acc3_lo);
                            acc3_hi = _mm256_fmadd_ps(vs3_hi, vw_hi, acc3_hi);

                            wp_base = wp_base.add(out_len * 2);
                        },
                        {
                            let vw = _mm256_loadu_ps(wp_base);
                            let vs0 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                            let vs1 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                            let vs2 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                            let vs3 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));

                            acc0_lo = _mm256_fmadd_ps(vs0, vw, acc0_lo);
                            acc1_lo = _mm256_fmadd_ps(vs1, vw, acc1_lo);
                            acc2_lo = _mm256_fmadd_ps(vs2, vw, acc2_lo);
                            acc3_lo = _mm256_fmadd_ps(vs3, vw, acc3_lo);
                        }
                    );

                    let acc0 = _mm256_add_ps(acc0_lo, acc0_hi);
                    let acc1 = _mm256_add_ps(acc1_lo, acc1_hi);
                    let acc2 = _mm256_add_ps(acc2_lo, acc2_hi);
                    let acc3 = _mm256_add_ps(acc3_lo, acc3_hi);

                    _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
                },
                {
                    for i in 0..4 {
                        let frame_idx = f + i;
                        let mut sum = *residual.get_unchecked(frame_idx * out_len + out_c);
                        if do_bias {
                            sum += *bias.get_unchecked(out_c);
                        }
                        let mut wp = weights.as_ptr().add(out_c);
                        for in_c in 0..in_len {
                            let w = *wp;
                            sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                            wp = wp.add(out_len);
                        }
                        *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum;
                    }
                }
            );
        },
        {
            let in_frame = &in_frames[f * in_len..(f + 1) * in_len];
            let out_frame = &mut out_frames[f * out_len..(f + 1) * out_len];
            let res_frame = &residual[f * out_len..(f + 1) * out_len];

            let mut out_c = 0;
            while out_c + 8 <= out_len {
                let mut accum = _mm256_loadu_ps(res_frame.as_ptr().add(out_c));
                if do_bias {
                    accum = _mm256_add_ps(accum, _mm256_loadu_ps(bias.as_ptr().add(out_c)));
                }
                let mut wp = weights.as_ptr().add(out_c);
                for in_c in 0..in_len {
                    let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                    let vw = _mm256_loadu_ps(wp);
                    accum = _mm256_fmadd_ps(vs, vw, accum);
                    wp = wp.add(out_len);
                }
                _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
                out_c += 8;
            }
            while out_c < out_len {
                let mut sum = if do_bias {
                    *bias.get_unchecked(out_c)
                } else {
                    0.0
                };
                sum += *res_frame.get_unchecked(out_c);
                let mut wp = weights.as_ptr().add(out_c);
                for in_c in 0..in_len {
                    sum += *in_frame.get_unchecked(in_c) * *wp;
                    wp = wp.add(out_len);
                }
                *out_frame.get_unchecked_mut(out_c) = sum;
                out_c += 1;
            }
        }
    );
}

/// Const-generic fused residual GEMM kernel (AVX2, native f32 weights).
///
/// # Safety
/// Preconditions are identical to [`fused_gemm_residual_batch_f32_avx2`].
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_gemm_residual_batch_f32_const<const IN: usize, const OUT: usize>(
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
    debug_assert_eq!(in_frames.len(), num_frames * IN);
    debug_assert_eq!(out_frames.len(), num_frames * OUT);
    debug_assert_eq!(residual.len(), num_frames * OUT);
    debug_assert!(weights.len() >= IN * OUT);
    if do_bias {
        debug_assert!(bias.len() >= OUT);
    }
    assert!(in_frames.len() == num_frames * IN);
    assert!(out_frames.len() == num_frames * OUT);
    assert!(residual.len() == num_frames * OUT);
    assert!(weights.len() >= IN * OUT);
    if do_bias {
        assert!(bias.len() >= OUT);
    }

    let mut f = 0;
    gemm_batch_frame_loop_avx2!(
        f,
        num_frames,
        {
            let mut out_c = 0;
            gemm_batch_outc_loop_avx2!(
                out_c,
                OUT,
                {
                    let res0 = _mm256_loadu_ps(residual.as_ptr().add(f * OUT + out_c));
                    let res1 = _mm256_loadu_ps(residual.as_ptr().add((f + 1) * OUT + out_c));
                    let res2 = _mm256_loadu_ps(residual.as_ptr().add((f + 2) * OUT + out_c));
                    let res3 = _mm256_loadu_ps(residual.as_ptr().add((f + 3) * OUT + out_c));

                    let b = if do_bias {
                        _mm256_loadu_ps(bias.as_ptr().add(out_c))
                    } else {
                        _mm256_setzero_ps()
                    };

                    let mut acc0_lo = _mm256_add_ps(res0, b);
                    let mut acc0_hi = _mm256_setzero_ps();
                    let mut acc1_lo = _mm256_add_ps(res1, b);
                    let mut acc1_hi = _mm256_setzero_ps();
                    let mut acc2_lo = _mm256_add_ps(res2, b);
                    let mut acc2_hi = _mm256_setzero_ps();
                    let mut acc3_lo = _mm256_add_ps(res3, b);
                    let mut acc3_hi = _mm256_setzero_ps();

                    let mut in_c = 0;
                    gemm_batch_inner_dual_avx2!(
                        in_c,
                        IN,
                        {
                            let wp_lo = weights.as_ptr().add(in_c * OUT + out_c);
                            let vw_lo = _mm256_loadu_ps(wp_lo);
                            let wp_hi = weights.as_ptr().add((in_c + 1) * OUT + out_c);
                            let vw_hi = _mm256_loadu_ps(wp_hi);

                            let vs0_lo = _mm256_set1_ps(*in_frames.get_unchecked(f * IN + in_c));
                            let vs0_hi =
                                _mm256_set1_ps(*in_frames.get_unchecked(f * IN + in_c + 1));
                            acc0_lo = _mm256_fmadd_ps(vs0_lo, vw_lo, acc0_lo);
                            acc0_hi = _mm256_fmadd_ps(vs0_hi, vw_hi, acc0_hi);

                            let vs1_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * IN + in_c));
                            let vs1_hi =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * IN + in_c + 1));
                            acc1_lo = _mm256_fmadd_ps(vs1_lo, vw_lo, acc1_lo);
                            acc1_hi = _mm256_fmadd_ps(vs1_hi, vw_hi, acc1_hi);

                            let vs2_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * IN + in_c));
                            let vs2_hi =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * IN + in_c + 1));
                            acc2_lo = _mm256_fmadd_ps(vs2_lo, vw_lo, acc2_lo);
                            acc2_hi = _mm256_fmadd_ps(vs2_hi, vw_hi, acc2_hi);

                            let vs3_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * IN + in_c));
                            let vs3_hi =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * IN + in_c + 1));
                            acc3_lo = _mm256_fmadd_ps(vs3_lo, vw_lo, acc3_lo);
                            acc3_hi = _mm256_fmadd_ps(vs3_hi, vw_hi, acc3_hi);
                        },
                        {
                            let wp = weights.as_ptr().add(in_c * OUT + out_c);
                            let vw = _mm256_loadu_ps(wp);
                            let vs0 = _mm256_set1_ps(*in_frames.get_unchecked(f * IN + in_c));
                            let vs1 = _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * IN + in_c));
                            let vs2 = _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * IN + in_c));
                            let vs3 = _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * IN + in_c));

                            acc0_lo = _mm256_fmadd_ps(vs0, vw, acc0_lo);
                            acc1_lo = _mm256_fmadd_ps(vs1, vw, acc1_lo);
                            acc2_lo = _mm256_fmadd_ps(vs2, vw, acc2_lo);
                            acc3_lo = _mm256_fmadd_ps(vs3, vw, acc3_lo);
                        }
                    );

                    let acc0 = _mm256_add_ps(acc0_lo, acc0_hi);
                    let acc1 = _mm256_add_ps(acc1_lo, acc1_hi);
                    let acc2 = _mm256_add_ps(acc2_lo, acc2_hi);
                    let acc3 = _mm256_add_ps(acc3_lo, acc3_hi);

                    _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * OUT + out_c), acc0);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * OUT + out_c), acc1);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * OUT + out_c), acc2);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * OUT + out_c), acc3);
                },
                {
                    for i in 0..4 {
                        let frame_idx = f + i;
                        let mut sum = *residual.get_unchecked(frame_idx * OUT + out_c);
                        if do_bias {
                            sum += *bias.get_unchecked(out_c);
                        }
                        for in_c in 0..IN {
                            let w = *weights.get_unchecked(in_c * OUT + out_c);
                            sum += *in_frames.get_unchecked(frame_idx * IN + in_c) * w;
                        }
                        *out_frames.get_unchecked_mut(frame_idx * OUT + out_c) = sum;
                    }
                }
            );
        },
        {
            let in_frame = &in_frames[f * IN..(f + 1) * IN];
            let out_frame = &mut out_frames[f * OUT..(f + 1) * OUT];
            let res_frame = &residual[f * OUT..(f + 1) * OUT];

            let mut out_c = 0;
            while out_c + 8 <= OUT {
                let mut accum = _mm256_loadu_ps(res_frame.as_ptr().add(out_c));
                if do_bias {
                    accum = _mm256_add_ps(accum, _mm256_loadu_ps(bias.as_ptr().add(out_c)));
                }
                for in_c in 0..IN {
                    let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                    let weight_ptr = weights.as_ptr().add(in_c * OUT + out_c);
                    let vw = _mm256_loadu_ps(weight_ptr);
                    accum = _mm256_fmadd_ps(vs, vw, accum);
                }
                _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
                out_c += 8;
            }
            while out_c < OUT {
                let mut sum = if do_bias {
                    *bias.get_unchecked(out_c)
                } else {
                    0.0
                };
                sum += *res_frame.get_unchecked(out_c);
                for in_c in 0..IN {
                    sum +=
                        *in_frame.get_unchecked(in_c) * *weights.get_unchecked(in_c * OUT + out_c);
                }
                *out_frame.get_unchecked_mut(out_c) = sum;
                out_c += 1;
            }
        }
    );
}

/// Dedicated 12x12 SIMD (YMM + XMM) batch residual GEMM kernel.
///
/// # Safety
/// Preconditions are identical to [`fused_gemm_residual_batch_avx2`].
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_gemm_residual_batch_f32_12x12(
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
    debug_assert_eq!(in_frames.len(), num_frames * 12);
    debug_assert_eq!(out_frames.len(), num_frames * 12);
    debug_assert_eq!(residual.len(), num_frames * 12);
    debug_assert!(weights.len() >= 144);
    if do_bias {
        debug_assert!(bias.len() >= 12);
    }
    assert!(in_frames.len() == num_frames * 12);
    assert!(out_frames.len() == num_frames * 12);
    assert!(residual.len() == num_frames * 12);
    assert!(weights.len() >= 144);
    if do_bias {
        assert!(bias.len() >= 12);
    }

    use core::arch::x86_64::{
        _mm_add_ps, _mm_fmadd_ps, _mm_loadu_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    let mut f = 0;
    while f + 4 <= num_frames {
        let res0_lo = _mm256_loadu_ps(residual.as_ptr().add(f * 12));
        let res1_lo = _mm256_loadu_ps(residual.as_ptr().add((f + 1) * 12));
        let res2_lo = _mm256_loadu_ps(residual.as_ptr().add((f + 2) * 12));
        let res3_lo = _mm256_loadu_ps(residual.as_ptr().add((f + 3) * 12));

        let b_lo = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };

        let mut acc0_lo = _mm256_add_ps(res0_lo, b_lo);
        let mut acc1_lo = _mm256_add_ps(res1_lo, b_lo);
        let mut acc2_lo = _mm256_add_ps(res2_lo, b_lo);
        let mut acc3_lo = _mm256_add_ps(res3_lo, b_lo);

        let res0_hi = _mm_loadu_ps(residual.as_ptr().add(f * 12 + 8));
        let res1_hi = _mm_loadu_ps(residual.as_ptr().add((f + 1) * 12 + 8));
        let res2_hi = _mm_loadu_ps(residual.as_ptr().add((f + 2) * 12 + 8));
        let res3_hi = _mm_loadu_ps(residual.as_ptr().add((f + 3) * 12 + 8));

        let b_hi = if do_bias {
            _mm_loadu_ps(bias.as_ptr().add(8))
        } else {
            _mm_setzero_ps()
        };

        let mut acc0_hi = _mm_add_ps(res0_hi, b_hi);
        let mut acc1_hi = _mm_add_ps(res1_hi, b_hi);
        let mut acc2_hi = _mm_add_ps(res2_hi, b_hi);
        let mut acc3_hi = _mm_add_ps(res3_hi, b_hi);

        for in_c in 0..12 {
            let wp_lo = weights.as_ptr().add(in_c * 12);
            let vw_lo = _mm256_loadu_ps(wp_lo);
            let vw_hi = _mm_loadu_ps(wp_lo.add(8));

            let vs0 = *in_frames.get_unchecked(f * 12 + in_c);
            let vs1 = *in_frames.get_unchecked((f + 1) * 12 + in_c);
            let vs2 = *in_frames.get_unchecked((f + 2) * 12 + in_c);
            let vs3 = *in_frames.get_unchecked((f + 3) * 12 + in_c);

            acc0_lo = _mm256_fmadd_ps(_mm256_set1_ps(vs0), vw_lo, acc0_lo);
            acc1_lo = _mm256_fmadd_ps(_mm256_set1_ps(vs1), vw_lo, acc1_lo);
            acc2_lo = _mm256_fmadd_ps(_mm256_set1_ps(vs2), vw_lo, acc2_lo);
            acc3_lo = _mm256_fmadd_ps(_mm256_set1_ps(vs3), vw_lo, acc3_lo);

            acc0_hi = _mm_fmadd_ps(_mm_set1_ps(vs0), vw_hi, acc0_hi);
            acc1_hi = _mm_fmadd_ps(_mm_set1_ps(vs1), vw_hi, acc1_hi);
            acc2_hi = _mm_fmadd_ps(_mm_set1_ps(vs2), vw_hi, acc2_hi);
            acc3_hi = _mm_fmadd_ps(_mm_set1_ps(vs3), vw_hi, acc3_hi);
        }

        _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * 12), acc0_lo);
        _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * 12), acc1_lo);
        _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * 12), acc2_lo);
        _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * 12), acc3_lo);

        _mm_storeu_ps(out_frames.as_mut_ptr().add(f * 12 + 8), acc0_hi);
        _mm_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * 12 + 8), acc1_hi);
        _mm_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * 12 + 8), acc2_hi);
        _mm_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * 12 + 8), acc3_hi);

        f += 4;
    }

    while f < num_frames {
        let res_lo = _mm256_loadu_ps(residual.as_ptr().add(f * 12));
        let b_lo = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };
        let mut acc_lo = _mm256_add_ps(res_lo, b_lo);

        let res_hi = _mm_loadu_ps(residual.as_ptr().add(f * 12 + 8));
        let b_hi = if do_bias {
            _mm_loadu_ps(bias.as_ptr().add(8))
        } else {
            _mm_setzero_ps()
        };
        let mut acc_hi = _mm_add_ps(res_hi, b_hi);

        for in_c in 0..12 {
            let wp_lo = weights.as_ptr().add(in_c * 12);
            let vw_lo = _mm256_loadu_ps(wp_lo);
            let vw_hi = _mm_loadu_ps(wp_lo.add(8));

            let vs = *in_frames.get_unchecked(f * 12 + in_c);
            acc_lo = _mm256_fmadd_ps(_mm256_set1_ps(vs), vw_lo, acc_lo);
            acc_hi = _mm_fmadd_ps(_mm_set1_ps(vs), vw_hi, acc_hi);
        }

        _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * 12), acc_lo);
        _mm_storeu_ps(out_frames.as_mut_ptr().add(f * 12 + 8), acc_hi);

        f += 1;
    }
}
