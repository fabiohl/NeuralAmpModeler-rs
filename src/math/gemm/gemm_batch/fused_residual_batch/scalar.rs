// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Scalar / generic reference implementation for fused residual batch GEMM kernels.

/// Generic scalar implementation for fused residual batch GEMM.
///
/// Computes `out_frames = residual + bias (optional) + weights * in_frames`.
///
/// # Safety
/// Preconditions (slice bounds and matrix dimensions) must be satisfied.
pub unsafe fn fused_gemm_residual_batch_scalar(
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
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    for frame_idx in 0..num_frames {
        for out_c in 0..out_len {
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
}

/// Generic scalar implementation for fused residual batch GEMM with native f32 weights.
///
/// # Safety
/// Preconditions (slice bounds and matrix dimensions) must be satisfied.
pub unsafe fn fused_gemm_residual_batch_f32_scalar(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    residual: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    fused_gemm_residual_batch_scalar(
        in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
    );
}
