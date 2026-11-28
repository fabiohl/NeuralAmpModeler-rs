// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// Specialized GEMM micro-kernels utilize unsafe SIMD intrinsics and parameter lists.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

//! Scalar / generic reference implementation for fused residual batch GEMM kernels.

/// Generic scalar implementation for fused residual batch GEMM.
///
/// Computes `out_frames = residual + bias (optional) + weights * in_frames`.
///
/// # Safety
/// Let `in_len = in_frames.len() / num_frames` and
/// `out_len = out_frames.len() / num_frames` (the same truncated divisions used
/// below; `num_frames == 0` returns early). The caller must ensure:
/// - `residual.len() >= num_frames * out_len`,
/// - `weights.len() >= in_len * out_len`,
/// - `bias.len() >= out_len` when `do_bias`.
///
/// `in_frames` and `out_frames` are only indexed below
/// `num_frames * in_len` / `num_frames * out_len`, both of which are `<=` their
/// slice lengths by the truncated division, so trailing elements (when the
/// lengths are not exact multiples of `num_frames`) are ignored safely.
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
            // SAFETY: `frame_idx * out_len + out_c < num_frames * out_len <=
            // residual.len()` by the caller contract.
            let mut sum = unsafe { *residual.get_unchecked(frame_idx * out_len + out_c) };
            if do_bias {
                // SAFETY: `out_c < out_len <= bias.len()` when `do_bias`.
                sum += unsafe { *bias.get_unchecked(out_c) };
            }
            for in_c in 0..in_len {
                // SAFETY: `in_c * out_len + out_c < in_len * out_len <=
                // weights.len()` by the caller contract.
                let w = unsafe { *weights.get_unchecked(in_c * out_len + out_c) };
                // SAFETY: `frame_idx * in_len + in_c < num_frames * in_len <=
                // in_frames.len()` (truncated-division bound).
                sum += unsafe { *in_frames.get_unchecked(frame_idx * in_len + in_c) } * w;
            }
            // SAFETY: `frame_idx * out_len + out_c < num_frames * out_len <=
            // out_frames.len()` and the `&mut` is exclusive.
            unsafe { *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum };
        }
    }
}

/// Generic scalar implementation for fused residual batch GEMM with native f32 weights.
///
/// # Safety
/// Preconditions are identical to [`fused_gemm_residual_batch_scalar`].
pub unsafe fn fused_gemm_residual_batch_f32_scalar(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    residual: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    // SAFETY: forwards the caller's preconditions verbatim.
    unsafe {
        fused_gemm_residual_batch_scalar(
            in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
        );
    }
}
