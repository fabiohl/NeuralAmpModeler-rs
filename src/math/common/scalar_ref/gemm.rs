// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// Batch Matrix Processing (GEMM).
/// GEMM stands for "General Matrix Multiplication". It's the heart of neural networks.
/// This function processes multiple audio "frames" at once.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn fused_add_gemm_batch_fallback(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32], // "Bias" is a fixed offset added at the end (like the 'b' in y = ax + b).
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    // Figure out how much space each frame occupies in memory.
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    // For each frame in the batch...
    for f in 0..num_frames {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe {
            // ...call the function that processes a single vector (GEMV).
            fused_add_gemv_fallback(
                in_frames.get_unchecked(f * in_len..(f + 1) * in_len),
                weights,
                bias,
                out_frames.get_unchecked_mut(f * out_len..(f + 1) * out_len),
                do_bias,
            );
        }
    }
}

/// Matrix-Vector Multiplication (GEMV) that ADDS to the existing result.
/// Think of it as injecting a new processing layer on top of what was already there.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn fused_add_gemv_fallback(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    // For each output "neuron"...
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        // We start with the 'bias' (offset) or with zero.
        let mut sum = if do_bias { b } else { 0.0 };

        // We iterate through all inputs and corresponding weights.
        for in_c in 0..in_len {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                let w = *weights.get_unchecked(in_c * out_len + out_c);
                // Multiplies the input by the weight and accumulates.
                sum += *in_frame.get_unchecked(in_c) * w;
            }
        }
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe {
            // IMPORTANT: Here we use '+=' to ADD to what was already in the output buffer.
            *out_frame.get_unchecked_mut(out_c) += sum;
        }
    }
}

/// Matrix-Vector Multiplication (GEMV) that OVERWRITES the result.
/// Unlike the previous one, this erases what was in the output and places the new value.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn gemv_overwrite_fallback(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        let mut sum = if do_bias { b } else { 0.0 };
        for in_c in 0..in_len {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                let w = *weights.get_unchecked(in_c * out_len + out_c);
                sum += *in_frame.get_unchecked(in_c) * w;
            }
        }
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe {
            // IMPORTANT: Here we use '=' to CLEAR and set the new value.
            *out_frame.get_unchecked_mut(out_c) = sum;
        }
    }
}

/// Residual Matrix Multiplication in a batch.
/// In "Residual" neural networks, we add the processing result to the original signal.
/// It's like saying: "Change the sound just a little bit relative to what it was."
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn fused_gemm_residual_batch_fallback(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    residual: &[f32], // This is the "clean" signal that will be added at the end.
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
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe {
            for (out_c, &b) in bias.iter().enumerate().take(out_len) {
                let mut sum = if do_bias { b } else { 0.0 };
                for in_c in 0..in_len {
                    let w = *weights.get_unchecked(in_c * out_len + out_c);
                    sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                }
                // Final result = (Matrix Processing) + (Original Residual Signal).
                *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) =
                    sum + *residual.get_unchecked(frame_idx * out_len + out_c);
            }
        }
    }
}

/// Matrix-Vector Multiplication (Overwrite) using BF16 input and weights.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn gemv_overwrite_bf16_fallback(
    in_frame: &[u16],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        let mut sum = if do_bias { b } else { 0.0 };
        for in_c in 0..in_len {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                // Unpack BF16 input -> f32.
                let s = f32::from_bits((*in_frame.get_unchecked(in_c) as u32) << 16);
                // Unpack BF16 weight -> f32 and multiply.
                sum += s * f32::from_bits(
                    (*weights.get_unchecked(in_c * out_len + out_c) as u32) << 16,
                );
            }
        }
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe {
            *out_frame.get_unchecked_mut(out_c) = sum;
        }
    }
}

/// Batch GEMV overwrite with bias using native f32 weights and inputs.
///
/// Performs `num_frames` independent matrix-vector multiplications.
/// Layout: input is frame-major `[f0_in.., f1_in.., ...]`, output is
/// frame-major `[f0_out.., f1_out.., ...]`. Weights are column-major
/// `weights[in_c * OUT + out_c]`.
///
/// Always initialises the accumulator with the bias.
/// This is the scalar reference oracle for the SIMD kernels.
#[inline]
pub fn gemv_with_bias_f32_fallback(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
) {
    if num_frames == 0 {
        return;
    }
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;
    for n in 0..num_frames {
        for out_c in 0..out_len {
            let mut sum = bias[out_c];
            for in_c in 0..in_len {
                sum += in_frames[n * in_len + in_c] * weights[in_c * out_len + out_c];
            }
            out_frames[n * out_len + out_c] = sum;
        }
    }
}

/// Batch GEMV overwrite without bias using native f32 weights and inputs.
///
/// Performs `num_frames` independent matrix-vector multiplications.
/// Layout: input is frame-major `[f0_in.., f1_in.., ...]`, output is
/// frame-major `[f0_out.., f1_out.., ...]`. Weights are column-major
/// `weights[in_c * OUT + out_c]`.
///
/// Does not add bias; pure overwrite.
/// This is the scalar reference oracle for the SIMD kernels.
#[inline]
pub fn gemv_no_bias_f32_fallback(
    in_frames: &[f32],
    weights: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
) {
    if num_frames == 0 {
        return;
    }
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;
    for n in 0..num_frames {
        for out_c in 0..out_len {
            let mut sum = 0.0;
            for in_c in 0..in_len {
                sum += in_frames[n * in_len + in_c] * weights[in_c * out_len + out_c];
            }
            out_frames[n * out_len + out_c] = sum;
        }
    }
}

/// Fused residual batch GEMM with native f32 weights — scalar reference oracle.
///
/// Computes `output = residual + bias + weights * input` in a single pass,
/// matching the operational semantics of `fused_gemm_residual_batch_f32_avx2`
/// and `fused_gemm_residual_batch_f32_avx512`.
#[inline]
pub fn fused_gemm_residual_batch_f32_fallback(
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
            let mut sum = residual[frame_idx * out_len + out_c];
            if do_bias {
                sum += bias[out_c];
            }
            for in_c in 0..in_len {
                sum += in_frames[frame_idx * in_len + in_c] * weights[in_c * out_len + out_c];
            }
            out_frames[frame_idx * out_len + out_c] = sum;
        }
    }
}
