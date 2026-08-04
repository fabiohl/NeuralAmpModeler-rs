// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Interface definitions for GEMM (General Matrix Multiply) and GEMV kernels.

/// Trait for single-vector GEMV (matrix-vector multiplication) kernels.
pub trait GemvKernel {
    /// Fused add + GEMV kernel with f32 weights.
    ///
    /// # Safety
    /// Slices must be valid and non-aliasing.
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[f32],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// GEMV kernel with overwrite.
    ///
    /// # Safety
    /// Slices must be valid and non-aliasing.
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[f32],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );
}

/// Trait for batched GEMM kernels.
pub trait GemmBatchKernel {
    /// Fused add + batched GEMM kernel with f32 weights.
    ///
    /// # Safety
    /// Slices must be valid and non-aliasing.
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[f32],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// Fused residual batched GEMM kernel.
    ///
    /// # Safety
    /// Slices must be valid and non-aliasing.
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[f32],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );
}

/// Trait for dot product kernels.
pub trait DotProductKernel {
    /// Computes the dot product between two f32 vectors.
    ///
    /// # Safety
    /// `a` and `b` must be valid slices.
    unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32;

    /// Computes 4 simultaneous dot products with native f32 weights.
    ///
    /// # Safety
    /// `weights.len() >= state.len()`.
    unsafe fn dot_product_4x_f32(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4];
}
