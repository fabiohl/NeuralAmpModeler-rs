// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Abstract interfaces and traits for SIMD, GEMM, vector reduction, and DSP math operations.

pub mod gemm;
pub mod reduce;
pub mod simd;

pub use gemm::*;
pub use reduce::*;
pub use simd::*;

use super::InstructionSet;

/// Abstraction trait for static dispatch of SIMD mathematical operations.
///
/// # Safety
///
/// All implementations of this trait use x86-64 SIMD intrinsics that require
/// specific CPU features (AVX2+FMA minimum, enforced project-wide via `x86-64-v3`
/// in `.cargo/config.toml`). Dynamic dispatch only selects higher extensions
/// (AVX-512, AVX-512VNNI+BF16) at runtime via `SimdMathConfig::current()`.
///
/// Each method documents its own preconditions below. In general:
/// - **All** pointer/slice arguments must be valid (non-null, within allocation,
///   properly aligned for the target ISA).
/// - **Convolution** functions additionally require coefficient pointers aligned to
///   32 bytes (AVX2) or 64 bytes (AVX-512); all other operations use unaligned
///   loads and stores.
/// - **Length** invariants (e.g., `weights.len() >= input_len * output_len`) are
///   documented per method. Callers must uphold them.
/// - **Shared mutability** must not occur: no slice aliasing between `&mut` and
///   any other access during the call.
/// - Input values must be finite IEEE 754 `f32`; behavior with NaN/Inf is
///   implementation-defined.
/// - BF16-typed slices (`&[u16]`) must contain valid BF16 bit patterns.
///
/// # Operation Groups
///
/// The operations of this trait are organized into the following groups:
/// - **(A) Dot Products**: Scalar/4x/dual-frame dot products (e.g., `dot_product`).
/// - **(B) GEMV/GEMM Fused**: Fused matrix-vector/matrix-matrix kernels (e.g., `fused_add_gemv`).
/// - **(C) Activations**: Tanh/sigmoid activation functions and gated fusions (e.g., `tanh_slice`).
/// - **(D) Conversions**: f32 ↔ bf16 conversion utilities (e.g., `f32_to_bf16`).
/// - **(E) LSTM Gates**: Specific kernels for LSTM cell gates (e.g., `fused_lstm_gates_dyn`).
/// - **(F) Complex MAC**: Complex multiply-accumulate spectral kernels.
/// - **(G) FFT Butterfly**: SIMD Radix-2 DIT FFT butterfly stages.
/// - **(H) BatchNorm**: Frame-major batch normalization affine transform.
pub trait SimdMath {
    /// SIMD register type used (e.g.: __m256 or __m512).
    type V: Copy;

    /// Indicates whether this implementation uses weights and signals in BF16 format.
    const IS_BF16: bool = false;

    /// ISA that this implementation targets (compile-time constant for monomorphization).
    const ISA: InstructionSet;

    // --- (A) Dot Products ---

    /// Computes the dot product between two f32 vectors.
    ///
    /// Effective length is `min(a.len(), b.len())`. No alignment required.
    ///
    /// # Safety
    /// `a` and `b` must be valid slices.
    unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32;

    /// Computes the dot product between two BF16 vectors.
    ///
    /// Effective length is `min(a.len(), b.len())`. No alignment required.
    ///
    /// # Safety
    /// `a` and `b` must be valid slices with valid BF16 bit patterns.
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32;

    /// Computes 4 simultaneous BF16 dot products (interleaved) with f32 input.
    ///
    /// No alignment required. The output is `[a·w0, a·w1, a·w2, a·w3]`
    /// where each `w_k` is the k-th column of the interleaved weight matrix.
    ///
    /// # Safety
    /// `weights.len() >= state.len()`. Both slices must be valid
    /// and accessible for reading.
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4];

    /// Computes 4 simultaneous BF16 dot products (interleaved) for 2 parallel frames.
    /// Returns `(results_f0, results_f1)`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= max(state_f0.len(), state_f1.len())`.
    /// All slices must be valid and accessible for reading.
    unsafe fn dot_product_4x_interleaved_dual_frame(
        weights: &[[u16; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]);

    /// Computes 4 simultaneous dot products with native f32 weights.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= state.len()`. Each element of `weights` must be
    /// a valid `[f32; 4]` row. Both slices must be accessible for reading.
    unsafe fn dot_product_4x_f32(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4];

    /// Computes 4 simultaneous dot products with native f32 weights
    /// for 2 parallel frames.
    /// Returns `(results_f0, results_f1)`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= max(state_f0.len(), state_f1.len())`.
    /// All slices must be valid and accessible for reading.
    unsafe fn dot_product_4x_f32_dual(
        weights: &[[f32; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]);

    /// Computes 8 simultaneous dot products with native f32 weights.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= state.len()`. Each element of `weights` must be
    /// a valid `[f32; 8]` row. Both slices must be accessible for reading.
    unsafe fn dot_product_8x_f32(weights: &[[f32; 8]], state: &[f32]) -> [f32; 8];

    /// Computes 8 simultaneous dot products with native f32 weights
    /// for 2 parallel frames.
    /// Returns `(results_f0, results_f1)`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= max(state_f0.len(), state_f1.len())`.
    /// All slices must be valid and accessible for reading.
    unsafe fn dot_product_8x_f32_dual(
        weights: &[[f32; 8]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 8], [f32; 8]);

    /// Computes 16 simultaneous dot products with native f32 weights.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= state.len()`. Each element of `weights` must be
    /// a valid `[f32; 16]` row. Both slices must be accessible for reading.
    unsafe fn dot_product_16x_f32(weights: &[[f32; 16]], state: &[f32]) -> [f32; 16];

    /// Computes 16 simultaneous dot products with native f32 weights
    /// for 2 parallel frames.
    /// Returns `(results_f0, results_f1)`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= max(state_f0.len(), state_f1.len())`.
    /// All slices must be valid and accessible for reading.
    unsafe fn dot_product_16x_f32_dual(
        weights: &[[f32; 16]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 16], [f32; 16]);

    /// Fused accumulate variant: computes 4 simultaneous dot products with native f32
    /// weights and adds the `init` accumulator (bias + mixin).
    ///
    /// Equivalent to `dot_product_4x_f32(weights, state)` followed by element-wise
    /// addition of `init`. Fusing both avoids an extra pass over the output array.
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= state.len()`.
    unsafe fn dot_product_4x_f32_accumulate(
        weights: &[[f32; 4]],
        state: &[f32],
        init: &[f32; 4],
    ) -> [f32; 4];

    /// Fused accumulate dual-frame variant: computes 4 simultaneous dot products for
    /// 2 parallel frames and adds the respective `init_f0` / `init_f1` accumulators.
    /// Returns `(results_f0, results_f1)`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= max(state_f0.len(), state_f1.len())`.
    unsafe fn dot_product_4x_f32_dual_accumulate(
        weights: &[[f32; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
        init_f0: &[f32; 4],
        init_f1: &[f32; 4],
    ) -> ([f32; 4], [f32; 4]);

    /// Fused accumulate variant: computes 8 simultaneous dot products with native f32
    /// weights and adds the `init` accumulator.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= state.len()`.
    unsafe fn dot_product_8x_f32_accumulate(
        weights: &[[f32; 8]],
        state: &[f32],
        init: &[f32; 8],
    ) -> [f32; 8];

    /// Fused accumulate dual-frame variant: computes 8 simultaneous dot products for
    /// 2 parallel frames and adds the respective `init_f0` / `init_f1` accumulators.
    /// Returns `(results_f0, results_f1)`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= max(state_f0.len(), state_f1.len())`.
    unsafe fn dot_product_8x_f32_dual_accumulate(
        weights: &[[f32; 8]],
        state_f0: &[f32],
        state_f1: &[f32],
        init_f0: &[f32; 8],
        init_f1: &[f32; 8],
    ) -> ([f32; 8], [f32; 8]);

    /// Fused accumulate variant: computes 16 simultaneous dot products with native f32
    /// weights and adds the `init` accumulator.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= state.len()`.
    unsafe fn dot_product_16x_f32_accumulate(
        weights: &[[f32; 16]],
        state: &[f32],
        init: &[f32; 16],
    ) -> [f32; 16];

    /// Fused accumulate dual-frame variant: computes 16 simultaneous dot products for
    /// 2 parallel frames and adds the respective `init_f0` / `init_f1` accumulators.
    /// Returns `(results_f0, results_f1)`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() >= max(state_f0.len(), state_f1.len())`.
    unsafe fn dot_product_16x_f32_dual_accumulate(
        weights: &[[f32; 16]],
        state_f0: &[f32],
        state_f1: &[f32],
        init_f0: &[f32; 16],
        init_f1: &[f32; 16],
    ) -> ([f32; 16], [f32; 16]);

    /// Computes 4 simultaneous BF16 dot products with separate weight vectors.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// Each of `w0`, `w1`, `w2`, `w3` must have length `>= in_frame.len()`.
    /// All slices must be valid and contain valid BF16 bit patterns.
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4];

    /// Horizontal sum of `N` consecutive f32 values starting at `ptr`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `ptr` must point to at least `N` valid, initialized `f32` elements.
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32;

    // --- (B) GEMV/GEMM Fused ---

    /// Fused add + GEMV kernel (f32 weights).
    ///
    /// Computes `out += weights^T * in_frame [+ bias]` in a single pass.
    /// No alignment required.
    ///
    /// # Safety
    /// Let `in_len = in_frame.len()`, `out_len = out_frame.len()`.
    /// `weights.len() >= in_len * out_len`.
    /// `bias.len() >= out_len` (when `do_bias` is `true`).
    /// All slices must be valid; `out_frame` must not alias `in_frame`.
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[f32],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// Fused add + batch GEMM kernel (f32 weights).
    ///
    /// No alignment required. Frames are batched in groups of 4 (AVX2)
    /// or 8 (AVX-512) for throughput.
    ///
    /// # Safety
    /// `in_frames.len()` and `out_frames.len()` must divide exactly by `num_frames`.
    /// Let `in_len = in_frames.len() / num_frames`, `out_len = out_frames.len() / num_frames`.
    /// `weights.len() >= in_len * out_len`.
    /// `bias.len() >= out_len` (when `do_bias` is `true`).
    /// `num_frames > 0`. All slices must be valid and non-aliasing.
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[f32],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// Fused residual batch GEMM kernel (f32 weights).
    ///
    /// Computes `out = weights^T * in + residual [+ bias]`.
    /// No alignment required.
    ///
    /// # Safety
    /// Same preconditions as `fused_add_gemm_batch`, plus:
    /// `residual.len()` must divide exactly by `num_frames`,
    /// with `residual.len() / num_frames == out_len`.
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[f32],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// Fused residual batch GEMM kernel with native f32 weights.
    ///
    /// Used where the 1x1 projection operates on full-precision f32 weights.
    /// Fuses GEMV + bias + residual addition into a single SIMD pass.
    /// No alignment required.
    ///
    /// # Safety
    /// Same preconditions as `fused_gemm_residual_batch`.
    unsafe fn fused_gemm_residual_batch_f32(
        in_frames: &[f32],
        weights: &[f32],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// GEMV kernel with overwrite (f16c-quantized weights).
    ///
    /// Computes `out = weights^T * in [+ bias]` (no accumulation).
    /// No alignment required.
    ///
    /// # Safety
    /// Let `in_len = in_frame.len()`, `out_len = out_frame.len()`.
    /// `weights.len() >= in_len * out_len`.
    /// `bias.len() >= out_len` (when `do_bias` is `true`).
    /// All slices must be valid; `out_frame` must not alias `in_frame`.
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[f32],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// GEMV overwrite in batch (f32 weights).
    ///
    /// Delegates to per-frame `gemv_overwrite`. No alignment required.
    ///
    /// # Safety
    /// Same preconditions as `fused_add_gemm_batch`.
    unsafe fn gemv_overwrite_batch(
        in_frames: &[f32],
        weights: &[f32],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// GEMV kernel with overwrite (BF16 input and weights).
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// Let `in_len = in_frame.len()`, `out_len = out_frame.len()`.
    /// `weights.len() >= in_len * out_len`.
    /// `bias.len() >= out_len` (when `do_bias` is `true`).
    /// All slices must be valid and contain valid BF16 bit patterns.
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// GEMV overwrite in batch using native f32 weights (always adds bias).
    ///
    /// Used for mixed-precision head projection where the final stage requires
    /// full FP32 precision while the backbone runs quantized.
    /// No alignment required. Shape-dependent dispatch selects an optimized
    /// path based on `out_len` (1, ≤4, or ≥8).
    ///
    /// # Safety
    /// `in_frames.len()` and `out_frames.len()` must divide exactly by `num_frames`.
    /// Let `in_len = in_frames.len() / num_frames`, `out_len = out_frames.len() / num_frames`.
    /// `weights.len() == in_len * out_len`.
    /// `bias.len() >= out_len`.
    /// `num_frames > 0`. All slices must be valid and non-aliasing.
    unsafe fn gemv_with_bias_f32(
        in_frames: &[f32],
        weights: &[f32],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
    );

    /// GEMV overwrite in batch using native f32 weights (no bias).
    ///
    /// Used for mixed-precision head projection where the final stage requires
    /// full FP32 precision while the backbone runs quantized.
    /// Overwrites without adding bias. No alignment required.
    ///
    /// # Safety
    /// `in_frames.len()` and `out_frames.len()` must divide exactly by `num_frames`.
    /// Let `in_len = in_frames.len() / num_frames`, `out_len = out_frames.len() / num_frames`.
    /// `weights.len() == in_len * out_len`.
    /// `num_frames > 0`. All slices must be valid and non-aliasing.
    unsafe fn gemv_no_bias_f32(
        in_frames: &[f32],
        weights: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
    );

    /// GEMV with overwrite for 4 simultaneous LSTM gates (f16c-quantized).
    ///
    /// Computes `gates = W^T * in_frame [+ bias]` where the weight matrix
    /// contains concatenated gate weights: `[W_input | W_forget | W_cell | W_output]`.
    /// No alignment required.
    ///
    /// # Safety
    /// `weights.len() == in_frame.len() * hidden_size * 4`.
    /// `out_gates.len() >= hidden_size * 4`.
    /// `bias.len() >= hidden_size * 4` (when `do_bias` is `true`).
    /// All slices must be valid and non-aliasing.
    unsafe fn gemv_overwrite_4gate(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    );

    /// GEMV with overwrite for 4 simultaneous LSTM gates (BF16 input and weights).
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// Same preconditions as `gemv_overwrite_4gate`, plus all `u16` slices
    /// must contain valid BF16 bit patterns.
    unsafe fn gemv_overwrite_bf16_4gate(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    );

    // --- (C) Activations ---

    /// Accumulates `src` into `dest`: `dest[i] += src[i]`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `dest.len() == src.len()`. Both slices must be valid
    /// and must not alias.
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]);

    /// Fused Tanh + Head Accumulate.
    ///
    /// Computes `head_input[i] += tanh(block[i])`.
    /// No alignment required.
    ///
    /// # Safety
    /// `head_input.len() == block.len()`. Both slices must be valid
    /// and must not alias.
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]);

    /// Fused ReLU + Head Accumulate.
    ///
    /// Computes `head_input[i] += max(0.0, block[i])`.
    /// No alignment required.
    ///
    /// # Safety
    /// `head_input.len() == block.len()`. Both slices must be valid
    /// and must not alias.
    unsafe fn relu_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]);

    /// Fused Gated Activation + Head Accumulate.
    ///
    /// For each channel group of size `ch`, applies `tanh` to the gated portion
    /// and accumulates into the head:
    /// `head_input[i*ch + j] += tanh(block[i*ch + j])`.
    /// No alignment required.
    ///
    /// # Safety
    /// `head_input.len() == block.len()` and both divide exactly by `ch`.
    /// Both slices must be valid and must not alias. `ch > 0`.
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    );

    /// Fused Tanh + Head Overwrite.
    ///
    /// Computes `head_input[i] = tanh(block[i])`.
    /// No alignment required.
    ///
    /// # Safety
    /// `head_input.len() == block.len()`. Both slices must be valid
    /// and must not alias.
    unsafe fn tanh_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]);

    /// Fused ReLU + Head Overwrite.
    ///
    /// Computes `head_input[i] = max(0.0, block[i])`.
    /// No alignment required.
    ///
    /// # Safety
    /// `head_input.len() == block.len()`. Both slices must be valid
    /// and must not alias.
    unsafe fn relu_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]);

    /// Fused Seed + Tanh + Head Accumulate.
    ///
    /// Computes `head_input[i] = seed[i] + tanh(block[i])`.
    /// Eliminates the separate `copy_from_slice(seed)` before
    /// `tanh_and_accumulate_block` on the first layer of a cascaded array.
    /// No alignment required.
    ///
    /// # Safety
    /// `head_input.len() == block.len() == seed.len()`.
    /// All slices must be valid and must not alias.
    unsafe fn tanh_and_accumulate_with_seed(
        head_input: &mut [f32],
        block: &mut [f32],
        seed: &[f32],
    );

    /// Fused Seed + ReLU + Head Accumulate.
    ///
    /// Computes `head_input[i] = seed[i] + max(0.0, block[i])`.
    /// Eliminates the separate `copy_from_slice(seed)` before
    /// `relu_and_accumulate_block` on the first layer of a cascaded array.
    /// No alignment required.
    ///
    /// # Safety
    /// `head_input.len() == block.len() == seed.len()`.
    /// All slices must be valid and must not alias.
    unsafe fn relu_and_accumulate_with_seed(
        head_input: &mut [f32],
        block: &mut [f32],
        seed: &[f32],
    );

    /// Fused Gated Activation + Head Overwrite.
    ///
    /// For each channel group of size `ch`, applies `tanh` to the gated portion
    /// and overwrites the head:
    /// `head_input[i*ch + j] = tanh(block[i*ch + j])`.
    /// No alignment required.
    ///
    /// # Safety
    /// `head_input.len() == block.len()` and both divide exactly by `ch`.
    /// Both slices must be valid and must not alias. `ch > 0`.
    unsafe fn gated_activation_and_overwrite_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    );

    /// Computes `max(energy(l), energy(r))` as max mean-square energy.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `l` and `r` must be valid slices.
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32;

    /// Computes the mean-square energy: `(1/N) * Σ x_i²`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `data` must be a valid slice.
    unsafe fn compute_energy(data: &[f32]) -> f32;

    /// Computes `max(|a[i] - b[i]|)`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `a.len() == b.len()`. Both slices must be valid.
    unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32;

    /// Computes `(max(|left|), max(|right|))`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `left.len() == right.len()`. Both slices must be valid.
    unsafe fn compute_peak_abs_stereo(left: &[f32], right: &[f32]) -> (f32, f32);

    /// Computes `max(|x[i]|)` for a single channel.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `data` must be a valid slice.
    unsafe fn compute_peak_abs_mono(data: &[f32]) -> f32;

    /// Applies Tanh element-wise to `slice`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn tanh_slice(slice: &mut [f32]);

    /// Applies Sigmoid element-wise to `slice`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn sigmoid_slice(slice: &mut [f32]);

    /// Applies high-fidelity Tanh element-wise to `slice`.
    ///
    /// Uses polynomial exp-based approximation (~2.4e-7 error vs `f32::tanh`).
    /// Higher cost, lower error, lower aliasing. No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn tanh_slice_hf(slice: &mut [f32]) {
        // SAFETY: default implementation delegates to standard tanh_slice.
        unsafe {
            Self::tanh_slice(slice);
        }
    }

    /// Applies high-fidelity Sigmoid element-wise to `slice`.
    ///
    /// Uses polynomial exp-based approximation (~2.1e-7 error vs `f32::exp`).
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn sigmoid_slice_hf(slice: &mut [f32]) {
        // SAFETY: default implementation delegates to standard sigmoid_slice.
        unsafe {
            Self::sigmoid_slice(slice);
        }
    }

    /// Applies ReLU element-wise to `slice`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn relu_slice(slice: &mut [f32]);

    /// Applies PReLU element-wise: `slice[i] = max(0, slice[i]) + slopes[i] * min(0, slice[i])`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` and `slopes` must be valid slices with `slopes.len() >= slice.len()`.
    unsafe fn prelu_slice(slice: &mut [f32], slopes: &[f32]);

    /// Applies Softsign element-wise to `slice`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn softsign_slice(slice: &mut [f32]);

    /// Applies SiLU element-wise to `slice`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn silu_slice(slice: &mut [f32]);

    /// Applies high-fidelity SiLU element-wise to `slice`.
    ///
    /// Uses polynomial exp-based sigmoid (~2.1e-7 error vs `f32::exp`).
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn silu_slice_hf(slice: &mut [f32]) {
        // SAFETY: default implementation delegates to standard silu_slice.
        unsafe {
            Self::silu_slice(slice);
        }
    }

    /// Applies HardTanh element-wise to `slice`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn hard_tanh_slice(slice: &mut [f32]);

    /// Applies HardSwish element-wise to `slice`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn hard_swish_slice(slice: &mut [f32]);

    /// Applies FastTanh (rational approximation) element-wise to `slice`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn fast_tanh_slice(slice: &mut [f32]);

    /// Applies LeakyHardTanh element-wise.
    ///
    /// Uses piecewise-linear mapping with configurable saturation and leak slopes.
    /// No alignment required.
    ///
    /// # Safety
    /// `slice` must be a valid mutable slice.
    unsafe fn leaky_hard_tanh_slice(
        slice: &mut [f32],
        min_val: f32,
        max_val: f32,
        min_slope: f32,
        max_slope: f32,
    );

    /// Tanh activation on a block (shorthand for `tanh_slice`).
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `buf` must be a valid mutable slice.
    unsafe fn activation_tanh_block(buf: &mut [f32]);

    // --- (D) Conversions ---

    /// Conversion from F32 to BF16: `dest[i] = bf16(src[i])`.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `src.len() == dest.len()`. Both slices must be valid
    /// and must not alias.
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]);

    /// Stores the contents of a SIMD register as BF16 (truncated).
    ///
    /// Writes to memory via unaligned 128-bit store. No alignment required.
    ///
    /// # Safety
    /// `ptr` must be valid for at least enough `u16` elements to hold
    /// the register contents (8 elements for AVX2 `__m256`, 16 for AVX-512 `__m512`).
    unsafe fn store_bf16(ptr: *mut u16, v: Self::V);

    // --- (E) LSTM Gates ---

    /// Fused kernel for dynamic LSTM gate processing.
    ///
    /// Combines sigmoid/tanh activation of the 4 gates with cell and hidden
    /// state update in a single SIMD pass. No alignment required.
    ///
    /// # Safety
    /// `gates.len() >= hidden_size * 4`.
    /// `cell_state.len() == hidden_size`.
    /// `cell_error.len() == hidden_size`.
    /// `hidden_state.len() == hidden_size`.
    /// All slices must be valid and non-aliasing. `hidden_size > 0`.
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        cell_error: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    );

    /// Stereo convolution (used in the resampler).
    ///
    /// Computes the dot product between a coefficient bank and two input buffers (L/R).
    /// Coefficients are loaded with **aligned** SIMD loads for performance.
    ///
    /// # Safety
    /// `coeffs` must be aligned to 32 bytes (AVX2 `__m256`) or 64 bytes (AVX-512 `__m512`)
    /// depending on the concrete implementation.
    /// `coeffs`, `input_l`, and `input_r` must each point to at least `taps` valid `f32` elements.
    /// Input pointers use unaligned loads — no alignment required.
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32);

    /// Dual stereo convolution (reuses input loads).
    ///
    /// Computes the dot product between two coefficient banks and two input buffers (L/R).
    /// Both coefficient banks are loaded with **aligned** SIMD loads.
    ///
    /// # Safety
    /// `coeffs0` and `coeffs1` must be aligned to 32 bytes (AVX2) or 64 bytes (AVX-512).
    /// `coeffs0`, `coeffs1`, `input_l`, and `input_r` must each point to at least `taps`
    /// valid `f32` elements. Input pointers use unaligned loads — no alignment required.
    unsafe fn convolve_stereo_dual(
        coeffs0: *const f32,
        coeffs1: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> ((f32, f32), (f32, f32));

    /// Mono convolution (used in the resampler).
    ///
    /// Coefficients are loaded with **aligned** SIMD loads for performance.
    ///
    /// # Safety
    /// `coeffs` must be aligned to 32 bytes (AVX2 `__m256`) or 64 bytes (AVX-512 `__m512`).
    /// `coeffs` and `input` must each point to at least `taps` valid `f32` elements.
    /// Input uses unaligned loads — no alignment required.
    unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32;

    /// Dual mono convolution (reuses input loads).
    ///
    /// Both coefficient banks are loaded with **aligned** SIMD loads.
    ///
    /// # Safety
    /// `coeffs0` and `coeffs1` must be aligned to 32 bytes (AVX2) or 64 bytes (AVX-512).
    /// `coeffs0`, `coeffs1`, and `input` must each point to at least `taps`
    /// valid `f32` elements. Input uses unaligned loads — no alignment required.
    unsafe fn convolve_mono_dual(
        coeffs0: *const f32,
        coeffs1: *const f32,
        input: *const f32,
        taps: usize,
    ) -> (f32, f32);

    /// Applies gain and detects clipping in mono.
    ///
    /// Computes `data[i] *= gain` and returns `true` if any `|data[i]| > 1.0`.
    /// No alignment required.
    ///
    /// # Safety
    /// `data` must be a valid mutable slice.
    unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool;

    /// Applies gain and detects clipping in stereo.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `left` and `right` must be valid mutable slices.
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool;

    /// Applies constant gain in stereo without clipping detection.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `left` and `right` must be valid mutable slices.
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32);

    /// Applies constant gain to a mono buffer.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `data` must be a valid mutable slice.
    unsafe fn apply_gain(data: &mut [f32], gain: f32);

    /// Applies a linear gain ramp to a mono buffer.
    ///
    /// Computes `data[i] *= start + i * step`.
    /// No alignment required.
    ///
    /// # Safety
    /// `data` must be a valid mutable slice.
    unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32);

    /// Crossfade blend: `out[i] = out[i] * (1 - t) + pending[i] * t`.
    ///
    /// Computed as `fma(pending[i] - out[i], t, out[i])` for single-rounding precision.
    /// No alignment required.
    ///
    /// # Safety
    /// `out` and `pending` must be valid slices.
    /// Effective length is `min(out.len(), pending.len())`.
    unsafe fn crossfade_blend_mono(out: &mut [f32], pending: &[f32], t: f32);

    /// Applies a linear gain ramp in stereo.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `left` and `right` must be valid mutable slices.
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32);

    /// Fused gain + dither: `data[i] = data[i] * gain + offset` in a single pass.
    ///
    /// Eliminates the overhead of two separate passes (`apply_gain` then
    /// `apply_dither_add`) on the same buffer. Uses FMA for single-rounding
    /// precision where the ISA supports it.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `data` must be a valid mutable slice.
    unsafe fn apply_gain_then_dither(data: &mut [f32], gain: f32, offset: f32);

    /// Adds a broadcast constant (dither offset) to every element of a mono buffer.
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `data` must be a valid mutable slice.
    unsafe fn apply_dither_add(data: &mut [f32], offset: f32);

    // --- (F) Complex MAC ---

    /// Complex multiply-accumulate (overwrite): spectral kernel.
    ///
    /// Computes the element-wise complex multiplication of two vectors
    /// `(h_re, h_im)` and `(x_re, x_im)` writing the result to `(out_re, out_im)`.
    ///
    /// `out_re[i] = h_re[i] * x_re[i] - h_im[i] * x_im[i]`
    /// `out_im[i] = h_re[i] * x_im[i] + h_im[i] * x_re[i]`
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `n = h_re.len() == h_im.len() == x_re.len() == x_im.len() == out_re.len() == out_im.len()`.
    /// All slices must be valid and must not alias.
    unsafe fn complex_mac_overwrite(
        h_re: &[f32],
        h_im: &[f32],
        x_re: &[f32],
        x_im: &[f32],
        out_re: &mut [f32],
        out_im: &mut [f32],
    );

    /// Complex multiply-accumulate (accumulate): spectral kernel.
    ///
    /// Computes the element-wise complex multiplication of two vectors
    /// `(h_re, h_im)` and `(x_re, x_im)` adding the result to `(acc_re, acc_im)`.
    ///
    /// `acc_re[i] += h_re[i] * x_re[i] - h_im[i] * x_im[i]`
    /// `acc_im[i] += h_re[i] * x_im[i] + h_im[i] * x_re[i]`
    ///
    /// No alignment required.
    ///
    /// # Safety
    /// `n = h_re.len() == h_im.len() == x_re.len() == x_im.len() == acc_re.len() == acc_im.len()`.
    /// All slices must be valid and must not alias.
    unsafe fn complex_mac_accumulate(
        h_re: &[f32],
        h_im: &[f32],
        x_re: &[f32],
        x_im: &[f32],
        acc_re: &mut [f32],
        acc_im: &mut [f32],
    );

    // --- (G) FFT Butterfly ---

    /// SIMD Radix-2 DIT FFT butterfly stage for one group.
    ///
    /// Processes `half` butterflies for a single group starting at
    /// `group_start`. Each butterfly kernel computes:
    ///
    /// ```text
    /// t_re = w_re * re_bot - w_im * im_bot
    /// t_im = w_re * im_bot + w_im * re_bot
    /// re[top] = re[top] + t_re    re[bot] = re[top] - t_re
    /// im[top] = im[top] + t_im    im[bot] = im[top] - t_im
    /// ```
    ///
    /// where `top = group_start + j`, `bot = group_start + j + half`,
    /// and the twiddle factors `(w_re, w_im)` for `j = 0..half` are
    /// stored contiguously in `tw_re`/`tw_im`.
    ///
    /// When `inverse` is `true`, the twiddle imaginary part is
    /// conjugated (`w_im = -w_im`) before multiplication.
    ///
    /// # Safety
    /// - `re` and `im` point to `n` valid `f32` values.
    /// - `tw_re`/`tw_im` point to `half` valid `f32` values each.
    /// - `group_start + 2 * half <= n`.
    /// - CPU supports the required SIMD ISA (verified by dispatch).
    /// - Source and destination may alias within the same array
    ///   (in-place butterfly).
    unsafe fn fft_butterfly_stage(
        re: *mut f32,
        im: *mut f32,
        half: usize,
        tw_re: *const f32,
        tw_im: *const f32,
        group_start: usize,
        inverse: bool,
    );

    // --- (H) BatchNorm ---

    /// Frame-major batch normalization affine transform.
    ///
    /// Computes `data[f * n_ch + c] = data[f * n_ch + c] * scale[c] + offset[c]`
    /// for all frames and channels in a single pass.
    /// Processes frames in the outer loop for contiguous channel access.
    ///
    /// # Safety
    /// `data.len() == num_frames * n_ch`. `scale` and `offset` must each have
    /// at least `n_ch` elements. `num_frames > 0`.
    unsafe fn batch_norm_process(
        data: &mut [f32],
        scale: &[f32],
        offset: &[f32],
        n_ch: usize,
        num_frames: usize,
    );
}
