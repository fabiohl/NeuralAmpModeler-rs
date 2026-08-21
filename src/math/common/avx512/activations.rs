// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 dispatch macro for activation functions (same API surface as AVX2).
//!
//! Provides both `impl_avx512_activations!()` (standard AVX-512F/VL)
//! and `impl_avx512vnni_bf16_activations!()` (VNNI-BF16 path) macros.
//! Delegates to AVX-512 kernels in [`crate::math::wavenet::accumulate`]
//! and [`crate::math::activations`]. Invoked by [`dispatch_simd!`].

macro_rules! impl_avx512_activations {
    () => {
        #[inline(always)]
        // SAFETY: dest and src are valid slices with dest.len() <= src.len();
        // CPU supports AVX-512F+VL (verified by dispatch). Kernel uses unaligned loads/stores.
        unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
            // SAFETY: dest and src satisfy the function's documented invariants.
            unsafe { crate::math::wavenet::accumulate::accumulate_head_avx512vl(dest, src) }
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices with head_input.len() >= block.len();
        // CPU supports AVX-512F+VL (verified by dispatch). Kernel uses unaligned loads/stores.
        unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
            // SAFETY: head_input and block satisfy the function's documented invariants.
            unsafe {
                crate::math::wavenet::accumulate::tanh_and_accumulate_block_avx512vl(
                    head_input, block,
                )
            }
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices with head_input.len() >= block.len();
        // CPU supports AVX-512F+VL (verified by dispatch). Kernel uses unaligned loads/stores.
        unsafe fn relu_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
            // SAFETY: head_input and block satisfy the function's documented invariants.
            unsafe {
                crate::math::wavenet::accumulate::relu_and_accumulate_block_avx512vl(
                    head_input, block,
                )
            }
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices with head_input.len() >= block.len();
        // block.len() % ch == 0; CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn gated_activation_and_accumulate_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            // SAFETY: head_input, block, and ch satisfy the function's documented invariants.
            unsafe {
                crate::math::wavenet::accumulate::gated_activation_and_accumulate_block_avx512vl(
                    head_input, block, ch,
                )
            }
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices of equal length;
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn tanh_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]) {
            // SAFETY: head_input and block satisfy the function's documented invariants.
            unsafe {
                crate::math::wavenet::accumulate::tanh_and_overwrite_block_avx512vl(
                    head_input, block,
                )
            }
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices of equal length;
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn relu_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]) {
            // SAFETY: head_input and block satisfy the function's documented invariants.
            unsafe {
                crate::math::wavenet::accumulate::relu_and_overwrite_block_avx512vl(
                    head_input, block,
                )
            }
        }

        #[inline(always)]
        // SAFETY: head_input, block, and seed are valid slices of equal length;
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn tanh_and_accumulate_with_seed(
            head_input: &mut [f32],
            block: &mut [f32],
            seed: &[f32],
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                crate::math::wavenet::accumulate::tanh_and_accumulate_with_seed_avx512vl(
                    head_input, block, seed,
                )
            }
        }

        #[inline(always)]
        // SAFETY: head_input, block, and seed are valid slices of equal length;
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn relu_and_accumulate_with_seed(
            head_input: &mut [f32],
            block: &mut [f32],
            seed: &[f32],
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                crate::math::wavenet::accumulate::relu_and_accumulate_with_seed_avx512vl(
                    head_input, block, seed,
                )
            }
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices of equal length;
        // block.len() % ch == 0; CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn gated_activation_and_overwrite_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            // SAFETY: head_input, block, and ch satisfy the function's documented invariants.
            unsafe {
                crate::math::wavenet::accumulate::gated_activation_and_overwrite_block_avx512vl(
                    head_input, block, ch,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512F (verified by dispatch).
        unsafe fn tanh_slice(slice: &mut [f32]) {
            crate::math::activations::tanh_slice_avx512(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512F (verified by dispatch).
        unsafe fn sigmoid_slice(slice: &mut [f32]) {
            crate::math::activations::sigmoid_slice_avx512(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn tanh_slice_hf(slice: &mut [f32]) {
            crate::math::activations::tanh::high_fidelity::tanh_poly_slice_avx512(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn sigmoid_slice_hf(slice: &mut [f32]) {
            crate::math::activations::sigmoid::high_fidelity::sigmoid_poly_slice_avx512(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512F (verified by dispatch).
        unsafe fn relu_slice(slice: &mut [f32]) {
            crate::math::activations::relu_slice_avx512(slice)
        }

        #[inline(always)]
        // SAFETY: slice and slopes are valid f32 slices with slopes.len() >= slice.len();
        // CPU supports AVX-512F (verified by dispatch).
        unsafe fn prelu_slice(slice: &mut [f32], slopes: &[f32]) {
            crate::math::activations::prelu_slice_avx512(slice, slopes)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512F (verified by dispatch).
        unsafe fn softsign_slice(slice: &mut [f32]) {
            crate::math::activations::softsign_slice_avx512(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512F (verified by dispatch).
        unsafe fn silu_slice(slice: &mut [f32]) {
            crate::math::activations::silu_slice_avx512(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn silu_slice_hf(slice: &mut [f32]) {
            crate::math::activations::silu_poly_slice_avx512(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512F (verified by dispatch).
        unsafe fn hard_tanh_slice(slice: &mut [f32]) {
            crate::math::activations::hard_tanh_slice_avx512(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512F (verified by dispatch).
        unsafe fn hard_swish_slice(slice: &mut [f32]) {
            crate::math::activations::hard_swish_slice_avx512(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512F (verified by dispatch).
        unsafe fn fast_tanh_slice(slice: &mut [f32]) {
            crate::math::activations::fast_tanh_slice_avx512(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; all scalar params (min_val, max_val,
        // min_slope, max_slope) are finite f32 values; CPU supports AVX-512F (verified by dispatch).
        unsafe fn leaky_hard_tanh_slice(
            slice: &mut [f32],
            min_val: f32,
            max_val: f32,
            min_slope: f32,
            max_slope: f32,
        ) {
            crate::math::activations::leaky_hard_tanh_slice_avx512(
                slice, min_val, max_val, min_slope, max_slope,
            )
        }

        #[inline(always)]
        // SAFETY: buf is a valid mutable f32 buffer; CPU supports AVX-512F (verified by dispatch).
        unsafe fn activation_tanh_block(buf: &mut [f32]) {
            crate::math::activations::tanh_slice_avx512(buf)
        }

        #[inline(always)]
        // SAFETY: gates, cell_state, and hidden_state are valid mutable f32 slices with
        // gates.len() == 4 * hidden_size, cell_state.len() == hidden_size,
        // hidden_state.len() == hidden_size; CPU supports AVX-512F (verified by dispatch).
        unsafe fn fused_lstm_gates_dyn(
            gates: &mut [f32],
            cell_state: &mut [f32],
            cell_error: &mut [f32],
            hidden_state: &mut [f32],
            hidden_size: usize,
        ) {
            // SAFETY: gates, cell_state, hidden_state, and hidden_size satisfy the function's
            // documented invariants.
            unsafe {
                crate::math::lstm::fused_lstm_gates_dyn_avx512(
                    gates,
                    cell_state,
                    cell_error,
                    hidden_state,
                    hidden_size,
                )
            }
        }
    };
}

macro_rules! impl_avx512vnni_bf16_activations {
    () => {
        #[inline(always)]
        // SAFETY: dest and src are valid slices; CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
            Avx512Math::accumulate_head(dest, src)
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices; CPU supports AVX-512 VNNI+BF16.
        unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
            Avx512Math::tanh_and_accumulate_block(head_input, block)
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices; CPU supports AVX-512 VNNI+BF16.
        unsafe fn relu_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
            Avx512Math::relu_and_accumulate_block(head_input, block)
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices; block.len() % ch == 0;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn gated_activation_and_accumulate_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            Avx512Math::gated_activation_and_accumulate_block(head_input, block, ch)
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices of equal length;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn tanh_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]) {
            Avx512Math::tanh_and_overwrite_block(head_input, block)
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices of equal length;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn relu_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]) {
            Avx512Math::relu_and_overwrite_block(head_input, block)
        }

        #[inline(always)]
        // SAFETY: head_input, block, and seed are valid slices of equal length;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn tanh_and_accumulate_with_seed(
            head_input: &mut [f32],
            block: &mut [f32],
            seed: &[f32],
        ) {
            Avx512Math::tanh_and_accumulate_with_seed(head_input, block, seed)
        }

        #[inline(always)]
        // SAFETY: head_input, block, and seed are valid slices of equal length;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn relu_and_accumulate_with_seed(
            head_input: &mut [f32],
            block: &mut [f32],
            seed: &[f32],
        ) {
            Avx512Math::relu_and_accumulate_with_seed(head_input, block, seed)
        }

        #[inline(always)]
        // SAFETY: head_input and block are valid mutable slices of equal length;
        // block.len() % ch == 0; CPU supports AVX-512 VNNI+BF16.
        unsafe fn gated_activation_and_overwrite_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            Avx512Math::gated_activation_and_overwrite_block(head_input, block, ch)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512 VNNI+BF16.
        unsafe fn tanh_slice(slice: &mut [f32]) {
            Avx512Math::tanh_slice(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512 VNNI+BF16.
        unsafe fn sigmoid_slice(slice: &mut [f32]) {
            Avx512Math::sigmoid_slice(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512 VNNI+BF16.
        unsafe fn tanh_slice_hf(slice: &mut [f32]) {
            Avx512Math::tanh_slice_hf(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512 VNNI+BF16.
        unsafe fn sigmoid_slice_hf(slice: &mut [f32]) {
            Avx512Math::sigmoid_slice_hf(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512 VNNI+BF16.
        unsafe fn relu_slice(slice: &mut [f32]) {
            Avx512Math::relu_slice(slice)
        }

        #[inline(always)]
        // SAFETY: slice and slopes are valid f32 slices; CPU supports AVX-512 VNNI+BF16.
        unsafe fn prelu_slice(slice: &mut [f32], slopes: &[f32]) {
            Avx512Math::prelu_slice(slice, slopes)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512 VNNI+BF16.
        unsafe fn softsign_slice(slice: &mut [f32]) {
            Avx512Math::softsign_slice(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512 VNNI+BF16.
        unsafe fn silu_slice(slice: &mut [f32]) {
            Avx512Math::silu_slice(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512 VNNI+BF16.
        unsafe fn hard_tanh_slice(slice: &mut [f32]) {
            Avx512Math::hard_tanh_slice(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512 VNNI+BF16.
        unsafe fn hard_swish_slice(slice: &mut [f32]) {
            Avx512Math::hard_swish_slice(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; CPU supports AVX-512 VNNI+BF16.
        unsafe fn fast_tanh_slice(slice: &mut [f32]) {
            Avx512Math::fast_tanh_slice(slice)
        }

        #[inline(always)]
        // SAFETY: slice is a valid mutable f32 buffer; all scalar params are finite f32;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn leaky_hard_tanh_slice(
            slice: &mut [f32],
            min_val: f32,
            max_val: f32,
            min_slope: f32,
            max_slope: f32,
        ) {
            Avx512Math::leaky_hard_tanh_slice(slice, min_val, max_val, min_slope, max_slope)
        }

        #[inline(always)]
        // SAFETY: buf is a valid mutable f32 buffer; CPU supports AVX-512 VNNI+BF16.
        unsafe fn activation_tanh_block(buf: &mut [f32]) {
            Avx512Math::activation_tanh_block(buf)
        }

        #[inline(always)]
        // SAFETY: gates, cell_state, and hidden_state are valid mutable f32 slices;
        // gates.len() == 4 * hidden_size, cell_state.len() == hidden_state.len() == hidden_size;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn fused_lstm_gates_dyn(
            gates: &mut [f32],
            cell_state: &mut [f32],
            cell_error: &mut [f32],
            hidden_state: &mut [f32],
            hidden_size: usize,
        ) {
            Avx512Math::fused_lstm_gates_dyn(
                gates,
                cell_state,
                cell_error,
                hidden_state,
                hidden_size,
            )
        }
    };
}

pub(super) use impl_avx512_activations;
pub(super) use impl_avx512vnni_bf16_activations;
