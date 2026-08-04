// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 dispatch macro for GEMV/GEMM/dot-product kernels (`impl_avx512_gemv!()`).
//!
//! Covers scalar `dot_product`, interleaved 4× and 1× variants,
//! dual-frame dispatch, `gemm_vec`, bf16 fallback stubs, and transpose.
//! Delegates to avx512-intrinsics kernels in [`crate::math::gemm`].
//! Invoked by [`dispatch_simd!`].

macro_rules! impl_avx512_gemv {
    () => {
        #[inline(always)]
        // SAFETY: a and b are valid slices; CPU supports AVX-512F+VL+F16C (verified by dispatch).
        // Kernel uses min(a.len(), b.len()) as bound with unaligned 512-bit loads.
        unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
            crate::math::gemm::dot_basic::dot_product_avx512(a, b)
        }
        #[inline(always)]
        // SAFETY: a and b are valid u16 slices; no AVX-512 requirement (fallback path).
        unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
            dot_product_bf16_fallback(a, b)
        }
        #[inline(always)]
        // SAFETY: weights is a valid &[[u16; 4]] slice, state is a valid f32 slice;
        // CPU supports AVX-512F+VL (verified by dispatch). Kernel uses unaligned loads.
        unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
            crate::math::gemm::dot_4x::dot_product_4x_interleaved_avx512(weights, state)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, and state_f1 are valid slices;
        // state_f0.len() == state_f1.len() == weights.len();
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn dot_product_4x_interleaved_dual_frame(
            weights: &[[u16; 4]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 4], [f32; 4]) {
            crate::math::gemm::dot_4x::dot_product_4x_interleaved_dual_frame_avx512(
                weights, state_f0, state_f1,
            )
        }
        #[inline(always)]
        // SAFETY: weights and state are valid slices; weights.len() >= state.len();
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn dot_product_4x_f32(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
            crate::math::gemm::dot_4x::dot_product_4x_f32_avx512(weights, state)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, and state_f1 are valid slices;
        // state_f0.len() == state_f1.len() == weights.len();
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn dot_product_4x_f32_dual(
            weights: &[[f32; 4]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 4], [f32; 4]) {
            crate::math::gemm::dot_4x::dot_product_4x_f32_dual_avx512(weights, state_f0, state_f1)
        }
        #[inline(always)]
        // SAFETY: weights and state are valid slices; weights.len() >= state.len();
        // CPU supports AVX-512F (verified by dispatch). AVX-512 implies AVX2/FMA,
        // so the 8x AVX2 kernel is valid to call from this context.
        unsafe fn dot_product_8x_f32(weights: &[[f32; 8]], state: &[f32]) -> [f32; 8] {
            crate::math::gemm::dot_8x::dot_product_8x_f32_avx2(weights, state)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1 are valid slices;
        // weights.len() >= state_f0.len() and weights.len() >= state_f1.len();
        // CPU supports AVX-512F (verified by dispatch). AVX-512 implies AVX2/FMA,
        // so the 8x dual-frame AVX2 kernel is valid to call from this context.
        unsafe fn dot_product_8x_f32_dual(
            weights: &[[f32; 8]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 8], [f32; 8]) {
            crate::math::gemm::dot_8x::dot_product_8x_f32_dual_avx2(weights, state_f0, state_f1)
        }
        #[inline(always)]
        // SAFETY: weights and state are valid slices; weights.len() >= state.len();
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn dot_product_16x_f32(weights: &[[f32; 16]], state: &[f32]) -> [f32; 16] {
            crate::math::gemm::dot_16x::dot_product_16x_f32_avx512(weights, state)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1 are valid slices;
        // weights.len() >= state_f0.len() and weights.len() >= state_f1.len();
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn dot_product_16x_f32_dual(
            weights: &[[f32; 16]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 16], [f32; 16]) {
            crate::math::gemm::dot_16x::dot_product_16x_f32_dual_avx512(weights, state_f0, state_f1)
        }
        #[inline(always)]
        // SAFETY: weights, state, init are valid slices; weights.len() >= state.len();
        // CPU supports AVX-512F+VL (verified by dispatch).
        // Fused kernel: init is loaded into the first accumulator register.
        unsafe fn dot_product_4x_f32_accumulate(
            weights: &[[f32; 4]],
            state: &[f32],
            init: &[f32; 4],
        ) -> [f32; 4] {
            crate::math::gemm::dot_4x::dot_product_4x_f32_accumulate_avx512(weights, state, init)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1, init_f0, init_f1 are valid slices;
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn dot_product_4x_f32_dual_accumulate(
            weights: &[[f32; 4]],
            state_f0: &[f32],
            state_f1: &[f32],
            init_f0: &[f32; 4],
            init_f1: &[f32; 4],
        ) -> ([f32; 4], [f32; 4]) {
            crate::math::gemm::dot_4x::dot_product_4x_f32_dual_accumulate_avx512(
                weights, state_f0, state_f1, init_f0, init_f1,
            )
        }
        #[inline(always)]
        // SAFETY: weights, state, init are valid slices; weights.len() >= state.len();
        // AVX-512 implies AVX2/FMA — delegates to the fused AVX2 8x accumulate kernel.
        unsafe fn dot_product_8x_f32_accumulate(
            weights: &[[f32; 8]],
            state: &[f32],
            init: &[f32; 8],
        ) -> [f32; 8] {
            crate::math::gemm::dot_8x::dot_product_8x_f32_accumulate_avx2(weights, state, init)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1, init_f0, init_f1 are valid slices;
        // AVX-512 implies AVX2/FMA — delegates to the fused AVX2 8x dual accumulate kernel.
        unsafe fn dot_product_8x_f32_dual_accumulate(
            weights: &[[f32; 8]],
            state_f0: &[f32],
            state_f1: &[f32],
            init_f0: &[f32; 8],
            init_f1: &[f32; 8],
        ) -> ([f32; 8], [f32; 8]) {
            crate::math::gemm::dot_8x::dot_product_8x_f32_dual_accumulate_avx2(
                weights, state_f0, state_f1, init_f0, init_f1,
            )
        }
        #[inline(always)]
        // SAFETY: weights, state, init are valid slices; weights.len() >= state.len();
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn dot_product_16x_f32_accumulate(
            weights: &[[f32; 16]],
            state: &[f32],
            init: &[f32; 16],
        ) -> [f32; 16] {
            crate::math::gemm::dot_16x::dot_product_16x_f32_accumulate_avx512(weights, state, init)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1, init_f0, init_f1 are valid slices;
        // CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn dot_product_16x_f32_dual_accumulate(
            weights: &[[f32; 16]],
            state_f0: &[f32],
            state_f1: &[f32],
            init_f0: &[f32; 16],
            init_f1: &[f32; 16],
        ) -> ([f32; 16], [f32; 16]) {
            crate::math::gemm::dot_16x::dot_product_16x_f32_dual_accumulate_avx512(
                weights, state_f0, state_f1, init_f0, init_f1,
            )
        }
        #[inline(always)]
        // SAFETY: four weight slices and in_frame are valid u16 slices of equal length;
        // fallback path, no AVX-512 requirement.
        unsafe fn dot_product_bf16_4x(
            w0: &[u16],
            w1: &[u16],
            w2: &[u16],
            w3: &[u16],
            in_frame: &[u16],
        ) -> [f32; 4] {
            dot_product_bf16_4x_fallback(w0, w1, w2, w3, in_frame)
        }
        #[inline(always)]
        // SAFETY: in_frame, weights, bias, out_frame are valid slices with
        // weights.len() >= in_frame.len() * out_frame.len(), bias.len() >= out_frame.len();
        // CPU supports AVX-512F+VL+F16C (verified by dispatch).
        unsafe fn fused_add_gemv(
            in_frame: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frame: &mut [f32],
            do_bias: bool,
        ) {
            if out_frame.len() == 16 {
                crate::math::gemm::gemv::fused_add_gemv_avx512_small(
                    in_frame, weights, bias, out_frame, do_bias,
                )
            } else {
                fused_add_gemv_fallback(in_frame, weights, bias, out_frame, do_bias)
            }
        }
        #[inline(always)]
        // SAFETY: in_frames, weights, bias, out_frames are valid slices with lengths matching
        // num_frames * dimensions; CPU supports AVX-512F+VL+F16C (verified by dispatch).
        unsafe fn fused_add_gemm_batch(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            // SAFETY: all slice arguments satisfy the function's documented invariants.
            unsafe {
                crate::math::gemm::gemm_batch::fused_add_gemm_batch_avx512(
                    in_frames, weights, bias, out_frames, num_frames, do_bias,
                )
            }
        }
        #[inline(always)]
        // SAFETY: in_frames, weights, bias, residual, out_frames are valid slices with
        // lengths matching num_frames * dimensions; CPU supports AVX-512F+VL+F16C.
        unsafe fn fused_gemm_residual_batch(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            residual: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            // SAFETY: all slice arguments satisfy the function's documented invariants.
            unsafe {
                crate::math::gemm::gemm_batch::fused_gemm_residual_batch_avx512(
                    in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
                )
            }
        }
        #[inline(always)]
        // SAFETY: in_frames, weights, bias, residual, out_frames are valid f32 slices with
        // lengths matching num_frames * dimensions; CPU supports AVX-512F+VL.
        unsafe fn fused_gemm_residual_batch_f32(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            residual: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            // SAFETY: all slice arguments satisfy the function's documented invariants.
            unsafe {
                crate::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_avx512(
                    in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
                )
            }
        }
        #[inline(always)]
        // SAFETY: in_frame, weights, bias, out_frame are valid slices with
        // weights.len() >= in_frame.len() * out_frame.len(); CPU supports AVX-512F+VL+F16C.
        unsafe fn gemv_overwrite(
            in_frame: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frame: &mut [f32],
            do_bias: bool,
        ) {
            if out_frame.len() == 16 {
                crate::math::gemm::gemv::gemv_overwrite_avx512_small(
                    in_frame, weights, bias, out_frame, do_bias,
                )
            } else {
                gemv_overwrite_fallback(in_frame, weights, bias, out_frame, do_bias)
            }
        }
        #[inline(always)]
        // SAFETY: in_frame (u16 quantized), weights (u16), bias (f32), out_frame (f32) are
        // valid slices with matching dimensions; CPU supports AVX-512F (fallback path).
        unsafe fn gemv_overwrite_bf16(
            in_frame: &[u16],
            weights: &[u16],
            bias: &[f32],
            out_frame: &mut [f32],
            do_bias: bool,
        ) {
            gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias)
        }
        #[inline(always)]
        // SAFETY: in_frame, weights (4*gate concatenated), bias, out_gates are valid slices
        // with weights.len() == 4 * in_frame.len() * hidden_size,
        // out_gates.len() == 4 * hidden_size, bias.len() == 4 * hidden_size;
        // CPU supports AVX-512F+VL+F16C (verified by dispatch).
        unsafe fn gemv_overwrite_4gate(
            _in_frame: &[f32],
            _weights: &[u16],
            _bias: &[f32],
            _out_gates: &mut [f32],
            _hidden_size: usize,
            _do_bias: bool,
        ) {
            unreachable!(
                "gemv_overwrite_4gate is unused; 4-gate dispatch uses direct kernel functions"
            );
        }
        #[inline(always)]
        // SAFETY: in_frame (u16 BF16), weights (u16 BF16), bias (f32), out_gates (f32) are
        // valid slices with weights.len() == 4 * in_frame.len() * hidden_size,
        // out_gates.len() == 4 * hidden_size; CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn gemv_overwrite_bf16_4gate(
            in_frame: &[u16],
            weights: &[u16],
            bias: &[f32],
            out_gates: &mut [f32],
            hidden_size: usize,
            do_bias: bool,
        ) {
            let ih = in_frame.len();
            let stride = ih * hidden_size;
            // SAFETY: weights indices [0..4*stride] are within bounds per function invariants;
            // AVX-512 VNNI+BF16 ISA verified by caller via dispatch.
            unsafe {
                crate::math::gemm::gemv_4gate::gemv_4gate_bf16_avx512(
                    in_frame,
                    &weights[0..stride],
                    &weights[stride..2 * stride],
                    &weights[2 * stride..3 * stride],
                    &weights[3 * stride..4 * stride],
                    bias,
                    out_gates,
                    do_bias,
                )
            }
        }
        #[inline(always)]
        // SAFETY: in_frames, weights, bias, out_frames are valid slices with lengths
        // matching num_frames * dimensions; CPU supports AVX-512F+VL+F16C.
        unsafe fn gemv_overwrite_batch(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            // SAFETY: all slice arguments satisfy the function's documented invariants.
            unsafe {
                crate::math::gemm::gemv::gemv_overwrite_batch_avx512(
                    in_frames, weights, bias, out_frames, num_frames, do_bias,
                )
            }
        }
        #[inline(always)]
        // SAFETY: in_frames, weights (f32), bias, out_frames are valid slices with
        // lengths matching num_frames * dimensions; CPU supports AVX-512F+VL.
        unsafe fn gemv_with_bias_f32(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
        ) {
            // SAFETY: all slice arguments satisfy the function's documented invariants.
            unsafe {
                crate::math::gemm::gemv::gemv_with_bias_f32_avx512(
                    in_frames, weights, bias, out_frames, num_frames,
                )
            }
        }
        #[inline(always)]
        // SAFETY: in_frames, weights (f32), out_frames are valid slices with
        // lengths matching num_frames * dimensions; CPU supports AVX-512F+VL.
        unsafe fn gemv_no_bias_f32(
            in_frames: &[f32],
            weights: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
        ) {
            // SAFETY: all slice arguments satisfy the function's documented invariants.
            unsafe {
                crate::math::gemm::gemv::gemv_no_bias_f32_avx512(
                    in_frames, weights, out_frames, num_frames,
                )
            }
        }
    };
}
