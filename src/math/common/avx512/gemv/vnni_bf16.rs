// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 VNNI+BF16 dispatch macro for GEMV/GEMM kernels
//! (`impl_avx512vnni_bf16_gemv!()`).
//!
//! Uses 512-bit VNNI-BF16 dot-product instructions for 2× throughput
//! over the base AVX-512 f32 path when the ISA is available.
//! Invoked by [`dispatch_simd!`] when `IS_BF16=true`.

macro_rules! impl_avx512vnni_bf16_gemv {
    () => {
        #[inline(always)]
        // SAFETY: a and b are valid slices; CPU supports AVX-512F+VL+F16C (verified by dispatch).
        unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
            crate::math::gemm::dot_basic::dot_product_avx512(a, b)
        }
        #[inline(always)]
        // SAFETY: a and b are valid u16 (BF16) slices; CPU supports AVX-512 VNNI+BF16
        // (verified by dispatch).
        unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
            // SAFETY: a and b satisfy the function's documented invariants.
            unsafe { crate::math::gemm::dot_basic::dot_product_bf16_avx512(a, b) }
        }
        #[inline(always)]
        // SAFETY: weights and state are valid slices; CPU supports AVX-512 VNNI+BF16.
        unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
            crate::math::gemm::dot_4x::dot_product_4x_interleaved_avx512(weights, state)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1 are valid slices; CPU supports AVX-512 VNNI+BF16.
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
        // SAFETY: weights and state are valid slices; CPU supports AVX-512 VNNI+BF16.
        unsafe fn dot_product_4x_f32(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
            crate::math::gemm::dot_4x::dot_product_4x_f32_avx512vl(weights, state)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1 are valid slices; CPU supports AVX-512 VNNI+BF16.
        unsafe fn dot_product_4x_f32_dual(
            weights: &[[f32; 4]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 4], [f32; 4]) {
            crate::math::gemm::dot_4x::dot_product_4x_f32_dual_avx512vl(weights, state_f0, state_f1)
        }
        #[inline(always)]
        // SAFETY: weights and state are valid slices; CPU supports AVX-512 VNNI+BF16.
        unsafe fn dot_product_8x_f32(weights: &[[f32; 8]], state: &[f32]) -> [f32; 8] {
            crate::math::gemm::dot_8x::dot_product_8x_f32_avx2(weights, state)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1 are valid slices;
        // CPU supports AVX-512 VNNI+BF16 (verified by dispatch). AVX-512 implies AVX2/FMA,
        // so the 8x dual-frame AVX2 kernel is valid to call from this context.
        unsafe fn dot_product_8x_f32_dual(
            weights: &[[f32; 8]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 8], [f32; 8]) {
            crate::math::gemm::dot_8x::dot_product_8x_f32_dual_avx2(weights, state_f0, state_f1)
        }
        #[inline(always)]
        // SAFETY: weights and state are valid slices; CPU supports AVX-512 VNNI+BF16.
        unsafe fn dot_product_16x_f32(weights: &[[f32; 16]], state: &[f32]) -> [f32; 16] {
            crate::math::gemm::dot_16x::dot_product_16x_f32_avx512(weights, state)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1 are valid slices;
        // CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn dot_product_16x_f32_dual(
            weights: &[[f32; 16]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 16], [f32; 16]) {
            crate::math::gemm::dot_16x::dot_product_16x_f32_dual_avx512(weights, state_f0, state_f1)
        }
        #[inline(always)]
        // SAFETY: weights, state, init are valid slices; weights.len() >= state.len();
        // CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn dot_product_4x_f32_accumulate(
            weights: &[[f32; 4]],
            state: &[f32],
            init: &[f32; 4],
        ) -> [f32; 4] {
            crate::math::gemm::dot_4x::dot_product_4x_f32_accumulate_avx512vl(weights, state, init)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1, init_f0, init_f1 are valid slices;
        // CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn dot_product_4x_f32_dual_accumulate(
            weights: &[[f32; 4]],
            state_f0: &[f32],
            state_f1: &[f32],
            init_f0: &[f32; 4],
            init_f1: &[f32; 4],
        ) -> ([f32; 4], [f32; 4]) {
            crate::math::gemm::dot_4x::dot_product_4x_f32_dual_accumulate_avx512vl(
                weights, state_f0, state_f1, init_f0, init_f1,
            )
        }
        #[inline(always)]
        // SAFETY: weights, state, init are valid slices; weights.len() >= state.len();
        // AVX-512 VNNI+BF16 implies AVX2/FMA — delegates to fused AVX2 8x accumulate.
        unsafe fn dot_product_8x_f32_accumulate(
            weights: &[[f32; 8]],
            state: &[f32],
            init: &[f32; 8],
        ) -> [f32; 8] {
            crate::math::gemm::dot_8x::dot_product_8x_f32_accumulate_avx2(weights, state, init)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1, init_f0, init_f1 are valid slices;
        // AVX-512 VNNI+BF16 implies AVX2/FMA — delegates to fused AVX2 8x dual accumulate.
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
        // CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn dot_product_16x_f32_accumulate(
            weights: &[[f32; 16]],
            state: &[f32],
            init: &[f32; 16],
        ) -> [f32; 16] {
            crate::math::gemm::dot_16x::dot_product_16x_f32_accumulate_avx512(weights, state, init)
        }
        #[inline(always)]
        // SAFETY: weights, state_f0, state_f1, init_f0, init_f1 are valid slices;
        // CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
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
        // SAFETY: four weight slices and in_frame are valid u16 slices of equal length.
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
        // SAFETY: in_frame, weights, bias, out_frame are valid slices;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn fused_add_gemv(
            in_frame: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frame: &mut [f32],
            do_bias: bool,
        ) {
            Avx512Math::fused_add_gemv(in_frame, weights, bias, out_frame, do_bias)
        }
        #[inline(always)]
        // SAFETY: in_frames, weights, bias, out_frames are valid slices;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn fused_add_gemm_batch(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            Avx512Math::fused_add_gemm_batch(
                in_frames, weights, bias, out_frames, num_frames, do_bias,
            )
        }
        #[inline(always)]
        // SAFETY: in_frames, weights, bias, residual, out_frames are valid slices;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn fused_gemm_residual_batch(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            residual: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            Avx512Math::fused_gemm_residual_batch(
                in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
            )
        }
        #[inline(always)]
        // SAFETY: in_frames, weights (f32), bias, residual, out_frames are valid slices;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn fused_gemm_residual_batch_f32(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            residual: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            Avx512Math::fused_gemm_residual_batch_f32(
                in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
            )
        }
        #[inline(always)]
        // SAFETY: in_frame, weights, bias, out_frame are valid slices;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn gemv_overwrite(
            in_frame: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frame: &mut [f32],
            do_bias: bool,
        ) {
            Avx512Math::gemv_overwrite(in_frame, weights, bias, out_frame, do_bias)
        }
        #[inline(always)]
        // SAFETY: in_frame (u16 BF16), weights (u16 BF16), bias (f32), out_frame (f32) are
        // valid slices; CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn gemv_overwrite_bf16(
            in_frame: &[u16],
            weights: &[u16],
            bias: &[f32],
            out_frame: &mut [f32],
            do_bias: bool,
        ) {
            crate::math::gemm::gemv_bf16::gemv_overwrite_bf16_avx512(
                in_frame, weights, bias, out_frame, do_bias,
            )
        }
        #[inline(always)]
        // SAFETY: in_frame, weights (4-gate concatenated), bias, out_gates are valid slices;
        // CPU supports AVX-512 VNNI+BF16.
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
        // valid slices; CPU supports AVX-512 VNNI+BF16.
        unsafe fn gemv_overwrite_bf16_4gate(
            _in_frame: &[u16],
            _weights: &[u16],
            _bias: &[f32],
            _out_gates: &mut [f32],
            _hidden_size: usize,
            _do_bias: bool,
        ) {
            unreachable!(
                "gemv_overwrite_bf16_4gate is unused; 4-gate dispatch uses direct kernel functions"
            );
        }
        #[inline(always)]
        // SAFETY: in_frames, weights, bias, out_frames are valid slices;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn gemv_overwrite_batch(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            Avx512Math::gemv_overwrite_batch(
                in_frames, weights, bias, out_frames, num_frames, do_bias,
            )
        }
        #[inline(always)]
        // SAFETY: in_frames, weights (f32), bias, out_frames are valid slices;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn gemv_with_bias_f32(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
        ) {
            // SAFETY: arguments satisfy the documented invariants.
            unsafe {
                crate::math::gemm::gemv::gemv_with_bias_f32_avx512(
                    in_frames, weights, bias, out_frames, num_frames,
                )
            }
        }
        #[inline(always)]
        // SAFETY: in_frames, weights (f32), out_frames are valid slices;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn gemv_no_bias_f32(
            in_frames: &[f32],
            weights: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
        ) {
            // SAFETY: arguments satisfy the documented invariants.
            unsafe {
                crate::math::gemm::gemv::gemv_no_bias_f32_avx512(
                    in_frames, weights, out_frames, num_frames,
                )
            }
        }
    };
}
