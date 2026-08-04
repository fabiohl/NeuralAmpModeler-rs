// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX2 (x86-64-v3) dispatch macro for GEMV/GEMM/dot-product kernels.
//!
//! Covers scalar `dot_product`, interleaved 4× and 1× dot products,
//! dual-frame interleaved, `gemm_vec` (row-major column batch), bf16
//! fallback stubs, and transpose helpers. Delegates to
//! [`crate::math::gemm`] AVX2 kernels. Invoked by [`dispatch_simd!`].

macro_rules! impl_avx2_gemv {
    () => {
        // Dot Product: Multiplies weights by signal and sums the result (the "DNA" of neural networks).
        // In AVX2, we use 256-bit registers that process 8 numbers at once.
        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::gemm::dot_basic::dot_product_avx2(a, b) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::gemm::dot_4x::dot_product_4x_interleaved_avx2(weights, state) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_4x_interleaved_dual_frame(
            weights: &[[u16; 4]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 4], [f32; 4]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::dot_4x::dot_product_4x_interleaved_dual_frame_avx2(
                    weights, state_f0, state_f1,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_4x_f32(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::gemm::dot_4x::dot_product_4x_f32_avx2(weights, state) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_4x_f32_dual(
            weights: &[[f32; 4]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 4], [f32; 4]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::dot_4x::dot_product_4x_f32_dual_avx2(
                    weights, state_f0, state_f1,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_8x_f32(weights: &[[f32; 8]], state: &[f32]) -> [f32; 8] {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::gemm::dot_8x::dot_product_8x_f32_avx2(weights, state) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_8x_f32_dual(
            weights: &[[f32; 8]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 8], [f32; 8]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::dot_8x::dot_product_8x_f32_dual_avx2(
                    weights, state_f0, state_f1,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_16x_f32(weights: &[[f32; 16]], state: &[f32]) -> [f32; 16] {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::gemm::dot_16x::dot_product_16x_f32_avx2(weights, state) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_16x_f32_dual(
            weights: &[[f32; 16]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 16], [f32; 16]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::dot_16x::dot_product_16x_f32_dual_avx2(
                    weights, state_f0, state_f1,
                )
            }
        }

        // --- Fused accumulate dot products (bias+mixin base init) ---

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_4x_f32_accumulate(
            weights: &[[f32; 4]],
            state: &[f32],
            init: &[f32; 4],
        ) -> [f32; 4] {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::dot_4x::dot_product_4x_f32_accumulate_avx2(weights, state, init)
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_4x_f32_dual_accumulate(
            weights: &[[f32; 4]],
            state_f0: &[f32],
            state_f1: &[f32],
            init_f0: &[f32; 4],
            init_f1: &[f32; 4],
        ) -> ([f32; 4], [f32; 4]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::dot_4x::dot_product_4x_f32_dual_accumulate_avx2(
                    weights, state_f0, state_f1, init_f0, init_f1,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_8x_f32_accumulate(
            weights: &[[f32; 8]],
            state: &[f32],
            init: &[f32; 8],
        ) -> [f32; 8] {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::dot_8x::dot_product_8x_f32_accumulate_avx2(weights, state, init)
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_8x_f32_dual_accumulate(
            weights: &[[f32; 8]],
            state_f0: &[f32],
            state_f1: &[f32],
            init_f0: &[f32; 8],
            init_f1: &[f32; 8],
        ) -> ([f32; 8], [f32; 8]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::dot_8x::dot_product_8x_f32_dual_accumulate_avx2(
                    weights, state_f0, state_f1, init_f0, init_f1,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_16x_f32_accumulate(
            weights: &[[f32; 16]],
            state: &[f32],
            init: &[f32; 16],
        ) -> [f32; 16] {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::dot_16x::dot_product_16x_f32_accumulate_avx2(
                    weights, state, init,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_16x_f32_dual_accumulate(
            weights: &[[f32; 16]],
            state_f0: &[f32],
            state_f1: &[f32],
            init_f0: &[f32; 16],
            init_f1: &[f32; 16],
        ) -> ([f32; 16], [f32; 16]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::dot_16x::dot_product_16x_f32_dual_accumulate_avx2(
                    weights, state_f0, state_f1, init_f0, init_f1,
                )
            }
        }

        // GEMV Operations: Matrix-Vector multiplication, used in almost all model layers.
        // The "fused" prefix indicates that the Bias vector addition is combined (fused) with the multiplication
        // to save memory accesses and processor instructions.
        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn fused_add_gemv(
            in_frame: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frame: &mut [f32],
            do_bias: bool,
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                // Delegates the computation to the optimized AVX2 matrix-vector multiplication kernel.
                super::super::gemm::gemv::fused_add_gemv_avx2(
                    in_frame, weights, bias, out_frame, do_bias,
                )
            }
        }

        /// Performs matrix multiplication on a batch of vectors via AVX2.
        /// Useful when processing multiple audio frames concurrently to reduce overheads.
        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn fused_add_gemm_batch(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                // Delegates the batch matrix-matrix multiplication (GEMM) computation to the AVX2 kernel.
                super::super::gemm::gemm_batch::fused_add_gemm_batch_avx2(
                    in_frames, weights, bias, out_frames, num_frames, do_bias,
                )
            }
        }

        /// Performs matrix-vector multiplication also adding the residual connection (skip connection)
        /// from the previous layer. Widely used in the WaveNet residual block architecture.
        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn fused_gemm_residual_batch(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            residual: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                // Delegates the multiplication with integrated residual sum and bias to the AVX2 kernel.
                super::super::gemm::gemm_batch::fused_gemm_residual_batch_avx2(
                    in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn fused_gemm_residual_batch_f32(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            residual: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::gemm_batch::fused_gemm_residual_batch_f32_avx2(
                    in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
                )
            }
        }

        /// Version that overwrites the output buffer directly with the matrix-vector multiplication result,
        /// without accumulating with pre-existing values in the buffer.
        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn gemv_overwrite(
            in_frame: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frame: &mut [f32],
            do_bias: bool,
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::gemv::gemv_overwrite_avx2(
                    in_frame, weights, bias, out_frame, do_bias,
                )
            }
        }

        /// Version that overwrites the output buffer accepting input data represented in BF16 (16-bit)
        /// and BF16 weights, performing accumulation in f32 to preserve fidelity.
        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn gemv_overwrite_bf16(
            _in_frame: &[u16],
            _weights: &[u16],
            _bias: &[f32],
            _out_frame: &mut [f32],
            _do_bias: bool,
        ) {
            unreachable!("AVX2 IS_BF16=false; BF16 paths are never reached at runtime")
        }

        // LSTM Gates (4-gate): Simultaneously computes the 4 memory controls of the LSTM network.
        // Gate computation (Input, Forget, Cell Candidate, and Output) shares the same input
        // states. Computing them in parallel drastically reduces cache jumps.
        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
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

        /// Equivalent to `gemv_overwrite_4gate` but processing input data represented
        /// in the BF16 reduced precision format.
        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn gemv_overwrite_bf16_4gate(
            _in_frame: &[u16],
            _weights: &[u16],
            _bias: &[f32],
            _out_gates: &mut [f32],
            _hidden_size: usize,
            _do_bias: bool,
        ) {
            unreachable!("AVX2 IS_BF16=false; BF16 paths are never reached at runtime")
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn gemv_overwrite_batch(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            let in_len = in_frames.len() / num_frames;
            let out_len = out_frames.len() / num_frames;
            for i in 0..num_frames {
                let in_slice = &in_frames[i * in_len..(i + 1) * in_len];
                let out_slice = &mut out_frames[i * out_len..(i + 1) * out_len];
                // SAFETY: in_slice and out_slice are valid sub-slices of the batch arrays;
                // AVX2+FMA ISA verified by caller via dispatch.
                unsafe {
                    super::super::gemm::gemv::gemv_overwrite_avx2(
                        in_slice, weights, bias, out_slice, do_bias,
                    )
                };
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn gemv_with_bias_f32(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::gemv::gemv_with_bias_f32_avx2(
                    in_frames, weights, bias, out_frames, num_frames,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn gemv_no_bias_f32(
            in_frames: &[f32],
            weights: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::gemm::gemv::gemv_no_bias_f32_avx2(
                    in_frames, weights, out_frames, num_frames,
                )
            }
        }

        #[inline(always)]
        // SAFETY: data, scale, offset are valid f32 slices; n_ch * num_frames == data.len();
        // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch). Kernel uses unaligned loads/stores.
        unsafe fn batch_norm_process(
            data: &mut [f32],
            scale: &[f32],
            offset: &[f32],
            n_ch: usize,
            num_frames: usize,
        ) {
            for f in 0..num_frames {
                let frame_start = f * n_ch;
                let mut c = 0;
                while c + 8 <= n_ch {
                    // SAFETY: c is bounds-checked (c+8 <= n_ch); frame_start+c is within data
                    // bounds (n_ch * num_frames). Unaligned 256-bit loads/stores valid for f32.
                    unsafe {
                        let x = _mm256_loadu_ps(data.as_ptr().add(frame_start + c));
                        let s = _mm256_loadu_ps(scale.as_ptr().add(c));
                        let o = _mm256_loadu_ps(offset.as_ptr().add(c));
                        let y = _mm256_fmadd_ps(x, s, o);
                        _mm256_storeu_ps(data.as_mut_ptr().add(frame_start + c), y);
                    }
                    c += 8;
                }
                for c in c..n_ch {
                    let idx = frame_start + c;
                    // SAFETY: c < n_ch ensures idx is within data bounds; scale/offset have
                    // at least n_ch elements (caller invariant).
                    unsafe {
                        *data.get_unchecked_mut(idx) = (*data.get_unchecked(idx))
                            .mul_add(*scale.get_unchecked(c), *offset.get_unchecked(c));
                    }
                }
            }
        }
    };
}
pub(crate) use impl_avx2_gemv;
