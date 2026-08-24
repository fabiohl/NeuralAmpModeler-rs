// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::{AlignedVec, SimdMath};

/// 1x1 Dense Layer (The Channel Mixer):
/// Think of this layer as a 'digital mixing console'. It blends the various
/// audio channels coming from the previous stage to create the final timbre combination.
#[derive(Clone)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    /// Weight matrix: Defines 'how much' of each channel goes into the mix.
    pub weights: AlignedVec<f32>,
    /// Bias: A basic 'volume' adjustment for each output channel.
    pub bias: AlignedVec<f32>,
    /// Flag indicating whether bias should be applied.
    pub do_bias: bool,
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    /// Full-precision f32 fused residual batch.
    ///
    /// Fuses the 1x1 GEMV with bias and residual addition into a single SIMD
    /// pass using the native f32 weight tensor.
    ///
    /// # Safety
    /// The caller must guarantee compatible sizes and buffer validity.
    #[inline(always)]
    pub unsafe fn process_residual_batch<M: SimdMath>(
        &self,
        input: &[f32],
        residual: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        // SAFETY: the const-generic GEMM kernels are only reached for their matching `IN`/`OUT`
        // shapes with buffers validated by this wrapper's caller contract, and
        // `M::fused_gemm_residual_batch_f32` requires lane counts satisfied by `input`/`output`
        // sized `num_frames` frames of `IN`/`OUT` (caller-guaranteed); `M` is dispatched by
        // runtime CPUID feature checks matching its `#[target_feature]` backend.
        unsafe {
            // Slots for which we have specialised const-generic kernels
            // (endereçamento imediato — no leaq chains).
            #[cfg(target_arch = "x86_64")]
            {
                if IN == 4 && OUT == 4 {
                    crate::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_const::<4, 4>(
                        input,
                        &self.weights,
                        &self.bias,
                        residual,
                        output,
                        num_frames,
                        self.do_bias,
                    );
                    return;
                }
                if IN == 8 && OUT == 8 {
                    crate::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_const::<8, 8>(
                        input,
                        &self.weights,
                        &self.bias,
                        residual,
                        output,
                        num_frames,
                        self.do_bias,
                    );
                    return;
                }
                if IN == 12 && OUT == 12 {
                    crate::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_12x12(
                        input,
                        &self.weights,
                        &self.bias,
                        residual,
                        output,
                        num_frames,
                        self.do_bias,
                    );
                    return;
                }
                if IN == 16 && OUT == 16 {
                    crate::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_const::<16, 16>(
                        input,
                        &self.weights,
                        &self.bias,
                        residual,
                        output,
                        num_frames,
                        self.do_bias,
                    );
                    return;
                }
            }
            M::fused_gemm_residual_batch_f32(
                input,
                &self.weights,
                &self.bias,
                residual,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    /// Full-precision f32 head projection.
    ///
    /// Dispatches to the appropriate SIMD kernel via the `SimdMath` trait,
    /// replacing the previous scalar triple-nested loop with shape-dependent
    /// vectorization (frame-batching for OUT≤4, channel-batching for OUT≥8).
    ///
    /// # Safety
    /// The caller must ensure that `in_frame` and `out_frame` have sizes
    /// compatible with `IN`, `OUT`, and `num_frames`, and that the SIMD
    /// instructions for `M` are available on the host CPU.
    #[inline(always)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        // SAFETY: the AVX2 broadcast GEMV kernels are only reached under `#[cfg(target_arch =
        // "x86_64")]` with the target-feature guarantee of the caller's dispatch path, and the
        // buffer sizes (`IN`/`OUT` per frame, `num_frames` frames) satisfy the kernel lane counts
        // per this wrapper's documented contract; the generic `M::gemv_*` paths require the same
        // sizes plus `M`'s backend features validated by the runtime CPUID dispatch.
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                if IN == 1 {
                    if self.do_bias {
                        crate::math::gemm::gemv::broadcast_scale_with_bias_f32_avx2::<OUT>(
                            input,
                            &self.weights,
                            &self.bias,
                            output,
                        );
                    } else {
                        crate::math::gemm::gemv::broadcast_scale_f32_avx2::<OUT>(
                            input,
                            &self.weights,
                            output,
                        );
                    }
                    return;
                }
            }
            if self.do_bias {
                M::gemv_with_bias_f32(input, &self.weights, &self.bias, output, num_frames);
            } else {
                M::gemv_no_bias_f32(input, &self.weights, output, num_frames);
            }
        }
    }
}
