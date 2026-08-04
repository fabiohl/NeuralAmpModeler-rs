// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::{AlignedVec, SimdMath};

/// 1x1 Dense Layer (The Channel Mixer) with runtime dimensions.
///
/// Think of this layer as a 'digital mixing console'. It blends the various
/// audio channels coming from the previous stage to create the final timbre combination.
#[derive(Clone)]
pub struct DenseLayerDyn {
    /// Number of input channels.
    pub in_ch: usize,
    /// Number of output channels.
    pub out_ch: usize,
    /// Weight matrix: Defines 'how much' of each channel goes into the mix.
    pub weights: AlignedVec<f32>,
    /// Bias: A basic 'volume' adjustment for each output channel.
    pub bias: AlignedVec<f32>,
    /// Flag indicating whether bias should be applied.
    pub do_bias: bool,
}

impl DenseLayerDyn {
    /// Residual Sum (The Final 'Shortcut'):
    /// This function mixes channels AND adds the original sound
    /// (residual) to the result, all without needing to copy extra data in memory.
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
        unsafe {
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
    /// The caller must ensure that `input` and `output` have sizes
    /// compatible with `in_ch`, `out_ch`, and `num_frames`, and that the SIMD
    /// instructions for `M` are available on the host CPU.
    #[inline(always)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            if self.do_bias {
                M::gemv_with_bias_f32(input, &self.weights, &self.bias, output, num_frames);
            } else {
                M::gemv_no_bias_f32(input, &self.weights, output, num_frames);
            }
        }
    }
}
