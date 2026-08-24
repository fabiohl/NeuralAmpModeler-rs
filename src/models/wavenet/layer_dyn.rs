// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::{WAVENET_MAX_NUM_FRAMES, WavenetProcessContext};
use super::conv1d_dyn::Conv1dDyn;
use super::dense_dyn::DenseLayerDyn;
use crate::common::diagnostics::NamErrorCode;
use crate::math::common::{AlignedVec, SimdMath};
use crate::models::a2::activations::ActivationType;

/// Complete Convolutional Cell (WaveNet Layer) with runtime dimensions.
#[derive(Clone)]
pub struct WaveNetLayerDyn {
    /// This layer's parametric dilated Causal 1D convolution mesh.
    pub conv1d: Conv1dDyn,
    /// Conditional injection mixing network.
    pub input_mixin: DenseLayerDyn,
    /// 1x1 decompression linear affine transform of the layer.
    pub one_by_one: DenseLayerDyn,
    /// Activation function for this layer (default: Tanh).
    pub activation: ActivationType,
    /// Pre-allocated scratch buffer for conditioning mixin output
    /// (size: `ch * WAVENET_MAX_NUM_FRAMES`).
    pub scratch_mixin: AlignedVec<f32>,
    /// Pre-allocated scratch buffer for Conv1D + mixin intermediate results
    /// (size: `ch * WAVENET_MAX_NUM_FRAMES`).
    pub scratch_conv: AlignedVec<f32>,
}

impl WaveNetLayerDyn {
    /// Creates a new dynamic WaveNet layer.
    ///
    /// Scratch buffers are pre-allocated with size `ch * WAVENET_MAX_NUM_FRAMES`,
    /// ensuring zero allocation on the hot-path and supporting CH > 16
    /// (vs the const-generic stack buffer `[f32; 1024]` which is limited to CH ≤ 16).
    pub fn new(
        ch: usize,
        conv1d: Conv1dDyn,
        input_mixin: DenseLayerDyn,
        one_by_one: DenseLayerDyn,
    ) -> Result<Self, NamErrorCode> {
        let scratch_size = ch * WAVENET_MAX_NUM_FRAMES;
        Ok(Self {
            conv1d,
            input_mixin,
            one_by_one,
            activation: ActivationType::Tanh,
            scratch_mixin: AlignedVec::new(scratch_size, 0.0f32)?,
            scratch_conv: AlignedVec::new(scratch_size, 0.0f32)?,
        })
    }

    /// Processes a full WaveNet layer with runtime-dimensional scratch buffers.
    ///
    /// Uses dual-frame Conv1D tiling (process_block) for SIMD amortization.
    /// Scratch buffers are pre-allocated `AlignedVec<f32>` on the heap,
    /// replacing the const-generic stack buffer `[f32; 1024]` to support CH > 16.
    ///
    /// Uses full-precision f32 weights (no quantization) and exact `f32::tanh`.
    ///
    /// # Safety
    /// Math dispatch via pointer to inlined intrinsic functions.
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath>(&mut self, ctx: WavenetProcessContext<'_>) {
        let ch = self.conv1d.in_ch;
        let scratch_len = ctx.num_frames * ch;
        debug_assert!(
            scratch_len <= self.scratch_mixin.len(),
            "process_block_internal: num_frames*ch ({}) exceeds scratch buffer ({})",
            scratch_len,
            self.scratch_mixin.len()
        );

        // SAFETY: the `debug_assert!` above bounds `scratch_len = num_frames * ch` within the
        // scratch buffers (both pre-allocated to `ch * WAVENET_MAX_NUM_FRAMES`), the
        // `layer_buffer` residual slice `lb_offset..lb_offset + scratch_len` is covered by the
        // caller contract on `buffer_start`/`num_frames`, and the inner `unsafe fn` calls
        // (`process_block`, activation and residual kernels) are reached under those same
        // validated preconditions with `M` dispatched by runtime CPUID checks.
        unsafe {
            let mixin_out = &mut self.scratch_mixin[..scratch_len];
            self.input_mixin
                .process_block::<M>(ctx.condition, mixin_out, ctx.num_frames);

            let conv_slice = &mut self.scratch_conv[..scratch_len];
            self.conv1d.process_block::<M>(
                ctx.layer_buffer,
                conv_slice,
                ctx.buffer_start,
                ctx.num_frames,
                Some(mixin_out),
            );

            if let Some(s) = ctx.seed {
                match &self.activation {
                    ActivationType::ReLU => {
                        M::relu_and_accumulate_with_seed(ctx.head_input, conv_slice, s)
                    }
                    _ => M::tanh_and_accumulate_with_seed(ctx.head_input, conv_slice, s),
                }
            } else if ctx.is_first_layer {
                match &self.activation {
                    ActivationType::ReLU => M::relu_and_overwrite_block(ctx.head_input, conv_slice),
                    _ => M::tanh_and_overwrite_block(ctx.head_input, conv_slice),
                }
            } else {
                match &self.activation {
                    ActivationType::ReLU => {
                        M::relu_and_accumulate_block(ctx.head_input, conv_slice)
                    }
                    _ => M::tanh_and_accumulate_block(ctx.head_input, conv_slice),
                }
            }

            let lb_offset = ctx.buffer_start * ch;
            let residual_slice = ctx
                .layer_buffer
                .get_unchecked(lb_offset..lb_offset + scratch_len);

            self.one_by_one.process_residual_batch::<M>(
                conv_slice,
                residual_slice,
                ctx.output,
                ctx.num_frames,
            );
        }
    }
}
