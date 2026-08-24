// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! ConvNet feed-forward block: Conv1D → BatchNorm → Activation.
//!
//! Each block is simpler than a WaveNet layer — there is no gating mechanism
//! (no input_mixin / 1x1) and no rechannel projection. Output feeds directly
//! into the next block or the post-stack head.

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::{AlignedVec, SimdMath};
use crate::models::a2::activations::ActivationType;
use crate::models::wavenet::Conv1dDyn;
use crate::models::wavenet::common::WaveNetLayerState;

use super::batch_norm::BatchNorm1D;

/// A single ConvNet feed-forward block.
///
/// Processing pipeline:
/// 1. Write input frames into the causal ring buffer
/// 2. Run causal Conv1D directly into the output slice
/// 3. Apply BatchNorm (affine transform) in-place on output
/// 4. Apply activation function in-place on output
#[derive(Clone)]
#[repr(align(64))]
pub struct ConvNetBlock {
    /// Causal 1D convolution (runtime dimensions).
    pub conv: Conv1dDyn,
    /// Batch normalization (pre-fused affine transform).
    pub bn: BatchNorm1D,
    /// Activation function applied after batch norm.
    pub activation: ActivationType,
    /// Ring buffer state for causal convolution lookback.
    pub state: WaveNetLayerState,
}

impl ConvNetBlock {
    /// Creates a new `ConvNetBlock` with zero-initialized weights.
    ///
    /// Weights must be populated via `set_conv_weights` / `set_conv_bias`
    /// and `set_bn_params` by the dispatcher.
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        do_bias: bool,
        activation: ActivationType,
        alloc_num: usize,
    ) -> std::io::Result<Self> {
        let num_blocks = out_ch.div_ceil(4);
        let weights_len = num_blocks * kernel * in_ch * 4;

        let conv = Conv1dDyn {
            weights: AlignedVec::new(weights_len, 0.0f32).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}"))
            })?,
            bias: AlignedVec::new(out_ch, 0.0f32).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}"))
            })?,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            num_blocks,
            interleave_width: 4,
            kernel,
        };

        let bn = BatchNorm1D::from_fused(out_ch, &vec![0.0f32; out_ch], &vec![0.0f32; out_ch])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?;

        let receptive_field = (kernel - 1) * dilation;
        let state = WaveNetLayerState::new(in_ch, receptive_field, alloc_num)?;

        Ok(Self {
            conv,
            bn,
            activation,
            state,
        })
    }

    /// Returns the receptive field contribution of this block.
    pub fn receptive_field(&self) -> usize {
        (self.conv.kernel - 1) * self.conv.dilation
    }

    /// Loads convolution weights from a flat f32 slice.
    pub fn set_conv_weights(&mut self, weights: &[f32]) {
        let len = self.conv.weights.len().min(weights.len());
        self.conv.weights[..len].copy_from_slice(&weights[..len]);
    }

    /// Loads convolution bias from a flat f32 slice.
    pub fn set_conv_bias(&mut self, bias: &[f32]) {
        let len = self.conv.bias.len().min(bias.len());
        self.conv.bias[..len].copy_from_slice(&bias[..len]);
    }

    /// Loads pre-fused batch norm parameters.
    pub fn set_bn_params(&mut self, scale: &[f32], offset: &[f32]) -> Result<(), NamErrorCode> {
        self.bn = BatchNorm1D::from_fused(self.conv.out_ch, scale, offset)?;
        Ok(())
    }

    /// Public dispatch wrapper that selects the optimal SIMD path.
    ///
    /// # Safety
    /// Input and output slices must have sizes compatible with the block dimensions:
    /// `input.len() == num_frames * in_ch`, `output.len() == num_frames * out_ch`.
    #[inline(always)]
    pub unsafe fn process_block(&mut self, input: &[f32], output: &mut [f32], num_frames: usize) {
        // SAFETY: this `unsafe fn` is only reachable under its documented
        // preconditions: `input.len() == num_frames * in_ch` and
        // `output.len() == num_frames * out_ch`; `dispatch_simd!` selects a
        // matching `#[target_feature]` backend on runtime CPUID checks.
        unsafe {
            crate::math::common::dispatch_simd!(
                self,
                process_block_internal,
                input,
                output,
                num_frames
            )
        };
    }

    /// SIMD-dispatched processing kernel.
    ///
    /// # Safety
    /// `input.len()` must equal `num_frames * in_ch`.
    /// `output.len()` must equal `num_frames * out_ch`.
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath>(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        let in_ch = self.conv.in_ch;
        let out_ch = self.conv.out_ch;
        let input_len = num_frames * in_ch;

        let buf_start = self.state.buffer_start * in_ch;
        self.state.layer_buffer[buf_start..buf_start + input_len]
            .copy_from_slice(&input[..input_len]);

        let out_slice = &mut output[..num_frames * out_ch];
        // SAFETY: `conv.process_block::<M>` is an `unsafe fn`; its preconditions
        // hold: `layer_buffer` is large enough for `buffer_start..buffer_start + num_frames`
        // plus kernel lookback, `out_slice` has `num_frames * out_ch` elements, and `M`
        // matches the CPU ISA (top-level `dispatch_simd!`).
        unsafe {
            self.conv.process_block::<M>(
                &self.state.layer_buffer,
                out_slice,
                self.state.buffer_start,
                num_frames,
                None,
            );
        }

        // SAFETY: `out_slice.len() == num_frames * out_ch` and the batch norm was
        // built with `self.conv.out_ch` channels (`set_bn_params`/constructor),
        // satisfying `process_simd`'s `data.len() == num_frames * num_channels`
        // contract; `M` matches the CPU ISA.
        unsafe {
            self.bn.process_simd::<M>(out_slice, num_frames);
        }

        // SAFETY: `out_slice` is a valid mutable slice of `num_frames * out_ch`
        // elements and `M` matches the CPU ISA; both are `apply_simd`'s
        // documented preconditions.
        unsafe {
            self.activation.apply_simd::<M>(out_slice);
        }

        self.state.advance_frames(num_frames, in_ch);
    }
}

#[cfg(test)]
#[path = "convnet_block_test.rs"]
mod tests;
