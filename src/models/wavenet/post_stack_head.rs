// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::AlignedVec;
use crate::math::common::SimdMath;
use crate::models::a2::activations::ActivationType;

use super::common::WaveNetLayerState;
use super::conv1d_dyn::Conv1dDyn;
use crate::loader::nam_json::model::HeadConfig;

/// Post-stack head sub-object for WaveNet / ConvNet architectures.
///
/// Contains a causal Conv1D + activation that processes the signal
/// after the stack of layer arrays, before the final `head_scale` gain.
/// Mirrors the `_Head` structure in NAMCore's `convnet.h:108-118`.
#[derive(Clone)]
#[repr(align(64))]
pub struct PostStackHead {
    /// Causal 1D convolution (dynamic runtime dimensions).
    pub conv: Conv1dDyn,
    /// Activation function applied after convolution.
    pub activation: ActivationType,
    /// Ring buffer state for causal convolution lookback.
    pub state: WaveNetLayerState,
}

impl PostStackHead {
    /// Creates a new `PostStackHead` from the parsed `HeadConfig` and the
    /// input channel count from the last layer array.
    ///
    /// Missing fields in `HeadConfig` fall back to sensible defaults:
    /// - `channels` → `in_channels` (same as the last array's head projection)
    /// - `out_channels` → 1 (mono output)
    /// - `kernel_size` → 3
    /// - `bias` → false
    /// - `activation` → "Tanh"
    ///
    /// Weight and bias arrays are zero-initialized and must be populated
    /// by the dispatcher via `set_weights` and `set_bias`.
    pub fn from_config(config: &HeadConfig, in_channels: usize) -> std::io::Result<Self> {
        let channels = config.channels.unwrap_or(in_channels);
        let out_channels = config.out_channels.unwrap_or(1);
        let kernel = config.kernel_size.unwrap_or(3);
        let do_bias = config.bias.unwrap_or(false);
        let activation = parse_activation(config.activation.as_deref().unwrap_or("Tanh"));

        let num_blocks = out_channels.div_ceil(4);
        let weights_len = num_blocks * kernel * channels * 4;
        let bias_len = out_channels;

        let weights = AlignedVec::new(weights_len, 0.0f32)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?;
        let bias = AlignedVec::new(bias_len, 0.0f32)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?;

        let receptive_field = kernel;
        let state = WaveNetLayerState::new(channels, receptive_field, 0)?;

        let conv = Conv1dDyn {
            weights,
            bias,
            do_bias,
            dilation: 1,
            in_ch: channels,
            out_ch: out_channels,
            num_blocks,
            interleave_width: 4,
            kernel,
        };

        Ok(Self {
            conv,
            activation,
            state,
        })
    }

    /// Returns the receptive field contribution of this head (kernel size).
    /// Must be added to the global model receptive field for prewarm.
    pub fn receptive_field(&self) -> usize {
        self.conv.kernel
    }

    /// Number of output channels produced by this head.
    pub fn out_channels(&self) -> usize {
        self.conv.out_ch
    }

    /// Number of input channels expected by this head.
    pub fn in_channels(&self) -> usize {
        self.conv.in_ch
    }

    /// Loads convolution weights from a flat f32 slice.
    pub fn set_weights(&mut self, weights: &[f32]) {
        let len = self.conv.weights.len().min(weights.len());
        self.conv.weights[..len].copy_from_slice(&weights[..len]);
    }

    /// Loads convolution bias from a flat f32 slice, if present.
    pub fn set_bias(&mut self, bias: &[f32]) {
        let len = self.conv.bias.len().min(bias.len());
        self.conv.bias[..len].copy_from_slice(&bias[..len]);
    }

    /// Public dispatch wrapper that selects the optimal SIMD path.
    ///
    /// # Safety
    /// Input and output slices must have sizes compatible with the head dimensions:
    /// `input.len() == num_frames * in_ch`, `output.len() == num_frames * out_ch`.
    /// The ring buffer state must have been properly initialized (via `prewarm` or
    /// sufficient prior processing) to cover the causal receptive field.
    #[inline(always)]
    pub unsafe fn process_block(&mut self, input: &[f32], output: &mut [f32], num_frames: usize) {
        // SAFETY: `dispatch_simd!` dispatches on runtime CPUID feature checks to a matching
        // `#[target_feature]` backend `M`; `input`/`output`/`num_frames` are the same values
        // validated by this safe wrapper's documented contract.
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
    /// Writes `num_frames` of input into the ring buffer, runs the causal
    /// Conv1D directly into the output slice, applies activation in-place,
    /// and leaves the results in output.
    ///
    /// Input layout: frame-interleaved `[f0_c0, f0_c1, ..., f1_c0, ...]`.
    /// Output layout: frame-interleaved `[f0_c0, f0_c1, ..., f1_c0, ...]`.
    ///
    /// # Safety
    /// `input` and `output` must have sizes `num_frames * in_ch` and
    /// `num_frames * out_ch` respectively. The ring buffer must have been
    /// properly initialized.
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
        // SAFETY: `input` has `input_len = num_frames * in_ch` f32s, and the state's ring buffer
        // is sized for `buffer_start + num_frames` frames (constructor + `advance_frames`
        // contract), so the destination at `buf_start` fits; the buffers are distinct and
        // non-null/`f32`-aligned.
        unsafe {
            core::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.state.layer_buffer.as_mut_ptr().add(buf_start),
                input_len,
            );
        }

        let out_slice = &mut output[..num_frames * out_ch];
        // SAFETY: `process_block` is an `unsafe fn` reached with `out_slice` sized
        // `num_frames * out_ch`, the layer buffer covering `buffer_start + num_frames` frames
        // (the copy above), and `M`'s backend features guaranteed by the `dispatch_simd!`
        // caller.
        unsafe {
            self.conv.process_block::<M>(
                &self.state.layer_buffer,
                out_slice,
                self.state.buffer_start,
                num_frames,
                None,
            );
        }

        // SAFETY: `apply_simd` processes `out_slice` (`num_frames * out_ch` f32s) with `M`'s
        // instructions, whose availability is guaranteed by the dispatch that selected `M`.
        unsafe {
            self.activation.apply_simd::<M>(out_slice);
        }

        self.state.advance_frames(num_frames, in_ch);
    }

    /// Public prewarm wrapper with SIMD dispatch.
    #[cold]
    pub fn prewarm(&mut self) {
        // SAFETY: `dispatch_simd!` dispatches on runtime CPUID feature checks to a matching
        // `#[target_feature]` backend `M`; the prewarm only touches the head's own state buffer.
        unsafe {
            crate::math::common::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// Fills the conv state buffer with a single frame of silence replicated
    /// backward to cover the entire receptive field.
    ///
    /// # Safety
    /// Must be called via `dispatch_simd!` macro. The state buffer must be
    /// properly allocated and the ring buffer start pointer must be valid.
    pub unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        let in_ch = self.conv.in_ch;
        let out_ch = self.conv.out_ch;
        let kernel = self.conv.kernel;

        let buf_start = self.state.buffer_start * in_ch;

        self.state.layer_buffer[buf_start..buf_start + in_ch].fill(0.0);

        let start_idx = self.state.buffer_start * in_ch;
        let src_range = start_idx..start_idx + in_ch;
        for offset in 1..=kernel {
            let dst_idx = (self.state.buffer_start - offset) * in_ch;
            self.state
                .layer_buffer
                .copy_within(src_range.clone(), dst_idx);
        }

        let mut out_frame = vec![0.0f32; out_ch];
        // SAFETY: `process_single_frame` is an `unsafe fn` reached with `out_frame` sized
        // `out_ch`, the state buffer covering `buffer_start`'s receptive field (backfilled above),
        // and `M` guaranteed by the `dispatch_simd!` caller.
        unsafe {
            self.conv.process_single_frame::<M>(
                &self.state.layer_buffer,
                &mut out_frame,
                self.state.buffer_start,
                None,
            );
        }
        // SAFETY: `out_frame` has `out_ch` f32s and `M`'s instructions are available per the
        // dispatch that selected it.
        unsafe {
            self.activation.apply_simd::<M>(&mut out_frame);
        }

        self.state.advance_frames(1, in_ch);
    }
}

/// Maps an activation function name string to an `ActivationType`.
///
/// Supported values match the variant names of `ActivationType`:
/// `"Tanh"`, `"HardTanh"`, `"FastTanh"`, `"ReLU"`, `"Sigmoid"`,
/// `"SiLU"`, `"HardSwish"`, `"Softsign"`.
///
/// Unrecognized strings fall back to `ActivationType::Tanh`.
pub fn parse_activation(name: &str) -> ActivationType {
    match name {
        "Tanh" => ActivationType::Tanh,
        "HardTanh" => ActivationType::HardTanh,
        "FastTanh" => ActivationType::FastTanh,
        "ReLU" => ActivationType::ReLU,
        "Sigmoid" => ActivationType::Sigmoid,
        "SiLU" => ActivationType::SiLU,
        "HardSwish" => ActivationType::HardSwish,
        "Softsign" => ActivationType::Softsign,
        _ => ActivationType::Tanh,
    }
}

#[cfg(test)]
#[path = "post_stack_head_test.rs"]
mod tests;
