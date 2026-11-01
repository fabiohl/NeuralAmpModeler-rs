// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::common::diagnostics::NamErrorCode;
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
    /// Returns whether this head still satisfies the unchecked kernel
    /// contract for `expected_in` input channels.
    #[inline]
    pub(crate) fn is_sound(&self, expected_in: usize) -> bool {
        self.conv.in_ch == expected_in
            && self.conv.is_sound()
            && self
                .conv
                .backreach()
                .is_some_and(|back| back <= self.state.receptive_field_size)
            && self.state.is_sound_for(self.conv.in_ch)
    }

    /// Creates a new `PostStackHead` from the parsed `HeadConfig` and the
    /// input channel count from the last layer array.
    ///
    /// Missing fields in `HeadConfig` fall back to sensible defaults:
    /// - `channels` → `in_channels` (same as the last array's head projection)
    /// - `out_channels` → 1 (mono output)
    /// - `kernel_size` → 3
    /// - `bias` → false
    /// - `activation` → `"Tanh"` (a present but unrecognized activation is
    ///   rejected with [`NamErrorCode::NamJsonUnknownActivation`], never
    ///   silently defaulted)
    ///
    /// Weight and bias arrays are zero-initialized and must be populated
    /// by the dispatcher via `set_weights` and `set_bias`.
    ///
    /// # Errors
    /// Rejects non-positive or oversized dimensions and unresolved activation
    /// specifications with a typed [`NamErrorCode`] before any allocation:
    /// zero channels/out-channels/kernel, `channels` above
    /// `MAX_WAVENET_FREE_CHANNELS`, `out_channels` above `MAX_A2_HEAD_CHANNELS`
    /// and `kernel` above `MAX_KERNEL` (the `Conv1dDyn` fixed tap array).
    /// These are the canonical loader ceilings, re-checked here as
    /// defense-in-depth (F-RES2-02) because `HeadConfig` is also constructible
    /// directly (unit tests, oracle) bypassing the topology validators.
    pub fn from_config(config: &HeadConfig, in_channels: usize) -> Result<Self, NamErrorCode> {
        use crate::loader::nam_json::validation::{
            MAX_A2_HEAD_CHANNELS, MAX_HEAD_KERNEL_SIZE, MAX_WAVENET_FREE_CHANNELS,
        };

        let channels = config.channels.unwrap_or(in_channels);
        let out_channels = config.out_channels.unwrap_or(1);
        let kernel = config.kernel_size.unwrap_or(3);
        let do_bias = config.bias.unwrap_or(false);
        let default_activation = serde_json::Value::String("Tanh".to_string());
        let activation = crate::loader::nam_json::try_parse_activation(
            config.activation.as_ref().unwrap_or(&default_activation),
        )?;

        if channels == 0 || out_channels == 0 || kernel == 0 {
            return Err(NamErrorCode::InvalidModelTopology);
        }
        if channels > MAX_WAVENET_FREE_CHANNELS {
            return Err(NamErrorCode::InvalidModelTopology);
        }
        if out_channels > MAX_A2_HEAD_CHANNELS {
            return Err(NamErrorCode::InvalidModelTopology);
        }
        if kernel > MAX_HEAD_KERNEL_SIZE {
            return Err(NamErrorCode::InvalidModelTopology);
        }

        let num_blocks = out_channels.div_ceil(4);
        // Defense-in-depth (F-RES2-02/F-RES2-03): checked size arithmetic so a
        // hostile dimension can never wrap into a smaller allocation in
        // release builds (overflow-checks off). Ceilings above already bound
        // real inputs; this keeps the computation itself total.
        let weights_len = num_blocks
            .checked_mul(4)
            .and_then(|v| v.checked_mul(channels))
            .and_then(|v| v.checked_mul(kernel))
            .ok_or(NamErrorCode::InvalidModelTopology)?;
        let bias_len = out_channels;

        let weights = AlignedVec::new(weights_len, 0.0f32)?;
        let bias = AlignedVec::new(bias_len, 0.0f32)?;

        let receptive_field = kernel;
        let state = WaveNetLayerState::new(channels, receptive_field, 0)
            .map_err(|_| NamErrorCode::InvalidModelTopology)?;

        // Release-stable padding/bounds proof (R-2): weights_len is exactly the
        // interleaved-4 padded total, validated again by the constructor.
        let conv =
            Conv1dDyn::try_from_parts(weights, bias, do_bias, 1, channels, out_channels, kernel, 4)
                .map_err(|_| NamErrorCode::InvalidModelTopology)?;

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

    /// Clears the head's causal delay-line state to the freshly-built
    /// condition, in place and without allocating.
    pub fn clear_state(&mut self) {
        self.state.clear();
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

#[cfg(test)]
#[path = "post_stack_head_test.rs"]
mod tests;
