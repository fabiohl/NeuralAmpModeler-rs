// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;
use crate::math::common::SimdMath;
use crate::models::a2::activations::ActivationType;

use super::common::WaveNetLayerState;
use super::conv1d_dyn::Conv1dDyn;
use crate::loader::nam_json::model::HeadConfig;

/// One stage of a post-stack head: a causal Conv1D plus the activation that
/// feeds it.
///
/// - Canonical NAMCore heads (`head.kernel_sizes` array) apply the activation
///   to the stage **input**, before the convolution, with an implicit bias.
/// - The legacy Rust-only single-stage head applies the stored activation to
///   the convolution output instead; that ordering is handled by
///   [`PostStackHead::process_block`], not by this struct.
#[derive(Clone)]
#[repr(align(64))]
pub struct HeadStage {
    /// Causal 1D convolution (dynamic runtime dimensions).
    pub conv: Conv1dDyn,
    /// Activation associated with this stage.
    pub activation: ActivationType,
    /// Ring buffer state for the causal convolution lookback.
    pub state: WaveNetLayerState,
}

/// Post-stack head sub-object for WaveNet / ConvNet architectures.
///
/// Two contracts coexist, selected by the parsed [`HeadConfig`]:
///
/// - **Canonical** (`head.kernel_sizes` present, NAMCore trainer export):
///   arbitrary number of stages, intermediate width `head.channels`, final
///   width `head.out_channels`, implicit bias on every convolution, activation
///   applied before each convolution and `head_scale` applied to the layer
///   array outputs **before** the head. Matches
///   `NAM/wavenet/model.cpp:21-44,664-673,855-866`.
/// - **Legacy** (only `head.kernel_size`, Rust-only format): one stage,
///   optional bias, convolution first and activation after, `head_scale`
///   applied after the head. This preserves the historical NeuralAmpModeler-rs
///   behavior for models that are not NAMCore exports.
#[derive(Clone)]
#[repr(align(64))]
pub struct PostStackHead {
    /// Head stages, in execution order (always at least one).
    pub stages: Vec<HeadStage>,
    /// Whether this head follows the canonical NAMCore contract.
    pub canonical: bool,
    /// Reusable inter-stage scratch, sized `max_stage_out * WAVENET_MAX_NUM_FRAMES`.
    ///
    /// Only allocated for canonical heads; legacy heads process in place and
    /// keep it empty.
    scratch: AlignedVec<f32>,
    /// Largest output width across all stages.
    max_stage_out: usize,
}

impl PostStackHead {
    /// Returns whether this head still satisfies the unchecked kernel
    /// contract for `expected_in` input channels.
    #[inline]
    pub(crate) fn is_sound(&self, expected_in: usize) -> bool {
        if self.stages.is_empty() {
            return false;
        }
        let mut in_ch = expected_in;
        for stage in &self.stages {
            if stage.conv.in_ch != in_ch
                || !stage.conv.is_sound()
                || !stage
                    .conv
                    .backreach()
                    .is_some_and(|back| back <= stage.state.receptive_field_size)
                || !stage.state.is_sound_for(stage.conv.in_ch)
            {
                return false;
            }
            in_ch = stage.conv.out_ch;
        }
        true
    }

    /// Creates a new [`PostStackHead`] from the parsed [`HeadConfig`] and the
    /// input channel count from the last layer array.
    ///
    /// Canonical heads (`kernel_sizes` present) follow NAMCore exactly:
    /// `channels` is the intermediate width for every non-final stage,
    /// `out_channels` the final width, every Conv1D carries a bias, and one
    /// activation instance is created per stage from the same specification.
    ///
    /// Legacy heads keep the historical defaults: `channels` → `in_channels`,
    /// `out_channels` → 1, `kernel_size` → 3, `bias` → false and
    /// `activation` → `"Tanh"`.
    ///
    /// # Errors
    /// Rejects non-positive or oversized dimensions, an empty
    /// `kernel_sizes` array, an ambiguous config declaring both
    /// `kernel_size` and `kernel_sizes`, unresolved activations, and PReLU
    /// slope vectors whose length matches neither `1` nor the stage input
    /// channel count.
    pub fn from_config(config: &HeadConfig, in_channels: usize) -> Result<Self, NamErrorCode> {
        use crate::loader::nam_json::validation::{
            MAX_A2_HEAD_CHANNELS, MAX_HEAD_KERNEL_SIZE, MAX_WAVENET_FREE_CHANNELS,
        };

        if config.kernel_size.is_some() && config.kernel_sizes.is_some() {
            return Err(NamErrorCode::InvalidModelTopology);
        }

        let channels = config.channels.unwrap_or(in_channels);
        let out_channels = config.out_channels.unwrap_or(1);
        let kernels = config.resolved_kernel_sizes();
        let do_bias = if config.is_canonical() {
            // NAMCore's detail::Head always constructs its Conv1D with bias.
            true
        } else {
            config.bias.unwrap_or(false)
        };

        let default_activation = serde_json::Value::String("Tanh".to_string());
        let activation_spec = config.activation.as_ref().unwrap_or(&default_activation);

        if in_channels == 0 || channels == 0 || out_channels == 0 || kernels.is_empty() {
            return Err(NamErrorCode::InvalidModelTopology);
        }
        if channels > MAX_WAVENET_FREE_CHANNELS || in_channels > MAX_WAVENET_FREE_CHANNELS {
            return Err(NamErrorCode::InvalidModelTopology);
        }
        if out_channels > MAX_A2_HEAD_CHANNELS {
            return Err(NamErrorCode::InvalidModelTopology);
        }

        let mut stages = Vec::with_capacity(kernels.len());
        // Canonical heads take the last layer array's head projection as the
        // first stage input. The legacy Rust-only head historically used
        // `channels` as the convolution input width instead.
        let mut stage_in = if config.is_canonical() {
            in_channels
        } else {
            channels
        };
        let mut max_stage_out = out_channels;

        for (i, &kernel) in kernels.iter().enumerate() {
            if kernel == 0 || kernel > MAX_HEAD_KERNEL_SIZE {
                return Err(NamErrorCode::InvalidModelTopology);
            }
            let stage_out = if i + 1 == kernels.len() {
                out_channels
            } else {
                channels
            };
            max_stage_out = max_stage_out.max(stage_out);

            let activation = crate::loader::nam_json::try_parse_activation(activation_spec)?;
            if let ActivationType::PReLU {
                ref negative_slopes,
            } = activation
                && (negative_slopes.is_empty()
                    || (negative_slopes.len() != 1 && negative_slopes.len() != stage_in))
            {
                // The activation operates on this stage's input width.
                return Err(NamErrorCode::InvalidModelTopology);
            }

            let num_blocks = stage_out.div_ceil(4);
            // Defense-in-depth (F-RES2-02/F-RES2-03): checked size arithmetic
            // so a hostile dimension can never wrap into a smaller allocation
            // in release builds (overflow-checks off).
            let weights_len = num_blocks
                .checked_mul(4)
                .and_then(|v| v.checked_mul(stage_in))
                .and_then(|v| v.checked_mul(kernel))
                .ok_or(NamErrorCode::InvalidModelTopology)?;
            let bias_len = stage_out;

            let weights = AlignedVec::new(weights_len, 0.0f32)?;
            let bias = AlignedVec::new(bias_len, 0.0f32)?;

            let state = WaveNetLayerState::new(stage_in, kernel, i)
                .map_err(|_| NamErrorCode::InvalidModelTopology)?;

            let conv = Conv1dDyn::try_from_parts(
                weights, bias, do_bias, 1, stage_in, stage_out, kernel, 4,
            )
            .map_err(|_| NamErrorCode::InvalidModelTopology)?;

            stages.push(HeadStage {
                conv,
                activation,
                state,
            });
            stage_in = stage_out;
        }

        let scratch = if config.is_canonical() {
            let len = max_stage_out
                .checked_mul(super::common::WAVENET_MAX_NUM_FRAMES)
                .ok_or(NamErrorCode::InvalidModelTopology)?;
            AlignedVec::new(len, 0.0f32)?
        } else {
            AlignedVec::new(0, 0.0f32)?
        };

        Ok(Self {
            stages,
            canonical: config.is_canonical(),
            scratch,
            max_stage_out,
        })
    }

    /// Returns the receptive field contribution of this head:
    /// `1 + Σ(kernel_i - 1)`, matching NAMCore's `detail::Head`.
    ///
    /// The caller adds `receptive_field() - 1` to the model prewarm count.
    pub fn receptive_field(&self) -> usize {
        1 + self
            .stages
            .iter()
            .map(|stage| stage.conv.kernel.saturating_sub(1))
            .sum::<usize>()
    }

    /// Number of output channels produced by this head.
    pub fn out_channels(&self) -> usize {
        self.stages.last().map_or(1, |stage| stage.conv.out_ch)
    }

    /// Number of input channels expected by this head.
    pub fn in_channels(&self) -> usize {
        self.stages.first().map_or(1, |stage| stage.conv.in_ch)
    }

    /// Returns whether this head uses the canonical NAMCore contract
    /// (activation before each convolution, `head_scale` applied before the
    /// head).
    #[inline]
    pub fn is_canonical(&self) -> bool {
        self.canonical
    }

    /// Clears every stage's causal delay-line state in place, without
    /// allocating.
    pub fn clear_state(&mut self) {
        for stage in &mut self.stages {
            stage.state.clear();
        }
    }

    /// Loads the first stage convolution weights from a flat f32 slice
    /// (legacy single-stage path).
    pub fn set_weights(&mut self, weights: &[f32]) {
        if let Some(stage) = self.stages.first_mut() {
            let len = stage.conv.weights.len().min(weights.len());
            stage.conv.weights[..len].copy_from_slice(&weights[..len]);
        }
    }

    /// Loads the first stage convolution bias from a flat f32 slice (legacy
    /// single-stage path).
    pub fn set_bias(&mut self, bias: &[f32]) {
        if let Some(stage) = self.stages.first_mut() {
            let len = stage.conv.bias.len().min(bias.len());
            stage.conv.bias[..len].copy_from_slice(&bias[..len]);
        }
    }

    /// Loads one stage's interleaved-4 convolution weights.
    pub fn set_stage_weights(&mut self, stage: usize, weights: &[f32]) {
        if let Some(stage) = self.stages.get_mut(stage) {
            let len = stage.conv.weights.len().min(weights.len());
            stage.conv.weights[..len].copy_from_slice(&weights[..len]);
        }
    }

    /// Loads one stage's convolution bias.
    pub fn set_stage_bias(&mut self, stage: usize, bias: &[f32]) {
        if let Some(stage) = self.stages.get_mut(stage) {
            let len = stage.conv.bias.len().min(bias.len());
            stage.conv.bias[..len].copy_from_slice(&bias[..len]);
        }
    }

    /// Applies the legacy single-stage head: causal conv, then activation.
    ///
    /// # Safety
    /// Input and output slices must have sizes compatible with the head
    /// dimensions: `input.len() == num_frames * in_ch`,
    /// `output.len() == num_frames * out_ch`. The ring buffer state must have
    /// been properly initialized (via `prewarm` or sufficient prior
    /// processing) to cover the causal receptive field.
    #[inline(always)]
    pub unsafe fn process_block(&mut self, input: &[f32], output: &mut [f32], num_frames: usize) {
        debug_assert!(!self.canonical);
        let Some(stage) = self.stages.first_mut() else {
            return;
        };
        let in_ch = stage.conv.in_ch;
        let out_ch = stage.conv.out_ch;
        let input_len = num_frames * in_ch;

        let buf_start = stage.state.buffer_start * in_ch;
        // SAFETY: `input` has `input_len = num_frames * in_ch` f32s, and the
        // state's ring buffer is sized for `buffer_start + num_frames` frames
        // (constructor + `advance_frames` contract), so the destination at
        // `buf_start` fits; the buffers are distinct.
        unsafe {
            core::ptr::copy_nonoverlapping(
                input.as_ptr(),
                stage.state.layer_buffer.as_mut_ptr().add(buf_start),
                input_len,
            );
        }

        let out_slice = &mut output[..num_frames * out_ch];
        // SAFETY: `out_slice` has `num_frames * out_ch` f32s and the layer
        // buffer covers the causal window for `buffer_start`.
        unsafe {
            crate::math::common::dispatch_simd!(
                stage.conv,
                process_block,
                &stage.state.layer_buffer,
                out_slice,
                stage.state.buffer_start,
                num_frames,
                None
            )
        };

        // SAFETY: `out_slice` has `num_frames * out_ch` f32s and the backend
        // selected by the dispatch matches the CPU ISA.
        unsafe { crate::math::common::dispatch_simd!(stage.activation, apply_simd, out_slice) };

        stage.state.advance_frames(num_frames, in_ch);
    }

    /// Applies the canonical NAMCore multi-stage head:
    /// `input * scale` → activation → conv, repeated per stage.
    ///
    /// # Safety
    /// `input.len() == num_frames * in_channels()` and
    /// `output.len() == num_frames * out_channels()`. The ring buffers must be
    /// initialized (via `prewarm` or sufficient prior processing).
    #[inline(always)]
    pub unsafe fn process_block_scaled<M: SimdMath>(
        &mut self,
        input: &[f32],
        scale: f32,
        output: &mut [f32],
        num_frames: usize,
    ) {
        debug_assert!(self.canonical);
        if self.stages.is_empty() || num_frames == 0 {
            return;
        }

        let Self {
            stages,
            scratch: scratch_buf,
            max_stage_out,
            ..
        } = self;
        let scratch = &mut scratch_buf[..num_frames * *max_stage_out];

        // Stage 0 consumes the externally provided (scaled) input.
        {
            let stage = &mut stages[0];
            let in_ch = stage.conv.in_ch;
            let input_len = num_frames * in_ch;
            let buf_start = stage.state.buffer_start * in_ch;
            // SAFETY: `input` has `num_frames * in_ch` f32s, the ring buffer
            // covers `buf_start + num_frames` frames, and the slices are
            // distinct.
            unsafe {
                M::copy_with_scale(
                    input,
                    &mut stage.state.layer_buffer[buf_start..buf_start + input_len],
                    scale,
                );
            }
            // SAFETY: the ring slice has `input_len` f32s and the activation
            // backend matches the CPU ISA.
            unsafe {
                stage.activation.apply_simd::<M>(
                    &mut stage.state.layer_buffer[buf_start..buf_start + input_len],
                );
            }
            let out_len = num_frames * stage.conv.out_ch;
            // SAFETY: destination `scratch[..out_len]` and the stage ring
            // buffer cover the documented sizes.
            unsafe {
                stage.conv.process_block::<M>(
                    &stage.state.layer_buffer,
                    &mut scratch[..out_len],
                    stage.state.buffer_start,
                    num_frames,
                    None,
                );
            }
            stage.state.advance_frames(num_frames, in_ch);
        }

        // Remaining stages consume the previous stage's output. NAMCore
        // activates the previous output before feeding it to the next conv.
        let mut last_out_len = num_frames * stages[0].conv.out_ch;
        for stage in &mut stages[1..] {
            let in_len = num_frames * stage.conv.in_ch;
            debug_assert!(in_len <= last_out_len);
            // SAFETY: `scratch[..in_len]` is the previous stage output; the
            // activation backend matches the CPU ISA.
            unsafe {
                stage.activation.apply_simd::<M>(&mut scratch[..in_len]);
            }
            let in_ch = stage.conv.in_ch;
            let buf_start = stage.state.buffer_start * in_ch;
            stage.state.layer_buffer[buf_start..buf_start + in_len]
                .copy_from_slice(&scratch[..in_len]);
            let out_len = num_frames * stage.conv.out_ch;
            // SAFETY: destination `scratch[..out_len]` and the stage ring
            // buffer cover the documented sizes. The convolution copies its
            // input into its ring buffer before writing the output, so reusing
            // `scratch` for both is safe.
            unsafe {
                stage.conv.process_block::<M>(
                    &stage.state.layer_buffer,
                    &mut scratch[..out_len],
                    stage.state.buffer_start,
                    num_frames,
                    None,
                );
            }
            stage.state.advance_frames(num_frames, in_ch);
            last_out_len = out_len;
        }

        output[..last_out_len].copy_from_slice(&scratch[..last_out_len]);
    }

    /// Public prewarm wrapper with SIMD dispatch.
    #[cold]
    pub fn prewarm(&mut self) {
        // SAFETY: `dispatch_simd!` dispatches on runtime CPUID feature checks
        // to a matching `#[target_feature]` backend `M`; the prewarm only
        // touches the head's own state buffers.
        unsafe {
            if self.canonical {
                crate::math::common::dispatch_simd!(self, prewarm_canonical_internal);
            } else {
                crate::math::common::dispatch_simd!(self, prewarm_internal);
            }
        }
    }

    /// Fills the legacy single-stage conv state buffer with a single frame of
    /// silence replicated backward to cover the receptive field, then applies
    /// the activation after the convolution (legacy order).
    ///
    /// # Safety
    /// Must be called via `dispatch_simd!` macro. The state buffer must be
    /// properly allocated and the ring buffer start pointer must be valid.
    pub unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        let Some(stage) = self.stages.first_mut() else {
            return;
        };
        let in_ch = stage.conv.in_ch;
        let out_ch = stage.conv.out_ch;
        let kernel = stage.conv.kernel;

        let buf_start = stage.state.buffer_start * in_ch;

        stage.state.layer_buffer[buf_start..buf_start + in_ch].fill(0.0);

        let start_idx = stage.state.buffer_start * in_ch;
        let src_range = start_idx..start_idx + in_ch;
        for offset in 1..=kernel {
            let dst_idx = (stage.state.buffer_start - offset) * in_ch;
            stage
                .state
                .layer_buffer
                .copy_within(src_range.clone(), dst_idx);
        }

        let mut out_frame = vec![0.0f32; out_ch];
        // SAFETY: `out_frame` has `out_ch` f32s, the state buffer covers the
        // receptive field, and `M` is guaranteed by the `dispatch_simd!`
        // caller.
        unsafe {
            stage.conv.process_single_frame::<M>(
                &stage.state.layer_buffer,
                &mut out_frame,
                stage.state.buffer_start,
                None,
            );
        }
        // SAFETY: `out_frame` has `out_ch` f32s and `M`'s instructions are
        // available per the dispatch.
        unsafe {
            stage.activation.apply_simd::<M>(&mut out_frame);
        }

        stage.state.advance_frames(1, in_ch);
    }

    /// Steady-state prewarm for the canonical (activation-before-conv) head.
    ///
    /// For each stage it fills the ring buffer with the constant stage input
    /// produced by silence (`activation(0)` for the first stage, then the
    /// activation of the previous stage's steady output), computes the steady
    /// convolution output and advances one frame, mirroring the legacy
    /// prewarm's single-frame contract without allocating on the audio thread.
    ///
    /// # Safety
    /// Must be called via `dispatch_simd!`; the ring buffers must be valid.
    pub unsafe fn prewarm_canonical_internal<M: SimdMath>(&mut self) {
        if self.stages.is_empty() {
            return;
        }

        let mut input = vec![0.0f32; self.stages[0].conv.in_ch];
        for stage in &mut self.stages {
            // Activation is applied before this stage's convolution.
            // SAFETY: `input` has exactly this stage's input channel count.
            unsafe {
                stage.activation.apply_simd::<M>(&mut input);
            }

            let in_ch = stage.conv.in_ch;
            let kernel = stage.conv.kernel;
            let buf_start = stage.state.buffer_start * in_ch;
            for offset in 0..=kernel {
                let start = buf_start - offset * in_ch;
                stage.state.layer_buffer[start..start + in_ch].copy_from_slice(&input);
            }

            let mut out_frame = vec![0.0f32; stage.conv.out_ch];
            // SAFETY: `out_frame` has `out_ch` f32s, the state buffer covers
            // the receptive field, and `M` is guaranteed by the dispatch.
            unsafe {
                stage.conv.process_single_frame::<M>(
                    &stage.state.layer_buffer,
                    &mut out_frame,
                    stage.state.buffer_start,
                    None,
                );
            }
            stage.state.advance_frames(1, in_ch);
            input = out_frame;
        }
    }
}

#[cfg(test)]
#[path = "post_stack_head_test.rs"]
mod tests;
