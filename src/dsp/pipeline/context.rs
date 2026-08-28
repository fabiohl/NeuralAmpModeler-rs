// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Context and working buffers for the DSP pipeline.

use crate::common::spsc::RtStatusFlags;
use crate::dsp::adaptive::AdaptiveCompute;
use crate::dsp::cabsim::adapter::{CabSimAdapter, CabSimPair};
use crate::dsp::gate::{DynamicHysteresis, GateParams};
use crate::dsp::resampler::NamResampler;
use crate::models::StaticModel;

use super::bridge::DspBridgeWriter;
use crate::dsp::oversample::OversampleEngine;

/// Data context for the DSP pipeline hot-path execution.
///
/// Encapsulates references to all stateful components, controllers, and atomic flags
/// required by the multi-stage DSP pipeline during real-time sample block processing.
pub struct DspPipelineContext<'a> {
    /// Active polyphase resampler for host-to-model sample rate conversion.
    pub resampler: &'a mut NamResampler,
    /// Optional half-band oversampling engine for the left channel (2×/4× rate).
    pub os_l: &'a mut OversampleEngine,
    /// Optional half-band oversampling engine for the right channel (2×/4× rate).
    pub os_r: &'a mut OversampleEngine,
    /// Active neural model instance for the left channel (`None` = bypass).
    pub active_model_l: &'a mut Option<Box<StaticModel>>,
    /// Active neural model instance for the right channel (`None` = bypass).
    pub active_model_r: &'a mut Option<Box<StaticModel>>,
    /// Input gain linear multiplier applied in `apply_input_stage` (pipeline-level).
    /// Set to `1.0` when downstream host integration applies smoothed gain externally;
    /// otherwise carries combined input and model pre-gain scaling factor.
    pub input_gain_mult: f32,
    /// Output gain linear multiplier applied in `apply_output_stage` (pipeline-level).
    /// Set to `1.0` when downstream host integration applies smoothed gain externally;
    /// otherwise carries combined output and model post-gain scaling factor.
    pub output_gain_mult: f32,
    /// Noise Gate threshold and timing parameters.
    pub gate_params: &'a GateParams,
    /// Dynamic hysteresis tracking envelope state for silence detection.
    pub silence_hysteresis: &'a mut DynamicHysteresis,
    /// Dynamic hysteresis tracking envelope state for mono signal detection.
    pub mono_hysteresis: &'a mut DynamicHysteresis,
    /// Opening threshold in squared linear amplitude (`linear_thresh^2`).
    pub threshold_open_sq: f32,
    /// Closing threshold in squared linear amplitude (`linear_thresh^2`).
    pub threshold_close_sq: f32,
    /// Flag indicating mono signal path optimization (single channel model run).
    pub process_mono: &'a mut bool,
    /// Atomic real-time status bitmask for communicating non-blocking flags to main thread.
    pub rt_status: &'a RtStatusFlags,
    /// Adaptive compute controller tracking FSM load state and soft-degrade.
    pub adaptive: &'a mut AdaptiveCompute,
    /// Reference to the audio monitoring bridge (`None` = no active listener).
    pub bridge_writer: Option<DspBridgeWriter>,
    /// Active cab-sim convolution adapter — shared-state path (`None` = not
    /// attached). Both channels run through this single adapter's mutable
    /// state, so this path provides no stereo decoupling; prefer
    /// [`conv_pair`](DspPipelineContext::conv_pair) for stereo content. When
    /// both are attached, `conv_pair` takes precedence and `conv` is ignored.
    pub conv: Option<&'a mut CabSimAdapter>,
    /// Active stereo-decoupled cab-sim pair (`None` = not attached, zero
    /// cost). Independent L/R adapters guarantee no convolucional state is
    /// shared between channels.
    pub conv_pair: Option<&'a mut CabSimPair>,
}

impl<'a> DspPipelineContext<'a> {
    /// Creates the pipeline context from its fifteen mandatory components.
    ///
    /// The optional [`bridge_writer`](DspPipelineContext::bridge_writer),
    /// [`conv`](DspPipelineContext::conv) and
    /// [`conv_pair`](DspPipelineContext::conv_pair) pointers default to
    /// `None` (no listener / cab-sim bypass); attach them with the chainable
    /// [`with_bridge_writer`](Self::with_bridge_writer),
    /// [`with_conv`](Self::with_conv) and
    /// [`with_conv_pair`](Self::with_conv_pair) methods.
    #[expect(
        clippy::too_many_arguments,
        reason = "DspPipelineContext mirrors the public struct literal; all non-optional components are mandatory"
    )]
    pub fn from_parts(
        resampler: &'a mut NamResampler,
        os_l: &'a mut OversampleEngine,
        os_r: &'a mut OversampleEngine,
        active_model_l: &'a mut Option<Box<StaticModel>>,
        active_model_r: &'a mut Option<Box<StaticModel>>,
        input_gain_mult: f32,
        output_gain_mult: f32,
        gate_params: &'a GateParams,
        silence_hysteresis: &'a mut DynamicHysteresis,
        mono_hysteresis: &'a mut DynamicHysteresis,
        threshold_open_sq: f32,
        threshold_close_sq: f32,
        process_mono: &'a mut bool,
        rt_status: &'a RtStatusFlags,
        adaptive: &'a mut AdaptiveCompute,
    ) -> Self {
        Self {
            resampler,
            os_l,
            os_r,
            active_model_l,
            active_model_r,
            input_gain_mult,
            output_gain_mult,
            gate_params,
            silence_hysteresis,
            mono_hysteresis,
            threshold_open_sq,
            threshold_close_sq,
            process_mono,
            rt_status,
            adaptive,
            bridge_writer: None,
            conv: None,
            conv_pair: None,
        }
    }

    /// Attaches the audio monitoring bridge writer (chainable).
    pub fn with_bridge_writer(mut self, bridge_writer: DspBridgeWriter) -> Self {
        self.bridge_writer = Some(bridge_writer);
        self
    }

    /// Attaches the active cab-sim convolution adapter (chainable).
    pub fn with_conv(mut self, conv: &'a mut CabSimAdapter) -> Self {
        self.conv = Some(conv);
        self
    }

    /// Attaches the active stereo-decoupled cab-sim pair (chainable).
    pub fn with_conv_pair(mut self, conv_pair: &'a mut CabSimPair) -> Self {
        self.conv_pair = Some(conv_pair);
        self
    }
}

/// Intermediate working buffers for the DSP pipeline stages.
///
/// Pre-allocated 64-byte aligned slices passed to pipeline processing stages to hold
/// intermediate rate-converted, oversampled, and model-processed audio blocks without RT heap allocations.
pub struct DspBuffers<'a> {
    /// Intermediate post-resampler buffer for left channel at neural model sample rate.
    pub resamp_mid_l: &'a mut [f32],
    /// Intermediate post-resampler buffer for right channel at neural model sample rate.
    pub resamp_mid_r: &'a mut [f32],
    /// Final resampler output buffer for left channel after downsampling back to host rate.
    pub resamp_out_l: &'a mut [f32],
    /// Final resampler output buffer for right channel after downsampling back to host rate.
    pub resamp_out_r: &'a mut [f32],
    /// Neural model output buffer for left channel at model rate.
    pub model_out_l: &'a mut [f32],
    /// Neural model output buffer for right channel at model rate.
    pub model_out_r: &'a mut [f32],
    /// Oversampled input buffer for left channel (pre-neural model, 2× or 4× host rate).
    pub os_in_l: &'a mut [f32],
    /// Oversampled input buffer for right channel (pre-neural model, 2× or 4× host rate).
    pub os_in_r: &'a mut [f32],
    /// Oversampled model output buffer for left channel (post-neural model, 2× or 4× host rate).
    pub os_model_l: &'a mut [f32],
    /// Oversampled model output buffer for right channel (post-neural model, 2× or 4× host rate).
    pub os_model_r: &'a mut [f32],
    /// Scratch buffer for WaveNet crossfade second-pass output in inference stage,
    /// used when processing is chunked to avoid overlap with accumulated output.
    pub crossfade_scratch_l: &'a mut [f32],
    /// Scratch buffer for WaveNet crossfade second-pass output (right channel).
    pub crossfade_scratch_r: &'a mut [f32],
}

impl<'a> DspBuffers<'a> {
    /// Creates the buffer set from all twelve working slices (mirrors the struct literal).
    ///
    /// # Examples
    ///
    /// ```
    /// # let mut a = [0.0f32; 8]; let mut b = [0.0f32; 8];
    /// # let mut c = [0.0f32; 8]; let mut d = [0.0f32; 8];
    /// # let mut e = [0.0f32; 8]; let mut f = [0.0f32; 8];
    /// # let mut g = [0.0f32; 8]; let mut h = [0.0f32; 8];
    /// # let mut i = [0.0f32; 8]; let mut j = [0.0f32; 8];
    /// # let mut k = [0.0f32; 8]; let mut l = [0.0f32; 8];
    /// use neural_amp_modeler_rs::dsp::pipeline::DspBuffers;
    ///
    /// let bufs = DspBuffers::from_parts(
    ///     &mut a, &mut b, &mut c, &mut d,
    ///     &mut e, &mut f, &mut g, &mut h,
    ///     &mut i, &mut j, &mut k, &mut l,
    /// );
    /// assert!(bufs.resamp_mid_l.len() == 8);
    /// ```
    #[expect(
        clippy::too_many_arguments,
        reason = "DspBuffers mirrors the public struct literal; all working slices are mandatory"
    )]
    pub fn from_parts(
        resamp_mid_l: &'a mut [f32],
        resamp_mid_r: &'a mut [f32],
        resamp_out_l: &'a mut [f32],
        resamp_out_r: &'a mut [f32],
        model_out_l: &'a mut [f32],
        model_out_r: &'a mut [f32],
        os_in_l: &'a mut [f32],
        os_in_r: &'a mut [f32],
        os_model_l: &'a mut [f32],
        os_model_r: &'a mut [f32],
        crossfade_scratch_l: &'a mut [f32],
        crossfade_scratch_r: &'a mut [f32],
    ) -> Self {
        Self {
            resamp_mid_l,
            resamp_mid_r,
            resamp_out_l,
            resamp_out_r,
            model_out_l,
            model_out_r,
            os_in_l,
            os_in_r,
            os_model_l,
            os_model_r,
            crossfade_scratch_l,
            crossfade_scratch_r,
        }
    }

    /// Creates the buffer set from the ten stage slices, leaving both crossfade
    /// scratches empty (`&mut []`).
    ///
    /// Attach scratch slices with [`with_crossfade_scratch_l`](Self::with_crossfade_scratch_l) /
    /// [`with_crossfade_scratch_r`](Self::with_crossfade_scratch_r) when chunked
    /// WaveNet crossfade processing is used.
    #[expect(
        clippy::too_many_arguments,
        reason = "DspBuffers mirrors the public struct literal; all stage slices are mandatory"
    )]
    pub fn new(
        resamp_mid_l: &'a mut [f32],
        resamp_mid_r: &'a mut [f32],
        resamp_out_l: &'a mut [f32],
        resamp_out_r: &'a mut [f32],
        model_out_l: &'a mut [f32],
        model_out_r: &'a mut [f32],
        os_in_l: &'a mut [f32],
        os_in_r: &'a mut [f32],
        os_model_l: &'a mut [f32],
        os_model_r: &'a mut [f32],
    ) -> Self {
        Self::from_parts(
            resamp_mid_l,
            resamp_mid_r,
            resamp_out_l,
            resamp_out_r,
            model_out_l,
            model_out_r,
            os_in_l,
            os_in_r,
            os_model_l,
            os_model_r,
            &mut [],
            &mut [],
        )
    }

    /// Attaches the left-channel crossfade scratch slice (chainable).
    pub fn with_crossfade_scratch_l(mut self, crossfade_scratch_l: &'a mut [f32]) -> Self {
        self.crossfade_scratch_l = crossfade_scratch_l;
        self
    }

    /// Attaches the right-channel crossfade scratch slice (chainable).
    pub fn with_crossfade_scratch_r(mut self, crossfade_scratch_r: &'a mut [f32]) -> Self {
        self.crossfade_scratch_r = crossfade_scratch_r;
        self
    }
}
