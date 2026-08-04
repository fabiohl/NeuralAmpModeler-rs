// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Context and working buffers for the DSP pipeline.

use crate::common::spsc::RtStatusFlags;
use crate::dsp::adaptive::AdaptiveCompute;
use crate::dsp::cabsim::adapter::CabSimAdapter;
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
    /// Active cab-sim convolution adapter (`None` = cab-sim bypass, zero cost).
    pub conv: Option<&'a mut CabSimAdapter>,
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
}
