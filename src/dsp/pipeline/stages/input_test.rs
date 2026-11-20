// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::common::params::AdaptiveComputeMode;
use crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION;
use crate::dsp::adaptive::AdaptiveCompute;
use crate::dsp::gate::{DynamicHysteresis, GateParams};
use crate::dsp::oversample::{OversampleEngine, OversampleFactor};
use crate::dsp::pipeline::MAX_RESAMP_BUF;
use crate::dsp::resampler::NamResampler;

#[test]
fn input_stage_stereo_clamps_oversized_n_samples_and_flags_violation() {
    let mut resampler = NamResampler::new_simple(48_000, 48_000).unwrap();
    let mut os_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut os_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut model_l = None;
    let mut model_r = None;
    let gate_params = GateParams {
        mono_epsilon: -1.0,
        ..Default::default()
    };
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = false;
    let rt_status = RtStatusFlags::default();
    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

    let mut ctx = DspPipelineContext::from_parts(
        &mut resampler,
        &mut os_l,
        &mut os_r,
        &mut model_l,
        &mut model_r,
        2.0,
        1.0,
        &gate_params,
        &mut silence_hysteresis,
        &mut mono_hysteresis,
        0.0,
        0.0,
        &mut process_mono,
        &rt_status,
        &mut adaptive,
    );

    let mut samples_l = vec![0.25f32; 256];
    let mut samples_r = vec![-0.125f32; 256];

    let state = apply_input_stage(&mut samples_l, &mut samples_r, 1024, &mut ctx);
    assert_eq!(state, GateState::Open);
    assert!(rt_status.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));

    let expected_l = 0.25 * 2.0 + DENORMAL_DITHER_OFFSET;
    let expected_r = -0.125 * 2.0 + DENORMAL_DITHER_OFFSET;
    for &s in &samples_l {
        assert!((s - expected_l).abs() < 1e-7);
    }
    #[cfg(feature = "dual-mono")]
    for &s in &samples_r {
        assert!((s - expected_r).abs() < 1e-7);
    }
    #[cfg(not(feature = "dual-mono"))]
    let _ = expected_r;
}

#[test]
fn input_stage_mono_clamps_oversized_n_samples_and_flags_violation() {
    let mut resampler = NamResampler::new_simple(48_000, 48_000).unwrap();
    let mut os_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut os_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut model_l = None;
    let mut model_r = None;
    let gate_params = GateParams::default();
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = false;
    let rt_status = RtStatusFlags::default();
    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

    let mut ctx = DspPipelineContext::from_parts(
        &mut resampler,
        &mut os_l,
        &mut os_r,
        &mut model_l,
        &mut model_r,
        2.0,
        1.0,
        &gate_params,
        &mut silence_hysteresis,
        &mut mono_hysteresis,
        0.0,
        0.0,
        &mut process_mono,
        &rt_status,
        &mut adaptive,
    );

    let mut samples_l = vec![0.25f32; 256];
    let state = apply_input_stage_mono(&mut samples_l, 1024, &mut ctx);
    assert_eq!(state, GateState::Open);
    assert!(*ctx.process_mono);
    assert!(rt_status.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));

    let expected_l = 0.25 * 2.0 + DENORMAL_DITHER_OFFSET;
    for &s in &samples_l {
        assert!((s - expected_l).abs() < 1e-7);
    }
}
