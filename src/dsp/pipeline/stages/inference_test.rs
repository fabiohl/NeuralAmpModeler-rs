// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};
use crate::common::params::AdaptiveComputeMode;
use crate::common::spsc::RtStatusFlags;
use crate::dsp::adaptive::AdaptiveCompute;
use crate::dsp::gate::{DynamicHysteresis, GateParams};
use crate::dsp::oversample::{OversampleEngine, OversampleFactor};
use crate::loader::dispatcher::build_model;
use crate::loader::nam_json::parse_nam_json;
use crate::models::{NamModel, StaticModel};
use std::fs;
use std::path::PathBuf;

fn load_test_model(name: &str) -> Box<StaticModel> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(name);
    let json_data = fs::read_to_string(path).expect("Failed to read model file");
    let model_data = parse_nam_json(&json_data).expect("Failed to process model JSON");
    build_model(&model_data).expect("Failed to build model")
}

/// Runs `run_inference_streaming` with a real NAM model over irregular
/// blocks, validating exact cardinality, conservation and zero allocation.
fn check_streaming_inference(host: u32, model_rate: u32, n: usize, iterations: usize) {
    let mut model = load_test_model("BossWN-nano.nam");
    model.prewarm(2048);

    let mut stream =
        StreamingResampleBuffer::new(host, model_rate, 8192).expect("streaming buffer failed");
    let latency = stream.latency_samples() as u64;

    let mut os_engine_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut os_engine_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let gate_params = GateParams::default();
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = false;
    let rt_status = RtStatusFlags::default();
    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

    let mut os_buf: [f32; MAX_RESAMP_BUF * 6] = [0.0f32; MAX_RESAMP_BUF * 6];
    let (os_in_l, rest) = os_buf.split_at_mut(MAX_RESAMP_BUF);
    let (os_in_r, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_l, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_r, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (scratch_l, scratch_r) = rest.split_at_mut(MAX_RESAMP_BUF);

    let in_l = vec![0.05f32; n];
    let in_r = vec![0.02f32; n];
    let mut out_l = vec![f32::NAN; n];
    let mut out_r = vec![f32::NAN; n];

    let mut model_opt: Option<Box<StaticModel>> = Some(model);
    let mut model_r_opt: Option<Box<StaticModel>> = None;
    let mut resampler = NamResampler::new_simple(host, model_rate).unwrap();
    let mut ctx = DspPipelineContext::from_parts(
        &mut resampler,
        &mut os_engine_l,
        &mut os_engine_r,
        &mut model_opt,
        &mut model_r_opt,
        1.0,
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

    let _guard = TrackingGuard::new();
    for _ in 0..iterations {
        let written = run_inference_streaming(
            &in_l,
            &in_r,
            &mut out_l,
            &mut out_r,
            n,
            &mut ctx,
            &mut stream,
            os_in_l,
            os_in_r,
            os_model_l,
            os_model_r,
            scratch_l,
            scratch_r,
        );
        assert_eq!(written, n, "host={host} model={model_rate} n={n}");
        assert!(
            out_l[..n].iter().all(|x| x.is_finite()),
            "non-finite L at host={host} model={model_rate} n={n}"
        );
        assert!(
            out_r[..n].iter().all(|x| x.is_finite()),
            "non-finite R at host={host} model={model_rate} n={n}"
        );
        assert_eq!(stream.input_pending(), 0);
        assert_eq!(stream.model_pending(), 0);
        assert!(stream.output_pending() <= stream.output_capacity_actual());
    }

    let allocs = get_alloc_count();
    drop(_guard);
    assert_eq!(allocs, 0, "streaming inference must be zero-alloc");

    assert_eq!(
        stream.output_real_total(),
        stream.input_total().saturating_sub(latency),
        "conservation violated: host={host} model={model_rate} n={n}"
    );
    assert_eq!(stream.underflow_total(), 0);
}

const TEST_HOST_RATES: &[u32] = &[44_100, 48_000, 96_000, 192_000];
const IRREGULAR_BLOCKS: &[usize] = &[1, 7, 31, 63, 64, 65, 127, 256];

#[test]
#[ignore = "long: teste pesado (> 2.0s em debug)"]
fn test_run_inference_streaming_irregular_blocks() {
    for &host in TEST_HOST_RATES {
        for &n in IRREGULAR_BLOCKS {
            check_streaming_inference(host, 48_000, n, 16);
        }
    }
}

#[test]
#[ignore = "long: teste pesado (> 2.0s em debug)"]
fn test_run_inference_streaming_max_block() {
    check_streaming_inference(44_100, 48_000, 8192, 8);
    check_streaming_inference(96_000, 48_000, 8192, 8);
    check_streaming_inference(48_000, 48_000, 8192, 8);
}

#[test]
#[ignore = "long: teste pesado (> 2.0s em debug)"]
fn test_run_inference_streaming_oversized_block_subchunked() {
    // Host block larger than the adapter's max_block must be processed in
    // sub-blocks without dropping input or fabricating output.
    let host = 44_100u32;
    let model_rate = 48_000u32;
    let n = 700usize;
    let iterations = 16usize;
    let max_block = 256usize;

    let mut model = load_test_model("BossWN-nano.nam");
    model.prewarm(2048);
    let mut stream =
        StreamingResampleBuffer::new(host, model_rate, max_block).expect("streaming buffer failed");
    let latency = stream.latency_samples() as u64;

    let mut os_engine_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut os_engine_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let gate_params = GateParams::default();
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = false;
    let rt_status = RtStatusFlags::default();
    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);
    let mut model_opt: Option<Box<StaticModel>> = Some(model);
    let mut model_r_opt: Option<Box<StaticModel>> = None;
    let mut resampler = NamResampler::new_simple(host, model_rate).unwrap();
    let mut ctx = DspPipelineContext::from_parts(
        &mut resampler,
        &mut os_engine_l,
        &mut os_engine_r,
        &mut model_opt,
        &mut model_r_opt,
        1.0,
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

    let mut os_buf: [f32; MAX_RESAMP_BUF * 6] = [0.0f32; MAX_RESAMP_BUF * 6];
    let (os_in_l, rest) = os_buf.split_at_mut(MAX_RESAMP_BUF);
    let (os_in_r, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_l, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_r, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (scratch_l, scratch_r) = rest.split_at_mut(MAX_RESAMP_BUF);

    let in_l = vec![0.05f32; n];
    let in_r = vec![0.02f32; n];
    let mut out_l = vec![f32::NAN; n];
    let mut out_r = vec![f32::NAN; n];

    let _guard = TrackingGuard::new();
    for _ in 0..iterations {
        let written = run_inference_streaming(
            &in_l,
            &in_r,
            &mut out_l,
            &mut out_r,
            n,
            &mut ctx,
            &mut stream,
            os_in_l,
            os_in_r,
            os_model_l,
            os_model_r,
            scratch_l,
            scratch_r,
        );
        assert_eq!(written, n, "oversized block must produce exactly n");
        assert!(
            out_l[..n].iter().all(|x| x.is_finite()),
            "non-finite output for oversized block"
        );
        assert_eq!(stream.input_pending(), 0);
        assert!(stream.output_pending() <= stream.output_capacity_actual());
    }
    let allocs = get_alloc_count();
    drop(_guard);
    assert_eq!(allocs, 0, "sub-chunked processing must be zero-alloc");

    assert_eq!(
        stream.output_real_total(),
        stream.input_total().saturating_sub(latency),
        "sub-chunked conservation violated"
    );
    assert_eq!(stream.underflow_total(), 0);
}

#[test]
fn test_run_inference_streaming_short_output_slice_returns_zero_and_raises_flag() {
    let host = 48_000u32;
    let model_rate = 48_000u32;
    let mut stream =
        StreamingResampleBuffer::new(host, model_rate, 256).expect("streaming buffer failed");
    let mut os_engine_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut os_engine_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let gate_params = GateParams::default();
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = false;
    let rt_status = RtStatusFlags::default();
    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);
    let mut model_opt: Option<Box<StaticModel>> = None;
    let mut model_r_opt: Option<Box<StaticModel>> = None;
    let mut resampler = NamResampler::new_simple(host, model_rate).unwrap();
    let mut ctx = DspPipelineContext::from_parts(
        &mut resampler,
        &mut os_engine_l,
        &mut os_engine_r,
        &mut model_opt,
        &mut model_r_opt,
        1.0,
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

    let mut os_buf: [f32; MAX_RESAMP_BUF * 6] = [0.0f32; MAX_RESAMP_BUF * 6];
    let (os_in_l, rest) = os_buf.split_at_mut(MAX_RESAMP_BUF);
    let (os_in_r, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_l, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_r, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (scratch_l, scratch_r) = rest.split_at_mut(MAX_RESAMP_BUF);

    let in_l = vec![0.05f32; 64];
    let in_r = vec![0.02f32; 64];
    let mut out_l = vec![99.0f32; 1]; // short output: 1 element vs 64
    let mut out_r = vec![99.0f32; 1];

    let written = run_inference_streaming(
        &in_l,
        &in_r,
        &mut out_l,
        &mut out_r,
        64,
        &mut ctx,
        &mut stream,
        os_in_l,
        os_in_r,
        os_model_l,
        os_model_r,
        scratch_l,
        scratch_r,
    );

    assert_eq!(written, 0, "short output slice must return 0");
    assert!(
        rt_status.check_flag(crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION),
        "short output slice must raise RT_STATUS_HOST_CONTRACT_VIOLATION"
    );
    assert_eq!(out_l[0], 0.0, "safe slice portion must be zeroed");
    assert_eq!(out_r[0], 0.0, "safe slice portion must be zeroed");
}

#[test]
fn test_run_inference_48khz_dsp_buffers_new_crossfade_no_contract_violation() {
    use crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION;
    use crate::dsp::pipeline::{DspBuffers, capture_dsp_pipeline};

    let mut model = load_test_model("BossWN-nano.nam");
    assert!(model.supports_layer_skip());
    model.prewarm(2048);

    let mut resampler = NamResampler::new_simple(48_000, 48_000).unwrap();
    let mut os_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut os_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let gate_params = GateParams::default();
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = false;

    let update_flags = RtStatusFlags::default();
    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
    for _ in 0..3 {
        adaptive.update(900, 1000, 48_000, &update_flags);
    }
    assert!(adaptive.is_crossfading());

    let rt_status = RtStatusFlags::default();
    let mut model_opt: Option<Box<StaticModel>> = Some(model);
    let mut model_r_opt: Option<Box<StaticModel>> = None;

    let mut resamp_mid_l = vec![0.0f32; MAX_RESAMP_BUF];
    let mut resamp_mid_r = vec![0.0f32; MAX_RESAMP_BUF];
    let mut resamp_out_l = vec![0.0f32; MAX_RESAMP_BUF];
    let mut resamp_out_r = vec![0.0f32; MAX_RESAMP_BUF];
    let mut model_out_l = vec![0.0f32; MAX_RESAMP_BUF];
    let mut model_out_r = vec![0.0f32; MAX_RESAMP_BUF];
    let mut os_in_l = vec![0.0f32; MAX_RESAMP_BUF];
    let mut os_in_r = vec![0.0f32; MAX_RESAMP_BUF];
    let mut os_model_l = vec![0.0f32; MAX_RESAMP_BUF];
    let mut os_model_r = vec![0.0f32; MAX_RESAMP_BUF];

    let n = 256usize;
    let mut samples_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
    let mut samples_r = samples_l.clone();

    let ctx = DspPipelineContext::from_parts(
        &mut resampler,
        &mut os_l,
        &mut os_r,
        &mut model_opt,
        &mut model_r_opt,
        1.0,
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
    let bufs = DspBuffers::new(
        &mut resamp_mid_l,
        &mut resamp_mid_r,
        &mut resamp_out_l,
        &mut resamp_out_r,
        &mut model_out_l,
        &mut model_out_r,
        &mut os_in_l,
        &mut os_in_r,
        &mut os_model_l,
        &mut os_model_r,
    );

    let n_out = capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48_000);
    assert_eq!(n_out, n);
    assert!(!rt_status.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));
    assert_eq!(
        rt_status
            .status_bits
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert!(resamp_out_l[..n].iter().all(|x| x.is_finite()));
    assert!(resamp_out_l[..n].iter().any(|&x| x.abs() > 1e-4));
}

#[test]
fn test_pure_mono_empty_r_slices_no_panic_no_violation() {
    use crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION;

    for &crossfade in &[false, true] {
        for &host_rate in &[48_000u32, 44_100u32] {
            let mut model = load_test_model("BossWN-nano.nam");
            model.prewarm(2048);

            let mut resampler = NamResampler::new_simple(host_rate, 48_000).unwrap();
            let mut os_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
            let mut os_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
            let gate_params = GateParams::default();
            let mut silence_hysteresis = DynamicHysteresis::new();
            let mut mono_hysteresis = DynamicHysteresis::new();
            let mut process_mono = true;
            let update_flags = RtStatusFlags::default();
            let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
            if crossfade {
                for _ in 0..3 {
                    adaptive.update(900, 1000, host_rate, &update_flags);
                }
                assert!(adaptive.is_crossfading());
            }

            let rt_status = RtStatusFlags::default();
            let mut model_opt: Option<Box<StaticModel>> = Some(model);
            let mut model_r_opt: Option<Box<StaticModel>> = None;
            let mut ctx = DspPipelineContext::from_parts(
                &mut resampler,
                &mut os_l,
                &mut os_r,
                &mut model_opt,
                &mut model_r_opt,
                1.0,
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

            let n = 128usize;
            let mut samples_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.25).collect();
            let mut resamp_mid_l = vec![0.0f32; MAX_RESAMP_BUF];
            let mut resamp_out_l = vec![0.0f32; MAX_RESAMP_BUF];
            let mut model_out_l = vec![0.0f32; MAX_RESAMP_BUF];
            let mut scratch_l = vec![0.0f32; MAX_RESAMP_BUF];

            let written = run_inference(
                &mut samples_l,
                &mut [],
                n,
                &mut ctx,
                &mut resamp_mid_l,
                &mut [],
                &mut resamp_out_l,
                &mut [],
                &mut model_out_l,
                &mut [],
                &mut [],
                &mut [],
                &mut [],
                &mut [],
                &mut scratch_l,
                &mut [],
            );
            assert!(written > 0);
            assert!(!rt_status.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));
            assert!(resamp_out_l[..written].iter().all(|x| x.is_finite()));

            // Also test run_inference_streaming with empty R slices
            let mut stream = StreamingResampleBuffer::new(host_rate, 48_000, 1024).unwrap();
            let in_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.25).collect();
            let mut out_l = vec![0.0f32; n];
            for _ in 0..4 {
                let s_written = run_inference_streaming(
                    &in_l,
                    &[],
                    &mut out_l,
                    &mut [],
                    n,
                    &mut ctx,
                    &mut stream,
                    &mut [],
                    &mut [],
                    &mut [],
                    &mut [],
                    &mut scratch_l,
                    &mut [],
                );
                assert_eq!(s_written, n);
                assert!(!rt_status.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));
                assert!(out_l.iter().all(|x| x.is_finite()));
            }
        }
    }
}

#[test]
fn test_wavenet_crossfade_oversampled_sine_continuity_and_zero_alloc() {
    use crate::dsp::pipeline::stages::apply_output_stage;

    const SR: u32 = 48_000;
    const BLOCK: usize = 128;

    let run_scenario = |factor: OversampleFactor, trigger_crossfade: bool| -> (Vec<f32>, usize) {
        let mut model = load_test_model("BossWN-nano.nam");
        model.prewarm(2048);

        let mult = factor.multiplier();
        let mut os_l = OversampleEngine::new(factor, MAX_RESAMP_BUF).unwrap();
        let mut os_r = OversampleEngine::new(factor, MAX_RESAMP_BUF).unwrap();
        let mut resampler = NamResampler::new_simple(SR, SR).unwrap();
        let gate_params = GateParams::default();
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = true;
        let rt_status = RtStatusFlags::default();
        let mut adaptive = AdaptiveCompute::new(if trigger_crossfade {
            AdaptiveComputeMode::Conservative
        } else {
            AdaptiveComputeMode::Off
        });

        let mut model_opt: Option<Box<StaticModel>> = Some(model);
        let mut model_r_opt: Option<Box<StaticModel>> = None;

        let mut resamp_mid_l = vec![0.0f32; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0f32; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0f32; MAX_RESAMP_BUF];
        let mut resamp_out_r = vec![0.0f32; MAX_RESAMP_BUF];
        let mut model_out_l = vec![0.0f32; MAX_RESAMP_BUF];
        let mut model_out_r = vec![0.0f32; MAX_RESAMP_BUF];
        let mut os_in_l = vec![0.0f32; MAX_RESAMP_BUF * mult];
        let mut os_in_r = vec![0.0f32; MAX_RESAMP_BUF * mult];
        let mut os_model_l = vec![0.0f32; MAX_RESAMP_BUF * mult];
        let mut os_model_r = vec![0.0f32; MAX_RESAMP_BUF * mult];
        let mut scratch_l = vec![0.0f32; MAX_RESAMP_BUF];
        let mut scratch_r = vec![0.0f32; MAX_RESAMP_BUF];

        let mut samples_l = vec![0.0f32; BLOCK];
        let mut samples_r = vec![0.0f32; BLOCK];
        let mut collected = Vec::with_capacity(BLOCK * 36);
        let mut phase = 0usize;

        let _guard = TrackingGuard::new();
        for block_idx in 0..36usize {
            for i in 0..BLOCK {
                let s =
                    (2.0 * std::f32::consts::PI * 220.0 * (phase as f32) / SR as f32).sin() * 0.25;
                samples_l[i] = s;
                samples_r[i] = s;
                phase += 1;
            }

            if trigger_crossfade && block_idx == 6 {
                // Trigger Full -> Reduced crossfade (32 ms = 1536 samples = 12 blocks)
                for _ in 0..3 {
                    adaptive.update(900, 1000, SR, &rt_status);
                }
                assert!(adaptive.is_crossfading());
            } else if trigger_crossfade && block_idx == 20 {
                // Trigger Reduced -> Full recovery crossfade
                for _ in 0..5 {
                    adaptive.update(100, 1000, SR, &rt_status);
                }
                assert!(adaptive.is_crossfading());
            }

            let mut ctx = DspPipelineContext::from_parts(
                &mut resampler,
                &mut os_l,
                &mut os_r,
                &mut model_opt,
                &mut model_r_opt,
                1.0,
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

            let n_pw = run_inference(
                &mut samples_l,
                &mut samples_r,
                BLOCK,
                &mut ctx,
                &mut resamp_mid_l,
                &mut resamp_mid_r,
                &mut resamp_out_l,
                &mut resamp_out_r,
                &mut model_out_l,
                &mut model_out_r,
                &mut os_in_l,
                &mut os_in_r,
                &mut os_model_l,
                &mut os_model_r,
                &mut scratch_l,
                &mut scratch_r,
            );
            assert_eq!(n_pw, BLOCK);

            apply_output_stage(
                &mut resamp_out_l,
                &mut resamp_out_r,
                n_pw,
                1.0,
                &mut silence_hysteresis,
                &rt_status,
                true,
                &mut adaptive,
                SR,
            );

            if block_idx >= 4 {
                collected.extend_from_slice(&resamp_out_l[..n_pw]);
            }
        }
        let allocs = get_alloc_count();
        drop(_guard);
        (collected, allocs)
    };

    for &factor in &[
        OversampleFactor::Off,
        OversampleFactor::X2,
        OversampleFactor::X4,
    ] {
        let (baseline, baseline_allocs) = run_scenario(factor, false);
        let (crossfade, crossfade_allocs) = run_scenario(factor, true);

        assert_eq!(baseline_allocs, 0);
        assert_eq!(
            crossfade_allocs, 0,
            "oversampled crossfade must be zero-alloc"
        );
        assert!(crossfade.iter().all(|x| x.is_finite()));

        let (base_jump_idx, max_step_base) = baseline
            .windows(2)
            .enumerate()
            .map(|(i, w)| (i, (w[1] - w[0]).abs()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        // `collected` starts at block 4, so blocks 4..19 (indices 0..15*BLOCK) cover
        // steady Full -> Reduced crossfade -> steady Reduced, where all layers start warm.
        let degrade_end = (19 - 4) * BLOCK;
        let (degrade_idx, max_step_degrade) = crossfade[..degrade_end]
            .windows(2)
            .enumerate()
            .map(|(i, w)| (i, (w[1] - w[0]).abs()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let (recover_idx, max_step_recover) = crossfade
            .windows(2)
            .enumerate()
            .map(|(i, w)| (i, (w[1] - w[0]).abs()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();

        // Full -> Reduced (blocks 4..19): with continuous per-sample ramp, uninterrupted OS FIR
        // delay lines, post_first_starts sync, and dominant-branch state preservation, step diff
        // remains small across the entire transition (Measured: X2 <= 0.034, X4 <= 0.035, Off <= 0.075;
        // unfixed OS chunking/state-desync produced 0.136).
        assert!(
            max_step_degrade < 0.045_f32.max(max_step_base * 1.5),
            "Degrade discontinuity with factor={factor:?}: max_step_degrade={max_step_degrade} at idx={degrade_idx} (block={}, off={}) vs max_step_base={max_step_base} (at {base_jump_idx})",
            4 + degrade_idx / BLOCK,
            degrade_idx % BLOCK,
        );

        // Check crossfade entry/exit block boundaries (blocks 6, 18, 20, 32 relative to block 4).
        for boundary_block in [6usize, 18, 20, 32] {
            let b_idx = (boundary_block - 4) * BLOCK;
            let step = (crossfade[b_idx] - crossfade[b_idx - 1]).abs();
            assert!(
                step <= max_step_base * 1.2 + 0.01,
                "Boundary jump at block={boundary_block} factor={factor:?}: step={step} vs max_step_base={max_step_base}"
            );
        }

        // Reduced -> Full wakes dormant upper layers (e.g. d=128/256/512) while `t` ramps up;
        // verify it stays bounded below 0.12 (measured ~0.0855 at 2x OS, without large pops > 0.15).
        assert!(
            max_step_recover < 0.12,
            "Recovery discontinuity with factor={factor:?}: max_step_recover={max_step_recover} at idx={recover_idx} (block={}, off={}) vs max_step_base={max_step_base}",
            4 + recover_idx / BLOCK,
            recover_idx % BLOCK,
        );
    }
}
