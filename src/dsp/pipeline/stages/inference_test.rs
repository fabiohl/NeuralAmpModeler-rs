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
fn test_run_inference_streaming_irregular_blocks() {
    for &host in TEST_HOST_RATES {
        for &n in IRREGULAR_BLOCKS {
            check_streaming_inference(host, 48_000, n, 16);
        }
    }
}

#[test]
fn test_run_inference_streaming_max_block() {
    check_streaming_inference(44_100, 48_000, 8192, 8);
    check_streaming_inference(96_000, 48_000, 8192, 8);
    check_streaming_inference(48_000, 48_000, 8192, 8);
}

#[test]
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
