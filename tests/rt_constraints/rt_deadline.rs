// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  RT deadline gate — asserts that all SKUs meet the 1.33 ms processing
//  deadline for a 64-sample block at 48 kHz.
//
//  Uses `LatencyHistogram` to measure per-block processing time and enforces
//  an `assert!` on p99 latency. This gate catches performance regressions
//  before they cause xruns in production.
//
//  ## Running
//
//  ```sh
//  cargo test --release --test rt_deadline -- --nocapture
//  taskset -c 0 cargo test --release --test rt_deadline -- --nocapture
//  ```
//
//  ## Constants
//
//  - `RT_DEADLINE_US`: 1330 (1.33 ms @ 48 kHz, 64-sample block)
//  - `WARMUP_BLOCKS`: 256 (stabilize CPU caches and branch predictor)
//  - `MEASURE_BLOCKS`: 2048 (sufficient for stable p99)
//  - `BLOCK_SIZE`: 64

use super::common;
use common::*;

use std::fs;

use neural_amp_modeler_rs::dsp::telemetry::LatencyHistogram;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::NamModel;

/// RT deadline for 64 samples at 48 kHz: 1.33 ms.
const RT_DEADLINE_US: u64 = 1330;

/// Number of warmup blocks before measurement (stabilize CPU state).
const WARMUP_BLOCKS: usize = 256;

/// Number of measured blocks for stable p50/p99 statistics.
const MEASURE_BLOCKS: usize = 2048;

/// DSP block size in samples (standard 48 kHz JACK/PipeWire buffer).
const BLOCK_SIZE: usize = 64;

/// Loads a model from `tests/fixtures/models/<filename>`, returning `None`
/// if the file does not exist (skip gracefully).
fn load_model(filename: &str) -> Option<neural_amp_modeler_rs::models::StaticModel> {
    let path = model_path(filename);
    if !path.exists() {
        eprintln!("SKIP: {} not found.", filename);
        return None;
    }
    let json_data =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to read {filename}: {e}"));
    let model_data =
        parse_nam_json(&json_data).unwrap_or_else(|e| panic!("Failed to parse {filename}: {e}"));
    let mut model = build_model(&model_data)
        .unwrap_or_else(|e| panic!("Dispatcher failed for {filename}: {e}"));
    model.prewarm(2048);
    Some(*model)
}

/// Measures p99 processing time for a model, asserts it is under the RT deadline.
fn assert_rt_deadline(label: &str, model: &mut neural_amp_modeler_rs::models::StaticModel) {
    let input = generate_sine_440hz(BLOCK_SIZE);
    let out_ch = match model {
        neural_amp_modeler_rs::models::StaticModel::ConvNet(c) => c.out_channels(),
        _ => 1,
    };
    let output_size = out_ch * BLOCK_SIZE;
    let mut output = vec![0.0f32; output_size];
    let hist = LatencyHistogram::new();

    // Warmup
    for _ in 0..WARMUP_BLOCKS {
        model.process(&input, &mut output);
    }

    // Measurement
    for _ in 0..MEASURE_BLOCKS {
        let start = std::time::Instant::now();
        model.process(&input, &mut output);
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        hist.record(elapsed_ns);
    }

    let p50 = hist.get_percentile(0.50) / 1000;
    let p99 = hist.get_percentile(0.99) / 1000;
    let exact_max = hist.get_exact_max() / 1000;

    println!(
        "[{label}] P50={p50}μs  P99={p99}μs  exact_max={exact_max}μs  \
         deadline={RT_DEADLINE_US}μs  blocks={MEASURE_BLOCKS}"
    );

    if !cfg!(debug_assertions) {
        assert!(
            p99 < RT_DEADLINE_US,
            "[{label}] P99={p99}μs exceeds RT deadline {RT_DEADLINE_US}μs — regression detected"
        );
        assert!(
            output.iter().all(|s| s.is_finite()),
            "[{label}] Non-finite output sample detected"
        );
    }
}

// ── WaveNet SKUs ──

#[test]
fn test_rt_deadline_wavenet_standard() {
    let mut model = load_model("BossWN-standard.nam").unwrap();
    assert_rt_deadline("WaveNet-Standard", &mut model);
}

#[test]
fn test_rt_deadline_wavenet_feather() {
    if let Some(mut model) = load_model("BossWN-feather.nam") {
        assert_rt_deadline("WaveNet-Feather", &mut model);
    }
}

#[test]
fn test_rt_deadline_wavenet_lite() {
    if let Some(mut model) = load_model("BossWN-lite.nam") {
        assert_rt_deadline("WaveNet-Lite", &mut model);
    }
}

#[test]
fn test_rt_deadline_wavenet_nano() {
    if let Some(mut model) = load_model("BossWN-nano.nam") {
        assert_rt_deadline("WaveNet-Nano", &mut model);
    }
}

// ── A2 SKUs ──

#[test]
fn test_rt_deadline_a2_full() {
    if let Some(mut model) = load_model("wavenet_a2_full.nam") {
        assert_rt_deadline("A2-Full", &mut model);
    }
}

#[test]
fn test_rt_deadline_a2_lite() {
    if let Some(mut model) = load_model("wavenet_a2_lite.nam") {
        assert_rt_deadline("A2-Lite", &mut model);
    }
}

// ── LSTM SKUs ──

#[test]
fn test_rt_deadline_lstm_1x16() {
    if let Some(mut model) = load_model("BossLSTM-1x16.nam") {
        assert_rt_deadline("LSTM-1x16", &mut model);
    }
}

#[test]
fn test_rt_deadline_lstm_2x8() {
    if let Some(mut model) = load_model("BossLSTM-2x8.nam") {
        assert_rt_deadline("LSTM-2x8", &mut model);
    }
}

// ── Linear / ConvNet ──

#[test]
fn test_rt_deadline_linear() {
    if let Some(mut model) = load_model("linear_test.nam") {
        assert_rt_deadline("Linear", &mut model);
    }
}

#[test]
fn test_rt_deadline_convnet() {
    if let Some(mut model) = load_model("convnet_test.nam") {
        assert_rt_deadline("ConvNet", &mut model);
    }
}

// ── Container / Adaptive States ──

/// Tests the container model (A2-Full + A2-Lite submodels) at all
/// three adaptive states: Full (1.0), Reduced (0.25), Minimal (0.0).
///
/// This validates that the channel-reduction and layer-skipping
/// mechanisms of the adaptive FSM produce models that still meet
/// the RT deadline — and that the code path for slimmable swap
/// does not introduce latency regression.
#[test]
fn test_rt_deadline_adaptive_states() {
    let path = model_path("wavenet_a2_container.nam");
    if !path.exists() {
        eprintln!("SKIP: wavenet_a2_container.nam not found.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read container model");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse container model");
    let mut model = build_model(&model_data).expect("Dispatcher failed for container model");
    model.prewarm(2048);

    // Full — quality = 1.0
    model.set_slimmable_size(1.0, None);
    // Process a few blocks to settle crossfade if any
    let input = generate_sine_440hz(BLOCK_SIZE);
    let mut output = vec![0.0f32; BLOCK_SIZE];
    for _ in 0..64 {
        model.process(&input, &mut output);
    }
    assert_rt_deadline("Container-Full", &mut model);

    // Reduced — quality = 0.25
    model.set_slimmable_size(0.25, None);
    for _ in 0..64 {
        model.process(&input, &mut output);
    }
    assert_rt_deadline("Container-Reduced", &mut model);

    // Minimal — quality = 0.0
    model.set_slimmable_size(0.0, None);
    for _ in 0..64 {
        model.process(&input, &mut output);
    }
    assert_rt_deadline("Container-Minimal", &mut model);
}

// ── WaveNet Dynamic (free-geometry fallback path) ──

#[test]
fn test_rt_deadline_wavenet_dynamic() {
    if let Some(mut model) = load_model("wavenet_dyn_free.nam") {
        assert_rt_deadline("WaveNet-Dynamic", &mut model);
    }
}

// ── LSTM Dynamic ──

#[test]
fn test_rt_deadline_lstm_dynamic() {
    if let Some(mut model) = load_model("lstm_dyn_test.nam") {
        assert_rt_deadline("LSTM-Dynamic", &mut model);
    }
}

// ── A2 Dynamic ──

#[test]
fn test_rt_deadline_a2_dynamic() {
    if let Some(mut model) = load_model("a2_dynamic_gated_ch8.nam") {
        assert_rt_deadline("A2-Dynamic-Gated", &mut model);
    }
}
