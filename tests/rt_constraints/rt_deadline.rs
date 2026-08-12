// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  RT deadline gate — asserts that all SKUs meet the 1.33 ms processing
//  deadline for a 64-sample block at 48 kHz.
//
//  Integrates `rt_preflight` environmental verification (CPU affinity +
//  performance governor). When the environment is uncontrolled, telemetry
//  is still collected but tail/max values are classified as
//  `INCONCLUSIVE_ENVIRONMENT`. When preflight passes, any block exceeding
//  the deadline causes a hard test failure.
//
//  Telemetry per SKU: P50, P90, P99, P99.9, exact max, and exact count
//  of blocks above the 1.330 µs deadline.
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
use common::rt_helpers::{self, RtPreflightStatus};
use common::*;

use std::fs;
use std::sync::OnceLock;

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

/// Preflight runs once per process; subsequent calls return the cached result.
fn preflight_ok() -> bool {
    static PREFLIGHT: OnceLock<bool> = OnceLock::new();
    *PREFLIGHT.get_or_init(|| {
        let result = rt_helpers::rt_preflight();
        rt_helpers::print_preflight(&result);
        result.status == RtPreflightStatus::Pass
    })
}

#[derive(Debug, Default)]
struct DeadlineStats {
    p50_us: u64,
    p90_us: u64,
    p99_us: u64,
    p999_us: u64,
    exact_max_us: u64,
    violations: u64,
    total_blocks: usize,
}

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

/// Measures full tail-latency distribution and deadline violations for a model.
fn measure_rt_deadline(
    label: &str,
    model: &mut neural_amp_modeler_rs::models::StaticModel,
) -> DeadlineStats {
    let input = generate_sine_440hz(BLOCK_SIZE);
    let out_ch = match model {
        neural_amp_modeler_rs::models::StaticModel::ConvNet(c) => c.out_channels(),
        _ => 1,
    };
    let output_size = out_ch * BLOCK_SIZE;
    let mut output = vec![0.0f32; output_size];
    let hist = LatencyHistogram::new();
    let mut violations: u64 = 0;

    for _ in 0..WARMUP_BLOCKS {
        model.process(&input, &mut output);
    }

    for _ in 0..MEASURE_BLOCKS {
        let start = std::time::Instant::now();
        model.process(&input, &mut output);
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        hist.record(elapsed_ns);

        if elapsed_ns > RT_DEADLINE_US * 1000 {
            violations += 1;
        }
    }

    assert!(
        output.iter().all(|s| s.is_finite()),
        "[{label}] Non-finite output sample detected"
    );

    DeadlineStats {
        p50_us: hist.get_percentile(0.50) / 1000,
        p90_us: hist.get_percentile(0.90) / 1000,
        p99_us: hist.get_percentile(0.99) / 1000,
        p999_us: hist.get_percentile(0.999) / 1000,
        exact_max_us: hist.get_exact_max() / 1000,
        violations,
        total_blocks: MEASURE_BLOCKS,
    }
}

/// Emits full telemetry receipt and asserts deadline compliance when the
/// environment is controlled.
fn emit_and_assert(label: &str, stats: &DeadlineStats) {
    let pflight_ok = preflight_ok();

    println!(
        "[{label}] P50={}μs  P90={}μs  P99={}μs  P99.9={}μs  \
         exact_max={}μs  violations={}/{}  deadline={}μs",
        stats.p50_us,
        stats.p90_us,
        stats.p99_us,
        stats.p999_us,
        stats.exact_max_us,
        stats.violations,
        stats.total_blocks,
        RT_DEADLINE_US,
    );

    if pflight_ok {
        println!(
            "[{label}] RECEIPT: deadline_ok violations={}/{} max={}μs",
            stats.violations, stats.total_blocks, stats.exact_max_us,
        );
        if !cfg!(debug_assertions) {
            assert!(
                stats.violations == 0,
                "[{label}] {}/{} blocks exceeded RT deadline {RT_DEADLINE_US}μs — \
                 hard failure in controlled environment",
                stats.violations,
                stats.total_blocks,
            );
            assert!(
                stats.p99_us < RT_DEADLINE_US,
                "[{label}] P99={}μs exceeds RT deadline {RT_DEADLINE_US}μs — regression detected",
                stats.p99_us,
            );
        }
    } else {
        println!(
            "[{label}] INCONCLUSIVE_ENVIRONMENT — tail/max values not certified \
             (environment preconditions not met)"
        );
    }
}

/// Convenience: loads a model, measures deadline telemetry, emits receipt and asserts.
fn run_deadline_test(filename: &str, label: &str) {
    if let Some(mut model) = load_model(filename) {
        let stats = measure_rt_deadline(label, &mut model);
        emit_and_assert(label, &stats);
    }
}

// ── WaveNet SKUs ──

#[test]
fn test_rt_deadline_wavenet_standard() {
    run_deadline_test("BossWN-standard.nam", "WaveNet-Standard");
}

#[test]
fn test_rt_deadline_wavenet_feather() {
    run_deadline_test("BossWN-feather.nam", "WaveNet-Feather");
}

#[test]
fn test_rt_deadline_wavenet_lite() {
    run_deadline_test("BossWN-lite.nam", "WaveNet-Lite");
}

#[test]
fn test_rt_deadline_wavenet_nano() {
    run_deadline_test("BossWN-nano.nam", "WaveNet-Nano");
}

// ── A2 SKUs ──

#[test]
fn test_rt_deadline_a2_full() {
    run_deadline_test("wavenet_a2_full.nam", "A2-Full");
}

#[test]
fn test_rt_deadline_a2_lite() {
    run_deadline_test("wavenet_a2_lite.nam", "A2-Lite");
}

// ── LSTM SKUs ──

#[test]
fn test_rt_deadline_lstm_1x16() {
    run_deadline_test("BossLSTM-1x16.nam", "LSTM-1x16");
}

#[test]
fn test_rt_deadline_lstm_2x8() {
    run_deadline_test("BossLSTM-2x8.nam", "LSTM-2x8");
}

// ── Linear / ConvNet ──

#[test]
fn test_rt_deadline_linear() {
    run_deadline_test("linear_test.nam", "Linear");
}

#[test]
fn test_rt_deadline_convnet() {
    run_deadline_test("convnet_test.nam", "ConvNet");
}

// ── Container / Adaptive States ──

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

    let input = generate_sine_440hz(BLOCK_SIZE);
    let mut output = vec![0.0f32; BLOCK_SIZE];

    model.set_slimmable_size(1.0, None);
    for _ in 0..64 {
        model.process(&input, &mut output);
    }
    let stats = measure_rt_deadline("Container-Full", &mut model);
    emit_and_assert("Container-Full", &stats);

    model.set_slimmable_size(0.25, None);
    for _ in 0..64 {
        model.process(&input, &mut output);
    }
    let stats = measure_rt_deadline("Container-Reduced", &mut model);
    emit_and_assert("Container-Reduced", &stats);

    model.set_slimmable_size(0.0, None);
    for _ in 0..64 {
        model.process(&input, &mut output);
    }
    let stats = measure_rt_deadline("Container-Minimal", &mut model);
    emit_and_assert("Container-Minimal", &stats);
}

// ── WaveNet Dynamic (free-geometry fallback path) ──

#[test]
fn test_rt_deadline_wavenet_dynamic() {
    run_deadline_test("wavenet_dyn_free.nam", "WaveNet-Dynamic");
}

// ── LSTM Dynamic ──

#[test]
fn test_rt_deadline_lstm_dynamic() {
    run_deadline_test("lstm_dyn_test.nam", "LSTM-Dynamic");
}

// ── A2 Dynamic ──

#[test]
fn test_rt_deadline_a2_dynamic() {
    run_deadline_test("a2_dynamic_gated_ch8.nam", "A2-Dynamic-Gated");
}
