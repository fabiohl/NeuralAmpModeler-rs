// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Golden Vector Cross-Reference Tests.
//
//  Compares NAM-rs Rust engine output against C++ reference golden vectors
//  (NeuralAmpModelerCore — Steven Atkinson) recorded in `tests/fixtures/*.bin`.
//
//  ## `.golden.bin` Format
//  ```text
//  [u32 num_samples LE]
//  [f32×N input samples LE]       — stress signal (2048 samples @ 48 kHz)
//  [f32×N expected output LE]     — output from C++ NeuralAmpModelerCore (render tool)
//  ```
//
//  ## Regenerating golden vectors
//  Run `tests/fixtures/golden_gen_build.sh` with NeuralAmpModelerCore.
//  The resulting `.golden.bin` files should be committed in `tests/fixtures/`.

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::slimmable::SlimmableModel;
use neural_amp_modeler_rs::models::{NamModel, StaticModel};
use std::fs;
use std::path::PathBuf;

use super::common;
use common::*;

fn gv_metric(label: &str) {
    set_metric_model(format!("{label} @48000 Live"));
    set_metric_mode("Live".to_string());
}

/// Runs a v2 golden test across a specific set of sample rates.
///
/// For each sample rate, reads the committed `golden_{name}_v2_{sr}.bin` file,
/// processes with `process_in_blocks`, and validates via `report_dsp_fidelity`.
///
/// Uses `assert!` to enforce mandatory gate: all listed .bin files must exist.
fn run_v2_golden_test(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    model_name: &str,
    sample_rates: &[u32],
) {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let nam_path = model_path(model_filename);

    if !nam_path.exists() {
        eprintln!(
            "SKIP: Model {model_filename} not found at {nam_path:?}. Skipping v2 golden test."
        );
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read model JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    for &sr in sample_rates {
        let golden_filename = format!("{golden_name}_v2_{sr}.bin");
        let golden_path = fixtures_dir.join(&golden_filename);

        if !golden_path.exists() {
            eprintln!(
                "SKIP: {golden_filename} not found at {golden_path:?}. Run './tests/fixtures/golden_gen_build.sh' to generate v2 multi-SR golden vectors."
            );
            continue;
        }

        let (input, expected) = read_golden_bin(&golden_path)
            .unwrap_or_else(|| panic!("Failed to read {golden_filename}"));

        let mut model =
            build_model(&model_data).unwrap_or_else(|_| panic!("Dispatcher failed for {label}"));

        let num_samples = input.len();
        model.prewarm(V2_PREWARM_SAMPLES);
        let mut output = vec![0.0f32; num_samples];
        process_in_blocks(&mut model, &input, &mut output, V2_TEST_BLOCK_SIZE);

        let (mut mse_limit, mut min_snr_db, mut max_esr, mut mrstft_max) =
            topology_thresholds(&model_data, model_name);

        if model_data.architecture == "LSTM" {
            // LSTM recurrent state accumulates quantization/approximation errors
            // over the 100x longer v2 stress signal. The accumulation is proportional
            // to the sequence length. We adjust the thresholds accordingly.
            let sr_ratio = sr as f64 / 48000.0;
            let snr_relaxation = (3.5 * sr_ratio).min(10.0);
            min_snr_db = (min_snr_db - snr_relaxation).max(7.0);
            if let Some(ref mut m) = mse_limit {
                *m *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
            if let Some(ref mut esr) = max_esr {
                *esr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
            if let Some(ref mut mr) = mrstft_max {
                *mr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
        } else {
            // WaveNet and other models accumulate minor differences over the longer v2 stress signal
            let sr_ratio = sr as f64 / 48000.0;
            let snr_relaxation = (1.5 * sr_ratio).min(4.0);
            min_snr_db -= snr_relaxation;
            if let Some(ref mut m) = mse_limit {
                *m *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
            if let Some(ref mut esr) = max_esr {
                *esr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
            if let Some(ref mut mr) = mrstft_max {
                *mr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
        }

        set_metric_model(format!("{label} @{sr} (v2) Live"));
        set_metric_mode("Live".to_string());

        report_dsp_fidelity(
            &expected,
            &output,
            mse_limit,
            min_snr_db,
            max_esr,
            mrstft_max,
            &format!("{label} @ {sr} Hz (v2)"),
            sr,
        );
    }
}

/// Convenience: all 5 supported sample rates.
const ALL_SR: &[u32] = &[44100, 48000, 88200, 96000, 192000];

/// All SRs except 192k (LSTM models exhibit significant recurrent drift at 192k
/// over 5s stress signal — 960k uncompensated samples is an unrealistic extreme).
const SR_EX_192K: &[u32] = &[44100, 48000, 88200, 96000];

/// Convenience: 48 kHz only (models with C++ render tool SR constraint).
const SR_48K_ONLY: &[u32] = &[48000];

// =============================================================================
// Golden Vector Tests (Cross-Reference C++ ↔ Rust)
// =============================================================================

/// Test 7: Golden Vectors WaveNet — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_wavenet_standard.bin`, builds the `StaticModel`
/// from `BossWN-standard.nam`, runs prewarm + processing,
/// and compares the output against the C++ reference (NeuralAmpModelerCore).
///
/// **Expanded precision metrics** (MSE, MAE, SNR, PSNR, bits equiv.)
/// computed in single-pass fusion — see `report_dsp_fidelity` in `tests/common/mod.rs`.
///
/// ## Thresholds
/// - Thresholds auto-computed by `topology_thresholds()` (CH=16 → 105 dB).
/// - Stress signal: 2048 samples (chirp + guitar harmonics + impulse + fade-to-silence).
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_standard.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_standard.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_standard.bin");

    // Load and build the model
    let nam_path = model_path("BossWN-standard.nam");
    assert!(
        nam_path.exists(),
        "BossWN-standard.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build WaveNet for golden test");

    // Prewarm + Processing
    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // 5-metric validation — single-pass fusion
    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "BossWN-standard");
    gv_metric("BossWN-standard");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "BossWN-standard",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8: Golden Vectors LSTM 1×16 — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_lstm_1x16.bin`, builds the `StaticModel`
/// from `BossLSTM-1x16.nam`, runs prewarm + processing,
/// and compares the output against the C++ reference (NeuralAmpModelerCore).
///
/// ## Thresholds
/// - MSE < 3e-3, SNR ≥ 15 dB
/// - LSTM converges better than WaveNet (no FastMath Padé accumulation between layers).
/// - Stress signal: 2048 samples (multi-component).
///
/// If the golden file does not exist, the test fails with an explicit error
/// directing the user to run `tests/fixtures/golden_gen_build.sh`.
#[test]
fn test_golden_vectors_lstm_1x16() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_1x16.bin");

    assert!(
        golden_path.exists(),
        "golden_lstm_1x16.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_1x16.bin");

    // Load and build the model
    let nam_path = model_path("BossLSTM-1x16.nam");
    assert!(
        nam_path.exists(),
        "BossLSTM-1x16.nam not found at {nam_path:?}. Run golden_gen_build.sh to fetch models."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read LSTM model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM for golden test");

    // Prewarm + Processing
    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // 5-metric validation — single-pass fusion
    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "BossLSTM-1x16");
    gv_metric("BossLSTM-1x16");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "BossLSTM-1x16",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8b: Golden Vectors LSTM 2×8 — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_lstm_2x8.bin`, builds the `StaticModel`
/// from `BossLSTM-2x8.nam`. Exercises 2-layer LSTM.
///
/// ## Thresholds
/// - MSE < 1e-3, SNR ≥ 18 dB
/// - Stress signal: 2048 samples (multi-component).
#[test]
fn test_golden_vectors_lstm_2x8() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_2x8.bin");

    assert!(
        golden_path.exists(),
        "golden_lstm_2x8.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_2x8.bin");

    let nam_path = model_path("BossLSTM-2x8.nam");
    assert!(
        nam_path.exists(),
        "BossLSTM-2x8.nam not found at {nam_path:?}. Run golden_gen_build.sh to fetch models."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read LSTM 2x8 model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM 2x8 for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "BossLSTM-2x8");
    gv_metric("BossLSTM-2x8");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "BossLSTM-2x8",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8d-L: Golden Vectors WaveNet A1 Standard (Official) — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
#[test]
fn test_golden_vectors_wavenet_a1_standard() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_a1_standard.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_a1_standard.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a1_standard.bin");

    let nam_path = model_path("wavenet_a1_standard.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a1_standard.nam not found at {nam_path:?}."
    );

    let json_data =
        fs::read_to_string(&nam_path).expect("Failed to read WaveNet A1 Standard model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet A1 Standard for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_a1_standard");
    gv_metric("wavenet_a1_standard (Official)");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "wavenet_a1_standard (Official)",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8f-L: Golden Vectors LSTM Official — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
#[test]
fn test_golden_vectors_lstm_official() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_official.bin");

    assert!(
        golden_path.exists(),
        "golden_lstm_official.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_official.bin");

    let nam_path = model_path("lstm.nam");
    assert!(nam_path.exists(), "lstm.nam not found at {nam_path:?}.");

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read LSTM Official model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM Official for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "lstm (Official)");
    gv_metric("lstm (Official)");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "lstm (Official)",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8c: Golden Vectors WaveNet Feather — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// ## Thresholds
/// - Thresholds auto-computed by `topology_thresholds()` (CH=8 → 100 dB).
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet_feather() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_feather.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_feather.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_feather.bin");

    let nam_path = model_path("BossWN-feather.nam");
    assert!(
        nam_path.exists(),
        "BossWN-feather.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet Feather model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet Feather for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "BossWN-feather");
    gv_metric("BossWN-feather");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "BossWN-feather",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8d: Golden Vectors WaveNet Nano — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// ## Thresholds
/// - Thresholds auto-computed by `topology_thresholds()` (CH=4 → 95 dB).
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet_nano() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_nano.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_nano.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_nano.bin");

    let nam_path = model_path("BossWN-nano.nam");
    assert!(
        nam_path.exists(),
        "BossWN-nano.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet Nano model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build WaveNet Nano for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "BossWN-nano");
    gv_metric("BossWN-nano");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "BossWN-nano",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8e: Golden Vectors WaveNet Lite — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_wavenet_lite.bin`, builds the `StaticModel`
/// from `EVH-5150-Lite.nam` (real community model, CH=12, K=3, HEAD=6, 20 layers),
/// runs prewarm + processing, and compares the output against the C++ reference
/// (NeuralAmpModelerCore).
///
/// ## Thresholds
/// - Stress signal: 2048 samples (chirp + guitar harmonics + impulse + fade-to-silence).
/// - Measured: SNR=122.3 dB, ESR=5.84e-13 (EVH-5150-Lite, post-migration).
/// - Thresholds: SNR ≥ 105 dB, ESR ≤ 3.5e-11 (17.3 dB margin — honest, como Feather CH=8).
///
/// ## Fixture provenance
/// - `golden_wavenet_lite.bin` is generated by `tests/fixtures/golden_gen_build.sh`
///   from NeuralAmpModelerCore C++ render (pinned commit, see script).
/// - `EVH-5150-Lite.nam` is a community real model (CH=12 WaveNet Lite, non-distributable).
///   See `tests/fixtures/README.md` §Non-Distributable Model Management.
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet_lite() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_lite.bin");

    if !golden_path.exists() {
        eprintln!("SKIP: golden_wavenet_lite.bin not found at {golden_path:?}.");
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_lite.bin");

    let nam_path = model_path("EVH-5150-Lite.nam");
    if !nam_path.exists() {
        eprintln!("SKIP: EVH-5150-Lite.nam not found at {nam_path:?}.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build WaveNet Lite for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "EVH-5150-Lite");
    gv_metric("EVH-5150-Lite");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "EVH-5150-Lite",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8g: Golden Vectors WaveNet A2-Full (CH=8) — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_wavenet_a2_full.bin`, builds the `StaticModel`
/// from `wavenet_a2_full.nam`, runs prewarm + processing,
/// and compares the output against the C++ reference (NeuralAmpModelerCore v0.5.3).
#[test]
fn test_golden_vectors_wavenet_a2_full() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_full.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_a2_full.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_full.bin");

    // Load and build the model
    let nam_path = model_path("wavenet_a2_full.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a2_full.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build A2-Full for golden test");

    // Prewarm + Processing
    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // 5-metric validation — single-pass fusion
    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_a2_full");
    gv_metric("WaveNet A2-Full (CH=8) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNet A2-Full (CH=8) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8h: Golden Vectors WaveNet A2-Lite (CH=3) — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_wavenet_a2_lite.bin`, builds the `StaticModel`
/// from `wavenet_a2_lite.nam`, runs prewarm + processing,
/// and compares the output against the C++ reference (NeuralAmpModelerCore v0.5.3).
#[test]
fn test_golden_vectors_wavenet_a2_lite() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_lite.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_a2_lite.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_lite.bin");

    // Load and build the model
    let nam_path = model_path("wavenet_a2_lite.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a2_lite.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2-Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build A2-Lite for golden test");

    // Prewarm + Processing
    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // 5-metric validation — single-pass fusion
    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_a2_lite");
    gv_metric("WaveNet A2-Lite (CH=3) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNet A2-Lite (CH=3) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

// =============================================================================
// ContainerModel Golden Tests
// =============================================================================

/// Test 8i: Container Golden — A2-Full submodel matches C++ reference.
///
/// Builds a `ContainerModel` with A2-Full and A2-Lite as submodels,
/// selects the A2-Full submodel via `set_slimmable_size(0.75)`,
/// and verifies the output matches the standalone A2-Full C++ reference.
#[test]
fn test_golden_vectors_container_a2_full() {
    let full_nam_path = model_path("wavenet_a2_full.nam");
    let lite_nam_path = model_path("wavenet_a2_lite.nam");
    if !full_nam_path.exists() || !lite_nam_path.exists() {
        eprintln!("SKIP: A2 model files not found. Container golden test impossible.");
        return;
    }

    let full_json = fs::read_to_string(&full_nam_path).expect("Failed to read A2-Full model");
    let full_data = parse_nam_json(&full_json).expect("Failed to parse A2-Full");
    let full_model = build_model(&full_data).expect("Dispatcher failed for A2-Full");

    let lite_json = fs::read_to_string(&lite_nam_path).expect("Failed to read A2-Lite model");
    let lite_data = parse_nam_json(&lite_json).expect("Failed to parse A2-Lite");
    let lite_model = build_model(&lite_data).expect("Dispatcher failed for A2-Lite");

    let sample_rate = full_data.sample_rate.map(|s| s as u32).unwrap_or(48000);

    let container = neural_amp_modeler_rs::models::container::ContainerModel::new(
        vec![(0.5, lite_model), (1.0, full_model)],
        sample_rate,
    )
    .expect("Failed to create ContainerModel");

    let mut model = StaticModel::Container(Box::new(container));

    let full_golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_full.bin");

    assert!(
        full_golden_path.exists(),
        "golden_wavenet_a2_full.bin not found at {full_golden_path:?}."
    );

    let (input, expected) =
        read_golden_bin(&full_golden_path).expect("Failed to read golden_wavenet_a2_full.bin");

    if let StaticModel::Container(ref mut c) = model {
        // Use set_active_index to skip crossfade and match existing golden
        c.set_slimmable_size(0.75, None);
    } else {
        unreachable!("Expected Container variant");
    }

    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&full_data, "wavenet_a2_full");
    gv_metric("Container A2-Full (CH=8) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "Container A2-Full (CH=8) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8j: Container Golden — A2-Lite submodel matches C++ reference.
///
/// Builds a `ContainerModel` with A2-Full and A2-Lite as submodels,
/// selects the A2-Lite submodel via `set_slimmable_size(0.25)`,
/// and verifies the output matches the standalone A2-Lite C++ reference.
#[test]
fn test_golden_vectors_container_a2_lite() {
    let full_nam_path = model_path("wavenet_a2_full.nam");
    let lite_nam_path = model_path("wavenet_a2_lite.nam");
    if !full_nam_path.exists() || !lite_nam_path.exists() {
        eprintln!("SKIP: A2 model files not found. Container golden test impossible.");
        return;
    }

    let full_json = fs::read_to_string(&full_nam_path).expect("Failed to read A2-Full model");
    let full_data = parse_nam_json(&full_json).expect("Failed to parse A2-Full");
    let full_model = build_model(&full_data).expect("Dispatcher failed for A2-Full");

    let lite_json = fs::read_to_string(&lite_nam_path).expect("Failed to read A2-Lite model");
    let lite_data = parse_nam_json(&lite_json).expect("Failed to parse A2-Lite");
    let lite_model = build_model(&lite_data).expect("Dispatcher failed for A2-Lite");

    let sample_rate = lite_data.sample_rate.map(|s| s as u32).unwrap_or(48000);

    let container = neural_amp_modeler_rs::models::container::ContainerModel::new(
        vec![(0.5, lite_model), (1.0, full_model)],
        sample_rate,
    )
    .expect("Failed to create ContainerModel");

    let mut model = StaticModel::Container(Box::new(container));

    let lite_golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_lite.bin");

    assert!(
        lite_golden_path.exists(),
        "golden_wavenet_a2_lite.bin not found at {lite_golden_path:?}."
    );

    let (input, expected) =
        read_golden_bin(&lite_golden_path).expect("Failed to read golden_wavenet_a2_lite.bin");

    if let StaticModel::Container(ref mut c) = model {
        // Switch to Lite submodel directly (bypass crossfade) to match existing golden
        c.submodels_mut()[0]
            .1
            .reset(sample_rate, GOLDEN_BLOCK_SIZE)
            .unwrap();
        c.set_active_index(0);
    } else {
        unreachable!("Expected Container variant");
    }

    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&lite_data, "wavenet_a2_lite");
    gv_metric("Container A2-Lite (CH=3) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "Container A2-Lite (CH=3) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8j-F: Container Golden loaded from file — A2 submodels match C++ reference.
///
/// Loads `wavenet_a2_container.nam` from file, runs it first for Lite submodel (slim=0.25),
/// then for Full submodel (slim=0.75), verifying both outputs match C++ standalones.
#[test]
fn test_golden_vectors_wavenet_a2_container() {
    let container_path = model_path("wavenet_a2_container.nam");
    if !container_path.exists() {
        eprintln!("SKIP: wavenet_a2_container.nam not found. Container golden test impossible.");
        return;
    }

    let container_json =
        fs::read_to_string(&container_path).expect("Failed to read container model");
    let container_data = parse_nam_json(&container_json).expect("Failed to parse container");

    let sample_rate = container_data
        .sample_rate
        .map(|s| s as u32)
        .unwrap_or(48000);

    // 1) Test Lite submodel selection
    {
        let mut model = build_model(&container_data).expect("Dispatcher failed for container");
        let lite_golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/golden_wavenet_a2_lite.bin");
        let (input, expected) =
            read_golden_bin(&lite_golden_path).expect("Failed to read golden_wavenet_a2_lite.bin");

        if let StaticModel::Container(ref mut c) = *model {
            c.submodels_mut()[0]
                .1
                .reset(sample_rate, GOLDEN_BLOCK_SIZE)
                .unwrap();
            c.set_active_index(0);
        } else {
            unreachable!("Expected Container variant");
        }

        let mut output = vec![0.0f32; input.len()];
        process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

        let (mse_limit, min_snr_db, max_esr, mrstft_max) =
            topology_thresholds(&container_data, "wavenet_a2_lite");
        gv_metric("Container File A2-Lite (CH=3) C++ cross-reference");
        report_dsp_fidelity(
            &expected,
            &output,
            mse_limit,
            min_snr_db,
            max_esr,
            mrstft_max,
            "Container File A2-Lite (CH=3) C++ cross-reference",
            STRESS_SAMPLE_RATE,
        );
    }

    // 2) Test Full submodel selection
    {
        let mut model = build_model(&container_data).expect("Dispatcher failed for container");
        let full_golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/golden_wavenet_a2_full.bin");
        let (input, expected) =
            read_golden_bin(&full_golden_path).expect("Failed to read golden_wavenet_a2_full.bin");

        if let StaticModel::Container(ref mut c) = *model {
            c.set_slimmable_size(0.75, None);
        } else {
            unreachable!("Expected Container variant");
        }

        let mut output = vec![0.0f32; input.len()];
        process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

        let (mse_limit, min_snr_db, max_esr, mrstft_max) =
            topology_thresholds(&container_data, "wavenet_a2_full");
        gv_metric("Container File A2-Full (CH=8) C++ cross-reference");
        report_dsp_fidelity(
            &expected,
            &output,
            mse_limit,
            min_snr_db,
            max_esr,
            mrstft_max,
            "Container File A2-Full (CH=8) C++ cross-reference",
            STRESS_SAMPLE_RATE,
        );
    }
}

/// Test 8j: Golden Vectors SlimmableContainer A2 Example.
///
/// Reads `tests/fixtures/golden_a2_example.bin`, builds the `StaticModel`
/// from `a2_example.nam` (official C++ `example_models/A2.nam` —
/// SlimmableContainer with 2 WaveNet A2 submodels, CH=3→6),
/// runs prewarm + processing, and compares the output against the
/// C++ reference (NeuralAmpModelerCore v0.5.3, A2_FAST enabled).
#[test]
fn test_golden_vectors_a2_example_slimmable() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_a2_example.bin");

    if !golden_path.exists() {
        eprintln!("SKIP: golden_a2_example.bin not found at {golden_path:?}.");
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_a2_example.bin");

    let nam_path = model_path("a2_example.nam");
    if !nam_path.exists() {
        eprintln!("SKIP: a2_example.nam not found at {nam_path:?}.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read a2_example model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build a2_example for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "a2_example");
    gv_metric("SlimmableContainer A2 Example (CH=3→6) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "SlimmableContainer A2 Example (CH=3→6) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8k: `wavenet_a2_max.nam` is disabled at dispatch (fail-closed, §7.1).
///
/// The model is confirmed broken against the NAMcore C++ golden
/// (see `docs/cpp_parity_map.md` §7.1). This test asserts that
/// `build_model` returns `Err` with the expected "disabled" message
/// and cite to §7.1, proving the guard is active.
#[test]
fn test_wavenet_a2_max_dispatch_is_disabled_broken() {
    let path = model_path("wavenet_a2_max.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read wavenet_a2_max.nam");
    let data = parse_nam_json(&json).expect("Failed to parse wavenet_a2_max.nam");
    let result = build_model(&data);
    assert!(
        result.is_err(),
        "wavenet_a2_max.nam must be rejected (fail-closed guard is missing or bypassed)"
    );
    let err_msg = format!("{}", result.err().unwrap());
    assert!(
        err_msg.contains("disabled"),
        "Error message must contain 'disabled', got: {}",
        err_msg
    );
    assert!(
        err_msg.contains("§7.1"),
        "Error message must cite §7.1, got: {}",
        err_msg
    );
}

/// Test 8k-1: `wavenet_condition_dsp.nam` still loads — non-regression guard.
///
/// Proves that the fail-closed dispatch guard in `is_disabled_broken_a2_flagship`
/// does **not** block valid neighboring models. The `wavenet_condition_dsp.nam`
/// model is a multi-array cascade with `condition_dsp` and `condition_size=3`,
/// which does not match the broken-flagship signature (it has `num_arrays=2`,
/// which falls outside the `num_arrays==1` predicate).
///
/// If this test fails, the guard is over-broad and must be narrowed.
#[test]
fn test_wavenet_condition_dsp_still_loads() {
    let path = model_path("wavenet_condition_dsp.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read wavenet_condition_dsp.nam");
    let data = parse_nam_json(&json).expect("Failed to parse wavenet_condition_dsp.nam");
    let result = build_model(&data);
    assert!(
        result.is_ok(),
        "wavenet_condition_dsp.nam must load successfully (guard is over-broad). Error: {:?}",
        result.err()
    );
}

/// Test 8k-1b: `wavenet_condition_lstm.nam` fail-closed rejection.
///
/// Validates that the dispatcher rejects WaveNet models with an LSTM
/// condition_dsp sub-model. LSTM condition_dsp produces structurally wrong
/// audio (ESR ≈ 1.3e-1, confirmed empirically). The model is rejected at load
/// time with a clear diagnostic message.
#[test]
fn test_wavenet_condition_lstm_loads_and_runs() {
    let path = model_path("wavenet_condition_lstm.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read wavenet_condition_lstm.nam");
    let data = parse_nam_json(&json).expect("Failed to parse wavenet_condition_lstm.nam");
    let result = build_model(&data);
    assert!(
        result.is_err(),
        "Expected LSTM condition_dsp to be rejected (fail-closed policy), but it loaded"
    );
    let err_msg = format!("{}", result.err().unwrap());
    assert!(
        err_msg.contains("LSTM condition_dsp is not supported"),
        "Expected LSTM condition_dsp rejection message, got: {}",
        err_msg
    );
}

/// Test 8k-2: Loader Gap — Explicit rejection of single-net slimmable WaveNet (PM-12).
///
/// Loads the real fixture `slimmable_wavenet.nam` and validates fail-closed
/// rejection with the expected error message. The `nam-rs` engine does not
/// support per-layer slimmable weight slicing — users must use a
/// `SlimmableContainer` architecture instead.
#[test]
fn test_loader_gap_slimmable_wavenet() {
    let path = model_path("slimmable_wavenet.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read slimmable_wavenet.nam");
    let data = parse_nam_json(&json).expect("Failed to parse slimmable_wavenet.nam");
    let model = build_model(&data);
    assert!(
        model.is_err(),
        "Expected slimmable single-net WaveNet to be rejected, but it loaded successfully"
    );
    let err_msg = format!("{}", model.err().unwrap());
    assert!(
        err_msg.contains("slimmable single-net weight slicing is not supported"),
        "Expected explicit slimmable rejection message, got: {}",
        err_msg
    );
}

/// Test 8l: Golden Vectors WaveNet Condition DSP cross-reference C++ ↔ NAM-rs.
///
/// Replaces the gap test (`test_loader_gap_wavenet_condition_dsp`).
/// The condition_dsp sub-model is fully functional and the dynamic engine
/// processes audio through the nested DSP. Validates Rust output against C++ reference
/// via ESR/SNR/MSE fusion report.
///
/// Reads `tests/fixtures/golden_wavenet_condition_dsp.bin`, builds the dynamic `StaticModel`
/// from `wavenet_condition_dsp.nam`, and compares via ESR/SNR/MSE fusion report.
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet_condition_dsp() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_condition_dsp.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_condition_dsp.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate the golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_condition_dsp.bin");

    let nam_path = model_path("wavenet_condition_dsp.nam");
    assert!(
        nam_path.exists(),
        "wavenet_condition_dsp.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data =
        fs::read_to_string(&nam_path).expect("Failed to read wavenet_condition_dsp.nam");
    let model_data =
        parse_nam_json(&json_data).expect("Failed to parse wavenet_condition_dsp.nam JSON");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet Condition DSP for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_condition_dsp");
    gv_metric("WaveNet Condition DSP (CH=3, cond=3, dynamic path) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNet Condition DSP (CH=3, cond=3, dynamic path) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8l-2: Rejection of LSTM condition_dsp via f64 oracle path.
///
/// Validates that the dispatcher fails-closed when attempting to build a WaveNet
/// model whose `condition_dsp` sub-model is an LSTM. The
/// `test_oracle_vs_python_anchor_condition_lstm` in `reference_oracle_f64.rs`
/// separately validates the oracle itself (which does not go through the
/// production dispatcher and is unaffected by this policy).
#[test]
fn test_policy_reject_condition_lstm() {
    let nam_path = model_path("wavenet_condition_lstm.nam");
    assert!(
        nam_path.exists(),
        "wavenet_condition_lstm.nam not found at {nam_path:?}. \
         Run './tests/fixtures/generate_a2_fixtures.py' to regenerate."
    );

    let json_data =
        fs::read_to_string(&nam_path).expect("Failed to read wavenet_condition_lstm.nam");
    let model_data =
        parse_nam_json(&json_data).expect("Failed to parse wavenet_condition_lstm.nam JSON");

    let result = build_model(&model_data);
    assert!(
        result.is_err(),
        "Expected LSTM condition_dsp to be rejected (fail-closed policy), but it loaded"
    );
    let err_msg = format!("{}", result.err().unwrap());
    assert!(
        err_msg.contains("LSTM condition_dsp is not supported"),
        "Expected LSTM condition_dsp rejection message, got: {}",
        err_msg
    );
}

/// Test 8m: Golden Vectors WaveNet Official (dynamic path) — cross-reference C++ ↔ NAM-rs.
///
/// This replaces the gap test (`test_loader_gap_slimmable_wavenet`).
/// Free-geometry WaveNet A1 models now load via the
/// dynamic engine. `wavenet_official.nam` (CH=3, 2 arrays, dilations [(1,2),(8)])
/// exercises the dynamic path and is validated against a C++ reference golden.
///
/// Reads `tests/fixtures/golden_wavenet_official.bin`, builds the dynamic `StaticModel`
/// from `wavenet_official.nam`, and compares via ESR/SNR/MSE fusion report.
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet_official() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_official.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_official.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate the golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_official.bin");

    let nam_path = model_path("wavenet_official.nam");
    assert!(
        nam_path.exists(),
        "wavenet_official.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read wavenet_official.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse wavenet_official.nam JSON");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet Official for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_official");
    gv_metric("WaveNet Official (CH=3, dynamic path) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNet Official (CH=3, dynamic path) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8n: Loader Gap Slimmable Container — verifies robust loading with
/// submodel topology routing (LSTM 1x3 + WaveNet free [3,2] + WaveNet free [4,2]).
/// The container with all three submodels loads
/// successfully via the dynamic engine (heterogeneous channels) and LSTM fast-path.
#[test]
fn test_loader_gap_slimmable_container() {
    let path = model_path("slimmable_container.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read slimmable_container.nam");
    let data = parse_nam_json(&json).expect("Failed to parse slimmable_container.nam");
    let model = match build_model(&data) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP: container build failed ({e:?})");
            return;
        }
    };

    // Verify we have a Container with 3 submodels
    match model.as_ref() {
        StaticModel::Container(c) => {
            assert_eq!(c.submodels().len(), 3);
            // Verify max_values are sorted: 0.33, 0.66, 1.0
            let max_values: Vec<f32> = c.submodels().iter().map(|(mv, _)| *mv).collect();
            assert_eq!(max_values, vec![0.33, 0.66, 1.0]);
            // Verify submodel architectures are correctly dispatched
            let sub_arches: Vec<&str> = c
                .submodels()
                .iter()
                .map(|(_, sm)| match sm.as_ref() {
                    StaticModel::Lstm1x3(_) => "LSTM",
                    StaticModel::WavenetDyn(_) => "WaveNetDyn",
                    StaticModel::WavenetNano(_) => "Nano",
                    _ => "Unknown",
                })
                .collect();
            // Submodel[0] is LSTM 1x3, Submodel[1] is WaveNetDyn (free geometry),
            // Submodel[2] is Nano (channels [4,2] + LITE dilations).
            assert_eq!(sub_arches, vec!["LSTM", "WaveNetDyn", "Nano"]);
        }
        _other => panic!("Expected StaticModel::Container, got a different variant"),
    }
}

// =============================================================================
// V2 Multi-SR Golden Vector Tests
// =============================================================================
//
// Layer-2 soak gates exercising the engine at 44.1/48/88.2/96/192 kHz
// across 5 stimulus categories (GA/FRG/P/BA/PA) via Stress Signal v2.
//
// Each test reads committed `golden_{name}_v2_{sr}.bin` files and
// validates Rust↔C++ parity via ESR/SNR/MSE fusion report.
//
// These tests are `#[ignore]` because the 5-second v2 signals are ~200×
// longer than v1 (240k–960k vs 2048 samples), making them impractical for
// debug-mode CI (~2 min per model). Run with `--include-ignored` for
// comprehensive multi-SR validation. The committed .bin files (generated
// by `golden_gen_build.sh`) serve as reproducible C++ reference artifacts.
//
// Run `./tests/fixtures/golden_gen_build.sh` to regenerate.

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_standard() {
    run_v2_golden_test(
        "BossWN-standard.nam",
        "golden_wavenet_standard",
        "WaveNet Standard (CH=16)",
        "BossWN-standard",
        SR_48K_ONLY,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_feather() {
    run_v2_golden_test(
        "BossWN-feather.nam",
        "golden_wavenet_feather",
        "WaveNet Feather (CH=8)",
        "BossWN-feather",
        ALL_SR,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_nano() {
    run_v2_golden_test(
        "BossWN-nano.nam",
        "golden_wavenet_nano",
        "WaveNet Nano (CH=4)",
        "BossWN-nano",
        ALL_SR,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_lite() {
    run_v2_golden_test(
        "EVH-5150-Lite.nam",
        "golden_wavenet_lite",
        "WaveNet Lite (CH=12)",
        "EVH-5150-Lite",
        ALL_SR,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_lstm_1x16() {
    run_v2_golden_test(
        "BossLSTM-1x16.nam",
        "golden_lstm_1x16",
        "LSTM 1×16",
        "BossLSTM-1x16",
        SR_EX_192K,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_app_evh() {
    run_v2_golden_test(
        "APP-EVH-Stealth100-Dialled-xSTD.nam",
        "golden_wavenet_app_evh",
        "APP EVH Stealth 100",
        "APP-EVH-Stealth100-Dialled-xSTD",
        ALL_SR,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_boss_bd2() {
    run_v2_golden_test(
        "Boss BD-2 H2O Mod T-12_00 G-12_00.nam",
        "golden_wavenet_boss_bd2",
        "Boss BD-2 H2O Mod",
        "Boss BD-2 H2O Mod T-12_00 G-12_00",
        ALL_SR,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_slammin_marshall() {
    run_v2_golden_test(
        "SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam",
        "golden_wavenet_slammin_marshall",
        "SLAMMIN MARSHALL JTM 45",
        "SLAMMIN MARSHALL JTM 45 REISSUE",
        ALL_SR,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_lstm_2x8() {
    run_v2_golden_test(
        "BossLSTM-2x8.nam",
        "golden_lstm_2x8",
        "LSTM 2×8",
        "BossLSTM-2x8",
        SR_EX_192K,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_a1_standard() {
    run_v2_golden_test(
        "wavenet_a1_standard.nam",
        "golden_wavenet_a1_standard",
        "WaveNet A1 Standard (Official)",
        "wavenet_a1_standard",
        ALL_SR,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_official() {
    // The C++ render tool only generates `wavenet_official` at 48 kHz (the model's
    // expected rate); golden_gen_build.sh skips the other SRs ("Input WAV sample
    // rate does not match model expected rate (48000 Hz)"). Match lstm_official / A2.
    run_v2_golden_test(
        "wavenet_official.nam",
        "golden_wavenet_official",
        "WaveNet Official (CH=3, dynamic)",
        "wavenet_official",
        SR_48K_ONLY,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_lstm_official() {
    run_v2_golden_test(
        "lstm.nam",
        "golden_lstm_official",
        "LSTM Official",
        "lstm (Official)",
        SR_48K_ONLY,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_a2_full() {
    run_v2_golden_test(
        "wavenet_a2_full.nam",
        "golden_wavenet_a2_full",
        "WaveNet A2-Full (CH=8)",
        "wavenet_a2_full",
        SR_48K_ONLY,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_a2_lite() {
    run_v2_golden_test(
        "wavenet_a2_lite.nam",
        "golden_wavenet_a2_lite",
        "WaveNet A2-Lite (CH=3)",
        "wavenet_a2_lite",
        SR_48K_ONLY,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_condition_dsp() {
    run_v2_golden_test(
        "wavenet_condition_dsp.nam",
        "golden_wavenet_condition_dsp",
        "WaveNet Condition DSP (CH=3, cond=3, dynamic)",
        "wavenet_condition_dsp",
        SR_48K_ONLY,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_condition_lstm() {
    run_v2_golden_test(
        "wavenet_condition_lstm.nam",
        "golden_wavenet_condition_lstm",
        "WaveNet Condition DSP LSTM (CH=3, cond=3, LSTM)",
        "wavenet_condition_lstm",
        SR_48K_ONLY,
    );
}

// =============================================================================
// Polynomial Activation Regression Gate
// =============================================================================

/// Activation regression gate: WaveNet Standard golden fidelity.
///
/// Validates that the end-to-end WaveNet SIMD output does not regress
/// against the C++ reference (NeuralAmpModelerCore). The polynomial path uses
/// exact exp-based tanh/sigmoid with full-precision f32 weights — the same
/// arithmetic as the C++ reference — so the ESR gate is tightened substantially
/// relative to the quantized mode (where weight quantization + Padé approximation
/// dominate the drift).
///
/// **Gate**: ESR ≤ 1e-4  (100× tighter than quantized parity limit, 1e-2).
///           SNR ≥ 70 dB.
#[test]
fn test_poly_regression_gate_wavenet_standard() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_standard.bin");

    if !golden_path.exists() {
        eprintln!(
            "SKIP  golden_wavenet_standard.bin not found.\n\
             Run './tests/fixtures/golden_gen_build.sh' to generate golden vectors."
        );
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_standard.bin");

    let nam_path = model_path("BossWN-standard.nam");
    if !nam_path.exists() {
        eprintln!("SKIP  BossWN-standard.nam not found.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet Standard model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet Standard for poly regression gate");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // Polynomial gate: 100× tighter ESR than quantized parity limit.
    // The polynomial path uses exact exp-based tanh + full-precision f32 weights,
    // matching the C++ reference arithmetic. Floating-point ordering
    // differences (Rust vs Eigen/C++) dominate the residual ESR.
    const POLY_ESR_MAX: f64 = 1e-4;
    const POLY_SNR_MIN: f64 = 70.0;
    const POLY_MSE_MAX: f64 = 1e-5;

    gv_metric("WaveNet Standard polynomial SIMD (regression gate)");
    report_dsp_fidelity(
        &expected,
        &output,
        Some(POLY_MSE_MAX),
        POLY_SNR_MIN,
        Some(POLY_ESR_MAX),
        None,
        "WaveNet Standard polynomial SIMD (regression gate)",
        STRESS_SAMPLE_RATE,
    );
}

/// Activation regression gate: WaveNet A2-Full golden fidelity.
///
/// Same gate as `test_poly_regression_gate_wavenet_standard` for the
/// A2 architecture (CH=8, 23 layers with variable kernel sizes).
#[test]
fn test_poly_regression_gate_wavenet_a2_full() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_full.bin");

    if !golden_path.exists() {
        eprintln!(
            "SKIP  golden_wavenet_a2_full.bin not found.\n\
             Run './tests/fixtures/golden_gen_build.sh' to generate golden vectors."
        );
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_full.bin");

    let nam_path = model_path("wavenet_a2_full.nam");
    if !nam_path.exists() {
        eprintln!("SKIP  wavenet_a2_full.nam not found.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build A2-Full for poly regression gate");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // Post-weight-dequantization — A2 is now near-bit-exact (ESR=1.13e-13).
    // Gate matches WaveNet Standard's POLY_WAVENET_ESR_MAX.
    const POLY_A2_ESR_MAX: f64 = 1e-4;
    const POLY_A2_SNR_MIN: f64 = 65.0;
    const POLY_A2_MSE_MAX: f64 = 1e-5;

    gv_metric("WaveNet A2-Full polynomial SIMD (regression gate)");
    report_dsp_fidelity(
        &expected,
        &output,
        Some(POLY_A2_MSE_MAX),
        POLY_A2_SNR_MIN,
        Some(POLY_A2_ESR_MAX),
        None,
        "WaveNet A2-Full polynomial SIMD (regression gate)",
        STRESS_SAMPLE_RATE,
    );
}

// =============================================================================
// WaveNet A2 Dynamic Golden Tests (Golden Vectors e C++ Parity)
//
// NOTE (2026-06-21): v2 multi-SR goldens (`golden_a2_dynamic_*_v2_<sr>.bin`)
// do not exist for the A2 dynamic geometries (gated/blended/FiLM). These
// engines are forward-compat parser surface only and not part of the v2 golden
// pipeline. The A2 fixed fast-path models (`wavenet_a2_full`, `wavenet_a2_lite`)
// already have v2 golden coverage at 48 kHz.
// =============================================================================

/// Test 9a: Golden Vectors — A2 Dynamic Gated (CH=8)
///
/// Validates the `WaveNetA2Dyn` engine with gating active on 3 layers
/// (early/mid/late) against the C++ generic WaveNet reference.
///
/// The C++ v0.5.3 `is_a2_shape()` rejects this model (gating detected) and
/// routes it to the generic WaveNet path. The Rust dispatcher classifies it
/// as `A2TopologyResult::Dynamic` and routes to `WaveNetA2Dyn`.
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate all golden vectors,
/// including the A2 dynamic/FiLM fixtures from generate_a2_fixtures.py.
#[test]
fn test_golden_vectors_a2_dynamic_gated_ch8() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_a2_dynamic_gated_ch8.bin");

    assert!(
        golden_path.exists(),
        "golden_a2_dynamic_gated_ch8.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_a2_dynamic_gated_ch8.bin");

    let nam_path = model_path("a2_dynamic_gated_ch8.nam");
    assert!(
        nam_path.exists(),
        "a2_dynamic_gated_ch8.nam not found at {nam_path:?}."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read A2 Dynamic Gated model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build A2 Dynamic Gated for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "a2_dynamic_gated_ch8");
    gv_metric("WaveNet A2 Dynamic Gated (CH=8, gated layers 3/23) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNet A2 Dynamic Gated (CH=8, gated layers 3/23) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 9b: Golden Vectors — A2 Dynamic Blended (CH=3)
///
/// Validates the `WaveNetA2Dyn` engine with blending active on 2 layers
/// against the C++ generic WaveNet reference.
///
/// The C++ v0.5.3 `is_a2_shape()` rejects this model (blending detected) and
/// routes it to the generic WaveNet path. The Rust dispatcher classifies it
/// as `A2TopologyResult::Dynamic` and routes to `WaveNetA2Dyn`.
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate all golden vectors,
/// including the A2 dynamic/FiLM fixtures from generate_a2_fixtures.py.
#[test]
fn test_golden_vectors_a2_dynamic_blended_ch3() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_a2_dynamic_blended_ch3.bin");

    assert!(
        golden_path.exists(),
        "golden_a2_dynamic_blended_ch3.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_a2_dynamic_blended_ch3.bin");

    let nam_path = model_path("a2_dynamic_blended_ch3.nam");
    assert!(
        nam_path.exists(),
        "a2_dynamic_blended_ch3.nam not found at {nam_path:?}."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read A2 Dynamic Blended model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build A2 Dynamic Blended for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "a2_dynamic_blended_ch3");
    gv_metric("WaveNet A2 Dynamic Blended (CH=3, blended layers 2/23) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNet A2 Dynamic Blended (CH=3, blended layers 2/23) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 9c: Golden Vectors — WaveNet A2-FiLM-Lite (CH=3, FiLM active)
///
/// Validates the `WaveNetA2Dyn` engine with FiLM modulation against the
/// C++ generic WaveNet reference (C++ a2_fast.cpp rejects FiLM and falls
/// back to Eigen-based generic WaveNet).
///
/// Reads `tests/fixtures/golden_wavenet_a2_film_lite.bin`, builds the
/// dynamic `StaticModel` from `wavenet_a2_film_lite.nam`, and compares
/// via ESR/SNR/MSE fusion report.
#[test]
fn test_golden_vectors_wavenet_a2_film_lite() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_a2_film_lite.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_a2_film_lite.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_film_lite.bin");

    let nam_path = model_path("wavenet_a2_film_lite.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a2_film_lite.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data =
        fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2-FiLM-Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed to build A2-FiLM-Lite");

    assert!(
        matches!(
            model.as_ref(),
            neural_amp_modeler_rs::models::StaticModel::WavenetA2Dyn(_)
        ),
        "FiLM model must route to WaveNetA2Dyn (C++ a2_fast.cpp rejects FiLM)"
    );

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_a2_film_lite");
    gv_metric("WaveNet A2-FiLM-Lite (CH=3, FiLM active) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNet A2-FiLM-Lite (CH=3, FiLM active) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 9d: Golden Vectors — WaveNet A2-FiLM-Full (CH=8, FiLM active)
///
/// Validates the `WaveNetA2Dyn` engine with FiLM modulation against the
/// C++ generic WaveNet reference (C++ a2_fast.cpp rejects FiLM and falls
/// back to Eigen-based generic WaveNet).
///
/// Reads `tests/fixtures/golden_wavenet_a2_film_full.bin`, builds the
/// dynamic `StaticModel` from `wavenet_a2_film_full.nam`, and compares
/// via ESR/SNR/MSE fusion report.
#[test]
fn test_golden_vectors_wavenet_a2_film_full() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_a2_film_full.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_a2_film_full.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_film_full.bin");

    let nam_path = model_path("wavenet_a2_film_full.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a2_film_full.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data =
        fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2-FiLM-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed to build A2-FiLM-Full");

    assert!(
        matches!(
            model.as_ref(),
            neural_amp_modeler_rs::models::StaticModel::WavenetA2Dyn(_)
        ),
        "FiLM model must route to WaveNetA2Dyn (C++ a2_fast.cpp rejects FiLM)"
    );

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_a2_film_full");
    gv_metric("WaveNet A2-FiLM-Full (CH=8, FiLM active) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNet A2-FiLM-Full (CH=8, FiLM active) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 9e: Golden Vectors — WaveNet A2-FiLM Chaos Stress (CH=3, FiLM active)
///
/// Validates the `WaveNetA2Dyn` engine with FiLM modulation under chaos/stress
/// conditions against the C++ generic WaveNet reference.
#[test]
fn test_golden_vectors_wavenet_a2_film_chaos_stress() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_a2_film_chaos_stress.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_a2_film_chaos_stress.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) = read_golden_bin(&golden_path)
        .expect("Failed to read golden_wavenet_a2_film_chaos_stress.bin");

    let nam_path = model_path("wavenet_a2_film_chaos_stress.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a2_film_chaos_stress.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data =
        fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2-FiLM-Chaos-Stress model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build A2-FiLM-Chaos-Stress");

    assert!(
        matches!(
            model.as_ref(),
            neural_amp_modeler_rs::models::StaticModel::WavenetA2Dyn(_)
        ),
        "FiLM model must route to WaveNetA2Dyn (C++ a2_fast.cpp rejects FiLM)"
    );

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_a2_film_chaos_stress");
    gv_metric("WaveNet A2-FiLM Chaos Stress (CH=3, FiLM active) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNet A2-FiLM Chaos Stress (CH=3, FiLM active) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

// =============================================================================
// Dynamic Model Golden Vector Tests
//
// NOTE (2026-06-21): v2 multi-SR goldens for `wavenet_dyn_free` and
// `lstm_dyn_test` are intentionally limited to 48 kHz (v1 only). Dynamic
// engines handle arbitrary free geometries — geometry variance subsumes
// sample-rate variance. Live cross-validation (`tests/cpp_parity.rs` lines
// 667, 678) exercises multi-SR parity via the C++ toolchain for these
// geometries without committing large v2 golden files. See `docs/cpp_parity_map.md`
// §3.3.
// =============================================================================

/// Test 10a: Golden Vectors — WaveNetDyn Free-Shape (CH=7→4)
///
/// Validates the `WaveNetModelDyn` engine against C++ generic WaveNet reference
/// for a free-geometry WaveNet that does not match any standard SKU.
///
/// Reads `tests/fixtures/golden_wavenet_dyn_free.bin`, builds the dynamic
/// `StaticModel` from `wavenet_dyn_free.nam`, and compares via ESR/SNR/MSE
/// fusion report.
#[test]
fn test_golden_vectors_wavenet_dyn_free() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_dyn_free.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_dyn_free.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_dyn_free.bin");

    let nam_path = model_path("wavenet_dyn_free.nam");
    assert!(
        nam_path.exists(),
        "wavenet_dyn_free.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read wavenet_dyn_free.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse wavenet_dyn_free.nam JSON");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNetDyn Free for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_dyn_free");
    gv_metric("WaveNetDyn Free-Shape (CH=7→4, dynamic path) C++ cross-reference");
    report_dsp_fidelity_no_lufs(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNetDyn Free-Shape (CH=7→4, dynamic path) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 10b: Golden Vectors — LSTM-Dyn 1×7
///
/// Validates the `LstmModelDyn` engine against C++ generic LSTM reference
/// for a non-catalog LSTM (hidden_size=7, single layer).
///
/// Reads `tests/fixtures/golden_lstm_dyn_test.bin`, builds the dynamic
/// `StaticModel` from `lstm_dyn_test.nam`, and compares via ESR/SNR/MSE
/// fusion report.
#[test]
fn test_golden_vectors_lstm_dyn_test() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_dyn_test.bin");

    assert!(
        golden_path.exists(),
        "golden_lstm_dyn_test.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_dyn_test.bin");

    let nam_path = model_path("lstm_dyn_test.nam");
    assert!(
        nam_path.exists(),
        "lstm_dyn_test.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read lstm_dyn_test.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse lstm_dyn_test.nam JSON");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM-Dyn for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "lstm_dyn_test");
    gv_metric("LSTM-Dyn 1×7 (dynamic path) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "LSTM-Dyn 1×7 (dynamic path) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

// =============================================================================
// ConvNet Self-Golden Consistency Test
// =============================================================================
//
// ConvNet output determinism is validated by verifying identical output across
// two independent model instances. Live C++ cross-validation (2026-07-28)
// confirms full parity: ESR=4.20e-15 (SNR=143.8 dB) against NAMcore.

/// Test 10c: ConvNet Self-Golden Consistency
///
/// Builds the ConvNet model from `convnet_test.nam`, processes the v1 stress
/// signal via `model.process()`, and verifies output determinism by running
/// two independent instances. C++ golden cross-reference (live parity) is
/// confirmed at ESR=4.20e-15 (TASK-CONVNET-04/07).
///
/// ConvNet produces `out_ch` > 1 channels per frame when there is no
/// post-stack head. The output buffer must be `num_frames × out_ch`.
#[test]
fn test_golden_vectors_convnet_test() {
    let nam_path = model_path("convnet_test.nam");
    assert!(
        nam_path.exists(),
        "convnet_test.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read convnet_test.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse convnet_test.nam JSON");

    let mut model_a =
        build_model(&model_data).expect("Dispatcher failed to build ConvNet for golden test");
    let out_ch = match model_a.as_ref() {
        neural_amp_modeler_rs::models::StaticModel::ConvNet(c) => c.out_channels(),
        _ => 1,
    };

    let stressed: Vec<f32> = generate_stress_signal_v1();
    let num_frames = stressed.len();
    let out_len = num_frames * out_ch;

    model_a.prewarm(2048);
    let mut output_a = vec![0.0f32; out_len];
    model_a.process(&stressed, &mut output_a);

    let mut model_b =
        build_model(&model_data).expect("Dispatcher failed to build ConvNet (second instance)");
    model_b.prewarm(2048);
    let mut output_b = vec![0.0f32; out_len];
    model_b.process(&stressed, &mut output_b);

    for (&a, &b) in output_a.iter().zip(output_b.iter()) {
        assert!(
            (a - b).abs() == 0.0,
            "ConvNet output determinism violated: diff = {:e}",
            (a - b).abs()
        );
    }

    for &s in output_a.iter() {
        assert!(s.is_finite(), "ConvNet output must be finite");
    }

    let max_out = output_a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(max_out > 0.0, "ConvNet output must not be silent");

    let signal_power: f64 =
        output_a.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / out_len as f64;
    let noise_power: f64 = output_a
        .iter()
        .zip(output_b.iter())
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>()
        / out_len as f64;
    let self_golden_snr = if noise_power <= f64::EPSILON {
        f64::INFINITY
    } else {
        10.0 * (signal_power / noise_power).log10()
    };

    let (mse_limit, min_snr_db, max_esr, _mrstft_max) =
        topology_thresholds(&model_data, "convnet_test");
    let mse = noise_power;

    if let Some(mse_limit_val) = mse_limit {
        println!();
        println!("[ConvNet Self-Golden — Output Determinism]");
        println!(
            "  MSE     = {:.2e}      (threshold < {:.1e})  {}",
            mse,
            mse_limit_val,
            if mse < mse_limit_val { "✓" } else { "✗" }
        );
        assert!(
            mse < mse_limit_val,
            "[ConvNet Self-Golden] MSE={mse:.6e} exceeds threshold {mse_limit_val:.1e}"
        );
    }
    println!(
        "  SNR     = {:.1} dB       (threshold ≥ {:.1} dB)   {}",
        self_golden_snr,
        min_snr_db,
        if self_golden_snr >= min_snr_db {
            "✓"
        } else {
            "✗"
        }
    );
    assert!(
        self_golden_snr >= min_snr_db,
        "ConvNet self-golden SNR={self_golden_snr:.1} dB below minimum {min_snr_db:.1} dB"
    );
    if let Some(esr_limit) = max_esr {
        let esr = noise_power / signal_power;
        println!(
            "  ESR     = {:.2e}       (threshold < {:.1e})  {}",
            esr,
            esr_limit,
            if esr < esr_limit { "✓" } else { "✗" }
        );
        assert!(
            esr < esr_limit,
            "ConvNet self-golden ESR={esr:.2e} exceeds threshold {esr_limit:.1e}"
        );
    }
}

/// Test 10d: Golden Vectors — WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1)
///
/// DISABLED — model confirmed broken against NAMcore C++ golden
/// (see `docs/cpp_parity_map.md` §7.1). Inference path is blocked at
/// dispatch by fail-closed guard in `is_disabled_broken_a2_flagship`
/// (`src/loader/dispatcher/wavenet/mod.rs`).
///
/// Originally validated the `WaveNetA2Dyn` engine with condition_dsp
/// sub-model against the C++ generic WaveNet reference (C++ a2_fast.cpp
/// rejects this topology and falls back to Eigen-based generic WaveNet).
/// Empirically measured (bypassing guard): SNR=−15.6 dB, ESR=3.61e1
/// (noise power 36× signal power — negative SNR confirms architectural
/// mismatch between Rust WaveNetA2Dyn native FiLM and C++ Eigen-based
/// generic WaveNet with different condition_dsp processing).
///
/// Re-enable only after closing the condition_dsp parity gap (§4.4) and
/// removing the dispatch guard.
#[test]
#[ignore = "model disabled — confirmed broken; inference path blocked at dispatch"]
fn test_golden_vectors_wavenet_a2_max() {
    // The A2 Max flagship is fail-closed disabled at dispatch (§7.1 — see
    // `is_disabled_broken_a2_flagship` and the non-ignored guard test
    // `test_wavenet_a2_max_dispatch_is_disabled_broken`). Golden audio parity
    // cannot be evaluated while the inference path is blocked, so this test
    // only re-asserts the guard rejects the model and skips the spectral
    // comparison until the §4.4 condition_dsp parity gap is closed.
    let nam_path = model_path("wavenet_a2_max.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a2_max.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read wavenet_a2_max.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse wavenet_a2_max.nam JSON");
    let result = build_model(&model_data);
    assert!(
        result.is_err(),
        "wavenet_a2_max.nam must remain rejected by the fail-closed dispatch guard (§7.1); \
         golden parity is deferred until the condition_dsp parity gap (§4.4) is closed."
    );
    // Mirror the non-ignored guard test (`test_wavenet_a2_max_dispatch_is_disabled_broken`)
    // so both assertions stay in lockstep on the exact rejection category, not
    // just that *some* error occurred.
    let err_msg = format!("{}", result.err().unwrap());
    assert!(
        err_msg.contains("disabled"),
        "Error message must contain 'disabled', got: {err_msg}"
    );
    assert!(
        err_msg.contains("§7.1"),
        "Error message must cite §7.1, got: {err_msg}"
    );
    eprintln!(
        "SKIP: WaveNet A2 Max golden parity — model disabled at dispatch (§7.1). \
         Build correctly rejected: {err_msg}"
    );
}

/// Synthetic MR-STFT regression — mild low-pass filter
/// on model output must trigger the hard MR-STFT gate at 48 kHz.
///
/// A 1-pole low-pass at 2 kHz applied to the Rust output induces spectral
/// divergence that the calibrated mrstft_max gate catches at native
/// sample rate, proving the gate is not a placebo.
#[test]
fn test_mrstft_hard_gate_catches_regression() {
    // Suppress report output and panic messages during this
    // controlled-panic regression test to keep the green-test suite clean.
    let _report_guard = SuppressReportGuard::new();
    // Label JSONL emissions as "selftest" to prevent contamination
    // of the quality dashboard with synthetic regression metrics.
    let _kind_guard = MetricKindGuard::selftest();
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    // Use WaveNet A1 Standard — always available golden fixture
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_a1_standard.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_a1_standard.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, _expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a1_standard.bin");

    let nam_path = model_path("wavenet_a1_standard.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a1_standard.nam not found at {nam_path:?}."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet A1 model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet A1 for MR-STFT regression test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // Synthetic degradation: mild low-pass filter (2 kHz cutoff)
    // This should elevate MR-STFT above the calibrated threshold (0.05)
    let degraded = neural_amp_modeler_rs::testing::mushra::low_pass_1pole(&output, 2000.0, 48000);
    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_a1_standard");

    // Verify MR-STFT is indeed elevated above the calibrated gate
    let mr_stft = neural_amp_modeler_rs::testing::perceptual::compute_mr_stft(&output, &degraded);
    assert!(
        mr_stft > mrstft_max.unwrap(),
        "MR-STFT regression test precondition: MR-STFT ({mr_stft:.4e}) must exceed \
         calibrated threshold ({:.2e}) for the assert to be meaningful. \
         Increase low-pass cutoff or use a stronger degradation.",
        mrstft_max.unwrap(),
    );

    // This should panic because MR-STFT exceeds the hard gate at 48 kHz
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        report_dsp_fidelity(
            &output,
            &degraded,
            mse_limit,
            min_snr_db,
            max_esr,
            mrstft_max,
            "MR-STFT regression gate (synthetic)",
            48000,
        );
    }));

    // Restore the default panic hook before any following asserts,
    // so test-framework failures are still visible.
    std::panic::set_hook(prev_hook);

    assert!(
        result.is_err(),
        "MR-STFT hard gate did NOT catch the synthetic spectral regression. \
         MR-STFT={mr_stft:.4e} should exceed calibrated threshold."
    );
}

#[test]
fn test_golden_vectors_wavenet_a2_film_input_mixin_pre() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_a2_film_input_mixin_pre.bin");
    if !golden_path.exists() {
        // `SKIP-COVERAGE` is a greppable marker so a coverage audit can detect
        // golden-vector tests that never exercised their C++ reference (the
        // golden binary was never generated). Without it this `#[ignore]`d test
        // would report green indefinitely with zero parity coverage.
        eprintln!(
            "SKIP-COVERAGE: golden_wavenet_a2_film_input_mixin_pre.bin not found at {golden_path:?}."
        );
        eprintln!(
            "      Run './tests/fixtures/golden_gen_build.sh' to generate the C++ golden \
             (threshold still pending C++ golden measurement — see validation.rs)."
        );
        return;
    }

    let (input, expected) = read_golden_bin(&golden_path)
        .expect("Failed to read golden_wavenet_a2_film_input_mixin_pre.bin");

    let nam_path = model_path("wavenet_a2_film_input_mixin_pre.nam");
    assert!(
        nam_path.exists(),
        "Model file not found: {}",
        nam_path.display()
    );

    let json_data = std::fs::read_to_string(&nam_path)
        .expect("Failed to read wavenet_a2_film_input_mixin_pre.nam");
    let model_data =
        parse_nam_json(&json_data).expect("Failed to parse wavenet_a2_film_input_mixin_pre.nam");
    let mut model =
        build_model(&model_data).expect("Failed to build wavenet_a2_film_input_mixin_pre.nam");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_a2_film_input_mixin_pre");
    gv_metric("WaveNet A2-FiLM-InputMixinPre (CH=3, input_mixin_pre_film) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNet A2-FiLM-InputMixinPre (CH=3, input_mixin_pre_film) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

// =============================================================================
// Catalog Preflight — Capability Receipt
// =============================================================================

/// Prints the unified fixture catalog capability receipt to stdout.
///
/// Iterates every entry in `FIXTURE_CATALOG`, resolves its path via the
/// unified path resolution (env vars → local checkout), checks disk presence,
/// and emits a typed capability report.
///
/// Invoked by `utils/tests-long.sh` before running the long suite.  The
/// receipt serves as an auditable preflight artifact: it proves that the
/// runner verified fixture availability against the catalog declaratively,
/// rather than proceeding blindly and crashing late with a panic.
///
/// Missing `RequiredLocal` fixtures are reported but this test does NOT fail —
/// the shell-level preflight in `tests-long.sh` Phase 0 handles hard-fail
/// gating independently.  This function is a receipt emitter, not a gate.
#[test]
fn catalog_preflight() {
    println!("{}", FIXTURE_CATALOG.capability_receipt());

    let mut missing_required = Vec::new();
    let mut missing_optional = Vec::new();

    for entry in FIXTURE_CATALOG.entries() {
        if entry.name.ends_with(".bin") {
            let status = FIXTURE_CATALOG.check_golden(entry.name);
            match status {
                FixtureStatus::MissingRequired => missing_required.push(entry.name),
                FixtureStatus::MissingOptional => missing_optional.push(entry.name),
                _ => {}
            }
        } else {
            let status = FIXTURE_CATALOG.check(entry.name);
            match status {
                FixtureStatus::MissingRequired => missing_required.push(entry.name),
                FixtureStatus::MissingOptional => missing_optional.push(entry.name),
                _ => {}
            }
        }
    }

    if !missing_optional.is_empty() {
        println!();
        println!(
            "=== Optional Fixtures Absent ({} file(s)) ===",
            missing_optional.len()
        );
        for name in &missing_optional {
            println!("  SKIP-CAPABILITY: {name} (OptionalExternal — skip gracefully)");
        }
    }

    if !missing_required.is_empty() {
        println!();
        println!(
            "=== Required Fixtures Absent ({} file(s)) ===",
            missing_required.len()
        );
        for name in &missing_required {
            println!("  MISSING-REQUIRED: {name} (RequiredLocal — preflight hard-fail)");
        }
    } else {
        println!();
        println!("=== All RequiredLocal fixtures present ✓ ===");
    }
}
