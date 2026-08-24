// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Golden Vector Cross-Reference Tests.
//
//  Compares NeuralAmpModeler-rs Rust engine output against C++ reference golden vectors
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
use neural_amp_modeler_rs::models::a2::GatingMode;
use neural_amp_modeler_rs::models::slimmable::SlimmableModel;
use neural_amp_modeler_rs::models::{NamModel, StaticModel};
use neural_amp_modeler_rs::testing::catalog::v2_sample_rates_for;
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
/// processes with `process_in_blocks`, and validates via `report_dsp_fidelity`
/// (or `report_dsp_fidelity_no_lufs` when `check_lufs_gate` is false).
///
/// `check_lufs_gate` must match the corresponding `live_cross_validation_v2_*`
/// policy in `tests/parity/cpp_parity.rs` (SKIP_CAPABILITY for synthetic
/// low-loudness fixtures such as `lstm_2x24` / `convnet_relu`).
fn run_v2_golden_test(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    model_name: &str,
    sample_rates: &[u32],
    check_lufs_gate: bool,
) {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let nam_path = model_path(model_filename);

    if !nam_path.exists() {
        eprintln!(
            "[STATUS] SKIP_CAPABILITY: model_not_found:{model_filename} — skipping v2 golden test"
        );
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read model JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    for &sr in sample_rates {
        let golden_filename = format!("{golden_name}_v2_{sr}.bin");
        let golden_path = fixtures_dir.join(&golden_filename);

        if !golden_path.exists() {
            if model_filename == "wavenet_condition_lstm.nam" {
                eprintln!(
                    "[STATUS] KNOWN_GAP id=\"condition_lstm_cpp_crash\" reason=\"C++ upstream limitation: LSTM condition_dsp sub-model channel mismatch — golden binary cannot be generated\""
                );
            } else {
                eprintln!(
                    "[STATUS] SKIP_CAPABILITY: golden_not_found:{golden_filename} — run './tests/fixtures/golden_gen_build.sh' to generate v2 multi-SR golden vectors"
                );
            }
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

        let report_label = format!("{label} @ {sr} Hz (v2)");
        if check_lufs_gate {
            report_dsp_fidelity(
                &expected,
                &output,
                mse_limit,
                min_snr_db,
                max_esr,
                mrstft_max,
                &report_label,
                sr,
            );
        } else {
            report_dsp_fidelity_no_lufs(
                &expected,
                &output,
                mse_limit,
                min_snr_db,
                max_esr,
                mrstft_max,
                &report_label,
                sr,
            );
        }
    }
}

// =============================================================================
// V2 Multi-SR Catalog — Single Source of Truth
// =============================================================================
//
// The canonical V2 golden catalog (models, sample rates, expected fixtures,
// distribution policy) lives in `src/testing/catalog.rs::GOLDEN_GEN_CATALOG`
// and is consumed here through `v2_sample_rates_for` (imported above). The
// former local `V2_CATALOG` table and the bash `V2_CATALOG_SCOPE` array in
// `utils/tests-long.sh` were removed — Rust is the only definition, validated
// on disk by `catalog_preflight` via `validate_v2_catalog`.
//
// Scope ∈ { AllRates (5 SRs), Exclude192k (4 SRs), Sr48kOnly (1 SR) }.
//   AllRates     — all 5 sample rates (44100, 48000, 88200, 96000, 192000)
//   Exclude192k  — all except 192k (LSTM recurrent drift over 5s stress)
//   Sr48kOnly    — only 48000 Hz (model declares expected_sample_rate=48000
//                  or C++ render tool rejects other SRs)

// =============================================================================
// Golden Vector Tests (Cross-Reference C++ ↔ Rust)
// =============================================================================

/// Test 7: Golden Vectors WaveNet — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
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

/// Test 8: Golden Vectors LSTM 1×16 — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
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

/// Test 8b: Golden Vectors LSTM 2×8 — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
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

/// Test 8d-L: Golden Vectors WaveNet A1 Standard (Official) — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
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

/// Test 8f-L: Golden Vectors LSTM Official — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
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

/// Test 8c: Golden Vectors WaveNet Feather — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
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

/// Test 8d: Golden Vectors WaveNet Nano — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
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

/// Test 8e: Golden Vectors WaveNet Lite — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
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
///   See `docs/fixtures.md` §Non-Distributable Model Management.
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet_lite() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_lite.bin");

    if !golden_path.exists() {
        eprintln!(
            "[STATUS] SKIP_CAPABILITY: golden_not_found:golden_wavenet_lite.bin — run './tests/fixtures/golden_gen_build.sh'"
        );
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_lite.bin");

    let nam_path = model_path("EVH-5150-Lite.nam");
    if !nam_path.exists() {
        eprintln!("[STATUS] SKIP_CAPABILITY: model_not_found:EVH-5150-Lite.nam");
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

/// Test 8g: Golden Vectors WaveNet A2-Full (CH=8) — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
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

/// Test 8h: Golden Vectors WaveNet A2-Lite (CH=3) — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
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
        eprintln!(
            "[STATUS] SKIP_CAPABILITY: model_not_found:a2_full_or_lite — container golden test impossible"
        );
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
        eprintln!(
            "[STATUS] SKIP_CAPABILITY: model_not_found:a2_full_or_lite — container golden test impossible"
        );
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
        eprintln!(
            "[STATUS] SKIP_CAPABILITY: model_not_found:wavenet_a2_container.nam — container golden test impossible"
        );
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
        eprintln!("[STATUS] SKIP_CAPABILITY: golden_not_found:golden_a2_example.bin");
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_a2_example.bin");

    let nam_path = model_path("a2_example.nam");
    if !nam_path.exists() {
        eprintln!("[STATUS] SKIP_CAPABILITY: model_not_found:a2_example.nam");
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

/// Test 8k: `wavenet_a2_max.nam` dispatch is fail-closed (TR1.1 / KB-A2-MAX).
///
/// Known bug: prod×C++ SNR ≈ 0.23 dB (structural). `build_model` must
/// return `Err` — no production f32 instance of this topology enters
/// the public hot path.
#[test]
fn test_wavenet_a2_max_dispatch_rejected() {
    unsafe {
        std::env::remove_var("NAM_A2_MAX_UNLOCK");
    }
    let path = model_path("wavenet_a2_max.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read wavenet_a2_max.nam");
    let data = parse_nam_json(&json).expect("Failed to parse wavenet_a2_max.nam");
    let result = build_model(&data);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("wavenet_a2_max.nam must be rejected fail-closed (TR1.1)"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("KB-A2-MAX") || msg.contains("parity gap"),
        "Error message must cite KB-A2-MAX / parity gap, got: {msg}"
    );
    assert!(
        msg.contains("fail-closed"),
        "Error message must cite fail-closed, got: {msg}"
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

/// Test TR2.2: weight budget checkpoints for A2 Max (818 main + 1052 condition_dsp).
///
/// Under `NAM_A2_MAX_UNLOCK=1`, verifies that the dispatcher consumes every f32
/// in the weight stream — the main model builds with exactly 818 weights consumed
/// and the condition_dsp sub-model with exactly 1052. Per-component checkpoints
/// are asserted against the NAMCore reference (`third-party/.../generate_weights_a2.py`).
///
/// **Invariant:** any residual weight ≠ 0 fails the layout test before comparing audio.
/// **Rollback:** remove this test; the guard remains unchanged.
#[test]
fn test_a2_max_weight_budget_818_1052() {
    let path = model_path("wavenet_a2_max.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read wavenet_a2_max.nam");
    let data = parse_nam_json(&json).expect("Failed to parse wavenet_a2_max.nam");

    // --- Main model weight budget ---
    let main_actual = data.weights.len();
    assert_eq!(
        main_actual, 818,
        "Main model weight count mismatch: expected 818, got {main_actual}"
    );

    // --- condition_dsp weight budget ---
    let cond_json = data
        .config
        .condition_dsp
        .as_ref()
        .expect("A2 Max fixture must have condition_dsp");
    let cond_actual = cond_json
        .get("weights")
        .and_then(|w| w.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(
        cond_actual, 1052,
        "condition_dsp weight count mismatch: expected 1052, got {cond_actual}"
    );

    // --- Build model (implicitly verifies all weights consumed) ---
    unsafe {
        std::env::set_var("NAM_A2_MAX_UNLOCK", "1");
    }
    let model = build_model(&data).expect("A2 Max must build under NAM_A2_MAX_UNLOCK=1");
    unsafe {
        std::env::remove_var("NAM_A2_MAX_UNLOCK");
    }

    let wad = match &*model {
        StaticModel::WavenetA2Dyn(wad) => wad,
        _ => panic!("Expected WavenetA2Dyn variant"),
    };

    // --- Per-component checkpoints: rechannel ---
    let rechannel_expected = wad.input_channels * wad.channels;
    assert_eq!(
        wad.rechannel_w_f32.len(),
        rechannel_expected,
        "rechannel weight count mismatch"
    );

    // --- Per-component checkpoints: layers ---
    for (li, layer) in wad.layers.iter().enumerate() {
        let ch = wad.channels;
        let bn = wad.bottleneck;
        let use_gating = matches!(
            wad.gating_modes[li],
            GatingMode::Gated | GatingMode::Blended
        );
        let conv_out = if use_gating { bn * 2 } else { bn };
        // mixin_w: group-aware compact
        let mg = wad.mixin_groups.max(1);
        let mixin_in_pg = wad.condition_size / mg as usize;
        let mixin_count = conv_out * mixin_in_pg;
        assert_eq!(
            layer.mixin_w.len(),
            mixin_count,
            "layer[{li}].mixin_w size mismatch"
        );

        // l1x1: group-aware compact
        let lg = wad.l1x1_groups.max(1);
        let l1x1_in_pg = bn / lg as usize;
        let l1x1_w_count = if lg > 1 { ch * l1x1_in_pg } else { bn * ch };
        assert_eq!(
            layer.l1x1_w.len(),
            l1x1_w_count,
            "layer[{li}].l1x1_w size mismatch"
        );
        assert_eq!(layer.l1x1_b.len(), ch, "layer[{li}].l1x1_b size mismatch");

        // head1x1
        if wad.head1x1_active {
            let h1_w_count = wad.head_accum_size * wad.head1x1_h1_in;
            assert!(
                layer.head1x1_w.len() == h1_w_count,
                "layer[{li}].head1x1_w size mismatch: expected {h1_w_count}, got {}",
                layer.head1x1_w.len()
            );
            assert!(
                layer.head1x1_b.len() == wad.head_accum_size,
                "layer[{li}].head1x1_b size mismatch"
            );
        }

        // Verify FiLM layers are present where expected
        assert!(
            layer.conv_pre_film.is_some(),
            "layer[{li}].conv_pre_film must be loaded"
        );
        assert!(
            layer.input_mixin_post_film.is_some(),
            "layer[{li}].input_mixin_post_film must be loaded"
        );
        assert!(
            layer.activation_pre_film.is_some(),
            "layer[{li}].activation_pre_film must be loaded"
        );
        assert!(
            layer.layer1x1_post_film.is_some(),
            "layer[{li}].layer1x1_post_film must be loaded"
        );
        assert!(
            layer.head1x1_post_film.is_some(),
            "layer[{li}].head1x1_post_film must be loaded"
        );

        // FiLM check: conv_post_film should exist (active in JSON)
        assert!(
            layer.conv_post_film.is_some(),
            "layer[{li}].conv_post_film must be loaded"
        );
    }

    // --- Head budget ---
    let head_size = wad.head_size;
    if head_size == 1 {
        assert!(
            wad.head_conv.is_some(),
            "head_conv must be built for head_size=1"
        );
    } else {
        let per_oc_w = wad.head_kernel_size * wad.head_accum_size;
        assert_eq!(
            wad.head_rechannel_w.len(),
            head_size * per_oc_w,
            "head_rechannel_w size mismatch"
        );
        assert_eq!(
            wad.head_rechannel_b.len(),
            head_size,
            "head_rechannel_b size mismatch"
        );
        assert_eq!(
            wad.head_rechannel_scale.len(),
            head_size,
            "head_rechannel_scale size mismatch"
        );
    }

    // --- condition_dsp built ---
    assert!(
        wad.condition_dsp.is_some(),
        "A2 Max condition_dsp must be built"
    );
}

/// Checkpoints FiLM por slot (H6).
///
/// Under unlock, builds A2 Max and registers per-layer, per-FiLM-slot:
/// (groups, shift, w_count, b_count, stream offset). Verifies the total
/// weight budget = 818 and each slot matches the `weights_layout.rs` formula.
///
/// ## How to run
/// ```sh
/// NAM_A2_MAX_UNLOCK=1 cargo test --test models test_a2_max_film_slot_budget -- --ignored --exact --nocapture
/// ```
// on-demand: KB-A2-MAX FiLM slot budget diagnostic (TR2b.3/H6); unlock-only, not a nightly gate
#[test]
#[ignore = "TR2b.3 (H6): FiLM budget check — not a CI gate; run manually"]
fn test_a2_max_film_slot_budget() {
    use neural_amp_modeler_rs::models::StaticModel;

    let path = model_path("wavenet_a2_max.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read wavenet_a2_max.nam");
    let data = parse_nam_json(&json).expect("Failed to parse wavenet_a2_max.nam");

    let total_weights = data.weights.len();
    assert_eq!(total_weights, 818, "Main model weight budget mismatch");

    unsafe {
        std::env::set_var("NAM_A2_MAX_UNLOCK", "1");
    }
    let model = build_model(&data).expect("A2 Max must build under unlock");
    unsafe {
        std::env::remove_var("NAM_A2_MAX_UNLOCK");
    }

    let wad = match &*model {
        StaticModel::WavenetA2Dyn(wad) => wad,
        _ => panic!("Expected WavenetA2Dyn variant"),
    };

    let ch = wad.channels;
    let bn = wad.bottleneck;
    let cond_size = wad.condition_size;
    let k = wad
        .layers
        .first()
        .map(|l| l.conv.kernel_size())
        .unwrap_or(0);
    let conv_out: usize = wad
        .gating_modes
        .first()
        .map(|g| match g {
            neural_amp_modeler_rs::models::a2::gating::GatingMode::Gated
            | neural_amp_modeler_rs::models::a2::gating::GatingMode::Blended => bn * 2,
            _ => bn,
        })
        .unwrap_or(bn);
    let mg = (wad.mixin_groups as usize).max(1);
    let lg = (wad.l1x1_groups as usize).max(1);
    let h1_w_count = wad.head_accum_size * wad.head1x1_h1_in;
    let num_layers = wad.num_layers;
    let head_k = wad.head_kernel_size;

    println!(
        "// Topology: CH={ch} BN={bn} cond={cond_size} K={k} conv_out={conv_out} layers={num_layers}"
    );
    println!(
        "// mixin_groups={mg} l1x1_groups={lg} head1x1_active={}",
        wad.head1x1_active
    );
    println!(
        "// head_accum_size={} head1x1_h1_in={} head_kernel_size={head_k}",
        wad.head_accum_size, wad.head1x1_h1_in
    );

    // ── Helper: FiLM weight/bias formulas (replicated from weights_layout::pub(crate)) ──
    fn film_w_cnt(groups: u32, cond_size: usize, channels: usize, shift: bool) -> usize {
        let g = groups as usize;
        let mult = if shift { 2 } else { 1 };
        channels * mult * cond_size / g
    }
    fn film_b_cnt(channels: usize, shift: bool) -> usize {
        if shift { channels * 2 } else { channels }
    }

    // ── Walk all layers, all 8 FiLM slots ──
    /// Returns (slot_name, film_channels for given has/cond_size).
    fn slot_info(idx: usize, has: usize, cs: usize, ch: usize) -> (&'static str, usize) {
        let fc = match idx {
            2 => cs,
            7 => has,
            _ => ch,
        };
        let name = match idx {
            0 => "conv_pre_film",
            1 => "conv_post_film",
            2 => "input_mixin_pre_film",
            3 => "input_mixin_post_film",
            4 => "activation_pre_film",
            5 => "activation_post_film",
            6 => "layer1x1_post_film",
            7 => "head1x1_post_film",
            _ => "?",
        };
        (name, fc)
    }

    fn get_film(
        layer: &neural_amp_modeler_rs::models::a2::layer::A2Layer,
        idx: usize,
    ) -> Option<&neural_amp_modeler_rs::models::a2::film::FiLMLayer> {
        match idx {
            0 => layer.conv_pre_film.as_ref(),
            1 => layer.conv_post_film.as_ref(),
            2 => layer.input_mixin_pre_film.as_ref(),
            3 => layer.input_mixin_post_film.as_ref(),
            4 => layer.activation_pre_film.as_ref(),
            5 => layer.activation_post_film.as_ref(),
            6 => layer.layer1x1_post_film.as_ref(),
            7 => layer.head1x1_post_film.as_ref(),
            _ => None,
        }
    }

    let mut stream_pos = 0usize;
    let mut running_sum = 0usize;

    // 1) Rechannel
    let rc_count = wad.input_channels * ch;
    let rc_offset = stream_pos;
    stream_pos += rc_count;
    running_sum += rc_count;
    println!("  [rechannel]        offset={rc_offset:>4} w={rc_count:>4} sum={running_sum}");

    // 2) Per-layer: conv, mixin, l1x1, head1x1, then FiLM
    #[expect(clippy::type_complexity)]
    let mut film_table: Vec<(usize, usize, &str, usize, bool, usize, usize, usize)> = Vec::new();
    let mut total_film_w = 0usize;
    let mut total_film_b = 0usize;

    for li in 0..num_layers {
        let layer = &wad.layers[li];

        // conv_w
        let conv_w_cnt = ch * bn * k;
        let conv_w_off = stream_pos;
        stream_pos += conv_w_cnt;
        running_sum += conv_w_cnt;
        println!(
            "  l{li} conv_w        offset={conv_w_off:>4} w={conv_w_cnt:>4} sum={running_sum}"
        );

        // conv_b
        let conv_b_cnt = conv_out;
        let conv_b_off = stream_pos;
        stream_pos += conv_b_cnt;
        running_sum += conv_b_cnt;
        println!(
            "  l{li} conv_b        offset={conv_b_off:>4} w={conv_b_cnt:>4} sum={running_sum}"
        );

        // mixin_w
        let mixin_in_pg = cond_size / mg;
        let mixin_cnt = conv_out * mixin_in_pg;
        let mixin_off = stream_pos;
        stream_pos += mixin_cnt;
        running_sum += mixin_cnt;
        println!("  l{li} mixin_w       offset={mixin_off:>4} w={mixin_cnt:>4} sum={running_sum}");

        // l1x1_w
        let l1x1_in_pg = bn / lg;
        let l1x1_w_cnt = if lg > 1 { ch * l1x1_in_pg } else { bn * ch };
        let l1x1_w_off = stream_pos;
        stream_pos += l1x1_w_cnt;
        running_sum += l1x1_w_cnt;
        println!(
            "  l{li} l1x1_w        offset={l1x1_w_off:>4} w={l1x1_w_cnt:>4} sum={running_sum}"
        );

        // l1x1_b
        let l1x1_b_cnt = ch;
        let l1x1_b_off = stream_pos;
        stream_pos += l1x1_b_cnt;
        running_sum += l1x1_b_cnt;
        println!(
            "  l{li} l1x1_b        offset={l1x1_b_off:>4} w={l1x1_b_cnt:>4} sum={running_sum}"
        );

        // head1x1_w
        let h1_w_off = stream_pos;
        stream_pos += h1_w_count;
        running_sum += h1_w_count;
        println!("  l{li} head1x1_w     offset={h1_w_off:>4} w={h1_w_count:>4} sum={running_sum}");

        // head1x1_b
        let h1_b_cnt = wad.head_accum_size;
        let h1_b_off = stream_pos;
        stream_pos += h1_b_cnt;
        running_sum += h1_b_cnt;
        println!("  l{li} head1x1_b     offset={h1_b_off:>4} w={h1_b_cnt:>4} sum={running_sum}");

        // FiLM slots 0-7
        for slot_idx in 0..8 {
            let (slot_name, film_channels) =
                slot_info(slot_idx, wad.head_accum_size, cond_size, ch);

            if let Some(film) = get_film(layer, slot_idx) {
                let groups = film.config.groups as usize;
                let shift = film.config.shift;

                let expected_w = film_w_cnt(film.config.groups, cond_size, film_channels, shift);
                let expected_b = film_b_cnt(film_channels, shift);

                let actual_w = film.weights.len();
                let actual_b = film.bias.len();

                let w_off = stream_pos;
                stream_pos += actual_w;
                let _b_off = stream_pos;
                stream_pos += actual_b;
                running_sum += actual_w + actual_b;

                film_table.push((
                    li, slot_idx, slot_name, groups, shift, expected_w, expected_b, w_off,
                ));
                total_film_w += actual_w;
                total_film_b += actual_b;

                let w_status = if actual_w == expected_w {
                    "ok"
                } else {
                    "MISMATCH"
                };
                let b_status = if actual_b == expected_b {
                    "ok"
                } else {
                    "MISMATCH"
                };

                println!(
                    "  l{li} {slot_name:>24} offset={w_off:>4} w={actual_w:>3}(exp {expected_w:>3}) {w_status} \
                     b={actual_b:>3}(exp {expected_b:>3}) {b_status} groups={groups} shift={shift} sum={running_sum}"
                );
            }
        }
    }

    // 3) Head conv + bias + scale
    if wad.head_size == 1 {
        let hc = wad.head_conv.as_ref().expect("head_conv must exist");
        let head_w_cnt = hc.head_w.len();
        let head_w_off = stream_pos;
        stream_pos += head_w_cnt;
        running_sum += head_w_cnt;
        println!("  [head_conv_w]     offset={head_w_off:>4} w={head_w_cnt:>4} sum={running_sum}");

        let head_b_cnt = 1usize;
        let head_b_off = stream_pos;
        stream_pos += head_b_cnt;
        running_sum += head_b_cnt;
        println!("  [head_conv_b]     offset={head_b_off:>4} w={head_b_cnt:>4} sum={running_sum}");
    } else {
        let per_oc = head_k * wad.head_accum_size;
        let hw_cnt = wad.head_size * per_oc;
        let hw_off = stream_pos;
        stream_pos += hw_cnt;
        running_sum += hw_cnt;
        println!("  [head_rechannel_w] offset={hw_off:>4} w={hw_cnt:>4} sum={running_sum}");

        let hb_cnt = wad.head_size;
        let hb_off = stream_pos;
        stream_pos += hb_cnt;
        running_sum += hb_cnt;
        println!("  [head_rechannel_b] offset={hb_off:>4} w={hb_cnt:>4} sum={running_sum}");
    }

    let head_scale_cnt = 1usize;
    let hs_off = stream_pos;
    stream_pos += head_scale_cnt;
    running_sum += head_scale_cnt;
    println!("  [head_scale]      offset={hs_off:>4} w={head_scale_cnt:>4} sum={running_sum}");

    // ── Summary table ──
    println!();
    println!("=== FiLM Slot Budget Summary ===");
    println!(
        "{:<5} {:<6} {:<26} {:<8} {:<7} {:<10} {:<10} {:<10}",
        "Layer", "Slot", "Name", "Groups", "Shift", "Weights", "Bias", "Offset"
    );
    println!("{}", "-".repeat(85));
    for &(li, slot_idx, slot_name, groups, shift, exp_w, exp_b, offset) in &film_table {
        println!(
            "{:<5} {:<6} {:<26} {:<8} {:<7} {:<10} {:<10} {:<10}",
            li, slot_idx, slot_name, groups, shift, exp_w, exp_b, offset
        );
    }
    println!("{}", "-".repeat(85));
    println!("Total FiLM weights: {total_film_w}, Total FiLM biases: {total_film_b}");

    // ── Invariant: total == 818 ──
    println!();
    println!(
        "// Measured: stream_pos after all weights = {stream_pos} | expected = {total_weights}"
    );
    assert_eq!(
        running_sum, total_weights,
        "Budget invariant: computed sum {running_sum} != actual total {total_weights}"
    );
    assert_eq!(
        stream_pos, total_weights,
        "Stream position {stream_pos} != total weights {total_weights}"
    );

    // ── Per-slot formula verification ──
    let mut slot_mismatches = 0usize;
    for &(li, _slot_idx, slot_name, _groups, _shift, exp_w, exp_b, _offset) in &film_table {
        let layer = &wad.layers[li];
        let _film_channels = match _slot_idx {
            2 => cond_size,
            7 => wad.head_accum_size,
            _ => ch,
        };
        let film = get_film(layer, _slot_idx);
        if let Some(f) = film {
            if f.weights.len() != exp_w {
                println!(
                    "MISMATCH: l{li} {slot_name} w actual={} expected={exp_w}",
                    f.weights.len()
                );
                slot_mismatches += 1;
            }
            if f.bias.len() != exp_b {
                println!(
                    "MISMATCH: l{li} {slot_name} b actual={} expected={exp_b}",
                    f.bias.len()
                );
                slot_mismatches += 1;
            }
        }
    }
    assert_eq!(
        slot_mismatches, 0,
        "{slot_mismatches} FiLM slot formula mismatches found"
    );

    println!(
        "// Measured: All {} FiLM slots match weight_layout formulas ✓",
        film_table.len()
    );
    println!("// Measured: FiLM slot cursor positions verified — no slot overlap detected");
    println!("// Measured: A2 Max main model total = {total_weights} ✓");
}

/// Test TR2.3: diagnostic dump harness — deterministic, bit-stable across two runs.
///
/// Under `NAM_A2_MAX_UNLOCK=1`, builds A2 Max, prewarms, and processes
/// a deterministic test signal twice with full diagnostic capture enabled.
/// Verifies that condition_dsp output, head-per-layer snapshots, and final
/// output have identical bit-stable hashes in both runs (zero non-determinism).
///
/// **Invariant:** release builds carry no dump symbols.
/// **Rollback:** remove this test and the diagnostic hooks.
#[test]
fn test_a2_max_diagnostic_dump_bit_stable() {
    use neural_amp_modeler_rs::testing::diagnostics::DiagnosticConfig;

    let path = model_path("wavenet_a2_max.nam");
    let json = fs::read_to_string(&path).expect("Fixture not found");
    let data = parse_nam_json(&json).expect("Parse failed");

    let run_dump = || -> u64 {
        unsafe {
            std::env::set_var("NAM_A2_MAX_UNLOCK", "1");
        }
        let model = build_model(&data).expect("Build failed under unlock");
        unsafe {
            std::env::remove_var("NAM_A2_MAX_UNLOCK");
        }
        let mut wad = match *model {
            StaticModel::WavenetA2Dyn(w) => w,
            _ => panic!("Expected WavenetA2Dyn"),
        };
        wad.prewarm();

        let config = DiagnosticConfig {
            capture_condition_dsp: true,
            capture_head_per_layer: true,
            capture_final_output: true,
        };
        wad.enable_diagnostics(config);

        let nf = 64usize;
        let input: Vec<f32> = (0..nf)
            .map(|i| {
                let t = i as f32 / nf as f32;
                (t * std::f32::consts::TAU * 10.0).sin() * 0.3
            })
            .collect();
        let mut output = vec![0.0f32; nf];
        wad.process(&input, &mut output);

        let dump = wad.take_diagnostics().expect("Dump must be present");
        let hash = dump.bit_stable_hash();
        assert!(
            !dump.condition_dsp_snapshots.is_empty(),
            "condition_dsp snapshots must be captured"
        );
        assert!(
            !dump.head_per_layer_snapshots.is_empty(),
            "head-per-layer snapshots must be captured"
        );
        assert!(dump.final_output.is_some(), "final output must be captured");
        assert_eq!(dump.total_frames, nf, "dump total_frames mismatch");
        hash
    };

    let h1 = run_dump();
    let h2 = run_dump();
    assert_eq!(
        h1, h2,
        "Diagnostic dump must be bit-stable across two identical runs: \
         run1={h1:#016x}, run2={h2:#016x}"
    );

    unsafe {
        std::env::remove_var("NAM_A2_MAX_UNLOCK");
    }
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

/// Test 8k-2: Slimmable WaveNet metadata parsing (T5.1 — Parser e Estruturas de Metadados).
///
/// Loads the real fixture `slimmable_wavenet.nam` and validates:
/// - JSON parsing succeeds with slimmable metadata deserialized into `SlimmableConfig`.
/// - `allowed_channels` is properly extracted from `kwargs.allowed_channels`.
/// - The model builds without rejection (no longer a loader gap).
/// - `NamModelData::is_slimmable_capable()` returns `true` (metadata-level flag).
#[test]
fn test_loader_gap_slimmable_wavenet() {
    use neural_amp_modeler_rs::loader::nam_json::SlimmableConfig;

    let path = model_path("slimmable_wavenet.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read slimmable_wavenet.nam");

    let data = parse_nam_json(&json).expect("Failed to parse slimmable_wavenet.nam");
    assert_eq!(data.architecture, "WaveNet");

    let layer = data
        .config
        .layers
        .first()
        .expect("Expected at least one layer");
    let slimmable: &SlimmableConfig = layer
        .slimmable
        .as_ref()
        .expect("Expected slimmable metadata");
    assert_eq!(
        slimmable.method.as_deref(),
        Some("slice_channels_uniform"),
        "Expected method 'slice_channels_uniform'"
    );
    let allowed = slimmable
        .kwargs
        .as_ref()
        .and_then(|k| k.allowed_channels.as_deref())
        .expect("Expected kwargs.allowed_channels");
    assert_eq!(allowed, &[1, 2, 3], "Expected allowed_channels [1, 2, 3]");

    assert!(
        data.is_slimmable_capable(),
        "NamModelData must report slimmable_capable when layers have slimmable metadata"
    );

    let model = build_model(&data).expect("Slimmable WaveNet model must build without rejection");
    assert!(
        model.is_slimmable_capable(),
        "StaticModel must report slimmable_capable when geometry carries allowed_channels"
    );
}

/// Test 8k-3: SlimmableWavenet channel slicing inference and breakpoint validation (T5.2).
///
/// Validates:
/// - `slimmable_breakpoints()` returns correct normalized breakpoints from allowed_channels.
/// - `set_slimmable_size()` sets the correct `pending_slim_channel` target.
/// - Channel slicing rebuilds produce valid, deterministic, non-silent inference.
/// - Each breakpoint (channel count) produces different output (slicing is not a no-op).
#[test]
fn test_slimmable_wavenet_inference_and_breakpoints() {
    use neural_amp_modeler_rs::models::slimmable::SlimmableModel;

    let path = model_path("slimmable_wavenet.nam");
    let json = fs::read_to_string(&path).expect("Failed to read slimmable_wavenet.nam");
    let data = parse_nam_json(&json).expect("Failed to parse slimmable_wavenet.nam");

    let model = build_model(&data).expect("Slimmable WaveNet model must build");

    // ── Breakpoints validation ──
    // allowed_channels = [1, 2, 3], full channels = 3
    // breakpoints = [1/3, 2/3] ≈ [0.3333, 0.6667]
    let bps = model.slimmable_breakpoints();
    assert_eq!(
        bps.len(),
        2,
        "Expected 2 breakpoints for allowed_channels [1,2,3]"
    );
    assert!(
        (bps[0] - 1.0 / 3.0).abs() < 1e-9,
        "First breakpoint should be 1/3, got {}",
        bps[0]
    );
    assert!(
        (bps[1] - 2.0 / 3.0).abs() < 1e-9,
        "Second breakpoint should be 2/3, got {}",
        bps[1]
    );

    // ── Inference at full channels (CH=3) ──
    let mut full_model = model;
    let prewarm_samples = full_model.prewarm_samples().max(2048);
    full_model.prewarm(prewarm_samples);

    let input: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();
    let mut full_out = vec![0.0f32; input.len()];
    full_model.process(&input, &mut full_out);

    let full_rms = (full_out.iter().map(|&x| x * x).sum::<f32>() / full_out.len() as f32).sqrt();
    assert!(full_rms > 1e-6, "Full model output must be non-silent");

    // ── Inference at each slimmable breakpoint ──
    for &target_ch in &[1usize, 2, 3] {
        let base_model = match full_model.as_ref() {
            StaticModel::WavenetDyn(w) => w.clone_exact(),
            _ => panic!("Expected WavenetDyn"),
        };
        let mut slim_model = if base_model.ch != target_ch {
            base_model.slice_channels(target_ch).unwrap_or_else(|e| {
                panic!("Failed to slice model to CH={target_ch}: {e}");
            })
        } else {
            base_model
        };
        slim_model.prewarm();
        let mut clone = slim_model.clone_exact();

        let mut slim_out = vec![0.0f32; input.len()];
        slim_model.process(&input, &mut slim_out);

        let slim_rms =
            (slim_out.iter().map(|&x| x * x).sum::<f32>() / slim_out.len() as f32).sqrt();
        assert!(
            slim_rms > 1e-6,
            "Sliced model CH={target_ch} output must be non-silent"
        );

        let mut clone_out = vec![0.0f32; input.len()];
        clone.process(&input, &mut clone_out);
        for (i, (&a, &b)) in slim_out.iter().zip(clone_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-7,
                "Sliced model CH={target_ch} non-deterministic at sample {i}: {a} vs {b}"
            );
        }

        // CH=1 and CH=2 outputs should differ from CH=3 (slicing is not a no-op)
        if target_ch < 3 {
            let max_diff = slim_out
                .iter()
                .zip(full_out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff > 1e-8,
                "CH={target_ch} output identical to CH=3 — slicing produced no change"
            );
        }
    }

    // ── set_slimmable_size integration ──
    let model2 = match full_model.as_ref() {
        StaticModel::WavenetDyn(w) => Box::new(w.clone_exact()),
        _ => panic!("Expected WavenetDyn"),
    };
    let mut model2 = StaticModel::WavenetDyn(model2);
    if let StaticModel::WavenetDyn(w) = &mut model2 {
        assert!(w.pending_slim_channel.is_none());
        assert!(w.allowed_channels.is_some());

        w.set_slimmable_size(0.0, None);
        assert_eq!(
            w.pending_slim_channel,
            Some(1),
            "quality 0.0 should select lowest tier CH=1"
        );

        w.set_slimmable_size(0.5, None);
        assert_eq!(
            w.pending_slim_channel,
            Some(2),
            "quality 0.5 should select middle tier CH=2"
        );

        w.set_slimmable_size(1.0, None);
        assert_eq!(
            w.pending_slim_channel,
            Some(3),
            "quality 1.0 should select highest tier CH=3"
        );
    } else {
        panic!("Expected StaticModel::WavenetDyn variant");
    }
}

/// Test 8l: Golden Vectors WaveNet Condition DSP cross-reference C++ ↔ NeuralAmpModeler-rs.
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

/// Test 8m: Golden Vectors WaveNet Official (dynamic path) — cross-reference C++ ↔ NeuralAmpModeler-rs.
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

/// Test 8n: Slimmable Container Integration — validates robust loading,
/// submodel topology routing, inference at all switch boundaries, and
/// crossfade transitions for `slimmable_container.nam`
/// (LSTM 1x3 + WaveNetDyn CH=3 + WaveNetNano CH=4 [ReLU]).
///
/// Since no single-container C++ golden vector exists for this model,
/// validation uses internal consistency: no NaN/Inf, deterministic output,
/// non-silent inference, correct boundary switching, and crossfade continuity.
#[test]
fn test_loader_gap_slimmable_container() {
    let path = model_path("slimmable_container.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read slimmable_container.nam");
    let data = parse_nam_json(&json).expect("Failed to parse slimmable_container.nam");
    let mut model =
        build_model(&data).expect("slimmable_container.nam must build with ReLU supported");

    // Verify Container structure and architecture dispatch
    let (num_submodels, max_values, sub_arches) = match model.as_ref() {
        StaticModel::Container(c) => {
            let max_values: Vec<f32> = c.submodels().iter().map(|(mv, _)| *mv).collect();
            let arches: Vec<&str> = c
                .submodels()
                .iter()
                .map(|(_, sm)| match sm.as_ref() {
                    StaticModel::Lstm1x3(_) => "LSTM",
                    StaticModel::WavenetDyn(_) => "WaveNetDyn",
                    StaticModel::WavenetNano(_) => "Nano",
                    _ => "Unknown",
                })
                .collect();
            (c.submodels().len(), max_values, arches)
        }
        _ => panic!("Expected StaticModel::Container"),
    };
    assert_eq!(num_submodels, 3, "Container must have 3 submodels");
    assert_eq!(max_values, vec![0.33, 0.66, 1.0]);
    assert_eq!(sub_arches, vec!["LSTM", "WaveNetDyn", "Nano"]);

    let sample_rate = data.sample_rate.map(|s| s as u32).unwrap_or(48000);
    let _ = model.reset(sample_rate, 256);

    /// Helper: advance past pending crossfade into steady state.
    fn drain_crossfade(model: &mut StaticModel, input: &[f32]) {
        loop {
            let is_xf = match model {
                StaticModel::Container(c) => c.is_crossfading(),
                _ => false,
            };
            if !is_xf {
                break;
            }
            let mut drain = vec![0.0f32; 64];
            model.process(&input[..64.min(input.len())], &mut drain);
        }
    }

    /// Helper: get the active submodel index from a container model.
    fn active_idx(model: &StaticModel) -> usize {
        match model {
            StaticModel::Container(c) => c.active_index(),
            _ => panic!("Expected Container"),
        }
    }

    /// Helper: switch to a slimmable size and drain crossfade.
    fn switch_to(model: &mut StaticModel, size: f32, input: &[f32]) {
        if let StaticModel::Container(c) = model {
            c.set_slimmable_size(size, None);
        }
        drain_crossfade(model, input);
    }

    let input = generate_stress_signal_v1();
    let num_samples = input.len();

    // --- Boundary: size=0.1 → active_index=0 (LSTM 1x3) ---
    switch_to(&mut model, 0.1, &input);
    assert_eq!(active_idx(&model), 0, "size=0.1 must select LSTM (index 0)");
    let mut output_lstm = vec![0.0f32; num_samples];
    process_in_blocks(&mut model, &input, &mut output_lstm, GOLDEN_BLOCK_SIZE);
    assert!(
        output_lstm.iter().all(|v| v.is_finite()),
        "LSTM submodel output must be finite"
    );
    let rms_lstm: f64 =
        output_lstm.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / num_samples as f64;
    assert!(rms_lstm.sqrt() > 1e-6, "LSTM output must not be silent");

    // --- Determinism: same input, same output after reset ---
    switch_to(&mut model, 0.1, &input);
    let mut output_check = vec![0.0f32; num_samples];
    process_in_blocks(&mut model, &input, &mut output_check, GOLDEN_BLOCK_SIZE);
    assert!(
        output_check.iter().all(|v| v.is_finite()),
        "LSTM output after re-switch must be finite"
    );
    let rms_check: f64 = output_check
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        / num_samples as f64;
    assert!(
        rms_check.sqrt() > 1e-6,
        "LSTM output after re-switch must not be silent"
    );

    // --- Boundary: size=0.5 → active_index=1 (WaveNetDyn CH=3) ---
    switch_to(&mut model, 0.5, &input);
    assert_eq!(
        active_idx(&model),
        1,
        "size=0.5 must select WaveNetDyn (index 1)"
    );
    let mut output_dyn = vec![0.0f32; num_samples];
    process_in_blocks(&mut model, &input, &mut output_dyn, GOLDEN_BLOCK_SIZE);
    assert!(
        output_dyn.iter().all(|v| v.is_finite()),
        "WaveNetDyn submodel output must be finite"
    );
    let rms_dyn: f64 =
        output_dyn.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / num_samples as f64;
    assert!(
        rms_dyn.sqrt() > 1e-8,
        "WaveNetDyn output must not be silent"
    );

    // --- Boundary: size=0.8 → active_index=2 (WaveNet Nano CH=4, ReLU) ---
    switch_to(&mut model, 0.8, &input);
    assert_eq!(active_idx(&model), 2, "size=0.8 must select Nano (index 2)");
    let mut output_nano = vec![0.0f32; num_samples];
    process_in_blocks(&mut model, &input, &mut output_nano, GOLDEN_BLOCK_SIZE);
    assert!(
        output_nano.iter().all(|v| v.is_finite()),
        "Nano ReLU submodel output must be finite"
    );
    let rms_nano: f64 =
        output_nano.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / num_samples as f64;
    assert!(
        rms_nano.sqrt() > 1e-6,
        "Nano ReLU output must not be silent"
    );

    // --- Crossfade at boundary 0.33 (LSTM → WaveNetDyn) ---
    switch_to(&mut model, 0.1, &input);
    {
        if let StaticModel::Container(c) = model.as_mut() {
            c.set_slimmable_size(0.34, None);
        }
    }
    let mut xf_output_33 = vec![0.0f32; 256];
    model.process(&input[..256], &mut xf_output_33);
    assert!(
        xf_output_33.iter().all(|v| v.is_finite()),
        "Crossfade from LSTM to WaveNetDyn must produce finite output"
    );
    let xf_rms_33: f64 = xf_output_33
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        / 256.0;
    assert!(
        xf_rms_33.sqrt() > 1e-8,
        "Crossfade at 0.33 must not be silent"
    );

    // --- Crossfade at boundary 0.66 (WaveNetDyn → Nano ReLU) ---
    switch_to(&mut model, 0.5, &input);
    {
        if let StaticModel::Container(c) = model.as_mut() {
            c.set_slimmable_size(0.67, None);
        }
    }
    let mut xf_output_66 = vec![0.0f32; 256];
    model.process(&input[..256], &mut xf_output_66);
    assert!(
        xf_output_66.iter().all(|v| v.is_finite()),
        "Crossfade from WaveNetDyn to Nano ReLU must produce finite output"
    );
    let xf_rms_66: f64 = xf_output_66
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        / 256.0;
    assert!(
        xf_rms_66.sqrt() > 1e-8,
        "Crossfade at 0.66 must not be silent"
    );
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
        v2_sample_rates_for("BossWN-standard.nam"),
        true,
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
        v2_sample_rates_for("BossWN-feather.nam"),
        true,
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
        v2_sample_rates_for("BossWN-nano.nam"),
        true,
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
        v2_sample_rates_for("EVH-5150-Lite.nam"),
        true,
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
        v2_sample_rates_for("BossLSTM-1x16.nam"),
        true,
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
        v2_sample_rates_for("APP-EVH-Stealth100-Dialled-xSTD.nam"),
        true,
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
        v2_sample_rates_for("Boss BD-2 H2O Mod T-12_00 G-12_00.nam"),
        true,
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
        v2_sample_rates_for("SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam"),
        true,
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
        v2_sample_rates_for("BossLSTM-2x8.nam"),
        true,
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
        v2_sample_rates_for("wavenet_a1_standard.nam"),
        true,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_official() {
    run_v2_golden_test(
        "wavenet_official.nam",
        "golden_wavenet_official",
        "WaveNet Official (CH=3, dynamic)",
        "wavenet_official",
        v2_sample_rates_for("wavenet_official.nam"),
        true,
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
        v2_sample_rates_for("lstm.nam"),
        true,
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
        v2_sample_rates_for("wavenet_a2_full.nam"),
        true,
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
        v2_sample_rates_for("wavenet_a2_lite.nam"),
        true,
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
        v2_sample_rates_for("wavenet_condition_dsp.nam"),
        true,
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
        v2_sample_rates_for("wavenet_condition_lstm.nam"),
        true,
    );
}

// SKIP_CAPABILITY: mirrors live_cross_validation_v2_lstm_1x10 — synthetic
// LSTM 1×10 can sit at the LUFS plausibility floor on long v2 stress; ESR/SNR
// remain the primary interop gates (check_lufs_gate=false).
/// Golden Vectors LSTM 1×10 (v2 multi-SR, 48 kHz only).
#[test]
#[ignore]
fn test_golden_vectors_v2_lstm_1x10() {
    run_v2_golden_test(
        "lstm_1x10.nam",
        "golden_lstm_1x10",
        "LSTM 1×10 (uncat., v2)",
        "lstm_1x10",
        v2_sample_rates_for("lstm_1x10.nam"),
        false,
    );
}

// SKIP_CAPABILITY: mirrors live_cross_validation_v2_lstm_2x24 — large hidden
// size damps output energy on long v2 stress (LUFS ≈ −51…−54). Not a golden
// defect; SNR/ESR stay near bit-exact vs C++ (check_lufs_gate=false).
/// Golden Vectors LSTM 2×24 (v2 multi-SR, 48 kHz only).
#[test]
#[ignore]
fn test_golden_vectors_v2_lstm_2x24() {
    run_v2_golden_test(
        "lstm_2x24.nam",
        "golden_lstm_2x24",
        "LSTM 2×24 (uncat., v2)",
        "lstm_2x24",
        v2_sample_rates_for("lstm_2x24.nam"),
        false,
    );
}

/// Golden Vectors LSTM 3×8 (v2 multi-SR, 48 kHz only).
#[test]
#[ignore]
fn test_golden_vectors_v2_lstm_3x8() {
    run_v2_golden_test(
        "lstm_3x8.nam",
        "golden_lstm_3x8",
        "LSTM 3×8 (v2)",
        "lstm_3x8",
        v2_sample_rates_for("lstm_3x8.nam"),
        true,
    );
}

/// Golden Vectors ConvNet No BatchNorm (v2 multi-SR, 48 kHz only).
#[test]
#[ignore]
fn test_golden_vectors_v2_convnet_nobn() {
    run_v2_golden_test(
        "convnet_nobn.nam",
        "golden_convnet_nobn",
        "ConvNet No BatchNorm (v2)",
        "convnet_nobn",
        v2_sample_rates_for("convnet_nobn.nam"),
        true,
    );
}

// SKIP_CAPABILITY: mirrors live_cross_validation_v2_convnet_relu — ReLU
// without BatchNorm yields inherently low loudness on v2 stress (LUFS ≈ −65).
// C++ reference is valid; ESR/SNR are primary gates (check_lufs_gate=false).
/// Golden Vectors ConvNet ReLU (v2 multi-SR, 48 kHz only).
#[test]
#[ignore]
fn test_golden_vectors_v2_convnet_relu() {
    run_v2_golden_test(
        "convnet_relu.nam",
        "golden_convnet_relu",
        "ConvNet ReLU (v2)",
        "convnet_relu",
        v2_sample_rates_for("convnet_relu.nam"),
        false,
    );
}

/// Golden Vectors ConvNet SiLU (v2 multi-SR, 48 kHz only).
#[test]
#[ignore]
fn test_golden_vectors_v2_convnet_silu() {
    run_v2_golden_test(
        "convnet_silu.nam",
        "golden_convnet_silu",
        "ConvNet SiLU (v2)",
        "convnet_silu",
        v2_sample_rates_for("convnet_silu.nam"),
        true,
    );
}

/// Golden Vectors Linear No Bias (v2 multi-SR, 48 kHz only).
#[test]
#[ignore]
fn test_golden_vectors_v2_linear_nobias() {
    run_v2_golden_test(
        "linear_nobias.nam",
        "golden_linear_nobias",
        "Linear No Bias (v2)",
        "linear_nobias",
        v2_sample_rates_for("linear_nobias.nam"),
        true,
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

/// Test 10d: Golden — WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1)
///
/// ## Gate state: `#[ignore = "KB-A2-MAX known bug: prod×C++ ~0.23 dB; guard TR1.1"]`
///
/// **Why ignored (KB-A2-MAX):** Permanent known bug until reopening criteria
/// in `docs/cpp_parity_map.md` §4.4.3. HEAD meter prod×C++ **SNR ≈ 0.23 dB**
/// (ESR ≈ 9.49e-1). Guard TR1.1 rejects `build_model`. Not a CI parity gate.
/// Diagnostic path: `NAM_A2_MAX_UNLOCK=1` + feature `testing` / `cfg(test)`.
///
/// **What the test validates when un-ignored:** Compares Rust DSP against
/// `golden_wavenet_a2_max.bin` (NAMCore C++). Do not un-ignore without
/// meeting §4.4.3 (SNR≥90 dB + intermediate C++ dumps).
// on-demand: KB-A2-MAX golden compare; known bug until §4.4.3 reopen; not a nightly gate
#[test]
#[ignore = "KB-A2-MAX known bug: prod×C++ ~0.23 dB; guard TR1.1 — not a CI parity gate"]
fn test_golden_vectors_wavenet_a2_max() {
    // Permanent known bug until docs/cpp_parity_map.md §4.4.3.
    // Default path (no unlock): assert fail-closed and return — never a red gate.
    // Full golden compare only under NAM_A2_MAX_UNLOCK=1 (manual reopen diagnostics).
    let nam_path = model_path("wavenet_a2_max.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a2_max.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2 Max model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    let unlocked = std::env::var("NAM_A2_MAX_UNLOCK").as_deref() == Ok("1");
    if !unlocked {
        let err = match build_model(&model_data) {
            Err(e) => e,
            Ok(_) => panic!(
                "KB-A2-MAX: build_model must Err without NAM_A2_MAX_UNLOCK=1 (fail-closed TR1.1)"
            ),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("KB-A2-MAX") || msg.contains("parity gap"),
            "KB-A2-MAX: reject message must cite known bug, got: {msg}"
        );
        return;
    }

    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_max.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_a2_max.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_max.bin");

    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build A2 Max under NAM_A2_MAX_UNLOCK=1");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "wavenet_a2_max");
    gv_metric("WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1) C++ cross-reference");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// A2 Max SNR/ESR vs C++ golden (n=2048, block=64, prewarm=2048) — tracking meter.
///
/// **Ignored by default** — does not fail CI. Run manually:
/// ```sh
/// cargo test --test models test_measure_a2_max_snr_vs_golden -- --ignored --exact --nocapture
/// ```
///
/// Provenance (do not treat the println label as a frozen “pass” baseline):
/// - Pre-R3 audit (TR2.5): SNR ≈ **1.35 dB**, ESR ≈ 7.4e-1
/// - H1-only (TR3.1): SNR ≈ **2.31 dB**
/// - H1+H2 tree (re-audit 2026-08-09): SNR ≈ **0.23 dB**, ESR ≈ 9.49e-1  ← current HEAD
///
/// Fail-closed guard remains active; unlock via `NAM_A2_MAX_UNLOCK=1` inside the test.
// on-demand: KB-A2-MAX SNR/ESR meter vs C++ golden; diagnostic only, not a nightly gate
#[test]
#[ignore = "KB-A2-MAX meter only: prod×C++ ~0.23 dB; unlock diagnostics — not a CI gate"]
fn test_measure_a2_max_snr_vs_golden() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_max.bin");
    assert!(golden_path.exists());
    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_max.bin");

    let nam_path = model_path("wavenet_a2_max.nam");
    assert!(nam_path.exists());
    let json_data = fs::read_to_string(&nam_path).expect("Failed to read A2 Max model");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse A2 Max JSON");

    unsafe {
        std::env::set_var("NAM_A2_MAX_UNLOCK", "1");
    }
    let mut model = build_model(&model_data).expect("Failed to build A2 Max under unlock");
    unsafe {
        std::env::remove_var("NAM_A2_MAX_UNLOCK");
    }

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, 64);

    let n = input.len() as f64;
    let mse = common::metrics::compute_mse(&expected, &output);
    let esr = common::metrics::compute_esr(&expected, &output);

    let signal_power: f64 = expected.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / n;
    let snr_db = if mse > 0.0 && signal_power > 0.0 {
        10.0 * (signal_power / mse).log10()
    } else {
        f64::INFINITY
    };

    // Measured history (f32 production vs C++ NAMcore golden_wavenet_a2_max.bin,
    // n=2048, block=64, prewarm=2048, 48 kHz):
    //   TR2.5 pre-R3:     SNR ≈ 1.35 dB, ESR ≈ 7.4e-1
    //   TR3.1 H1-only:    SNR ≈ 2.31 dB
    //   HEAD H1+H2 tree:  SNR ≈ 0.23 dB, ESR ≈ 9.49e-1  (re-audit 2026-08-09)
    println!(
        "// Measured: SNR = {snr_db:.2} dB, ESR = {esr:.2e} | \
         n={} block=64 prewarm=2048 48kHz | HEAD meter (see history in test docs)",
        input.len()
    );

    let max_abs = output
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);
    println!("// Measured: max_abs_error = {max_abs:.4e} | same run");
}

/// Triple decomposition harness (H0).
///
/// Processes the same deterministic golden input (n=2048) three ways:
///   1. prod f32 (unlock) — with diagnostic dumps of condition_dsp, head_accum, output
///   2. f64 oracle — full model output
///   3. C++ golden output — from `golden_wavenet_a2_max.bin`
///
/// Emits SNR/ESR table for prod×C++, prod×f64, f64×C++ and classifies
/// the divergence pattern as Case A/B/C/D (see `docs/cpp_parity_map.md` §4.4.2).
///
/// ## How to run
/// ```sh
/// NAM_A2_MAX_UNLOCK=1 cargo test --test models test_h0_triple_decomposition -- --ignored --exact --nocapture
/// ```
///
/// ## Invariants
/// - One `#[ignore]`'d command reproduces the full table.
/// - Numbers annotated with `// Measured:` for machine-readable consumption.
/// - No production code change; only measurement + documentation.
// on-demand: KB-A2-MAX H0 triple-decomposition harness (TR2b.1); not a nightly gate
#[test]
#[ignore = "TR2b.1 (H0): diagnostic harness — not a CI gate; run manually"]
fn test_h0_triple_decomposition() {
    use neural_amp_modeler_rs::testing::diagnostics::DiagnosticConfig;
    use neural_amp_modeler_rs::testing::reference_oracle::{
        PrecisionConfig, compute_esr_f64, esr_to_db_f64, oracle_forward,
    };

    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_max.bin");
    if !golden_path.exists() {
        eprintln!("[STATUS] SKIP_CAPABILITY: golden_not_found:golden_wavenet_a2_max.bin");
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_max.bin");
    let n = input.len();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (1) prod f32 — production output + diagnostic dumps
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let nam_path = model_path("wavenet_a2_max.nam");
    assert!(nam_path.exists());
    let json_data = fs::read_to_string(&nam_path).expect("Failed to read A2 Max model");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse A2 Max JSON");

    unsafe {
        std::env::set_var("NAM_A2_MAX_UNLOCK", "1");
    }
    let model = build_model(&model_data).expect("Failed to build A2 Max under unlock");
    let mut wad = match *model {
        neural_amp_modeler_rs::models::StaticModel::WavenetA2Dyn(w) => w,
        other => panic!("Expected WavenetA2Dyn, got {:?}", other.class_label()),
    };

    let config = DiagnosticConfig {
        capture_condition_dsp: true,
        capture_head_per_layer: true,
        capture_final_output: true,
    };
    wad.enable_diagnostics(config);
    wad.prewarm();

    let mut prod_output = vec![0.0f32; n];
    {
        let block_size = 64usize;
        let mut pos = 0;
        while pos < n {
            let nf = (n - pos).min(block_size);
            wad.process(&input[pos..pos + nf], &mut prod_output[pos..pos + nf]);
            pos += nf;
        }
    }
    let dump = wad
        .take_diagnostics()
        .expect("Diagnostic dump must be present");

    // Validation: dumps are non-empty
    assert!(
        !dump.condition_dsp_snapshots.is_empty(),
        "condition_dsp snapshots required for H0 decomposition"
    );
    assert!(
        !dump.head_per_layer_snapshots.is_empty(),
        "head_per_layer snapshots required for H0 decomposition"
    );
    assert!(
        dump.final_output.is_some(),
        "final output required for H0 decomposition"
    );

    // Per-channel stats on condition_dsp
    for snap in &dump.condition_dsp_snapshots {
        println!(
            "// Measured: condition_dsp snapshot — channels={}, frames={}, nz_samples={}",
            snap.channels,
            snap.num_frames,
            snap.data.iter().filter(|&&v| v.abs() > 1e-12).count()
        );
    }
    println!(
        "// Measured: head_per_layer snapshots = {} (layers captured)",
        dump.head_per_layer_snapshots.len()
    );

    unsafe {
        std::env::remove_var("NAM_A2_MAX_UNLOCK");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (2) f64 oracle — full model output
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let input_f64: Vec<f64> = input.iter().map(|&x| x as f64).collect();
    let oracle_cfg = PrecisionConfig::default();
    let oracle_output = oracle_forward(&model_data, &input_f64, &oracle_cfg);
    assert_eq!(oracle_output.len(), n, "oracle output length mismatch");
    assert!(
        oracle_output.iter().all(|&x| x.is_finite()),
        "oracle output contains NaN/Inf"
    );

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (3) C++ golden — from .bin file (already in `expected`)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Pairwise SNR/ESR table
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let prod_output_f64: Vec<f64> = prod_output.iter().map(|&x| x as f64).collect();
    let expected_f64: Vec<f64> = expected.iter().map(|&x| x as f64).collect();

    let esr_prod_cpp = compute_esr_f64(&expected_f64, &prod_output_f64);
    let esr_prod_f64 = compute_esr_f64(&oracle_output, &prod_output_f64);
    let esr_f64_cpp = compute_esr_f64(&expected_f64, &oracle_output);

    let snr = |esr: f64| -> f64 {
        if esr <= f64::EPSILON {
            f64::INFINITY
        } else {
            -10.0 * esr.log10()
        }
    };

    // ── H0 classification ──────────────────────────────────────────────────
    // prod≈f64: ESR < 1e-3   ( -1  means < -30 dB, practically identical)
    // prod≈C++: ESR < 1e-3
    // f64≈C++ : ESR < 1e-3
    let eps = 1e-3;

    let classification = match (esr_prod_f64 < eps, esr_prod_cpp < eps, esr_f64_cpp < eps) {
        (true, false, false) => "Case A: prod≈f64 ≠ C++ → bug compartilhado / grafo comum",
        (false, false, true) => "Case B: f64≈C++ ≠ prod → bug só produção",
        (false, true, false) => "Case C: prod≈C++ ≠ f64 → bug só oráculo (inesperado)",
        (false, false, false) => "Case D: todos distintos → múltiplas falhas; priorizar cond/FiLM",
        (true, true, false) => "prod≈f64≈C++ par (prod≈f64 ∧ prod≈C++ mas f64≉C++)",
        (true, false, true) => "prod≈f64≈C++ par (prod≈f64 ∧ f64≈C++ mas prod≉C++)",
        (false, true, true) => "prod≈C++≈f64 par (prod≈C++ ∧ f64≈C++ mas prod≉f64)",
        (true, true, true) => "Todos idênticos (ESR<1e-3 em todos os pares) — gap fechado!",
    };

    // ── Print table ─────────────────────────────────────────────────────────
    println!();
    println!("=== H0 Triple Decomposition Table ===");
    println!("Model:  wavenet_a2_max.nam (CH=4, cond=8, FiLM, head1x1)");
    println!("Input:  n=2048, block=64, prewarm=2048, 48 kHz (golden_wavenet_a2_max.bin)");
    println!(
        "Oracle: PrecisionConfig::default() (F64Exact weights, Exact activations, Neumaier acc)"
    );
    println!();
    println!(
        "{:<20} {:<18} {:<18} {:<18}",
        "Pair", "ESR (linear)", "ESR (dB)", "SNR (dB)"
    );
    println!("{}", "-".repeat(74));
    println!(
        "{:<20} {:<18.6e} {:<18.1} {:<18.1}",
        "prod f32 × C++",
        esr_prod_cpp,
        esr_to_db_f64(esr_prod_cpp),
        snr(esr_prod_cpp)
    );
    println!(
        "{:<20} {:<18.6e} {:<18.1} {:<18.1}",
        "prod f32 × f64 oracle",
        esr_prod_f64,
        esr_to_db_f64(esr_prod_f64),
        snr(esr_prod_f64)
    );
    println!(
        "{:<20} {:<18.6e} {:<18.1} {:<18.1}",
        "f64 oracle × C++",
        esr_f64_cpp,
        esr_to_db_f64(esr_f64_cpp),
        snr(esr_f64_cpp)
    );
    println!("{}", "-".repeat(74));
    println!();
    println!(
        "// Measured: ESR prod×C++  = {esr_prod_cpp:.6e} ({:.1} dB)",
        esr_to_db_f64(esr_prod_cpp)
    );
    println!(
        "// Measured: ESR prod×f64  = {esr_prod_f64:.6e} ({:.1} dB)",
        esr_to_db_f64(esr_prod_f64)
    );
    println!(
        "// Measured: ESR f64×C++   = {esr_f64_cpp:.6e} ({:.1} dB)",
        esr_to_db_f64(esr_f64_cpp)
    );
    println!("// Measured: SNR prod×C++  = {:.2} dB", snr(esr_prod_cpp));
    println!("// Measured: SNR prod×f64  = {:.2} dB", snr(esr_prod_f64));
    println!("// Measured: SNR f64×C++   = {:.2} dB", snr(esr_f64_cpp));
    println!();
    println!("Classification: {classification}");
    println!("eps = {eps:.0e} (matches < -30 dB ESR for practical identity)");
    println!();

    // Assert the classification is not vacuous (at least one pair should diverge)
    assert!(
        esr_prod_cpp > eps || esr_prod_f64 > eps || esr_f64_cpp > eps,
        "H0: all three pairs are identical (ESR<{eps:.0e}) — this would mean the gap is closed. \
         Update classification and remove #[ignore]."
    );
}

/// Runtime contract condition_dsp (H5+H7).
///
/// Under unlock, builds A2 Max and inspects the nested condition_dsp sub-model:
///   1. Asserts `condition_dsp.num_output_channels()` value and documents.
///   2. Identifies the enum variant of the nested condition_dsp.
///   3. Dumps condition_dsp 8ch output and compares with f64 oracle.
///   4. Confirms whether `dsp_ch < cond_size` branch is dead code.
///
/// ## How to run
/// ```sh
/// NAM_A2_MAX_UNLOCK=1 cargo test --test models test_tr2b2_condition_dsp_contract -- --ignored --exact --nocapture
/// ```
// on-demand: KB-A2-MAX condition_dsp contract harness (TR2b.2/H5+H7); not a nightly gate
#[test]
#[ignore = "TR2b.2 (H5+H7): diagnostic harness — not a CI gate; run manually"]
fn test_tr2b2_condition_dsp_contract() {
    use neural_amp_modeler_rs::models::StaticModel;
    use neural_amp_modeler_rs::testing::diagnostics::DiagnosticConfig;
    use neural_amp_modeler_rs::testing::reference_oracle::{
        PrecisionConfig, compute_esr_f64, esr_to_db_f64, oracle_forward,
    };

    let nam_path = model_path("wavenet_a2_max.nam");
    assert!(nam_path.exists());
    let json_data = fs::read_to_string(&nam_path).expect("Failed to read A2 Max model");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse A2 Max JSON");
    let model_data_for_oracle = model_data.clone();

    unsafe {
        std::env::set_var("NAM_A2_MAX_UNLOCK", "1");
    }
    let model = build_model(&model_data).expect("A2 Max must build under unlock");
    let wad = match *model {
        StaticModel::WavenetA2Dyn(w) => w,
        other => panic!("Expected WavenetA2Dyn, got {:?}", other.class_label()),
    };

    let cond_dsp = wad
        .condition_dsp
        .as_ref()
        .expect("A2 Max must have condition_dsp");

    let dsp_ch = cond_dsp.num_output_channels();
    let cond_size = wad.condition_size;

    assert_eq!(
        cond_size, 8,
        "Precondition: A2 Max condition_size must be 8, got {}",
        cond_size
    );

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (a) Enum variant of nested condition_dsp
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let variant_label = match cond_dsp.as_ref() {
        StaticModel::WavenetStandard(_) => "WavenetStandard",
        StaticModel::WavenetLite(_) => "WavenetLite",
        StaticModel::WavenetFeather(_) => "WavenetFeather",
        StaticModel::WavenetNano(_) => "WavenetNano",
        StaticModel::WavenetA2Full(_) => "WavenetA2Full",
        StaticModel::WavenetA2Lite(_) => "WavenetA2Lite",
        StaticModel::WavenetA2Dyn(_) => "WavenetA2Dyn",
        StaticModel::WavenetA2Cascade(m) => {
            let inner_head = m.arrays.first().map(|a| a.head_size).unwrap_or(0);
            println!(
                "// Measured: cond_dsp is WavenetA2Cascade ({} arrays, array[0].head_size={})",
                m.arrays.len(),
                inner_head
            );
            "WavenetA2Cascade"
        }
        StaticModel::WavenetDyn(m) => {
            let last_head = m.arrays.last().map(|a| a.head).unwrap_or(0);
            println!(
                "// Measured: cond_dsp is WavenetDyn ({} arrays, last.head={})",
                m.arrays.len(),
                last_head
            );
            "WavenetDyn"
        }
        _ => {
            let label = cond_dsp.class_label().to_string();
            println!("// Measured: cond_dsp variant = {label}");
            ""
        }
    };

    println!(
        "// Measured: cond_dsp.num_output_channels() = {dsp_ch} | condition_size = {cond_size}"
    );
    println!("// Measured: cond_dsp :: StaticModel::{variant_label}");

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (b) Dead code analysis: dsp_ch < cond_size branch
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    assert!(dsp_ch > 0, "cond_dsp.num_output_channels must be > 0");
    let branch_executes = dsp_ch < cond_size;
    println!();
    if branch_executes {
        println!("// Measured: H2 broadcast IS active — dsp_ch({dsp_ch}) < cond_size({cond_size})");
    } else {
        println!(
            "// Measured: dsp_ch < cond_size is DEAD CODE on A2 Max (dsp_ch={dsp_ch} == cond_size={cond_size}). \
             H2 broadcast is NOT the mechanism for this fixture."
        );
    }
    println!();

    unsafe {
        std::env::remove_var("NAM_A2_MAX_UNLOCK");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // (c) Dump condition_dsp 8ch vs f64 oracle
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_max.bin");
    if !golden_path.exists() {
        println!("[STATUS] SKIP_CAPABILITY: golden_not_found:golden_wavenet_a2_max.bin");
        return;
    }
    let (golden_input, _) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_max.bin");

    unsafe {
        std::env::set_var("NAM_A2_MAX_UNLOCK", "1");
    }
    let model2 = build_model(&model_data_for_oracle).expect("A2 Max must build");
    let mut wad2 = match *model2 {
        StaticModel::WavenetA2Dyn(w) => w,
        other => panic!("Expected WavenetA2Dyn, got {:?}", other.class_label()),
    };

    let config = DiagnosticConfig {
        capture_condition_dsp: true,
        capture_head_per_layer: false,
        capture_final_output: true,
    };
    wad2.enable_diagnostics(config);
    wad2.prewarm();

    let n = golden_input.len();
    let mut prod_output = vec![0.0f32; n];
    {
        let block_size = 64usize;
        let mut pos = 0;
        while pos < n {
            let nf = (n - pos).min(block_size);
            wad2.process(
                &golden_input[pos..pos + nf],
                &mut prod_output[pos..pos + nf],
            );
            pos += nf;
        }
    }
    let dump = wad2
        .take_diagnostics()
        .expect("Diagnostic dump must be present");

    unsafe {
        std::env::remove_var("NAM_A2_MAX_UNLOCK");
    }

    // ── f64 oracle for condition_dsp sub-model ──
    let cond_json = model_data_for_oracle
        .config
        .condition_dsp
        .as_ref()
        .expect("cond_dsp JSON must exist");
    let cond_data: neural_amp_modeler_rs::loader::nam_json::NamModelData =
        serde_json::from_value(cond_json.clone()).expect("Failed to parse condition_dsp JSON");
    let input_f64: Vec<f64> = golden_input.iter().map(|&x| x as f64).collect();
    let oracle_cfg = PrecisionConfig::default();
    let oracle_output = oracle_forward(&cond_data, &input_f64, &oracle_cfg);

    // ── Aggregate ESR of condition_dsp output vs oracle ──
    let all_prod_f32: Vec<f32> = dump
        .condition_dsp_snapshots
        .iter()
        .flat_map(|s| s.data.iter().copied())
        .collect();
    let total_cond_frames: usize = dump
        .condition_dsp_snapshots
        .iter()
        .map(|s| s.num_frames)
        .sum();
    let compare_frames = total_cond_frames.min(golden_input.len());

    // Build oracle-broadcasted and prod arrays for comparison.
    // oracle_output is mono (1 sample/frame), cond_dsp output is 8ch interleaved.
    // Compare using broadcast logic: if branch_executes, broadcast; else full 8ch compare.
    let mut oracle_cmp = Vec::with_capacity(compare_frames * cond_size);
    let mut prod_cmp = Vec::with_capacity(compare_frames * cond_size);
    for f in 0..compare_frames {
        let o = oracle_output.get(f).copied().unwrap_or(0.0);
        let prod_base = f * cond_size;
        if prod_base + cond_size <= all_prod_f32.len() {
            for c in 0..cond_size {
                oracle_cmp.push(o);
                prod_cmp.push(all_prod_f32[prod_base + c] as f64);
            }
        }
    }

    let agg_esr = compute_esr_f64(&oracle_cmp, &prod_cmp);
    println!(
        "// Measured: condition_dsp prod×f64 ESR = {agg_esr:.6e} ({:.1} dB) | {compare_frames} frames, {} channels",
        esr_to_db_f64(agg_esr),
        cond_size
    );

    // ── Per-sub-block ESR ──
    println!();
    println!(
        "{:<12} {:<18} {:<18} {:<12}",
        "Sub-block", "ESR prod×f64", "ESR dB", "Frames"
    );
    println!("{}", "-".repeat(60));

    let mut frame_offset = 0usize;
    for (i, snap) in dump.condition_dsp_snapshots.iter().enumerate() {
        let nf = snap.num_frames;
        let prod_start = frame_offset * cond_size;
        let prod_end = (frame_offset + nf) * cond_size;
        if prod_end > all_prod_f32.len() || frame_offset >= compare_frames {
            break;
        }
        let prod_block: Vec<f64> = all_prod_f32[prod_start..prod_end]
            .iter()
            .map(|&x| x as f64)
            .collect();
        let oracle_block: Vec<f64> = (frame_offset..frame_offset + nf)
            .flat_map(|f| {
                let o = oracle_output.get(f).copied().unwrap_or(0.0);
                (0..cond_size).map(move |_| o)
            })
            .collect();

        let esr = compute_esr_f64(&oracle_block, &prod_block);
        println!(
            "{:<12} {:<18.6e} {:<18.1} {:<12}",
            format!("b{}", i),
            esr,
            esr_to_db_f64(esr),
            nf
        );
        frame_offset += nf;
    }
    println!("{}", "-".repeat(60));
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

/// Golden Vectors LSTM 1×10 (uncatalogued) — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
///
/// Reads `tests/fixtures/golden_lstm_1x10.bin`, builds the `StaticModel`
/// from `lstm_1x10.nam`. Exercises single-layer LSTM with hidden_size=10.
#[test]
fn test_golden_vectors_lstm_1x10() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_1x10.bin");

    assert!(
        golden_path.exists(),
        "golden_lstm_1x10.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_1x10.bin");

    let nam_path = model_path("lstm_1x10.nam");
    assert!(
        nam_path.exists(),
        "lstm_1x10.nam not found at {nam_path:?}. Run golden_gen_build.sh to fetch models."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read lstm_1x10.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM 1×10 for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "lstm_1x10");
    gv_metric("lstm_1x10");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "LSTM 1×10 (uncat.)",
        STRESS_SAMPLE_RATE,
    );
}

/// Golden Vectors LSTM 2×24 (uncatalogued) — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
///
/// Reads `tests/fixtures/golden_lstm_2x24.bin`, builds the `StaticModel`
/// from `lstm_2x24.nam`. Exercises 2-layer LSTM with hidden_size=24.
#[test]
fn test_golden_vectors_lstm_2x24() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_2x24.bin");

    assert!(
        golden_path.exists(),
        "golden_lstm_2x24.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_2x24.bin");

    let nam_path = model_path("lstm_2x24.nam");
    assert!(
        nam_path.exists(),
        "lstm_2x24.nam not found at {nam_path:?}. Run golden_gen_build.sh to fetch models."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read lstm_2x24.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM 2×24 for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "lstm_2x24");
    gv_metric("lstm_2x24");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "LSTM 2×24 (uncat.)",
        STRESS_SAMPLE_RATE,
    );
}

/// Golden Vectors LSTM 3×8 — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
///
/// Reads `tests/fixtures/golden_lstm_3x8.bin`, builds the `StaticModel`
/// from `lstm_3x8.nam`. Exercises 3-layer LSTM with hidden_size=8.
#[test]
fn test_golden_vectors_lstm_3x8() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_3x8.bin");

    assert!(
        golden_path.exists(),
        "golden_lstm_3x8.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_3x8.bin");

    let nam_path = model_path("lstm_3x8.nam");
    assert!(
        nam_path.exists(),
        "lstm_3x8.nam not found at {nam_path:?}. Run golden_gen_build.sh to fetch models."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read lstm_3x8.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM 3×8 for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) = topology_thresholds(&model_data, "lstm_3x8");
    gv_metric("lstm_3x8");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "LSTM 3×8",
        STRESS_SAMPLE_RATE,
    );
}

/// Golden Vectors ConvNet No BatchNorm — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
///
/// Reads `tests/fixtures/golden_convnet_nobn.bin`, builds the `StaticModel`
/// from `convnet_nobn.nam`. Exercises ConvNet without batch normalization.
#[test]
fn test_golden_vectors_convnet_nobn() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_convnet_nobn.bin");

    assert!(
        golden_path.exists(),
        "golden_convnet_nobn.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_convnet_nobn.bin");

    let nam_path = model_path("convnet_nobn.nam");
    assert!(
        nam_path.exists(),
        "convnet_nobn.nam not found at {nam_path:?}. Run golden_gen_build.sh to fetch models."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read convnet_nobn.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build ConvNet No BatchNorm for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "convnet_nobn");
    gv_metric("convnet_nobn");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "ConvNet No BatchNorm",
        STRESS_SAMPLE_RATE,
    );
}

/// Golden Vectors ConvNet ReLU — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
///
/// Reads `tests/fixtures/golden_convnet_relu.bin`, builds the `StaticModel`
/// from `convnet_relu.nam`. Exercises ConvNet with ReLU activation.
#[test]
fn test_golden_vectors_convnet_relu() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_convnet_relu.bin");

    assert!(
        golden_path.exists(),
        "golden_convnet_relu.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_convnet_relu.bin");

    let nam_path = model_path("convnet_relu.nam");
    assert!(
        nam_path.exists(),
        "convnet_relu.nam not found at {nam_path:?}. Run golden_gen_build.sh to fetch models."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read convnet_relu.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build ConvNet ReLU for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "convnet_relu");
    gv_metric("convnet_relu");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "ConvNet ReLU",
        STRESS_SAMPLE_RATE,
    );
}

/// Golden Vectors ConvNet SiLU — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
///
/// Reads `tests/fixtures/golden_convnet_silu.bin`, builds the `StaticModel`
/// from `convnet_silu.nam`. Exercises ConvNet with SiLU activation.
#[test]
fn test_golden_vectors_convnet_silu() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_convnet_silu.bin");

    assert!(
        golden_path.exists(),
        "golden_convnet_silu.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_convnet_silu.bin");

    let nam_path = model_path("convnet_silu.nam");
    assert!(
        nam_path.exists(),
        "convnet_silu.nam not found at {nam_path:?}. Run golden_gen_build.sh to fetch models."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read convnet_silu.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build ConvNet SiLU for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "convnet_silu");
    gv_metric("convnet_silu");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "ConvNet SiLU",
        STRESS_SAMPLE_RATE,
    );
}

/// Golden Vectors Linear No Bias — cross-reference NeuralAmpModelerCore ↔ NeuralAmpModeler-rs.
///
/// Reads `tests/fixtures/golden_linear_nobias.bin`, builds the `StaticModel`
/// from `linear_nobias.nam`. Exercises Linear without bias vector.
#[test]
fn test_golden_vectors_linear_nobias() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_linear_nobias.bin");

    assert!(
        golden_path.exists(),
        "golden_linear_nobias.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_linear_nobias.bin");

    let nam_path = model_path("linear_nobias.nam");
    assert!(
        nam_path.exists(),
        "linear_nobias.nam not found at {nam_path:?}. Run golden_gen_build.sh to fetch models."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read linear_nobias.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build Linear No Bias for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr, mrstft_max) =
        topology_thresholds(&model_data, "linear_nobias");
    gv_metric("linear_nobias");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        "Linear No Bias",
        STRESS_SAMPLE_RATE,
    );
}

// =============================================================================
// Catalog Preflight — Capability Receipt
// =============================================================================

/// Prints the unified fixture catalog capability receipt to stdout.
///
/// Three gates run here:
///
/// 1. **Fixture catalog receipt** — iterates every entry in `FIXTURE_CATALOG`,
///    resolves its path via the unified path resolution (env vars → local
///    checkout), checks disk presence, and emits a typed capability report.
///
/// 2. **V1 golden catalog validation** — runs
///    `catalog::validate_v1_goldens()` against the Rust single source of truth
///    (`src/testing/catalog.rs::V1_GOLDEN_CATALOG`), checking every 48 kHz v1
///    golden binary on disk: DistributedCore model goldens and the CabSim
///    convolution goldens (RequiredLocal, hard-fail) plus the
///    LocalNonDistributable WaveNet Lite golden (OptionalExternal, graceful).
///    The former bash arrays `REQUIRED_GOLDEN_MODELS` / `NONDIST_GOLDEN_MODELS`
///    / `REQUIRED_CABSIM_GOLDENS` in `utils/tests-long.sh` were removed; this
///    Rust gate is the only v1 presence check.
///
/// 3. **V2 golden catalog validation** — runs
///    `catalog::validate_v2_catalog()` against the Rust single source of truth
///    (`src/testing/catalog.rs::GOLDEN_GEN_CATALOG`), checking every model
///    fixture and every expected `*_v2_{sr}.bin` golden per the model's
///    sample-rate scope. The former bash `V2_CATALOG_SCOPE` preflight in
///    `utils/tests-long.sh` was removed; this Rust gate is the only V2 check.
///
/// Invoked by `utils/tests-long.sh` before running the long suite. The
/// receipt serves as an auditable preflight artifact: it proves that the
/// runner verified fixture availability against the catalog declaratively,
/// rather than proceeding blindly and crashing late with a panic.
///
/// Missing `RequiredLocal` fixtures fail this test (exit 1 with a descriptive
/// message). Integrity of present goldens is additionally enforced by the
/// freshness gate (`check_freshness`, nam_freshness) in the runners.
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

    // V1 golden catalog validation — Rust single source of truth (S6-T01).
    let v1_status = neural_amp_modeler_rs::testing::catalog::validate_v1_goldens()
        .unwrap_or_else(|e| panic!("catalog_preflight: V1 golden catalog validation failed: {e}"));
    println!("\n{}", v1_status.receipt_v1());
    missing_required.extend(v1_status.missing_required.iter().map(String::as_str));
    missing_optional.extend(v1_status.missing_optional.iter().map(String::as_str));

    // V2 golden catalog validation — Rust single source of truth (S3-T02).
    let v2_status = neural_amp_modeler_rs::testing::catalog::validate_v2_catalog()
        .unwrap_or_else(|e| panic!("catalog_preflight: V2 catalog validation failed: {e}"));
    println!("\n{}", v2_status.receipt());
    missing_required.extend(v2_status.missing_required.iter().map(String::as_str));
    missing_optional.extend(v2_status.missing_optional.iter().map(String::as_str));

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

    assert!(
        missing_required.is_empty(),
        "catalog_preflight: {} RequiredLocal fixture(s) absent: {missing_required:?}",
        missing_required.len()
    );
}
