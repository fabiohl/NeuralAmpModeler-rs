// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Investigation [T5.5]: LSTM activation precision — SNR gain analysis.
//
//  Measures the SNR gain from using exact `f32::tanh` (libm) vs the production
//  Padé [5,4] rational approximant in LSTM fused gates.  Runs the same golden
//  vector input through both paths and reports the SNR delta.
//
//  ## Axis-B split (contract [S2-T01])
//
//  - **Structural** (Phase 1, debug): model loading, geometry dispatch and
//    buffer allocation — cheap, deterministic, codegen-agnostic.
//  - **SNR oracles** (Phase 2, `--release` only): float measurements against
//    golden `.bin` vectors / the f64 reference oracle.  `tests-quick.sh`
//    Phase 1 filters them via
//    `--skip lstm_activation_precision::test_lstm_activation_precision_gain`
//    (substring skip also covers `..._gain_stress_v2`); Phase 2 runs the
//    whole module under release.

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::math::activations::{
    ActivationPrecision, set_thread_local_activation_precision,
};
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::perceptual::compute_snr_db;
use std::fs;
use std::path::PathBuf;

use super::common;
use common::*;

const GOLDEN_BLOCK_SIZE: usize = 64;

// ── Structural (Phase 1 — debug, Axis-B) ─────────────────────────────────────

/// Maps a dispatched [`StaticModel`] back to its compile-time LSTM profile name.
///
/// The static profiles encode the model geometry as const generics, so the
/// variant itself is the dimension validation.
fn lstm_variant_name(model: &neural_amp_modeler_rs::models::StaticModel) -> &'static str {
    match model {
        neural_amp_modeler_rs::models::StaticModel::Lstm1x3(_) => "Lstm1x3",
        neural_amp_modeler_rs::models::StaticModel::Lstm1x8(_) => "Lstm1x8",
        neural_amp_modeler_rs::models::StaticModel::Lstm1x12(_) => "Lstm1x12",
        neural_amp_modeler_rs::models::StaticModel::Lstm1x16(_) => "Lstm1x16",
        neural_amp_modeler_rs::models::StaticModel::Lstm1x24(_) => "Lstm1x24",
        neural_amp_modeler_rs::models::StaticModel::Lstm1x40(_) => "Lstm1x40",
        neural_amp_modeler_rs::models::StaticModel::Lstm2x8(_) => "Lstm2x8",
        neural_amp_modeler_rs::models::StaticModel::Lstm2x12(_) => "Lstm2x12",
        neural_amp_modeler_rs::models::StaticModel::Lstm2x16(_) => "Lstm2x16",
        neural_amp_modeler_rs::models::StaticModel::Lstm2x24(_) => "Lstm2x24",
        _ => "LstmDyn",
    }
}

/// [S2-T01] Axis-B structural LSTM checks (Phase 1 — debug).
///
/// Loads the three distributed LSTM fixtures (RequiredLocal), validates their
/// declared geometry (`architecture` / `num_layers` / `hidden_size`) against
/// the dispatcher's static profile table, and exercises prewarm + block
/// processing on zeroed input (allocation / buffer-integrity checks).
///
/// No float comparisons or SNR gates: codegen-agnostic by construction.
#[test]
fn test_lstm_activation_precision_structural() {
    const PROFILES: &[(&str, usize, usize, &str)] = &[
        ("BossLSTM-1x16.nam", 1, 16, "Lstm1x16"),
        ("BossLSTM-2x8.nam", 2, 8, "Lstm2x8"),
        ("lstm.nam", 1, 3, "Lstm1x3"),
    ];

    for &(filename, layers, hidden, profile) in PROFILES {
        let nam_path = model_path(filename);
        assert!(
            nam_path.exists(),
            "Structural LSTM check requires {filename} (RequiredLocal fixture) at {nam_path:?}"
        );

        let json_data = fs::read_to_string(&nam_path).expect("Failed to read model");
        let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

        // Dimension validation: JSON topology must match the expected profile.
        assert_eq!(model_data.architecture, "LSTM", "{filename} must be LSTM");
        assert_eq!(
            model_data.config.num_layers,
            Some(layers),
            "{filename} must declare {layers} LSTM layer(s)"
        );
        assert_eq!(
            model_data.config.hidden_size,
            Some(hidden),
            "{filename} must declare hidden size {hidden}"
        );

        let mut model = build_model(&model_data)
            .unwrap_or_else(|e| panic!("Dispatcher failed for {filename}: {e}"));
        assert_eq!(
            lstm_variant_name(&model),
            profile,
            "{filename} must dispatch to static profile {profile}"
        );

        // Allocation / buffer-integrity: prewarm + process zeroed blocks.
        model.prewarm(2048);
        for block_size in [1usize, 64, 1024] {
            let input = vec![0.0f32; block_size];
            let mut output = vec![f32::NAN; block_size];
            process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);
            assert_eq!(
                output.len(),
                block_size,
                "{filename}: output length mismatch"
            );
            assert!(
                output.iter().all(|s| s.is_finite()),
                "{filename}: zero-input block of {block_size} produced non-finite samples"
            );
        }
    }
}

// ── SNR oracles (Phase 2 — --release, Axis-B) ────────────────────────────────
// These float measurements are filtered from Phase 1 via
// `--skip lstm_activation_precision::test_lstm_activation_precision_gain`
// in `utils/tests-quick.sh` and run exclusively under `--release`.

#[derive(Debug)]
struct SnrResult {
    model_name: String,
    gain_db: f64,
}

impl SnrResult {
    fn verdict(&self) -> &'static str {
        if self.gain_db < 2.0 {
            "NEGLIGIBLE — FastMath is adequate"
        } else if self.gain_db < 6.0 {
            "MODEST — consider documenting tradeoff"
        } else {
            "SIGNIFICANT — higher precision warranted"
        }
    }
}

fn process_in_blocks_scalar(
    model: &mut neural_amp_modeler_rs::models::StaticModel,
    input: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let total = input.len();
    let mut pos = 0;
    while pos < total {
        let end = (pos + block_size).min(total);
        model.process_scalar(&input[pos..end], &mut output[pos..end]);
        pos = end;
    }
}

fn measure_lstm_snr(golden_path: &str, model_filename: &str, label: &str) -> (f64, f64) {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let golden_full = fixtures_dir.join(golden_path);

    if !golden_full.exists() {
        eprintln!("SKIP: {} not found at {:?}.", golden_path, golden_full);
        return (f64::NAN, f64::NAN);
    }

    let nam_path = model_path(model_filename);
    if !nam_path.exists() {
        eprintln!("SKIP: {} not found.", model_filename);
        return (f64::NAN, f64::NAN);
    }

    let (input, expected) =
        read_golden_bin(&golden_full).unwrap_or_else(|| panic!("Failed to read {}", golden_path));

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    // --- FastMath path (SIMD Padé tanh + minimax sigmoid) ---
    let mut model_fast = build_model(&model_data).expect("Dispatcher failed");
    model_fast.prewarm(2048);
    let mut output_fast = vec![0.0f32; input.len()];
    process_in_blocks(&mut model_fast, &input, &mut output_fast, GOLDEN_BLOCK_SIZE);
    let snr_fast = compute_snr_db(&expected, &output_fast);

    // --- Exact tanh path (f32::tanh via libm, scalar minimax sigmoid) ---
    let mut model_exact = build_model(&model_data).expect("Dispatcher failed");
    model_exact.prewarm(2048);
    let mut output_exact = vec![0.0f32; input.len()];
    process_in_blocks_scalar(
        &mut model_exact,
        &input,
        &mut output_exact,
        GOLDEN_BLOCK_SIZE,
    );
    let snr_exact = compute_snr_db(&expected, &output_exact);

    println!(
        "{label:>22}  SIMD(Padé): {snr_fast:6.1} dB  |  Scalar(libm): {snr_exact:6.1} dB  |  Δ={gain:+.1} dB",
        gain = snr_exact - snr_fast,
    );

    // S2.T6: forensic JSONL sink (kind `activation`) — human log unchanged.
    // Only finite pairs are reported; skips stay human-log-only.
    if snr_fast.is_finite() && snr_exact.is_finite() {
        common::report_activation(label, snr_fast, snr_exact);
    }

    (snr_fast, snr_exact)
}

/// [T5.5] Precision investigation: measure SNR gain of exact tanh vs FastMath Padé.
///
/// Runs the 3 LSTM golden vectors through both SIMD (FastMath) and scalar
/// (exact `f32::tanh`) paths, computing SNR against the C++ reference.
/// Reports per-model SNR gain and overall verdict.
///
/// Phase-2 oracle (Axis-B): float measurement against golden `.bin` — runs
/// exclusively under `--release`; filtered from Phase 1 via `--skip`.
#[test]
fn test_lstm_activation_precision_gain() {
    eprintln!();
    eprintln!("══════════════════════════════════════════════════════════════════");
    eprintln!("  [T5.5] LSTM Activation Precision Investigation");
    eprintln!("  Comparing FastMath (Padé tanh) vs Exact tanh (libm f32::tanh)");
    eprintln!("  SNR vs C++ golden vectors (NeuralAmpModelerCore)");
    eprintln!("══════════════════════════════════════════════════════════════════");

    let results = [
        measure_lstm_snr("golden_lstm_1x16.bin", "BossLSTM-1x16.nam", "LSTM 1×16"),
        measure_lstm_snr("golden_lstm_2x8.bin", "BossLSTM-2x8.nam", "LSTM 2×8"),
        measure_lstm_snr("golden_lstm_official.bin", "lstm.nam", "LSTM Official"),
    ];

    let valid: Vec<_> = results
        .iter()
        .filter(|(fast, exact)| fast.is_finite() && exact.is_finite())
        .collect();

    if valid.is_empty() {
        eprintln!("\n  All tests SKIPPED (golden files or .nam models missing).");
        return;
    }

    let parsed: Vec<SnrResult> = results
        .iter()
        .enumerate()
        .filter(|(_, (f, e))| f.is_finite() && e.is_finite())
        .map(|(i, &(snr_fastmath_db, snr_exact_tanh_db))| SnrResult {
            model_name: ["LSTM 1×16", "LSTM 2×8", "LSTM Official"][i].to_string(),
            gain_db: snr_exact_tanh_db - snr_fastmath_db,
        })
        .collect();

    let min_gain = parsed
        .iter()
        .map(|r| r.gain_db)
        .fold(f64::INFINITY, f64::min);
    let max_gain = parsed
        .iter()
        .map(|r| r.gain_db)
        .fold(f64::NEG_INFINITY, f64::max);
    let avg_gain = parsed.iter().map(|r| r.gain_db).sum::<f64>() / parsed.len() as f64;

    eprintln!();
    eprintln!("  ────────────────────────────────────────────────────────────────");
    eprintln!("  Summary:");
    for r in &parsed {
        eprintln!(
            "    {:<16}  SNR gain: {:+5.1} dB  →  {}",
            r.model_name,
            r.gain_db,
            r.verdict()
        );
    }
    let separator = "-".repeat(16);
    eprintln!(
        "    {separator}  SNR gain range: [{:+.1}, {:.1}] dB  avg: {:.1} dB",
        min_gain, max_gain, avg_gain,
    );
    eprintln!("  ────────────────────────────────────────────────────────────────");

    if avg_gain < 3.0 {
        eprintln!(
            "  VERDICT: Fast Padé [5,4] is adequate for LSTM goldens.\n\
               The SNR gain from exact tanh is negligible (< 3 dB avg).\n\
               Note: `Standard` (exact-grade) is the universal production\n\
               default; `Fast` remains available as an explicit opt-in."
        );
    } else if avg_gain < 6.0 {
        eprintln!(
            "  VERDICT: Modest SNR gain detected ({} dB avg).\n\
               Consider an optional high-precision LSTM feature flag\n\
               for users who prioritize accuracy over speed.",
            avg_gain
        );
    } else {
        eprintln!(
            "  VERDICT: Significant SNR gain ({} dB avg).\n\
               A higher-precision LSTM tanh kernel is warranted.",
            avg_gain
        );
    }
    eprintln!();
}

/// [T5.5] Precision investigation: measure SNR gain of exact-grade `Standard` vs `Fast` Padé on full stress v2.
///
/// Runs the 3 LSTM models through both SIMD (`Fast`) and SIMD (`Standard`)
/// paths, computing SNR against the f64 exact reference oracle on the stress v2 signal.
///
/// Phase-2 oracle (Axis-B): float measurement against the f64 reference
/// oracle — runs exclusively under `--release`; filtered from Phase 1 via
/// `--skip` (substring skip of `test_lstm_activation_precision_gain`).
#[test]
fn test_lstm_activation_precision_gain_stress_v2() {
    eprintln!();
    eprintln!("══════════════════════════════════════════════════════════════════");
    eprintln!("  [T5.5] LSTM Activation Precision Investigation (STRESS V2)");
    eprintln!("  Comparing Fast (Padé tanh) vs Standard (exp-based, exact-grade)");
    eprintln!("  SNR vs f64 Exact Reference Oracle");
    eprintln!("══════════════════════════════════════════════════════════════════");

    // Make sure we restore Standard precision in case of panic
    let _restore = PrecisionGuard::new(ActivationPrecision::Standard);

    let results = [
        measure_lstm_snr_stress_v2("BossLSTM-1x16.nam", "LSTM 1×16"),
        measure_lstm_snr_stress_v2("BossLSTM-2x8.nam", "LSTM 2×8"),
        measure_lstm_snr_stress_v2("lstm.nam", "LSTM Official"),
    ];

    let valid: Vec<_> = results
        .iter()
        .filter(|(fast, exact)| fast.is_finite() && exact.is_finite())
        .collect();

    if valid.is_empty() {
        eprintln!("\n  All tests SKIPPED (models missing).");
        return;
    }

    let parsed: Vec<SnrResult> = results
        .iter()
        .enumerate()
        .filter(|(_, (f, e))| f.is_finite() && e.is_finite())
        .map(|(i, &(snr_fastmath_db, snr_exact_tanh_db))| SnrResult {
            model_name: ["LSTM 1×16", "LSTM 2×8", "LSTM Official"][i].to_string(),
            gain_db: snr_exact_tanh_db - snr_fastmath_db,
        })
        .collect();

    let min_gain = parsed
        .iter()
        .map(|r| r.gain_db)
        .fold(f64::INFINITY, f64::min);
    let max_gain = parsed
        .iter()
        .map(|r| r.gain_db)
        .fold(f64::NEG_INFINITY, f64::max);
    let avg_gain = parsed.iter().map(|r| r.gain_db).sum::<f64>() / parsed.len() as f64;

    eprintln!();
    eprintln!("  ────────────────────────────────────────────────────────────────");
    eprintln!("  Summary (Stress V2):");
    for r in &parsed {
        eprintln!(
            "    {:<16}  SNR gain: {:+5.1} dB  →  {}",
            r.model_name,
            r.gain_db,
            r.verdict()
        );
    }
    let separator = "-".repeat(16);
    eprintln!(
        "    {separator}  SNR gain range: [{:+.1}, {:.1}] dB  avg: {:.1} dB",
        min_gain, max_gain, avg_gain,
    );
    eprintln!("  ────────────────────────────────────────────────────────────────");

    if avg_gain < 3.0 {
        eprintln!(
            "  VERDICT: Fast Padé [5,4] is adequate for LSTM.\n\
               The SNR gain from exact tanh is negligible (< 3 dB avg).\n\
               Note: `Standard` (exact-grade) is the universal production\n\
               default; `Fast` remains available as an explicit opt-in."
        );
    } else if avg_gain < 6.0 {
        eprintln!(
            "  VERDICT: Modest SNR gain detected ({} dB avg).\n\
               Consider an optional high-precision LSTM feature flag\n\
               for users who prioritize accuracy over speed.",
            avg_gain
        );
    } else {
        eprintln!(
            "  VERDICT: Significant SNR gain ({} dB avg).\n\
               A higher-precision LSTM tanh kernel is warranted.",
            avg_gain
        );
    }
    eprintln!();
}

fn measure_lstm_snr_stress_v2(model_filename: &str, label: &str) -> (f64, f64) {
    use neural_amp_modeler_rs::testing::reference_oracle::{PrecisionConfig, oracle_forward};
    use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2_default;

    let nam_path = model_path(model_filename);
    if !nam_path.exists() {
        eprintln!("SKIP: {} not found.", model_filename);
        return (f64::NAN, f64::NAN);
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    let stress_signal = generate_stress_signal_v2_default(48000); // 240k samples
    let stress_f64: Vec<f64> = stress_signal.iter().map(|&x| x as f64).collect();

    // Compute expected output via f64 reference oracle
    let oracle_out = oracle_forward(&model_data, &stress_f64, &PrecisionConfig::default());
    let expected: Vec<f32> = oracle_out.iter().map(|&x| x as f32).collect();

    // --- Fast path (SIMD Padé tanh + minimax sigmoid) ---
    let mut model_fast = build_model(&model_data).expect("Dispatcher failed");
    let mut output_fast = vec![0.0f32; stress_signal.len()];
    let mut pos = 0;
    {
        let _guard = set_thread_local_activation_precision(Some(ActivationPrecision::Fast));
        while pos < stress_signal.len() {
            let nf = (stress_signal.len() - pos).min(64);
            model_fast.process(
                &stress_signal[pos..pos + nf],
                &mut output_fast[pos..pos + nf],
            );
            pos += nf;
        }
    }
    let snr_fast = compute_snr_db(&expected, &output_fast);

    // --- Standard path (SIMD exact-grade polynomial tanh + sigmoid) ---
    let mut model_exact = build_model(&model_data).expect("Dispatcher failed");
    let mut output_exact = vec![0.0f32; stress_signal.len()];
    pos = 0;
    {
        let _guard = set_thread_local_activation_precision(Some(ActivationPrecision::Standard));
        while pos < stress_signal.len() {
            let nf = (stress_signal.len() - pos).min(64);
            model_exact.process(
                &stress_signal[pos..pos + nf],
                &mut output_exact[pos..pos + nf],
            );
            pos += nf;
        }
    }
    let snr_exact = compute_snr_db(&expected, &output_exact);

    println!(
        "{label:>22}  Fast(Padé): {snr_fast:6.1} dB  |  Standard(exact): {snr_exact:6.1} dB  |  Δ={gain:+.1} dB",
        gain = snr_exact - snr_fast,
    );

    // S2.T6: forensic JSONL sink (kind `activation`) — human log unchanged.
    if snr_fast.is_finite() && snr_exact.is_finite() {
        common::report_activation(label, snr_fast, snr_exact);
    }

    (snr_fast, snr_exact)
}
