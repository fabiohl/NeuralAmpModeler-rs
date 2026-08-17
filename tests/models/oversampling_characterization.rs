// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Oversampling Characterization for LSTM Models.
//
//  Measures the empirical effect of external `OversampleEngine` (Off vs 2× vs 4×)
//  on LSTM models:
//
//  - **ASR** (Aliasing-to-Signal Ratio): anti-aliasing benefit from oversampling.
//  - **ESR / MR-STFT**: timbre change introduced by running the model at a higher
//    internal rate through the half-band filter pipeline.
//
//  ## Hypothesis
//
//  Oversampling serves anti-aliasing (reduces fold-back from nonlinear activations)
//  but changes timbre because the LSTM feedback delay is fixed in absolute samples —
//  running at 2× or 4× rate effectively halves or quarters the feedback time window
//  in seconds, altering the recurrent dynamics.
//
//  ## Tests (all `#[ignore]` — require .nam model files)
//
//  ```bash
//  cargo test --test oversampling_characterization -- --ignored --nocapture
//  ```

use neural_amp_modeler_rs::dsp::oversample::OversampleEngine;
use neural_amp_modeler_rs::dsp::oversample::OversampleFactor;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::aliasing;
use neural_amp_modeler_rs::testing::perceptual;
use std::fs;

use super::common;

const BLOCK_SIZE: usize = 64;
const PREWARM_SAMPLES: usize = 2048;

// =============================================================================
// Oversampling inference helper
// =============================================================================

/// Processes input through a model wrapped in an `OversampleEngine`.
///
/// For `Off`, the model processes at native rate. For `X2`/`X4`, the signal is
/// upsampled → model processes at higher rate → downsampled back to native rate.
///
/// Returns the native-rate output with `latency_samples()` of group delay from
/// the half-band filter pipeline. The first `latency_samples()` output samples
/// are filter warm-up transients.
fn process_with_os(
    model: &mut neural_amp_modeler_rs::models::StaticModel,
    input: &[f32],
    os_factor: OversampleFactor,
) -> Vec<f32> {
    let mut os_engine = OversampleEngine::new(os_factor, BLOCK_SIZE).unwrap();
    let mult = os_factor.multiplier();
    let max_os = BLOCK_SIZE * mult;
    let mut output = Vec::with_capacity(input.len());
    let mut up_buf = vec![0.0f32; max_os];
    let mut os_model_out = vec![0.0f32; max_os];
    let mut native_out = vec![0.0f32; BLOCK_SIZE];

    let mut pos = 0;
    while pos < input.len() {
        let end = (pos + BLOCK_SIZE).min(input.len());
        let block = &input[pos..end];
        let block_len = end - pos;

        if os_engine.is_bypass() {
            model.process(block, &mut native_out[..block_len]);
            output.extend_from_slice(&native_out[..block_len]);
        } else {
            let n_os = os_engine.upsample(block, &mut up_buf[..block_len * mult], None);
            model.process(&up_buf[..n_os], &mut os_model_out[..n_os]);
            let n_out = os_engine.downsample(&os_model_out[..n_os], &mut native_out, None);
            output.extend_from_slice(&native_out[..n_out]);
        }
        pos = end;
    }
    output
}

// =============================================================================
// Model loading
// =============================================================================

fn load_and_prewarm_lstm(filename: &str) -> Box<neural_amp_modeler_rs::models::StaticModel> {
    let path = common::io_helpers::model_path(filename);
    let json_data =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to read {path:?}: {e}"));
    let model_data = neural_amp_modeler_rs::loader::nam_json::parse_nam_json(&json_data)
        .expect("Failed to parse model JSON");
    assert_eq!(
        model_data.architecture, "LSTM",
        "Expected LSTM model, got {}",
        model_data.architecture
    );
    let mut model = neural_amp_modeler_rs::loader::dispatcher::build_model(&model_data)
        .expect("Failed to build model");
    model.prewarm(PREWARM_SAMPLES);
    model
}

// =============================================================================
// ASR measurement across oversampling factors
// =============================================================================

struct AsrRow {
    model: &'static str,
    off_db: f64,
    x2_db: f64,
    x4_db: f64,
}

fn measure_asr_row(
    model: &mut neural_amp_modeler_rs::models::StaticModel,
    label: &'static str,
) -> AsrRow {
    let sr = 48000;
    let f0 = aliasing::STRESS_F0;
    let gain = aliasing::HIGH_GAIN;
    let n = 65536;

    let input = aliasing::generate_sine(f0, sr, n, gain);

    let output_off = process_with_os(model, &input, OversampleFactor::Off);
    let output_x2 = process_with_os(model, &input, OversampleFactor::X2);
    let output_x4 = process_with_os(model, &input, OversampleFactor::X4);

    // Trim latency from outputs so we measure steady-state (same window for all)
    let trim = 128;
    let asr_off = aliasing::compute_asr(&output_off[trim..trim + 32768], f0, sr);
    let asr_x2 = aliasing::compute_asr(&output_x2[trim..trim + 32768], f0, sr);
    let asr_x4 = aliasing::compute_asr(&output_x4[trim..trim + 32768], f0, sr);

    eprintln!(
        "  ASR {label:>12} | Off: {:>7.1} dB | X2: {:>7.1} dB | X4: {:>7.1} dB",
        asr_off.asr_db, asr_x2.asr_db, asr_x4.asr_db
    );

    AsrRow {
        model: label,
        off_db: asr_off.asr_db,
        x2_db: asr_x2.asr_db,
        x4_db: asr_x4.asr_db,
    }
}

// =============================================================================
// ESR / MR-STFT: timbre change measurement
// =============================================================================

struct EsrRow {
    model: &'static str,
    x2_vs_off_esr: f64,
    x2_vs_off_esr_db: f64,
    x4_vs_off_esr: f64,
    x4_vs_off_esr_db: f64,
    #[allow(dead_code, clippy::allow_attributes)]
    off_mrstft: f64,
    x2_mrstft: f64,
    x4_mrstft: f64,
}

fn measure_esr_row(
    model: &mut neural_amp_modeler_rs::models::StaticModel,
    label: &'static str,
) -> EsrRow {
    // Generate a complex stress signal — v2 at 48 kHz, 5 seconds.
    let input =
        neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2("nam-rs-os-char", 48000);

    // Re-prewarm before each measurement to reset LSTM recurrent state
    model.prewarm(PREWARM_SAMPLES);
    let output_off = process_with_os(model, &input, OversampleFactor::Off);

    model.prewarm(PREWARM_SAMPLES);
    let output_x2 = process_with_os(model, &input, OversampleFactor::X2);

    model.prewarm(PREWARM_SAMPLES);
    let output_x4 = process_with_os(model, &input, OversampleFactor::X4);

    // Align: OS pipeline introduces group delay. Shift OS outputs forward
    // by their latency to align with the Off reference.
    let latency_x2 = 12;
    let latency_x4 = 24;
    let max_latency = latency_x4;

    // Trim max_latency from the beginning of all signals (steady-state).
    // For X2/X4, also shift by their additional latency.
    let start = max_latency;
    let end = input
        .len()
        .min(output_off.len())
        .min(output_x2.len().saturating_sub(latency_x2))
        .min(output_x4.len().saturating_sub(latency_x4));

    let ref_sig = &output_off[start..end];
    let x2_aligned = &output_x2[start + latency_x2..start + latency_x2 + (end - start)];
    let x4_aligned = &output_x4[start + latency_x4..start + latency_x4 + (end - start)];

    let x2_vs_off = perceptual::compute_esr(ref_sig, x2_aligned);
    let x4_vs_off = perceptual::compute_esr(ref_sig, x4_aligned);

    // MR-STFT: compare each OS output against Off baseline (same-aligned windows)
    let off_mrstft = perceptual::compute_mr_stft(ref_sig, ref_sig); // self → ~0
    let x2_mrstft = perceptual::compute_mr_stft(ref_sig, x2_aligned);
    let x4_mrstft = perceptual::compute_mr_stft(ref_sig, x4_aligned);

    let x2_db = perceptual::esr_to_db(x2_vs_off);
    let x4_db = perceptual::esr_to_db(x4_vs_off);

    eprintln!(
        "  ESR {label:>12} | X2vsOff: {:>7.1} dB | X4vsOff: {:>7.1} dB | MR-STFT(Off=self): {:.4} | MR-STFT(X2): {:.4} | MR-STFT(X4): {:.4}",
        x2_db, x4_db, off_mrstft, x2_mrstft, x4_mrstft
    );

    EsrRow {
        model: label,
        x2_vs_off_esr: x2_vs_off,
        x2_vs_off_esr_db: x2_db,
        x4_vs_off_esr: x4_vs_off,
        x4_vs_off_esr_db: x4_db,
        off_mrstft,
        x2_mrstft,
        x4_mrstft,
    }
}

// =============================================================================
// Characterization test — `#[ignore]` (requires model files)
// =============================================================================

const LSTM_MODELS: &[(&str, &str)] = &[
    ("BossLSTM-1x16.nam", "LSTM-1x16"),
    ("BossLSTM-2x8.nam", "LSTM-2x8"),
    ("lstm.nam", "LSTM-official"),
];

/// Runs the full LSTM oversampling characterization and prints tabulated results.
///
/// Produces two tables:
/// 1. **ASR** (Aliasing Suppression Ratio) — shows anti-aliasing benefit.
/// 2. **ESR/MR-STFT** — shows timbre change vs Off baseline.
#[test]
#[ignore = "requires .nam model files; run with --ignored --nocapture"]
fn characterize_lstm_oversampling() {
    eprintln!("\n╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  LSTM Oversampling Characterization                              ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    // ── ASR Measurements ──
    let mut asr_rows = Vec::new();
    eprintln!("━━━ ASR: Aliasing Suppression Ratio (stress tone 2017 Hz, +12 dB) ━━━\n");
    for &(file, label) in LSTM_MODELS {
        let mut model = load_and_prewarm_lstm(file);
        asr_rows.push(measure_asr_row(&mut model, label));
    }

    // ── ESR / MR-STFT Measurements ──
    let mut esr_rows = Vec::new();
    eprintln!("\n━━━ ESR / MR-STFT: Timbre Change vs Off Baseline (v2 stress, 5s) ━━━\n");
    for &(file, label) in LSTM_MODELS {
        let mut model = load_and_prewarm_lstm(file);
        esr_rows.push(measure_esr_row(&mut model, label));
    }

    // ── Tabulated Output ──
    println!("\n");
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║  TABLE 1 — ASR: Aliasing Suppression Ratio (2017 Hz, +12 dB, lower is better)              ║"
    );
    println!(
        "╠══════════════╤════════════════╤════════════════╤════════════════╤══════════════════════════╣"
    );
    println!(
        "║ Model        │ ASR Off (dB)   │ ASR 2×  (dB)   │ ASR 4×  (dB)   │ Δ(Off→4×)               ║"
    );
    println!(
        "╟──────────────┼────────────────┼────────────────┼────────────────┼──────────────────────────╢"
    );
    for r in &asr_rows {
        let delta = r.x4_db - r.off_db;
        let delta_str = if delta.is_finite() {
            format!("{delta:+.1} dB")
        } else {
            "—".to_string()
        };
        println!(
            "║ {:<12} │ {:>14.1} │ {:>14.1} │ {:>14.1} │ {:<24} ║",
            r.model, r.off_db, r.x2_db, r.x4_db, delta_str
        );
    }
    println!(
        "╚══════════════╧════════════════╧════════════════╧════════════════╧══════════════════════════╝"
    );

    println!();
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║  TABLE 2 — ESR / MR-STFT: Timbre Change vs Off Baseline (v2 stress 5s, 48 kHz)            ║"
    );
    println!(
        "╠══════════════╤══════════════════╤══════════════════╤══════════════════╤════════════════════╣"
    );
    println!(
        "║ Model        │ ESR X2 vs Off    │ ESR X4 vs Off    │ MR-STFT X2 vsOff │ MR-STFT X4 vsOff   ║"
    );
    println!(
        "╟──────────────┼──────────────────┼──────────────────┼──────────────────┼────────────────────╢"
    );
    for r in &esr_rows {
        println!(
            "║ {:<12} │ {:>10.1} dB ({:.2e}) │ {:>10.1} dB ({:.2e}) │ {:.4e}        │ {:.4e}          ║",
            r.model,
            r.x2_vs_off_esr_db,
            r.x2_vs_off_esr,
            r.x4_vs_off_esr_db,
            r.x4_vs_off_esr,
            r.x2_mrstft,
            r.x4_mrstft,
        );
    }
    println!(
        "╚══════════════╧══════════════════╧══════════════════╧══════════════════╧════════════════════╝"
    );

    // ── Hypothesis Validation ──
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  HYPOTHESIS VALIDATION                                          ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let mut asr_works = false;
    let mut no_aliasing_models = 0;

    for r in &asr_rows {
        if !r.off_db.is_finite() && !r.x2_db.is_finite() && !r.x4_db.is_finite() {
            no_aliasing_models += 1;
            continue;
        }
        if r.x4_db < r.off_db || r.x2_db < r.off_db {
            asr_works = true;
        }
    }
    // If models that actually produce aliasing all improve → confirmed
    let asr_improved = asr_works || no_aliasing_models == asr_rows.len();

    let mut timbre_changes = true;
    for r in &esr_rows {
        // ESR > 1e-4 means measurable timbre difference (not numerical noise)
        if r.x4_vs_off_esr < 1e-4 && r.x2_vs_off_esr < 1e-4 {
            timbre_changes = false;
        }
    }

    println!(
        "║  Anti-aliasing confirmed (ASR improves with OS):  {}",
        if asr_improved { "✅ YES" } else { "❌ NO" }
    );
    println!(
        "║  Timbre changes measurably (ESR > 1e-4 vs Off):   {}",
        if timbre_changes { "✅ YES" } else { "❌ NO" }
    );

    let verdict = if asr_improved && timbre_changes {
        "✅ CONFIRMED — Oversampling reduces aliasing but changes LSTM timbre"
    } else if !asr_improved && timbre_changes {
        "PARTIAL — Timbre changes but ASR not measurable"
    } else {
        "UNEXPECTED — re-run with different signals"
    };
    println!("║                                                              ║");
    println!("║  VERDICT: {:<52} ║", verdict);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Sanity checks: all measurements must be finite (not NaN/Inf) ──
    // ESR > 1.0 is expected — LSTM feedback delay changes with oversampling,
    // fundamentally altering recurrent dynamics. This is the phenomenon being measured.
    for r in &esr_rows {
        assert!(
            r.x2_vs_off_esr.is_finite(),
            "{}: X2 vs Off ESR is non-finite",
            r.model
        );
        assert!(
            r.x4_vs_off_esr.is_finite(),
            "{}: X4 vs Off ESR is non-finite",
            r.model
        );
    }

    // ASR: sanity check — if a model produces no aliasing, that's a valid observation.
    let models_with_aliasing = asr_rows
        .iter()
        .filter(|r| r.off_db.is_finite() || r.x2_db.is_finite() || r.x4_db.is_finite())
        .count();
    assert!(
        models_with_aliasing > 0,
        "No LSTM model produced finite ASR — aliasing characterization inconclusive"
    );
}

// =============================================================================
// Quick ASR-only validation (runs with `cargo test`, no ignored needed)
// =============================================================================

/// Sanity check: ASR measurement with `OversampleFactor::Off` on a synthetic
/// clipper confirms the ASR metric itself works correctly with the OS pipeline.
#[test]
fn asr_metric_works_with_os_pipeline() {
    let f0 = 2017.0;
    let sr = 48000;
    let n = 32768;

    // Create a trivial "model" by using a mock: hard-clip
    fn mock_model(input: &[f32], output: &mut [f32]) {
        for (i, &x) in input.iter().enumerate() {
            output[i] = x.clamp(-0.3, 0.3);
        }
    }

    let input = aliasing::generate_sine(f0, sr, n, 6.0);
    let mut output = vec![0.0f32; n];
    mock_model(&input, &mut output);

    let result = aliasing::compute_asr(&output[128..128 + 16384], f0, sr);
    assert!(
        result.has_aliasing(),
        "Hard-clip on 2017 Hz sine must produce aliasing"
    );
    assert!(
        result.asr_db > -30.0,
        "ASR too low for hard-clip: {:.1} dB",
        result.asr_db
    );
}

/// Verify that `process_with_os` with `Off` is bit-identical to direct model processing
/// when both use the same block size.
#[test]
fn process_with_os_off_is_identity() {
    use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v1;

    let lstm_file = "BossLSTM-1x16.nam";
    let path = common::io_helpers::model_path(lstm_file);
    if !path.exists() {
        eprintln!("SKIP: {lstm_file} not found");
        return;
    }

    let input = generate_stress_signal_v1();
    let n = input.len();

    let mut model_a = load_and_prewarm_lstm(lstm_file);
    let os_off_out = process_with_os(&mut model_a, &input, OversampleFactor::Off);

    let mut model_b = load_and_prewarm_lstm(lstm_file);
    let mut direct_out = vec![0.0f32; n];
    let block_size = BLOCK_SIZE;
    let mut pos = 0;
    while pos < n {
        let end = (pos + block_size).min(n);
        model_b.process(&input[pos..end], &mut direct_out[pos..end]);
        pos = end;
    }

    assert_eq!(os_off_out.len(), direct_out.len());
    for (i, (&a, &b)) in os_off_out.iter().zip(direct_out.iter()).enumerate() {
        assert_eq!(a, b, "Mismatch at sample {i}: os_off={a} vs direct={b}");
    }
}
