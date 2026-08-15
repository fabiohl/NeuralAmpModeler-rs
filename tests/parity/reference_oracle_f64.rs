// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Reference oracle f64 — ESR measurement and source decomposition.
//
//  Runs the f64 oracle against production f32 output for all model families,
//  reporting the absolute error floor and isolating contributions from
//  weight quantization, activation approximation, and accumulation precision.
//
//  ## External anchoring
//
//  The Rust f64 oracle is validated against a PyTorch/NumPy f64 reference
//  (pre-generated anchors in `tests/fixtures/f64_anchors/`). The oracle
//  passes with ESR < 1e-12 for all three model families, proving it is
//  a correct ground-truth reference.
//
//  ## Caveat
//  Production models use f16c-quantized weights (LSTM, WaveNet DenseLayers).
//  The oracle uses full f32→f64 precision weights. The ESR(f32 vs f64) thus
//  includes the quantization error, which is often the dominant term.

use std::io::Read;
use std::path::PathBuf;

use super::common;
use common::A2_ESR_LIMIT;
use common::A2_FILM_ESR_LIMIT;
use common::CONVNET_ESR_LIMIT;
use common::LSTM_1X16_DRIFT_LEGACY_ESR_LIMIT;
use common::LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT;
use common::LSTM_2X8_DRIFT_PAIRED_ESR_LIMIT;
use common::LSTM_ESR_LIMIT;
use common::PrecisionGuard;
use common::WAVENET_ESR_LIMIT;

use neural_amp_modeler_rs::loader::nam_json::model::NamModelData;
use neural_amp_modeler_rs::loader::nam_json::parse::parse_nam_json;
use neural_amp_modeler_rs::math::activations::{ActivationPrecision, set_activation_tls};
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::reference_oracle::{
    AccumulationMode, ActivationMode, PrecisionConfig, WeightPrecision, compute_esr_f64,
    esr_to_db_f64, oracle_forward, run_decomposition,
};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("models")
}

fn models_dir_nondist() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("models-nondist")
}

fn resolve_model_path(model_filename: &str) -> PathBuf {
    let path = models_dir().join(model_filename);
    if path.exists() {
        return path;
    }
    let nondist = models_dir_nondist().join(model_filename);
    if nondist.exists() {
        return nondist;
    }
    let community =
        neural_amp_modeler_rs::testing::fixtures::community_models_dir().join(model_filename);
    if community.exists() {
        return community;
    }
    path
}

fn anchors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("f64_anchors")
}

fn load_f64_binary(path: &PathBuf) -> Vec<f64> {
    // Anchor regeneration policy: docs/cpp_parity_map.md §4.5.1.
    // Anchors may only be regenerated when a C++ golden exists and
    // test_summary_table passes before regeneration, OR with explicit
    // human review documented in the commit. Regenerating from the
    // oracle itself is circular and does not prove correctness.
    let mut f = std::fs::File::open(path).expect("Failed to open f64 binary");
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf).expect("Failed to read count");
    let n = u32::from_le_bytes(buf) as usize;
    let mut data = Vec::with_capacity(n);
    let mut sample_buf = [0u8; 8];
    for _ in 0..n {
        f.read_exact(&mut sample_buf)
            .expect("Failed to read sample");
        data.push(f64::from_le_bytes(sample_buf));
    }
    data
}

fn load_and_parse(path: &PathBuf) -> NamModelData {
    let json = std::fs::read_to_string(path).expect("Failed to read model");
    parse_nam_json(&json).expect("Failed to parse NAM JSON")
}

fn run_f32_inference(model_data: &NamModelData, input: &[f32]) -> Vec<f32> {
    let mut model = neural_amp_modeler_rs::loader::dispatcher::build_model(model_data)
        .expect("Failed to build model");
    model.prewarm(2048);
    let total = input.len();
    let mut output = vec![0.0f32; total];
    let mut pos = 0;
    while pos < total {
        let nf = (total - pos).min(64);
        model.process(&input[pos..pos + nf], &mut output[pos..pos + nf]);
        pos += nf;
    }
    output
}

fn gen_sweep(n: usize, sample_rate: f64) -> Vec<f64> {
    let mut sig = Vec::with_capacity(n);
    let f0 = 220.0;
    let f1 = 880.0;
    let duration = n as f64 / sample_rate;
    for i in 0..n {
        let t = i as f64 / sample_rate;
        let phase = 2.0 * std::f64::consts::PI * (f0 * t + 0.5 * (f1 - f0) * t * t / duration);
        sig.push(phase.sin() * 0.3);
    }
    sig
}

fn print_decomposition(
    result: &neural_amp_modeler_rs::testing::reference_oracle::DecompositionResult,
) {
    let db = |v: f64| -> f64 {
        if v <= 0.0 {
            f64::NEG_INFINITY
        } else {
            10.0 * v.log10()
        }
    };
    println!(
        "{} Decomposition:\n\
           ESR(f32 vs f64 oracle):  {:.2e} ({:.1} dB)\n\
           ΔESR f16c weights:       {:.2e} ({:.1} dB)\n\
           ΔESR bf16 weights:       {:.2e} ({:.1} dB)\n\
           ΔESR Pade activation:    {:.2e} ({:.1} dB)\n\
           ΔESR f32 accumulation:   {:.2e} ({:.1} dB)\n\
           ΔESR combined (F16C+Padé+F32): {:.2e} ({:.1} dB)",
        result.label,
        result.esr_f32_vs_f64,
        db(result.esr_f32_vs_f64),
        result.esr_quant_f16c_display(),
        db(result.esr_quant_f16c_display()),
        result.esr_quant_bf16_display(),
        db(result.esr_quant_bf16_display()),
        result.esr_activation_display(),
        db(result.esr_activation_display()),
        result.esr_accumulation_display(),
        db(result.esr_accumulation_display()),
        result.esr_combined_display(),
        db(result.esr_combined_display()),
    );
}

// ── External anchor validation ─────────────────────────────────────

/// Anchor validation: Rust oracle vs independent NumPy f64 reference.
/// The pre-generated anchors are in tests/fixtures/f64_anchors/.
/// Format: [u32 LE count] [f64 LE × count]
///
/// MAINTENANCE CONTRACT (Gate Calibration Policy Rule 6, docs/perceptual_validation.md):
/// any change to `src/testing/reference_oracle.rs` invalidates these anchors.
/// In the SAME change set, regenerate them with
/// `tests/fixtures/scripts/validate_oracle_f64.py` and confirm the script's
/// output also matches the f32 production engine (≈ f32/f16c floor) — proving
/// the reference is independently correct, not merely a mirror of the oracle.
/// Never leave these tests `#[ignore]`d without a tracked restoration task.

#[test]
fn test_oracle_vs_python_anchor_wavenet() {
    let path = models_dir().join("wavenet_official.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("wavenet_official_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "WaveNet Official: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "WaveNet Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

#[test]
fn test_oracle_vs_python_anchor_condition_dsp() {
    let path = models_dir().join("wavenet_condition_dsp.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("wavenet_condition_dsp_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "Condition DSP: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "Condition DSP Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

#[test]
fn test_oracle_vs_python_anchor_condition_lstm() {
    let path = models_dir().join("wavenet_condition_lstm.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("wavenet_condition_lstm_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "Condition DSP LSTM: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "Condition DSP LSTM Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

#[test]
fn test_oracle_vs_python_anchor_lstm() {
    let path = models_dir().join("lstm.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("lstm_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "LSTM H=3: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "LSTM Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

#[test]
fn test_oracle_vs_python_anchor_a2() {
    let path = models_dir().join("wavenet_a2_lite.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("a2_lite_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "A2 Lite: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "A2 Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

// ── FiLM A2 Python anchor validation ─────────────────────────────

#[test]
fn test_oracle_vs_python_anchor_a2_film_lite() {
    let path = models_dir().join("wavenet_a2_film_lite.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("wavenet_a2_film_lite_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "A2-FiLM-Lite: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "A2-FiLM-Lite Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

#[test]
fn test_oracle_vs_python_anchor_a2_film_full() {
    let path = models_dir().join("wavenet_a2_film_full.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("wavenet_a2_film_full_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "A2-FiLM-Full: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "A2-FiLM-Full Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

#[test]
fn test_oracle_vs_python_anchor_a2_film_input_mixin_pre() {
    let path = models_dir().join("wavenet_a2_film_input_mixin_pre.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor =
        load_f64_binary(&anchors_dir().join("wavenet_a2_film_input_mixin_pre_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "A2-FiLM-InputMixinPre: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "A2-FiLM-InputMixinPre Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

#[test]
fn test_oracle_vs_python_anchor_a2_gated() {
    let path = models_dir().join("a2_dynamic_gated_ch8.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("a2_dynamic_gated_ch8_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "A2-Dyn-Gated: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-15,
        "A2-Dyn-Gated Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

#[test]
fn test_oracle_vs_python_anchor_a2_blended() {
    let path = models_dir().join("a2_dynamic_blended_ch3.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("a2_dynamic_blended_ch3_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "A2-Dyn-Blended: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-15,
        "A2-Dyn-Blended Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

// ── Re-derived fidelity gates (prewarm-paired) ──────────
// Oracle ESR limits are defined in tests/common/constants.rs as pub const
// (WAVENET_ESR_LIMIT, LSTM_ESR_LIMIT, A2_ESR_LIMIT) and shared between
// reference_oracle_f64.rs and threshold_calibration.rs for cross-test access.
// All gates comply with the Gate Calibration Policy in docs/perceptual_validation.md
// (Rules 1–5: validated reference, below placebo, provenance comment, link to
// independent measurement, sanity-check Σ sources ≈ total).

// ── Prewarm-paired ESR gate tests ────────────────────────────────────
// Measures ESR(f32 production vs f64 ideal oracle) with paired prewarm
// (24k warmup + 256 measurement sweep @ 48 kHz) to eliminate transient-
// mismatch from cold-start history buffers. Limits = measured × 2.

/// Runs a prewarm-paired ESR measurement: feeds both production (f32) and the
/// f64 ideal oracle with 24k warmup + 256 measurement samples, then computes
/// ESR on the post-prewarm window only.
fn run_oracle_esr_paired(model_filename: &str, label: &str) -> f64 {
    let path = resolve_model_path(model_filename);
    let md = load_and_parse(&path);
    let total = 24_000 + 256;
    let input_f64 = gen_sweep(total, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();

    let mut model =
        neural_amp_modeler_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
    println!("MODEL CLASS LABEL: {}", model.class_label());
    let mut prod_output = vec![0.0f32; input_f32.len()];
    let mut pos = 0;
    while pos < input_f32.len() {
        let nf = (input_f32.len() - pos).min(64);
        model.process(&input_f32[pos..pos + nf], &mut prod_output[pos..pos + nf]);
        pos += nf;
    }

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());

    let prod_last_f64: Vec<f64> = prod_output[24_000..].iter().map(|&x| x as f64).collect();
    let oracle_last = &oracle[24_000..];

    let esr = compute_esr_f64(oracle_last, &prod_last_f64);
    let verbose = std::env::var("NAM_ORACLE_VERBOSE").is_ok();
    if verbose {
        println!(
            "PROD FIRST 10: {:?}",
            &prod_last_f64[..10.min(prod_last_f64.len())]
        );
        println!(
            "ORACLE FIRST 10: {:?}",
            &oracle_last[..10.min(oracle_last.len())]
        );
    }
    println!(
        "{} ESR(f32 vs oracle, prewarm-paired, last 256 of 24k+256): {:.2e} ({:.1} dB)",
        label,
        esr,
        esr_to_db_f64(esr)
    );
    esr
}

#[test]
fn test_oracle_wavenet() {
    let esr = run_oracle_esr_paired("wavenet_official.nam", "WaveNet");
    assert!(
        esr < WAVENET_ESR_LIMIT,
        "WaveNet ESR={:.6e} exceeds calibrated limit {}",
        esr,
        WAVENET_ESR_LIMIT
    );
}

#[test]
fn test_oracle_lstm() {
    let esr = run_oracle_esr_paired("lstm.nam", "LSTM");
    assert!(
        esr < LSTM_ESR_LIMIT,
        "LSTM ESR={:.6e} exceeds calibrated limit {}",
        esr,
        LSTM_ESR_LIMIT
    );
}

#[test]
fn test_oracle_a2() {
    let esr = run_oracle_esr_paired("wavenet_a2_lite.nam", "A2");
    assert!(
        esr < A2_ESR_LIMIT,
        "A2 ESR={:.6e} exceeds calibrated limit {}",
        esr,
        A2_ESR_LIMIT
    );
}

// ── Decomposition tests ────────────────────────────────────────────────────
// Gating assertion uses prewarm-paired ESR; decomposition breakdown
// is diagnostic (cold-start run_decomposition prints per-component ΔESR).

#[test]
fn test_decomposition_wavenet() {
    use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2_default;

    let path = models_dir().join("wavenet_official.nam");
    let md = load_and_parse(&path);

    const WARMUP_LEN: usize = 24_000;
    const MEASURE_LEN: usize = 4_096;
    let total = WARMUP_LEN + MEASURE_LEN;

    let stress_signal = generate_stress_signal_v2_default(48000);
    assert!(
        stress_signal.len() >= total,
        "Stress signal is too short ({} < {})",
        stress_signal.len(),
        total
    );
    let input_f32 = &stress_signal[0..total];
    let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();

    let mut model =
        neural_amp_modeler_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
    let mut prod_output = vec![0.0f32; total];
    let mut pos = 0;
    {
        let _guard =
            neural_amp_modeler_rs::math::activations::set_thread_local_activation_precision(Some(
                neural_amp_modeler_rs::math::activations::ActivationPrecision::Fast,
            ));
        while pos < total {
            let nf = (total - pos).min(64);
            model.process(&input_f32[pos..pos + nf], &mut prod_output[pos..pos + nf]);
            pos += nf;
        }
    }
    let prod_output_f64: Vec<f64> = prod_output.iter().map(|&x| x as f64).collect();

    let result = run_decomposition_paired(
        "WaveNet-official",
        "WaveNet",
        &md,
        &prod_output_f64,
        &input_f64,
        WARMUP_LEN,
    );
    print_decomposition(&result);

    let sum_sources = result.esr_quant_f16c_display()
        + result.esr_activation_display()
        + result.esr_accumulation_display();
    let total_esr = result.esr_f32_vs_f64;
    let ratio = if sum_sources > 0.0 {
        total_esr / sum_sources
    } else {
        0.0
    };
    let inverse_ratio = if total_esr > 0.0 {
        sum_sources / total_esr
    } else {
        0.0
    };

    println!(
        "WaveNet Rule 5: Total ESR = {:.6e}, Sum of sources = {:.6e}, Ratio = {:.2}, InvRatio = {:.2}",
        total_esr, sum_sources, ratio, inverse_ratio
    );

    assert!(
        result.esr_combined_display() > 0.0,
        "Combined ΔESR should be non-zero"
    );
}

#[test]
fn test_decomposition_lstm() {
    use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2_default;

    let path = models_dir().join("lstm.nam");
    let md = load_and_parse(&path);

    const WARMUP_LEN: usize = 24_000;
    const MEASURE_LEN: usize = 4_096;
    let total = WARMUP_LEN + MEASURE_LEN;

    let stress_signal = generate_stress_signal_v2_default(48000);
    assert!(
        stress_signal.len() >= total,
        "Stress signal is too short ({} < {})",
        stress_signal.len(),
        total
    );
    let input_f32 = &stress_signal[0..total];
    let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();

    let mut model =
        neural_amp_modeler_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
    let mut prod_output = vec![0.0f32; total];
    let mut pos = 0;
    {
        let _guard =
            neural_amp_modeler_rs::math::activations::set_thread_local_activation_precision(Some(
                neural_amp_modeler_rs::math::activations::ActivationPrecision::Fast,
            ));
        while pos < total {
            let nf = (total - pos).min(64);
            model.process(&input_f32[pos..pos + nf], &mut prod_output[pos..pos + nf]);
            pos += nf;
        }
    }
    let prod_output_f64: Vec<f64> = prod_output.iter().map(|&x| x as f64).collect();

    let result = run_decomposition_paired(
        "LSTM-H3",
        "LSTM",
        &md,
        &prod_output_f64,
        &input_f64,
        WARMUP_LEN,
    );
    print_decomposition(&result);

    let sum_sources = result.esr_quant_f16c_display()
        + result.esr_activation_display()
        + result.esr_accumulation_display();
    let total_esr = result.esr_f32_vs_f64;
    let ratio = if sum_sources > 0.0 {
        total_esr / sum_sources
    } else {
        0.0
    };
    let inverse_ratio = if total_esr > 0.0 {
        sum_sources / total_esr
    } else {
        0.0
    };

    println!(
        "LSTM Rule 5: Total ESR = {:.6e}, Sum of sources = {:.6e}, Ratio = {:.2}, InvRatio = {:.2}",
        total_esr, sum_sources, ratio, inverse_ratio
    );

    assert!(
        result.esr_combined_display() > 0.0,
        "Combined ΔESR should be non-zero"
    );
}

fn run_decomposition_paired(
    label: &str,
    architecture: &str,
    model_data: &NamModelData,
    production_output: &[f64],
    input_signal: &[f64],
    warmup_len: usize,
) -> neural_amp_modeler_rs::testing::reference_oracle::DecompositionResult {
    let oracle_cfg = PrecisionConfig::default();

    let oracle_out = oracle_forward(model_data, input_signal, &oracle_cfg);
    let oracle_paired = &oracle_out[warmup_len..];
    let prod_paired = &production_output[warmup_len..];
    let esr_f32_vs_f64 = compute_esr_f64(oracle_paired, prod_paired);

    let mut cfg_f16c = oracle_cfg;
    cfg_f16c.weight_precision = WeightPrecision::F16C;
    let out_f16c = oracle_forward(model_data, input_signal, &cfg_f16c);
    let esr_f16c = compute_esr_f64(oracle_paired, &out_f16c[warmup_len..]);

    let mut cfg_bf16 = oracle_cfg;
    cfg_bf16.weight_precision = WeightPrecision::BF16;
    let out_bf16 = oracle_forward(model_data, input_signal, &cfg_bf16);
    let esr_bf16 = compute_esr_f64(oracle_paired, &out_bf16[warmup_len..]);

    let mut cfg_act = oracle_cfg;
    cfg_act.activation = ActivationMode::PadeMinimax;
    let out_act = oracle_forward(model_data, input_signal, &cfg_act);
    let esr_act = compute_esr_f64(oracle_paired, &out_act[warmup_len..]);

    let mut cfg_acc = oracle_cfg;
    cfg_acc.accumulation = AccumulationMode::F32Plain;
    let out_acc = oracle_forward(model_data, input_signal, &cfg_acc);
    let esr_acc = compute_esr_f64(oracle_paired, &out_acc[warmup_len..]);

    let combined_cfg = PrecisionConfig {
        weight_precision: WeightPrecision::F16C,
        activation: ActivationMode::PadeMinimax,
        accumulation: AccumulationMode::F32Plain,
    };
    let out_combined = oracle_forward(model_data, input_signal, &combined_cfg);
    let esr_combined = compute_esr_f64(oracle_paired, &out_combined[warmup_len..]);

    neural_amp_modeler_rs::testing::reference_oracle::DecompositionResult {
        label: label.to_string(),
        architecture: architecture.to_string(),
        esr_f32_vs_f64,
        esr_quant_f16c: Some(esr_f16c),
        esr_quant_bf16: Some(esr_bf16),
        esr_activation: Some(esr_act),
        esr_accumulation: Some(esr_acc),
        esr_combined: Some(esr_combined),
    }
}

#[test]
fn test_decomposition_boss_lstm_1x16() {
    use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2_default;

    let path = models_dir().join("BossLSTM-1x16.nam");
    let md = load_and_parse(&path);

    const WARMUP_LEN: usize = 24_000;
    const MEASURE_LEN: usize = 4_096;
    let total = WARMUP_LEN + MEASURE_LEN;

    let stress_signal = generate_stress_signal_v2_default(48000);
    assert!(
        stress_signal.len() >= total,
        "Stress signal is too short ({} < {})",
        stress_signal.len(),
        total
    );
    let input_f32 = &stress_signal[0..total];
    let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();

    let mut model =
        neural_amp_modeler_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
    let mut prod_output = vec![0.0f32; total];
    let mut pos = 0;
    {
        let _guard =
            neural_amp_modeler_rs::math::activations::set_thread_local_activation_precision(Some(
                neural_amp_modeler_rs::math::activations::ActivationPrecision::Fast,
            ));
        while pos < total {
            let nf = (total - pos).min(64);
            model.process(&input_f32[pos..pos + nf], &mut prod_output[pos..pos + nf]);
            pos += nf;
        }
    }
    let prod_output_f64: Vec<f64> = prod_output.iter().map(|&x| x as f64).collect();

    let result = run_decomposition_paired(
        "BossLSTM-1x16",
        "LSTM",
        &md,
        &prod_output_f64,
        &input_f64,
        WARMUP_LEN,
    );
    print_decomposition(&result);

    let sum_sources = result.esr_quant_f16c_display()
        + result.esr_activation_display()
        + result.esr_accumulation_display();
    let total_esr = result.esr_f32_vs_f64;
    let ratio = if sum_sources > 0.0 {
        total_esr / sum_sources
    } else {
        0.0
    };
    let inverse_ratio = if total_esr > 0.0 {
        sum_sources / total_esr
    } else {
        0.0
    };

    println!(
        "BossLSTM-1x16 Rule 5: Total ESR = {:.6e}, Sum of sources = {:.6e}, Ratio = {:.2}",
        total_esr, sum_sources, ratio
    );

    assert!(
        ratio <= 10.0 && inverse_ratio <= 10.0,
        "Sanity check failed: total ESR ({:.2e}) is not consistent with sum of sources ({:.2e}) (ratio = {:.2})",
        total_esr,
        sum_sources,
        ratio
    );
}

#[test]
fn test_decomposition_boss_lstm_2x8() {
    use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2_default;

    let path = models_dir().join("BossLSTM-2x8.nam");
    let md = load_and_parse(&path);

    const WARMUP_LEN: usize = 24_000;
    const MEASURE_LEN: usize = 4_096;
    let total = WARMUP_LEN + MEASURE_LEN;

    let stress_signal = generate_stress_signal_v2_default(48000);
    assert!(
        stress_signal.len() >= total,
        "Stress signal is too short ({} < {})",
        stress_signal.len(),
        total
    );
    let input_f32 = &stress_signal[0..total];
    let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();

    let mut model =
        neural_amp_modeler_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
    let mut prod_output = vec![0.0f32; total];
    let mut pos = 0;
    {
        let _guard =
            neural_amp_modeler_rs::math::activations::set_thread_local_activation_precision(Some(
                neural_amp_modeler_rs::math::activations::ActivationPrecision::Fast,
            ));
        while pos < total {
            let nf = (total - pos).min(64);
            model.process(&input_f32[pos..pos + nf], &mut prod_output[pos..pos + nf]);
            pos += nf;
        }
    }
    let prod_output_f64: Vec<f64> = prod_output.iter().map(|&x| x as f64).collect();

    let result = run_decomposition_paired(
        "BossLSTM-2x8",
        "LSTM",
        &md,
        &prod_output_f64,
        &input_f64,
        WARMUP_LEN,
    );
    print_decomposition(&result);

    let sum_sources = result.esr_quant_f16c_display()
        + result.esr_activation_display()
        + result.esr_accumulation_display();
    let total_esr = result.esr_f32_vs_f64;
    let ratio = if sum_sources > 0.0 {
        total_esr / sum_sources
    } else {
        0.0
    };
    let inverse_ratio = if total_esr > 0.0 {
        sum_sources / total_esr
    } else {
        0.0
    };

    println!(
        "BossLSTM-2x8 Rule 5: Total ESR = {:.6e}, Sum of sources = {:.6e}, Ratio = {:.2}",
        total_esr, sum_sources, ratio
    );

    assert!(
        ratio <= 10.0 && inverse_ratio <= 10.0,
        "Sanity check failed: total ESR ({:.2e}) is not consistent with sum of sources ({:.2e}) (ratio = {:.2})",
        total_esr,
        sum_sources,
        ratio
    );
}

#[test]
fn test_decomposition_a2() {
    use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2_default;

    let path = models_dir().join("wavenet_a2_lite.nam");
    let md = load_and_parse(&path);

    const WARMUP_LEN: usize = 24_000;
    const MEASURE_LEN: usize = 4_096;
    let total = WARMUP_LEN + MEASURE_LEN;

    let stress_signal = generate_stress_signal_v2_default(48000);
    assert!(
        stress_signal.len() >= total,
        "Stress signal is too short ({} < {})",
        stress_signal.len(),
        total
    );
    let input_f32 = &stress_signal[0..total];
    let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();

    let mut model =
        neural_amp_modeler_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
    let mut prod_output = vec![0.0f32; total];
    let mut pos = 0;
    {
        let _guard =
            neural_amp_modeler_rs::math::activations::set_thread_local_activation_precision(Some(
                neural_amp_modeler_rs::math::activations::ActivationPrecision::Fast,
            ));
        while pos < total {
            let nf = (total - pos).min(64);
            model.process(&input_f32[pos..pos + nf], &mut prod_output[pos..pos + nf]);
            pos += nf;
        }
    }
    let prod_output_f64: Vec<f64> = prod_output.iter().map(|&x| x as f64).collect();

    let result = run_decomposition_paired(
        "A2-Lite",
        "WaveNet",
        &md,
        &prod_output_f64,
        &input_f64,
        WARMUP_LEN,
    );
    print_decomposition(&result);

    let sum_sources = result.esr_quant_f16c_display()
        + result.esr_activation_display()
        + result.esr_accumulation_display();
    let total_esr = result.esr_f32_vs_f64;
    let ratio = if sum_sources > 0.0 {
        total_esr / sum_sources
    } else {
        0.0
    };
    let inverse_ratio = if total_esr > 0.0 {
        sum_sources / total_esr
    } else {
        0.0
    };

    println!(
        "A2 Rule 5: Total ESR = {:.6e}, Sum of sources = {:.6e}, Ratio = {:.2}, InvRatio = {:.2}",
        total_esr, sum_sources, ratio, inverse_ratio
    );

    assert!(
        result.esr_combined_display() > 0.0,
        "Combined ΔESR should be non-zero"
    );
}

// ── Combined simulation acceptance tests ───────────────────────────────────
// Uses prewarm-paired measurement (24k prewarm + 256 sweep) to
// eliminate transient-mismatch artifacts and assert ESR < 1e-2.

fn combined_config() -> PrecisionConfig {
    PrecisionConfig {
        weight_precision: WeightPrecision::F16C,
        activation: ActivationMode::PadeMinimax,
        accumulation: AccumulationMode::F32Plain,
    }
}

/// Runs a prewarm-paired combined simulation test.
/// Feeds both production and oracle with 24k warmup + 256 measurement samples,
/// then asserts ESR on the post-prewarm window.
fn run_combined_paired_test(model_filename: &str, label: &str) {
    let path = models_dir().join(model_filename);
    let md = load_and_parse(&path);
    let total = 24_000 + 256;
    let input_f64 = gen_sweep(total, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();

    let mut model =
        neural_amp_modeler_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
    let mut prod_output = vec![0.0f32; input_f32.len()];
    let mut pos = 0;
    while pos < input_f32.len() {
        let nf = (input_f32.len() - pos).min(64);
        model.process(&input_f32[pos..pos + nf], &mut prod_output[pos..pos + nf]);
        pos += nf;
    }

    let combined = oracle_forward(&md, &input_f64, &combined_config());
    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());

    let prod_last_f64: Vec<f64> = prod_output[24_000..].iter().map(|&x| x as f64).collect();
    let combined_last = &combined[24_000..];
    let oracle_last = &oracle[24_000..];

    let esr_combined_vs_oracle = compute_esr_f64(oracle_last, combined_last);
    let esr_combined_vs_prod = compute_esr_f64(combined_last, &prod_last_f64);

    println!(
        "{} CombinedSim (prewarm-paired):\n  \
           ΔESR(combined vs oracle):     {:.2e} ({:.1} dB)\n  \
           ESR(combined vs production):  {:.2e} ({:.1} dB)",
        label,
        esr_combined_vs_oracle,
        esr_to_db_f64(esr_combined_vs_oracle),
        esr_combined_vs_prod,
        esr_to_db_f64(esr_combined_vs_prod),
    );
    assert!(
        esr_combined_vs_oracle > 0.0,
        "Combined simulation must be active (non-zero ΔESR)"
    );
    assert!(
        esr_combined_vs_prod < 1e-2,
        "Oracle does not faithfully model production: ESR {:.2e} ≥ 1e-2 ({})",
        esr_combined_vs_prod,
        label,
    );
}

#[test]
fn test_combined_simulation_wavenet() {
    run_combined_paired_test("wavenet_official.nam", "WaveNet");
}

#[test]
fn test_combined_simulation_lstm() {
    run_combined_paired_test("lstm.nam", "LSTM");
}

#[test]
fn test_combined_simulation_a2() {
    run_combined_paired_test("wavenet_a2_lite.nam", "A2");
}

// ── ConvNet oracle tests ────────────────────────────────────────────

#[test]
fn test_oracle_convnet() {
    let esr = run_oracle_esr_paired("convnet_test.nam", "ConvNet");
    assert!(
        esr < CONVNET_ESR_LIMIT,
        "ConvNet ESR={:.6e} exceeds calibrated limit {}",
        esr,
        CONVNET_ESR_LIMIT
    );
}

#[test]
fn test_decomposition_convnet() {
    use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2_default;

    let path = models_dir().join("convnet_test.nam");
    let md = load_and_parse(&path);

    const WARMUP_LEN: usize = 24_000;
    const MEASURE_LEN: usize = 4_096;
    let total = WARMUP_LEN + MEASURE_LEN;

    let stress_signal = generate_stress_signal_v2_default(48000);
    assert!(
        stress_signal.len() >= total,
        "Stress signal is too short ({} < {})",
        stress_signal.len(),
        total
    );
    let input_f32 = &stress_signal[0..total];
    let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();

    let mut model =
        neural_amp_modeler_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
    let mut prod_output = vec![0.0f32; total];
    let mut pos = 0;
    {
        let _guard =
            neural_amp_modeler_rs::math::activations::set_thread_local_activation_precision(Some(
                neural_amp_modeler_rs::math::activations::ActivationPrecision::Fast,
            ));
        while pos < total {
            let nf = (total - pos).min(64);
            model.process(&input_f32[pos..pos + nf], &mut prod_output[pos..pos + nf]);
            pos += nf;
        }
    }
    let prod_output_f64: Vec<f64> = prod_output.iter().map(|&x| x as f64).collect();

    let result = run_decomposition_paired(
        "ConvNet-test",
        "ConvNet",
        &md,
        &prod_output_f64,
        &input_f64,
        WARMUP_LEN,
    );
    print_decomposition(&result);

    let sum_sources = result.esr_quant_f16c_display()
        + result.esr_activation_display()
        + result.esr_accumulation_display();
    let total_esr = result.esr_f32_vs_f64;
    let ratio = if sum_sources > 0.0 {
        total_esr / sum_sources
    } else {
        0.0
    };
    let inverse_ratio = if total_esr > 0.0 {
        sum_sources / total_esr
    } else {
        0.0
    };

    println!(
        "ConvNet Rule 5: Total ESR = {:.6e}, Sum of sources = {:.6e}, Ratio = {:.2}, InvRatio = {:.2}",
        total_esr, sum_sources, ratio, inverse_ratio
    );

    assert!(
        result.esr_combined_display() > 0.0,
        "Combined ΔESR should be non-zero"
    );
}

#[test]
fn test_combined_simulation_convnet() {
    run_combined_paired_test("convnet_test.nam", "ConvNet");
}

#[test]
fn test_oracle_warmup_paired_convnet() {
    run_warmup_paired_test("convnet_test.nam", "ConvNet", 1.00);
}

#[test]
fn test_oracle_vs_python_anchor_convnet() {
    let path = models_dir().join("convnet_test.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("convnet_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "ConvNet: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "ConvNet Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

// ── FiLM A2 oracle tests ────────────────────────────────────────────

#[test]
fn test_oracle_a2_film_lite() {
    let esr = run_oracle_esr_paired("wavenet_a2_film_lite.nam", "A2-FiLM-Lite");
    assert!(
        esr < A2_FILM_ESR_LIMIT,
        "A2-FiLM-Lite ESR={:.6e} exceeds calibrated limit {}",
        esr,
        A2_FILM_ESR_LIMIT
    );
}

#[test]
fn test_oracle_a2_film_full() {
    let esr = run_oracle_esr_paired("wavenet_a2_film_full.nam", "A2-FiLM-Full");
    assert!(
        esr < A2_FILM_ESR_LIMIT,
        "A2-FiLM-Full ESR={:.6e} exceeds calibrated limit {}",
        esr,
        A2_FILM_ESR_LIMIT
    );
}

#[test]
fn test_oracle_a2_film_input_mixin_pre() {
    let esr = run_oracle_esr_paired(
        "wavenet_a2_film_input_mixin_pre.nam",
        "A2-FiLM-InputMixinPre",
    );
    assert!(
        esr < A2_FILM_ESR_LIMIT,
        "A2-FiLM-InputMixinPre ESR={:.6e} exceeds calibrated limit {}",
        esr,
        A2_FILM_ESR_LIMIT
    );
}

#[test]
fn test_combined_simulation_a2_film() {
    run_combined_paired_test("wavenet_a2_film_lite.nam", "A2-FiLM-Lite");
    run_combined_paired_test("wavenet_a2_film_full.nam", "A2-FiLM-Full");
    run_combined_paired_test(
        "wavenet_a2_film_input_mixin_pre.nam",
        "A2-FiLM-InputMixinPre",
    );
}

// ── A2 Generic oracle tests ────────────────────────────────────────

use common::A2_GENERIC_ESR_LIMIT;

// on-demand: KB-A2-MAX paired prod×f64 vs NumPy anchor; not a nightly gate (H0 Case D)
#[test]
#[ignore = "KB-A2-MAX known bug §4.4: fail-closed TR1.1; paired prod×f64 not a gate (H0 Case D)"]
fn test_oracle_vs_python_anchor_a2_generic() {
    let path = models_dir().join("wavenet_a2_max.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("a2_max_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "A2 Generic: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "A2 Generic Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

// on-demand: KB-A2-MAX paired prod×f64 ESR; not a nightly gate (H0 Case D)
#[test]
#[ignore = "KB-A2-MAX known bug §4.4: fail-closed TR1.1; paired prod×f64 not a gate (H0 Case D)"]
fn test_oracle_a2_generic() {
    let esr = run_oracle_esr_paired("wavenet_a2_max.nam", "A2-Generic");
    assert!(
        esr < A2_GENERIC_ESR_LIMIT,
        "A2-Generic ESR={:.6e} exceeds calibrated limit {}",
        esr,
        A2_GENERIC_ESR_LIMIT
    );
}

// on-demand: KB-A2-MAX paired decomposition; not a nightly gate (H0 Case D)
#[test]
#[ignore = "KB-A2-MAX known bug §4.4: fail-closed TR1.1; paired prod×f64 not a gate (H0 Case D)"]
fn test_decomposition_a2_generic() {
    let path = models_dir().join("wavenet_a2_max.nam");
    let md = load_and_parse(&path);

    let esr_paired = run_oracle_esr_paired("wavenet_a2_max.nam", "A2-Generic");
    assert!(
        esr_paired < A2_GENERIC_ESR_LIMIT,
        "A2-Generic paired ESR={:.6e} exceeds calibrated limit {}",
        esr_paired,
        A2_GENERIC_ESR_LIMIT
    );

    let input_f64 = gen_sweep(256, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
    let prod_f32 = run_f32_inference(&md, &input_f32);
    let prod_f64: Vec<f64> = prod_f32.iter().map(|&x| x as f64).collect();

    let result = run_decomposition("A2-Generic-Max", "WaveNet/A2", &md, &prod_f64, &input_f64);
    print_decomposition(&result);
    assert!(
        result.esr_combined_display() > 0.0,
        "Combined ΔESR should be non-zero"
    );
}

// on-demand: KB-A2-MAX combined paired simulation; not a nightly gate (H0 Case D)
#[test]
#[ignore = "KB-A2-MAX known bug §4.4: fail-closed TR1.1; paired prod×f64 not a gate (H0 Case D)"]
fn test_combined_simulation_a2_generic() {
    run_combined_paired_test("wavenet_a2_max.nam", "A2-Generic");
}

/// Standalone oracle-validation for A2 Max — oracle-only, no production dependency.
///
/// T7.1 (Reconciliação) / T7.2 (Calibração de Gates): validates that the f64
/// oracle correctly processes wavenet_a2_max.nam after the corrected topology
/// (head1x1 per-layer, grouped projections, K=1 header, FiLM slot 7 dims).
///
/// **Calibration (T7.2):**
///   Measured: ESR(f64 vs F32-sim) = 6.00e-15 (SNR = 142.2 dB)
///   Gate:     SNR > 120 dB (ESR < 1e-12)
///
/// **Gates (T7.1):**
/// - Oracle output is finite (zero NaN/Inf) and non-zero.
/// - Oracle is deterministic (bit-exact across repeated runs).
/// - f64 oracle vs F32-simulated oracle SNR > 120 dB (FP32 quantization floor).
///
/// **Note (T7.2):** Triple agreement (Python NumPy, C++ NAMCore, Rust f64) is
/// pending the condition_dsp A2 multi-channel gap tracked in §4.4 of
/// docs/cpp_parity_map.md. The Python anchor (validate_oracle_f64.py) has been
/// updated with grouped projection support but requires further reconciliation
/// for exact parity. The standalone test provides the primary oracle gate.
#[test]
fn test_oracle_a2_max_standalone() {
    let path = models_dir().join("wavenet_a2_max.nam");
    let md = load_and_parse(&path);
    let total = 24_000 + 256;
    let input_f64 = gen_sweep(total, 48000.0);

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let oracle_last = &oracle[24_000..];

    assert_eq!(
        oracle.len(),
        input_f64.len(),
        "oracle output length mismatch"
    );
    assert!(
        oracle_last.iter().all(|&x| x.is_finite()),
        "oracle output contains NaN or Inf"
    );
    assert!(
        oracle_last.iter().any(|&x| x.abs() > 1e-12),
        "oracle output is all zeros — model may be silent"
    );

    let oracle2 = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    assert!(
        oracle
            .iter()
            .zip(oracle2.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "oracle is non-deterministic"
    );

    let f32_cfg = PrecisionConfig {
        weight_precision: WeightPrecision::F32,
        activation: ActivationMode::Exact,
        accumulation: AccumulationMode::F32Plain,
    };
    let oracle_f32sim = oracle_forward(&md, &input_f64, &f32_cfg);
    let esr_f64_vs_f32sim = compute_esr_f64(&oracle[24_000..], &oracle_f32sim[24_000..]);

    println!(
        "A2 Max oracle standalone: ESR(f64 vs F32-sim, prewarm-paired) = {:.2e} ({:.1} dB)",
        esr_f64_vs_f32sim,
        esr_to_db_f64(esr_f64_vs_f32sim)
    );
    assert!(
        esr_f64_vs_f32sim < 1e-12,
        "A2 Max f64 vs F32-sim SNR={:.1} dB < 120 dB — exceeds FP32 quantization floor",
        esr_to_db_f64(esr_f64_vs_f32sim)
    );
}

// ── Paired prewarm diagnostic — warmup hypothesis ────────────────────

/// Diagnostic: measures ESR(oracle vs production) with paired prewarm.
///
/// Hypothesis (AC-7): the current ESR gap is dominated by **transient
/// mismatch** — the oracle starts with zero state while production has
/// pre-warmed history buffers. If correct, feeding both with the same 24k
/// prewarm samples before measuring 256 sweep should collapse the gap.
///
/// We use `combined_config()` (F16C + Padé Minimax + f32 accumulation) on
/// the oracle side to match production precision as closely as possible.
fn run_warmup_paired_test(model_filename: &str, label: &str, original_esr: f64) {
    let path = models_dir().join(model_filename);
    let md = load_and_parse(&path);
    let total = 24_000 + 256;
    let input_f64 = gen_sweep(total, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();

    let mut model =
        neural_amp_modeler_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
    let mut prod_output = vec![0.0f32; input_f32.len()];
    let mut pos = 0;
    while pos < input_f32.len() {
        let nf = (input_f32.len() - pos).min(64);
        model.process(&input_f32[pos..pos + nf], &mut prod_output[pos..pos + nf]);
        pos += nf;
    }

    let oracle = oracle_forward(&md, &input_f64, &combined_config());
    let prod_last_f64: Vec<f64> = prod_output[24_000..].iter().map(|&x| x as f64).collect();
    let oracle_last = &oracle[24_000..];

    let esr_full = compute_esr_f64(
        &oracle,
        &prod_output.iter().map(|&x| x as f64).collect::<Vec<f64>>(),
    );
    let esr_warmed = compute_esr_f64(oracle_last, &prod_last_f64);

    println!(
        "{} Paired Prewarm (24k+256):\n\
           ESR(full 24k+256, cold):      {:.2e} ({:.1} dB)\n\
           ESR(last 256, after prewarm): {:.2e} ({:.1} dB)\n\
           Hypothesis: gap should shrink from ~{} to < 0.1 if prewarm is root cause",
        label,
        esr_full,
        esr_to_db_f64(esr_full),
        esr_warmed,
        esr_to_db_f64(esr_warmed),
        original_esr,
    );
}

#[test]
fn test_oracle_warmup_paired_wavenet() {
    run_warmup_paired_test("wavenet_official.nam", "WaveNet", 2.47);
}

#[test]
fn test_oracle_warmup_paired_lstm() {
    run_warmup_paired_test("lstm.nam", "LSTM", 1.06);
}

#[test]
fn test_oracle_warmup_paired_a2() {
    run_warmup_paired_test("wavenet_a2_lite.nam", "A2", 0.126);
}

// ── Summary table ──────────────────────────────────────────────────────────

#[test]
fn test_summary_table() {
    let models: Vec<(&str, &str)> = vec![
        ("wavenet_official.nam", "WaveNet"),
        ("lstm.nam", "LSTM"),
        ("wavenet_a2_lite.nam", "WaveNet(A2)"),
        ("convnet_test.nam", "ConvNet"),
        ("wavenet_a2_film_lite.nam", "A2-FiLM-Lite"),
        ("wavenet_a2_film_full.nam", "A2-FiLM-Full"),
        (
            "wavenet_a2_film_input_mixin_pre.nam",
            "A2-FiLM-InputMixinPre",
        ),
        ("BossWN-standard.nam", "BossWN-standard"),
        ("BossWN-feather.nam", "BossWN-feather"),
        ("BossWN-nano.nam", "BossWN-nano"),
        ("wavenet_a1_standard.nam", "wavenet_a1_standard"),
        ("BossLSTM-1x16.nam", "BossLSTM-1x16"),
        ("BossLSTM-2x8.nam", "BossLSTM-2x8"),
        ("lstm_dyn_test.nam", "LSTMDyn"),
        ("wavenet_a2_full.nam", "A2Full"),
        ("a2_dynamic_gated_ch8.nam", "A2DynGated"),
        ("a2_dynamic_blended_ch3.nam", "A2DynBlended"),
        ("wavenet_a2_film_chaos_stress.nam", "A2FiLMChaos"),
        ("wavenet_dyn_free.nam", "WaveNetDynFree"),
        ("wavenet_condition_dsp.nam", "WaveNetCondDSP"),
        ("EVH-5150-Lite.nam", "EVH-5150-Lite"),
    ];

    println!("\n=== ESR(f32 vs f64 oracle) Summary (prewarm-paired: 24k + 256) ===");
    println!(
        "{:<40} {:<20} {:<15} {:<15}",
        "Model", "Family", "ESR linear", "ESR (dB)"
    );
    println!("{}", "-".repeat(90));

    for (filename, family) in &models {
        let path = resolve_model_path(filename);
        if !path.exists() {
            println!(
                "{:<40} {:<20} {:<15} {:<15}",
                filename, family, "N/A", "N/A"
            );
            continue;
        }
        let esr = run_oracle_esr_paired(filename, family);
        println!(
            "{:<40} {:<20} {:<15.6e} {:<15.1}",
            filename,
            family,
            esr,
            esr_to_db_f64(esr)
        );
    }
    println!("{}", "-".repeat(90));
}

// ── RCA diagnostic: recurrent state drift ───────────────────────────────

fn print_blockwise_esr_table(esr_blocks: &[f64], block_size: usize, sample_rate: u32) {
    let block_duration = block_size as f64 / sample_rate as f64;
    println!(
        "\nTabela ESR Blockwise (block_size = {} amostras / {:.3}s):",
        block_size, block_duration
    );
    println!(
        "{:<6} | {:<15} | {:<12} | {:<10} | Seção do Sinal",
        "Bloco", "Janela Tempo", "ESR linear", "ESR (dB)"
    );
    println!("{}", "-".repeat(85));

    for (idx, &esr) in esr_blocks.iter().enumerate() {
        let start_s = idx as f64 * block_duration;
        let end_s = (idx + 1) as f64 * block_duration;
        let t_mid = (start_s + end_s) / 2.0;

        let section = if t_mid < 1.0 {
            "Single note Low-E (bend+vibrato)"
        } else if t_mid < 2.0 {
            "Power chord E2+E3+B3"
        } else if t_mid < 2.5 {
            "Palm-mute attack-release (16 hits)"
        } else if t_mid < 3.5 {
            "Pinch harmonic train + saw sweep"
        } else if t_mid < 4.5 {
            "Bass amp: low-A + transient pluck"
        } else {
            "Slow chord ringing decay (C-E-G)"
        };

        let esr_db = esr_to_db_f64(esr);

        println!(
            "{:<6} | [{:.2}s - {:.2}s] | {:<12.6e} | {:<10.1} | {}",
            idx, start_s, end_s, esr, esr_db, section
        );
    }
}

#[test]
#[ignore]
fn t33_diagnostic_recurrent_drift_lstm_1x16() {
    use neural_amp_modeler_rs::testing::perceptual::compute_esr;
    use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2_default;

    let path = models_dir().join("BossLSTM-1x16.nam");
    let md = load_and_parse(&path);

    let sample_rate = 48000;
    let stress_signal = generate_stress_signal_v2_default(sample_rate);

    let mut output: Vec<f32> = vec![0.0f32; stress_signal.len()];
    let mut model =
        neural_amp_modeler_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
    model.prewarm(24_000);
    let mut pos = 0;
    while pos < stress_signal.len() {
        let nf = (stress_signal.len() - pos).min(64);
        model.process(&stress_signal[pos..pos + nf], &mut output[pos..pos + nf]);
        pos += nf;
    }

    let input_f64: Vec<f64> = stress_signal.iter().map(|&x| x as f64).collect();
    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let oracle_f32: Vec<f32> = oracle.iter().map(|&x| x as f32).collect();

    // NAMCore golden comparison skipped here — already handled in cpp_parity.
    // Golden files available at: {models_dir}/../../fixtures/golden_lstm_1x16_v2_48000.bin

    let segment_sizes = &[
        512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 120000, 240000,
    ];

    println!("\n=== Recurrent State Drift Analysis ===");
    println!("Model:   BossLSTM-1×16");
    println!("Signal:  v2 stress, 5s @ 48 kHz (240k samples)");
    println!(
        "Baseline pareado esperado (LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT/2): {:.2e}",
        LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT / 2.0
    );
    println!(
        "{:<12} {:<18} {:<12} {:<18}",
        "Samples", "ESR(vs oracle)", "ESR dB", "Time (ms)"
    );
    println!("{}", "-".repeat(60));

    for &n_samples in segment_sizes.iter() {
        let n = n_samples.min(stress_signal.len());

        let mut segment_out = vec![0.0f32; n];
        let mut model_fresh = neural_amp_modeler_rs::loader::dispatcher::build_model(&md)
            .expect("Failed to build model");
        model_fresh.prewarm(24_000);
        let mut pos = 0;
        while pos < n {
            let nf = (n - pos).min(64);
            model_fresh.process(
                &stress_signal[pos..pos + nf],
                &mut segment_out[pos..pos + nf],
            );
            pos += nf;
        }

        let esr_vs_oracle = compute_esr(&oracle_f32[..n], &segment_out[..n]);
        let time_ms = n as f64 / 48.0; // ms
        println!(
            "{:<12} {:<18.6e} {:<12.1} {:<18.1}",
            n,
            esr_vs_oracle,
            10.0 * esr_vs_oracle.log10(),
            time_ms
        );
    }

    // Final comparison: full output vs oracle
    let esr_full = compute_esr(&oracle_f32, &output);
    println!(
        "\nFull 240k: ESR={:.6e} ({:.1} dB)",
        esr_full,
        10.0 * esr_full.log10()
    );

    // Blockwise ESR Analysis
    use neural_amp_modeler_rs::testing::perceptual::compute_esr_blockwise;
    let esr_blocks_48k = compute_esr_blockwise(&oracle_f32, &output, 48_000);
    print_blockwise_esr_table(&esr_blocks_48k, 48_000, 48000);

    let esr_blocks_12k = compute_esr_blockwise(&oracle_f32, &output, 12_000);
    print_blockwise_esr_table(&esr_blocks_12k, 12_000, 48000);

    assert!(
        esr_full < LSTM_1X16_DRIFT_LEGACY_ESR_LIMIT,
        "Legacy LSTM 1x16 drift ESR limit exceeded: {:.6e} >= {:.6e}",
        esr_full,
        LSTM_1X16_DRIFT_LEGACY_ESR_LIMIT
    );
}

// ── Paired prewarm LSTM diagnostic helper ──────────────────────────────

fn run_paired_drift_diagnostic(
    model_filename: &str,
    label: &str,
    block_size: usize,
) -> (f64, Vec<f64>) {
    use neural_amp_modeler_rs::testing::perceptual::{compute_esr, compute_esr_blockwise};
    use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2_default;

    let path = models_dir().join(model_filename);
    let md = load_and_parse(&path);
    let stress_signal = generate_stress_signal_v2_default(48000); // 240k samples
    let stress_f64: Vec<f64> = stress_signal.iter().map(|&x| x as f64).collect();

    // Produção: SEM model.prewarm(zeros) — processa o sinal real desde t=0,
    // igual ao oráculo, eliminando o mismatch de estado inicial.
    let mut model =
        neural_amp_modeler_rs::loader::dispatcher::build_model(&md).expect("build_model");
    let mut output = vec![0.0f32; stress_signal.len()];
    let mut pos = 0;
    while pos < stress_signal.len() {
        let nf = (stress_signal.len() - pos).min(64);
        model.process(&stress_signal[pos..pos + nf], &mut output[pos..pos + nf]);
        pos += nf;
    }

    let oracle = oracle_forward(&md, &stress_f64, &PrecisionConfig::default());
    let oracle_f32: Vec<f32> = oracle.iter().map(|&x| x as f32).collect();

    const N_WARMUP: usize = 24_000;
    let esr_tail = compute_esr(&oracle_f32[N_WARMUP..], &output[N_WARMUP..]);
    println!(
        "\n{} (sem mismatch de estado inicial), cauda de {} amostras: ESR={:.6e} ({:.1} dB)",
        label,
        stress_signal.len() - N_WARMUP,
        esr_tail,
        10.0 * esr_tail.log10()
    );

    let esr_blocks = compute_esr_blockwise(&oracle_f32, &output, block_size);
    print_blockwise_esr_table(&esr_blocks, block_size, 48000);

    (esr_tail, esr_blocks)
}

// ── Paired prewarm LSTM 1x16 diagnostic ──────────────────────────────

#[test]
#[ignore]
fn t33b_diagnostic_recurrent_drift_lstm_1x16_paired() {
    let _guard = PrecisionGuard::new(ActivationPrecision::Standard);

    // 1. Run with Fast precision (Padé/minimax approximation, opt-in)
    set_activation_tls(ActivationPrecision::Fast);
    let (esr_tail_fast, _) =
        run_paired_drift_diagnostic("BossLSTM-1x16.nam", "LSTM 1x16 paired (Fast)", 48_000);
    let _ = run_paired_drift_diagnostic("BossLSTM-1x16.nam", "LSTM 1x16 paired (Fast)", 12_000);

    // 2. Run with Standard precision (exact-grade, universal default)
    set_activation_tls(ActivationPrecision::Standard);
    let (esr_tail_std, _) =
        run_paired_drift_diagnostic("BossLSTM-1x16.nam", "LSTM 1x16 paired (Standard)", 48_000);
    let _ = run_paired_drift_diagnostic("BossLSTM-1x16.nam", "LSTM 1x16 paired (Standard)", 12_000);

    println!("\nLSTM 1x16 Diagnostic Comparison:");
    println!(
        "  Fast ESR_tail:     {:.6e} ({:.1} dB)",
        esr_tail_fast,
        10.0 * esr_tail_fast.log10()
    );
    println!(
        "  Standard ESR_tail: {:.6e} ({:.1} dB)",
        esr_tail_std,
        10.0 * esr_tail_std.log10()
    );
    println!("  Delta:             {:.6e}", esr_tail_fast - esr_tail_std);

    // Fast mode gate
    assert!(
        esr_tail_fast < LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT,
        "Fast LSTM 1x16 paired drift ESR limit exceeded: {:.6e} >= {:.6e}",
        esr_tail_fast,
        LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT
    );

    // Standard confirmation gate: expected to drop to near zero (< 1e-8) because there is no cold-start mismatch
    assert!(
        esr_tail_std < 1.0e-8,
        "Standard LSTM 1x16 paired drift ESR not near zero: {:.6e}",
        esr_tail_std
    );
}

// ── Paired prewarm LSTM 2x8 diagnostic ──────────────────────────────

#[test]
#[ignore]
fn t33c_diagnostic_recurrent_drift_lstm_2x8_paired() {
    let _guard = PrecisionGuard::new(ActivationPrecision::Standard);

    // 1. Run with Fast precision (Padé/minimax approximation, opt-in)
    set_activation_tls(ActivationPrecision::Fast);
    let (esr_tail_fast, _) =
        run_paired_drift_diagnostic("BossLSTM-2x8.nam", "LSTM 2x8 paired (Fast)", 48_000);
    let _ = run_paired_drift_diagnostic("BossLSTM-2x8.nam", "LSTM 2x8 paired (Fast)", 12_000);

    // 2. Run with Standard precision (exact-grade, universal default)
    set_activation_tls(ActivationPrecision::Standard);
    let (esr_tail_std, _) =
        run_paired_drift_diagnostic("BossLSTM-2x8.nam", "LSTM 2x8 paired (Standard)", 48_000);
    let _ = run_paired_drift_diagnostic("BossLSTM-2x8.nam", "LSTM 2x8 paired (Standard)", 12_000);

    println!("\nLSTM 2x8 Diagnostic Comparison:");
    println!(
        "  Fast ESR_tail:     {:.6e} ({:.1} dB)",
        esr_tail_fast,
        10.0 * esr_tail_fast.log10()
    );
    println!(
        "  Standard ESR_tail: {:.6e} ({:.1} dB)",
        esr_tail_std,
        10.0 * esr_tail_std.log10()
    );
    println!("  Delta:             {:.6e}", esr_tail_fast - esr_tail_std);

    // Fast mode gate
    assert!(
        esr_tail_fast < LSTM_2X8_DRIFT_PAIRED_ESR_LIMIT,
        "Fast LSTM 2x8 paired drift ESR limit exceeded: {:.6e} >= {:.6e}",
        esr_tail_fast,
        LSTM_2X8_DRIFT_PAIRED_ESR_LIMIT
    );

    // Standard control gate (ensure it stays below limit)
    assert!(
        esr_tail_std < LSTM_2X8_DRIFT_PAIRED_ESR_LIMIT,
        "Standard LSTM 2x8 paired drift ESR limit exceeded: {:.6e} >= {:.6e}",
        esr_tail_std,
        LSTM_2X8_DRIFT_PAIRED_ESR_LIMIT
    );
}
