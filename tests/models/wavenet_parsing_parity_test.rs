// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! End-to-end WaveNet parsing, fail-closed validation, FiLM defaults, PReLU dimensions,
//! and `condition_dsp` channel contract tests against `namcore_render` (C++) and `f64` oracle.
//!
//! Validates through the full loader (`load_and_build_model_from_bytes`) AND `parse_nam_json` +
//! `build_model` so JSON parsing, topology detection, weight-cursor consumption, and inference
//! parity are all exercised together without `layer_raw` bypasses.

use crate::common::generate_sine_440hz;
use crate::common::metrics::compute_esr;
use neural_amp_modeler_rs::SystemSnapshot;
use neural_amp_modeler_rs::common::diagnostics::NamErrorCode;
use neural_amp_modeler_rs::loader::{LoadError, LoadOptions, load_and_build_model_from_bytes};
use neural_amp_modeler_rs::models::{NamModel, StaticModel};
use neural_amp_modeler_rs::testing::perceptual::compute_snr_db;
use neural_amp_modeler_rs::testing::reference_oracle::{
    OracleError, PrecisionConfig, try_oracle_forward,
};
use neural_amp_modeler_rs::testing::wav::{read_wav_f32, write_wav_f32};
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

struct TempFileGuard {
    paths: Vec<PathBuf>,
}

impl TempFileGuard {
    fn new(prefix: &str) -> (Self, PathBuf, PathBuf, PathBuf) {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let nam = dir.join(format!("nam_s06_{pid}_{prefix}.nam"));
        let in_wav = dir.join(format!("nam_s06_{pid}_{prefix}_in.wav"));
        let out_wav = dir.join(format!("nam_s06_{pid}_{prefix}_out.wav"));
        let guard = Self {
            paths: vec![nam.clone(), in_wav.clone(), out_wav.clone()],
        };
        (guard, nam, in_wav, out_wav)
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        for p in &self.paths {
            let _ = fs::remove_file(p);
        }
    }
}

fn find_namcore_render_bin() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("build/namcore_render/tools/render"),
        manifest_dir.join("build/namcore_render/Release/render"),
        manifest_dir.join("build/namcore_render/Debug/render"),
        manifest_dir.join("build/namcore_render/render"),
    ];
    candidates.into_iter().find(|p| p.exists() && p.is_file())
}

#[expect(clippy::too_many_arguments)]
fn build_minimal_a2_json(
    in_ch: Option<usize>,
    ch: usize,
    cond_size: usize,
    kernel_size: usize,
    is_gated: bool,
    activation: Value,
    extra_layer_fields: Value,
    active_film_shift_count: usize,
    condition_dsp: Option<Value>,
    sample_rate: f64,
) -> Value {
    let effective_in = in_ch.unwrap_or(1);
    let out_ch = if is_gated { ch * 2 } else { ch };

    let mut weights: Vec<f32> = Vec::new();
    // 1. Rechannel (in_ch * ch)
    for i in 0..(effective_in * ch) {
        weights.push(0.45 + 0.05 * ((i % 5) as f32));
    }
    // 2. Single layer (kernel_size * ch * out_ch + bias)
    let conv_w_count = kernel_size * ch * out_ch;
    for i in 0..conv_w_count {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        weights.push(sign * (0.18 + 0.02 * ((i % 7) as f32)));
    }
    for i in 0..out_ch {
        weights.push(0.05 * ((i as f32) + 1.0));
    }
    // input_mixin (out_ch * cond_size)
    for i in 0..(out_ch * cond_size) {
        weights.push(0.22 - 0.03 * ((i % 4) as f32));
    }
    // layer1x1 (ch * ch + ch)
    for i in 0..(ch * ch) {
        weights.push(0.30 + 0.04 * ((i % 3) as f32));
    }
    for i in 0..ch {
        weights.push(0.02 * ((i as f32) + 1.0));
    }
    // Active FiLMs (scale+shift weights: 2 * ch * cond_size, biases: 2 * ch)
    for _ in 0..active_film_shift_count {
        for i in 0..(2 * ch * cond_size) {
            weights.push(0.10 + 0.02 * ((i % 5) as f32));
        }
        // scale biases (~1.0), shift biases (~0.05)
        for i in 0..ch {
            weights.push(1.0 + 0.03 * (i as f32));
        }
        for i in 0..ch {
            weights.push(0.05 + 0.01 * (i as f32));
        }
    }
    // 3. Head (head_size=1, kernel_size=1, bias=true -> ch + 1)
    for i in 0..ch {
        weights.push(0.35 + 0.05 * (i as f32));
    }
    weights.push(0.01);
    // 4. Head scale
    weights.push(0.25);

    let mut layer = json!({
        "input_size": effective_in,
        "condition_size": cond_size,
        "head_size": 1,
        "channels": ch,
        "bottleneck": ch,
        "kernel_size": kernel_size,
        "dilations": [1],
        "activation": activation,
        "head": {
            "out_channels": 1,
            "kernel_size": 1,
            "bias": true
        },
        "layer1x1": {
            "active": true,
            "groups": 1
        },
        "head1x1": {
            "active": false,
            "out_channels": ch,
            "groups": 1
        }
    });
    if let (Some(dst), Some(src)) = (layer.as_object_mut(), extra_layer_fields.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }

    let mut config = json!({
        "layers": [layer],
        "head_scale": 0.25
    });
    if let Some(ic) = in_ch {
        config["in_channels"] = json!(ic);
    }
    if let Some(cdsp) = condition_dsp {
        config["condition_dsp"] = cdsp;
    }

    json!({
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": config,
        "weights": weights,
        "sample_rate": sample_rate
    })
}

fn load_rust_model(nam_val: &Value) -> Result<Box<StaticModel>, LoadError> {
    let sys = SystemSnapshot::capture();
    let bytes = nam_val.to_string().into_bytes();
    let loaded = load_and_build_model_from_bytes(&bytes, &sys, false, LoadOptions::default())?;
    Ok(loaded.model_l.expect("loaded mono L model"))
}

fn run_cpp_render(
    render_bin: &PathBuf,
    tag: &str,
    nam_val: &Value,
    input: &[f32],
    sr: u32,
) -> Result<Vec<f32>, String> {
    let (_guard, nam_path, in_wav, out_wav) = TempFileGuard::new(tag);
    fs::write(&nam_path, nam_val.to_string()).map_err(|e| format!("write .nam: {e}"))?;
    write_wav_f32(&in_wav, input, sr).map_err(|e| format!("write wav: {e}"))?;

    let out = Command::new(render_bin)
        .arg(&nam_path)
        .arg(&in_wav)
        .arg(&out_wav)
        .output()
        .map_err(|e| format!("exec render: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    let (samples, out_sr) = read_wav_f32(&out_wav).map_err(|e| format!("read out wav: {e}"))?;
    if out_sr != sr {
        return Err(format!("sample rate mismatch: expected {sr}, got {out_sr}"));
    }
    Ok(samples)
}

fn run_parity_suite_for_case(
    tag: &str,
    nam_val: &Value,
    min_snr_cpp: f64,
    max_esr_cpp: f64,
    max_esr_f64: f64,
) {
    let sr = 48000u32;
    let n = 1024usize;
    let mut input = generate_sine_440hz(n);
    // Add non-trivial harmonic variation to stress non-linearities & FiLM
    for (i, s) in input.iter_mut().enumerate() {
        *s = 0.5 * *s
            + 0.25 * (2.0 * std::f32::consts::PI * 1130.0 * (i as f32) / (sr as f32)).sin();
    }

    // 1. Load & weight cursor validation in Rust
    let mut rust_model = load_rust_model(nam_val)
        .unwrap_or_else(|e| panic!("[{tag}] Rust loader failed unexpectedly: {e:?}"));
    rust_model.reset(sr, 64).expect("reset");
    let mut rust_out = vec![0.0f32; n];
    for (ic, oc) in input.chunks(64).zip(rust_out.chunks_mut(64)) {
        rust_model.process(ic, oc);
    }
    assert!(
        rust_out.iter().any(|x| x.abs() > 1e-6) && rust_out.iter().all(|x| x.is_finite()),
        "[{tag}] Rust output must be finite and non-silent"
    );

    // 2. f64 Reference Oracle audio parity (with matching prewarm zeros prepended)
    let md: neural_amp_modeler_rs::loader::nam_json::NamModelData =
        serde_json::from_value(nam_val.clone()).expect("deserialize NamModelData");
    let prewarm_len = rust_model.prewarm_samples();
    let mut input_f64 = vec![0.0f64; prewarm_len + n];
    for (i, &x) in input.iter().enumerate() {
        input_f64[prewarm_len + i] = x as f64;
    }
    let oracle_res = try_oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let f64_summary = match oracle_res {
        Ok(oracle_full) => {
            let oracle_out: Vec<f32> = oracle_full[prewarm_len..prewarm_len + n]
                .iter()
                .map(|&v| v as f32)
                .collect();
            let esr_f64 = compute_esr(&oracle_out, &rust_out);
            let snr_f64 = compute_snr_db(&oracle_out, &rust_out);
            assert!(
                esr_f64 < max_esr_f64,
                "[{tag}] f64 ESR too high: {esr_f64} (SNR={snr_f64:.2} dB)"
            );
            format!("SNR={snr_f64:.2} dB, ESR={esr_f64:.3e}")
        }
        Err(OracleError::UnsupportedTopology { .. }) => {
            "skipped (oracle topology scope)".to_string()
        }
        Err(e) => panic!("[{tag}] f64 oracle failed unexpectedly: {e}"),
    };

    // 3. Independent C++ NeuralAmpModelerCore parity (when namcore_render binary is available)
    if let Some(render_bin) = find_namcore_render_bin() {
        let cpp_out = run_cpp_render(&render_bin, tag, nam_val, &input, sr)
            .unwrap_or_else(|err| panic!("[{tag}] C++ namcore_render rejected valid model: {err}"));
        assert_eq!(cpp_out.len(), n);
        let esr_cpp = compute_esr(&cpp_out, &rust_out);
        let snr_cpp = compute_snr_db(&cpp_out, &rust_out);
        eprintln!("[{tag}] C++ parity: SNR={snr_cpp:.2} dB, ESR={esr_cpp:.3e}; f64: {f64_summary}");
        assert!(
            esr_cpp < max_esr_cpp && snr_cpp > min_snr_cpp,
            "[{tag}] C++ parity failed: SNR={snr_cpp:.2} dB (min {min_snr_cpp}), ESR={esr_cpp:.3e} (max {max_esr_cpp})"
        );
    } else {
        eprintln!(
            "[STATUS] SKIP_OPTIONAL_FIXTURE: namcore_render binary not found ({tag}); f64: {f64_summary}"
        );
    }
}

#[test]
fn test_s06_wavenet_gating_legacy_and_precedence_cpp_and_f64_parity() {
    // Legacy `gated: true` (no `gating_mode`, no `secondary_activation` -> Sigmoid)
    let legacy_gated_true = build_minimal_a2_json(
        None,
        2,
        1,
        2,
        true,
        json!("LeakyReLU"),
        json!({"gated": true}),
        0,
        None,
        48000.0,
    );
    // Measured (vs namcore_render C++ & f64 oracle):
    //   C++: SNR=143.51 dB, ESR=4.454e-15; f64: SNR=143.89 dB, ESR=4.080e-15
    run_parity_suite_for_case("legacy_gated_true", &legacy_gated_true, 80.0, 1e-4, 1e-4);

    // Intentional P2 A1 guardrail: Tanh WaveNet declaring legacy gated=true must fail closed.
    let a1_tanh_gated = build_minimal_a2_json(
        None,
        2,
        1,
        2,
        true,
        json!("Tanh"),
        json!({"gated": true}),
        0,
        None,
        48000.0,
    );
    let p2_err = load_rust_model(&a1_tanh_gated)
        .err()
        .expect("P2 guardrail must reject Tanh WaveNet with gated=true");
    assert_eq!(p2_err.code(), NamErrorCode::InvalidModelTopology);

    // Legacy `gated: false`
    let legacy_gated_false = build_minimal_a2_json(
        None,
        2,
        1,
        2,
        false,
        json!("ReLU"),
        json!({"gated": false}),
        0,
        None,
        48000.0,
    );
    // Measured: C++ SNR=143.15 dB, ESR=4.847e-15; f64 SNR=144.43 dB, ESR=3.605e-15
    run_parity_suite_for_case("legacy_gated_false", &legacy_gated_false, 100.0, 1e-8, 1e-8);

    // Precedence: `gating_mode: "none"` overrides `gated: true` (must consume 1x bottleneck weights)
    let mode_none_overrides_gated = build_minimal_a2_json(
        None,
        2,
        1,
        2,
        false,
        json!("ReLU"),
        json!({"gated": true, "gating_mode": "none"}),
        0,
        None,
        48000.0,
    );
    // Measured: C++ SNR=143.15 dB, ESR=4.847e-15; f64 SNR=144.43 dB, ESR=3.605e-15
    run_parity_suite_for_case(
        "gating_mode_none_overrides_gated_true",
        &mode_none_overrides_gated,
        100.0,
        1e-8,
        1e-8,
    );

    // Precedence: `gating_mode: "gated"` overrides `gated: false` (uses secondary_activation)
    let mode_gated_overrides_false = build_minimal_a2_json(
        None,
        2,
        1,
        2,
        true,
        json!("ReLU"),
        json!({"gated": false, "gating_mode": "gated", "secondary_activation": "Sigmoid"}),
        0,
        None,
        48000.0,
    );
    // Measured: C++ SNR=143.53 dB, ESR=4.439e-15; f64 SNR=143.91 dB, ESR=4.064e-15
    run_parity_suite_for_case(
        "gating_mode_gated_overrides_gated_false",
        &mode_gated_overrides_false,
        100.0,
        1e-8,
        1e-8,
    );
}

#[test]
fn test_s06_wavenet_prelu_primary_and_secondary_forms_cpp_and_f64_parity() {
    // 1. Secondary PReLU simple string `"PReLU"` (default 0.01 slope) + primary per-channel vector PReLU
    let prelu_simple_and_vector = build_minimal_a2_json(
        None,
        2,
        1,
        2,
        true,
        json!({"type": "PReLU", "negative_slopes": [0.15, 0.35]}),
        json!({"gating_mode": "gated", "secondary_activation": "PReLU"}),
        0,
        None,
        48000.0,
    );
    // Measured: C++ SNR=142.75 dB, ESR=5.306e-15; f64 SNR=144.24 dB, ESR=3.764e-15
    run_parity_suite_for_case(
        "prelu_primary_vec_sec_simple",
        &prelu_simple_and_vector,
        100.0,
        1e-8,
        1e-8,
    );

    // 2. Secondary PReLU scalar object `{"type": "PReLU", "negative_slope": 0.25}`
    let prelu_scalar = build_minimal_a2_json(
        None,
        2,
        1,
        2,
        true,
        json!("ReLU"),
        json!({"gating_mode": "gated", "secondary_activation": {"type": "PReLU", "negative_slope": 0.25}}),
        0,
        None,
        48000.0,
    );
    // Measured: C++ SNR=142.83 dB, ESR=5.214e-15; f64 SNR=144.33 dB, ESR=3.691e-15
    run_parity_suite_for_case("prelu_sec_scalar", &prelu_scalar, 100.0, 1e-8, 1e-8);

    // 3. Secondary PReLU per-channel vector `{"type": "PReLU", "negative_slopes": [0.12, 0.38]}`
    let prelu_vector = build_minimal_a2_json(
        None,
        2,
        1,
        2,
        true,
        json!("ReLU"),
        json!({"gating_mode": "blended", "secondary_activation": {"type": "PReLU", "negative_slopes": [0.12, 0.38]}}),
        0,
        None,
        48000.0,
    );
    // Measured: C++ SNR=142.63 dB, ESR=5.451e-15; f64 SNR=144.27 dB, ESR=3.741e-15
    run_parity_suite_for_case("prelu_sec_vector_blended", &prelu_vector, 100.0, 1e-8, 1e-8);
}

#[test]
fn test_s06_wavenet_film_empty_object_and_false_cpp_and_f64_parity() {
    // FiLM `{}` must default `active: true, shift: true, groups: 1` (matches C++ `wavenet/model.cpp:1209`)
    let film_empty_obj = build_minimal_a2_json(
        None,
        2,
        1,
        2,
        false,
        json!("ReLU"),
        json!({"conv_pre_film": {}, "conv_post_film": false}),
        1,
        None,
        48000.0,
    );
    // Measured: C++ SNR=142.52 dB, ESR=5.593e-15; f64 SNR=143.58 dB, ESR=4.389e-15
    run_parity_suite_for_case("film_empty_object", &film_empty_obj, 100.0, 1e-8, 1e-8);
}

#[test]
fn test_s06_wavenet_condition_dsp_and_multichannel_parent_matrix() {
    // 1. Valid condition_dsp 1->1 matching mono parent
    let cond_1to1 = build_minimal_a2_json(
        Some(1),
        2,
        1,
        2,
        false,
        json!("ReLU"),
        json!({}),
        0,
        Some(json!({
            "version": "0.6.0",
            "architecture": "Linear",
            "config": {"receptive_field": 1, "bias": false, "in_channels": 1, "out_channels": 1},
            "weights": [0.75],
            "sample_rate": 48000.0
        })),
        48000.0,
    );
    // Measured: C++ SNR=142.61 dB, ESR=5.477e-15 (oracle skipped: Linear condition_dsp)
    run_parity_suite_for_case("condition_dsp_1to1_mono", &cond_1to1, 100.0, 1e-8, 1e-8);

    // 2. Preserved P5 broadcast: condition_dsp 1->1 with condition_size=3 on mono parent
    // (C++ rejects dsp_ch != condition_size; Rust keeps documented P5 channel-0 broadcast).
    let cond_broadcast = build_minimal_a2_json(
        Some(1),
        2,
        3,
        2,
        false,
        json!("ReLU"),
        json!({}),
        0,
        Some(json!({
            "version": "0.6.0",
            "architecture": "Linear",
            "config": {"receptive_field": 1, "bias": false, "in_channels": 1, "out_channels": 1},
            "weights": [0.75],
            "sample_rate": 48000.0
        })),
        48000.0,
    );
    let mut b_model = load_rust_model(&cond_broadcast)
        .expect("Documented P5 condition_dsp broadcast (1 < condition_size=3) must stay valid");
    let b_in: Vec<f32> = generate_sine_440hz(128)
        .into_iter()
        .map(|s| 0.5 * s)
        .collect();
    let mut b_out = vec![0.0f32; 128];
    b_model.reset(48000, 64).expect("reset");
    b_model.process(&b_in, &mut b_out);
    assert!(b_out.iter().any(|x| x.abs() > 1e-6) && b_out.iter().all(|x| x.is_finite()));

    // 3. Valid multichannel parent (in_channels=2) with Linear 2->2 shared condition_dsp (condition_size=2)
    let multi_parent_2to2 = build_minimal_a2_json(
        Some(2),
        2,
        2,
        2,
        false,
        json!("ReLU"),
        json!({}),
        0,
        Some(json!({
            "version": "0.6.0",
            "architecture": "Linear",
            "config": {"receptive_field": 1, "bias": false, "in_channels": 2, "out_channels": 2},
            "weights": [0.65],
            "sample_rate": 48000.0
        })),
        48000.0,
    );
    let mut m_model = load_rust_model(&multi_parent_2to2)
        .expect("Multichannel parent in_channels=2 with condition_dsp 2->2 must build");
    let mut m_in = vec![0.0f32; 128 * 2];
    for f in 0..128 {
        m_in[f * 2] = 0.4 * ((f as f32) * 0.05).sin();
        m_in[f * 2 + 1] = -0.3 * ((f as f32) * 0.08).cos();
    }
    let mut m_out = vec![0.0f32; 128];
    m_model.reset(48000, 64).expect("reset");
    m_model.process(&m_in, &mut m_out);
    assert!(m_out.iter().any(|x| x.abs() > 1e-6) && m_out.iter().all(|x| x.is_finite()));

    // 3b. Valid multichannel parent (in_channels=2, condition_size=2) with Linear 2->1 condition_dsp (P5 broadcast)
    let multi_parent_2to1 = build_minimal_a2_json(
        Some(2),
        2,
        2,
        2,
        false,
        json!("ReLU"),
        json!({}),
        0,
        Some(json!({
            "version": "0.6.0",
            "architecture": "Linear",
            "config": {"receptive_field": 1, "bias": false, "in_channels": 2, "out_channels": 1},
            "weights": [0.5, -0.3],
            "sample_rate": 48000.0
        })),
        48000.0,
    );
    let mut m21_model = load_rust_model(&multi_parent_2to1)
        .expect("Multichannel parent in_channels=2 with condition_dsp 2->1 must build");
    let mut m21_out = vec![0.0f32; 128];
    m21_model.reset(48000, 64).expect("reset");
    m21_model.process(&m_in, &mut m21_out);
    assert!(m21_out.iter().any(|x| x.abs() > 1e-6) && m21_out.iter().all(|x| x.is_finite()));

    // 4. Valid multichannel parent (in_channels=2, condition_size=2) without condition_dsp
    let multi_parent_no_cond = build_minimal_a2_json(
        Some(2),
        2,
        2,
        2,
        false,
        json!("ReLU"),
        json!({}),
        0,
        None,
        48000.0,
    );
    let mut m_nc_model = load_rust_model(&multi_parent_no_cond).expect(
        "Multichannel parent in_channels=2, condition_size=2 without condition_dsp must build",
    );
    let mut m_nc_out = vec![0.0f32; 128];
    m_nc_model.reset(48000, 64).expect("reset");
    m_nc_model.process(&m_in, &mut m_nc_out);
    assert!(m_nc_out.iter().any(|x| x.abs() > 1e-6) && m_nc_out.iter().all(|x| x.is_finite()));
}

#[test]
fn test_s06_wavenet_fail_closed_invalid_parsing_and_topology_matrix() {
    // Every case below must fail closed through `load_and_build_model_from_bytes` with a specific
    // LoadError/NamErrorCode (zero silent fallbacks to Sigmoid, ReLU, None gating, or inactive FiLM).
    let invalid_cases: Vec<(&str, Value, Option<NamErrorCode>, bool)> = vec![
        // (name, nam_json, expected_error_code, cpp_also_rejects)
        (
            "gating_mode_bogus",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                false,
                json!("ReLU"),
                json!({"gating_mode": "bogus"}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            true,
        ),
        (
            "gating_mode_null",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                false,
                json!("ReLU"),
                json!({"gating_mode": null}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            true,
        ),
        (
            "gating_mode_array_with_null",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                false,
                json!("ReLU"),
                json!({"gating_mode": [null]}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            true,
        ),
        (
            "gated_non_bool_string",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                false,
                json!("ReLU"),
                json!({"gated": "yes"}),
                0,
                None,
                48000.0,
            ),
            None, // serde type error -> NamJsonParseError (invalid JSON field type)
            true,
        ),
        (
            "gated_true_with_invalid_secondary_activation_fails_via_gating_mode",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                true,
                json!("ReLU"),
                json!({"gating_mode": "gated", "secondary_activation": null}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::NamJsonUnknownActivation),
            true,
        ),
        (
            "gating_mode_gated_with_empty_secondary",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                true,
                json!("ReLU"),
                json!({"gating_mode": "gated", "secondary_activation": ""}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::NamJsonUnknownActivation),
            true,
        ),
        (
            "gating_mode_blended_with_unknown_secondary",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                true,
                json!("ReLU"),
                json!({"gating_mode": "blended", "secondary_activation": "Bogus"}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::NamJsonUnknownActivation),
            true,
        ),
        (
            "prelu_empty_negative_slopes",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                false,
                json!({"type": "PReLU", "negative_slopes": []}),
                json!({}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::NamJsonUnknownActivation),
            false, // C++ Release disables assert
        ),
        (
            "primary_prelu_wrong_dimension_len3_for_ch2",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                false,
                json!({"type": "PReLU", "negative_slopes": [0.1, 0.2, 0.3]}),
                json!({}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            false,
        ),
        (
            "secondary_prelu_wrong_dimension_len3_for_ch2",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                true,
                json!("ReLU"),
                json!({"gating_mode": "gated", "secondary_activation": {"type": "PReLU", "negative_slopes": [0.1, 0.2, 0.3]}}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            false,
        ),
        (
            "film_invalid_string_value",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                false,
                json!("ReLU"),
                json!({"conv_post_film": "bad"}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            true,
        ),
        (
            "film_invalid_true_bool",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                false,
                json!("ReLU"),
                json!({"conv_post_film": true}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            true,
        ),
        (
            "film_invalid_active_type",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                false,
                json!("ReLU"),
                json!({"conv_post_film": {"active": "yes"}}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            true,
        ),
        (
            "film_empty_object_without_film_weights_fails_weight_cursor",
            build_minimal_a2_json(
                None,
                2,
                1,
                2,
                false,
                json!("ReLU"),
                json!({"conv_post_film": {}}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::WeightCountMismatch),
            true,
        ),
        (
            "condition_dsp_2to1_on_mono_parent",
            build_minimal_a2_json(
                Some(1),
                2,
                1,
                2,
                false,
                json!("ReLU"),
                json!({}),
                0,
                Some(json!({
                    "version": "0.6.0",
                    "architecture": "Linear",
                    "config": {"receptive_field": 1, "bias": false, "in_channels": 2, "out_channels": 1},
                    "weights": [0.5, -0.5],
                    "sample_rate": 48000.0
                })),
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            true,
        ),
        (
            "condition_dsp_1to1_on_2ch_parent",
            build_minimal_a2_json(
                Some(2),
                2,
                2,
                2,
                false,
                json!("ReLU"),
                json!({}),
                0,
                Some(json!({
                    "version": "0.6.0",
                    "architecture": "Linear",
                    "config": {"receptive_field": 1, "bias": false, "in_channels": 1, "out_channels": 1},
                    "weights": [0.5],
                    "sample_rate": 48000.0
                })),
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            true,
        ),
        (
            "condition_dsp_out_exceeds_condition_size",
            build_minimal_a2_json(
                Some(1),
                2,
                1,
                2,
                false,
                json!("ReLU"),
                json!({}),
                0,
                Some(json!({
                    "version": "0.6.0",
                    "architecture": "Linear",
                    "config": {"receptive_field": 1, "bias": false, "in_channels": 1, "out_channels": 2},
                    "weights": [0.5, -0.5],
                    "sample_rate": 48000.0
                })),
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            true,
        ),
        (
            "condition_dsp_sample_rate_mismatch",
            build_minimal_a2_json(
                Some(1),
                2,
                1,
                2,
                false,
                json!("ReLU"),
                json!({}),
                0,
                Some(json!({
                    "version": "0.6.0",
                    "architecture": "Linear",
                    "config": {"receptive_field": 1, "bias": false, "in_channels": 1, "out_channels": 1},
                    "weights": [0.5],
                    "sample_rate": 44100.0
                })),
                48000.0,
            ),
            Some(NamErrorCode::NamJsonSampleRateMismatch),
            true,
        ),
        (
            "multichannel_parent_without_cond_dsp_mismatched_condition_size",
            build_minimal_a2_json(
                Some(2),
                2,
                3,
                2,
                false,
                json!("ReLU"),
                json!({}),
                0,
                None,
                48000.0,
            ),
            Some(NamErrorCode::InvalidModelTopology),
            true,
        ),
    ];

    let render_bin = find_namcore_render_bin();
    let probe_in = vec![0.0f32; 64];

    for (name, nam_val, expected_code, cpp_also_rejects) in invalid_cases {
        let err = load_rust_model(&nam_val).err().unwrap_or_else(|| {
            panic!("[{name}] Expected fail-closed rejection, but model loaded!")
        });
        if let Some(code) = expected_code {
            assert_eq!(
                err.code(),
                code,
                "[{name}] Unexpected LoadError::code(): got {:?} ({err})",
                err.code()
            );
        }
        if cpp_also_rejects && let Some(ref bin) = render_bin {
            let cpp_res = run_cpp_render(bin, &format!("neg_{name}"), &nam_val, &probe_in, 48000);
            assert!(
                cpp_res.is_err(),
                "[{name}] Expected C++ namcore_render to also reject this model, but it succeeded!"
            );
        }
    }
}

#[test]
fn test_s06_wavenet_config_in_channels_vs_layers0_and_dynamic_a1_routes() {
    // 1. config.in_channels != layers[0].input_size must fail with InvalidModelTopology
    let mut mismatch = build_minimal_a2_json(
        Some(2),
        2,
        2,
        2,
        false,
        json!("ReLU"),
        json!({}),
        0,
        None,
        48000.0,
    );
    mismatch["config"]["layers"][0]["input_size"] = json!(1);
    let err = load_rust_model(&mismatch)
        .err()
        .expect("mismatched in_channels must fail");
    assert_eq!(err.code(), NamErrorCode::InvalidModelTopology);

    // 2. PostStackHead PReLU wrong slope dimension in dynamic WaveNet (no A2 features -> A1 dyn path)
    // Construct a minimal free-geometry A1 dynamic WaveNet with `head.activation: PReLU[len=2]` when head.channels=1
    let a1_dyn_bad_head_prelu = json!({
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "head_size": 1,
                "channels": 1,
                "kernel_size": 1,
                "dilations": [1],
                "activation": "Tanh",
                "gated": false,
                "head_bias": true
            }],
            "head": {
                "channels": 1,
                "out_channels": 1,
                "kernel_size": 1,
                "bias": false,
                "activation": {"type": "PReLU", "negative_slopes": [0.1, 0.2]}
            },
            "head_scale": 1.0
        },
        // weights: rechannel(1) + conv(2) + mixin(1) + l1x1(2) + head_rechannel(2) + head_scale(1) + post_stack_head(1)
        "weights": vec![0.1; 10],
        "sample_rate": 48000.0
    });
    let err = load_rust_model(&a1_dyn_bad_head_prelu)
        .err()
        .expect("PostStackHead PReLU dimension mismatch must fail closed");
    assert_eq!(
        err.code(),
        NamErrorCode::InvalidModelTopology,
        "unexpected error: {err}"
    );

    // 3. WaveNet A1 dynamic with `condition_dsp` `2->1` on mono parent must fail closed
    let a1_dyn_bad_cond_in = json!({
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "head_size": 1,
                "channels": 1,
                "kernel_size": 1,
                "dilations": [1],
                "activation": "ReLU",
                "gated": false,
                "head_bias": true
            }],
            "head_scale": 1.0,
            "condition_dsp": {
                "version": "0.6.0",
                "architecture": "Linear",
                "config": {"receptive_field": 1, "bias": false, "in_channels": 2, "out_channels": 1},
                "weights": [0.5, -0.5],
                "sample_rate": 48000.0
            }
        },
        "weights": vec![0.1; 10],
        "sample_rate": 48000.0
    });
    let err = load_rust_model(&a1_dyn_bad_cond_in)
        .err()
        .expect("WaveNet A1 dynamic condition_dsp 2->1 on mono parent must fail");
    assert_eq!(err.code(), NamErrorCode::InvalidModelTopology);

    // 4. WaveNet A1 guardrail: Tanh WaveNet with `conv_post_film: {}` or `"bad"` must fail closed
    for film_val in [json!({}), json!("bad")] {
        let a1_film = build_minimal_a2_json(
            None,
            2,
            1,
            2,
            false,
            json!("Tanh"),
            json!({"conv_post_film": film_val}),
            0,
            None,
            48000.0,
        );
        let err = load_rust_model(&a1_film)
            .err()
            .expect("WaveNet A1 guardrail must reject FiLM `{}` or invalid FiLM");
        assert_eq!(err.code(), NamErrorCode::InvalidModelTopology);
    }

    // 5. `layer1x1_post_film: {}` with `layer1x1.active == false` must fail closed
    let invalid_l1x1_film = build_minimal_a2_json(
        None,
        2,
        1,
        2,
        false,
        json!("ReLU"),
        json!({"layer1x1": {"active": false, "groups": 1}, "layer1x1_post_film": {}}),
        0,
        None,
        48000.0,
    );
    let err = load_rust_model(&invalid_l1x1_film)
        .err()
        .expect("layer1x1_post_film:{} when layer1x1.active==false must fail");
    assert_eq!(err.code(), NamErrorCode::InvalidModelTopology);

    // 6. 2-array A2 cascade where array[1] has `condition_size: 1` < `condition_dsp` out_channels=2
    let mut cascade_bad_cond = build_minimal_a2_json(
        Some(1),
        2,
        2,
        2,
        false,
        json!(["ReLU"]),
        json!({}),
        0,
        Some(json!({
            "version": "0.6.0",
            "architecture": "Linear",
            "config": {"receptive_field": 1, "bias": false, "in_channels": 1, "out_channels": 2},
            "weights": [0.5, -0.5],
            "sample_rate": 48000.0
        })),
        48000.0,
    );
    let arr0 = cascade_bad_cond["config"]["layers"][0].clone();
    let mut arr1 = arr0.clone();
    arr1["input_size"] = json!(2);
    arr1["condition_size"] = json!(1);
    cascade_bad_cond["config"]["layers"] = json!([arr0, arr1]);
    // Exact cascade weights: arr0(2+10+4+6+3+1=26) + arr1(4+10+2+6+3+1=26) = 52
    cascade_bad_cond["weights"] = json!(vec![0.1f32; 52]);
    let err = load_rust_model(&cascade_bad_cond)
        .err()
        .expect("2-array cascade array[1].condition_size < cond_out must fail closed");
    assert_eq!(
        err.code(),
        NamErrorCode::InvalidModelTopology,
        "unexpected error: {err}"
    );

    // 7. Valid 44100 Hz parent + 44100 Hz condition_dsp builds cleanly
    let valid_441 = build_minimal_a2_json(
        Some(1),
        2,
        1,
        2,
        false,
        json!("ReLU"),
        json!({}),
        0,
        Some(json!({
            "version": "0.6.0",
            "architecture": "Linear",
            "config": {"receptive_field": 1, "bias": false, "in_channels": 1, "out_channels": 1},
            "weights": [0.75],
            "sample_rate": 44100.0
        })),
        44100.0,
    );
    let mut m441 = load_rust_model(&valid_441).expect("44.1 kHz matching condition_dsp must build");
    let mut o441 = [0.0f32; 32];
    m441.reset(44100, 32).expect("reset 44.1k");
    m441.process(&[0.25f32; 32], &mut o441);
    assert!(o441.iter().any(|x| x.abs() > 1e-6) && o441.iter().all(|x| x.is_finite()));
}
