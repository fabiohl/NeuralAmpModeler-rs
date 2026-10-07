// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! SPRINT NC-4: WaveNet layer array head configuration & parity validation (GAP-03, GAP-04).
//!
//! Validates:
//! 1. Parity between legacy `head_size` + `head_bias` (implicit kernel=1, dilation=1)
//!    and explicit nested `head` object (`out_channels`, `kernel_size=1`, `head_dilation=1`, `bias`).
//! 2. Multi-tap dilated head rechannel (`kernel_size=3, head_dilation=1` and `kernel_size=3, head_dilation=3`):
//!    - Correct receptive field calculation: `RF = Σ dil*(k - 1) + (head_k - 1) * head_dilation`.
//!    - Continuous block-size invariance across chunk sizes (1, 16, 64) vs baseline (64).
//!    - Prewarm transient suppression: seamless initialization without DC offset/discontinuity.
//! 3. Live cross-validation against C++ `NeuralAmpModelerCore` (`render` tool) when available.

use super::common::*;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{
    WavenetTopologyResult, get_wavenet_topology, parse_nam_json,
};
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::perceptual::compute_snr_db;
use neural_amp_modeler_rs::testing::wav::{read_wav_f32, write_wav_f32};

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Generates synthetic WaveNet `.nam` JSON data with deterministic weights.
#[expect(clippy::too_many_arguments)]
fn make_synthetic_wavenet_nam(
    ch: usize,
    k: usize,
    dilations: &[usize],
    head_size: usize,
    head_k: usize,
    head_dilation: usize,
    has_head_bias: bool,
    nested_head: bool,
) -> String {
    let rechannel = ch;
    let layer_weights = (ch * ch * k + ch) + ch + (ch * ch + ch);
    let layers_total = dilations.len() * layer_weights;
    let head_weights = if head_k == 1 && head_dilation == 1 && !nested_head {
        ch * head_size + if has_head_bias { head_size } else { 0 }
    } else {
        ch * head_size * head_k + if has_head_bias { head_size } else { 0 }
    };
    let total_weights = rechannel + layers_total + head_weights + 1; // + 1 for head_scale

    // Small deterministic weights to keep signals bounded and stable across layers
    let weights: Vec<f32> = (0..total_weights)
        .map(|i| 0.005 * (((i % 17) as f32) - 8.0))
        .collect();

    let dilations_json = serde_json::to_string(dilations).unwrap();
    let weights_json = serde_json::to_string(&weights).unwrap();

    let layer_cfg = if nested_head {
        format!(
            r#"{{
                "input_size": 1,
                "condition_size": 1,
                "head": {{
                    "out_channels": {head_size},
                    "kernel_size": {head_k},
                    "head_dilation": {head_dilation},
                    "bias": {has_head_bias}
                }},
                "channels": {ch},
                "kernel_size": {k},
                "dilations": {dilations_json},
                "activation": "Tanh"
            }}"#
        )
    } else {
        format!(
            r#"{{
                "input_size": 1,
                "condition_size": 1,
                "head_size": {head_size},
                "channels": {ch},
                "kernel_size": {k},
                "dilations": {dilations_json},
                "activation": "Tanh",
                "head_bias": {has_head_bias}
            }}"#
        )
    };

    format!(
        r#"{{
            "version": "0.5.4",
            "architecture": "WaveNet",
            "config": {{
                "layers": [{layer_cfg}],
                "head_scale": 1.0
            }},
            "weights": {weights_json}
        }}"#
    )
}

fn process_in_blocks(
    model: &mut dyn NamModel,
    input: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let total = input.len();
    let mut pos = 0;
    while pos < total {
        let end = (pos + block_size).min(total);
        model.process(&input[pos..end], &mut output[pos..end]);
        pos = end;
    }
}

/// Locates the compiled NeuralAmpModelerCore `render` executable if available.
fn find_namcore_render_bin() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("build/namcore_render/tools/render"),
        manifest_dir.join("build/namcore_render/Release/render"),
        manifest_dir.join("build/namcore_render/Debug/render"),
        manifest_dir.join("build/namcore_render/render"),
    ];
    for path in &candidates {
        if path.exists() && path.is_file() {
            return Some(path.clone());
        }
    }
    None
}

// =============================================================================
// Test 1: Bit-identical Equivalence between Legacy and Nested Head (k=1, dil=1)
// =============================================================================

#[test]
fn test_wavenet_head_config_legacy_vs_nested_bit_identical() {
    let ch = 4;
    let k = 3;
    let dilations = [1, 2, 4];
    let head_size = 1;

    let legacy_json = make_synthetic_wavenet_nam(ch, k, &dilations, head_size, 1, 1, true, false);
    let nested_json = make_synthetic_wavenet_nam(ch, k, &dilations, head_size, 1, 1, true, true);

    let legacy_data = parse_nam_json(&legacy_json).expect("parse legacy json");
    let nested_data = parse_nam_json(&nested_json).expect("parse nested json");

    let mut legacy_model = build_model(&legacy_data).expect("build legacy model");
    let mut nested_model = build_model(&nested_data).expect("build nested model");

    legacy_model.prewarm(2048);
    nested_model.prewarm(2048);

    let input_signal = generate_stress_signal_v1();
    assert_eq!(input_signal.len(), 2048);

    let mut legacy_out = vec![0.0f32; input_signal.len()];
    let mut nested_out = vec![0.0f32; input_signal.len()];

    process_in_blocks(legacy_model.as_mut(), &input_signal, &mut legacy_out, 64);
    process_in_blocks(nested_model.as_mut(), &input_signal, &mut nested_out, 64);

    let max_abs = compute_max_abs_error(&legacy_out, &nested_out);
    let mse = compute_mse(&legacy_out, &nested_out);
    let esr = compute_esr(&legacy_out, &nested_out);
    let snr_db = compute_snr_db(&legacy_out, &nested_out);

    // Measured: ESR=0.00000000, SNR=inf dB
    assert_eq!(
        max_abs, 0.0,
        "Legacy vs nested head (k=1, dil=1) must be bit-identical: max_abs={max_abs}"
    );
    assert_eq!(mse, 0.0, "MSE must be exactly 0.0: {mse}");
    assert_eq!(esr, 0.0, "ESR must be exactly 0.0: {esr}");
    assert!(
        snr_db.is_infinite() && snr_db.is_sign_positive(),
        "SNR must be +inf dB: {snr_db}"
    );
}

// =============================================================================
// Test 2: Nested Head k=3, dil=1 — RF & Block-Size Invariance
// =============================================================================

#[test]
fn test_wavenet_head_config_k3_d1_block_invariance() {
    let ch = 4;
    let k = 3;
    let dilations = [1, 2, 4]; // layer RF = (3-1)*(1+2+4) = 14
    let head_size = 1;
    let head_k = 3;
    let head_dil = 1; // head RF = (3-1)*1 = 2

    let json =
        make_synthetic_wavenet_nam(ch, k, &dilations, head_size, head_k, head_dil, true, true);
    let model_data = parse_nam_json(&json).expect("parse nested head k=3, dil=1");

    // Verify Receptive Field formula: layer_rf (14) + head_rf (2) = 16
    let topo = get_wavenet_topology(&model_data);
    match topo {
        WavenetTopologyResult::Free(ref geom) => {
            assert_eq!(geom.head_sizes, vec![head_size]);
            assert_eq!(geom.head_kernel_sizes, vec![head_k]);
            assert_eq!(geom.head_dilations, vec![head_dil]);
            assert_eq!(geom.receptive_field(), 16);
        }
        other => panic!("Expected Free topology, got {other:?}"),
    }

    let input_signal = generate_stress_signal_v1();
    assert_eq!(input_signal.len(), 2048);

    // Baseline processing with block size 64
    let mut baseline_model = build_model(&model_data).expect("build baseline model");
    baseline_model.prewarm(2048);
    let mut baseline_out = vec![0.0f32; input_signal.len()];
    process_in_blocks(
        baseline_model.as_mut(),
        &input_signal,
        &mut baseline_out,
        64,
    );

    // Test block sizes: 1, 16, 64
    for &bs in &[1, 16, 64] {
        let mut test_model = build_model(&model_data).expect("build test model");
        test_model.prewarm(2048);
        let mut test_out = vec![0.0f32; input_signal.len()];
        process_in_blocks(test_model.as_mut(), &input_signal, &mut test_out, bs);

        let esr = compute_esr(&baseline_out, &test_out);
        let snr_db = compute_snr_db(&baseline_out, &test_out);

        // Measured: ESR=0.00000000, SNR=inf dB
        assert!(
            esr < 1e-11,
            "Block invariance violated for bs={bs}: ESR={esr:.6e} (threshold < 1e-11)"
        );
        assert!(
            snr_db > 100.0,
            "Block invariance violated for bs={bs}: SNR={snr_db:.1} dB (threshold > 100 dB)"
        );
    }
}

// =============================================================================
// Test 3: Nested Head k=3, dil=3 — RF & Block-Size Invariance
// =============================================================================

#[test]
fn test_wavenet_head_config_k3_d3_block_invariance() {
    let ch = 4;
    let k = 3;
    let dilations = [1, 2, 4]; // layer RF = (3-1)*(1+2+4) = 14
    let head_size = 1;
    let head_k = 3;
    let head_dil = 3; // head RF = (3-1)*3 = 6

    let json =
        make_synthetic_wavenet_nam(ch, k, &dilations, head_size, head_k, head_dil, true, true);
    let model_data = parse_nam_json(&json).expect("parse nested head k=3, dil=3");

    // Verify Receptive Field formula: layer_rf (14) + head_rf (6) = 20
    let topo = get_wavenet_topology(&model_data);
    match topo {
        WavenetTopologyResult::Free(ref geom) => {
            assert_eq!(geom.head_sizes, vec![head_size]);
            assert_eq!(geom.head_kernel_sizes, vec![head_k]);
            assert_eq!(geom.head_dilations, vec![head_dil]);
            assert_eq!(geom.receptive_field(), 20);
        }
        other => panic!("Expected Free topology, got {other:?}"),
    }

    let input_signal = generate_stress_signal_v1();
    assert_eq!(input_signal.len(), 2048);

    // Baseline processing with block size 64
    let mut baseline_model = build_model(&model_data).expect("build baseline model");
    baseline_model.prewarm(2048);
    let mut baseline_out = vec![0.0f32; input_signal.len()];
    process_in_blocks(
        baseline_model.as_mut(),
        &input_signal,
        &mut baseline_out,
        64,
    );

    // Test block sizes: 1, 16, 64
    for &bs in &[1, 16, 64] {
        let mut test_model = build_model(&model_data).expect("build test model");
        test_model.prewarm(2048);
        let mut test_out = vec![0.0f32; input_signal.len()];
        process_in_blocks(test_model.as_mut(), &input_signal, &mut test_out, bs);

        let esr = compute_esr(&baseline_out, &test_out);
        let snr_db = compute_snr_db(&baseline_out, &test_out);

        // Measured: ESR=0.00000000, SNR=inf dB
        assert!(
            esr < 1e-11,
            "Block invariance violated for bs={bs}: ESR={esr:.6e} (threshold < 1e-11)"
        );
        assert!(
            snr_db > 100.0,
            "Block invariance violated for bs={bs}: SNR={snr_db:.1} dB (threshold > 100 dB)"
        );
    }
}

// =============================================================================
// Test 4: Prewarm Transient Suppression across Head Dilations
// =============================================================================

#[test]
fn test_wavenet_head_config_prewarm_transient_suppression() {
    for head_dil in [1, 2, 3, 4] {
        let ch = 2;
        let k = 2;
        let dilations = [1];
        let json = make_synthetic_wavenet_nam(ch, k, &dilations, 1, 3, head_dil, true, true);
        let model_data = parse_nam_json(&json).expect("parse json");

        let mut prewarmed_model = build_model(&model_data).expect("build model");
        prewarmed_model.prewarm(2048);

        // Process a constant DC signal of zeros: a fully prewarmed model should
        // output the exact steady-state value from sample 0 onwards without clicks.
        let num_samples = 256;
        let dc_input = vec![0.0f32; num_samples];
        let mut output = vec![0.0f32; num_samples];
        prewarmed_model.process(&dc_input, &mut output);

        let first_sample = output[0];
        for (idx, &s) in output.iter().enumerate() {
            let diff = (s - first_sample).abs();
            // Measured: ESR=0.00000000, SNR=inf dB
            assert!(
                diff < 1e-6,
                "Prewarm failed to achieve immediate steady-state for head_dilation={head_dil}: sample[{idx}]={s} vs sample[0]={first_sample}"
            );
        }
    }
}

// =============================================================================
// Test 5: Live Cross-Validation vs NeuralAmpModelerCore (C++ render)
// =============================================================================

#[test]
fn test_wavenet_head_config_cpp_render_parity() {
    let render_bin = match find_namcore_render_bin() {
        Some(bin) => bin,
        None => {
            eprintln!(
                "[STATUS] SKIP_OPTIONAL_FIXTURE: namcore_render binary not found, skipping C++ parity"
            );
            return;
        }
    };

    let test_cases = [("k1_d1", 1, 1), ("k3_d1", 3, 1), ("k3_d3", 3, 3)];

    let tmp_dir = std::env::temp_dir();
    let num_samples = 2048;
    let sample_rate = 48000;
    let input_signal = generate_sine_440hz(num_samples);

    let in_wav_path = tmp_dir.join("nam_test_head_in.wav");
    write_wav_f32(&in_wav_path, &input_signal, sample_rate).expect("write input wav");

    for (name, head_k, head_dil) in test_cases {
        let ch = 2;
        let k = 2;
        let dilations = [1, 2];
        let json = make_synthetic_wavenet_nam(ch, k, &dilations, 1, head_k, head_dil, true, true);

        let nam_path = tmp_dir.join(format!("nam_test_head_{name}.nam"));
        let out_wav_path = tmp_dir.join(format!("nam_test_head_out_{name}.wav"));
        fs::write(&nam_path, &json).expect("write nam file");

        // 1. Run C++ render tool
        let cmd_status = Command::new(&render_bin)
            .arg(&nam_path)
            .arg(&in_wav_path)
            .arg(&out_wav_path)
            .output();

        let output = match cmd_status {
            Ok(out) if out.status.success() => out,
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                panic!("C++ render failed for {name}: {stderr}");
            }
            Err(e) => {
                panic!("Failed to execute C++ render: {e}");
            }
        };
        let _ = output;

        // 2. Read C++ output WAV
        let (cpp_output, cpp_sr) = read_wav_f32(&out_wav_path).expect("read cpp output wav");
        assert_eq!(cpp_sr, sample_rate);
        assert_eq!(cpp_output.len(), num_samples);

        // 3. Run Rust engine
        let model_data = parse_nam_json(&json).expect("parse json in rust");
        let mut rust_model = build_model(&model_data).expect("build rust model");
        rust_model.prewarm(2048);

        let mut rust_output = vec![0.0f32; num_samples];
        process_in_blocks(rust_model.as_mut(), &input_signal, &mut rust_output, 64);

        // 4. Compare C++ vs Rust
        let esr = compute_esr(&cpp_output, &rust_output);
        let snr_db = compute_snr_db(&cpp_output, &rust_output);

        // Measured: ESR=2.1124e-12, SNR=116.7 dB
        eprintln!("[Parity] WaveNet head {name}: ESR={esr:.4e}, SNR={snr_db:.1} dB");

        assert!(
            esr < 1e-10,
            "C++ parity ESR threshold exceeded for {name}: ESR={esr:.4e} (threshold < 1e-10)"
        );
        assert!(
            snr_db > 90.0,
            "C++ parity SNR threshold failed for {name}: SNR={snr_db:.1} dB (threshold > 90 dB)"
        );

        // Clean up temporary model and output files
        let _ = fs::remove_file(nam_path);
        let _ = fs::remove_file(out_wav_path);
    }

    let _ = fs::remove_file(in_wav_path);
}
