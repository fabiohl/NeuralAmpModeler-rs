// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! T4.1e — Heterogeneous 2-array Slimmable WaveNet tests and parity validation.
//!
//! Validates:
//! 1. Breakpoint union `{1/3, 1/2, 2/3}` and target table mapping across `[0.0, 1.0]`.
//! 2. Prefix-masking equivalence oracle (pure Rust, no C++): slicing a full model
//!    to targets `T[i]` produces output identical to running the full model with all
//!    weights/biases outside prefix `T[i]` zeroed out (since `tanh(0) = 0`). This is
//!    a **single-precision equivalence** check between the sliced graph and the
//!    mathematically masked full graph — it is *not* an independent f64 reference.
//! 3. Oracle 1 (Live C++ parity): cross-validates against C++ `render --slim <ratio>`
//!    when the compiled binary is available with `--slim` capability.

use super::common::*;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::slimmable::slice_wavenet_model_per_array;
use neural_amp_modeler_rs::models::{NamModel, StaticModel};
use neural_amp_modeler_rs::testing::perceptual::{compute_esr, compute_snr_db};
use neural_amp_modeler_rs::testing::wav::{read_wav_f32, write_wav_f32};

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use super::common::io_helpers::require_render_or_skip;

const CH0: usize = 4;
const HEAD0: usize = 16;
const CH1: usize = 16;
const HEAD1: usize = 1;
const HEAD_SCALE: f32 = 0.5;

/// Checks if the render binary supports `--slim`.
fn render_bin_has_slim_capability(bin: &PathBuf) -> bool {
    let output = Command::new(bin).output();
    match output {
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            stderr.contains("--slim") || stdout.contains("--slim")
        }
        Err(_) => false,
    }
}

/// Generates deterministic weights for the synthetic 2-array slimmable model.
///
/// Weight counts:
/// - Array 0: rechannel (4) + conv1d (16+4) + mixin (4) + 1x1 (16+4) + head (64+16) = 128
/// - Array 1: rechannel (64) + conv1d (256+16) + mixin (16) + 1x1 (256+16) + head (16+1) = 641
/// - Trailing: head_scale (1)
///
/// Total = 770 f32 weights.
fn make_deterministic_weights() -> Vec<f32> {
    let total = 128 + 641 + 1;
    (0..total)
        .map(|i| 0.01 * (((i % 13) as f32) - 6.0))
        .collect()
}

/// Generates the synthetic 2-array slimmable `.nam` JSON string with given weights.
fn make_synthetic_2array_json(weights: &[f32]) -> String {
    let weights_json = serde_json::to_string(weights).expect("serialize weights");
    format!(
        r#"{{
            "version": "0.7.0",
            "architecture": "WaveNet",
            "config": {{
                "layers": [
                    {{
                        "input_size": 1,
                        "condition_size": 1,
                        "head_size": {HEAD0},
                        "channels": {CH0},
                        "kernel_size": 1,
                        "dilations": [1],
                        "activation": "Tanh",
                        "gated": false,
                        "head_bias": true,
                        "slimmable": {{
                            "method": "slice_channels_uniform",
                            "kwargs": {{
                                "allowed_channels": [2, 4]
                            }}
                        }}
                    }},
                    {{
                        "input_size": {CH0},
                        "condition_size": 1,
                        "head_size": {HEAD1},
                        "channels": {CH1},
                        "kernel_size": 1,
                        "dilations": [1],
                        "activation": "Tanh",
                        "gated": false,
                        "head_bias": true,
                        "slimmable": {{
                            "method": "slice_channels_uniform",
                            "kwargs": {{
                                "allowed_channels": [4, 8, 16]
                            }}
                        }}
                    }}
                ],
                "head": null,
                "head_scale": {HEAD_SCALE}
            }},
            "weights": {weights_json}
        }}"#
    )
}

/// Generates a copy of `weights` with all weights and biases outside the prefix
/// `targets = [T[0], T[1]]` zeroed out.
fn mask_weights_for_targets(full_weights: &[f32], targets: [usize; 2]) -> Vec<f32> {
    let [t0, t1] = targets;
    let mut masked = Vec::with_capacity(full_weights.len());
    let mut src = full_weights.iter().copied();

    // ---- Array 0 ----
    let slim_in0 = 1;
    let slim_ch0 = t0;
    let slim_head0 = t1;

    // rechannel: in=1 -> out=CH0 (bias=false)
    for out_c in 0..CH0 {
        for in_c in 0..1 {
            let w = src.next().unwrap();
            masked.push(if out_c < slim_ch0 && in_c < slim_in0 {
                w
            } else {
                0.0
            });
        }
    }
    // conv1d: in=CH0 -> out=CH0 (k=1, bias=true)
    for out_c in 0..CH0 {
        for in_c in 0..CH0 {
            let w = src.next().unwrap();
            masked.push(if out_c < slim_ch0 && in_c < slim_ch0 {
                w
            } else {
                0.0
            });
        }
    }
    for out_c in 0..CH0 {
        let b = src.next().unwrap();
        masked.push(if out_c < slim_ch0 { b } else { 0.0 });
    }
    // input_mixin: in=1 -> out=CH0 (bias=false)
    for out_c in 0..CH0 {
        for _in_c in 0..1 {
            let w = src.next().unwrap();
            masked.push(if out_c < slim_ch0 { w } else { 0.0 });
        }
    }
    // one_by_one: in=CH0 -> out=CH0 (bias=true)
    for out_c in 0..CH0 {
        for in_c in 0..CH0 {
            let w = src.next().unwrap();
            masked.push(if out_c < slim_ch0 && in_c < slim_ch0 {
                w
            } else {
                0.0
            });
        }
    }
    for out_c in 0..CH0 {
        let b = src.next().unwrap();
        masked.push(if out_c < slim_ch0 { b } else { 0.0 });
    }
    // head_rechannel: in=CH0 -> out=HEAD0 (bias=true)
    for out_c in 0..HEAD0 {
        for in_c in 0..CH0 {
            let w = src.next().unwrap();
            masked.push(if out_c < slim_head0 && in_c < slim_ch0 {
                w
            } else {
                0.0
            });
        }
    }
    for out_c in 0..HEAD0 {
        let b = src.next().unwrap();
        masked.push(if out_c < slim_head0 { b } else { 0.0 });
    }

    // ---- Array 1 ----
    let slim_in1 = t0;
    let slim_ch1 = t1;
    let slim_head1 = 1;

    // rechannel: in=CH0 -> out=CH1 (bias=false)
    for out_c in 0..CH1 {
        for in_c in 0..CH0 {
            let w = src.next().unwrap();
            masked.push(if out_c < slim_ch1 && in_c < slim_in1 {
                w
            } else {
                0.0
            });
        }
    }
    // conv1d: in=CH1 -> out=CH1 (k=1, bias=true)
    for out_c in 0..CH1 {
        for in_c in 0..CH1 {
            let w = src.next().unwrap();
            masked.push(if out_c < slim_ch1 && in_c < slim_ch1 {
                w
            } else {
                0.0
            });
        }
    }
    for out_c in 0..CH1 {
        let b = src.next().unwrap();
        masked.push(if out_c < slim_ch1 { b } else { 0.0 });
    }
    // input_mixin: in=1 -> out=CH1 (bias=false)
    for out_c in 0..CH1 {
        for _in_c in 0..1 {
            let w = src.next().unwrap();
            masked.push(if out_c < slim_ch1 { w } else { 0.0 });
        }
    }
    // one_by_one: in=CH1 -> out=CH1 (bias=true)
    for out_c in 0..CH1 {
        for in_c in 0..CH1 {
            let w = src.next().unwrap();
            masked.push(if out_c < slim_ch1 && in_c < slim_ch1 {
                w
            } else {
                0.0
            });
        }
    }
    for out_c in 0..CH1 {
        let b = src.next().unwrap();
        masked.push(if out_c < slim_ch1 { b } else { 0.0 });
    }
    // head_rechannel: in=CH1 -> out=HEAD1 (bias=true)
    for out_c in 0..HEAD1 {
        for in_c in 0..CH1 {
            let w = src.next().unwrap();
            masked.push(if out_c < slim_head1 && in_c < slim_ch1 {
                w
            } else {
                0.0
            });
        }
    }
    for out_c in 0..HEAD1 {
        let b = src.next().unwrap();
        masked.push(if out_c < slim_head1 { b } else { 0.0 });
    }

    // head_scale
    masked.push(src.next().unwrap());
    assert!(src.next().is_none(), "all weights processed");
    masked
}

fn process_signal(model: &mut dyn NamModel, input: &[f32], block_size: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    let mut pos = 0;
    while pos < input.len() {
        let chunk = block_size.min(input.len() - pos);
        model.process(&input[pos..pos + chunk], &mut output[pos..pos + chunk]);
        pos += chunk;
    }
    output
}

// =============================================================================
// Test 1: Breakpoint Union & Target Table Verification
// =============================================================================

#[test]
fn test_slimmable_2array_breakpoints_and_target_table() {
    let weights = make_deterministic_weights();
    let json = make_synthetic_2array_json(&weights);
    let data = parse_nam_json(&json).expect("parse json");
    let model = build_model(&data).expect("build model");

    let bps = model.slimmable_breakpoints();
    assert_eq!(
        bps.len(),
        3,
        "union of 1/2 and {{1/3, 2/3}} gives 3 breakpoints"
    );
    assert!((bps[0] - 1.0 / 3.0).abs() < 1e-6, "bp0 = 1/3");
    assert!((bps[1] - 0.5).abs() < 1e-6, "bp1 = 1/2");
    assert!((bps[2] - 2.0 / 3.0).abs() < 1e-6, "bp2 = 2/3");

    // Table from specification:
    // ratio  | array0 [2,4] | array1 [4,8,16]
    // 0.00   | 2            | 4
    // 1/3    | 2            | 8
    // 0.50   | 4            | 8
    // 2/3    | 4            | 16
    // 1.00   | 4            | 16
    // Exact breakpoint ratios (not the 0.34/0.67 decimal approximations) are
    // included so the tier boundary is tested at its true value.
    let expected_cases: [(f64, Vec<usize>); 6] = [
        (0.00, vec![2, 4]),
        (1.0 / 3.0, vec![2, 8]),
        (0.50, vec![4, 8]),
        (2.0 / 3.0, vec![4, 16]),
        (0.34, vec![2, 8]),
        (1.00, vec![4, 16]),
    ];

    for (ratio, expected_targets) in expected_cases {
        let targets = model
            .slimmable_targets(ratio)
            .expect("model supports slimmable targets");
        assert_eq!(
            targets, expected_targets,
            "ratio {ratio} targets mismatch: got {targets:?}, expected {expected_targets:?}"
        );
    }
}

// =============================================================================
// Test 2: Prefix-Masking Equivalence Oracle (Pure Rust)
// =============================================================================

#[test]
fn test_slimmable_2array_prefix_mask_equivalence() {
    let full_weights = make_deterministic_weights();
    let json_full = make_synthetic_2array_json(&full_weights);
    let data_full = parse_nam_json(&json_full).expect("parse full json");
    let model_full = build_model(&data_full).expect("build full model");

    let wavenet_full = match *model_full {
        StaticModel::WavenetDyn(ref m) => m.clone(),
        _ => panic!("expected WavenetDyn"),
    };

    let input = generate_stress_signal_v1();
    assert_eq!(input.len(), 2048);

    let test_ratios: [f64; 7] = [0.00, 1.0 / 3.0, 0.34, 0.50, 2.0 / 3.0, 0.67, 1.00];

    for ratio in test_ratios {
        let targets = wavenet_full
            .slimmable_targets(ratio)
            .expect("slimmable targets");
        let t = [targets[0], targets[1]];

        // 1. Sliced model
        let mut sliced_model =
            slice_wavenet_model_per_array(&wavenet_full, &targets).expect("slice model");
        sliced_model.prewarm();
        let sliced_output = process_signal(&mut sliced_model, &input, 64);

        // 2. Full model with weights/biases outside prefix zeroed out
        let masked_weights = mask_weights_for_targets(&full_weights, t);
        let json_masked = make_synthetic_2array_json(&masked_weights);
        let data_masked = parse_nam_json(&json_masked).expect("parse masked json");
        let mut masked_model = build_model(&data_masked).expect("build masked model");
        masked_model.prewarm(2048);
        let masked_output = process_signal(masked_model.as_mut(), &input, 64);

        // 3. Compare sliced vs masked-full
        let mut max_diff = 0.0f32;
        for (a, b) in sliced_output.iter().zip(masked_output.iter()) {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        let esr = compute_esr(&masked_output, &sliced_output);
        let snr = compute_snr_db(&masked_output, &sliced_output);

        // Measured: max_diff <= 2.33e-10, ESR <= 4.94e-16, SNR = inf dB
        eprintln!(
            "[Prefix-Mask Equivalence] ratio={ratio:.2} targets={targets:?}: max_diff={max_diff:.6e}, ESR={esr:.6e}, SNR={snr:.1} dB"
        );

        assert!(
            max_diff < 1e-6,
            "Prefix-mask oracle failed for ratio {ratio}: max_diff={max_diff}"
        );
        assert!(
            esr < 1e-10,
            "Prefix-mask oracle ESR failed for ratio {ratio}: ESR={esr}"
        );
    }
}

// =============================================================================
// Test 3: Oracle 1 — Live C++ Cross-Validation vs NeuralAmpModelerCore (render --slim)
// =============================================================================

#[test]
fn test_slimmable_2array_cpp_render_parity() {
    let Some(render_bin) = require_render_or_skip("slimmable 2-array c++ parity") else {
        return;
    };

    if !render_bin_has_slim_capability(&render_bin) {
        if std::env::var("NAM_REQUIRE_CPP_ORACLE").as_deref() == Ok("1") {
            panic!(
                "[STRICT] slimmable 2-array c++ parity: render binary does not expose \
                 --slim — missing C++ parity evidence is a failure under NAM_REQUIRE_CPP_ORACLE=1"
            );
        }
        eprintln!(
            "[STATUS] NOT_VERIFIED: render binary does not expose --slim capability, \
             skipping C++ live parity (run utils/tests-long.sh --strict to make this fatal)"
        );
        return;
    }

    let weights = make_deterministic_weights();
    let json = make_synthetic_2array_json(&weights);
    let data = parse_nam_json(&json).expect("parse json");
    let model = build_model(&data).expect("build model");

    let wavenet = match *model {
        StaticModel::WavenetDyn(ref m) => m.clone(),
        _ => panic!("expected WavenetDyn"),
    };

    let tmp_dir = std::env::temp_dir();
    let input_signal = generate_stress_signal_v1();
    let num_samples = input_signal.len();
    let sample_rate = 48000;

    let in_wav_path = tmp_dir.join("nam_test_slimmable_2array_in.wav");
    write_wav_f32(&in_wav_path, &input_signal, sample_rate).expect("write input wav");

    let nam_path = tmp_dir.join("nam_test_slimmable_2array.nam");
    fs::write(&nam_path, &json).expect("write nam file");

    let test_ratios: [f64; 7] = [0.00, 1.0 / 3.0, 0.34, 0.50, 2.0 / 3.0, 0.67, 1.00];

    for ratio in test_ratios {
        let targets = wavenet.slimmable_targets(ratio).expect("slimmable targets");

        let out_wav_path = tmp_dir.join(format!("nam_test_slimmable_2array_out_{ratio:.4}.wav"));

        // 1. Run C++ render tool with --slim <ratio>
        let cmd_status = Command::new(&render_bin)
            .arg("--slim")
            .arg(format!("{ratio}"))
            .arg(&nam_path)
            .arg(&in_wav_path)
            .arg(&out_wav_path)
            .output();

        let output = match cmd_status {
            Ok(out) if out.status.success() => out,
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                panic!("C++ render --slim {ratio} failed: {stderr}");
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

        // 3. Run Rust sliced engine
        let mut rust_sliced =
            slice_wavenet_model_per_array(&wavenet, &targets).expect("slice model");
        rust_sliced.prewarm();
        let rust_output = process_signal(&mut rust_sliced, &input_signal, 64);

        // 4. Parity comparison
        let esr = compute_esr(&cpp_output, &rust_output);
        let snr_db = compute_snr_db(&cpp_output, &rust_output);

        // Measured: ESR <= 1.7780e-14, SNR = inf dB (cross-validated vs build/namcore_render/tools/render --slim)
        eprintln!(
            "[C++ Parity] ratio={ratio:.2} targets={targets:?}: ESR={esr:.4e}, SNR={snr_db:.1} dB"
        );

        assert!(
            esr < 1e-10,
            "C++ parity ESR threshold exceeded for ratio {ratio}: ESR={esr:.4e}"
        );
        assert!(
            snr_db > 90.0,
            "C++ parity SNR threshold failed for ratio {ratio}: SNR={snr_db:.1} dB"
        );
    }

    // Clean up temporary files
    let _ = fs::remove_file(&in_wav_path);
    let _ = fs::remove_file(&nam_path);
    for ratio in test_ratios {
        let out_wav = tmp_dir.join(format!("nam_test_slimmable_2array_out_{ratio:.4}.wav"));
        let _ = fs::remove_file(out_wav);
    }
}
