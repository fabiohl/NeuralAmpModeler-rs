// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Integration test to verify the encode→decode round-trip of `.namb` v2 layouts.
//
//  Covers 11 topologies (7 LSTM + 4 WaveNet) across layouts:
//  - `Original` — no transposition, bit-exact identical
//  - `GateMajorLstm` — LSTM weights pre-transposed [Gate][IH][H]
//  - `Interleaved4WaveNet` — WaveNet weights interleaved 4-wide

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{
    NamConfig, NamLayerConfig, NamModelData, NamWavenetTopology, WeightsLayout, parse_nam_json,
};
use neural_amp_modeler_rs::loader::namb::parse_namb;
use neural_amp_modeler_rs::loader::namb_encoder::encode_namb;
use neural_amp_modeler_rs::math::activations::ActivationPrecision;
use neural_amp_modeler_rs::models::NamModel;
use std::fs;
use std::path::PathBuf;

use crate::common::PrecisionGuard;
use crate::common::generate_sine_440hz as generate_sine;

// =============================================================================
// Shared helpers
// =============================================================================

fn model_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(filename);
    path
}

fn compute_mse(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum();
    sum / (n as f64)
}

// =============================================================================
// LSTM — Synthetic model builder (pre-existing)
// =============================================================================

fn make_synthetic_lstm(num_layers: usize, hidden_size: usize) -> NamModelData {
    let mut weights = Vec::new();
    let mut current_input_size = 1;
    let mut val = 0.05f32;

    for _ in 0..num_layers {
        let ih = current_input_size + hidden_size;
        let w_size = 4 * hidden_size * ih;
        for _ in 0..w_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        let b_size = 4 * hidden_size;
        for _ in 0..b_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        for _ in 0..hidden_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        for _ in 0..hidden_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        current_input_size = hidden_size;
    }

    for _ in 0..hidden_size {
        weights.push(val);
        val = (val + 0.007) % 0.3;
    }
    weights.push(0.01);

    NamModelData {
        version: Some("0.5.0".to_string()),
        architecture: "LSTM".to_string(),
        config: NamConfig {
            layers: vec![],
            head: None,
            head_scale: Some(1.0),
            num_layers: Some(num_layers),
            hidden_size: Some(hidden_size),
            receptive_field: None,
            bias: None,
            submodels: None,
            ..Default::default()
        },
        weights,
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

// =============================================================================
// LSTM — Expected GateMajor weight computation (independent verification)
// =============================================================================

/// Mirrors `transpose_lstm_gate_major` from the encoder for bit-exact verification.
fn expected_gate_major_weights(data: &NamModelData) -> Vec<f32> {
    let hidden_size = data.config.hidden_size.unwrap();
    let num_layers = data.config.num_layers.unwrap_or(1);

    let mut cursor = 0;
    let mut out = Vec::with_capacity(data.weights.len());
    let mut current_input_size = 1;

    for _l in 0..num_layers {
        let ih = current_input_size + hidden_size;
        let h = hidden_size;
        let layer_size = 4 * h * ih;

        let raw = &data.weights[cursor..cursor + layer_size];
        let mut transposed = vec![0.0f32; layer_size];
        for k in 0..4 {
            for i in 0..h {
                for j in 0..ih {
                    let val = raw[k * h * ih + i * ih + j];
                    transposed[k * h * ih + j * h + i] = val;
                }
            }
        }
        out.extend(transposed);
        cursor += layer_size;

        // bias (pass-through)
        let bias_size = 4 * h;
        out.extend_from_slice(&data.weights[cursor..cursor + bias_size]);
        cursor += bias_size;

        // hidden_init + cell_init (pass-through)
        let state_size = 2 * h;
        out.extend_from_slice(&data.weights[cursor..cursor + state_size]);
        cursor += state_size;

        current_input_size = hidden_size;
    }

    // head_weights + head_bias (pass-through)
    if cursor < data.weights.len() {
        out.extend_from_slice(&data.weights[cursor..]);
    }

    out
}

// =============================================================================
// WaveNet — Helpers and synthetic model builder
// =============================================================================

static STD_DILATIONS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
static LITE_DILATIONS_0: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
static LITE_DILATIONS_1: &[usize] = &[128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

/// Configuration for each supported WaveNet topology.
struct WavenetTopoCfg {
    channels: usize,
    head: usize,
    dilations_0: &'static [usize],
    dilations_1: &'static [usize],
    topology: NamWavenetTopology,
}

fn wavenet_topologies() -> Vec<WavenetTopoCfg> {
    vec![
        WavenetTopoCfg {
            channels: 16,
            head: 8,
            dilations_0: STD_DILATIONS,
            dilations_1: STD_DILATIONS,
            topology: NamWavenetTopology::Standard,
        },
        WavenetTopoCfg {
            channels: 12,
            head: 6,
            dilations_0: LITE_DILATIONS_0,
            dilations_1: LITE_DILATIONS_1,
            topology: NamWavenetTopology::Lite,
        },
        WavenetTopoCfg {
            channels: 8,
            head: 4,
            dilations_0: LITE_DILATIONS_0,
            dilations_1: LITE_DILATIONS_1,
            topology: NamWavenetTopology::Feather,
        },
        WavenetTopoCfg {
            channels: 4,
            head: 2,
            dilations_0: LITE_DILATIONS_0,
            dilations_1: LITE_DILATIONS_1,
            topology: NamWavenetTopology::Nano,
        },
    ]
}

/// Builds a synthetic WaveNet `NamModelData` with deterministic weights.
///
/// Weight layout (Original / JSON):
/// ```text
/// [array1.rechannel[CH*1]]
///   for each dilation: [conv1d[CH*CH*K]][bias[CH]][mixin[CH*1]][1x1[CH*CH]][1x1_bias[CH]]
/// [array1.head_rechannel[HEAD*CH]]
///
/// [array2.rechannel[HEAD*CH]]       (IN=CH_array1, CH=HEAD)
///   for each dilation: [conv1d[HEAD*HEAD*K]][bias[HEAD]][mixin[HEAD*1]][1x1[HEAD*HEAD]][1x1_bias[HEAD]]
/// [array2.head_rechannel[1*HEAD]][head_bias[1]]
/// [head_scale[1]]
/// ```
fn make_synthetic_wavenet(cfg: &WavenetTopoCfg) -> NamModelData {
    let ch = cfg.channels;
    let head = cfg.head;
    const K: usize = 3;

    let mut weights = Vec::new();
    let mut val = 0.05f32;

    let mut push_seq = |n: usize| {
        for _ in 0..n {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
    };

    // ---- Array 1 ----
    push_seq(ch); // rechannel

    for _ in 0..cfg.dilations_0.len() {
        push_seq(ch * ch * K); // conv1d
        push_seq(ch); // bias
        push_seq(ch); // mixin
        push_seq(ch * ch); // 1x1
        push_seq(ch); // 1x1 bias
    }

    push_seq(head * ch); // head_rechannel (no bias: has_head_bias=false)

    // ---- Array 2 (IN=ch, CH=head, HEAD=1) ----
    push_seq(head * ch); // rechannel (IN=ch, CH=head)

    for _ in 0..cfg.dilations_1.len() {
        push_seq(head * head * K); // conv1d
        push_seq(head); // bias
        push_seq(head); // mixin
        push_seq(head * head); // 1x1
        push_seq(head); // 1x1 bias
    }

    push_seq(head); // head_rechannel (has_head_bias=true, weights)
    weights.push(0.04); // head_bias
    weights.push(0.06); // head_scale

    NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![
                NamLayerConfig {
                    input_size: Some(1),
                    condition_size: Some(1),
                    head_size: Some(head),
                    channels: Some(ch),
                    kernel_size: Some(K),
                    dilations: Some(cfg.dilations_0.to_vec()),
                    activation: Some("Tanh".to_string()),
                    gated: Some(false),
                    head_bias: Some(false),
                    ..Default::default()
                },
                NamLayerConfig {
                    input_size: Some(ch),
                    condition_size: Some(1),
                    head_size: Some(1),
                    channels: Some(head),
                    kernel_size: Some(K),
                    dilations: Some(cfg.dilations_1.to_vec()),
                    activation: Some("Tanh".to_string()),
                    gated: Some(false),
                    head_bias: Some(true),
                    ..Default::default()
                },
            ],
            head: None,
            head_scale: None,
            num_layers: None,
            hidden_size: None,
            receptive_field: None,
            bias: None,
            submodels: None,
            ..Default::default()
        },
        weights,
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

// =============================================================================
// WaveNet — Expected Interleaved4 weight computation (independent verification)
// =============================================================================

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Mirrors `transpose_wavenet_interleaved4` from the encoder for bit-exact verification.
fn expected_interleaved4_weights(data: &NamModelData) -> Vec<f32> {
    let mut cursor = 0;
    let mut out = Vec::with_capacity(data.weights.len());

    for layer_cfg in &data.config.layers {
        let in_ch = layer_cfg.input_size.unwrap_or(1);
        let ch = layer_cfg.channels.unwrap_or(16);
        let cond_ch = layer_cfg.condition_size.unwrap_or(1);
        let k = layer_cfg.kernel_size.unwrap_or(3);
        let head_ch = layer_cfg.head_size.unwrap_or(8);
        let dilations = layer_cfg.dilations.as_ref().unwrap();
        let gated = layer_cfg.gated.unwrap_or(false);
        let conv_out_ch = if gated { 2 * ch } else { ch };

        // 1. Rechannel: [OUT][IN] -> [IN][OUT]
        let size = ch * in_ch;
        let raw = &data.weights[cursor..cursor + size];
        for in_c in 0..in_ch {
            for out_c in 0..ch {
                out.push(raw[out_c * in_ch + in_c]);
            }
        }
        cursor += size;

        // 2. Convolution layers
        for _ in 0..dilations.len() {
            // Conv1D: [OUT][IN][K] -> interleaved 4-wide
            let size = conv_out_ch * ch * k;
            let raw = &data.weights[cursor..cursor + size];
            let num_blocks = ceil_div(conv_out_ch, 4);
            for b in 0..num_blocks {
                for ki in 0..k {
                    for in_c in 0..ch {
                        for lane in 0..4 {
                            let out_c = b * 4 + lane;
                            if out_c < conv_out_ch {
                                out.push(raw[(out_c * ch + in_c) * k + ki]);
                            } else {
                                out.push(0.0);
                            }
                        }
                    }
                }
            }
            cursor += size;

            // Bias (pass-through)
            out.extend_from_slice(&data.weights[cursor..cursor + conv_out_ch]);
            cursor += conv_out_ch;

            // Input Mixin: [OUT][COND] -> [COND][OUT]
            let size = ch * cond_ch;
            let raw = &data.weights[cursor..cursor + size];
            for in_c in 0..cond_ch {
                for out_c in 0..ch {
                    out.push(raw[out_c * cond_ch + in_c]);
                }
            }
            cursor += size;

            // 1x1: [OUT][IN] -> [IN][OUT]
            let size = ch * ch;
            let raw = &data.weights[cursor..cursor + size];
            for in_c in 0..ch {
                for out_c in 0..ch {
                    out.push(raw[out_c * ch + in_c]);
                }
            }
            cursor += size;

            // 1x1 Bias (pass-through)
            out.extend_from_slice(&data.weights[cursor..cursor + ch]);
            cursor += ch;
        }

        // 3. Head Rechannel
        let size = head_ch * ch;
        let raw = &data.weights[cursor..cursor + size];
        for in_c in 0..ch {
            for out_c in 0..head_ch {
                out.push(raw[out_c * ch + in_c]);
            }
        }
        cursor += size;

        // Head Rechannel Bias (pass-through)
        if layer_cfg.head_bias.unwrap_or(false) {
            out.extend_from_slice(&data.weights[cursor..cursor + head_ch]);
            cursor += head_ch;
        }
    }

    // head_scale (pass-through)
    if cursor < data.weights.len() {
        out.extend_from_slice(&data.weights[cursor..]);
    }

    out
}

// =============================================================================
// Tests — Original Round-trip (LSTM + WaveNet)
// =============================================================================

/// Original round-trip: encode without transposition → decode → identical weights.
fn test_original_roundtrip(data: &NamModelData) {
    let namb = encode_namb(data, 2, WeightsLayout::Original).unwrap();
    let decoded = parse_namb(&namb).unwrap();

    assert_eq!(
        decoded.weights_layout,
        WeightsLayout::Original,
        "Layout must be Original after round-trip"
    );
    assert_eq!(
        decoded.weights.len(),
        data.weights.len(),
        "Weight count must be preserved"
    );
    assert!(
        decoded.weights == data.weights,
        "Weights must be bit-exact identical in Original round-trip"
    );
}

#[test]
fn test_lstm_original_roundtrip() {
    let topologies = [
        (1, 8),
        (1, 12),
        (1, 16),
        (1, 24),
        (1, 40),
        (2, 8),
        (2, 12),
        (2, 16),
        (2, 24),
    ];
    for &(layers, hidden) in &topologies {
        let data = make_synthetic_lstm(layers, hidden);
        test_original_roundtrip(&data);
    }
}

#[test]
fn test_wavenet_original_roundtrip() {
    for cfg in &wavenet_topologies() {
        let data = make_synthetic_wavenet(cfg);
        test_original_roundtrip(&data);
    }
}

// =============================================================================
// Tests — GateMajorLstm Round-trip with bit-exact weight verification
// =============================================================================

#[test]
fn test_lstm_gate_major_roundtrip() {
    let _guard = PrecisionGuard::new(ActivationPrecision::Standard);

    let topologies = [
        (1, 8),
        (1, 12),
        (1, 16),
        (1, 24),
        (1, 40),
        (2, 8),
        (2, 12),
        (2, 16),
        (2, 24),
    ];
    for &(num_layers, hidden_size) in &topologies {
        let orig_data = make_synthetic_lstm(num_layers, hidden_size);

        // 1. Encode → decode
        let namb = encode_namb(&orig_data, 2, WeightsLayout::GateMajorLstm).unwrap();
        let decoded = parse_namb(&namb).unwrap();
        assert_eq!(decoded.weights_layout, WeightsLayout::GateMajorLstm);

        // 2. Bit-exact verification of transposed weights
        let expected = expected_gate_major_weights(&orig_data);
        assert_eq!(
            decoded.weights, expected,
            "GateMajor weights must be bit-exact identical to expected for LSTM {}x{}",
            num_layers, hidden_size
        );

        // 3. Inference (sanity check)
        let mut model_orig = build_model(&orig_data).unwrap();
        model_orig.prewarm(1024);
        let mut model_v2 = build_model(&decoded).unwrap();
        model_v2.prewarm(1024);

        let input = generate_sine(512);
        let mut out_orig = vec![0.0f32; 512];
        let mut out_v2 = vec![0.0f32; 512];
        model_orig.process(&input, &mut out_orig);
        model_v2.process(&input, &mut out_v2);

        let mse = compute_mse(&out_orig, &out_v2);
        assert!(
            mse < 1e-12,
            "LSTM {}x{} GateMajor divergence! MSE = {:e}",
            num_layers,
            hidden_size,
            mse
        );
    }
}

// =============================================================================
// Tests — Interleaved4WaveNet Round-trip with bit-exact verification
// =============================================================================

#[test]
fn test_wavenet_interleaved4_roundtrip() {
    let _guard = PrecisionGuard::new(ActivationPrecision::Standard);

    for cfg in &wavenet_topologies() {
        let orig_data = make_synthetic_wavenet(cfg);

        // 1. Encode → decode
        let namb = encode_namb(&orig_data, 2, WeightsLayout::Interleaved4WaveNet).unwrap();
        let decoded = parse_namb(&namb).unwrap();
        assert_eq!(decoded.weights_layout, WeightsLayout::Interleaved4WaveNet);

        // 2. Bit-exact verification of transposed weights
        let expected = expected_interleaved4_weights(&orig_data);
        assert_eq!(
            decoded.weights, expected,
            "Interleaved4 weights must be bit-exact identical to expected for WaveNet {:?}",
            cfg.topology
        );
        assert_eq!(
            decoded.weights.len(),
            expected.len(),
            "Weight count preserved for WaveNet {:?}",
            cfg.topology
        );

        // 3. Inference (sanity check)
        let mut model_orig = build_model(&orig_data).unwrap();
        model_orig.prewarm(2048);
        let mut model_v2 = build_model(&decoded).unwrap();
        model_v2.prewarm(2048);

        let input = generate_sine(512);
        let mut out_orig = vec![0.0f32; 512];
        let mut out_v2 = vec![0.0f32; 512];
        model_orig.process(&input, &mut out_orig);
        model_v2.process(&input, &mut out_v2);

        let mse = compute_mse(&out_orig, &out_v2);
        assert!(
            mse < 1e-10,
            "WaveNet {:?} Interleaved4 divergence! MSE = {:e}",
            cfg.topology,
            mse
        );
    }
}

// =============================================================================
// Test with real model (pre-existing, preserved)
// =============================================================================

#[test]
fn test_real_lstm_2x8_roundtrip() {
    let path = model_path("BossLSTM-2x8.nam");
    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).unwrap();
    let original_data = parse_nam_json(&json_data).unwrap();

    let mut model_orig = build_model(&original_data).unwrap();
    model_orig.prewarm(1024);

    let namb_v2 = encode_namb(&original_data, 2, WeightsLayout::GateMajorLstm).unwrap();

    let v2_data = parse_namb(&namb_v2).unwrap();
    assert_eq!(v2_data.weights_layout, WeightsLayout::GateMajorLstm);

    let mut model_v2 = build_model(&v2_data).unwrap();
    model_v2.prewarm(1024);

    let _guard = PrecisionGuard::new(ActivationPrecision::Standard);

    let input = generate_sine(512);
    let mut out_orig = vec![0.0f32; 512];
    let mut out_v2 = vec![0.0f32; 512];

    model_orig.process(&input, &mut out_orig);
    model_v2.process(&input, &mut out_v2);

    let mse = compute_mse(&out_orig, &out_v2);
    println!("[BossLSTM-2x8 v2 Parity] MSE: {:.2e}", mse);
    assert!(
        mse < 1e-12,
        "Real model BossLSTM-2x8 divergence! MSE={:e}",
        mse
    );
}
