// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Hostile model parsing and DoS hardening integration tests (Sprint 6 / Epic F).
//!
//! Verifies fail-closed resilience against adversarial model payloads:
//! - F1 (R-10): Integer overflow & panic-free transposer and buffer allocation.
//! - F2 (R-11): DoS bounds enforcement on A2 dynamic allocations (head channels, layer channels, head size).
//! - F3 (R-12): Rejection of degenerate topologies (`hidden_size == 0`, `dilation == 0`).
//! - F4 (R-13): Fail-closed `A2Conv1d::try_new` constructor and caller integration.

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{
    MAX_A2_HEAD_CHANNELS, WavenetTopologyResult, get_convnet_topology, get_lstm_topology,
    get_wavenet_topology, parse_nam_json,
};
use neural_amp_modeler_rs::loader::namb_encoder::ensure_capacity;
use neural_amp_modeler_rs::loader::transpose::wavenet::transpose_wavenet_interleaved4;
use neural_amp_modeler_rs::math::common::AlignedVec;
use neural_amp_modeler_rs::models::a2::conv1d::A2Conv1d;

// ── F1: Arithmetic Hardening & Integer Overflow Protection ───────────────────

#[test]
fn test_namb_ensure_capacity_overflow_rejected() {
    let buf: Vec<f32> = Vec::new();
    // Attempt cursor + needed that overflows usize
    let res = ensure_capacity(&buf, usize::MAX - 10, 20, "test_overflow".to_string());
    assert!(res.is_err(), "ensure_capacity must fail on usize overflow");
    let err = res.unwrap_err();
    assert!(
        err.to_string().contains("Overflow"),
        "error must explicitly report integer overflow: {err}"
    );
}

#[test]
fn test_wavenet_transpose_overflow_checked() {
    let hostile_json = r#"{
        "version": "0.5.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [{
                "input_size": 18446744073709551615,
                "condition_size": 1,
                "channels": 18446744073709551615,
                "kernel_size": 3,
                "dilations": [1],
                "activation": "Tanh"
            }],
            "head_size": 16
        },
        "weights": [0.0],
        "sample_rate": 48000.0
    }"#;
    let data = parse_nam_json(hostile_json).expect("valid JSON syntax");
    let res = transpose_wavenet_interleaved4(&data);
    assert!(
        res.is_err(),
        "transpose_wavenet_interleaved4 must fail gracefully on integer overflow"
    );
}

#[test]
fn test_wavenet_transpose_truncated_weights_rejected() {
    let valid_json = r#"{
        "version": "0.5.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "channels": 16,
                "kernel_size": 3,
                "dilations": [1],
                "activation": "Tanh"
            }],
            "head_size": 16
        },
        "weights": [0.0],
        "sample_rate": 48000.0
    }"#;
    let data = parse_nam_json(valid_json).expect("valid JSON syntax");
    let res = transpose_wavenet_interleaved4(&data);
    assert!(
        res.is_err(),
        "transpose_wavenet_interleaved4 must reject truncated raw weights"
    );
}

// ── F2: DoS Hardening & Allocation Caps (A2 Dynamic) ─────────────────────────

#[test]
fn test_hostile_a2_oversized_head_channels_rejected() {
    let hostile_json = format!(
        r#"{{
            "version": "0.6.0",
            "architecture": "WaveNet",
            "config": {{
                "layers": [{{
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 16,
                    "bottleneck": 16,
                    "kernel_sizes": [3],
                    "dilations": [1],
                    "activation": "LeakyReLU",
                    "head1x1": {{
                        "active": true,
                        "out_channels": {}
                    }}
                }}],
                "head_size": 16,
                "head_scale": 1.0
            }},
            "weights": [],
            "sample_rate": 48000.0
        }}"#,
        MAX_A2_HEAD_CHANNELS + 1
    );

    let parsed = parse_nam_json(&hostile_json).expect("JSON syntax is valid");
    let build_res = build_model(&parsed);
    assert!(
        build_res.is_err(),
        "A2 model with head1x1_out_channels > MAX_A2_HEAD_CHANNELS must be rejected"
    );
    let err = build_res.err().unwrap().to_string();
    assert!(
        err.contains("head1x1_out_channels"),
        "error message should cite head1x1_out_channels bounds: {err}"
    );
}

#[test]
fn test_hostile_a2_zero_head_channels_rejected() {
    let hostile_json = r#"{
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "channels": 16,
                "bottleneck": 16,
                "kernel_sizes": [3],
                "dilations": [1],
                "activation": "LeakyReLU",
                "head1x1": {
                    "active": true,
                    "out_channels": 0
                }
            }],
            "head_size": 16,
            "head_scale": 1.0
        },
        "weights": [],
        "sample_rate": 48000.0
    }"#;

    let parsed = parse_nam_json(hostile_json).expect("JSON syntax is valid");
    let build_res = build_model(&parsed);
    assert!(
        build_res.is_err(),
        "A2 model with head1x1_out_channels == 0 must be rejected"
    );
    let err = build_res.err().unwrap().to_string();
    assert!(
        err.contains("head1x1_out_channels"),
        "error message should cite head1x1_out_channels bounds: {err}"
    );
}

#[test]
fn test_hostile_a2_oversized_channels_rejected() {
    let hostile_json = r#"{
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "channels": 256,
                "bottleneck": 16,
                "kernel_sizes": [3],
                "dilations": [1],
                "activation": "LeakyReLU"
            }],
            "head_size": 16,
            "head_scale": 1.0
        },
        "weights": [],
        "sample_rate": 48000.0
    }"#;

    let parsed = parse_nam_json(hostile_json).expect("JSON syntax is valid");
    let build_res = build_model(&parsed);
    assert!(
        build_res.is_err(),
        "A2 model with channels > MAX_A2_DYN_CHANNELS must be rejected"
    );
}

#[test]
fn test_hostile_a2_oversized_head_size_rejected() {
    let hostile_json = r#"{
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "channels": 16,
                "bottleneck": 16,
                "kernel_sizes": [3],
                "dilations": [1],
                "activation": "LeakyReLU"
            }],
            "head_size": 2048,
            "head_scale": 1.0
        },
        "weights": [],
        "sample_rate": 48000.0
    }"#;

    let parsed = parse_nam_json(hostile_json).expect("JSON syntax is valid");
    let build_res = build_model(&parsed);
    assert!(
        build_res.is_err(),
        "A2 model with head_size > MAX_HEAD_SIZE must be rejected"
    );
}

// ── F3: Degenerate Topology Validation (LSTM, ConvNet, WaveNet) ──────────────

#[test]
fn test_degenerate_lstm_zero_hidden_size_rejected() {
    let hostile_json = r#"{
        "version": "0.5.0",
        "architecture": "LSTM",
        "config": {
            "num_layers": 1,
            "input_size": 1,
            "hidden_size": 0
        },
        "weights": [],
        "sample_rate": 48000.0
    }"#;

    let parsed = parse_nam_json(hostile_json).expect("JSON syntax is valid");
    let topo_res = get_lstm_topology(&parsed);
    assert!(
        topo_res.is_err(),
        "LSTM with hidden_size == 0 must be rejected by get_lstm_topology"
    );
    let err = topo_res.err().unwrap().to_string();
    assert!(
        err.contains("hidden_size=0"),
        "expected error to mention zero hidden_size: {err}"
    );

    let build_res = build_model(&parsed);
    assert!(
        build_res.is_err(),
        "LSTM with hidden_size == 0 must fail build_model"
    );
}

#[test]
fn test_degenerate_convnet_zero_dilation_rejected() {
    // 1. FlatCpp format: dilations array containing 0
    let flat_json = r#"{
        "version": "0.5.0",
        "architecture": "ConvNet",
        "config": {
            "channels": 4,
            "dilations": [0, 1],
            "batchnorm": false,
            "activation": "ReLU"
        },
        "weights": [],
        "sample_rate": 48000.0
    }"#;

    let parsed_flat = parse_nam_json(flat_json).expect("JSON syntax is valid");
    assert!(
        get_convnet_topology(&parsed_flat).is_none(),
        "ConvNet flat format with dilation == 0 must return None"
    );
    assert!(
        build_model(&parsed_flat).is_err(),
        "ConvNet flat format with dilation == 0 must fail build_model"
    );

    // 2. Layers format: layer dilations containing 0
    let layers_json = r#"{
        "version": "0.5.0",
        "architecture": "ConvNet",
        "config": {
            "layers": [{
                "channels": 4,
                "kernel_size": 2,
                "dilations": [1, 0, 2],
                "activation": "ReLU"
            }]
        },
        "weights": [],
        "sample_rate": 48000.0
    }"#;

    let parsed_layers = parse_nam_json(layers_json).expect("JSON syntax is valid");
    assert!(
        get_convnet_topology(&parsed_layers).is_none(),
        "ConvNet layers format with dilation == 0 must return None"
    );
    assert!(
        build_model(&parsed_layers).is_err(),
        "ConvNet layers format with dilation == 0 must fail build_model"
    );
}

#[test]
fn test_degenerate_wavenet_zero_dilation_rejected() {
    let hostile_json = r#"{
        "version": "0.5.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "channels": 16,
                "kernel_size": 3,
                "dilations": [1, 0, 4],
                "activation": "Tanh"
            }],
            "head_size": 16
        },
        "weights": [],
        "sample_rate": 48000.0
    }"#;

    let parsed = parse_nam_json(hostile_json).expect("JSON syntax is valid");
    match get_wavenet_topology(&parsed) {
        WavenetTopologyResult::Rejected(reason) => {
            assert!(
                reason.contains("dilation") && reason.contains("0"),
                "expected rejection reason to identify zero dilation: {reason}"
            );
        }
        other => panic!("expected WavenetTopologyResult::Rejected, got {other:?}"),
    }

    let build_res = build_model(&parsed);
    assert!(
        build_res.is_err(),
        "WaveNet with layer dilation == 0 must fail build_model"
    );
}

// ── F4: A2Conv1d::try_new Fail-Closed Invariants ─────────────────────────────

#[test]
fn test_a2_conv1d_try_new_fail_closed_validation() {
    let weights = AlignedVec::new(64, 0.0f32).unwrap();
    let bias = AlignedVec::new(16, 0.0f32).unwrap();

    // Zero kernel size
    assert!(
        A2Conv1d::try_new(weights.clone(), bias.clone(), true, 1, 4, 4, 0).is_err(),
        "zero kernel_size must be rejected"
    );

    // Zero input channels
    assert!(
        A2Conv1d::try_new(weights.clone(), bias.clone(), true, 1, 0, 4, 3).is_err(),
        "zero in_ch must be rejected"
    );

    // Zero output channels
    assert!(
        A2Conv1d::try_new(weights.clone(), bias.clone(), true, 1, 4, 0, 3).is_err(),
        "zero out_ch must be rejected"
    );

    // Undersized bias
    let small_bias = AlignedVec::new(2, 0.0f32).unwrap();
    assert!(
        A2Conv1d::try_new(weights, small_bias, true, 1, 4, 8, 3).is_err(),
        "undersized bias must be rejected"
    );
}
