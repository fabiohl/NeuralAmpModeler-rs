// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Hostile model parsing and DoS hardening integration tests.
//!
//! Verifies fail-closed resilience against adversarial model payloads:
//! - Arithmetic Hardening: Integer overflow & panic-free transposer and buffer allocation.
//! - DoS Bounds Enforcement: Bounds validation on A2 dynamic allocations (head channels, layer channels, head size).
//! - Degenerate Topologies: Rejection of invalid topologies (`hidden_size == 0`, `dilation == 0`).
//! - Fail-Closed Constructors: Safe `A2Conv1d::try_new` constructor and caller integration.

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{
    MAX_A2_HEAD_CHANNELS, MAX_HEAD_CHANNELS, MAX_HEAD_KERNEL_SIZE, MAX_HEAD_OUT_CHANNELS,
    WavenetTopologyResult, get_convnet_topology, get_lstm_topology, get_wavenet_topology,
    parse_nam_json,
};
use neural_amp_modeler_rs::loader::namb_encoder::ensure_capacity;
use neural_amp_modeler_rs::loader::transpose::wavenet::transpose_wavenet_interleaved4;
use neural_amp_modeler_rs::math::common::AlignedVec;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::a2::conv1d::A2Conv1d;

// ── Arithmetic Hardening & Integer Overflow Protection ────────────────────────

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

// ── DoS Hardening & Allocation Caps (A2 Dynamic) ──────────────────────────────

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

// ── A2-Dynamic `groups` Divisibility (F-RES2-01/F-RES2-06) ────────────────────

/// Single-array A2-Dynamic WaveNet JSON (channels=12, bottleneck=12,
/// non-gated LeakyReLU, one conv layer with kernel 3, layer-array head K=16)
/// with configurable group counts — the minimal topology that reaches the
/// grouped-conv/mixin/l1x1/head1x1 reshape arithmetic.
fn a2_dynamic_groups_json(
    groups_input: u64,
    mixin_groups: u64,
    l1x1_groups: u64,
    head1x1_groups: Option<u64>,
) -> String {
    let head1x1 = match head1x1_groups {
        None => "null".to_string(),
        Some(g) => format!(r#"{{"active": true, "out_channels": 12, "groups": {g}}}"#),
    };
    format!(
        r#"{{
            "version": "0.6.0",
            "architecture": "WaveNet",
            "config": {{
                "in_channels": 1,
                "head_scale": 0.02,
                "layers": [{{
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 12,
                    "bottleneck": 12,
                    "kernel_sizes": [3],
                    "dilations": [1],
                    "activation": [{{"type": "LeakyReLU", "negative_slope": 0.01}}],
                    "layer1x1": {{"active": true, "groups": {l1x1_groups}}},
                    "head1x1": {head1x1},
                    "head": {{"out_channels": 1, "kernel_size": 16, "bias": true}},
                    "groups_input": {groups_input},
                    "groups_input_mixin": {mixin_groups}
                }}]
            }},
            "weights": [],
            "sample_rate": 48000.0
        }}"#
    )
}

/// Weight stream length for the topology above with the given `groups_input`:
/// rechannel (1×12) + conv ((12×12/groups)×3 + bias 12) + mixin (12×1) +
/// layer1x1 (12×12 + bias 12) + head (16×12 + 1 + 1).
fn a2_dynamic_weight_count(groups_input: u64) -> usize {
    let groups_input = groups_input.max(1) as usize;
    12 + (12 * 12 / groups_input) * 3 + 12 + 12 + 144 + 12 + 192 + 1 + 1
}

/// `groups_input: 7` with `channels: 12` (12 % 7 == 5) must be rejected with
/// a typed `Err` at the parsing point — never a panic in any build profile.
#[test]
fn test_hostile_a2_groups_input_non_divisor_rejected() {
    let json = a2_dynamic_groups_json(7, 1, 1, None);
    let parsed = parse_nam_json(&json).expect("JSON syntax is valid");
    let build_res = build_model(&parsed)
        .err()
        .expect("non-divisor groups_input must be rejected, not panic");
    let err = build_res.to_string();
    assert!(
        err.contains("groups_input (7)"),
        "error should identify the non-divisor groups_input: {err}"
    );
    assert!(
        err.contains("hostile JSON rejection"),
        "rejection must follow the hostile-JSON pattern: {err}"
    );
}

/// `groups_input: 2` exactly divides `channels` and `conv_out` (both 12):
/// the grouped topology must build and run successfully with a well-formed
/// weight stream.
#[test]
fn test_hostile_a2_groups_input_divisor_builds() {
    let json = a2_dynamic_groups_json(2, 1, 1, None);
    let mut parsed = parse_nam_json(&json).expect("JSON syntax is valid");
    parsed.weights = vec![0.0f32; a2_dynamic_weight_count(2)];

    let mut model = build_model(&parsed)
        .expect("divisor groups_input (2 | 12) must build a valid A2-Dynamic model");
    model.prewarm(64);

    let input = [0.1f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);
    for &s in &output {
        assert!(s.is_finite(), "zero-weight model must emit finite samples");
    }
}

/// `groups_input_mixin: 3` with `condition_size: 1` does not divide the mixin
/// reshape operands — rejected at the parsing point.
#[test]
fn test_hostile_a2_mixin_groups_non_divisor_rejected() {
    let json = a2_dynamic_groups_json(1, 3, 1, None);
    let parsed = parse_nam_json(&json).expect("JSON syntax is valid");
    let err = build_model(&parsed)
        .err()
        .expect("non-divisor groups_input_mixin must be rejected")
        .to_string();
    assert!(
        err.contains("groups_input_mixin (3)"),
        "error should identify the non-divisor groups_input_mixin: {err}"
    );
}

/// `layer1x1.groups: 5` with bottleneck/channels 12 does not divide the
/// layer1x1 reshape operands — rejected at the parsing point.
#[test]
fn test_hostile_a2_l1x1_groups_non_divisor_rejected() {
    let json = a2_dynamic_groups_json(1, 1, 5, None);
    let parsed = parse_nam_json(&json).expect("JSON syntax is valid");
    let err = build_model(&parsed)
        .err()
        .expect("non-divisor layer1x1.groups must be rejected")
        .to_string();
    assert!(
        err.contains("layer1x1.groups (5)"),
        "error should identify the non-divisor layer1x1.groups: {err}"
    );
}

/// Active `head1x1` with `groups: 7` and bottleneck/out_channels 12 does not
/// divide the head1x1 reshape operands — rejected at the parsing point.
#[test]
fn test_hostile_a2_head1x1_groups_non_divisor_rejected() {
    let json = a2_dynamic_groups_json(1, 1, 1, Some(7));
    let parsed = parse_nam_json(&json).expect("JSON syntax is valid");
    let err = build_model(&parsed)
        .err()
        .expect("non-divisor head1x1.groups must be rejected")
        .to_string();
    assert!(
        err.contains("head1x1.groups (7)"),
        "error should identify the non-divisor head1x1.groups: {err}"
    );
}

// ── Degenerate Topology Validation (LSTM, ConvNet, WaveNet) ───────────────────

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

// ── A2Conv1d::try_new Fail-Closed Invariants ──────────────────────────────────

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

// ── Post-Stack Head Ceilings (F-RES2-02) ─────────────────────────────────────
//
// The `head` sub-object is the only loader dimension with a declared ceiling:
// `head.channels` / `head.out_channels` / `head.kernel_size` are capped at the
// canonical layer ceilings before any allocation or size multiplication.
// Each hostile dimension must fail closed (`Err`, never panic or wrap) on
// both WaveNet free-geometry and ConvNet `Layers` paths.

/// Builds a minimal WaveNet free-geometry JSON with the given `head` object
/// literal (e.g. `"null"` or `"{\"channels\": ...}"`).
fn wavenet_free_head_json(head_literal: &str) -> String {
    format!(
        r#"{{
            "version": "0.5.4",
            "architecture": "WaveNet",
            "config": {{
                "layers": [
                    {{
                        "input_size": 1, "condition_size": 1, "head_size": 4,
                        "channels": 8, "kernel_size": 3, "dilations": [1, 2],
                        "activation": "Tanh", "gated": false, "head_bias": false
                    }},
                    {{
                        "input_size": 1, "condition_size": 1, "head_size": 4,
                        "channels": 8, "kernel_size": 3, "dilations": [1, 2],
                        "activation": "Tanh", "gated": false, "head_bias": true
                    }}
                ],
                "head": {head_literal},
                "head_scale": 0.02
            }},
            "weights": [0.0],
            "sample_rate": 48000.0
        }}"#
    )
}

/// Builds a minimal ConvNet `Layers` JSON with the given `head` object literal.
fn convnet_layers_head_json(head_literal: &str) -> String {
    format!(
        r#"{{
            "version": "0.5.4",
            "architecture": "ConvNet",
            "config": {{
                "layers": [{{
                    "channels": 4, "kernel_size": 2, "dilations": [1],
                    "activation": "ReLU"
                }}],
                "head": {head_literal},
                "head_scale": 1.0
            }},
            "weights": [0.0],
            "sample_rate": 48000.0
        }}"#
    )
}

#[test]
fn test_hostile_head_channels_extreme_rejected_wavenet() {
    for extreme in [MAX_HEAD_CHANNELS + 1, usize::MAX] {
        let json = wavenet_free_head_json(&format!(
            r#"{{"channels": {extreme}, "bias": false, "out_channels": 1, "activation": "Tanh", "kernel_size": 1}}"#
        ));
        let parsed = parse_nam_json(&json).expect("JSON syntax is valid");
        match get_wavenet_topology(&parsed) {
            WavenetTopologyResult::Rejected(reason) => assert!(
                reason.contains("head"),
                "rejection should identify the head ceiling, got: {reason}"
            ),
            other => panic!("head.channels={extreme} must be rejected, got {other:?}"),
        }
        assert!(
            build_model(&parsed).is_err(),
            "head.channels={extreme} must fail build_model"
        );
    }
}

#[test]
fn test_hostile_head_out_channels_extreme_rejected_wavenet() {
    for extreme in [MAX_HEAD_OUT_CHANNELS + 1, usize::MAX] {
        let json = wavenet_free_head_json(&format!(
            r#"{{"channels": 4, "bias": false, "out_channels": {extreme}, "activation": "Tanh", "kernel_size": 1}}"#
        ));
        let parsed = parse_nam_json(&json).expect("JSON syntax is valid");
        match get_wavenet_topology(&parsed) {
            WavenetTopologyResult::Rejected(reason) => assert!(
                reason.contains("head"),
                "rejection should identify the head ceiling, got: {reason}"
            ),
            other => panic!("head.out_channels={extreme} must be rejected, got {other:?}"),
        }
        assert!(
            build_model(&parsed).is_err(),
            "head.out_channels={extreme} must fail build_model"
        );
    }
}

#[test]
fn test_hostile_head_kernel_extreme_rejected_wavenet() {
    for extreme in [MAX_HEAD_KERNEL_SIZE + 1, usize::MAX] {
        let json = wavenet_free_head_json(&format!(
            r#"{{"channels": 4, "bias": false, "out_channels": 1, "activation": "Tanh", "kernel_size": {extreme}}}"#
        ));
        let parsed = parse_nam_json(&json).expect("JSON syntax is valid");
        match get_wavenet_topology(&parsed) {
            WavenetTopologyResult::Rejected(reason) => assert!(
                reason.contains("head"),
                "rejection should identify the head ceiling, got: {reason}"
            ),
            other => panic!("head.kernel_size={extreme} must be rejected, got {other:?}"),
        }
        assert!(
            build_model(&parsed).is_err(),
            "head.kernel_size={extreme} must fail build_model"
        );
    }
}

#[test]
fn test_hostile_head_channels_extreme_rejected_convnet() {
    for extreme in [MAX_HEAD_CHANNELS + 1, usize::MAX] {
        let json = convnet_layers_head_json(&format!(
            r#"{{"channels": {extreme}, "bias": false, "out_channels": 1, "activation": "Tanh", "kernel_size": 1}}"#
        ));
        let parsed = parse_nam_json(&json).expect("JSON syntax is valid");
        assert!(
            get_convnet_topology(&parsed).is_none(),
            "ConvNet head.channels={extreme} must be rejected by get_convnet_topology"
        );
        assert!(
            build_model(&parsed).is_err(),
            "ConvNet head.channels={extreme} must fail build_model"
        );
    }
}

#[test]
fn test_hostile_head_out_channels_extreme_rejected_convnet() {
    for extreme in [MAX_HEAD_OUT_CHANNELS + 1, usize::MAX] {
        let json = convnet_layers_head_json(&format!(
            r#"{{"channels": 4, "bias": false, "out_channels": {extreme}, "activation": "Tanh", "kernel_size": 1}}"#
        ));
        let parsed = parse_nam_json(&json).expect("JSON syntax is valid");
        assert!(
            get_convnet_topology(&parsed).is_none(),
            "ConvNet head.out_channels={extreme} must be rejected by get_convnet_topology"
        );
        assert!(
            build_model(&parsed).is_err(),
            "ConvNet head.out_channels={extreme} must fail build_model"
        );
    }
}

#[test]
fn test_hostile_head_kernel_extreme_rejected_convnet() {
    for extreme in [MAX_HEAD_KERNEL_SIZE + 1, usize::MAX] {
        let json = convnet_layers_head_json(&format!(
            r#"{{"channels": 4, "bias": false, "out_channels": 1, "activation": "Tanh", "kernel_size": {extreme}}}"#
        ));
        let parsed = parse_nam_json(&json).expect("JSON syntax is valid");
        assert!(
            get_convnet_topology(&parsed).is_none(),
            "ConvNet head.kernel_size={extreme} must be rejected by get_convnet_topology"
        );
        assert!(
            build_model(&parsed).is_err(),
            "ConvNet head.kernel_size={extreme} must fail build_model"
        );
    }
}

/// A small valid `head` must keep building on both paths (guard against
/// ceiling regressions on legitimate models).
#[test]
fn test_valid_head_at_ceiling_builds() {
    let wavenet = wavenet_free_head_json(&format!(
        r#"{{"channels": {MAX_HEAD_CHANNELS}, "bias": false, "out_channels": 1, "activation": "Tanh", "kernel_size": 1}}"#
    ));
    let parsed = parse_nam_json(&wavenet).expect("JSON syntax is valid");
    assert!(
        matches!(
            get_wavenet_topology(&parsed),
            WavenetTopologyResult::Known(_) | WavenetTopologyResult::Free(_)
        ),
        "head at the ceiling must stay accepted"
    );

    let convnet = convnet_layers_head_json(
        r#"{"channels": 4, "bias": false, "out_channels": 1, "activation": "Tanh", "kernel_size": 1}"#,
    );
    let parsed_conv = parse_nam_json(&convnet).expect("JSON syntax is valid");
    assert!(
        get_convnet_topology(&parsed_conv).is_some(),
        "small valid ConvNet head must stay accepted"
    );
}
