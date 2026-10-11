// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Loading-time activation contract: the declared activation is executed
//! exactly, or the model is rejected with a typed error.
//!
//! Pins two behaviours introduced by the NAMcore parity remediation:
//!
//! 1. A known activation is applied faithfully (no silent `Tanh` fallback).
//!    The canonical `LeakyReLU` reproduction must yield the C++ result.
//! 2. An unknown activation name is rejected with
//!    [`NamErrorCode::NamJsonUnknownActivation`] (`E1221`) in every family
//!    (ConvNet, WaveNet A1 standard, WaveNet A1 dynamic, post-stack head).

use neural_amp_modeler_rs::common::diagnostics::NamErrorCode;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::perceptual::{compute_esr, compute_snr_db};

use super::common::io_helpers::{cpp_render_mono, require_render_or_skip};

const BLOCK: usize = 64;

/// ConvNet FlatCpp format with `channels=1`, a single dilation and no
/// batch norm. Weight layout: `[conv_w, conv_b, head_w, head_b]`
/// (`conv_w` is `out*in*kernel = 1*1*2`, C++ hardcodes `kernel_size=2`).
fn convnet_flat_json(activation: &str) -> String {
    format!(
        r#"{{
            "version": "0.5.4",
            "architecture": "ConvNet",
            "config": {{
                "channels": 1,
                "dilations": [1],
                "batchnorm": false,
                "activation": "{activation}"
            }},
            "weights": [0.0, 1.0, -1.0, 1.0, 0.0],
            "sample_rate": 48000
        }}"#
    )
}

/// Minimal free-geometry WaveNet A1 (2 arrays, non-catalog channels) whose
/// first layer declares `first_activation` and whose optional head may declare
/// its own activation.
fn wavenet_a1_free_json(first_activation: &str, head_activation: Option<&str>) -> String {
    let head = match head_activation {
        Some(a) => format!(
            r#"{{"channels": 2, "out_channels": 1, "kernel_size": 1, "bias": false, "activation": "{a}"}}"#
        ),
        None => "null".to_string(),
    };
    format!(
        r#"{{
            "version": "0.5.4",
            "architecture": "WaveNet",
            "config": {{
                "layers": [
                    {{
                        "channels": 2, "kernel_size": 3, "head_size": 1,
                        "dilations": [1], "gated": false, "head_bias": false,
                        "activation": "{first_activation}"
                    }},
                    {{
                        "channels": 2, "kernel_size": 3, "head_size": 1,
                        "dilations": [1], "gated": false, "head_bias": true,
                        "activation": "Tanh"
                    }}
                ],
                "head": {head},
                "head_scale": 0.02
            }},
            "weights": [0.0],
            "sample_rate": 48000
        }}"#
    )
}

/// Catalog-matching WaveNet Standard geometry (16 channels, canonical
/// dilations) so the const-generic A1 static builder is exercised.
fn wavenet_standard_json(activation: &str) -> String {
    let dils = "1,2,4,8,16,32,64,128,256,512";
    format!(
        r#"{{
            "version": "0.5.4",
            "architecture": "WaveNet",
            "config": {{
                "layers": [
                    {{
                        "channels": 16, "kernel_size": 3, "head_size": 8,
                        "dilations": [{dils}], "gated": false, "head_bias": false,
                        "activation": "{activation}"
                    }},
                    {{
                        "channels": 16, "kernel_size": 3, "head_size": 8,
                        "dilations": [{dils}], "gated": false, "head_bias": true,
                        "activation": "{activation}"
                    }}
                ],
                "head": null,
                "head_scale": 0.02
            }},
            "weights": [0.0],
            "sample_rate": 48000
        }}"#
    )
}

fn build(json: &str) -> anyhow::Result<Box<neural_amp_modeler_rs::models::StaticModel>> {
    let data = parse_nam_json(json).expect("JSON parses");
    build_model(&data)
}

fn build_err(json: &str) -> anyhow::Error {
    match build(json) {
        Ok(_) => panic!("expected model build to be rejected: {json}"),
        Err(e) => e,
    }
}

#[test]
fn convnet_leaky_relu_matches_cpp() {
    // Canonical reproduction: conv bias of -1 through LeakyReLU(0.01) yields
    // -0.01, *not* tanh(-1) ≈ -0.761594. A silent Tanh fallback would fail here.
    let json = convnet_flat_json("LeakyReLU");
    let mut model = build(&json).expect("LeakyReLU must be accepted");
    let input = [0.0f32; BLOCK];
    let mut output = [0.0f32; BLOCK];
    model.process(&input, &mut output);

    for (i, &s) in output.iter().enumerate() {
        assert!(
            (s - (-0.01)).abs() < 1e-6,
            "frame {i}: expected -0.01 (C++ LeakyReLU), got {s}"
        );
    }

    // Authoritative oracle: the NAMcore render tool must produce the same
    // output for a non-trivial excitation, proving the LeakyReLU reproduction
    // matches C++ beyond the single -0.01 constant case.
    let Some(render_bin) = require_render_or_skip("convnet leaky-relu c++ parity") else {
        return;
    };
    let sample_rate = 48000u32;
    let exc: Vec<f32> = (0..BLOCK * 8)
        .map(|i| 0.5 * ((i as f32) * 0.07).sin())
        .collect();
    let cpp = cpp_render_mono(&render_bin, &json, &exc, sample_rate, "cnn_lrelu");

    let mut cnn = build(&json).expect("rebuild for render parity");
    cnn.reset(sample_rate, BLOCK).expect("reset convnet");
    let mut rust_out = vec![0.0f32; exc.len()];
    for (ic, oc) in exc.chunks(BLOCK).zip(rust_out.chunks_mut(BLOCK)) {
        cnn.process(ic, oc);
    }
    let esr = compute_esr(&cpp, &rust_out);
    let snr = compute_snr_db(&cpp, &rust_out);
    assert!(
        esr < 1e-4,
        "ConvNet LeakyReLU ESR too high vs C++ render: {esr}"
    );
    assert!(
        snr > 80.0,
        "ConvNet LeakyReLU SNR too low vs C++ render: {snr} dB"
    );
}

#[test]
fn convnet_unknown_activation_is_rejected() {
    let err = build_err(&convnet_flat_json("Gelu"));
    assert_eq!(
        err.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err}"
    );
}

#[test]
fn wavenet_a1_standard_unknown_activation_is_rejected() {
    let err = build_err(&wavenet_standard_json("Gelu"));
    assert_eq!(
        err.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err}"
    );
}

#[test]
fn wavenet_a1_dynamic_unknown_activation_is_rejected() {
    let err = build_err(&wavenet_a1_free_json("Gelu", None));
    assert_eq!(
        err.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err}"
    );
}

#[test]
fn post_stack_head_unknown_activation_is_rejected() {
    // ConvNet `layers` format reaches the post-stack head builder before reading
    // the head weights, so a 4-float block weight budget suffices.
    let json = r#"{
        "version": "0.5.4",
        "architecture": "ConvNet",
        "config": {
            "layers": [
                {"channels": 1, "kernel_size": 1, "dilations": [1], "activation": "Tanh"}
            ],
            "head": {"channels": 1, "out_channels": 1, "kernel_size": 1, "bias": false, "activation": "Gelu"},
            "head_scale": 1.0
        },
        "weights": [0.0, 0.0, 0.0, 0.0],
        "sample_rate": 48000
    }"#;
    let err = build_err(json);
    assert_eq!(
        err.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err}"
    );
}

#[test]
fn wavenet_a2_dynamic_unknown_activation_is_rejected() {
    let path = crate::common::model_path("wavenet_a2_lite.nam");
    let json_str = std::fs::read_to_string(&path).expect("read fixture");
    let mut val: serde_json::Value = serde_json::from_str(&json_str).expect("parse json");
    val["config"]["layers"][0]["activation"] = serde_json::json!("Gelu");
    let err = build_err(&val.to_string());
    assert_eq!(
        err.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err}"
    );
}

#[test]
fn wavenet_a2_cascade_unknown_activation_is_rejected() {
    let path = crate::common::model_path("wavenet_a2_max.nam");
    let json_str = std::fs::read_to_string(&path).expect("read fixture");
    let mut val: serde_json::Value = serde_json::from_str(&json_str).expect("parse json");
    val["config"]["layers"][0]["activation"] = serde_json::json!("Gelu");
    let err = build_err(&val.to_string());
    assert_eq!(
        err.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err}"
    );
}

#[test]
fn wavenet_a2_unknown_object_is_rejected() {
    let path = crate::common::model_path("wavenet_a2_lite.nam");
    let json_str = std::fs::read_to_string(&path).expect("read fixture");
    let mut val: serde_json::Value = serde_json::from_str(&json_str).expect("parse json");
    // Single object with unknown type
    val["config"]["layers"][0]["activation"] = serde_json::json!({"type": "UnknownActivation"});
    let err = build_err(&val.to_string());
    assert_eq!(
        err.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err}"
    );

    // Array of 23 unknown objects
    val["config"]["layers"][0]["activation"] =
        serde_json::json!(vec![serde_json::json!({"type": "UnknownActivation"}); 23]);
    let err_arr = build_err(&val.to_string());
    assert_eq!(
        err_arr.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err_arr}"
    );
}

#[test]
fn convnet_layers_activation_object_and_array_contracts() {
    let base = r#"{
        "version": "0.5.4",
        "architecture": "ConvNet",
        "config": {
            "layers": [
                {"channels": 1, "kernel_size": 1, "dilations": [1], "activation": ACT}
            ],
            "head": null,
            "head_scale": 1.0
        },
        "weights": [0.0, 0.0, 0.0, 0.0, 0.0],
        "sample_rate": 48000
    }"#;

    // 1. Unknown object -> E1221
    let json_err = base.replace("ACT", r#"{"type": "Gelu"}"#);
    let err = build_err(&json_err);
    assert_eq!(
        err.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err}"
    );

    // 2. Array -> E1221
    let json_err_arr = base.replace("ACT", r#"["Tanh", "Tanh"]"#);
    let err_arr = build_err(&json_err_arr);
    assert_eq!(
        err_arr.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err_arr}"
    );

    // 3. Null -> E1221
    let json_err_null = base.replace("ACT", "null");
    let err_null = build_err(&json_err_null);
    assert_eq!(
        err_null.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err_null}"
    );

    // 4. Valid object -> builds successfully
    let json_ok = base.replace("ACT", r#"{"type": "ReLU"}"#);
    assert!(build(&json_ok).is_ok());
}

#[test]
fn wavenet_a1_layers_activation_object_and_array_contracts() {
    let make_json = |act: &str| {
        format!(
            r#"{{
                "version": "0.5.4",
                "architecture": "WaveNet",
                "config": {{
                    "layers": [
                        {{
                            "channels": 2, "kernel_size": 3, "head_size": 1,
                            "dilations": [1], "gated": false, "head_bias": false,
                            "activation": {act}
                        }},
                        {{
                            "channels": 2, "kernel_size": 3, "head_size": 1,
                            "dilations": [1], "gated": false, "head_bias": true,
                            "activation": "Tanh"
                        }}
                    ],
                    "head": null,
                    "head_scale": 0.02
                }},
                "weights": [0.0],
                "sample_rate": 48000
            }}"#
        )
    };

    // 1. Valid object Tanh -> passes activation validation (fails later on weights if weights insufficient, but never E1215 or E1221)
    if let Err(e) = build(&make_json(r#"{"type": "Tanh"}"#)) {
        assert_ne!(
            e.downcast_ref::<NamErrorCode>(),
            Some(&NamErrorCode::NamJsonUnknownActivation),
            "valid Tanh object must not trigger E1221"
        );
        assert_ne!(
            e.downcast_ref::<NamErrorCode>(),
            Some(&NamErrorCode::NamJsonUnsupportedTopology),
            "valid Tanh object must not trigger E1215"
        );
    }

    // 2. Unsupported valid NAM activation (FastTanh) -> E1215
    let err_topo = build_err(&make_json(r#"{"type": "FastTanh"}"#));
    assert_eq!(
        err_topo.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnsupportedTopology),
        "expected E1215, got: {err_topo}"
    );

    // 3. Unknown object -> E1221
    let err_unknown = build_err(&make_json(r#"{"type": "Gelu"}"#));
    assert_eq!(
        err_unknown.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err_unknown}"
    );

    // 4. Array -> E1221
    let err_arr = build_err(&make_json(r#"["Tanh", "Tanh"]"#));
    assert_eq!(
        err_arr.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err_arr}"
    );

    // 5. Null -> E1221
    let err_null = build_err(&make_json("null"));
    assert_eq!(
        err_null.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonUnknownActivation),
        "expected E1221, got: {err_null}"
    );
}
