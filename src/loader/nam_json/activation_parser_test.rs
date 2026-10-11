// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::A2_NUM_LAYERS;

fn make_a2_layer_json() -> serde_json::Value {
    let mut activations = Vec::new();
    for _ in 0..A2_NUM_LAYERS {
        activations.push(serde_json::json!({
            "type": "LeakyReLU",
            "negative_slope": 0.01
        }));
    }
    let mut gating = Vec::new();
    for _ in 0..A2_NUM_LAYERS {
        gating.push(serde_json::json!("none"));
    }
    let secondary: Vec<serde_json::Value> = vec![serde_json::Value::Null; A2_NUM_LAYERS];

    serde_json::json!({
        "activation": activations,
        "gating_mode": gating,
        "secondary_activation": secondary
    })
}

#[test]
fn test_parse_standard_a2_activations() {
    let raw = make_a2_layer_json();
    let config = parse_layer_activations(&raw, A2_NUM_LAYERS).expect("should parse");

    assert_eq!(config.activations.len(), A2_NUM_LAYERS);
    for act in &config.activations {
        assert_eq!(
            act,
            &ActivationType::LeakyReLU {
                negative_slope: 0.01
            }
        );
    }
    for mode in &config.gating_modes {
        assert_eq!(mode, &GatingMode::None);
    }
    for sec in &config.secondary_activations {
        assert!(
            sec.is_none(),
            "expected None secondary activation, got {sec:?}"
        );
    }
}

#[test]
fn test_parse_heterogeneous_activations() {
    let raw = serde_json::json!({
        "activation": [
            {"type": "LeakyReLU", "negative_slope": 0.01},
            {"type": "Tanh"},
            {"type": "ReLU"},
            {"type": "Sigmoid"},
            {"type": "SiLU"},
            {"type": "HardTanh"},
            {"type": "FastTanh"},
            {"type": "Softsign"},
            {"type": "HardSwish"},
            {"type": "PReLU", "negative_slopes": [0.01]},
            {"type": "LeakyHardTanh", "min_val": -1.0, "max_val": 1.0, "min_slope": 0.01, "max_slope": 0.01},
            {"type": "Tanh"},
            {"type": "ReLU"},
            {"type": "Tanh"},
            {"type": "Sigmoid"},
            {"type": "SiLU"},
            {"type": "Tanh"},
            {"type": "HardTanh"},
            {"type": "LeakyReLU", "negative_slope": 0.02},
            {"type": "FastTanh"},
            {"type": "Softsign"},
            {"type": "LeakyReLU", "negative_slope": 0.01},
            {"type": "Tanh"}
        ]
    });

    let activations = parse_activations_from_json(&raw, 23)
        .expect("should parse heterogeneous activations")
        .expect("should return Some vector");
    assert_eq!(activations.len(), 23);
    assert_eq!(
        activations[0],
        ActivationType::LeakyReLU {
            negative_slope: 0.01
        }
    );
    assert_eq!(activations[1], ActivationType::Tanh);
    assert_eq!(activations[2], ActivationType::ReLU);
    assert_eq!(activations[3], ActivationType::Sigmoid);
    assert_eq!(activations[4], ActivationType::SiLU);
    assert_eq!(
        activations[18],
        ActivationType::LeakyReLU {
            negative_slope: 0.02
        }
    );
}

#[test]
fn test_parse_gating_modes() {
    let raw = serde_json::json!({
        "gating_mode": [
            "none", "gated", "blended", "none", "none",
            "none", "none", "none", "none", "none",
            "none", "none", "none", "none", "none",
            "none", "none", "none", "none", "none",
            "gated", "blended", "none"
        ]
    });

    let modes = parse_gating_modes_from_json(&raw, 23);
    assert_eq!(modes.len(), 23);
    assert_eq!(modes[0], GatingMode::None);
    assert_eq!(modes[1], GatingMode::Gated);
    assert_eq!(modes[2], GatingMode::Blended);
    assert_eq!(modes[20], GatingMode::Gated);
    assert_eq!(modes[21], GatingMode::Blended);
    assert_eq!(modes[22], GatingMode::None);
}

#[test]
fn test_parse_gating_modes_absent() {
    let raw = serde_json::json!({});
    let modes = parse_gating_modes_from_json(&raw, 23);
    assert_eq!(modes.len(), 23);
    for mode in &modes {
        assert_eq!(mode, &GatingMode::None);
    }
}

#[test]
fn test_parse_gating_modes_null() {
    let raw = serde_json::json!({"gating_mode": null});
    let modes = parse_gating_modes_from_json(&raw, 23);
    assert_eq!(modes.len(), 23);
    for mode in &modes {
        assert_eq!(mode, &GatingMode::None);
    }
}

#[test]
fn test_parse_secondary_activations() {
    let raw = serde_json::json!({
        "secondary_activation": [
            null,
            {"type": "Sigmoid"},
            {"type": "Tanh"},
            null, null, null, null, null, null, null,
            null, null, null, null, null, null, null,
            null, null, null, null, null, null
        ]
    });

    let sec = parse_secondary_activations_from_json(&raw, 23);
    assert_eq!(sec.len(), 23);
    assert!(sec[0].is_none());
    assert_eq!(sec[1], Some(ActivationType::Sigmoid));
    assert_eq!(sec[2], Some(ActivationType::Tanh));
    assert!(sec[3].is_none());
    assert!(sec[22].is_none());
}

#[test]
fn test_parse_secondary_activations_absent() {
    let raw = serde_json::json!({});
    let sec = parse_secondary_activations_from_json(&raw, 23);
    assert_eq!(sec.len(), 23);
    for s in &sec {
        assert!(s.is_none());
    }
}

#[test]
fn test_parse_a2_max_secondary_activation_forms() {
    let scalar = serde_json::json!({ "secondary_activation": "Hardswish" });
    assert_eq!(
        parse_secondary_activations_from_json(&scalar, 2),
        vec![Some(ActivationType::HardSwish); 2]
    );
    let mixed = serde_json::json!({ "secondary_activation": [
        { "type": "LeakyHardtanh", "min_val": 0.0, "max_val": 0.9,
          "min_slope": 0.0, "max_slope": 0.02 },
        "ReLU", "Sigmoid"
    ] });
    assert_eq!(
        parse_secondary_activations_from_json(&mixed, 3),
        vec![
            Some(ActivationType::LeakyHardTanh {
                min_val: 0.0,
                max_val: 0.9,
                min_slope: 0.0,
                max_slope: 0.02,
            }),
            Some(ActivationType::ReLU),
            Some(ActivationType::Sigmoid),
        ]
    );
}

#[test]
fn test_parse_activations_wrong_length_rejected() {
    let raw = serde_json::json!({
        "activation": [
            {"type": "LeakyReLU", "negative_slope": 0.01}
        ]
    });
    assert_eq!(
        parse_activations_from_json(&raw, 23),
        Err(NamErrorCode::NamJsonUnknownActivation)
    );
}

#[test]
fn test_parse_activations_missing_is_ok_none() {
    let raw = serde_json::json!({});
    assert_eq!(parse_activations_from_json(&raw, 23), Ok(None));
}

#[test]
fn test_parse_activations_null_rejected() {
    let raw = serde_json::json!({ "activation": null });
    assert_eq!(
        parse_activations_from_json(&raw, 23),
        Err(NamErrorCode::NamJsonUnknownActivation)
    );
}

#[test]
fn test_parse_layer_activations_missing_defaults_to_leaky_relu() {
    let raw = serde_json::json!({});
    let config = parse_layer_activations(&raw, 23).expect("should default");
    assert_eq!(config.activations.len(), 23);
    for act in &config.activations {
        assert_eq!(
            act,
            &ActivationType::LeakyReLU {
                negative_slope: 0.01,
            }
        );
    }
}

#[test]
fn test_parse_layer_activations_unknown_rejected() {
    let raw = serde_json::json!({ "activation": "Gelu" });
    assert_eq!(
        parse_layer_activations(&raw, 23),
        Err(NamErrorCode::NamJsonUnknownActivation)
    );
}

#[test]
fn test_parse_layer_activations_unknown_object_rejected() {
    let raw = serde_json::json!({ "activation": { "unknown": true } });
    assert_eq!(
        parse_layer_activations(&raw, 23),
        Err(NamErrorCode::NamJsonUnknownActivation)
    );
}

#[test]
fn test_try_parse_activation_null_param_rejected() {
    assert_eq!(
        try_parse_activation(&serde_json::json!({ "type": "LeakyReLU", "negative_slope": null })),
        Err(NamErrorCode::NamJsonUnknownActivation)
    );
}

#[test]
fn test_parse_layer_activations_integrated() {
    let raw = make_a2_layer_json();
    let config = parse_layer_activations(&raw, A2_NUM_LAYERS).expect("should parse");
    assert_eq!(config.activations.len(), A2_NUM_LAYERS);
    assert_eq!(config.gating_modes.len(), A2_NUM_LAYERS);
    assert_eq!(config.secondary_activations.len(), A2_NUM_LAYERS);
}

// =============================================================================
// `try_parse_activation` — NAMcore `ActivationConfig::from_json` parity
// =============================================================================

/// The full C++ name map (`activations.cpp:64-76`) resolves to a typed variant.
#[test]
fn test_try_parse_activation_all_canonical_names() {
    use super::try_parse_activation;
    let cases: &[(&str, ActivationType)] = &[
        ("Tanh", ActivationType::Tanh),
        ("Hardtanh", ActivationType::HardTanh),
        ("Fasttanh", ActivationType::FastTanh),
        ("ReLU", ActivationType::ReLU),
        (
            "LeakyReLU",
            ActivationType::LeakyReLU {
                negative_slope: 0.01,
            },
        ),
        (
            "PReLU",
            ActivationType::PReLU {
                negative_slopes: vec![0.01],
            },
        ),
        ("Sigmoid", ActivationType::Sigmoid),
        ("SiLU", ActivationType::SiLU),
        ("Hardswish", ActivationType::HardSwish),
        (
            "LeakyHardtanh",
            ActivationType::LeakyHardTanh {
                min_val: -1.0,
                max_val: 1.0,
                min_slope: 0.01,
                max_slope: 0.01,
            },
        ),
        (
            "LeakyHardTanh",
            ActivationType::LeakyHardTanh {
                min_val: -1.0,
                max_val: 1.0,
                min_slope: 0.01,
                max_slope: 0.01,
            },
        ),
        ("Softsign", ActivationType::Softsign),
    ];
    for (name, expected) in cases {
        let got = try_parse_activation(&serde_json::Value::String((*name).to_string()));
        assert_eq!(got.as_ref(), Ok(expected), "name {name}");
    }
}

#[test]
fn test_try_parse_activation_unknown_name_is_e1221() {
    use super::try_parse_activation;
    use crate::common::diagnostics::NamErrorCode;
    assert_eq!(
        try_parse_activation(&serde_json::json!("Gelu")),
        Err(NamErrorCode::NamJsonUnknownActivation)
    );
}

#[test]
fn test_try_parse_activation_leaky_relu_default_slope() {
    use super::try_parse_activation;
    let got = try_parse_activation(&serde_json::json!({"type": "LeakyReLU"})).expect("parse");
    assert_eq!(
        got,
        ActivationType::LeakyReLU {
            negative_slope: 0.01
        }
    );
}

#[test]
fn test_try_parse_activation_prelu_vector_and_scalar() {
    use super::try_parse_activation;
    assert_eq!(
        try_parse_activation(&serde_json::json!({"type": "PReLU", "negative_slopes": [0.1, 0.2]}))
            .expect("parse"),
        ActivationType::PReLU {
            negative_slopes: vec![0.1, 0.2]
        }
    );
    // Scalar form maps to a single-element slope vector (C++ `ActivationsPReLU(float)`).
    assert_eq!(
        try_parse_activation(&serde_json::json!({"type": "PReLU", "negative_slope": 0.25}))
            .expect("parse"),
        ActivationType::PReLU {
            negative_slopes: vec![0.25]
        }
    );
    // Absent slopes default to 0.01.
    assert_eq!(
        try_parse_activation(&serde_json::json!({"type": "PReLU"})).expect("parse"),
        ActivationType::PReLU {
            negative_slopes: vec![0.01]
        }
    );
}

#[test]
fn test_try_parse_activation_non_string_non_object_is_e1221() {
    use super::try_parse_activation;
    use crate::common::diagnostics::NamErrorCode;
    assert_eq!(
        try_parse_activation(&serde_json::json!(42)),
        Err(NamErrorCode::NamJsonUnknownActivation)
    );
    assert_eq!(
        try_parse_activation(&serde_json::json!({"no_type": true})),
        Err(NamErrorCode::NamJsonUnknownActivation)
    );
}

#[test]
fn test_try_parse_activation_leaky_hard_tanh_custom_params() {
    use super::try_parse_activation;
    let got = try_parse_activation(&serde_json::json!({
        "type": "LeakyHardtanh",
        "min_val": -2.0,
        "max_val": 3.0,
        "min_slope": 0.1,
        "max_slope": 0.2,
    }))
    .expect("parse");
    assert_eq!(
        got,
        ActivationType::LeakyHardTanh {
            min_val: -2.0,
            max_val: 3.0,
            min_slope: 0.1,
            max_slope: 0.2,
        }
    );
}
