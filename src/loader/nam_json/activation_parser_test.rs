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

    let modes = parse_gating_modes_from_json(&raw, 23).expect("should parse gating modes");
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
    let modes =
        parse_gating_modes_from_json(&raw, 23).expect("absent gating_mode defaults to None");
    assert_eq!(modes.len(), 23);
    for mode in &modes {
        assert_eq!(mode, &GatingMode::None);
    }
}

#[test]
fn test_parse_gating_modes_null_rejected() {
    let raw = serde_json::json!({"gating_mode": null});
    assert_eq!(
        parse_gating_modes_from_json(&raw, 23),
        Err(NamErrorCode::InvalidModelTopology)
    );
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

    let sec = parse_secondary_activations_from_json(&raw, 23).expect("should parse secondary");
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
    let sec = parse_secondary_activations_from_json(&raw, 23).expect("absent secondary is ok");
    assert_eq!(sec.len(), 23);
    for s in &sec {
        assert!(s.is_none());
    }
}

#[test]
fn test_parse_a2_max_secondary_activation_forms() {
    let scalar = serde_json::json!({ "secondary_activation": "Hardswish" });
    assert_eq!(
        parse_secondary_activations_from_json(&scalar, 2).expect("Hardswish parses"),
        vec![Some(ActivationType::HardSwish); 2]
    );
    let mixed = serde_json::json!({ "secondary_activation": [
        { "type": "LeakyHardtanh", "min_val": 0.0, "max_val": 0.9,
          "min_slope": 0.0, "max_slope": 0.02 },
        "ReLU", "Sigmoid"
    ] });
    assert_eq!(
        parse_secondary_activations_from_json(&mixed, 3).expect("mixed parses"),
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
fn test_parse_layer_activations_absent_fields_use_standard_a2_layer_count() {
    let raw = serde_json::json!({});
    let config = parse_layer_activations(&raw, A2_NUM_LAYERS).expect("absent fields default");
    assert_eq!(config.activations.len(), A2_NUM_LAYERS);
    assert_eq!(config.gating_modes.len(), A2_NUM_LAYERS);
    assert_eq!(config.secondary_activations.len(), A2_NUM_LAYERS);
    assert!(config.activations.iter().all(|act| act
        == &ActivationType::LeakyReLU {
            negative_slope: 0.01
        }));
    assert!(config.gating_modes.iter().all(|m| m == &GatingMode::None));
    assert!(config.secondary_activations.iter().all(Option::is_none));
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

// — F06 / T5.6: dimensions validated before any per-layer replication ————

#[test]
fn test_activation_helpers_accept_supported_layer_boundaries() {
    let raw = serde_json::json!({});
    for n in [0usize, 1, 23, 64] {
        let cfg = parse_layer_activations(&raw, n).expect("supported layer count");
        assert_eq!(cfg.activations.len(), n);
        assert_eq!(cfg.gating_modes.len(), n);
        assert_eq!(cfg.secondary_activations.len(), n);

        assert_eq!(
            parse_activations_from_json(&raw, n).expect("activation helper"),
            None
        );
        assert_eq!(
            parse_gating_modes_from_json(&raw, n)
                .expect("gating helper")
                .len(),
            n
        );
        assert_eq!(
            parse_secondary_activations_from_json(&raw, n)
                .expect("secondary helper")
                .len(),
            n
        );
    }
}

#[test]
fn test_activation_helpers_reject_65_layers_as_unsupported_geometry() {
    let raw = serde_json::json!({
        "activation": "Tanh",
        "gating_mode": "none",
        "secondary_activation": null
    });
    assert_eq!(
        parse_layer_activations(&raw, 65),
        Err(NamErrorCode::InvalidModelTopology)
    );
    assert_eq!(
        parse_activations_from_json(&raw, 65),
        Err(NamErrorCode::InvalidModelTopology)
    );
    assert_eq!(
        parse_gating_modes_from_json(&raw, 65),
        Err(NamErrorCode::InvalidModelTopology)
    );
    assert_eq!(
        parse_secondary_activations_from_json(&raw, 65),
        Err(NamErrorCode::InvalidModelTopology)
    );
}

#[test]
fn test_activation_helpers_reject_usize_max_without_panic() {
    let raw = serde_json::json!({});
    assert_eq!(
        parse_layer_activations(&raw, usize::MAX),
        Err(NamErrorCode::OutOfMemory)
    );
    assert_eq!(
        parse_activations_from_json(&raw, usize::MAX),
        Err(NamErrorCode::OutOfMemory)
    );
    assert_eq!(
        parse_gating_modes_from_json(&raw, usize::MAX),
        Err(NamErrorCode::OutOfMemory)
    );
    assert_eq!(
        parse_secondary_activations_from_json(&raw, usize::MAX),
        Err(NamErrorCode::OutOfMemory)
    );
}

#[test]
fn test_parse_activation_config_validates_before_defaults() {
    let cfg = crate::loader::nam_json::NamLayerConfig::default();
    assert_eq!(
        cfg.parse_activation_config(65),
        Err(NamErrorCode::InvalidModelTopology)
    );
    assert_eq!(
        cfg.parse_activation_config(usize::MAX),
        Err(NamErrorCode::OutOfMemory)
    );
    let ok = cfg.parse_activation_config(64).expect("64 layers accepted");
    assert_eq!(ok.activations.len(), 64);
}

#[test]
fn test_prelu_slope_vector_and_replication_limits() {
    let slopes_513: Vec<f64> = (0..513).map(|i| i as f64 * 0.001).collect();
    let over =
        serde_json::json!({ "activation": { "type": "PReLU", "negative_slopes": slopes_513 } });
    assert_eq!(
        parse_layer_activations(&over, 23),
        Err(NamErrorCode::InvalidModelTopology)
    );

    let slopes_512: Vec<f64> = (0..512).map(|i| i as f64 * 0.001).collect();
    let max =
        serde_json::json!({ "activation": { "type": "PReLU", "negative_slopes": slopes_512 } });
    let cfg = parse_layer_activations(&max, 64).expect("64 layers x 512 slopes accepted");
    assert_eq!(cfg.activations.len(), 64);
    assert!(cfg.activations.iter().all(|a| matches!(
        a,
        ActivationType::PReLU { negative_slopes } if negative_slopes.len() == 512
    )));

    let slopes_256: Vec<f64> = (0..256).map(|i| i as f64 * 0.001).collect();
    let mid =
        serde_json::json!({ "activation": { "type": "PReLU", "negative_slopes": slopes_256 } });
    parse_layer_activations(&mid, 23).expect("256-slope vector accepted for a 256-channel layer");
}

#[test]
fn test_gating_legacy_blended_and_divergent_lists_at_boundaries() {
    let legacy = serde_json::json!({ "gated": true });
    let cfg = parse_layer_activations(&legacy, 64).expect("legacy gated at 64 layers");
    assert!(cfg.gating_modes.iter().all(|m| m == &GatingMode::Gated));
    assert!(
        cfg.secondary_activations
            .iter()
            .all(|s| s == &Some(ActivationType::Sigmoid))
    );

    let blended = serde_json::json!({ "gating_mode": "blended" });
    let modes = parse_gating_modes_from_json(&blended, 64).expect("blended at 64 layers");
    assert!(modes.iter().all(|m| m == &GatingMode::Blended));

    let divergent_activation = serde_json::json!({ "activation": ["Tanh"] });
    assert_eq!(
        parse_activations_from_json(&divergent_activation, 2),
        Err(NamErrorCode::NamJsonUnknownActivation)
    );
    let divergent_gating = serde_json::json!({ "gating_mode": ["none"] });
    assert_eq!(
        parse_gating_modes_from_json(&divergent_gating, 2),
        Err(NamErrorCode::InvalidModelTopology)
    );
    let divergent_secondary = serde_json::json!({
        "gating_mode": ["none", "gated"],
        "secondary_activation": ["Tanh"]
    });
    assert_eq!(
        parse_layer_activations(&divergent_secondary, 2),
        Err(NamErrorCode::InvalidModelTopology)
    );
}
