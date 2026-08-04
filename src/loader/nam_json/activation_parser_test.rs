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

    let activations =
        parse_activations_from_json(&raw, 23).expect("should parse heterogeneous activations");
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
fn test_parse_activations_wrong_length_rejected() {
    let raw = serde_json::json!({
        "activation": [
            {"type": "LeakyReLU", "negative_slope": 0.01}
        ]
    });
    assert!(parse_activations_from_json(&raw, 23).is_none());
}

#[test]
fn test_parse_activations_missing_rejected() {
    let raw = serde_json::json!({});
    assert!(parse_activations_from_json(&raw, 23).is_none());
}

#[test]
fn test_parse_layer_activations_integrated() {
    let raw = make_a2_layer_json();
    let config = parse_layer_activations(&raw, A2_NUM_LAYERS).expect("should parse");
    assert_eq!(config.activations.len(), A2_NUM_LAYERS);
    assert_eq!(config.gating_modes.len(), A2_NUM_LAYERS);
    assert_eq!(config.secondary_activations.len(), A2_NUM_LAYERS);
}
