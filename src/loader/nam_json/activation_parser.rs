// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Advanced activation and gating parser for A2 layer arrays (F8).
//!
//! Parses the per-layer activation, gating mode, and secondary activation arrays
//! from the raw layer JSON, returning typed Rust vectors for consumption by the
//! dynamic A2 engine (WaveNetA2Dyn) and topology classification.
//!
//! Each function expects a single layer array's raw JSON (the `layer_raw` field
//! of `NamLayerConfig`) and validates array length against `num_layers` (23 for
//! standard A2).

use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::GatingMode;

/// Parsed per-layer activation configuration for a single A2 layer array.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerActivationConfig {
    /// Primary activation per layer (23 elements for standard A2).
    pub activations: Vec<ActivationType>,
    /// Gating mode per layer (23 elements).
    pub gating_modes: Vec<GatingMode>,
    /// Secondary activation per layer (23 elements, `None` when absent/null).
    pub secondary_activations: Vec<Option<ActivationType>>,
}

/// Parses the full set of activation-related fields from a single layer array's
/// raw JSON.
///
/// Extracts `activation`, `gating_mode`, and `secondary_activation` arrays,
/// validating that each has exactly `num_layers` entries and deserializing
/// each entry to its typed Rust equivalent.
///
/// Returns `None` when the `activation` array is missing or invalid, as A2
/// models require this field.
pub fn parse_layer_activations(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Option<LayerActivationConfig> {
    let activations = parse_activations_from_json(raw, num_layers)?;
    let gating_modes = parse_gating_modes_from_json(raw, num_layers);
    let secondary_activations = parse_secondary_activations_from_json(raw, num_layers);

    Some(LayerActivationConfig {
        activations,
        gating_modes,
        secondary_activations,
    })
}

/// Parses the `activation` JSON array into `Vec<ActivationType>`.
///
/// Returns `None` if the array is missing, has the wrong length, or
/// contains entries that cannot be deserialized as valid `ActivationType`.
///
/// Also handles single-object activations (e.g. `{"type":"Softsign"}`)
/// used by A2 generic topologies and condition_dsp sub-models, replicating the
/// single activation across all layers.
pub fn parse_activations_from_json(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Option<Vec<ActivationType>> {
    // Single-object activation: replicate across all layers (A2 generic).
    if let Some(obj) = raw.get("activation").and_then(|v| v.as_object()) {
        let at: ActivationType =
            serde_json::from_value(serde_json::Value::Object(obj.clone())).ok()?;
        return Some(vec![at; num_layers]);
    }

    let arr = raw.get("activation").and_then(|v| v.as_array())?;
    if arr.len() != num_layers {
        return None;
    }
    let mut out = Vec::with_capacity(num_layers);
    for entry in arr {
        let at: ActivationType = serde_json::from_value(entry.clone()).ok()?;
        out.push(at);
    }
    Some(out)
}

/// Parses the `gating_mode` JSON array into `Vec<GatingMode>`.
///
/// If the field is absent or null, returns a vector of `GatingMode::None`
/// with length `num_layers`.
///
/// Also handles single-string gating mode (e.g. `"gated"`)
/// used by A2 generic topologies, replicating the value across all layers.
pub fn parse_gating_modes_from_json(raw: &serde_json::Value, num_layers: usize) -> Vec<GatingMode> {
    // Single-string gating mode: replicate across all layers (A2 generic).
    if let Some(s) = raw.get("gating_mode").and_then(|v| v.as_str()) {
        let mode = match s {
            "gated" => GatingMode::Gated,
            "blended" => GatingMode::Blended,
            _ => GatingMode::None,
        };
        return vec![mode; num_layers];
    }

    let arr = match raw.get("gating_mode") {
        None | Some(serde_json::Value::Null) => {
            return vec![GatingMode::None; num_layers];
        }
        Some(v) => match v.as_array() {
            Some(a) if a.len() == num_layers => a,
            _ => return vec![GatingMode::None; num_layers],
        },
    };
    let mut out = Vec::with_capacity(num_layers);
    for entry in arr {
        let mode = match entry.as_str() {
            Some("gated") => GatingMode::Gated,
            Some("blended") => GatingMode::Blended,
            _ => GatingMode::None,
        };
        out.push(mode);
    }
    out
}

/// Parses the `secondary_activation` JSON array into `Vec<Option<ActivationType>>`.
///
/// If the field is absent or null, returns a vector of `None` with length
/// `num_layers`. Individual null entries are mapped to `None`.
///
/// Also handles single-object/single-string secondary activation
/// (e.g. `"Hardswish"`), replicating across all layers.
pub fn parse_secondary_activations_from_json(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Vec<Option<ActivationType>> {
    // Single-value secondary activation (object or string): replicate across all layers.
    if let Some(v) = raw.get("secondary_activation") {
        if let Some(s) = v.as_str() {
            if s.is_empty() || s.eq_ignore_ascii_case("none") {
                return vec![None; num_layers];
            }
            let at = serde_json::from_value(serde_json::Value::String(s.to_string())).ok();
            return vec![at; num_layers];
        }
        if v.is_object() {
            let at = serde_json::from_value(v.clone()).ok();
            return vec![at; num_layers];
        }
    }

    let arr = match raw.get("secondary_activation") {
        None | Some(serde_json::Value::Null) => {
            return vec![None; num_layers];
        }
        Some(v) => match v.as_array() {
            Some(a) if a.len() == num_layers => a,
            _ => return vec![None; num_layers],
        },
    };
    let mut out = Vec::with_capacity(num_layers);
    for entry in arr {
        if entry.is_null() {
            out.push(None);
        } else {
            let at = serde_json::from_value(entry.clone()).ok();
            out.push(at);
        }
    }
    out
}

#[cfg(test)]
#[path = "activation_parser_test.rs"]
mod tests;
