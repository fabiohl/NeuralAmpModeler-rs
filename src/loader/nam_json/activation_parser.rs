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

use crate::common::diagnostics::NamErrorCode;
use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::GatingMode;

/// Parses a single activation specification into a typed [`ActivationType`].
///
/// Mirrors the C++ `nam::activations::ActivationConfig::from_json`
/// (`activations.cpp:59-130`) exactly:
///
/// - a bare string is looked up in the canonical name map
///   (`Tanh`, `Hardtanh`, `Fasttanh`, `ReLU`, `LeakyReLU`, `PReLU`, `Sigmoid`,
///   `SiLU`, `Hardswish`, `LeakyHardtanh`/`LeakyHardTanh`, `Softsign`);
/// - an object requires a `type` field, then applies the per-type C++ parameter
///   defaults (`LeakyReLU` slope `0.01`; `LeakyHardtanh`
///   `(-1, 1, 0.01, 0.01)`; `PReLU` scalar `0.01`);
/// - any other JSON type, or an unknown name, is rejected with
///   [`NamErrorCode::NamJsonUnknownActivation`] (`E1221`).
///
/// No approximation or silent fallback is performed: the executed activation
/// is exactly the declared one, or the model is rejected.
pub(crate) fn try_parse_activation(v: &serde_json::Value) -> Result<ActivationType, NamErrorCode> {
    let (type_name, obj) = match v {
        serde_json::Value::String(s) => (s.as_str(), None),
        serde_json::Value::Object(map) => {
            let t = map
                .get("type")
                .and_then(|t| t.as_str())
                .ok_or(NamErrorCode::NamJsonUnknownActivation)?;
            (t, Some(map))
        }
        _ => return Err(NamErrorCode::NamJsonUnknownActivation),
    };

    let param = |key: &str, default: f32| -> Result<f32, NamErrorCode> {
        match obj.and_then(|m| m.get(key)) {
            None => Ok(default),
            Some(x) => x
                .as_f64()
                .map(|f| f as f32)
                .ok_or(NamErrorCode::NamJsonUnknownActivation),
        }
    };

    match type_name {
        "Tanh" => Ok(ActivationType::Tanh),
        "Hardtanh" | "HardTanh" => Ok(ActivationType::HardTanh),
        "Fasttanh" | "FastTanh" => Ok(ActivationType::FastTanh),
        "ReLU" => Ok(ActivationType::ReLU),
        "LeakyReLU" => Ok(ActivationType::LeakyReLU {
            negative_slope: param("negative_slope", 0.01)?,
        }),
        "PReLU" => {
            // C++ `from_json` checks `negative_slope` (scalar) before
            // `negative_slopes` (vector); a missing slope defaults to 0.01
            // (`activations.cpp:103-113`, `:158`).
            if let Some(m) = obj {
                if let Some(scalar) = m.get("negative_slope") {
                    let slope = scalar
                        .as_f64()
                        .ok_or(NamErrorCode::NamJsonUnknownActivation)?
                        as f32;
                    return Ok(ActivationType::PReLU {
                        negative_slopes: vec![slope],
                    });
                }
                if let Some(vector) = m.get("negative_slopes") {
                    let arr = vector
                        .as_array()
                        .ok_or(NamErrorCode::NamJsonUnknownActivation)?;
                    let mut slopes = Vec::with_capacity(arr.len());
                    for e in arr {
                        slopes
                            .push(e.as_f64().ok_or(NamErrorCode::NamJsonUnknownActivation)? as f32);
                    }
                    return Ok(ActivationType::PReLU {
                        negative_slopes: slopes,
                    });
                }
            }
            Ok(ActivationType::PReLU {
                negative_slopes: vec![0.01],
            })
        }
        "Sigmoid" => Ok(ActivationType::Sigmoid),
        "SiLU" => Ok(ActivationType::SiLU),
        "Hardswish" | "HardSwish" => Ok(ActivationType::HardSwish),
        "LeakyHardtanh" | "LeakyHardTanh" => Ok(ActivationType::LeakyHardTanh {
            min_val: param("min_val", -1.0)?,
            max_val: param("max_val", 1.0)?,
            min_slope: param("min_slope", 0.01)?,
            max_slope: param("max_slope", 0.01)?,
        }),
        "Softsign" => Ok(ActivationType::Softsign),
        _ => Err(NamErrorCode::NamJsonUnknownActivation),
    }
}

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
/// each entry to its typed Rust equivalent via `try_parse_activation`.
///
/// Returns `Ok(`[`LayerActivationConfig`]`)` with the documented default
/// (`LeakyReLU` with slope 0.01) only when the `activation` field is absent.
/// Rejects unknown or invalid activations with [`NamErrorCode::NamJsonUnknownActivation`].
pub fn parse_layer_activations(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Result<LayerActivationConfig, NamErrorCode> {
    let activations = match parse_activations_from_json(raw, num_layers)? {
        Some(acts) => acts,
        None => vec![
            ActivationType::LeakyReLU {
                negative_slope: crate::models::a2::A2_LEAKY_SLOPE,
            };
            num_layers
        ],
    };
    let gating_modes = parse_gating_modes_from_json(raw, num_layers);
    let secondary_activations = parse_secondary_activations_from_json(raw, num_layers);

    Ok(LayerActivationConfig {
        activations,
        gating_modes,
        secondary_activations,
    })
}

/// Parses the `activation` JSON field into `Option<Vec<ActivationType>>`.
///
/// Returns `Ok(None)` when the `activation` field is absent.
/// If present, validates and deserializes via `try_parse_activation`:
/// - Single-string or single-object activations are replicated across all layers (A2 generic).
/// - Arrays must have length `num_layers` and every entry must parse successfully.
///
/// Rejects wrong array lengths, null, unknown names, or malformed objects with
/// [`NamErrorCode::NamJsonUnknownActivation`] (`E1221`).
pub fn parse_activations_from_json(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Result<Option<Vec<ActivationType>>, NamErrorCode> {
    let act_val = match raw.get("activation") {
        None => return Ok(None),
        Some(v) => v,
    };

    // Single-string activation: replicate across all layers (A2 generic).
    if let Some(s) = act_val.as_str() {
        let at = try_parse_activation(&serde_json::Value::String(s.to_string()))?;
        return Ok(Some(vec![at; num_layers]));
    }

    // Single-object activation: replicate across all layers (A2 generic).
    if let Some(obj) = act_val.as_object() {
        let at = try_parse_activation(&serde_json::Value::Object(obj.clone()))?;
        return Ok(Some(vec![at; num_layers]));
    }

    // Array of activations: validate length and parse each entry.
    if let Some(arr) = act_val.as_array() {
        if arr.len() != num_layers {
            return Err(NamErrorCode::NamJsonUnknownActivation);
        }
        let mut out = Vec::with_capacity(num_layers);
        for entry in arr {
            let at = try_parse_activation(entry)?;
            out.push(at);
        }
        return Ok(Some(out));
    }

    // Any other JSON type (null, number, bool): rejected as invalid activation.
    Err(NamErrorCode::NamJsonUnknownActivation)
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
    let parse = |value: &serde_json::Value| {
        // ActivationType is internally tagged; NAM also permits a bare name.
        let tagged = match value.as_str() {
            Some(name) => serde_json::json!({ "type": name }),
            None => value.clone(),
        };
        serde_json::from_value(tagged).ok()
    };
    // Single-value secondary activation (object or string): replicate across all layers.
    if let Some(v) = raw.get("secondary_activation") {
        if let Some(s) = v.as_str() {
            if s.is_empty() || s.eq_ignore_ascii_case("none") {
                return vec![None; num_layers];
            }
            let at = parse(v);
            return vec![at; num_layers];
        }
        if v.is_object() {
            let at = parse(v);
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
            let at = parse(entry);
            out.push(at);
        }
    }
    out
}

#[cfg(test)]
#[path = "activation_parser_test.rs"]
mod tests;
