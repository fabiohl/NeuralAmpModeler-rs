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

/// Maximum number of internal layers accepted by the activation helpers (D-06).
///
/// This is the A2 per-layer-array cap already enforced by the topology scan
/// (`MAX_DILATIONS_PER_ARRAY`), restated here because these helpers are public
/// and replicate defaults/specs per layer *before* any topology validation.
/// `0` remains accepted where it was accepted before.
const MAX_ACTIVATION_LAYERS: usize = 64;

/// Maximum length of a single `PReLU` `negative_slopes` vector (D-06).
///
/// Equal to the widest channel ceiling any consumer accepts (512, shared by
/// ConvNet, WaveNet free geometry and post-stack heads), so no model with a
/// channel-consistent slope vector is rejected. Bounding it prevents a small
/// number of layers from amplifying a huge vector through replication.
const MAX_PRELU_SLOPES_PER_VECTOR: usize = 512;

/// Validates `num_layers` before any `vec![_; num_layers]` replication.
///
/// Distinguishes an impossible capacity (the byte count for the vector itself
/// overflows `usize`, reported as [`NamErrorCode::OutOfMemory`]) from a
/// geometry that is representable but unsupported by this parser (reported as
/// [`NamErrorCode::InvalidModelTopology`]).
pub(super) fn validate_num_layers(num_layers: usize) -> Result<(), NamErrorCode> {
    if num_layers
        .checked_mul(std::mem::size_of::<ActivationType>())
        .is_none()
    {
        return Err(NamErrorCode::OutOfMemory);
    }
    if num_layers > MAX_ACTIVATION_LAYERS {
        return Err(NamErrorCode::InvalidModelTopology);
    }
    Ok(())
}

/// Cost of one activation when replicated `num_layers` times, in slope slots.
///
/// Non-`PReLU` activations cost one slot per layer; `PReLU` costs its slope
/// vector length. The multiplication is checked and the aggregate is capped so
/// a vector cannot be cloned an unbounded number of times.
fn ensure_replication_cost(
    activation: &ActivationType,
    num_layers: usize,
) -> Result<(), NamErrorCode> {
    let per_layer = match activation {
        ActivationType::PReLU { negative_slopes } => negative_slopes.len(),
        _ => 1,
    };
    let total = num_layers
        .checked_mul(per_layer)
        .ok_or(NamErrorCode::OutOfMemory)?;
    if total > MAX_ACTIVATION_LAYERS * MAX_PRELU_SLOPES_PER_VECTOR {
        return Err(NamErrorCode::InvalidModelTopology);
    }
    Ok(())
}

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
                    if arr.len() > MAX_PRELU_SLOPES_PER_VECTOR {
                        return Err(NamErrorCode::InvalidModelTopology);
                    }
                    let mut slopes = Vec::with_capacity(arr.len());
                    for e in arr {
                        slopes
                            .push(e.as_f64().ok_or(NamErrorCode::NamJsonUnknownActivation)? as f32);
                    }
                    if slopes.is_empty() {
                        return Err(NamErrorCode::NamJsonUnknownActivation);
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
///
/// # Errors
///
/// - [`NamErrorCode::OutOfMemory`] when `num_layers` is so large that the
///   per-layer vector size itself overflows.
/// - [`NamErrorCode::InvalidModelTopology`] when `num_layers` exceeds the A2
///   per-layer-array ceiling (64).
/// - [`NamErrorCode::NamJsonUnknownActivation`] for unknown or malformed
///   activation specs.
/// - [`NamErrorCode::InvalidModelTopology`] for malformed gating fields or
///   divergent `secondary_activation` lengths.
pub fn parse_layer_activations(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Result<LayerActivationConfig, NamErrorCode> {
    validate_num_layers(num_layers)?;
    let activations = match parse_activations_from_json(raw, num_layers)? {
        Some(acts) => acts,
        None => vec![
            ActivationType::LeakyReLU {
                negative_slope: crate::models::a2::A2_LEAKY_SLOPE,
            };
            num_layers
        ],
    };
    let (gating_modes, used_legacy_gated) = parse_gating_modes_with_source(raw, num_layers)?;
    let secondary_activations =
        parse_secondary_activations_for_modes(raw, &gating_modes, used_legacy_gated)?;

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
///
/// # Errors
///
/// - [`NamErrorCode::OutOfMemory`] when `num_layers` is so large that the
///   per-layer vector size overflows.
/// - [`NamErrorCode::InvalidModelTopology`] when `num_layers` exceeds 64 or a
///   replicated `PReLU` vector would exceed the aggregate slope budget.
/// - [`NamErrorCode::NamJsonUnknownActivation`] for wrong array lengths, null,
///   unknown names or malformed entries.
pub fn parse_activations_from_json(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Result<Option<Vec<ActivationType>>, NamErrorCode> {
    validate_num_layers(num_layers)?;
    let act_val = match raw.get("activation") {
        None => return Ok(None),
        Some(v) => v,
    };

    // Single-string activation: replicate across all layers (A2 generic).
    if let Some(s) = act_val.as_str() {
        let at = try_parse_activation(&serde_json::Value::String(s.to_string()))?;
        ensure_replication_cost(&at, num_layers)?;
        return Ok(Some(vec![at; num_layers]));
    }

    // Single-object activation: replicate across all layers (A2 generic).
    if let Some(obj) = act_val.as_object() {
        let at = try_parse_activation(&serde_json::Value::Object(obj.clone()))?;
        ensure_replication_cost(&at, num_layers)?;
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

/// Internal helper returning `(gating_modes, used_legacy_gated)` so
/// [`parse_layer_activations`] preserves C++ precedence and legacy `gated: true`
/// behavior (`Sigmoid` secondary activation, ignoring stray `secondary_activation`).
fn parse_gating_modes_with_source(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Result<(Vec<GatingMode>, bool), NamErrorCode> {
    validate_num_layers(num_layers)?;
    if let Some(v) = raw.get("gating_mode") {
        if let Some(s) = v.as_str() {
            let mode = match s {
                "none" => GatingMode::None,
                "gated" => GatingMode::Gated,
                "blended" => GatingMode::Blended,
                _ => return Err(NamErrorCode::InvalidModelTopology),
            };
            return Ok((vec![mode; num_layers], false));
        }
        if let Some(arr) = v.as_array() {
            if arr.len() != num_layers {
                return Err(NamErrorCode::InvalidModelTopology);
            }
            let mut out = Vec::with_capacity(num_layers);
            for entry in arr {
                let mode = match entry.as_str() {
                    Some("none") => GatingMode::None,
                    Some("gated") => GatingMode::Gated,
                    Some("blended") => GatingMode::Blended,
                    _ => return Err(NamErrorCode::InvalidModelTopology),
                };
                out.push(mode);
            }
            return Ok((out, false));
        }
        return Err(NamErrorCode::InvalidModelTopology);
    }

    match raw.get("gated") {
        None => Ok((vec![GatingMode::None; num_layers], false)),
        Some(serde_json::Value::Bool(true)) => Ok((vec![GatingMode::Gated; num_layers], true)),
        Some(serde_json::Value::Bool(false)) => Ok((vec![GatingMode::None; num_layers], false)),
        Some(_) => Err(NamErrorCode::InvalidModelTopology),
    }
}

/// Parses the `gating_mode` (or legacy `gated`) field into `Vec<GatingMode>`.
///
/// Follows C++ `WaveNet::parse_config_json` precedence: `gating_mode` overrides
/// `gated`. Absent fields default to [`GatingMode::None`]. Invalid strings,
/// non-string elements (including `null`), wrong-length arrays, or non-boolean
/// `gated` values are rejected fail-closed.
///
/// # Errors
///
/// - [`NamErrorCode::OutOfMemory`] when `num_layers` is too large to represent.
/// - [`NamErrorCode::InvalidModelTopology`] when `num_layers` exceeds 64 or the
///   gating fields are malformed/divergent.
pub fn parse_gating_modes_from_json(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Result<Vec<GatingMode>, NamErrorCode> {
    parse_gating_modes_with_source(raw, num_layers).map(|(modes, _)| modes)
}

/// Context-aware `secondary_activation` resolution for a resolved gating vector.
fn parse_secondary_activations_for_modes(
    raw: &serde_json::Value,
    gating_modes: &[GatingMode],
    used_legacy_gated: bool,
) -> Result<Vec<Option<ActivationType>>, NamErrorCode> {
    let num_layers = gating_modes.len();
    if used_legacy_gated {
        return Ok(vec![Some(ActivationType::Sigmoid); num_layers]);
    }

    let any_active = gating_modes.iter().any(|&m| m != GatingMode::None);
    if !any_active {
        if let Some(arr) = raw.get("secondary_activation").and_then(|v| v.as_array())
            && arr.len() != num_layers
        {
            return Err(NamErrorCode::InvalidModelTopology);
        }
        return Ok(vec![None; num_layers]);
    }

    match raw.get("secondary_activation") {
        None => Ok(gating_modes
            .iter()
            .map(|&m| {
                if m == GatingMode::None {
                    None
                } else {
                    Some(ActivationType::Sigmoid)
                }
            })
            .collect()),
        Some(serde_json::Value::Array(arr)) => {
            if arr.len() != num_layers {
                return Err(NamErrorCode::InvalidModelTopology);
            }
            let mut out = Vec::with_capacity(num_layers);
            for (entry, &mode) in arr.iter().zip(gating_modes.iter()) {
                if mode == GatingMode::None {
                    out.push(None);
                } else {
                    out.push(Some(try_parse_activation(entry)?));
                }
            }
            Ok(out)
        }
        Some(v) => {
            let parsed = try_parse_activation(v)?;
            ensure_replication_cost(&parsed, num_layers)?;
            Ok(gating_modes
                .iter()
                .map(|&m| {
                    if m == GatingMode::None {
                        None
                    } else {
                        Some(parsed.clone())
                    }
                })
                .collect())
        }
    }
}

/// Parses the `secondary_activation` JSON field into `Vec<Option<ActivationType>>`.
///
/// Supports absent/`null`/`"none"`/`""` (all `None`), per-layer arrays (with `null`
/// permitted on slots), single-string bare names, and activation config objects
/// (including `PReLU` bare/scalar/vector) via `try_parse_activation`. Any
/// unrecognized or malformed entry fails closed with
/// [`NamErrorCode::NamJsonUnknownActivation`].
///
/// # Errors
///
/// - [`NamErrorCode::OutOfMemory`] when `num_layers` is too large to represent.
/// - [`NamErrorCode::InvalidModelTopology`] when `num_layers` exceeds 64 or a
///   replicated `PReLU` vector would exceed the aggregate slope budget.
/// - [`NamErrorCode::NamJsonUnknownActivation`] for unknown/malformed entries
///   or a divergent array length.
pub fn parse_secondary_activations_from_json(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Result<Vec<Option<ActivationType>>, NamErrorCode> {
    validate_num_layers(num_layers)?;
    match raw.get("secondary_activation") {
        None | Some(serde_json::Value::Null) => Ok(vec![None; num_layers]),
        Some(serde_json::Value::String(s)) if s.is_empty() || s.eq_ignore_ascii_case("none") => {
            Ok(vec![None; num_layers])
        }
        Some(serde_json::Value::Array(arr)) => {
            if arr.len() != num_layers {
                return Err(NamErrorCode::NamJsonUnknownActivation);
            }
            let mut out = Vec::with_capacity(num_layers);
            for entry in arr {
                if entry.is_null() {
                    out.push(None);
                } else {
                    out.push(Some(try_parse_activation(entry)?));
                }
            }
            Ok(out)
        }
        Some(v) => {
            let at = try_parse_activation(v)?;
            ensure_replication_cost(&at, num_layers)?;
            Ok(vec![Some(at); num_layers])
        }
    }
}

#[cfg(test)]
#[path = "activation_parser_test.rs"]
mod tests;
