// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Sample-rate provenance resolution across nested model graphs.
//!
//! The C++ NAMcore propagates an expected sample rate through `get_dsp`:
//! a model's declared `sample_rate` wins, and composite nodes (Sequential,
//! SlimmableContainer) resolve their rate from their children when they do
//! not declare their own. The Rust loader mirrors that rule here so that a
//! nested `Sequential`/`Container` never silently falls back to the 48 kHz
//! engine default while a child declares a concrete rate (F-13/F-16, DEC-01b).
//!
//! # Structural Guardrails (DoS hardening)
//!
//! Before descending into nested nodes or reading weights, structural limits
//! are strictly enforced to avoid resource exhaustion:
//! - Sequential recursion depth is capped at [`MAX_SEQUENTIAL_DEPTH`] (8) (`E1311`).
//! - Total sequential children across the entire tree are capped at
//!   [`MAX_SEQUENTIAL_TOTAL_CHILDREN`] (64) (`E1314`).
//! - SlimmableContainer recursion depth is capped at [`MAX_CONTAINER_DEPTH`] (4) (`E1210`).
//! - SlimmableContainer submodels count per container is capped at [`MAX_SUBMODELS`] (8) (`E1209`).
//!
//! Nodes are inspected directly by reference (`serde_json::Value`), avoiding
//! cloning subtrees or deserializing weights (`Vec<f32>`).
//! Rates are reduced directly without intermediate vector allocations.

use serde_json::Value;

use crate::common::diagnostics::NamErrorCode;
use crate::loader::dispatcher::container::MAX_CONTAINER_DEPTH;
use crate::loader::nam_json::NamModelData;
use crate::loader::nam_json::validation::schema::MAX_SUBMODELS;
use crate::loader::nam_json::validation::semantic::{
    MAX_SEQUENTIAL_DEPTH, MAX_SEQUENTIAL_TOTAL_CHILDREN,
};

#[derive(Debug, Default)]
struct StructuralBudget {
    total_sequential_children: usize,
}

/// Returns whether `rate` is losslessly representable by the public
/// [`LoadedModelPair::sample_rate`](crate::loader::LoadedModelPair) `u32`
/// contract.
///
/// Positive fractional values are rejected instead of truncated (0.5 would
/// silently become 0), values above `u32::MAX` are rejected instead of
/// saturated, and a `f32` that rounds to 2^32 is rejected because it exceeds
/// the `u32` range.
pub(crate) fn rate_is_representable_u32(rate: f32) -> bool {
    rate.is_finite() && rate >= 1.0 && rate.fract() == 0.0 && (rate as f64) <= u32::MAX as f64
}

/// Converts a validated rate to `u32`, or returns `None` when the value cannot
/// be represented exactly (see [`rate_is_representable_u32`]).
pub(crate) fn rate_to_u32(rate: f32) -> Option<u32> {
    rate_is_representable_u32(rate).then_some(rate as u32)
}

/// Applies the sample-rate contract to a JSON number parsed as `f64`.
///
/// The original literal is validated *before* the lossy reduction to `f32`,
/// so integers that are not exactly representable in `f32` (for example
/// 16_777_217) are rejected instead of silently rounded.
///
/// Non-finite values are rejected, non-positive values map to "unknown"
/// (`None`), matching the existing JSON lenience.
pub(crate) fn declared_rate_from_f64(num: f64) -> Result<Option<f32>, NamErrorCode> {
    if !num.is_finite() {
        return Err(NamErrorCode::NamJsonInvalidSampleRate);
    }
    if num <= 0.0 {
        return Ok(None);
    }
    let rate = num as f32;
    if rate_is_representable_u32(rate) && rate as f64 == num {
        Ok(Some(rate))
    } else {
        Err(NamErrorCode::NamJsonInvalidSampleRate)
    }
}

/// Reduces a child's resolved rate into the running `resolved` accumulator.
///
/// Follows DEC-01 / DEC-01c:
/// - Missing rate (`None`) is ignored and adopts the already-resolved rate.
/// - The first known rate (`Some(rate)`) establishes the governing rate.
/// - Any conflicting known rate triggers `SequentialSampleRateMismatch` (`E1309`).
#[inline]
fn reduce_rate(resolved: &mut Option<f32>, child_rate: Option<f32>) -> Result<(), NamErrorCode> {
    let Some(child_rate) = child_rate else {
        return Ok(());
    };
    match *resolved {
        None => {
            *resolved = Some(child_rate);
            Ok(())
        }
        Some(expected) => {
            if child_rate != expected {
                Err(NamErrorCode::SequentialSampleRateMismatch)
            } else {
                Ok(())
            }
        }
    }
}

/// Extracts the declared `sample_rate` from a JSON node by reference.
///
/// Follows C++ NAMcore lenience and RFC 7.1:
/// - Missing or null `sample_rate` returns `Ok(None)`.
/// - Any finite value <= 0.0 (including the `-1.0` C++ sentinel) returns `Ok(None)`.
/// - A finite positive number returns `Ok(Some(num))` only when it is a
///   positive integer exactly representable as `u32`/`f32`.
/// - Non-finite numbers, non-number types, fractional values, and values
///   outside the `u32`/exact-`f32` contract return
///   `Err(NamErrorCode::NamJsonInvalidSampleRate)`.
fn read_declared_rate(val: &Value) -> Result<Option<f32>, NamErrorCode> {
    match val.get("sample_rate") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(num) => match declared_rate_from_f64(num) {
                Ok(rate) => Ok(rate),
                Err(code) => {
                    // Preserve the offending original value for off-RT triage;
                    // the typed code alone cannot carry it through the
                    // structural recursion.
                    log::warn!(
                        "[Loader] Invalid field rejected: field='sample_rate', value={:?}, offset_bytes=0",
                        num
                    );
                    Err(code)
                }
            },
            None => Err(NamErrorCode::NamJsonInvalidSampleRate),
        },
    }
}

/// Recursively resolves the declared sample rate of a child node from its JSON `Value` by reference.
///
/// Enforces structural depth and child count limits before descending into nested children,
/// without cloning weights or deserializing model data.
fn resolve_value_rate(
    val: &Value,
    seq_depth: Option<usize>,
    container_depth: Option<usize>,
    budget: &mut StructuralBudget,
) -> Result<Option<f32>, NamErrorCode> {
    let declared_rate = read_declared_rate(val)?;
    let arch = val.get("architecture").and_then(Value::as_str);

    match arch {
        Some("Sequential") => {
            let next_seq_depth = match seq_depth {
                None => 1,
                Some(depth) => depth
                    .checked_add(1)
                    .ok_or(NamErrorCode::SequentialChildrenExceedLimit)?,
            };
            if next_seq_depth > MAX_SEQUENTIAL_DEPTH {
                return Err(NamErrorCode::SequentialRecursionDepthExceeded);
            }

            let mut resolved = declared_rate;
            if let Some(models) = val
                .get("config")
                .and_then(|c| c.get("models"))
                .and_then(Value::as_array)
            {
                for child in models {
                    budget.total_sequential_children = budget
                        .total_sequential_children
                        .checked_add(1)
                        .ok_or(NamErrorCode::SequentialChildrenExceedLimit)?;
                    if budget.total_sequential_children > MAX_SEQUENTIAL_TOTAL_CHILDREN {
                        return Err(NamErrorCode::SequentialChildrenExceedLimit);
                    }
                    let child_rate =
                        resolve_value_rate(child, Some(next_seq_depth), container_depth, budget)?;
                    reduce_rate(&mut resolved, child_rate)?;
                }
            }
            Ok(resolved)
        }
        Some("SlimmableContainer") => {
            let next_container_depth = match container_depth {
                None => 0,
                Some(depth) => depth
                    .checked_add(1)
                    .ok_or(NamErrorCode::NamJsonSubmodelsTooDeep)?,
            };
            if next_container_depth > MAX_CONTAINER_DEPTH {
                return Err(NamErrorCode::NamJsonSubmodelsTooDeep);
            }

            let mut resolved = declared_rate;
            if let Some(submodels) = val
                .get("config")
                .and_then(|c| c.get("submodels"))
                .and_then(Value::as_array)
            {
                if submodels.len() > MAX_SUBMODELS {
                    return Err(NamErrorCode::NamJsonSubmodelsExceedLimit);
                }
                for entry in submodels {
                    let model_val = entry
                        .get("model")
                        .ok_or(NamErrorCode::NamJsonInvalidEnvelope)?;
                    let sub_rate = resolve_value_rate(
                        model_val,
                        seq_depth,
                        Some(next_container_depth),
                        budget,
                    )?;
                    reduce_rate(&mut resolved, sub_rate)?;
                }
            }
            Ok(resolved)
        }
        _ => Ok(declared_rate),
    }
}

/// Resolves the *declared* effective sample rate of a model node.
///
/// - A non-composite node with its own `sample_rate` reports it directly.
/// - A `Sequential` node resolves recursively from its `config.models` children,
///   enforcing [`MAX_SEQUENTIAL_DEPTH`] and [`MAX_SEQUENTIAL_TOTAL_CHILDREN`].
/// - A `SlimmableContainer` node resolves recursively from its `config.submodels[].model`
///   children, enforcing [`MAX_CONTAINER_DEPTH`] and [`MAX_SUBMODELS`].
/// - Any other node with no declared rate reports `None`.
///
/// The 48 kHz global default is **never** materialized inside the recursion
/// (`None` propagates), otherwise a childless-default would masquerade as a
/// declared rate and create a false conflict against a real sibling rate
/// (DEC-01b).
///
/// A rate-less `SlimmableContainer` whose submodels declare conflicting known
/// rates is rejected with `SequentialSampleRateMismatch` (`E1309`). This is
/// intentionally **stricter** than the C++ container, which never cross-checks
/// submodels against each other — see DEC-01c in `docs/cpp_parity_map.md` §5.1.
pub(crate) fn effective_known_rate(data: &NamModelData) -> Result<Option<f32>, NamErrorCode> {
    let mut budget = StructuralBudget::default();

    match data.architecture.as_str() {
        "Sequential" => {
            let mut resolved = data.sample_rate;
            if let Some(children) = data.config.models.as_deref() {
                for child in children {
                    budget.total_sequential_children = budget
                        .total_sequential_children
                        .checked_add(1)
                        .ok_or(NamErrorCode::SequentialChildrenExceedLimit)?;
                    if budget.total_sequential_children > MAX_SEQUENTIAL_TOTAL_CHILDREN {
                        return Err(NamErrorCode::SequentialChildrenExceedLimit);
                    }
                    let child_rate = resolve_value_rate(child, Some(1), None, &mut budget)?;
                    reduce_rate(&mut resolved, child_rate)?;
                }
            }
            Ok(resolved)
        }
        "SlimmableContainer" => {
            let mut resolved = data.sample_rate;
            if let Some(submodels) = data.config.submodels.as_deref() {
                if submodels.len() > MAX_SUBMODELS {
                    return Err(NamErrorCode::NamJsonSubmodelsExceedLimit);
                }
                for entry in submodels {
                    let model_val = entry
                        .get("model")
                        .ok_or(NamErrorCode::NamJsonInvalidEnvelope)?;
                    let sub_rate = resolve_value_rate(model_val, None, Some(0), &mut budget)?;
                    reduce_rate(&mut resolved, sub_rate)?;
                }
            }
            Ok(resolved)
        }
        _ => Ok(data.sample_rate),
    }
}

#[cfg(test)]
#[path = "sample_rate_test.rs"]
mod tests;
