// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Topology and envelope validation for the `Sequential` architecture.
//!
//! Mirrors the C++ NAMcore `sequential.cpp` semantics at the topology level:
//! - `SequentialConfig::create` (L224–227): top-level `weights` must be empty.
//! - `build_models` (L82–108): `config.models` must be a present, non-empty
//!   array and every child must be a complete `.nam` envelope
//!   (`version`, `architecture`, `config`, `weights`).
//!
//! Additionally applies Rust-only robustness hardening declared as
//! intentional divergences from the C++ reference (which applies no limits),
//! see [`MAX_SEQUENTIAL_DEPTH`]/[`MAX_SEQUENTIAL_TOTAL_CHILDREN`] and
//! `docs/cpp_parity_map.md` §6.6:
//! - Maximum nesting depth for recursive `Sequential` children.
//! - Maximum total child count across the whole tree (checked arithmetic).
//! - Aggregate weight budget across the whole tree (checked arithmetic).
//!
//! The scan walks the raw JSON tree with constant stack depth (bounded by
//! [`MAX_SEQUENTIAL_DEPTH`]) and rejects a hostile tree before any child
//! model allocation or tensor decoding happens downstream.

use serde_json::Value;

use super::super::data::NamModelData;
use super::super::validation::{
    MAX_SEQUENTIAL_DEPTH, MAX_SEQUENTIAL_TOTAL_CHILDREN, MAX_SEQUENTIAL_TOTAL_WEIGHTS,
    validate_envelope,
};
use crate::common::diagnostics::NamErrorCode;

/// Validated shell of one direct child of a `Sequential` chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialChildTopology {
    /// Declared `architecture` of the child envelope (verbatim, case-sensitive).
    pub architecture: String,
    /// Whether the child declares another `Sequential` chain (recursion).
    pub is_sequential: bool,
    /// Declared weight count in the child envelope (`weights.len()`).
    pub weights_len: usize,
}

/// Validated topology of a `Sequential` model envelope.
///
/// Produced by [`get_sequential_topology`] after the full tree scan has
/// passed; consumed by the `Sequential` builder for chain construction
/// (channel links, sample-rate resolution, and per-stage buffer sizing are
/// model-level responsibilities downstream of this module).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequentialTopology {
    /// Direct children of the root `Sequential` (`config.models`, in order).
    pub children: Vec<SequentialChildTopology>,
    /// Total child count over the whole tree (nested `Sequential`
    /// descendants included), bounded by [`MAX_SEQUENTIAL_TOTAL_CHILDREN`].
    pub total_models: usize,
    /// Deepest `Sequential` nesting level reached (root = 1), bounded by
    /// [`MAX_SEQUENTIAL_DEPTH`].
    pub max_depth_reached: usize,
    /// Aggregate weight count over the whole tree (checked arithmetic),
    /// bounded by [`MAX_SEQUENTIAL_TOTAL_WEIGHTS`].
    pub aggregate_weights: usize,
}

/// Checks and returns the validated `Sequential` topology from model data.
///
/// Returns `Ok(None)` when `architecture` is not `"Sequential"` (case-sensitively —
/// lowercase `"sequential"` stays an unsupported architecture at dispatch),
/// `Ok(Some(topology))` when the whole tree passes validation, or the typed
/// [`NamErrorCode`] of the first failed rule:
/// - [`NamErrorCode::SequentialTopLevelWeightsNotEmpty`] (E1310): the root
///   `weights` array is not empty.
/// - [`NamErrorCode::SequentialEmptyModels`] (E1306): `config.models` is
///   missing, is not an array, or is empty — at the root or at any nested
///   `Sequential` level.
/// - [`NamErrorCode::SequentialIncompleteChild`] (E1307): a child is not a
///   complete `.nam` envelope (missing/ill-typed `version`, `architecture`,
///   `config`, or `weights`).
/// - [`NamErrorCode::SequentialRecursionDepthExceeded`] (E1311): Rust-only
///   hardening — nesting exceeds [`MAX_SEQUENTIAL_DEPTH`].
/// - [`NamErrorCode::SequentialChildrenExceedLimit`] (E1314): Rust-only
///   hardening — the tree breaches [`MAX_SEQUENTIAL_TOTAL_CHILDREN`],
///   [`MAX_SEQUENTIAL_TOTAL_WEIGHTS`], or overflows the checked budget.
pub fn get_sequential_topology(
    data: &NamModelData,
) -> Result<Option<SequentialTopology>, NamErrorCode> {
    if data.architecture != "Sequential" {
        return Ok(None);
    }
    validate_sequential_topology(data).map(Some)
}

/// Validate the root `Sequential` envelope and scan the whole tree.
///
/// Rule order mirrors the C++ reference: the top-level weights invariant
/// (`SequentialConfig::create` L224–227) is checked before the `models`
/// array invariants (`build_models` L83–88); every child envelope is then
/// validated with [`validate_envelope`] before its entry is counted.
fn validate_sequential_topology(data: &NamModelData) -> Result<SequentialTopology, NamErrorCode> {
    if !data.weights.is_empty() {
        return Err(NamErrorCode::SequentialTopLevelWeightsNotEmpty);
    }

    let Some(models) = data.config.models.as_ref() else {
        return Err(NamErrorCode::SequentialEmptyModels);
    };
    if models.is_empty() {
        return Err(NamErrorCode::SequentialEmptyModels);
    }

    let mut budget = ScanBudget {
        total_models: 0,
        aggregate_weights: 0,
        max_depth_reached: 1,
    };
    let children = scan_children(models, 1, &mut budget)?;

    Ok(SequentialTopology {
        children,
        total_models: budget.total_models,
        max_depth_reached: budget.max_depth_reached,
        aggregate_weights: budget.aggregate_weights,
    })
}

/// Bookkeeping accumulated by the pre-build tree scan.
struct ScanBudget {
    total_models: usize,
    aggregate_weights: usize,
    max_depth_reached: usize,
}

/// Validates one `Sequential` level and recurses into nested chains.
///
/// `level` is the nesting level of the `Sequential` owning `models` (root = 1).
fn scan_children(
    models: &[Value],
    level: usize,
    budget: &mut ScanBudget,
) -> Result<Vec<SequentialChildTopology>, NamErrorCode> {
    let mut shells = Vec::with_capacity(models.len());

    for (index, child) in models.iter().enumerate() {
        // Mirror `build_models` (L93–103): each child must be a complete NAM
        // envelope before anything descends into it. Legacy bare child
        // configs (e.g. `{receptive_field, bias}`) are rejected here.
        if let Err(reason) = validate_envelope(child) {
            log::warn!(
                "[Loader] Invalid field rejected: field='config.models[{}]', detail: {}",
                index,
                reason
            );
            return Err(NamErrorCode::SequentialIncompleteChild);
        }

        let Some(architecture) = child.get("architecture").and_then(Value::as_str) else {
            // Unreachable after `validate_envelope`; kept typed for defense.
            return Err(NamErrorCode::SequentialIncompleteChild);
        };
        let is_sequential = architecture == "Sequential";
        let weights_len = child
            .get("weights")
            .and_then(Value::as_array)
            .map_or(0, |weights| weights.len());

        budget.total_models = budget
            .total_models
            .checked_add(1)
            .ok_or(NamErrorCode::SequentialChildrenExceedLimit)?;
        if budget.total_models > MAX_SEQUENTIAL_TOTAL_CHILDREN {
            return Err(NamErrorCode::SequentialChildrenExceedLimit);
        }

        budget.aggregate_weights = budget
            .aggregate_weights
            .checked_add(weights_len)
            .ok_or(NamErrorCode::SequentialChildrenExceedLimit)?;
        if budget.aggregate_weights > MAX_SEQUENTIAL_TOTAL_WEIGHTS {
            return Err(NamErrorCode::SequentialChildrenExceedLimit);
        }

        let child_topology = SequentialChildTopology {
            architecture: architecture.to_string(),
            is_sequential,
            weights_len,
        };

        if is_sequential {
            // The child is the root of another `Sequential` chain: its own
            // envelope must satisfy the same top-level invariants before
            // this scan descends into it.
            let child_level = level
                .checked_add(1)
                .ok_or(NamErrorCode::SequentialChildrenExceedLimit)?;
            if child_level > MAX_SEQUENTIAL_DEPTH {
                return Err(NamErrorCode::SequentialRecursionDepthExceeded);
            }

            let child_weights_non_empty = child
                .get("weights")
                .and_then(Value::as_array)
                .is_some_and(|weights| !weights.is_empty());
            if child_weights_non_empty {
                return Err(NamErrorCode::SequentialTopLevelWeightsNotEmpty);
            }

            budget.max_depth_reached = budget.max_depth_reached.max(child_level);

            let Some(nested_models) = child
                .get("config")
                .and_then(|config| config.get("models"))
                .and_then(Value::as_array)
            else {
                return Err(NamErrorCode::SequentialEmptyModels);
            };
            if nested_models.is_empty() {
                return Err(NamErrorCode::SequentialEmptyModels);
            }

            // The nested level was validated with the same rule set; only the
            // root level shells are reported to the caller.
            scan_children(nested_models, child_level, budget)?;
        }

        shells.push(child_topology);
    }

    Ok(shells)
}
