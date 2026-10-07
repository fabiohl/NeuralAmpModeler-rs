// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `Sequential` architecture builder — chain construction over the validated
//! topology.
//!
//! The envelope/topology gate ([`crate::loader::nam_json::get_sequential_topology`])
//! runs first and rejects every hostile shape with a typed [`NamErrorCode`]
//! (top-level weights empty, non-empty `config.models`, complete child
//! envelopes, recursion depth and total child/weight budgets).
//!
//! On top of the validated topology, this builder mirrors the C++ chain
//! construction (`sequential.cpp:222-241` `SequentialConfig::create` per
//! child plus `nam::get_dsp` recursion for nested chains):
//!
//! 1. Each child raw JSON envelope is re-deserialized through the strict
//!    `NamModelData` serde path (weight caps, finite checks, `-1.0` unknown
//!    sample-rate normalization to `None`) and re-validated
//!    (`validate_model_data` — sub-models bypass `parse_nam_json` and must
//!    validate explicitly).
//! 2. Children are built recursively through the dispatcher's `build_model`
//!    (nested `Sequential` chains re-enter this builder, depth-bounded by
//!    the topology scan).
//! 3. The chain entity resolves its expected sample rate (DEC-01, unknowns
//!    ignored, conflicts fail-closed with `E1309`), validates the interior
//!    channel links (`E1308`), and owns the single-pass chain prewarm.
//!
//! Construction runs strictly off-RT (audio thread only ever sees the
//! finished [`StaticModel`]).

use log::{info, warn};

use crate::loader::nam_json::{NamModelData, get_sequential_topology, validate_model_data};
use crate::models::{NamModel, StaticModel, sequential::SequentialModel};

/// Builds a `Sequential` chain from already-validated model data.
pub(crate) fn build_sequential(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    if get_sequential_topology(data)
        .map_err(anyhow::Error::from)?
        .is_none()
    {
        anyhow::bail!("Sequential dispatcher invoked for a non-Sequential model");
    }

    let raw_children = data.config.models.as_deref().unwrap_or(&[]);

    let mut stages: Vec<StaticModel> = Vec::with_capacity(raw_children.len());
    let mut declared_rates: Vec<Option<f32>> = Vec::with_capacity(raw_children.len());
    for (index, raw_child) in raw_children.iter().enumerate() {
        // Complete-envelope + version/topology validation for the child
        // (mirrors `nam::get_dsp` on the child JSON). deserialize via the
        // strict serde path so weight caps, finiteness and the -1.0 unknown
        // marker all apply.
        let child_data: NamModelData = serde_json::from_value(raw_child.clone())?;
        if let Err(e) = validate_model_data(&child_data) {
            warn!(
                "[Dispatcher] Sequential child[{}] rejected by model validation: {}",
                index, e
            );
            return Err(anyhow::Error::from(e));
        }
        declared_rates.push(child_data.sample_rate);
        match super::build_model(&child_data) {
            // Unbox into stage storage: the chain keeps plain `StaticModel`
            // stages so stage calls never chase a second indirection.
            Ok(model) => stages.push(*model),
            // Child failures propagate bare so the typed chain (or family)
            // error code stays downcastable at the load boundary.
            Err(e) => {
                warn!("[Dispatcher] Sequential child[{}] build failed: {e}", index);
                return Err(e);
            }
        }
    }

    let chain = match SequentialModel::new(stages, declared_rates, data.sample_rate) {
        Ok(model) => model,
        Err(e) => {
            warn!("[Dispatcher] Sequential chain rejected: {e}");
            return Err(anyhow::Error::from(e));
        }
    };

    info!(
        "[Dispatcher] Sequential chain built: stages={}, in_ch={}, out_ch={}, \
         expected_sample_rate={} Hz, prewarm_samples={}",
        chain.num_stages(),
        chain.in_channels(),
        chain.out_channels(),
        chain.expected_sample_rate(),
        chain.prewarm_samples()
    );
    Ok(Box::new(StaticModel::Sequential(Box::new(chain))))
}
