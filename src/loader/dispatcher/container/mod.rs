// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! SlimmableContainer model parser/dispatcher.
//!
//! Parses the `SlimmableContainer` architecture (file version 0.7.0+).
//! Each submodel is a full independent `.nam` model, recursively built
//! via the main dispatcher. Recursion depth is capped to prevent DoS.

use crate::loader::nam_json::JsonError;
use crate::loader::nam_json::NamModelData;
use crate::models::StaticModel;
use crate::models::container::ContainerModel;
use anyhow::Context;

/// Maximum nesting depth for SlimmableContainer recursion.
const MAX_CONTAINER_DEPTH: usize = 4;

/// Builds a `Box<StaticModel>` for the `SlimmableContainer` architecture.
///
/// Parses `config.submodels[]`, recursively builds each submodel via the main
/// dispatcher, validates ordering / sample rate uniformity, and wraps them in
/// a `ContainerModel`.
pub fn build_container(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    build_container_inner(data, 0)
}

pub(crate) fn build_container_inner(
    data: &NamModelData,
    depth: usize,
) -> anyhow::Result<Box<StaticModel>> {
    if depth > MAX_CONTAINER_DEPTH {
        anyhow::bail!(JsonError::SubmodelsTooDeep {
            depth,
            max_depth: MAX_CONTAINER_DEPTH,
        });
    }

    let submodels_json = data
        .config
        .submodels
        .as_ref()
        .context("SlimmableContainer: missing 'config.submodels' array")?;

    if submodels_json.is_empty() {
        anyhow::bail!("SlimmableContainer: 'submodels' must be a non-empty array");
    }

    let container_sr = data.sample_rate.map(|s| s as u32).unwrap_or(48000);

    let mut submodels: Vec<(f32, Box<StaticModel>)> = Vec::with_capacity(submodels_json.len());

    for (i, entry) in submodels_json.iter().enumerate() {
        let max_value: f32 = entry
            .get("max_value")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .with_context(|| {
                format!(
                    "SlimmableContainer: submodel[{}] missing or invalid 'max_value'",
                    i
                )
            })?;

        let model_json = entry
            .get("model")
            .with_context(|| format!("SlimmableContainer: submodel[{}] missing 'model'", i))?;

        let inner_data: NamModelData =
            serde_json::from_value(model_json.clone()).with_context(|| {
                format!(
                    "SlimmableContainer: submodel[{}] failed to parse as a valid .nam model",
                    i
                )
            })?;

        let sub_sr = inner_data
            .sample_rate
            .map(|s| s as u32)
            .unwrap_or(container_sr);

        if sub_sr != container_sr {
            anyhow::bail!(
                "SlimmableContainer: submodel[{}] sample rate mismatch \
                 (container={}, submodel={})",
                i,
                container_sr,
                sub_sr
            );
        }

        if inner_data.architecture == "SlimmableContainer" {
            let model = build_container_inner(&inner_data, depth + 1).with_context(|| {
                format!(
                    "SlimmableContainer: nested container submodel[{}] build failed",
                    i
                )
            })?;
            submodels.push((max_value, model));
        } else {
            let model = super::build_model(&inner_data)
                .with_context(|| format!("SlimmableContainer: submodel[{}] build failed", i))?;
            submodels.push((max_value, model));
        }
    }

    let container = ContainerModel::new(submodels, container_sr)
        .context("SlimmableContainer: failed to create ContainerModel")?;

    Ok(Box::new(StaticModel::Container(Box::new(container))))
}
