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
use log::warn;

/// Maximum nesting depth for SlimmableContainer recursion.
///
/// # Invariants
/// Must remain strictly <= `MAX_UNIFIED_DEPTH` (8) defined in
/// `src/loader/dispatcher/wavenet/dynamic.rs` to bound memory consumption
/// and prevent stack exhaustion across nested container and condition models.
pub(crate) const MAX_CONTAINER_DEPTH: usize = 4;

/// Builds a `Box<StaticModel>` for the `SlimmableContainer` architecture.
///
/// Parses `config.submodels[]`, recursively builds each submodel via the main
/// dispatcher, validates ordering / sample rate uniformity, and wraps them in
/// a `ContainerModel`.
pub(crate) fn build_container(
    data: &NamModelData,
    prewarm: bool,
) -> anyhow::Result<Box<StaticModel>> {
    build_container_inner(data, 0, prewarm)
}

/// Recursive helper for building a container model while enforcing `MAX_CONTAINER_DEPTH`.
pub(crate) fn build_container_inner(
    data: &NamModelData,
    depth: usize,
    prewarm: bool,
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

    let container_declared = data.sample_rate;

    let mut submodels: Vec<(f32, Box<StaticModel>)> = Vec::with_capacity(submodels_json.len());
    let mut sub_rates: Vec<Option<f32>> = Vec::with_capacity(submodels_json.len());

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

        // Recursive declared rate: a nested Sequential/Container child
        // propagates its own children's rate instead of materializing 48 kHz.
        sub_rates.push(crate::loader::sample_rate::effective_known_rate(
            &inner_data,
        )?);

        if inner_data.architecture == "SlimmableContainer" {
            let model = build_container_inner(&inner_data, depth + 1, prewarm)
                .with_context(|| format!("Container -> submodel[{}] (nested container)", i))?;
            submodels.push((max_value, model));
        } else {
            let model = super::build_model(&inner_data)
                .with_context(|| format!("Container -> submodel[{}]", i))?;
            submodels.push((max_value, model));
        }
    }

    // C++ container rule: the container's own declared rate governs; when
    // absent, the first known submodel rate is used; a conflict between the
    // declared container rate and a known submodel rate is fail-closed. Only
    // when nothing declares a rate do we fall back to the 48 kHz default.
    let resolved =
        crate::models::sequential::resolve_known_sample_rate(&sub_rates, container_declared)
            .map_err(|e| {
                warn!("[Dispatcher] Container sample-rate resolution rejected: {e}");
                anyhow::Error::from(e)
            })?;
    let container_sr = resolved.unwrap_or(48000.0) as u32;

    let container = ContainerModel::new_with_prewarm(submodels, container_sr, prewarm)
        .context("Container: failed to create ContainerModel")?;

    Ok(Box::new(StaticModel::Container(Box::new(container))))
}

#[cfg(test)]
mod tests {
    use crate::common::diagnostics::SystemSnapshot;
    use crate::loader::LoadOptions;
    use crate::loader::load_and_build_model;
    use crate::models::StaticModel;
    use crate::testing::fixtures::model_path;

    #[test]
    fn test_container_builds_slimmable_with_relu_submodel() {
        let sys = SystemSnapshot::capture();
        let path = model_path("slimmable_container.nam");
        let result = load_and_build_model(&path, &sys, false, LoadOptions::default());
        let model = result.expect(
            "slimmable_container.nam must build successfully with ReLU activation supported",
        );

        let container = match model.model_l.as_ref().and_then(|m| match m.as_ref() {
            StaticModel::Container(c) => Some(c),
            _ => None,
        }) {
            Some(c) => c,
            None => panic!("Expected StaticModel::Container, got a different variant"),
        };

        assert_eq!(container.submodels().len(), 3);
        let max_values: Vec<f32> = container.submodels().iter().map(|(mv, _)| *mv).collect();
        assert_eq!(max_values, vec![0.33, 0.66, 1.0]);

        let sub_arches: Vec<&str> = container
            .submodels()
            .iter()
            .map(|(_, sm)| match sm.as_ref() {
                StaticModel::Lstm1x3(_) => "LSTM",
                StaticModel::WavenetDyn(_) => "WaveNetDyn",
                StaticModel::WavenetNano(_) => "Nano",
                _ => "Unknown",
            })
            .collect();
        assert_eq!(sub_arches, vec!["LSTM", "WaveNetDyn", "Nano"]);
    }
}
