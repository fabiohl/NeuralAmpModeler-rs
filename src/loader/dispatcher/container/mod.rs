// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! SlimmableContainer model parser/dispatcher.
//!
//! Parses the `SlimmableContainer` architecture (file version 0.7.0+).
//! Each submodel is a full independent `.nam` model, recursively built
//! via the main dispatcher. Recursion depth is capped to prevent DoS.

use crate::loader::nam_json::{JsonError, NamModelData, validate_model_data};
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
            serde::Deserialize::deserialize(model_json).with_context(|| {
                format!(
                    "SlimmableContainer: submodel[{}] failed to parse as a valid .nam model",
                    i
                )
            })?;

        // Submodels never pass through `parse_nam_json`: apply the same version
        // range and topology limits as the root and Sequential children before
        // building (NAMCore `container.cpp:163` → `get_dsp`, `get_dsp.cpp:146`).
        validate_model_data(&inner_data).map_err(|e| {
            warn!("[Dispatcher] Container submodel[{i}] rejected by model validation: {e}");
            anyhow::Error::from(e)
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
    let container_sr = crate::loader::sample_rate::rate_to_u32(resolved.unwrap_or(48000.0))
        .ok_or_else(|| {
            let value = resolved.unwrap_or(48000.0);
            warn!(
                "[Dispatcher] Container sample-rate resolution rejected: {value} is not a positive integer representable as u32"
            );
            anyhow::Error::from(JsonError::InvalidSampleRate {
                value,
                reason: "container sample rate must be a positive integer representable as u32",
            })
        })?;

    let container = ContainerModel::new_with_prewarm(submodels, container_sr, prewarm)
        .context("Container: failed to create ContainerModel")?;

    Ok(Box::new(StaticModel::Container(Box::new(container))))
}

#[cfg(test)]
mod tests {
    use crate::common::diagnostics::SystemSnapshot;
    use crate::loader::LoadOptions;
    use crate::loader::load_and_build_model;
    use crate::loader::nam_json::{JsonError, MAX_LAYERS, parse_nam_json};
    use crate::models::StaticModel;
    use crate::testing::fixtures::model_path;

    fn container_with_submodel(model: &str) -> String {
        format!(
            r#"{{"version":"0.7.0","architecture":"SlimmableContainer","config":{{"submodels":[{{"max_value":1.0,"model":{model}}}]}},"weights":[],"sample_rate":48000}}"#
        )
    }

    fn linear_submodel(version: &str) -> String {
        format!(
            r#"{{"version":"{version}","architecture":"Linear","config":{{"receptive_field":1,"bias":false}},"weights":[0.5],"sample_rate":48000}}"#
        )
    }

    #[test]
    fn test_container_submodel_valid_version_builds() {
        let data = parse_nam_json(&container_with_submodel(&linear_submodel("0.5.0")))
            .expect("root container must parse");
        assert!(
            super::build_container(&data, false).is_ok(),
            "submodel 0.5.0 inside the supported range must build"
        );
    }

    #[test]
    fn test_container_submodel_version_below_range_rejected() {
        let data = parse_nam_json(&container_with_submodel(&linear_submodel("0.4.0")))
            .expect("root container must parse");
        let err = super::build_container(&data, false)
            .err()
            .expect("submodel 0.4.0 must be rejected before build_model");
        assert!(
            matches!(
                err.downcast_ref::<JsonError>(),
                Some(JsonError::UnsupportedVersion { .. })
            ),
            "submodel version error must stay typed as JsonError::UnsupportedVersion: {err}"
        );
    }

    #[test]
    fn test_container_submodel_topology_above_limit_rejected() {
        let layers = std::iter::repeat_n("{}", MAX_LAYERS + 1)
            .collect::<Vec<_>>()
            .join(",");
        let wavenet = format!(
            r#"{{"version":"0.5.0","architecture":"WaveNet","config":{{"layers":[{layers}]}},"weights":[],"sample_rate":48000}}"#
        );
        let data =
            parse_nam_json(&container_with_submodel(&wavenet)).expect("root container must parse");
        let err = super::build_container(&data, false)
            .err()
            .expect("submodel above MAX_LAYERS must be rejected before build_model");
        assert!(
            matches!(
                err.downcast_ref::<JsonError>(),
                Some(JsonError::UnsupportedTopology { .. })
            ),
            "submodel topology error must stay typed as JsonError::UnsupportedTopology: {err}"
        );
    }

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
