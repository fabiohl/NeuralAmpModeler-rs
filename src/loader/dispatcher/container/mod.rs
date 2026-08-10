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
            let model = build_container_inner(&inner_data, depth + 1)
                .with_context(|| format!("Container -> submodel[{}] (nested container)", i))?;
            submodels.push((max_value, model));
        } else {
            let model = super::build_model(&inner_data)
                .with_context(|| format!("Container -> submodel[{}]", i))?;
            submodels.push((max_value, model));
        }
    }

    let container = ContainerModel::new(submodels, container_sr)
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
