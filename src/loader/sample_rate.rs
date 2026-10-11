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

use crate::common::diagnostics::NamErrorCode;
use crate::loader::nam_json::NamModelData;
use crate::models::sequential::resolve_known_sample_rate;

/// Resolves the *declared* effective sample rate of a model node.
///
/// - A node with its own `sample_rate` reports it directly.
/// - A `Sequential` node with no own rate resolves recursively from its
///   `config.models` children.
/// - A `SlimmableContainer` node with no own rate resolves recursively from
///   its `config.submodels[].model` children.
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
    if let Some(rate) = data.sample_rate {
        return Ok(Some(rate));
    }

    match data.architecture.as_str() {
        "Sequential" => {
            let children = data.config.models.as_deref().unwrap_or(&[]);
            let mut rates = Vec::with_capacity(children.len());
            for child in children {
                let child_data: NamModelData = serde_json::from_value(child.clone())
                    .map_err(|_| NamErrorCode::NamJsonParseError)?;
                rates.push(effective_known_rate(&child_data)?);
            }
            resolve_known_sample_rate(&rates, None)
        }
        "SlimmableContainer" => {
            let submodels = data.config.submodels.as_deref().unwrap_or(&[]);
            let mut rates = Vec::with_capacity(submodels.len());
            for entry in submodels {
                let model_json = entry
                    .get("model")
                    .ok_or(NamErrorCode::NamJsonInvalidEnvelope)?;
                let inner: NamModelData = serde_json::from_value(model_json.clone())
                    .map_err(|_| NamErrorCode::NamJsonParseError)?;
                rates.push(effective_known_rate(&inner)?);
            }
            resolve_known_sample_rate(&rates, None)
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::nam_json::parse_nam_json;

    fn inner_wavenet(rate: Option<f32>) -> String {
        let sr = match rate {
            Some(r) => format!(r#", "sample_rate": {r}"#),
            None => String::new(),
        };
        format!(
            r#"{{
                "version": "0.5.4",
                "architecture": "WaveNet",
                "config": {{
                    "layers": [{{
                        "input_size": 1, "condition_size": 1, "head_size": 4,
                        "channels": 8, "kernel_size": 3, "dilations": [1,2,4,8],
                        "activation": "Tanh", "gated": false, "head_bias": false
                    }}],
                    "head": null
                }},
                "weights": [0.0]{sr}
            }}"#
        )
    }

    fn rate_less_container(child_a: &str, child_b: &str) -> String {
        format!(
            r#"{{
                "version": "0.7.0",
                "architecture": "SlimmableContainer",
                "config": {{
                    "layers": [], "head": null,
                    "submodels": [
                        {{"max_value": 0.5, "model": {child_a}}},
                        {{"max_value": 1.0, "model": {child_b}}}
                    ]
                }},
                "weights": [0.0]
            }}"#
        )
    }

    /// DEC-01c: a rate-less container must reject a child-vs-child known-rate
    /// conflict (stricter than C++ `container.cpp:35-46`, which never
    /// cross-checks submodels).
    #[test]
    fn container_without_rate_rejects_child_rate_conflict() {
        let json =
            rate_less_container(&inner_wavenet(Some(44100.0)), &inner_wavenet(Some(48000.0)));
        let data = parse_nam_json(&json).expect("parse container");
        assert!(matches!(
            effective_known_rate(&data),
            Err(NamErrorCode::SequentialSampleRateMismatch)
        ));
    }

    /// DEC-01c: matching known rates resolve; all-unknown resolves to `None`
    /// (the 48 kHz default is materialized only at the top-level boundary).
    #[test]
    fn container_without_rate_resolves_matching_or_unknown() {
        let matching =
            rate_less_container(&inner_wavenet(Some(44100.0)), &inner_wavenet(Some(44100.0)));
        let data = parse_nam_json(&matching).expect("parse container");
        assert_eq!(effective_known_rate(&data), Ok(Some(44100.0)));

        let unknown = rate_less_container(&inner_wavenet(None), &inner_wavenet(None));
        let data = parse_nam_json(&unknown).expect("parse container");
        assert_eq!(effective_known_rate(&data), Ok(None));
    }
}
