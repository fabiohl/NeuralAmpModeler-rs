// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Slimmable metadata validation for WaveNet models.

use crate::loader::nam_json::model::NamLayerConfig;
use crate::loader::nam_json::validation::MAX_WAVENET_FREE_CHANNELS;

/// Validates slimmable metadata invariants across layer arrays.
pub fn validate_slimmable_metadata(
    layers: &[NamLayerConfig],
    channels: &[usize],
) -> Result<Option<Vec<Vec<usize>>>, String> {
    let any_slimmable = layers.iter().any(|l| l.slimmable.is_some());
    if !any_slimmable {
        return Ok(None);
    }

    // In a slimmable model, C++ slimmable.cpp:136-146 validates that ALL layer arrays
    // have head_kernel_size == 1 and all convolution groups == 1.
    for (j, layer) in layers.iter().enumerate() {
        let hk = layer
            .head_kernel_size
            .or_else(|| {
                layer
                    .layer_raw
                    .as_ref()
                    .and_then(|raw| raw.get("head"))
                    .and_then(|h| h.get("kernel_size"))
                    .and_then(|k| k.as_u64())
                    .map(|k| k as usize)
            })
            .unwrap_or(1);
        if hk != 1 {
            return Err(format!(
                "Layer {j} slimmable requires head_kernel_size==1, got {hk}."
            ));
        }

        // C++ `compute_slim_bottleneck` (`slimmable.cpp:80-85`) reduces to
        // `slim_bn = slim_ch` only when `bottleneck == channels`; otherwise it
        // back-computes `max(1, bottleneck * slim_ch / channels)`. The Rust
        // slicer has no representation for a distinct bottleneck, so reject
        // fail-closed instead of silently producing a wrong channel width.
        let full_ch = channels.get(j).copied().unwrap_or(0);
        let bottleneck = layer.bottleneck.unwrap_or(full_ch);
        if bottleneck != full_ch {
            return Err(format!(
                "Layer {j} slimmable requires bottleneck==channels ({full_ch}), got {bottleneck}."
            ));
        }

        if let Some(raw) = layer.layer_raw.as_ref().and_then(|r| r.as_object()) {
            // C++ `validate_groups` (`slimmable.cpp:88-98`) reads these keys as
            // concrete integers; a present-but-unparseable value (negative or
            // non-numeric) is malformed and must not fall back to the default.
            for key in ["groups", "groups_input", "groups_input_mixin"] {
                if let Some(value) = raw.get(key) {
                    let Some(v) = value.as_u64() else {
                        return Err(format!(
                            "Layer {j} slimmable requires {key} to be a non-negative integer."
                        ));
                    };
                    if v != 1 {
                        return Err(format!("Layer {j} slimmable requires {key}==1, got {v}."));
                    }
                }
            }
            for (key, default_active) in [("layer1x1", true), ("head1x1", false)] {
                let Some(block) = raw.get(key) else {
                    continue;
                };
                let Some(obj) = block.as_object() else {
                    return Err(format!(
                        "Layer {j} slimmable requires '{key}' to be an object when present."
                    ));
                };
                let active = match obj.get("active") {
                    Some(a) => a.as_bool().ok_or_else(|| {
                        format!("Layer {j} slimmable requires '{key}.active' to be a boolean.")
                    })?,
                    None => default_active,
                };
                match obj.get("groups") {
                    Some(groups_val) => {
                        let Some(groups) = groups_val.as_u64() else {
                            return Err(format!(
                                "Layer {j} slimmable requires '{key}.groups' to be a \
                                 non-negative integer."
                            ));
                        };
                        if active && groups != 1 {
                            return Err(format!(
                                "Layer {j} slimmable requires {key}.groups==1, got {groups}."
                            ));
                        }
                    }
                    None if active => {
                        return Err(format!(
                            "Layer {j} slimmable requires '{key}.groups' when active."
                        ));
                    }
                    None => {}
                }
            }
        }
    }

    let mut per_array: Vec<Vec<usize>> = Vec::with_capacity(layers.len());

    for (i, layer) in layers.iter().enumerate() {
        let Some(ref cfg) = layer.slimmable else {
            per_array.push(Vec::new());
            continue;
        };

        let method = cfg.method.as_deref().unwrap_or("");
        if method != "slice_channels_uniform" {
            return Err(format!(
                "Layer {i} has unsupported slimmable method '{method}'. \
                 Only 'slice_channels_uniform' is supported."
            ));
        }

        let full_ch = channels.get(i).copied().unwrap_or(0);
        // C++ slimmable.cpp:560-572:
        // If kwargs.allowed_channels is absent, defaults to [1..=full_ch].
        // If explicitly present (e.g. `[]`), leaves allowed list as provided (empty = fixed array).
        let ac: Vec<usize> = match cfg
            .kwargs
            .as_ref()
            .and_then(|k| k.allowed_channels.as_deref())
        {
            None => (1..=full_ch).collect(),
            Some(list) => list.to_vec(),
        };

        if !ac.is_empty() {
            if ac.len() > MAX_WAVENET_FREE_CHANNELS {
                return Err(format!(
                    "Layer {i} has {} slimmable breakpoints, exceeding maximum {} — DoS/OOM protection.",
                    ac.len(),
                    MAX_WAVENET_FREE_CHANNELS
                ));
            }
            for w in ac.windows(2) {
                if w[0] >= w[1] {
                    return Err(format!(
                        "Layer {i} slimmable channels must be strictly ascending: {:?}.",
                        ac
                    ));
                }
            }
            if ac.contains(&0) {
                return Err(format!(
                    "Layer {i} slimmable channels contain a zero value: {:?}.",
                    ac
                ));
            }
            if ac.iter().any(|&c| c > MAX_WAVENET_FREE_CHANNELS) {
                return Err(format!(
                    "Layer {i} slimmable channels contain a value exceeding maximum {}: {:?}.",
                    MAX_WAVENET_FREE_CHANNELS, ac
                ));
            }
            if ac.last().copied().unwrap_or(0) != full_ch {
                return Err(format!(
                    "Layer {i} slimmable allowed_channels last entry must equal full channels ({full_ch}): {:?}.",
                    ac
                ));
            }
        }

        per_array.push(ac);
    }

    // C++ slimmable.cpp:371-393: At least one layer array must have non-empty allowed_channels.
    if !per_array.iter().any(|a| !a.is_empty()) {
        return Err(
            "SlimmableWavenet: at least one layer array must have allowed_channels.".to_string(),
        );
    }

    Ok(Some(per_array))
}
