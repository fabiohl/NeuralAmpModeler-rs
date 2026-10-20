// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A1 guardrails: validates that WaveNet A1 models do not activate A2-specific features.

use crate::loader::nam_json::model::NamLayerConfig;
use crate::models::a2::weights_layout::FILM_KEYS;

/// Validates that an A1 model does not enable unsupported A2 features.
pub fn validate_a1_guardrail(layers: &[NamLayerConfig]) -> Result<(), String> {
    for (i, layer) in layers.iter().enumerate() {
        let Some(ref raw) = layer.layer_raw else {
            continue;
        };
        if let Some(gm) = raw.get("gating_mode") {
            if let Some(arr) = gm.as_array() {
                if arr
                    .iter()
                    .any(|v| !(v.as_str() == Some("none") || v.is_null()))
                {
                    return Err(format!(
                        "Layer {i} has non-none gating_mode — A2 feature not supported in WaveNet A1."
                    ));
                }
            } else if !gm.is_null() {
                return Err(format!(
                    "Layer {i} has non-array gating_mode — A2 feature not supported in WaveNet A1."
                ));
            }
        }
        if raw
            .get("head1x1")
            .and_then(|h| h.get("active"))
            .and_then(|a| a.as_bool())
            .unwrap_or(false)
        {
            return Err(format!(
                "Layer {i} has active head1x1 — A2 feature not supported in WaveNet A1."
            ));
        }
        if let Some(l1x1) = raw.get("layer1x1") {
            // C++ A1 (`wavenet/model.cpp:948-954`) reads `layer1x1["active"]`
            // and `layer1x1["groups"]` unconditionally whenever the key is
            // present, so a malformed block (non-object, or missing/ill-typed
            // keys) is rejected rather than silently defaulted. Absence of the
            // `layer1x1` key keeps the C++ default `active=true, groups=1`.
            // The Rust A1 engine always consumes the 1×1 block, so
            // `active=false` and `groups != 1` remain unsupported in the A1 engine.
            let Some(obj) = l1x1.as_object() else {
                return Err(format!(
                    "Layer {i} has a non-object layer1x1 — malformed (C++ expects an \
                     object with 'active' and 'groups')."
                ));
            };
            let Some(active) = obj.get("active").and_then(serde_json::Value::as_bool) else {
                return Err(format!(
                    "Layer {i} has layer1x1 with a missing or non-boolean 'active' — malformed."
                ));
            };
            let Some(groups) = obj.get("groups").and_then(serde_json::Value::as_u64) else {
                return Err(format!(
                    "Layer {i} has layer1x1 with a missing or non-integer 'groups' — malformed."
                ));
            };
            if !active {
                return Err(format!(
                    "Layer {i} has layer1x1.active=false — not supported in the Rust A1 \
                     engine (PO backlog)."
                ));
            }
            if groups != 1 {
                return Err(format!(
                    "Layer {i} has layer1x1.groups={groups} — not supported in the Rust A1 \
                     engine (PO backlog)."
                ));
            }
        }
        for &(key, _) in FILM_KEYS {
            if raw
                .get(key)
                .and_then(|f| f.get("active"))
                .and_then(|a| a.as_bool())
                .unwrap_or(false)
            {
                return Err(format!(
                    "Layer {i} has active {key} — A2 feature not supported in WaveNet A1."
                ));
            }
        }
        if layer.gated.unwrap_or(false) {
            return Err(format!(
                "Layer {i} has gated=true — A2 feature not supported in WaveNet A1."
            ));
        }
        if let Some(sec_act) = raw.get("secondary_activation") {
            let has_non_trivial = match sec_act {
                serde_json::Value::Null => false,
                serde_json::Value::String(s) => s != "none" && !s.is_empty(),
                serde_json::Value::Array(arr) => arr.iter().any(|v| {
                    !(v.is_null() || v.as_str().is_some_and(|s| s == "none" || s.is_empty()))
                }),
                _ => true,
            };
            if has_non_trivial {
                return Err(format!(
                    "Layer {i} has non-trivial secondary_activation — A2 feature not supported in WaveNet A1."
                ));
            }
        }
    }
    Ok(())
}
