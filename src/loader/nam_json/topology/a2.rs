// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Shape Detection — Hybrid Dispatch (A2 Fast-Path vs A2 Dynamic)

use super::super::data::NamModelData;

/// Valid A2 topologies.
#[derive(Debug, Clone, PartialEq)]
pub enum A2TopologyResult {
    /// Fast-path const-generic with exact channels (3 or 8).
    KnownFastPath(u8),
    /// Valid A2 geometry that falls outside the rigid fast-path.
    Dynamic,
}

/// Shape-based A2 detector: returns `Some(A2TopologyResult)` if the parsed model data
/// matches the A2 architectural signature. If the model strictly matches the
/// C++ fast-path criteria, it returns `KnownFastPath(CH)`, otherwise `Dynamic`.
///
/// Checks that require raw JSON (activation arrays, FiLM keys, etc.) are
/// skipped when `layer_raw` is `None` (structs constructed directly, not via
/// JSON deserialization). This keeps existing test helpers working while
/// enforcing full parity for models loaded from actual `.nam` files.
pub fn is_a2_shape(data: &NamModelData) -> Option<A2TopologyResult> {
    use crate::models::a2::{
        A2_DILATIONS, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS, A2_VALID_CHANNELS,
    };

    // 1. Architecture must be WaveNet
    if data.architecture != "WaveNet" {
        return None;
    }

    // T5.1: Slimmable WaveNet models go through the A1 free-geometry path
    // (get_wavenet_topology), which handles slimmable metadata parsing and
    // permitted-channel validation. The A2 fast-path and dynamic engine do not
    // support slimmable channel slicing.
    if data.config.layers.iter().any(|l| l.slimmable.is_some()) {
        return None;
    }

    // A2 models do not use Tanh as their main activation.
    // If any layer explicitly requests Tanh, this is an A1 (standard WaveNet) model.
    for l in &data.config.layers {
        if l.activation.as_deref() == Some("Tanh") {
            return None;
        }
    }

    // 2. At least one layer array (a2_fast.cpp:876-879).
    // Relaxed from != 1 to < 1 to support multi-array
    // cascade topologies (including hybrid condition_dsp sub-models
    // with FiLM features). Multi-array models must have
    // at least one A2-specific feature (head1x1, FiLM, or array/object
    // activation) to avoid misclassifying standard WaveNet models.
    let layers = &data.config.layers;
    if layers.is_empty() {
        return None;
    }
    if layers.len() > 1 {
        let mut has_a2_feature = false;
        let mut any_layer_raw = false;
        for l in layers.iter() {
            if let Some(ref raw) = l.layer_raw {
                any_layer_raw = true;
                // head1x1 active is an A2-specific feature.
                if raw
                    .get("head1x1")
                    .and_then(|h| h.get("active"))
                    .and_then(|a| a.as_bool())
                    .unwrap_or(false)
                {
                    has_a2_feature = true;
                    break;
                }
                // FiLM slots are A2-specific.
                for key in &[
                    "conv_pre_film",
                    "conv_post_film",
                    "input_mixin_pre_film",
                    "input_mixin_post_film",
                    "activation_pre_film",
                    "activation_post_film",
                    "layer1x1_post_film",
                    "head1x1_post_film",
                ] {
                    if raw
                        .get(key)
                        .and_then(|v| v.get("active"))
                        .and_then(|a| a.as_bool())
                        .unwrap_or(false)
                    {
                        has_a2_feature = true;
                        break;
                    }
                }
                // Array/object activation is A2-specific (standard WaveNet uses
                // string activation like "Tanh").
                if let Some(act) = raw.get("activation")
                    && (act.is_array() || act.is_object())
                {
                    has_a2_feature = true;
                    break;
                }
            }
        }
        if any_layer_raw && !has_a2_feature {
            // T5.1: condition_dsp is an A2-specific feature — the oracle always
            // routes models with condition_dsp through the A2 path. The builder
            // must match to avoid architecture mismatch.
            if data.config.condition_dsp.is_some() {
                has_a2_feature = true;
            }
        }
        if any_layer_raw && !has_a2_feature {
            return None;
        }
    }

    // 3. No post-stack head (a2_fast.cpp:881-884)
    if let Some(ref head) = data.config.head
        && !head.is_null()
    {
        return None;
    }

    // 4. head_scale present and numeric (a2_fast.cpp:886-890)
    data.config.head_scale.as_ref()?;

    let l0 = &layers[0];

    // 10. kernel_sizes must match A2_KERNEL_SIZES exactly for fast-path (a2_fast.cpp:910-918).
    // Non-canonical sizes indicate an A2-generic model → route to Dynamic.
    // S13.2: also accept models with scalar kernel_size (plural absent); those
    // are always A2-generic (Dynamic).
    let ks = match l0.kernel_sizes.as_deref() {
        Some(s) => s.to_vec(),
        None => {
            // kernel_size (scalar) + dilations → A2 generic
            if l0.kernel_size.is_some() && l0.dilations.is_some() {
                return Some(A2TopologyResult::Dynamic);
            }
            return None;
        }
    };
    if ks.len() != A2_NUM_LAYERS || ks.iter().zip(A2_KERNEL_SIZES.iter()).any(|(a, b)| *a != *b) {
        return Some(A2TopologyResult::Dynamic);
    }

    // 11. Dilations must match A2_DILATIONS exactly for fast-path (a2_fast.cpp:920-928).
    let dils = l0.dilations.as_deref()?;
    if dils.len() != A2_NUM_LAYERS || dils.iter().zip(A2_DILATIONS.iter()).any(|(a, b)| *a != *b) {
        return Some(A2TopologyResult::Dynamic);
    }

    // --- If we reached here, the model is unequivocally a WaveNet A2 fast-path candidate. ---
    // From here on, we validate whether it fits the const-generic fast-path.

    // 5. in_channels defaults to 1, must be 1 (a2_fast.cpp:892-894)
    if data.config.in_channels.unwrap_or(1) != 1 {
        return Some(A2TopologyResult::Dynamic);
    }

    // 6. input_size must be 1 (a2_fast.cpp:898)
    if l0.input_size != Some(1) {
        return Some(A2TopologyResult::Dynamic);
    }

    // 7. condition_size must be 1 (a2_fast.cpp:900)
    if l0.condition_size != Some(1) {
        return Some(A2TopologyResult::Dynamic);
    }

    let ch = match l0.channels {
        Some(c) => c as u8,
        None => return Some(A2TopologyResult::Dynamic),
    };

    // 8. channels and bottleneck must match (a2_fast.cpp:903-906)
    let bn = l0.bottleneck.unwrap_or(0);
    if ch as usize != bn {
        return Some(A2TopologyResult::Dynamic);
    }

    // 9. Channels must be exactly 3 or 8 (a2_fast.cpp:907-908)
    if !A2_VALID_CHANNELS.contains(&ch) {
        return Some(A2TopologyResult::Dynamic);
    }

    // ── Checks 12-19 require raw JSON (a2_fast.cpp:930-1000) ──
    if let Some(ref raw) = l0.layer_raw {
        // 12. All activations must be LeakyReLU(negative_slope ≈ 0.01)
        if !check_activations_are_leaky_relu(raw, A2_NUM_LAYERS, A2_LEAKY_SLOPE as f64) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 13. gating_mode: all "none" or absent
        if !check_gating_mode_all_none(raw, A2_NUM_LAYERS) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 14. secondary_activation: all null or absent
        if !check_secondary_activation_all_null(raw) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 15. head1x1 must be inactive
        if !check_head1x1_inactive(raw) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 16. layer1x1 active with groups=1
        if !check_layer1x1_active_groups1(raw) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 17. Layer-array head: out_channels=1, kernel_size=16, bias=true
        if !check_layer_array_head(raw) {
            return Some(A2TopologyResult::Dynamic);
        }

        if !check_film_all_inactive(raw) {
            // B.1.1 (F5): FiLM routing policy — C++ a2_fast.cpp rejects FiLM
            // models, routing them to the generic WaveNet engine. Rust fast-path
            // output diverges from C++ (measured CH=3 SNR 18.1 dB, CH=8 SNR 36.0 dB).
            // Route to WaveNetA2Dyn to match C++ behavior.
            return Some(A2TopologyResult::Dynamic);
        }

        // 19. groups_input == 1 AND groups_input_mixin == 1
        if !check_groups_are_1(raw) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 20. Not slimmable
        if !check_not_slimmable(raw) {
            return Some(A2TopologyResult::Dynamic);
        }
    }

    Some(A2TopologyResult::KnownFastPath(ch))
}

// ── Raw JSON helper functions for strict A2 shape checks ──

fn get_f64(val: &serde_json::Value) -> Option<f64> {
    val.as_f64().or_else(|| val.as_i64().map(|i| i as f64))
}

/// Checks that every entry in the `activation` array is a LeakyReLU with the
/// expected negative_slope.
fn check_activations_are_leaky_relu(
    raw: &serde_json::Value,
    num_layers: usize,
    slope: f64,
) -> bool {
    let arr = match raw.get("activation").and_then(|v| v.as_array()) {
        Some(a) if a.len() == num_layers => a,
        _ => return false,
    };
    for entry in arr {
        let obj = match entry.as_object() {
            Some(o) => o,
            None => return false,
        };
        if obj.get("type").and_then(|v| v.as_str()) != Some("LeakyReLU") {
            return false;
        }
        let ns = match obj.get("negative_slope").and_then(get_f64) {
            Some(n) => n,
            None => return false,
        };
        if (ns - slope).abs() > 1e-6 {
            return false;
        }
    }
    true
}

/// Checks that `gating_mode` is absent, null, or an array of all "none".
fn check_gating_mode_all_none(raw: &serde_json::Value, num_layers: usize) -> bool {
    match raw.get("gating_mode") {
        None | Some(serde_json::Value::Null) => true,
        Some(arr) => {
            let elems = match arr.as_array() {
                Some(a) if a.len() == num_layers => a,
                _ => return false,
            };
            elems
                .iter()
                .all(|v| v.as_str() == Some("none") || v.is_null())
        }
    }
}

/// Checks that `secondary_activation` is absent, null, or an array of all null.
fn check_secondary_activation_all_null(raw: &serde_json::Value) -> bool {
    match raw.get("secondary_activation") {
        None | Some(serde_json::Value::Null) => true,
        Some(arr) => match arr.as_array() {
            Some(a) => a.iter().all(|v| v.is_null()),
            None => false,
        },
    }
}

/// Checks that `head1x1` is absent, null, not an object, or active==false.
fn check_head1x1_inactive(raw: &serde_json::Value) -> bool {
    match raw.get("head1x1") {
        None | Some(serde_json::Value::Null) => true,
        Some(v) => match v.as_object() {
            Some(obj) => obj.get("active").and_then(|a| a.as_bool()) != Some(true),
            None => true,
        },
    }
}

/// Checks that `layer1x1` is present as an object, active==true, groups==1.
fn check_layer1x1_active_groups1(raw: &serde_json::Value) -> bool {
    let obj = match raw.get("layer1x1").and_then(|v| v.as_object()) {
        Some(o) => o,
        None => return false,
    };
    if obj.get("active").and_then(|v| v.as_bool()) != Some(true) {
        return false;
    }
    obj.get("groups").and_then(|g| g.as_u64()) == Some(1)
}

/// Checks the layer-array head: out_channels==1, kernel_size==16, bias==true.
fn check_layer_array_head(raw: &serde_json::Value) -> bool {
    use crate::models::a2::A2_HEAD_KERNEL_SIZE;
    let obj = match raw.get("head").and_then(|v| v.as_object()) {
        Some(o) => o,
        None => return false,
    };
    if obj.get("out_channels").and_then(|c| c.as_u64()) != Some(1) {
        return false;
    }
    if obj.get("kernel_size").and_then(|k| k.as_u64()) != Some(A2_HEAD_KERNEL_SIZE as u64) {
        return false;
    }
    obj.get("bias").and_then(|b| b.as_bool()) == Some(true)
}

/// Checks that all 8 FiLM keys are inactive (absent, null, not an object, or active==false).
fn check_film_all_inactive(raw: &serde_json::Value) -> bool {
    const FILM_KEYS: &[&str] = &[
        "conv_pre_film",
        "conv_post_film",
        "input_mixin_pre_film",
        "input_mixin_post_film",
        "activation_pre_film",
        "activation_post_film",
        "layer1x1_post_film",
        "head1x1_post_film",
    ];
    for key in FILM_KEYS {
        match raw.get(key) {
            None | Some(serde_json::Value::Null) => continue,
            Some(v) => match v.as_object() {
                Some(obj) => {
                    if obj.get("active").and_then(|a| a.as_bool()) == Some(true) {
                        return false;
                    }
                }
                None => continue,
            },
        }
    }
    true
}

/// Checks that `groups_input == 1` AND `groups_input_mixin == 1`.
fn check_groups_are_1(raw: &serde_json::Value) -> bool {
    let gi = raw
        .get("groups_input")
        .and_then(|v| v.as_u64())
        .unwrap_or(1);
    let gim = raw
        .get("groups_input_mixin")
        .and_then(|v| v.as_u64())
        .unwrap_or(1);
    gi == 1 && gim == 1
}

/// Checks that `slimmable` is absent or null.
fn check_not_slimmable(raw: &serde_json::Value) -> bool {
    matches!(raw.get("slimmable"), None | Some(serde_json::Value::Null))
}

#[cfg(test)]
#[path = "a2_test.rs"]
mod tests;
