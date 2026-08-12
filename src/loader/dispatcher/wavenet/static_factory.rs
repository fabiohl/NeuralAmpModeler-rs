// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Static factory helpers for WaveNet model construction (A1 and A2).
//!
//! Extracted from `mod.rs` to keep the dispatcher entry-point lean (~dispatch
//! only). Contains validation fns, A2 static/dynamic builders, and the A1
//! topology match.

use crate::loader::nam_json::{
    A2TopologyResult, NamModelData, WavenetTopologyResult, get_wavenet_topology,
};
use crate::math::common::AlignedVec;
use crate::models::StaticModel;
use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::GatingMode;
use crate::models::a2::params::{A2_DILATIONS, A2_KERNEL_SIZES, A2_LEAKY_SLOPE};
use crate::models::a2::weights_layout::FILM_KEYS;
use crate::models::a2::{WaveNetA2, WaveNetA2Cascade, WaveNetA2Dyn};
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;
use anyhow::bail;
use log::info;

// =============================================================================
// Validation
// =============================================================================

/// Validates the `activation` field in all layers of a WaveNet A1 model.
///
/// **Scope: A1 topologies only** (Standard, Lite, Feather, Nano). A2 models use
/// `LeakyReLU` (not `Tanh`) and are dispatched by `is_a2_shape` before this branch
/// is ever reached — so this function is never called for A2.
///
/// Called from `build_wavenet_typed` (which backs all A1 builders).
/// Returns an error if any layer declares an unsupported activation function.
pub(crate) fn validate_layer_activations(data: &NamModelData) -> anyhow::Result<()> {
    for (idx, layer) in data.config.layers.iter().enumerate() {
        let act = layer.activation.as_deref().unwrap_or("Tanh");
        if act != "Tanh" && act != "ReLU" {
            bail!(
                "Activation '{}' in layer {} is not supported. Only 'Tanh' and 'ReLU' are implemented.",
                act,
                idx
            );
        }
    }
    Ok(())
}

/// Rejects WaveNet models whose `condition_dsp` sub-model is an LSTM.
///
/// **Fail-closed stance:** LSTM `condition_dsp` models produce structurally
/// wrong audio (ESR ≈ 1.3e-1, -8.9 dB — plainly audible). The root cause is a
/// state-update divergence between the production LSTM and the f64 oracle. The
/// upstream C++ NAMcore does not process these files correctly either
/// (parity-first). This is a known limitation, not yet implemented.
///
/// **Scope:** Applies to any LSTM architecture embedded as `condition_dsp` inside a
/// WaveNet model. Standalone LSTM models (`.nam` with `architecture: "LSTM"`) are
/// unaffected.
#[cold]
#[inline(never)]
pub(crate) fn reject_condition_dsp_lstm(cond_dsp_data: &NamModelData) -> anyhow::Result<()> {
    if cond_dsp_data.architecture.eq_ignore_ascii_case("LSTM") {
        bail!(
            "LSTM condition_dsp is not supported — the sub-model embedded in this \
             WaveNet model uses an LSTM architecture which produces structurally \
             incorrect audio (ESR ≈ 1.3e-1). Upstream NAMcore also does not \
             support this combination. Use a standalone WaveNet or LSTM model \
             instead."
        );
    }
    Ok(())
}

// =============================================================================
// A2 static fast-path builders
// =============================================================================

/// Builds a WaveNet A2-Lite model (channels=3, 23-layer canonical dilation).
fn build_wavenet_a2_lite(
    data: &NamModelData,
    layer_raw: Option<serde_json::Value>,
) -> anyhow::Result<Box<StaticModel>> {
    let mut model = WaveNetA2::<3>::new()?;
    model.set_layer_raw(layer_raw);
    model
        .set_weights(&data.weights)
        .map_err(|e| anyhow::anyhow!("A2-Lite weight load failed: {e}"))?;
    info!(
        "[Dispatcher] WaveNet A2-Lite built — CH=3, layers=23, weights={}",
        data.weights.len()
    );
    Ok(Box::new(StaticModel::WavenetA2Lite(Box::new(model))))
}

/// Builds a WaveNet A2-Full model (channels=8, 23-layer canonical dilation).
fn build_wavenet_a2_full(
    data: &NamModelData,
    layer_raw: Option<serde_json::Value>,
) -> anyhow::Result<Box<StaticModel>> {
    let mut model = WaveNetA2::<8>::new()?;
    model.set_layer_raw(layer_raw);
    model
        .set_weights(&data.weights)
        .map_err(|e| anyhow::anyhow!("A2-Full weight load failed: {e}"))?;
    info!(
        "[Dispatcher] WaveNet A2-Full built — CH=8, layers=23, weights={}",
        data.weights.len()
    );
    Ok(Box::new(StaticModel::WavenetA2Full(Box::new(model))))
}

// =============================================================================
// A2 entry-point: static + dynamic
// =============================================================================

/// Builds any WaveNet A2 model (static fast-path or dynamic/cascade).
///
/// Called from [`super::build_wavenet`] after `is_a2_shape` confirms an A2
/// topology.
pub(crate) fn build_wavenet_a2(
    data: &NamModelData,
    topo: A2TopologyResult,
) -> anyhow::Result<Box<StaticModel>> {
    let layer_raw = data.config.layers.first().and_then(|l| l.layer_raw.clone());
    match topo {
        A2TopologyResult::KnownFastPath(3) => build_wavenet_a2_lite(data, layer_raw),
        A2TopologyResult::KnownFastPath(8) => build_wavenet_a2_full(data, layer_raw),
        A2TopologyResult::KnownFastPath(unexpected) => bail!(
            "Unexpected A2 channels in KnownFastPath: {}. Only 3 (Lite) and 8 (Full) are supported.",
            unexpected
        ),
        A2TopologyResult::Dynamic => build_wavenet_a2_dynamic(data),
    }
}

// =============================================================================
// A2 Max flagship fail-closed guard
// =============================================================================

/// Rejects WaveNet A2 models matching the `wavenet_a2_max.nam` structural class.
///
/// **Known bug (permanent until reopening criteria):** This class
/// (`condition_size >= 2` + FiLM active + `head1x1` groups > 1 +
/// `groups_input_mixin > 1` + nested `condition_dsp`) has a **structural**
/// production×NAMCore parity gap. Measured prod f32 × C++ golden:
/// **SNR ≈ 0.23 dB** (ESR ≈ 9.49e-1). The f64 oracle also diverges from C++ on
/// this fixture (H0 Case D) and must not adjudicate. Fail-closed: no public
/// `Ok` path while the gap is open. Diagnostic unlock:
/// `NAM_A2_MAX_UNLOCK=1` under `cfg(test)` / feature `testing` only.
///
/// See `docs/cpp_parity_map.md` §4.4 / §4.4.3.
///
/// **Scope:** A2 Dynamic models only.
#[cold]
#[inline(never)]
fn reject_wavenet_a2_max_class(data: &NamModelData) -> anyhow::Result<()> {
    let has_cond_dsp = data.config.condition_dsp.is_some();
    if !has_cond_dsp {
        return Ok(());
    }

    let mut has_film = false;
    let mut has_head1x1_groups = false;
    let mut has_mixin_groups = false;
    let mut cond_size_ge_2 = false;

    for l in &data.config.layers {
        if l.condition_size.unwrap_or(0) >= 2 {
            cond_size_ge_2 = true;
        }
        let Some(ref raw) = l.layer_raw else { continue };

        for &(key, _) in FILM_KEYS {
            if raw
                .get(key)
                .and_then(|v| v.get("active"))
                .and_then(|a| a.as_bool())
                .unwrap_or(false)
            {
                has_film = true;
                break;
            }
        }

        if raw
            .get("head1x1")
            .and_then(|h| h.get("active"))
            .and_then(|a| a.as_bool())
            .unwrap_or(false)
            && raw
                .get("head1x1")
                .and_then(|h| h.get("groups"))
                .and_then(|g| g.as_u64())
                .unwrap_or(1)
                > 1
        {
            has_head1x1_groups = true;
        }

        if raw
            .get("groups_input_mixin")
            .and_then(|g| g.as_u64())
            .unwrap_or(1)
            > 1
        {
            has_mixin_groups = true;
        }
    }

    if cond_size_ge_2 && has_film && has_head1x1_groups && has_mixin_groups {
        #[cfg(any(test, feature = "testing"))]
        {
            if std::env::var("NAM_A2_MAX_UNLOCK").as_deref() == Ok("1") {
                return Ok(());
            }
        }
        bail!(
            "A2 Max flagship topology is not supported (known bug KB-A2-MAX) — \
             production f32 diverges from the NAMCore C++ golden (measured \
             SNR ≈ 0.23 dB; structural parity gap). fail-closed until a future \
             investigation meets the reopening criteria in docs/cpp_parity_map.md \
             §4.4.3. Neighbors (A2 Full/Lite/FiLM, condition_dsp standalone) remain supported."
        );
    }
    Ok(())
}

// =============================================================================
// A2 dynamic / cascade builder
// =============================================================================

fn build_wavenet_a2_dynamic(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    reject_wavenet_a2_max_class(data)?;
    use crate::loader::nam_json::validation::{MAX_A2_DYN_BOTTLENECK, MAX_A2_DYN_CHANNELS};

    let num_arrays = data.config.layers.len();
    let total_weights = data.weights.len();
    let mut weight_pos: usize = 0;

    // Validate all arrays.
    for (ai, layer_cfg) in data.config.layers.iter().enumerate() {
        let ch = layer_cfg.channels.unwrap_or(0);
        let bn = layer_cfg.bottleneck.unwrap_or(ch);
        if ch > MAX_A2_DYN_CHANNELS {
            bail!(
                "A2-Dynamic array[{}] channels ({}) exceeds maximum {} (OOM/DoS protection)",
                ai,
                ch,
                MAX_A2_DYN_CHANNELS
            );
        }
        if bn > MAX_A2_DYN_BOTTLENECK {
            bail!(
                "A2-Dynamic array[{}] bottleneck ({}) exceeds maximum {} (OOM/DoS protection)",
                ai,
                bn,
                MAX_A2_DYN_BOTTLENECK
            );
        }
    }

    let l0 = &data.config.layers[0];
    let mut arrays: Vec<WaveNetA2Dyn> = Vec::with_capacity(num_arrays);

    for (ai, layer_cfg) in data.config.layers.iter().enumerate() {
        let channels = layer_cfg.channels.unwrap_or(0);
        let bottleneck = layer_cfg.bottleneck.unwrap_or(channels);
        let head_size = layer_cfg.head_size.unwrap_or(1);
        let input_channels = if ai == 0 {
            layer_cfg.input_size.unwrap_or(1)
        } else {
            data.config.layers[ai - 1].channels.unwrap_or(1)
        };

        let kernel_sizes = if let Some(ks) = layer_cfg.kernel_sizes.clone() {
            ks
        } else if let Some(ks_scalar) = layer_cfg.kernel_size {
            let dils = layer_cfg
                .dilations
                .clone()
                .unwrap_or_else(|| A2_DILATIONS.to_vec());
            vec![ks_scalar; dils.len()]
        } else {
            A2_KERNEL_SIZES.to_vec()
        };
        let dilations = layer_cfg
            .dilations
            .clone()
            .unwrap_or_else(|| A2_DILATIONS.to_vec());
        let num_layers = kernel_sizes.len();

        let act_cfg = layer_cfg
            .parse_activation_config(num_layers)
            .map(|cfg| (cfg.activations, cfg.gating_modes, cfg.secondary_activations));
        let (activations, gating_modes, secondary_activations) = match act_cfg {
            Some((a, g, s)) => (a, g, s),
            None => (
                vec![
                    ActivationType::LeakyReLU {
                        negative_slope: A2_LEAKY_SLOPE,
                    };
                    num_layers
                ],
                vec![GatingMode::None; num_layers],
                vec![None; num_layers],
            ),
        };

        let head1x1_active = layer_cfg
            .layer_raw
            .as_ref()
            .and_then(|raw| raw.get("head1x1"))
            .and_then(|h| h.get("active"))
            .and_then(|a| a.as_bool())
            .unwrap_or(false);

        let condition_size = layer_cfg.condition_size.unwrap_or(1);

        let head1x1_out_channels = layer_cfg
            .layer_raw
            .as_ref()
            .and_then(|raw| raw.get("head1x1"))
            .and_then(|h| h.get("out_channels"))
            .and_then(|a| a.as_u64())
            .unwrap_or(bottleneck as u64) as usize;
        let head_accum_size = if head1x1_active {
            head1x1_out_channels
        } else {
            bottleneck
        };
        let h1_groups = layer_cfg
            .layer_raw
            .as_ref()
            .and_then(|raw| raw.get("head1x1"))
            .and_then(|h| h.get("groups"))
            .and_then(|g| g.as_u64())
            .unwrap_or(1) as usize;
        let h1_in_size = if head1x1_active {
            bottleneck / h1_groups
        } else {
            bottleneck
        };

        // Head kernel size — legacy models use `head_size`/`head_bias` format
        // with implicit kernel=1; new models use `head.kernel_size`.
        let head_kernel_size = layer_cfg
            .layer_raw
            .as_ref()
            .and_then(|raw| raw.get("head"))
            .and_then(|h| h.get("kernel_size"))
            .and_then(|k| k.as_u64())
            .unwrap_or(1) as usize;

        let mut model = WaveNetA2Dyn::new(
            input_channels,
            channels,
            bottleneck,
            head_size,
            head_accum_size,
            h1_in_size,
            head_kernel_size,
            &kernel_sizes,
            &dilations,
            activations,
            gating_modes,
            secondary_activations,
        )?;
        model.head1x1_active = head1x1_active;
        model.head1x1_h1_in = h1_in_size;

        // Group configs (per-array, uniform across layers inside the array).
        model.mixin_groups = layer_cfg
            .layer_raw
            .as_ref()
            .and_then(|raw| raw.get("groups_input_mixin"))
            .and_then(|g| g.as_u64())
            .unwrap_or(1) as u32;
        model.l1x1_groups = layer_cfg
            .layer_raw
            .as_ref()
            .and_then(|raw| raw.get("layer1x1"))
            .and_then(|l| l.get("groups"))
            .and_then(|g| g.as_u64())
            .unwrap_or(1) as u32;

        model.set_layer_raw(layer_cfg.layer_raw.clone());
        model.condition_size = condition_size;
        model.cond_scratch = AlignedVec::new(condition_size, 0.0f32)
            .expect("cond_scratch allocation should succeed for test-sized condition size");

        model
            .load_weights_inner(&data.weights, &mut weight_pos, total_weights)
            .map_err(|e| anyhow::anyhow!("A2-Dynamic array[{ai}] weight load failed: {e}"))?;

        arrays.push(model);
    }

    if weight_pos != total_weights {
        info!(
            "[Dispatcher] A2-Dynamic: {} unconsumed weights after loading {} arrays (consumed {}, total {}). \
             Allowed for hybrid condition_dsp sub-models.",
            total_weights - weight_pos,
            num_arrays,
            weight_pos,
            total_weights
        );
    }

    let condition_size = l0.condition_size.unwrap_or(1);

    // Build condition_dsp sub-model if present.
    let condition_dsp = if let Some(ref cond_dsp_json) = data.config.condition_dsp {
        let cond_dsp_data: NamModelData = serde_json::from_value(cond_dsp_json.clone())?;
        reject_condition_dsp_lstm(&cond_dsp_data)?;
        let cond_model = crate::loader::dispatcher::build_model(&cond_dsp_data)?;
        let cond_out = cond_model.num_output_channels();
        if cond_out > condition_size {
            bail!(
                "condition_dsp output channels ({}) exceed A2 condition_size ({})",
                cond_out,
                condition_size
            );
        }
        info!(
            "[Dispatcher] A2-Dynamic condition_dsp built — architecture={}, output_channels={}",
            cond_dsp_data.architecture, cond_out
        );
        Some(cond_model)
    } else {
        None
    };

    if num_arrays == 1 {
        let mut model = arrays.remove(0);
        model.set_condition_dsp(condition_dsp, WAVENET_MAX_NUM_FRAMES);
        info!(
            "[Dispatcher] WaveNet A2-Dynamic built — CH={}, BN={}, layers={}, weights={}",
            model.channels,
            model.bottleneck,
            model.num_layers,
            data.weights.len()
        );
        return Ok(Box::new(StaticModel::WavenetA2Dyn(Box::new(model))));
    }

    let cascade = WaveNetA2Cascade::new(arrays, condition_dsp, condition_size);
    info!(
        "[Dispatcher] WaveNet A2-Cascade built — {} arrays, CH=[{}], weights={}",
        num_arrays,
        cascade
            .arrays
            .iter()
            .map(|a| a.channels.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        data.weights.len()
    );
    Ok(Box::new(StaticModel::WavenetA2Cascade(Box::new(cascade))))
}

// =============================================================================
// A1 topology builders
// =============================================================================

/// Detects the WaveNet A1 topology and branches to the correct submodule
/// builder.
///
/// Handles Standard, Lite, Feather, Nano, and Free/Dynamic topologies.
/// Called from [`super::build_wavenet`] when the model is not A2.
pub(crate) fn build_wavenet_a1(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    let topo = get_wavenet_topology(data);

    match topo {
        WavenetTopologyResult::Known(crate::loader::nam_json::NamWavenetTopology::Standard) => {
            let model = super::standard::build_wavenet_standard(data)?;
            Ok(Box::new(StaticModel::WavenetStandard(Box::new(model))))
        }
        WavenetTopologyResult::Known(crate::loader::nam_json::NamWavenetTopology::Lite) => {
            let model = super::lite::build_wavenet_lite(data)?;
            Ok(Box::new(StaticModel::WavenetLite(Box::new(model))))
        }
        WavenetTopologyResult::Known(crate::loader::nam_json::NamWavenetTopology::Feather) => {
            let model = super::feather::build_wavenet_feather(data)?;
            Ok(Box::new(StaticModel::WavenetFeather(Box::new(model))))
        }
        WavenetTopologyResult::Known(crate::loader::nam_json::NamWavenetTopology::Nano) => {
            let model = super::nano::build_wavenet_nano(data)?;
            Ok(Box::new(StaticModel::WavenetNano(Box::new(model))))
        }
        WavenetTopologyResult::Free(ref geom) => {
            let model = super::dynamic::build_wavenet_dynamic(data, geom)?;
            Ok(Box::new(StaticModel::WavenetDyn(Box::new(model))))
        }
        WavenetTopologyResult::Rejected(reason) => {
            let layer_info: Vec<(usize, usize)> = data
                .config
                .layers
                .iter()
                .map(|l| {
                    let ch = l.channels.unwrap_or(0);
                    let k = l.kernel_size.unwrap_or(0);
                    (ch, k)
                })
                .collect();
            bail!(
                "WaveNet model rejected: {reason}. \
                 Detected: {} layer(s) with geometry {layer_info:?}",
                data.config.layers.len(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::dispatcher::build_model;
    use crate::loader::nam_json::parse_nam_json;
    use std::fs;

    const A2_MAX_FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/models/wavenet_a2_max.nam"
    );

    #[test]
    fn test_a2_max_flag_controlled() {
        let json =
            fs::read_to_string(A2_MAX_FIXTURE).expect("Fixture wavenet_a2_max.nam not found");
        let data = parse_nam_json(&json).expect("Failed to parse fixture");

        unsafe {
            std::env::remove_var("NAM_A2_MAX_UNLOCK");
        }
        let result = build_model(&data);
        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("KB-A2-MAX") || msg.contains("parity gap"),
                    "Error must cite KB-A2-MAX / parity gap, got: {msg}"
                );
                assert!(
                    msg.contains("fail-closed"),
                    "Error must cite fail-closed, got: {msg}"
                );
            }
            Ok(_) => panic!("A2 Max must be rejected by default (no unlock flag set)"),
        }

        unsafe {
            std::env::set_var("NAM_A2_MAX_UNLOCK", "1");
        }
        let model = build_model(&data).expect("A2 Max must build under NAM_A2_MAX_UNLOCK=1");
        assert!(
            matches!(*model, StaticModel::WavenetA2Dyn(_)),
            "Expected WavenetA2Dyn variant under unlock"
        );

        unsafe {
            std::env::remove_var("NAM_A2_MAX_UNLOCK");
        }
    }
}
