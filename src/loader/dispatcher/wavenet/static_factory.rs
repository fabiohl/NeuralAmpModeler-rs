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
use crate::models::StaticModel;
use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::GatingMode;
use crate::models::a2::params::{A2_DILATIONS, A2_KERNEL_SIZES, A2_LEAKY_SLOPE};
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
        if act != "Tanh" {
            bail!(
                "Activation '{}' in layer {} is not supported. Only 'Tanh' is implemented.",
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

/// Predicate that identifies the known-broken WaveNet A2 flagship model
/// (`wavenet_a2_max.nam`) by structural signature.
///
/// The guard is **fail-closed**: it returns `true` only for models whose audio
/// output is confirmed wrong against the NAMcore C++ golden reference
/// (see [`docs/cpp_parity_map.md` §7.1][§7.1]). No `unsafe` code; the engine
/// and cascade code paths remain intact — this function merely prevents
/// construction at the dispatch layer.
///
/// **Detection signature** (narrow enough to preserve all other verified
/// A2 Dynamic fixtures):
///
/// - `num_arrays == 1` (single-array; multi-array cascade models are unaffected)
/// - `has_condition_dsp` (the model embeds a `condition_dsp` sub-model)
/// - `condition_size == 8` (the flagship's specific condition input width)
///
/// **Invariants:**
///
/// - No `unsafe` — pure structural check on parsed model metadata.
/// - Fail-closed: `true` → reject model before weights are touched.
/// - Engine code (`process.rs`, `cascade.rs`, `condition_dsp`) is never removed
///   or altered by this guard.
///
/// **Tradeoff (documented):** if a future model with correct output happens to
/// match this exact signature, it will be blocked. This is acceptable until the
/// `condition_dsp` parity gap against C++ is closed
/// ([`docs/cpp_parity_map.md` §4.4][§4.4]) — then the guard is removed or
/// narrowed.
///
/// [§7.1]: docs/cpp_parity_map.md
/// [§4.4]: docs/cpp_parity_map.md
#[cold]
#[inline(never)]
pub(crate) fn is_disabled_broken_a2_flagship(
    num_arrays: usize,
    condition_size: usize,
    has_condition_dsp: bool,
) -> bool {
    num_arrays == 1 && has_condition_dsp && condition_size == 8
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
// A2 dynamic / cascade builder
// =============================================================================

fn build_wavenet_a2_dynamic(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
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

    let condition_size = l0.condition_size.unwrap_or(1);
    let has_condition_dsp = data.config.condition_dsp.is_some();
    if is_disabled_broken_a2_flagship(num_arrays, condition_size, has_condition_dsp) {
        bail!(
            "WaveNet A2 flagship (single-array, condition_dsp, condition_size=8) is disabled: \
             confirmed wrong audio output vs NAMcore golden — see docs/cpp_parity_map.md §7.1. \
             Model is not removed; re-enable requires closing the condition_dsp parity gap (§4.4)."
        );
    }

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

        let mut model = WaveNetA2Dyn::new(
            input_channels,
            channels,
            bottleneck,
            head_size,
            head_accum_size,
            h1_in_size,
            &kernel_sizes,
            &dilations,
            activations,
            gating_modes,
            secondary_activations,
            head1x1_active,
        )?;
        model.set_layer_raw(layer_cfg.layer_raw.clone());
        model.condition_size = condition_size;

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
