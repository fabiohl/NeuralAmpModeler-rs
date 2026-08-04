// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of WaveNet topologies from model data.

use super::super::data::NamModelData;
use super::super::model::HeadConfig;

/// The closed and supported topologies within native WaveNet modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamWavenetTopology {
    /// Channels: 16 (Standard)
    Standard,
    /// Channels: 12 (Lite)
    Lite,
    /// Channels: 8 (Feather)
    Feather,
    /// Channels: 4 (Nano)
    Nano,
}

/// Description of a detected free (non-catalog) WaveNet A1 geometry.
///
/// Returned by [`get_wavenet_topology`] when the model is a valid WaveNet A1
/// but does not match any of the four catalog SKUs
/// (Standard/Lite/Feather/Nano). The dynamic engine consumes this
/// structure to build a runtime-dimensioned [`WaveNetModelDyn`].
///
/// ## Semver note
///
/// The `channels` and `head_sizes` fields changed from `usize` to `Vec<usize>`
/// to support per-array geometry (required for condition_dsp). This is a
/// breaking change from the v2.x API.
///
/// [`WaveNetModelDyn`]: crate::models::wavenet::WaveNetModelDyn
#[derive(Debug, Clone, PartialEq)]
pub struct FreeWavenetGeometry {
    /// Internal channels per layer-array (first array → `array[0]`).
    pub channels: Vec<usize>,
    /// Kernel size (same across all layers for legacy models;
    /// per-array kernel sizes for heterogeneous topologies).
    pub kernel_size: usize,
    /// Per-array kernel sizes, one per layer-array.
    /// For homogeneous topologies this is a repetition of `kernel_size`.
    pub kernel_sizes: Vec<usize>,
    /// Head projection size per layer-array.
    /// `head_sizes.last()` determines `NumOutputChannels()` for the model
    /// (C++ `wave_net_output_channels` returns `layer_array_params.back().head_size`).
    pub head_sizes: Vec<usize>,
    /// Dilations per layer-array.
    pub dilations: Vec<Vec<usize>>,
    /// Number of layer-arrays.
    pub num_arrays: usize,
    /// Condition input size (number of conditioning channels).
    /// Models with `condition_size > 1` flow into the dynamic engine
    /// (the const-generic fast-path is reserved for `COND=1`).
    pub condition_size: usize,
    /// Post-stack head configuration (Conv1D + activation) that processes
    /// the signal after all layer arrays, before `head_scale` and output.
    /// `None` means no post-stack head is present (standard behavior).
    pub post_stack_head: Option<HeadConfig>,
}

/// Result of WaveNet topology detection.
///
/// Distinguishes three states:
/// - `Known`: fast-path const-generic SKU (Standard/Lite/Feather/Nano).
/// - `Free`: valid geometry outside the catalog — destined for the dynamic engine.
/// - `Rejected`: invalid or missing shape data (no channels, no dilations, etc.).
#[derive(Debug, Clone, PartialEq)]
pub enum WavenetTopologyResult {
    /// Matches a known catalog SKU — use const-generic fast-path.
    Known(NamWavenetTopology),
    /// Valid geometry not in the catalog — use dynamic engine.
    Free(FreeWavenetGeometry),
    /// Rejected due to unsupported feature or invalid shape.
    Rejected(String),
}

static STD_DILATIONS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
static LITE_DILATIONS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
static LITE_DILATIONS_2: &[usize] = &[128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

/// Parses a version string in minimal SemVer format.
/// Supports pre-release or metadata suffixes and 'v' or 'V' prefix.
/// Returns `Some((major, minor, patch))` or `None` on parsing failure.
pub(crate) fn parse_semver(version: &str) -> Option<(u16, u16, u16)> {
    let clean = version.trim().trim_start_matches(['v', 'V']);
    let clean = clean.split('-').next()?.split('+').next()?;
    let mut parts = clean.split('.');
    let major = parts.next()?.trim().parse::<u16>().ok()?;
    let minor = parts.next().unwrap_or("0").trim().parse::<u16>().ok()?;
    let patch = parts.next().unwrap_or("0").trim().parse::<u16>().ok()?;
    Some((major, minor, patch))
}

impl NamModelData {
    /// Detects whether the model uses the WaveNet A2 architecture.
    ///
    /// A2 detection is layered:
    /// 1. **Primary — shape-based:** `is_a2_shape()` checks channels, dilations,
    ///    and single-layer structure. If shape matches, the model is A2.
    /// 2. **Secondary — activation:** any non-Tanh activation implies A2 semantics.
    /// 3. **Tertiary — version telemetry:** a SemVer >= 0.6.0 is logged as a
    ///    warning when the shape is not recognised, but version alone is NOT
    ///    sufficient to classify the model as A2. This prevents a WaveNet A1
    ///    model with a high version string from being silently misrouted.
    pub fn is_wavenet_a2(&self) -> bool {
        if self.architecture != "WaveNet" {
            return false;
        }

        // Primary: shape-based detection (channels + dilations + kernel sizes)
        if super::a2::is_a2_shape(self).is_some() {
            return true;
        }

        // Secondary: activation-based detection — only if the model has the
        // structural signature of an A2 (single layer array). Multi-array
        // WaveNet models with non-Tanh activation are A1 geometries with
        // a non-standard activation, not A2.
        if self.config.layers.len() == 1
            && let Some(layer) = self.config.layers.first()
            && layer.activation.as_deref().is_some_and(|a| a != "Tanh")
        {
            return true;
        }

        // Tertiary: version telemetry (warning only, does NOT classify)
        if let Some(ref v) = self.version
            && let Some(ver) = parse_semver(v)
            && ver >= (0, 6, 0)
        {
            log::warn!(
                "WaveNet model declares version {v} (>= 0.6.0), but its shape \
                 does not match any known A2 topology. Treating as non-A2. \
                 Channels/dilations may indicate an unsupported A2 variant \
                 or an A1 model with an unusually high version string."
            );
        }

        false
    }
}

/// Detects the WaveNet topology, distinguishing known SKUs from free geometries.
///
/// Returns [`WavenetTopologyResult`] with three possible outcomes:
/// - `Known(SKU)`: matches a catalog variant (Standard/Lite/Feather/Nano) —
///   use the const-generic fast-path. Only `condition_size=1` (or absent) models
///   can match a catalog SKU.
/// - `Free(geometry)`: valid A1 geometry outside the catalog — destined for the
///   dynamic engine. Any `condition_size` is accepted.
/// - `Rejected(reason)`: missing/invalid shape data.
///
/// Mirror of C++ `NeuralModel.cpp` (L:155-218) generalized to accept any valid
/// WaveNet A1 geometry, not only the four catalog SKUs.
pub fn get_wavenet_topology(data: &NamModelData) -> WavenetTopologyResult {
    use super::super::validation::{
        MAX_DILATION, MAX_DILATIONS_PER_ARRAY, MAX_HEAD_SIZE, MAX_KERNEL_SIZE,
        MAX_TOTAL_STATE_FRAMES, MAX_WAVENET_ARRAYS, MAX_WAVENET_FREE_CHANNELS,
    };

    // ── Architecture gate ──
    if data.architecture != "WaveNet" {
        return WavenetTopologyResult::Rejected("Not a WaveNet model.".to_string());
    }

    let layers = &data.config.layers;
    if layers.is_empty() {
        return WavenetTopologyResult::Rejected("WaveNet model has no layer arrays.".to_string());
    }

    if layers.len() > MAX_WAVENET_ARRAYS {
        return WavenetTopologyResult::Rejected(format!(
            "WaveNet model has {} layer arrays, exceeding maximum {} — \
             DoS/OOM protection.",
            layers.len(),
            MAX_WAVENET_ARRAYS
        ));
    }

    // ── Extract condition_size from first layer (all layers share the same value) ──
    let condition_size = layers[0].condition_size.unwrap_or(1);

    // ── Extract per-layer fields (gently — full validation deferred to free geometry) ──
    let mut first_channels: Option<usize> = None;
    let mut first_kernel_size: Option<usize> = None;
    let mut first_head_size: Option<usize> = None;
    let mut dilations: Vec<Vec<usize>> = Vec::with_capacity(layers.len());
    let mut head_sizes: Vec<usize> = Vec::with_capacity(layers.len());
    let mut channels: Vec<usize> = Vec::with_capacity(layers.len());
    let mut kernel_sizes: Vec<usize> = Vec::with_capacity(layers.len());

    for (i, layer) in layers.iter().enumerate() {
        if let Some(ref raw) = layer.layer_raw
            && raw
                .as_object()
                .is_some_and(|obj| obj.get("slimmable").is_some_and(|v| !v.is_null()))
        {
            return WavenetTopologyResult::Rejected(
                "slimmable single-net weight slicing is not supported; use SlimmableContainer instead"
                    .to_string(),
            );
        }
        let ch = match layer.channels {
            Some(c) if c > 0 => {
                if c > MAX_WAVENET_FREE_CHANNELS {
                    return WavenetTopologyResult::Rejected(format!(
                        "Layer {} channels ({}) exceeds maximum {} — \
                         OOM/DoS protection.",
                        i, c, MAX_WAVENET_FREE_CHANNELS
                    ));
                }
                c
            }
            _ => {
                return WavenetTopologyResult::Rejected(format!(
                    "Layer {} is missing or has invalid 'channels'.",
                    i
                ));
            }
        };
        let k = layer.kernel_size.filter(|&k| k > 0);
        if let Some(k) = k
            && k > MAX_KERNEL_SIZE
        {
            return WavenetTopologyResult::Rejected(format!(
                "Layer {} kernel_size ({}) exceeds maximum {} — \
                 DoS/OOM protection.",
                i, k, MAX_KERNEL_SIZE
            ));
        }
        let dils = match layer.dilations.as_deref() {
            Some(d) if !d.is_empty() => d.to_vec(),
            _ => {
                return WavenetTopologyResult::Rejected(format!(
                    "Layer {} is missing or has invalid 'dilations'.",
                    i
                ));
            }
        };
        if dils.len() > MAX_DILATIONS_PER_ARRAY {
            return WavenetTopologyResult::Rejected(format!(
                "Layer {} has {} dilations, exceeding maximum {} — \
                 DoS/OOM protection.",
                i,
                dils.len(),
                MAX_DILATIONS_PER_ARRAY
            ));
        }
        for (j, &d) in dils.iter().enumerate() {
            if d > MAX_DILATION {
                return WavenetTopologyResult::Rejected(format!(
                    "Layer {} dilation[{}] ({}) exceeds maximum {} — \
                     DoS/OOM protection.",
                    i, j, d, MAX_DILATION
                ));
            }
        }

        if i == 0 {
            first_channels = Some(ch);
            first_kernel_size = k;
            first_head_size = layer.head_size;
        }

        let hd = layer.head_size.unwrap_or(1);
        if hd == 0 {
            return WavenetTopologyResult::Rejected(format!(
                "Layer {} has invalid head_size=0.",
                i
            ));
        }
        if hd > MAX_HEAD_SIZE {
            return WavenetTopologyResult::Rejected(format!(
                "Layer {} head_size ({}) exceeds maximum {} — \
                 DoS/OOM protection.",
                i, hd, MAX_HEAD_SIZE
            ));
        }

        channels.push(ch);
        kernel_sizes.push(k.unwrap_or(0));
        head_sizes.push(hd);
        dilations.push(dils);
    }

    // ── Aggregate state budget check ──
    // Computes total receptive-field × channels across all layers.
    // Prevents DoS via amplification: a tiny model file (few KB of weights
    // with channels=1) can trigger ~1 GB of mirrored buffer allocation
    // when kernel_size × dilation × layer_count is extreme (F12).
    let total_state_frames: usize = dilations
        .iter()
        .zip(kernel_sizes.iter())
        .zip(channels.iter())
        .try_fold(0usize, |acc, ((dils, &k), &ch)| {
            let rf = k.saturating_sub(1);
            dils.iter().try_fold(acc, |a, &d| {
                a.checked_add(rf.saturating_mul(d).saturating_mul(ch))
            })
        })
        .unwrap_or(usize::MAX);
    if total_state_frames > MAX_TOTAL_STATE_FRAMES {
        return WavenetTopologyResult::Rejected(format!(
            "Aggregate state budget exceeded: {} total state frames vs max {} — \
             OOM/DoS protection (F12).",
            total_state_frames, MAX_TOTAL_STATE_FRAMES
        ));
    }

    let first_channels_val = match first_channels {
        Some(c) => c,
        None => {
            return WavenetTopologyResult::Rejected(
                "Layer 0 is missing or has invalid 'channels' — required for \
                 WaveNet topology detection."
                    .to_string(),
            );
        }
    };

    // ── Guardrail: Reject A1 models with A2-specific features ──
    // Models with gating_mode, head1x1, layer1x1, or FiLM features carry
    // A2 semantics and cannot be processed by WaveNet A1 (catalog or free).
    // (Finding 7.2.2 / Task T2.1; review: extended to cover catalog SKU path).
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
                    return WavenetTopologyResult::Rejected(format!(
                        "Layer {i} has non-none gating_mode — A2 feature not supported in WaveNet A1."
                    ));
                }
            } else if !gm.is_null() {
                return WavenetTopologyResult::Rejected(format!(
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
            return WavenetTopologyResult::Rejected(format!(
                "Layer {i} has active head1x1 — A2 feature not supported in WaveNet A1."
            ));
        }
        if raw
            .get("layer1x1")
            .and_then(|l| l.get("active"))
            .and_then(|a| a.as_bool())
            .unwrap_or(false)
        {
            return WavenetTopologyResult::Rejected(format!(
                "Layer {i} has active layer1x1 — A2 feature not supported in WaveNet A1."
            ));
        }
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
            if raw
                .get(key)
                .and_then(|f| f.get("active"))
                .and_then(|a| a.as_bool())
                .unwrap_or(false)
            {
                return WavenetTopologyResult::Rejected(format!(
                    "Layer {i} has active {key} — A2 feature not supported in WaveNet A1."
                ));
            }
        }
        if layer.gated.unwrap_or(false) {
            return WavenetTopologyResult::Rejected(format!(
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
                return WavenetTopologyResult::Rejected(format!(
                    "Layer {i} has non-trivial secondary_activation — A2 feature not supported in WaveNet A1."
                ));
            }
        }
    }

    // ── Try matching a known catalog SKU (fast-path) ──
    if layers.len() == 2 {
        let l0 = &layers[0];
        let l1 = &layers[1];

        let l0_gated = l0.gated.unwrap_or(false);
        let l1_gated = l1.gated.unwrap_or(false);
        let l0_head_bias = l0.head_bias.unwrap_or(false);
        let l1_head_bias = l1.head_bias.unwrap_or(false);

        let catalog_compatible = !l0_gated
            && !l1_gated
            && !l0_head_bias
            && l1_head_bias
            && condition_size <= 1
            && data.config.condition_dsp.is_none();

        if catalog_compatible {
            let dils_0 = &dilations[0];
            let dils_1 = &dilations[1];

            let result = match first_channels_val {
                16 if dils_0 == STD_DILATIONS && dils_1 == STD_DILATIONS => {
                    Some(NamWavenetTopology::Standard)
                }
                12 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
                    Some(NamWavenetTopology::Lite)
                }
                8 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
                    Some(NamWavenetTopology::Feather)
                }
                4 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
                    Some(NamWavenetTopology::Nano)
                }
                _ => None,
            };

            if let Some(sku) = result {
                return WavenetTopologyResult::Known(sku);
            }
        }
    }

    // ── Free geometry (valid A1, but not in catalog) ──
    let kernel_size = match first_kernel_size {
        Some(k) => k,
        None => {
            return WavenetTopologyResult::Rejected(
                "Layer 0 is missing or has invalid 'kernel_size' — required for \
                 free geometry WaveNet A1."
                    .to_string(),
            );
        }
    };
    if first_head_size.is_none_or(|h| h == 0) {
        return WavenetTopologyResult::Rejected(
            "Layer 0 is missing or has invalid 'head_size' — required for \
             WaveNet A1 geometries (determines the head projection dimension)."
                .to_string(),
        );
    }

    WavenetTopologyResult::Free(FreeWavenetGeometry {
        channels,
        kernel_size,
        kernel_sizes,
        head_sizes,
        condition_size,
        num_arrays: layers.len(),
        dilations,
        post_stack_head: data.config.parse_head(),
    })
}
