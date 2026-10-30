// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of WaveNet topologies from model data.

pub mod a2_validation;
pub mod budget;
pub mod guardrail;
pub mod sku;
pub mod slimmable;

pub use a2_validation::{A2TopologyVectors, validate_a2_layer_topology};
pub use budget::compute_state_budget;
pub use guardrail::validate_a1_guardrail;
pub use sku::try_match_catalog_sku;
pub use slimmable::validate_slimmable_metadata;

use super::super::data::NamModelData;
use super::super::model::{HeadConfig, NamLayerConfig};
use super::super::validation::{
    MAX_CONDITION_SIZE, MAX_DILATION, MAX_DILATIONS_PER_ARRAY, MAX_HEAD_SIZE, MAX_KERNEL_SIZE,
    MAX_WAVENET_ARRAYS, MAX_WAVENET_FREE_CHANNELS,
};
use crate::common::diagnostics::NamErrorCode;

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
    /// Head convolution kernel size per layer-array (default 1).
    pub head_kernel_sizes: Vec<usize>,
    /// Head convolution dilation per layer-array (default 1).
    pub head_dilations: Vec<usize>,
    /// Per-layer-array head_bias flag. `head_biases.last()` determines whether
    /// the final head_rechannel has a bias term (C++ `LayerArrayParams::HasHeadBias`).
    pub head_biases: Vec<bool>,
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
    /// Allowed channel breakpoints for slimmable WaveNet models, one list per
    /// layer-array (C++ `SlimmableWavenet::_per_array_allowed_channels`).
    ///
    /// When `Some`, the outer `Vec` has one entry per layer-array; an empty
    /// inner list means that array is non-slimmable (keeps full channels).
    /// A non-empty inner list must be sorted ascending with `back() == channels`
    /// for that array. Missing `kwargs.allowed_channels` defaults to
    /// `[1..=channels]` (C++ `SlimmableWavenetConfig::create`).
    /// `None` means the model is not slimmable-capable.
    pub allowed_channels: Option<Vec<Vec<usize>>>,
}

impl FreeWavenetGeometry {
    /// Computes the total receptive field for the free WaveNet geometry.
    pub fn receptive_field(&self) -> usize {
        let mut rf = 0;
        for i in 0..self.num_arrays {
            let k = self
                .kernel_sizes
                .get(i)
                .copied()
                .unwrap_or(self.kernel_size);
            let head_k = self.head_kernel_sizes.get(i).copied().unwrap_or(1);
            let head_dil = self.head_dilations.get(i).copied().unwrap_or(1);
            let head_rf = (head_k.saturating_sub(1)).saturating_mul(head_dil);
            let layer_rf: usize = self
                .dilations
                .get(i)
                .map(|dils| {
                    dils.iter()
                        .map(|&d| (k.saturating_sub(1)) * d)
                        .sum::<usize>()
                })
                .unwrap_or(0);
            rf += layer_rf + head_rf;
        }
        if let Some(ref post) = self.post_stack_head {
            let post_k = post.kernel_size.unwrap_or(1);
            rf += post_k.saturating_sub(1);
        }
        rf
    }
}

/// Topology construction or validation error carrying a typed diagnostic error code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyError {
    /// Diagnostic error code.
    pub code: NamErrorCode,
    /// Detailed reason string.
    pub reason: String,
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{} {}] {}",
            self.code.code(),
            self.code.mnemonic(),
            self.reason
        )
    }
}

impl std::error::Error for TopologyError {}

/// The outcome of WaveNet topology detection (`get_wavenet_topology`).
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
    Free(Box<FreeWavenetGeometry>),
    /// Rejected due to unsupported feature or invalid shape.
    Rejected {
        /// Typed diagnostics error code.
        code: NamErrorCode,
        /// Rejection reason description.
        reason: String,
    },
}

impl WavenetTopologyResult {
    /// Helper to construct a `Rejected` variant with [`NamErrorCode::InvalidModelTopology`] (`E1305`).
    pub fn rejected_topology(reason: impl Into<String>) -> Self {
        Self::Rejected {
            code: NamErrorCode::InvalidModelTopology,
            reason: reason.into(),
        }
    }

    /// Helper to construct a `Rejected` variant with [`NamErrorCode::InvalidFeatureTopology`] (`E1316`).
    pub fn rejected_feature(reason: impl Into<String>) -> Self {
        Self::Rejected {
            code: NamErrorCode::InvalidFeatureTopology,
            reason: reason.into(),
        }
    }
}

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

        // Slimmable WaveNet models go through the A1 free-geometry path.
        if self.config.layers.iter().any(|l| l.slimmable.is_some()) {
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

/// Entry-point: detects the WaveNet topology, distinguishing known SKUs from free geometries.
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
    // ── Architecture gate ──
    if data.architecture != "WaveNet" {
        return WavenetTopologyResult::Rejected {
            code: NamErrorCode::UnsupportedArchitecture,
            reason: "Not a WaveNet model.".to_string(),
        };
    }

    let layers = &data.config.layers;
    if layers.is_empty() {
        return WavenetTopologyResult::rejected_topology("WaveNet model has no layer arrays.");
    }

    if layers.len() > MAX_WAVENET_ARRAYS {
        return WavenetTopologyResult::rejected_topology(format!(
            "WaveNet model has {} layer arrays, exceeding maximum {} — \
             DoS/OOM protection.",
            layers.len(),
            MAX_WAVENET_ARRAYS
        ));
    }

    let condition_size = layers[0].condition_size.unwrap_or(1);
    if condition_size == 0 {
        return WavenetTopologyResult::rejected_topology(
            "WaveNet condition_size must be >= 1 — zero conditioning channels is invalid.",
        );
    }
    // H-03: free-geometry (A1) `condition_size` has no compile-time cap —
    // reject oversized values before the dynamic engine allocates
    // conditioning buffers proportional to it.
    if condition_size > MAX_CONDITION_SIZE {
        return WavenetTopologyResult::rejected_topology(format!(
            "WaveNet A1 condition_size ({condition_size}) exceeds maximum \
             {MAX_CONDITION_SIZE} — DoS/OOM protection (H-03)."
        ));
    }

    for (i, layer) in layers.iter().enumerate() {
        if let Some(cs) = layer.condition_size
            && cs != condition_size
        {
            return WavenetTopologyResult::rejected_topology(format!(
                "WaveNet layer[{i}] condition_size ({cs}) does not match layer[0] ({condition_size})."
            ));
        }
    }

    let extract = match extract_layer_metadata(layers) {
        Ok(m) => m,
        Err(reason) => return WavenetTopologyResult::rejected_topology(reason),
    };

    // ── Aggregate state budget check ──
    if let Err(reason) = compute_state_budget(
        &extract.dilations,
        &extract.kernel_sizes,
        &extract.channels,
        &extract.head_kernel_sizes,
        &extract.head_dilations,
    ) {
        return WavenetTopologyResult::rejected_topology(reason);
    }

    // ── Slimmable metadata validation (E1316) ──
    // Run before general A1 guardrails so slimmable invariants (method, channels,
    // head_kernel_size, groups) produce E1316 instead of generic E1305.
    let allowed_channels = match validate_slimmable_metadata(layers, &extract.channels) {
        Err(reason) => return WavenetTopologyResult::rejected_feature(reason),
        Ok(ac) => ac,
    };

    // ── Guardrail: Reject A1 models with A2-specific features (E1305) ──
    if let Err(reason) = validate_a1_guardrail(layers) {
        return WavenetTopologyResult::rejected_topology(reason);
    }

    // ── Try matching a known catalog SKU (fast-path) ──
    if layers.len() == 2 {
        let catalog_compatible = !layers[0].gated.unwrap_or(false)
            && !layers[1].gated.unwrap_or(false)
            && !layers[0].head_bias.unwrap_or(false)
            && layers[1].head_bias.unwrap_or(false)
            && condition_size <= 1
            && data.config.condition_dsp.is_none()
            && allowed_channels.is_none()
            && extract.head_kernel_sizes.iter().all(|&k| k == 1)
            && extract.head_dilations.iter().all(|&d| d == 1);

        if catalog_compatible
            && let Some(sku) = try_match_catalog_sku(extract.first_channels, &extract.dilations)
        {
            return WavenetTopologyResult::Known(sku);
        }
    }

    // ── Free geometry (valid A1, but not in catalog) ──
    let Some(kernel_size) = extract.first_kernel_size else {
        return WavenetTopologyResult::rejected_topology(
            "Layer 0 is missing or has invalid 'kernel_size' — required for \
             free geometry WaveNet A1.",
        );
    };
    if extract.first_head_size.is_none_or(|h| h == 0) {
        return WavenetTopologyResult::rejected_topology(
            "Layer 0 is missing or has invalid 'head_size' — required for \
             WaveNet A1 geometries (determines the head projection dimension).",
        );
    }

    // ── Post-stack head ceilings (F-RES2-02) ──
    // The head is the only loader dimension without a declared cap; validate
    // it before the geometry is accepted so hostile `head.channels` /
    // `head.out_channels` / `head.kernel_size` never reach the size arithmetic
    // in `PostStackHead::from_config` (now checked, but still fail-closed here
    // so the whole topology is rejected before any allocation).
    let in_channels = data
        .config
        .layers
        .last()
        .and_then(|l| l.head_size)
        .unwrap_or(1);
    let Ok(post_stack_head) = data.config.validate_head(in_channels) else {
        return WavenetTopologyResult::rejected_topology(
            "WaveNet free-geometry head exceeds canonical ceilings \
             (channels/out_channels/kernel_size) — OOM/DoS protection (F-RES2-02).",
        );
    };
    if allowed_channels.is_some() && post_stack_head.is_some() {
        return WavenetTopologyResult::rejected_feature(
            "SlimmableWavenet: post-stack head is not supported.",
        );
    }

    WavenetTopologyResult::Free(Box::new(FreeWavenetGeometry {
        channels: extract.channels,
        kernel_size,
        kernel_sizes: extract.kernel_sizes,
        head_sizes: extract.head_sizes,
        head_kernel_sizes: extract.head_kernel_sizes,
        head_dilations: extract.head_dilations,
        head_biases: extract.head_biases,
        condition_size,
        num_arrays: layers.len(),
        dilations: extract.dilations,
        post_stack_head,
        allowed_channels,
    }))
}

// ── Layer metadata extraction ─────────────────────────────────────────────

struct LayerMetadata {
    first_channels: usize,
    first_kernel_size: Option<usize>,
    first_head_size: Option<usize>,
    dilations: Vec<Vec<usize>>,
    head_sizes: Vec<usize>,
    head_kernel_sizes: Vec<usize>,
    head_dilations: Vec<usize>,
    channels: Vec<usize>,
    kernel_sizes: Vec<usize>,
    head_biases: Vec<bool>,
}

fn extract_layer_metadata(layers: &[NamLayerConfig]) -> Result<LayerMetadata, String> {
    let mut first_channels: Option<usize> = None;
    let mut first_kernel_size: Option<usize> = None;
    let mut first_head_size: Option<usize> = None;
    let mut dilations = Vec::with_capacity(layers.len());
    let mut head_sizes = Vec::with_capacity(layers.len());
    let mut head_kernel_sizes = Vec::with_capacity(layers.len());
    let mut head_dilations = Vec::with_capacity(layers.len());
    let mut channels = Vec::with_capacity(layers.len());
    let mut kernel_sizes = Vec::with_capacity(layers.len());
    let mut head_biases = Vec::with_capacity(layers.len());

    for (i, layer) in layers.iter().enumerate() {
        let ch = match layer.channels {
            Some(c) if c > 0 => {
                if c > MAX_WAVENET_FREE_CHANNELS {
                    return Err(format!(
                        "Layer {} channels ({}) exceeds maximum {} — OOM/DoS protection.",
                        i, c, MAX_WAVENET_FREE_CHANNELS
                    ));
                }
                c
            }
            _ => return Err(format!("Layer {} is missing or has invalid 'channels'.", i)),
        };
        let k = layer.kernel_size.filter(|&k| k > 0);
        if let Some(k) = k
            && k > MAX_KERNEL_SIZE
        {
            return Err(format!(
                "Layer {} kernel_size ({}) exceeds maximum {} — DoS/OOM protection.",
                i, k, MAX_KERNEL_SIZE
            ));
        }
        let dils = match layer.dilations.as_deref() {
            Some(d) if !d.is_empty() => d.to_vec(),
            _ => {
                return Err(format!(
                    "Layer {} is missing or has invalid 'dilations'.",
                    i
                ));
            }
        };
        if dils.len() > MAX_DILATIONS_PER_ARRAY {
            return Err(format!(
                "Layer {} has {} dilations, exceeding maximum {} — DoS/OOM protection.",
                i,
                dils.len(),
                MAX_DILATIONS_PER_ARRAY
            ));
        }
        for (j, &d) in dils.iter().enumerate() {
            if d == 0 {
                return Err(format!(
                    "Layer {} dilation[{}] is 0 — must be >= 1 (invalid dilation).",
                    i, j
                ));
            }
            if d > MAX_DILATION {
                return Err(format!(
                    "Layer {} dilation[{}] ({}) exceeds maximum {} — DoS/OOM protection.",
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
            return Err(format!("Layer {} has invalid head_size=0.", i));
        }
        if hd > MAX_HEAD_SIZE {
            return Err(format!(
                "Layer {} head_size ({}) exceeds maximum {} — DoS/OOM protection.",
                i, hd, MAX_HEAD_SIZE
            ));
        }
        let head_k = layer.head_kernel_size.unwrap_or(1);
        if head_k == 0 {
            return Err(format!("Layer {} has invalid head_kernel_size=0.", i));
        }
        if head_k > MAX_KERNEL_SIZE {
            return Err(format!(
                "Layer {} head_kernel_size ({}) exceeds maximum {} — DoS/OOM protection.",
                i, head_k, MAX_KERNEL_SIZE
            ));
        }
        let head_dil = layer.head_dilation.unwrap_or(1);
        if head_dil == 0 {
            return Err(format!("Layer {} has invalid head_dilation=0.", i));
        }
        if head_dil > MAX_DILATION {
            return Err(format!(
                "Layer {} head_dilation ({}) exceeds maximum {} — DoS/OOM protection.",
                i, head_dil, MAX_DILATION
            ));
        }

        channels.push(ch);
        kernel_sizes.push(k.unwrap_or(0));
        head_sizes.push(hd);
        head_kernel_sizes.push(head_k);
        head_dilations.push(head_dil);
        head_biases.push(layer.head_bias.unwrap_or(i == layers.len() - 1));
        dilations.push(dils);
    }

    let first_channels_val = first_channels.ok_or_else(|| {
        "Layer 0 is missing or has invalid 'channels' — required for \
         WaveNet topology detection."
            .to_string()
    })?;

    Ok(LayerMetadata {
        first_channels: first_channels_val,
        first_kernel_size,
        first_head_size,
        dilations,
        head_sizes,
        head_kernel_sizes,
        head_dilations,
        channels,
        kernel_sizes,
        head_biases,
    })
}
