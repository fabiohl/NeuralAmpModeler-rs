// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::common::diagnostics::ModelInfo;
use crate::models::StaticModel;
use std::path::Path;

/// Default input level in dBu for models that do not specify metadata.
pub(crate) const DEFAULT_INPUT_LEVEL_DBU: f32 = 12.0;
/// Default reference loudness in dB for normalization.
pub(crate) const DEFAULT_LOUDNESS_DB: f32 = -18.0;
/// Default reference sample rate (NAM standard).
pub(crate) const DEFAULT_SAMPLE_RATE: f32 = 48000.0;
/// Maximum allowed model file size (256 MiB).
pub(crate) const MAX_MODEL_BYTES: u64 = 256 * 1024 * 1024;

/// Maximum absolute plausible value (dBu) for level/loudness metadata.
///
/// Anything beyond ±60 dBu is outside the physically meaningful range for
/// audio equipment and is treated as hostile/saturated metadata.
pub(crate) const MAX_PLAUSIBLE_DBU: f32 = 60.0;
/// Maximum plausible `head_scale` linear multiplier.
///
/// `head_scale` scales the model head output; production models use values
/// around 0.02–6.0. Anything above 100.0 (≈40 dB) is treated as hostile.
pub(crate) const MAX_PLAUSIBLE_HEAD_SCALE: f32 = 100.0;

/// Typed error for hostile metadata floats rejected by post-parse validation.
///
/// [`load_and_build_model`](crate::loader::load_and_build_model) validates
/// `input_level_dbu`, `output_level_dbu`, `loudness`, and `head_scale` after
/// parsing — for both `.nam` (JSON) and `.namb` (binary) paths. Non-finite or
/// implausible values are rejected so they can never contaminate gain staging
/// (`db_to_linear`) with NaN/Inf or zero gain (F-14).
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum MetadataError {
    /// A metadata float is non-finite (NaN, +Inf, or -Inf).
    #[error("metadata field '{field}' is not finite (value: {value:e})")]
    NonFinite {
        /// Metadata field name.
        field: &'static str,
        /// The non-finite value.
        value: f32,
    },
    /// A dBu metadata float is outside the plausible range (±60 dBu).
    #[error(
        "metadata field '{field}' is outside the plausible range: \
         {value:e} dBu (allowed: ±{max} dBu)"
    )]
    DbOutOfRange {
        /// Metadata field name.
        field: &'static str,
        /// The implausible value.
        value: f32,
        /// Maximum absolute plausible value in dBu.
        max: f32,
    },
    /// `head_scale` (linear multiplier) is outside the plausible range.
    #[error(
        "metadata field 'head_scale' is outside the plausible range: \
         {value:e} (allowed: 0.0 < x <= {max})"
    )]
    HeadScaleOutOfRange {
        /// The implausible value.
        value: f32,
        /// Maximum plausible linear multiplier.
        max: f32,
    },
}

/// Pair of loaded models with calibration metadata.
pub struct LoadedModelPair {
    /// Model for the left channel.
    pub model_l: Option<Box<StaticModel>>,
    /// Model for the right channel.
    pub model_r: Option<Box<StaticModel>>,
    /// Input gain adjustment multiplier.
    pub input_mult_adj: f32,
    /// Output gain adjustment multiplier.
    pub output_mult_adj: f32,
    /// Native sample rate of the model.
    pub sample_rate: u32,
    /// Model architecture (e.g. "LSTM", "WaveNet").
    pub architecture: String,
    /// Model topology (e.g. "Standard", "1x64").
    pub topology: String,
    /// Optional model metadata.
    pub metadata: Option<crate::loader::nam_json::NamMetadata>,
    /// Weights layout format.
    pub weights_layout: String,
}

impl LoadedModelPair {
    /// Overall recorded loudness from model metadata, if available.
    pub fn loudness(&self) -> Option<f32> {
        self.metadata.as_ref().and_then(|m| m.loudness)
    }

    /// Expected input level (dBu) for proper gain staging, if available.
    pub fn input_level_dbu(&self) -> Option<f32> {
        self.metadata.as_ref().and_then(|m| m.input_level_dbu)
    }

    /// Expected output level (dBu) for the model, if available.
    pub fn output_level_dbu(&self) -> Option<f32> {
        self.metadata.as_ref().and_then(|m| m.output_level_dbu)
    }

    /// Whether the model metadata contains a loudness value.
    pub fn has_loudness(&self) -> bool {
        self.loudness().is_some()
    }

    /// Whether the model metadata contains an input level (dBu) value.
    pub fn has_input_level_dbu(&self) -> bool {
        self.input_level_dbu().is_some()
    }

    /// Whether the model metadata contains an output level (dBu) value.
    pub fn has_output_level_dbu(&self) -> bool {
        self.output_level_dbu().is_some()
    }

    /// Builds a [`ModelInfo`] snapshot from this loaded pair and the source file path.
    pub fn model_info(&self, path: &Path) -> ModelInfo {
        ModelInfo {
            arch_label: self.architecture.clone(),
            topology: self.topology.clone(),
            channels: self.model_l.as_ref().map(|m| m.channels()).unwrap_or(0),
            receptive_field: self
                .model_l
                .as_ref()
                .map(|m| m.receptive_field())
                .unwrap_or(0),
            model_sample_rate: self.sample_rate,
            weights_layout: self.weights_layout.clone(),
            path_basename: path.to_string_lossy().into_owned(),
        }
    }
}

impl std::fmt::Debug for LoadedModelPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModelPair")
            .field("model_l", &self.model_l.as_ref().map(|_| "StaticModel"))
            .field("model_r", &self.model_r.as_ref().map(|_| "StaticModel"))
            .field("input_mult_adj", &self.input_mult_adj)
            .field("output_mult_adj", &self.output_mult_adj)
            .field("sample_rate", &self.sample_rate)
            .field("architecture", &self.architecture)
            .field("topology", &self.topology)
            .field("metadata", &self.metadata)
            .field("weights_layout", &self.weights_layout)
            .finish()
    }
}
