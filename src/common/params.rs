// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Host-agnostic parameters for NeuralAmpModeler-rs.
//!
//! This module defines the complete processing configuration state,
//! allowing different host applications to manage and
//! synchronize parameters consistently.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub use crate::dsp::adaptive::AdaptiveComputeMode;
pub use crate::dsp::adaptive::SlimOverride;
use crate::dsp::oversample::OversampleFactor;
pub use crate::math::activations::ActivationPrecision;

const GATE_THRESHOLD_DB_DEFAULT: f32 = -70.0;

/// Global processing parameters for the plugin/application.
///
/// This structure encapsulates all controls available to the user,
/// from basic gains to the path of the loaded neural model.
///
/// ## State Compatibility
///
/// Snapshots of `ProcessingParams` serialized by previous versions of this crate
/// can be deserialized by subsequent versions via `serde_json::from_str` — missing
/// fields receive their default values (`#[serde(default)]`). This contract is
/// verified by regression tests in `tests/state_compat.rs`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessingParams {
    /// Input gain in decibels (dB). Default: 0.0.
    #[serde(default)]
    pub input_gain_db: f32,
    /// Output gain in decibels (dB). Default: 0.0.
    #[serde(default)]
    pub output_gain_db: f32,
    /// Noise Gate threshold in decibels (dB). Default: -70.0.
    /// This value maps to the `threshold_open_db` of the gate engine.
    #[serde(default = "default_gate_threshold_db")]
    pub gate_threshold_db: f32,
    /// Path to the loaded `.nam` or `.namb` model.
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    /// Base name of the model (filename only), used for portable lookup
    /// when the absolute path does not exist (cross-machine / cross-user).
    #[serde(default)]
    pub model_basename: Option<String>,
    /// SHA-256 hex digest of the model file, for content-based portable
    /// identity across machines and OSes.
    #[serde(default)]
    pub model_hash: Option<String>,
    /// Directories to search for the model if the absolute `model_path` does not exist.
    #[serde(default)]
    pub model_search_paths: Vec<PathBuf>,
    /// Bypass state (if `true`, audio passes without neural processing).
    #[serde(default)]
    pub bypass: bool,
    /// Adaptive compute mode for soft-degrade under CPU pressure.
    /// Default: `Off` for standalone; `Conservative` for host plugin.
    #[serde(default)]
    pub adaptive_compute: AdaptiveComputeMode,
    /// Manual slim override quality level. Default: `Auto` (FSM decides).
    #[serde(default)]
    pub slim_override: SlimOverride,
    /// Oversampling factor for the neural stage (anti-aliasing).
    /// Default: `Off` (lowest latency).
    #[serde(default)]
    pub oversample: OversampleFactor,
    /// Path to the loaded cab-sim impulse response (.wav).
    #[serde(default)]
    pub ir_path: Option<PathBuf>,
    /// SHA-256 hex digest of the IR file, for content-based portable
    /// identity across machines and OSes.
    #[serde(default)]
    pub ir_hash: Option<String>,
    /// Activation precision mode (`Standard` or `Fast`).
    /// Default: `Standard` (universal, exact-grade — matches NAMCore C++).
    #[serde(default)]
    pub activation_precision: ActivationPrecision,
}

fn default_gate_threshold_db() -> f32 {
    GATE_THRESHOLD_DB_DEFAULT
}

impl ProcessingParams {
    /// Creates a new `ProcessingParams` initialized to default values for fluent building.
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_amp_modeler_rs::prelude::*;
    ///
    /// let params = ProcessingParams::builder()
    ///     .with_input_gain_db(-3.0)
    ///     .with_output_gain_db(0.0)
    ///     .with_adaptive_compute(AdaptiveComputeMode::Conservative)
    ///     .with_oversample(OversampleFactor::X2);
    ///
    /// assert_eq!(params.input_gain_db, -3.0);
    /// assert_eq!(params.adaptive_compute, AdaptiveComputeMode::Conservative);
    /// ```
    pub fn builder() -> Self {
        Self::default()
    }

    /// Sets the input gain in decibels (dB).
    pub fn with_input_gain_db(mut self, db: f32) -> Self {
        self.input_gain_db = db;
        self
    }

    /// Sets the output gain in decibels (dB).
    pub fn with_output_gain_db(mut self, db: f32) -> Self {
        self.output_gain_db = db;
        self
    }

    /// Sets the noise gate threshold in decibels (dB).
    pub fn with_gate_threshold_db(mut self, db: f32) -> Self {
        self.gate_threshold_db = db;
        self
    }

    /// Sets the neural model path.
    pub fn with_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Sets the portable model base name.
    pub fn with_model_basename(mut self, name: impl Into<String>) -> Self {
        self.model_basename = Some(name.into());
        self
    }

    /// Sets the SHA-256 model content hash.
    pub fn with_model_hash(mut self, hash: impl Into<String>) -> Self {
        self.model_hash = Some(hash.into());
        self
    }

    /// Sets the directories to search for models if the primary path is missing.
    pub fn with_model_search_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.model_search_paths = paths;
        self
    }

    /// Appends a directory to search for models.
    pub fn with_model_search_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_search_paths.push(path.into());
        self
    }

    /// Sets the bypass state.
    pub fn with_bypass(mut self, bypass: bool) -> Self {
        self.bypass = bypass;
        self
    }

    /// Sets the adaptive compute mode.
    pub fn with_adaptive_compute(mut self, mode: AdaptiveComputeMode) -> Self {
        self.adaptive_compute = mode;
        self
    }

    /// Sets the slim override mode.
    pub fn with_slim_override(mut self, slim: SlimOverride) -> Self {
        self.slim_override = slim;
        self
    }

    /// Sets the oversampling factor.
    pub fn with_oversample(mut self, oversample: OversampleFactor) -> Self {
        self.oversample = oversample;
        self
    }

    /// Sets the cab-sim IR file path.
    pub fn with_ir_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.ir_path = Some(path.into());
        self
    }

    /// Sets the cab-sim IR content hash.
    pub fn with_ir_hash(mut self, hash: impl Into<String>) -> Self {
        self.ir_hash = Some(hash.into());
        self
    }

    /// Sets the activation precision mode.
    pub fn with_activation_precision(mut self, precision: ActivationPrecision) -> Self {
        self.activation_precision = precision;
        self
    }
}

impl Default for ProcessingParams {
    fn default() -> Self {
        Self {
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            gate_threshold_db: GATE_THRESHOLD_DB_DEFAULT,
            model_path: None,
            model_basename: None,
            model_hash: None,
            model_search_paths: Vec::new(),
            bypass: false,
            adaptive_compute: AdaptiveComputeMode::Off,
            slim_override: SlimOverride::Auto,
            oversample: OversampleFactor::Off,
            ir_path: None,
            ir_hash: None,
            activation_precision: ActivationPrecision::Standard,
        }
    }
}

/// Simplified parameter snapshot for the Real-Time audio thread.
/// Contains no heap-allocated fields (like PathBuf or String) to guarantee RT-safety when dropped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtProcessingParams {
    /// Input gain in decibels (dB).
    pub input_gain_db: f32,
    /// Output gain in decibels (dB).
    pub output_gain_db: f32,
    /// Noise Gate threshold in decibels (dB).
    pub gate_threshold_db: f32,
    /// Bypass state.
    pub bypass: bool,
    /// Adaptive compute mode.
    pub adaptive_compute: AdaptiveComputeMode,
    /// Manual slim override quality level.
    pub slim_override: SlimOverride,
    /// Oversampling factor for the neural stage (anti-aliasing).
    pub oversample: OversampleFactor,
    /// Activation precision mode.
    pub activation_precision: ActivationPrecision,
}

impl RtProcessingParams {
    /// Extract the RT-safe parameters from a full ProcessingParams.
    pub fn from_processing_params(params: &ProcessingParams) -> Self {
        Self {
            input_gain_db: params.input_gain_db,
            output_gain_db: params.output_gain_db,
            gate_threshold_db: params.gate_threshold_db,
            bypass: params.bypass,
            adaptive_compute: params.adaptive_compute,
            slim_override: params.slim_override,
            oversample: params.oversample,
            activation_precision: params.activation_precision,
        }
    }
}

impl Default for RtProcessingParams {
    fn default() -> Self {
        Self {
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            gate_threshold_db: GATE_THRESHOLD_DB_DEFAULT,
            bypass: false,
            adaptive_compute: AdaptiveComputeMode::Off,
            slim_override: SlimOverride::Auto,
            oversample: OversampleFactor::Off,
            activation_precision: ActivationPrecision::Standard,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_default() {
        let params = ProcessingParams::default();
        assert_eq!(params.input_gain_db, 0.0);
        assert_eq!(params.output_gain_db, 0.0);
        assert_eq!(params.gate_threshold_db, -70.0);
        assert_eq!(params.model_path, None);
        assert_eq!(params.model_basename, None);
        assert_eq!(params.model_hash, None);
        assert!(params.model_search_paths.is_empty());
        assert!(!params.bypass);
        assert_eq!(params.ir_path, None);
    }

    #[test]
    fn test_params_builder() {
        let params = ProcessingParams::builder()
            .with_input_gain_db(3.5)
            .with_output_gain_db(-2.0)
            .with_gate_threshold_db(-65.0)
            .with_model_path(PathBuf::from("/models/amp.nam"))
            .with_model_basename("amp.nam")
            .with_model_hash("abcd1234efgh5678")
            .with_model_search_path(PathBuf::from("/custom/models"))
            .with_bypass(true)
            .with_adaptive_compute(AdaptiveComputeMode::Conservative)
            .with_slim_override(SlimOverride::ForceFull)
            .with_oversample(OversampleFactor::X2)
            .with_ir_path(PathBuf::from("/irs/cab.wav"))
            .with_ir_hash("feedbeefcafe")
            .with_activation_precision(ActivationPrecision::Fast);

        assert_eq!(params.input_gain_db, 3.5);
        assert_eq!(params.output_gain_db, -2.0);
        assert_eq!(params.gate_threshold_db, -65.0);
        assert_eq!(params.model_path, Some(PathBuf::from("/models/amp.nam")));
        assert_eq!(params.model_basename.as_deref(), Some("amp.nam"));
        assert_eq!(params.model_hash.as_deref(), Some("abcd1234efgh5678"));
        assert_eq!(
            params.model_search_paths,
            vec![PathBuf::from("/custom/models")]
        );
        assert!(params.bypass);
        assert_eq!(params.adaptive_compute, AdaptiveComputeMode::Conservative);
        assert_eq!(params.slim_override, SlimOverride::ForceFull);
        assert_eq!(params.oversample, OversampleFactor::X2);
        assert_eq!(params.ir_path, Some(PathBuf::from("/irs/cab.wav")));
        assert_eq!(params.ir_hash.as_deref(), Some("feedbeefcafe"));
        assert_eq!(params.activation_precision, ActivationPrecision::Fast);
    }
}
