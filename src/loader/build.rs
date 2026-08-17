// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Model loading and building — reads `.nam`/`.namb` files, parses, calibrates,
//! and dispatches to the appropriate architecture builder.

use crate::common::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::loader::{dispatcher, nam_json, namb};
use crate::models::NamModel;
use log::{debug, info};
use std::path::Path;

use super::loaded_model_pair::{
    DEFAULT_INPUT_LEVEL_DBU, DEFAULT_LOUDNESS_DB, DEFAULT_SAMPLE_RATE, LoadedModelPair,
    MAX_MODEL_BYTES, MAX_PLAUSIBLE_DBU, MAX_PLAUSIBLE_HEAD_SCALE, MetadataError,
};

/// Validates metadata floats (`input_level_dbu`, `output_level_dbu`, `loudness`,
/// `head_scale`) for finiteness and plausible range (F-14).
///
/// Weights already reject NaN/Inf element-by-element; metadata does not go
/// through that path. A JSON value like `1e39` silently saturates to
/// `+Inf` when deserialized into `f32`, which would poison gain staging
/// (`db_to_linear`) with NaN/Inf/zero multipliers. This post-parse gate
/// applies uniformly to `.nam` and `.namb` loads.
fn validate_metadata_floats(
    meta: &crate::loader::nam_json::NamMetadata,
    head_scale: Option<f32>,
) -> anyhow::Result<()> {
    let db_fields: [(&'static str, Option<f32>); 3] = [
        ("input_level_dbu", meta.input_level_dbu),
        ("output_level_dbu", meta.output_level_dbu),
        ("loudness", meta.loudness),
    ];
    for (field, value) in db_fields {
        if let Some(v) = value {
            if !v.is_finite() {
                return Err(MetadataError::NonFinite { field, value: v }.into());
            }
            if v.abs() > MAX_PLAUSIBLE_DBU {
                return Err(MetadataError::DbOutOfRange {
                    field,
                    value: v,
                    max: MAX_PLAUSIBLE_DBU,
                }
                .into());
            }
        }
    }
    if let Some(v) = head_scale {
        if !v.is_finite() {
            return Err(MetadataError::NonFinite {
                field: "head_scale",
                value: v,
            }
            .into());
        }
        if v <= 0.0 || v > MAX_PLAUSIBLE_HEAD_SCALE {
            return Err(MetadataError::HeadScaleOutOfRange {
                value: v,
                max: MAX_PLAUSIBLE_HEAD_SCALE,
            }
            .into());
        }
    }
    Ok(())
}

/// Reads a model file into a byte buffer after validating its size.
///
/// Centralizes metadata retrieval, size validation against
/// [`MAX_MODEL_BYTES`], and byte reading previously duplicated for `.nam`
/// and `.namb` paths.
fn read_and_validate_model_bytes(
    path: &Path,
    path_str: &str,
    sys: &SystemSnapshot,
) -> anyhow::Result<Vec<u8>> {
    let len = std::fs::metadata(path)
        .map_err(|e| {
            NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                .message(format!("Failed to read metadata of \"{}\".", path_str))
                .hint("Please verify file access permissions.")
                .param("file", path_str)
                .param("io_error", &e)
                .emit();
            anyhow::Error::from(e)
        })?
        .len();
    if len > MAX_MODEL_BYTES {
        NamDiagnostic::new(NamErrorCode::ModelTooLarge, sys)
            .message(format!(
                "Model file \"{}\" is too large ({} bytes, max is {} bytes).",
                path_str, len, MAX_MODEL_BYTES
            ))
            .hint("Please check the file size and ensure it is a valid NAM model.")
            .param("file", path_str)
            .param("size_bytes", len)
            .emit();
        return Err(anyhow::anyhow!(
            "Model file \"{}\" exceeds maximum allowed size of {} MiB.",
            path_str,
            MAX_MODEL_BYTES / (1024 * 1024)
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| {
        NamDiagnostic::new(NamErrorCode::FileReadError, sys)
            .message(format!("Failed to read the file \"{}\".", path_str))
            .hint("Please verify file access permissions.")
            .param("file", path_str)
            .param("io_error", &e)
            .emit();
        anyhow::Error::from(e)
    })?;
    Ok(bytes)
}

/// Loads and builds a model pair from a file.
///
/// Returns `Ok(pair)` guaranteeing that `pair.model_l` is non-null (`Some`) and
/// ready for real-time audio processing.
///
/// When `stereo` is `false`, only the left-channel model is built (`model_r` is `None`),
/// avoiding unnecessary instantiation and prewarming. When `stereo` is `true`,
/// both `model_l` and `model_r` are built (`Some`).
///
/// If file reading, format parsing, metadata validation, or architecture dispatching/construction
/// fails for any requested channel, an error (`Err`) is returned.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use neural_amp_modeler_rs::loader::{load_and_build_model, LoadOptions};
/// use neural_amp_modeler_rs::SystemSnapshot;
///
/// // Capture system capabilities (SIMD feature set, CPU topology)
/// let sys = SystemSnapshot::capture();
///
/// // Load a model file (.nam or .namb)
/// let pair = load_and_build_model(
///     Path::new("path/to/model.nam"),
///     &sys,
///     false, // mono execution (set true for stereo)
///     LoadOptions::default(),
/// )
/// .expect("Failed to load model");
///
/// assert!(pair.model_l.is_some());
/// assert!(pair.model_r.is_none()); // Mono load: right channel is None
/// ```
pub fn load_and_build_model(
    path: &Path,
    sys: &SystemSnapshot,
    stereo: bool,
    options: crate::loader::LoadOptions,
) -> anyhow::Result<LoadedModelPair> {
    let path_str = path.to_string_lossy();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let ext_lower = ext.to_lowercase();

    info!("[Loader] Loading model from \"{}\"", path_str);

    // 1. Reading and Parsing
    let model_data = if ext_lower == "namb" {
        let bytes = read_and_validate_model_bytes(path, &path_str, sys)?;
        namb::parse_namb(&bytes).inspect_err(|e| {
            let code = match e.downcast_ref::<namb::NambError>() {
                Some(namb::NambError::Truncated { .. }) => NamErrorCode::NambTruncated,
                Some(namb::NambError::InvalidMagic(_)) => NamErrorCode::NambInvalidMagic,
                Some(namb::NambError::InvalidVersion(_)) => NamErrorCode::NambUnsupportedVersion,
                Some(namb::NambError::WeightsOffsetOutOfBounds { .. })
                | Some(namb::NambError::InvalidWeightsOffset { .. }) => NamErrorCode::NambTruncated,
                Some(namb::NambError::CrcMismatch { .. }) => NamErrorCode::NambCrc32Mismatch,
                Some(namb::NambError::CrcMissing { .. }) | Some(namb::NambError::CrcMissingV1) => {
                    NamErrorCode::NambCrc32Missing
                }
                Some(namb::NambError::WeightsTooLarge { .. }) => NamErrorCode::ModelTooLarge,
                Some(namb::NambError::NonFiniteWeight { .. }) => NamErrorCode::NambNonFiniteWeight,
                Some(namb::NambError::InvalidHeaderField { .. }) => {
                    NamErrorCode::NambInvalidHeaderField
                }
                None => NamErrorCode::ModelBuildFailed,
            };
            NamDiagnostic::new(code, sys)
                .message(format!("Invalid \".namb\" file: {}", path_str))
                .param("detail", e.to_string())
                .emit();
        })?
    } else if ext_lower == "nam" {
        let bytes = read_and_validate_model_bytes(path, &path_str, sys)?;
        let json = String::from_utf8(bytes).map_err(|e| {
            NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                .message(format!("File \"{}\" contains invalid UTF-8.", path_str))
                .hint("Only UTF-8 encoded .nam files are supported.")
                .param("file", &path_str)
                .param("utf8_error", &e)
                .emit();
            anyhow::Error::from(e)
        })?;
        nam_json::parse_nam_json(&json).inspect_err(|e| {
            let code = match e {
                nam_json::JsonError::WeightsExceedLimit { .. } => {
                    NamErrorCode::NamJsonWeightsExceedLimit
                }
                nam_json::JsonError::TrainingTooLarge { .. } => {
                    NamErrorCode::NamJsonTrainingTooLarge
                }
                nam_json::JsonError::TrainingTooDeep { .. } => NamErrorCode::NamJsonTrainingTooDeep,
                nam_json::JsonError::SubmodelsExceedLimit { .. } => {
                    NamErrorCode::NamJsonSubmodelsExceedLimit
                }
                nam_json::JsonError::SubmodelsTooDeep { .. } => {
                    NamErrorCode::NamJsonSubmodelsTooDeep
                }
                nam_json::JsonError::WeightNotFinite { .. } => NamErrorCode::NamJsonWeightNotFinite,
                nam_json::JsonError::InvalidSampleRate { .. } => {
                    NamErrorCode::NamJsonInvalidSampleRate
                }
                nam_json::JsonError::UnsupportedTopology { .. } => {
                    NamErrorCode::NamJsonUnsupportedTopology
                }
                nam_json::JsonError::InvalidVersionFormat { .. } => {
                    NamErrorCode::NamJsonInvalidVersionFormat
                }
                nam_json::JsonError::UnsupportedVersion { .. } => {
                    NamErrorCode::NamJsonUnsupportedVersion
                }
                nam_json::JsonError::UnsupportedMultiChannel { .. } => {
                    NamErrorCode::NamJsonUnsupportedMultiChannel
                }
                _ => NamErrorCode::NamJsonParseError,
            };
            NamDiagnostic::new(code, sys)
                .message(format!("Error parsing model JSON: {}", path_str))
                .param("detail", e)
                .emit();
        })?
    } else {
        return Err(anyhow::anyhow!("Unsupported file extension: {}", ext));
    };

    let model_version = model_data.version.as_deref().unwrap_or("(unknown)");
    let weights_count = model_data.weights.len();
    let model_sample_rate = model_data.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE);
    info!(
        "[Loader] Parsed model: arch=\"{}\", version={}, {} weights, sample_rate={:.0} Hz",
        model_data.architecture, model_version, weights_count, model_sample_rate
    );
    debug!(
        "[Loader] Model details: {:?} weights_layout",
        model_data.weights_layout
    );

    // 2. Metadata and Calibration Extraction
    let meta = model_data.metadata.clone().unwrap_or_default();
    validate_metadata_floats(&meta, model_data.config.head_scale).inspect_err(|e| {
        NamDiagnostic::new(NamErrorCode::InvalidMetadata, sys)
            .message(format!("Invalid model metadata in \"{}\"", path_str))
            .param("detail", e.to_string())
            .emit();
    })?;
    let in_level = meta.input_level_dbu.unwrap_or(DEFAULT_INPUT_LEVEL_DBU);
    let loudness = meta.loudness.unwrap_or(DEFAULT_LOUDNESS_DB);

    let input_db_adj = DEFAULT_INPUT_LEVEL_DBU - in_level;
    let output_db_adj = DEFAULT_LOUDNESS_DB - loudness;
    let nam_rate = model_data.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE) as u32;

    let lut = crate::math::dsp::gain_lut::get_gain_lut();
    let input_mult_adj = lut.db_to_linear(input_db_adj);
    let output_mult_adj = lut.db_to_linear(output_db_adj);

    let model_name = meta.name.as_deref().unwrap_or("(unnamed)");
    debug!(
        "[Loader] Metadata: name=\"{}\", in_level={:.1} dBu, loudness={:.1} dB, \
         input_adj={:+.1} dB, output_adj={:+.1} dB",
        model_name, in_level, loudness, input_db_adj, output_db_adj
    );

    // 3. Dispatcher (Build Model L/R)
    info!(
        "[Loader] Dispatching model build: arch=\"{}\", layout={:?}",
        model_data.architecture, model_data.weights_layout
    );
    let mut model_l = dispatcher::build_model(&model_data).inspect_err(|e| {
        let code = if e.to_string().contains("slimmable") {
            NamErrorCode::InvalidModelTopology
        } else {
            NamErrorCode::ModelBuildFailed
        };
        NamDiagnostic::new(code, sys)
            .message(format!("Failed to build model (L): {}", path_str))
            .param("detail", e.to_string())
            .emit();
    })?;

    // Size the model for the pipeline's documented maximum block size
    // (`MAX_RESAMP_BUF`), so containers/crossfade scratch and ring buffers
    // are pre-allocated for 8192-sample blocks before processing starts.
    model_l.set_max_buffer_size(crate::dsp::pipeline::MAX_RESAMP_BUF)?;

    if options.prewarm == Some(false) {
        model_l.set_prewarm_on_reset(false);
    } else {
        model_l.prewarm(model_l.prewarm_samples().max(2048));
    }

    let model_r = if stereo {
        let mut model = dispatcher::build_model(&model_data).inspect_err(|e| {
            let code = if e.to_string().contains("slimmable") {
                NamErrorCode::InvalidModelTopology
            } else {
                NamErrorCode::ModelBuildFailed
            };
            NamDiagnostic::new(code, sys)
                .message(format!("Failed to build model (R): {}", path_str))
                .param("detail", e.to_string())
                .emit();
        })?;
        model.set_max_buffer_size(crate::dsp::pipeline::MAX_RESAMP_BUF)?;
        if options.prewarm == Some(false) {
            model.set_prewarm_on_reset(false);
        } else {
            model.prewarm(model.prewarm_samples().max(2048));
        }
        Some(model)
    } else {
        None
    };

    let architecture = model_data.architecture.clone();
    let topology = if architecture == "WaveNet" {
        match nam_json::get_wavenet_topology(&model_data) {
            nam_json::WavenetTopologyResult::Known(nam_json::NamWavenetTopology::Standard) => {
                "Standard".to_string()
            }
            nam_json::WavenetTopologyResult::Known(nam_json::NamWavenetTopology::Lite) => {
                "Lite".to_string()
            }
            nam_json::WavenetTopologyResult::Known(nam_json::NamWavenetTopology::Feather) => {
                "Feather".to_string()
            }
            nam_json::WavenetTopologyResult::Known(nam_json::NamWavenetTopology::Nano) => {
                "Nano".to_string()
            }
            nam_json::WavenetTopologyResult::Free(_) => "WaveNet-Dynamic".to_string(),
            _ => {
                if let Some(topo) = nam_json::is_a2_shape(&model_data) {
                    match topo {
                        nam_json::A2TopologyResult::KnownFastPath(3) => "A2-Lite".to_string(),
                        nam_json::A2TopologyResult::KnownFastPath(8) => "A2-Full".to_string(),
                        nam_json::A2TopologyResult::KnownFastPath(_) => "A2-Unknown".to_string(),
                        nam_json::A2TopologyResult::Dynamic => "A2-Dynamic".to_string(),
                    }
                } else if architecture == "SlimmableContainer" {
                    "Container".to_string()
                } else {
                    "Custom".to_string()
                }
            }
        }
    } else if architecture == "LSTM" {
        match nam_json::get_lstm_topology(&model_data) {
            Ok(Some((layers, hidden))) => format!("{}x{}", layers, hidden),
            _ => "Custom".to_string(),
        }
    } else if architecture == "Linear" {
        match nam_json::get_linear_topology(&model_data) {
            Some((rf, has_bias, _impl)) => {
                if has_bias {
                    format!("RF{} (biased)", rf)
                } else {
                    format!("RF{}", rf)
                }
            }
            None => "Custom".to_string(),
        }
    } else if architecture == "ConvNet" {
        match nam_json::get_convnet_topology(&model_data) {
            Some(topo) => format!("B{}", topo.num_blocks),
            None => "Custom".to_string(),
        }
    } else {
        "Unknown".to_string()
    };
    let metadata = model_data.metadata.clone();
    let weights_layout_str = match model_data.weights_layout {
        crate::loader::nam_json::WeightsLayout::Original => "Original".to_string(),
        crate::loader::nam_json::WeightsLayout::GateMajorLstm => "GateMajorLstm".to_string(),
        crate::loader::nam_json::WeightsLayout::Interleaved4WaveNet => {
            "Interleaved4WaveNet".to_string()
        }
    };

    let channels = if model_r.is_some() {
        "stereo"
    } else {
        "mono (L only)"
    };
    info!(
        "[Loader] Model built successfully: arch=\"{}\", topology=\"{}\", \
         {} ch, layout={}, sample_rate={} Hz",
        architecture, topology, channels, weights_layout_str, nam_rate
    );

    Ok(LoadedModelPair {
        model_l: Some(model_l),
        model_r,
        input_mult_adj,
        output_mult_adj,
        sample_rate: nam_rate,
        architecture,
        topology,
        metadata,
        weights_layout: weights_layout_str,
    })
}

#[cfg(test)]
#[path = "build_test.rs"]
mod build_test;
