// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Model loading and building — reads `.nam`/`.namb` files, parses, calibrates,
//! and dispatches to the appropriate architecture builder.

use crate::common::diagnostics::{NamErrorCode, SystemSnapshot};
use crate::loader::{dispatcher, nam_json, namb};
use crate::models::NamModel;
use log::{debug, error, info};
use std::path::Path;

use super::error::LoadError;
use super::loaded_model_pair::{
    DEFAULT_INPUT_LEVEL_DBU, DEFAULT_LOUDNESS_DB, DEFAULT_SAMPLE_RATE, LoadedModelPair,
    MAX_MODEL_BYTES, validate_metadata_floats,
};

/// Reads a model file into a byte buffer after validating its size.
///
/// Centralizes metadata retrieval, size validation against
/// [`MAX_MODEL_BYTES`], and byte reading previously duplicated for `.nam`
/// and `.namb` paths.
///
/// Anti-TOCTOU (H-04): the file is opened **once** and read through
/// `take(MAX_MODEL_BYTES + 1)`. The previous `stat` + `fs::read` sequence left
/// a window where the file could grow past the cap between the two syscalls.
fn read_and_validate_model_bytes(
    path: &Path,
    path_str: &str,
) -> Result<Vec<u8>, super::error::LoadError> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|e| {
        // Structured failure diagnostic (size unknown at open time → 0).
        // Library loaders log and return the enriched error; the visual
        // support block is rendered only by CLIs via
        // `NamDiagnostic::support_block()`.
        error!(
            "[Loader] Model build failed: file='{}', size={} bytes, code={:?} — \
             failed to open the file (io_error: {}). Please verify file access permissions.",
            path_str,
            0,
            NamErrorCode::FileReadError,
            e
        );
        super::error::LoadError::Io(e)
    })?;

    // H-04: single open + bounded read. Reading `MAX_MODEL_BYTES + 1` bytes
    // detects an oversized file without ever buffering more than the cap.
    let mut bytes = Vec::new();
    file.by_ref()
        .take(MAX_MODEL_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| {
            error!(
                "[Loader] Model build failed: file='{}', size={} bytes, code={:?} — \
                 failed to read the file (io_error: {}). Please verify file access permissions.",
                path_str,
                bytes.len(),
                NamErrorCode::FileReadError,
                e
            );
            super::error::LoadError::Io(e)
        })?;

    if bytes.len() as u64 > MAX_MODEL_BYTES {
        error!(
            "[Loader] Model build failed: file='{}', size={} bytes, code={:?} — \
             model file is too large ({} bytes, max is {} bytes). Please check the file \
             size and ensure it is a valid NAM model.",
            path_str,
            bytes.len(),
            NamErrorCode::ModelTooLarge,
            bytes.len(),
            MAX_MODEL_BYTES
        );
        return Err(super::error::LoadError::ModelTooLarge);
    }
    Ok(bytes)
}

/// Loads and builds a model pair from a file.
///
/// Returns `Ok(pair)` guaranteeing that `pair.model_l` is non-null (`Some`) and
/// ready for real-time audio processing.
///
/// When `dual_mono` is `false`, only the left-channel model is built (`model_r` is `None`),
/// avoiding unnecessary instantiation and prewarming.
///
/// When `dual_mono` is `true`:
/// - With the `dual-mono` feature enabled (default), both `model_l` and `model_r` are built (`Some`).
/// - If the `dual-mono` feature is disabled at compile time, a warning is logged via `log::warn!`
///   and `pair.model_r` remains `None` to prevent unexpected memory allocation when the engine
///   is compiled for mono-only operation (Footgun F13).
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
///     false, // dual_mono: left-channel only (set true for independent L/R inference)
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
    dual_mono: bool,
    options: crate::loader::LoadOptions,
) -> Result<LoadedModelPair, LoadError> {
    let path_str = path.to_string_lossy();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let ext_lower = ext.to_lowercase();

    info!("[Loader] Loading model from \"{}\"", path_str);

    // Host context rides in the structured trace (debug level): every load
    // failure logged below stays correlated with the captured snapshot in the
    // LogBuffer, without the loader painting support blocks on stderr.
    debug!("[Loader] System snapshot: {:?}", sys);

    // 1. Reading and Parsing
    let (model_data, file_size) = if ext_lower == "namb" {
        let bytes = read_and_validate_model_bytes(path, &path_str)?;
        let file_size = bytes.len();
        let data = namb::parse_namb_typed(&bytes).map_err(|e| {
            let code = match &e {
                namb::NambError::Truncated { .. } => NamErrorCode::NambTruncated,
                namb::NambError::InvalidMagic(_) => NamErrorCode::NambInvalidMagic,
                namb::NambError::InvalidVersion(_) => NamErrorCode::NambUnsupportedVersion,
                namb::NambError::WeightsOffsetOutOfBounds { .. }
                | namb::NambError::InvalidWeightsOffset { .. } => NamErrorCode::NambTruncated,
                namb::NambError::CrcMismatch { .. } => NamErrorCode::NambCrc32Mismatch,
                namb::NambError::CrcMissing { .. } | namb::NambError::CrcMissingV1 => {
                    NamErrorCode::NambCrc32Missing
                }
                namb::NambError::WeightsTooLarge { .. } => NamErrorCode::ModelTooLarge,
                namb::NambError::NonFiniteWeight { .. } => NamErrorCode::NambNonFiniteWeight,
                namb::NambError::InvalidHeaderField { .. } => NamErrorCode::NambInvalidHeaderField,
                namb::NambError::MetadataNotUtf8 { .. } | namb::NambError::MetadataJson(_) => {
                    NamErrorCode::ModelBuildFailed
                }
            };
            // Structured failure diagnostic (path + size + code). The typed
            // error payload travels in the returned `LoadError::Namb`.
            error!(
                "[Loader] Model build failed: file='{}', size={} bytes, code={:?} — \
                 invalid \".namb\" file (detail: {}).",
                path_str, file_size, code, e
            );
            LoadError::from(e)
        })?;
        (data, file_size)
    } else if ext_lower == "nam" {
        let bytes = read_and_validate_model_bytes(path, &path_str)?;
        let file_size = bytes.len();
        let json = String::from_utf8(bytes).map_err(|e| {
            error!(
                "[Loader] Model build failed: file='{}', size={} bytes, code={:?} — \
                 file contains invalid UTF-8 (utf8_error: {}). Only UTF-8 encoded \
                 .nam files are supported.",
                path_str,
                file_size,
                NamErrorCode::FileReadError,
                e
            );
            LoadError::InvalidUtf8(e)
        })?;
        let data = nam_json::parse_nam_json(&json).map_err(|e| {
            let code = match &e {
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
            // Structured failure diagnostic (path + size + code). The typed
            // error payload travels in the returned `LoadError` variant.
            error!(
                "[Loader] Model build failed: file='{}', size={} bytes, code={:?} — \
                 error parsing model JSON (detail: {}).",
                path_str, file_size, code, e
            );
            match e {
                nam_json::JsonError::WeightsExceedLimit { .. }
                | nam_json::JsonError::TrainingTooLarge { .. } => LoadError::ModelTooLarge,
                nam_json::JsonError::WeightNotFinite { .. } => LoadError::NonFiniteWeights,
                nam_json::JsonError::UnsupportedTopology { .. }
                | nam_json::JsonError::UnsupportedVersion { .. }
                | nam_json::JsonError::UnsupportedMultiChannel { .. }
                | nam_json::JsonError::SubmodelsTooDeep { .. }
                | nam_json::JsonError::TrainingTooDeep { .. }
                | nam_json::JsonError::SubmodelsExceedLimit { .. } => {
                    LoadError::UnsupportedArchitecture(e.to_string())
                }
                nam_json::JsonError::InvalidVersionFormat { raw } => {
                    LoadError::UnsupportedArchitecture(format!("Invalid version format: {}", raw))
                }
                nam_json::JsonError::InvalidSampleRate { .. } => {
                    LoadError::UnsupportedArchitecture(e.to_string())
                }
                nam_json::JsonError::Serde(_) => LoadError::Json(e),
            }
        })?;
        (data, file_size)
    } else {
        error!(
            "[Loader] Model build failed: file='{}', size={} bytes, code={:?}",
            path_str,
            0,
            NamErrorCode::UnknownExtension
        );
        return Err(LoadError::UnsupportedExtension(ext.to_string()));
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
    validate_metadata_floats(&meta, model_data.config.head_scale).map_err(|e| {
        error!(
            "[Loader] Model build failed: file='{}', size={} bytes, code={:?} — \
             invalid model metadata (detail: {}).",
            path_str,
            file_size,
            NamErrorCode::InvalidMetadata,
            e
        );
        LoadError::InvalidMetadata(e)
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
    let mut model_l = dispatcher::build_model(&model_data).map_err(|e| {
        let code = if let Some(&code) = e.downcast_ref::<NamErrorCode>() {
            code
        } else if e.to_string().contains("slimmable") {
            NamErrorCode::InvalidModelTopology
        } else {
            NamErrorCode::ModelBuildFailed
        };
        error!(
            "[Loader] Model build failed: file='{}', size={} bytes, code={:?} — \
             failed to build model (L) (detail: {}).",
            path_str, file_size, code, e
        );
        if e.to_string().contains("slimmable") {
            LoadError::UnsupportedArchitecture(e.to_string())
        } else {
            LoadError::ModelBuildFailed(e.to_string())
        }
    })?;

    // Pre-allocate scratch containers, crossfade buffers, and delay lines for
    // the engine's documented maximum block size (`MAX_RESAMP_BUF` = 8192 samples).
    // INVARIANT: Performing this sizing once off-RT ensures that the audio callback
    // never triggers buffer growth or memory reallocation on the RT hot-path.
    model_l
        .set_max_buffer_size(crate::dsp::pipeline::MAX_RESAMP_BUF)
        .map_err(|e| LoadError::Internal(e.to_string()))?;

    // Prewarm state history: flushes receptive fields and recurrent cell states to
    // steady-state before audio processing starts.
    // INVARIANT: Eliminates cold-start transients, pops, and DC step responses on
    // the first processed audio block.
    if options.prewarm == Some(false) {
        model_l.set_prewarm_on_reset(false);
    } else {
        model_l.prewarm(model_l.prewarm_samples().max(2048));
    }

    #[cfg(feature = "dual-mono")]
    let build_dual_mono = dual_mono;

    #[cfg(not(feature = "dual-mono"))]
    let build_dual_mono = {
        if dual_mono {
            log::warn!(
                "[Loader] Requested dual-mono processing but feature 'dual-mono' is disabled at compile time; right-channel model will not be instantiated (model_r = None)."
            );
        }
        false
    };

    // Dual-mono channel isolation: instantiates an independent model replica for the
    // right channel. Both channels run strictly disjoint internal states (ring buffers,
    // delay lines, recurrent memory) to guarantee zero stereo crosstalk.
    let model_r = if build_dual_mono {
        let mut model = dispatcher::build_model(&model_data).map_err(|e| {
            let code = if let Some(&code) = e.downcast_ref::<NamErrorCode>() {
                code
            } else if e.to_string().contains("slimmable") {
                NamErrorCode::InvalidModelTopology
            } else {
                NamErrorCode::ModelBuildFailed
            };
            error!(
                "[Loader] Model build failed: file='{}', size={} bytes, code={:?} — \
                 failed to build model (R) (detail: {}).",
                path_str, file_size, code, e
            );
            if e.to_string().contains("slimmable") {
                LoadError::UnsupportedArchitecture(e.to_string())
            } else {
                LoadError::ModelBuildFailed(e.to_string())
            }
        })?;
        model
            .set_max_buffer_size(crate::dsp::pipeline::MAX_RESAMP_BUF)
            .map_err(|e| LoadError::Internal(e.to_string()))?;
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
        "dual-mono"
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
