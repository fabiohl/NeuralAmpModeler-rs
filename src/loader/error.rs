// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Strongly-typed public error for model loading and construction.

use super::loaded_model_pair::MetadataError;
use super::nam_json::JsonError;
use super::namb::NambError;
use super::wav::WavError;
use thiserror::Error;

/// Public error returned by [`load_and_build_model`](crate::loader::load_and_build_model)
/// and related loader facilities.
///
/// Each variant represents a distinct failure domain, enabling library consumers
/// (DAWs, audio hosts, plugins) to implement specialized recovery, user diagnostics,
/// and fallback logic without inspecting formatted strings or downcasting `anyhow::Error`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LoadError {
    /// The model file could not be opened or read (e.g. missing file, permission denied).
    #[error("I/O error reading model file: {0}")]
    Io(#[from] std::io::Error),

    /// The model file contains invalid UTF-8 data (`.nam` files must be valid UTF-8).
    #[error("Invalid UTF-8 encoding in model file: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    /// Structured `.namb` binary format error.
    #[error("NAMB binary format error: {0}")]
    Namb(#[from] NambError),

    /// Structured `.nam` JSON format error.
    #[error("NAM JSON format error: {0}")]
    Json(#[from] JsonError),

    /// Structured `.wav` impulse response format error.
    #[error("WAV impulse response format error: {0}")]
    Wav(#[from] WavError),

    /// The `.nam` JSON data is malformed or violates the expected schema.
    #[deprecated(since = "0.8.0", note = "use LoadError::Json instead")]
    #[error("Model JSON parse error: {0}")]
    JsonParse(#[source] serde_json::Error),

    /// The `.namb` file does not start with the expected magic bytes (`0x4E414D42`).
    #[deprecated(
        since = "0.8.0",
        note = "use LoadError::Namb(NambError::InvalidMagic) instead"
    )]
    #[error("Invalid .namb magic bytes: {0}")]
    NambInvalidMagic(String),

    /// The `.namb` file is truncated or has weight offsets exceeding file bounds.
    #[deprecated(
        since = "0.8.0",
        note = "use LoadError::Namb(NambError::Truncated { .. }) instead"
    )]
    #[error("Truncated or corrupt .namb file: {0}")]
    NambTruncated(String),

    /// The `.namb` file failed CRC32 integrity verification (corruption or incomplete download).
    #[deprecated(
        since = "0.8.0",
        note = "use LoadError::Namb(NambError::CrcMismatch { .. }) instead"
    )]
    #[error("CRC32 integrity mismatch in .namb file")]
    NambCrc32Mismatch,

    /// CRC32 integrity field is missing in a `.namb` file (policy requires checksum verification).
    #[deprecated(
        since = "0.8.0",
        note = "use LoadError::Namb(NambError::CrcMissing { .. }) instead"
    )]
    #[error("CRC32 integrity field missing in .namb file")]
    NambCrc32Missing,

    /// The file extension is not supported (only `.nam` and `.namb` are valid).
    #[error("Unsupported model file extension: {0}")]
    UnsupportedExtension(String),

    /// The model architecture or topology version is not supported by this engine build.
    #[error("Unsupported model architecture or topology: {0}")]
    UnsupportedArchitecture(String),

    /// The tensor dimensions declared in the file are inconsistent with weight counts.
    #[error("Inconsistent model dimensions: {0}")]
    DimensionMismatch(String),

    /// The model contains non-finite weights (`NaN` or `Inf`).
    #[error("Non-finite weights detected in model (NaN or Inf)")]
    NonFiniteWeights,

    /// The model file exceeds the maximum allowed size limit.
    #[error("Model file exceeds maximum allowed size")]
    ModelTooLarge,

    /// The model metadata contains invalid or hostile float values.
    #[error("Invalid model metadata: {0}")]
    InvalidMetadata(#[from] MetadataError),

    /// The file carries no usable `metadata` object (e.g. absent, JSON `null`,
    /// or outside the readable section for `.namb`).
    #[error("Model file contains no '{format}' metadata section")]
    MetadataNotFound {
        /// Source format the peek targeted (`"nam"` or `"namb"`).
        format: &'static str,
    },

    /// Model topology validation failed or requested unsupported feature topology.
    #[error("Invalid model topology ({code:?}): {reason}")]
    InvalidTopology {
        /// Diagnostic error code.
        code: crate::common::diagnostics::NamErrorCode,
        /// Detailed reason.
        reason: String,
    },

    /// Failed to instantiate or initialize the neural network model graph.
    #[error("Failed to build model instance: {0}")]
    ModelBuildFailed(String),

    /// An unclassified internal loader error occurred.
    #[error("Internal loader error: {0}")]
    Internal(String),
}

impl LoadError {
    /// Returns the typed diagnostic [`NamErrorCode`](crate::common::diagnostics::NamErrorCode) associated with this load failure.
    pub fn code(&self) -> crate::common::diagnostics::NamErrorCode {
        use crate::common::diagnostics::NamErrorCode;

        match self {
            Self::Io(_) => NamErrorCode::FileReadError,
            Self::InvalidUtf8(_) => NamErrorCode::NamJsonParseError,
            Self::Namb(e) => match e {
                NambError::Truncated { .. } => NamErrorCode::NambTruncated,
                NambError::InvalidMagic(_) => NamErrorCode::NambInvalidMagic,
                NambError::InvalidVersion(_) => NamErrorCode::NambUnsupportedVersion,
                NambError::WeightsOffsetOutOfBounds { .. }
                | NambError::InvalidWeightsOffset { .. }
                | NambError::InvalidHeaderField { .. } => NamErrorCode::NambInvalidHeaderField,
                NambError::CrcMismatch { .. } => NamErrorCode::NambCrc32Mismatch,
                NambError::CrcMissing { .. } | NambError::CrcMissingV1 => {
                    NamErrorCode::NambCrc32Missing
                }
                NambError::WeightsTooLarge { .. } => NamErrorCode::ModelTooLarge,
                NambError::NonFiniteWeight { .. } => NamErrorCode::NambNonFiniteWeight,
                NambError::MetadataNotUtf8 { .. } | NambError::MetadataJson(_) => {
                    NamErrorCode::NamJsonParseError
                }
            },
            Self::Json(e) => match e {
                JsonError::WeightsExceedLimit { .. } => NamErrorCode::NamJsonWeightsExceedLimit,
                JsonError::TrainingTooDeep { .. } => NamErrorCode::NamJsonTrainingTooDeep,
                JsonError::TrainingTooLarge { .. } => NamErrorCode::NamJsonTrainingTooLarge,
                JsonError::SubmodelsExceedLimit { .. } => NamErrorCode::NamJsonSubmodelsExceedLimit,
                JsonError::SubmodelsTooDeep { .. } => NamErrorCode::NamJsonSubmodelsTooDeep,
                JsonError::WeightNotFinite { .. } => NamErrorCode::NamJsonWeightNotFinite,
                JsonError::InvalidSampleRate { .. } => NamErrorCode::NamJsonInvalidSampleRate,
                JsonError::UnsupportedTopology { .. } => NamErrorCode::NamJsonUnsupportedTopology,
                JsonError::InvalidVersionFormat { .. } => NamErrorCode::NamJsonInvalidVersionFormat,
                JsonError::UnsupportedVersion { .. } => NamErrorCode::NamJsonUnsupportedVersion,
                JsonError::UnsupportedMultiChannel { .. } => {
                    NamErrorCode::NamJsonUnsupportedMultiChannel
                }
                JsonError::InvalidEnvelope { .. } => NamErrorCode::NamJsonInvalidEnvelope,
                JsonError::SequentialChildrenExceedLimit { .. } => {
                    NamErrorCode::SequentialChildrenExceedLimit
                }
                JsonError::Serde(_) => NamErrorCode::NamJsonParseError,
            },
            Self::Wav(e) => e.error_code(),
            #[expect(deprecated)]
            Self::JsonParse(_) => NamErrorCode::NamJsonParseError,
            #[expect(deprecated)]
            Self::NambInvalidMagic(_) => NamErrorCode::NambInvalidMagic,
            #[expect(deprecated)]
            Self::NambTruncated(_) => NamErrorCode::NambTruncated,
            #[expect(deprecated)]
            Self::NambCrc32Mismatch => NamErrorCode::NambCrc32Mismatch,
            #[expect(deprecated)]
            Self::NambCrc32Missing => NamErrorCode::NambCrc32Missing,
            Self::UnsupportedExtension(_) => NamErrorCode::UnknownExtension,
            Self::UnsupportedArchitecture(_) => NamErrorCode::UnsupportedArchitecture,
            Self::DimensionMismatch(_) => NamErrorCode::WeightCountMismatch,
            Self::NonFiniteWeights => NamErrorCode::NamJsonWeightNotFinite,
            Self::ModelTooLarge => NamErrorCode::ModelTooLarge,
            Self::InvalidMetadata(_) => NamErrorCode::InvalidMetadata,
            Self::MetadataNotFound { .. } => NamErrorCode::InvalidMetadata,
            Self::InvalidTopology { code, .. } => *code,
            Self::ModelBuildFailed(_) => NamErrorCode::ModelBuildFailed,
            Self::Internal(_) => NamErrorCode::ModelBuildFailed,
        }
    }
}
