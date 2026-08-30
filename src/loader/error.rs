// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Strongly-typed public error for model loading and construction.

use super::loaded_model_pair::MetadataError;
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

    /// The `.nam` JSON data is malformed or violates the expected schema.
    #[error("Model JSON parse error: {0}")]
    JsonParse(#[source] serde_json::Error),

    /// The `.namb` file does not start with the expected magic bytes (`0x4E414D42`).
    #[error("Invalid .namb magic bytes: {0}")]
    NambInvalidMagic(String),

    /// The `.namb` file is truncated or has weight offsets exceeding file bounds.
    #[error("Truncated or corrupt .namb file: {0}")]
    NambTruncated(String),

    /// The `.namb` file failed CRC32 integrity verification (corruption or incomplete download).
    #[error("CRC32 integrity mismatch in .namb file")]
    NambCrc32Mismatch,

    /// CRC32 integrity field is missing in a `.namb` file (policy requires checksum verification).
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

    /// Failed to instantiate or initialize the neural network model graph.
    #[error("Failed to build model instance: {0}")]
    ModelBuildFailed(String),

    /// An unclassified internal loader error occurred.
    #[error("Internal loader error: {0}")]
    Internal(String),
}
