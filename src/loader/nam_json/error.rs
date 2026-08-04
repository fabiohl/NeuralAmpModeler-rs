// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Typed errors of the `.nam` JSON parser.

/// Typed errors of the `.nam` JSON parser.
#[derive(Debug, PartialEq)]
pub enum JsonError {
    /// The `weights` array exceeds the float limit.
    WeightsExceedLimit {
        /// Number of floats received.
        got: usize,
        /// Maximum configured limit.
        max: usize,
    },
    /// The `metadata.training` field exceeds the JSON tree depth limit.
    TrainingTooDeep {
        /// Depth found.
        depth: usize,
        /// Maximum allowed depth.
        max_depth: usize,
    },
    /// The `metadata.training` field exceeds the size limit.
    TrainingTooLarge {
        /// Approximate size in bytes.
        size: usize,
        /// Maximum allowed size.
        max_size: usize,
    },
    /// The `submodels` array exceeds the count limit.
    SubmodelsExceedLimit {
        /// Number of submodels received.
        got: usize,
        /// Maximum allowed submodels.
        max: usize,
    },
    /// Container recursion depth exceeded (max depth enforced by dispatcher).
    SubmodelsTooDeep {
        /// Current recursion depth.
        depth: usize,
        /// Maximum allowed depth.
        max_depth: usize,
    },
    /// A weight value is non-finite (NaN, +Inf, or -Inf).
    WeightNotFinite {
        /// Index (0-based) of the non-finite float in the weights array.
        index: usize,
        /// The non-finite value.
        value: f32,
    },
    /// The sample rate is invalid (non-finite or <= 0.0).
    InvalidSampleRate {
        /// The invalid sample rate value.
        value: f32,
        /// Reason why it is invalid.
        reason: &'static str,
    },
    /// The topology specified in config exceeds OOM safety limits (MAX_LAYERS or MAX_HIDDEN_SIZE).
    UnsupportedTopology {
        /// The declared architecture.
        architecture: String,
        /// Description of the exceeded limit.
        issue: String,
        /// The specific limit that was exceeded.
        limit: usize,
    },
    /// The version string cannot be parsed as SemVer.
    InvalidVersionFormat {
        /// The raw version string that failed to parse.
        raw: String,
    },
    /// The version is outside the supported range (min 0.5.0, max 0.7.x).
    UnsupportedVersion {
        /// The raw version string.
        raw: String,
        /// Human-readable explanation.
        reason: String,
    },
    /// LSTM model has multi-channel I/O, which NAM-rs does not support
    /// (C++ NAMcore supports arbitrary channels but no known .nam model uses them).
    UnsupportedMultiChannel {
        /// The architecture for context.
        architecture: String,
        /// Which channel field contains the non-mono value ("in_channels" or "out_channels").
        field: &'static str,
        /// The non-mono channel value found.
        value: usize,
    },
    /// Generic serde_json parse error.
    Serde(String),
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WeightsExceedLimit { got, max } => {
                write!(
                    f,
                    "weights array exceeds limit ({} floats, max is {})",
                    got, max
                )
            }
            Self::TrainingTooDeep { depth, max_depth } => {
                write!(
                    f,
                    "metadata.training JSON tree too deep (depth {}, max is {})",
                    depth, max_depth
                )
            }
            Self::TrainingTooLarge { size, max_size } => {
                write!(
                    f,
                    "metadata.training JSON too large ({} bytes, max is {} bytes)",
                    size, max_size
                )
            }
            Self::SubmodelsExceedLimit { got, max } => {
                write!(
                    f,
                    "submodels array exceeds limit ({} submodels, max is {})",
                    got, max
                )
            }
            Self::SubmodelsTooDeep { depth, max_depth } => {
                write!(
                    f,
                    "container nesting too deep (depth {}, max is {})",
                    depth, max_depth
                )
            }
            Self::WeightNotFinite { index, value } => {
                write!(
                    f,
                    "weight at index {} is not finite (value: {:e})",
                    index, value
                )
            }
            Self::InvalidSampleRate { value, reason } => {
                write!(
                    f,
                    "sample rate is invalid (value: {}, reason: {})",
                    value, reason
                )
            }
            Self::UnsupportedTopology {
                architecture,
                issue,
                limit,
            } => {
                write!(
                    f,
                    "unsupported topology: architecture={} — {} (limit={})",
                    architecture, issue, limit
                )
            }
            Self::InvalidVersionFormat { raw } => {
                write!(f, "invalid version format: '{raw}' is not valid SemVer")
            }
            Self::UnsupportedVersion { raw, reason } => {
                write!(f, "unsupported version '{raw}': {reason}")
            }
            Self::UnsupportedMultiChannel {
                architecture,
                field,
                value,
            } => {
                write!(
                    f,
                    "{architecture} {field}={value} is not supported — NAM-rs only supports mono models. \
                     C++ NAMcore accepts multi-channel LSTM but no known production .nam model uses this feature."
                )
            }
            Self::Serde(msg) => write!(f, "JSON parse error: {}", msg),
        }
    }
}

impl std::error::Error for JsonError {}

impl From<serde_json::Error> for JsonError {
    fn from(e: serde_json::Error) -> Self {
        JsonError::Serde(e.to_string())
    }
}
