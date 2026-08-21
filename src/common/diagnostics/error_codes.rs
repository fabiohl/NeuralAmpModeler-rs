// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Typed error codes for precise failure triage.
//!
//! Numeric code convention:
//! - **E1xxx**: Model loading (I/O, parse, build)
//! - **E2xxx**: Audio backend / processing (init, stream, resampler, RT)
//! - **E3xxx**: SPSC / inter-thread communication
//! - **E4xxx**: Runtime / CLI (parsing, commands)
//! - **E5xxx**: System / hardware (CPU features, memory)
//!
//! ### Range reservation
//!
//! The **E2xxx–E4xxx** ranges are **reserved for downstream integrations**
//! (e.g., plugin wrappers, standalone hosts, offline renderers). They are declared here so the range
//! allocation is a single source of truth, but this host-agnostic core crate
//! never constructs them: the core library does not own an audio backend, an
//! SPSC host loop, or a CLI, and must not assume one. Downstream crates emit
//! these codes through the same diagnostic engine when wiring the core into
//! their own host surfaces. Changes to this reservation must be synchronized
//! with `docs/architecture.md` (§7).

use std::fmt;

/// Structured error code for precise failure triage.
///
/// Each variant maps to exactly one failure scenario. The numeric code
/// (used in the support block) follows the convention:
/// - **E1xxx**: Model loading (I/O, parse, build)
/// - **E2xxx**: Audio backend / processing (init, stream, resampler, RT)
/// - **E3xxx**: SPSC / inter-thread communication
/// - **E4xxx**: Runtime / CLI (parsing, commands)
/// - **E5xxx**: System / hardware (CPU features, memory)
///
/// The **E2xxx–E4xxx** ranges are reserved for downstream integrations
/// (e.g., plugin wrappers, standalone hosts, offline renderers) and are never constructed by this
/// host-agnostic core crate. See the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamErrorCode {
    // E1xxx — Model Loading
    /// Model file not found on the filesystem.
    FileNotFound,
    /// I/O read error while accessing the model file.
    FileReadError,
    /// Unknown file extension (expected .nam or .namb).
    UnknownExtension,
    /// Failed to parse JSON in .nam format.
    NamJsonParseError,
    /// JSON `weights` array exceeds float limit (MAX_WEIGHTS).
    NamJsonWeightsExceedLimit,
    /// JSON `metadata.training` field exceeds size limit (1 MiB).
    NamJsonTrainingTooLarge,
    /// JSON `metadata.training` field exceeds tree depth limit.
    NamJsonTrainingTooDeep,
    /// JSON `config.submodels` array exceeds the maximum count (8).
    NamJsonSubmodelsExceedLimit,
    /// JSON `config.submodels` exceeds container recursion depth limit.
    NamJsonSubmodelsTooDeep,
    /// JSON weight value is non-finite (NaN, +Inf, -Inf).
    NamJsonWeightNotFinite,
    /// JSON `sample_rate` field contains an invalid value (non-finite or <= 0).
    NamJsonInvalidSampleRate,
    /// JSON topology exceeds OOM safety limits (MAX_LAYERS or MAX_HIDDEN_SIZE).
    NamJsonUnsupportedTopology,
    /// JSON version string is not parseable as SemVer.
    NamJsonInvalidVersionFormat,
    /// JSON version is outside the supported range (min 0.5.0, max 0.7.x).
    NamJsonUnsupportedVersion,
    /// LSTM model has multi-channel I/O (NAM-rs supports mono only).
    NamJsonUnsupportedMultiChannel,
    /// Model metadata float is non-finite or outside the plausible range
    /// (`input_level_dbu`, `output_level_dbu`, `loudness`, `head_scale`).
    InvalidMetadata,
    /// NAMB non-finite weight detected in binary weight section.
    NambNonFiniteWeight,
    /// NAMB header field contains an invalid value (non-finite, <= 0, etc.).
    NambInvalidHeaderField,
    /// .namb CRC32 checksum does not match expected value.
    NambCrc32Mismatch,
    /// CRC32 integrity metadata missing (.namb v2+ flag absent, or v1 `crc32 == 0` sentinel).
    NambCrc32Missing,
    /// Invalid .namb magic signature.
    NambInvalidMagic,
    /// Unsupported .namb format version.
    NambUnsupportedVersion,
    /// Truncated .namb file (smaller than the minimum header).
    NambTruncated,
    /// Declared neural architecture is not supported (neither WaveNet nor LSTM).
    UnsupportedArchitecture,
    /// WaveNet/LSTM topology not detectable from the configuration.
    TopologyDetectionFailed,
    /// Weight count inconsistent with network geometry.
    WeightCountMismatch,
    /// Generic failure building the model via the dispatcher.
    ModelBuildFailed,
    /// Model file exceeds the maximum allowed size (256 MiB).
    ModelTooLarge,
    /// Slimmable metadata is invalid or inconsistent (channel mismatch, empty allowed list, etc.).
    InvalidModelTopology,

    // E2xxx — Audio Backend / Processing
    // Reserved for downstream integrations (e.g., plugin wrappers, standalone hosts):
    // this host-agnostic core crate never constructs these variants.
    /// Failed to initialize the audio backend (context/core).
    AudioInitFailed,
    /// Failed to connect the audio stream.
    StreamError,
    /// Failed to build the NamResampler (native Polyphase Sinc).
    ResamplerBuildFailed,
    /// SPSC resampler channel full (rebuild discarded).
    ResamplerChannelFull,
    /// Permission denied for real-time scheduling (no RT privileges).
    RtPriorityDenied,
    /// Failed to set CPU affinity on the DSP thread.
    CpuAffinityFailed,
    /// Audio processing exceeded the deadline budget.
    ProcessingOverload,

    // E3xxx — SPSC / Communication
    // Reserved for downstream integrations (e.g., plugin wrappers, standalone hosts):
    // this host-agnostic core crate never constructs these variants.
    /// CLI→DSP parameter SPSC channel full.
    ParamChannelFull,

    // E4xxx — Runtime / CLI
    // Reserved for downstream integrations (e.g., CLI tools, plugin wrappers):
    // this host-agnostic core crate never constructs these variants.
    /// Invalid gain value (not a valid f32 number).
    InvalidGainValue,
    /// Unknown CLI command.
    UnknownCommand,
    /// Failed to configure the Ctrl-C handler.
    CtrlCHandlerFailed,
    /// Failed to load a cab-sim impulse response (WAV file).
    IrLoadFailed,
    /// Overflow detected in the Garbage Collection (GC) channel.
    GcOverflow,
    /// Corrupted GC overflow buffer slot (inconsistent type/pointer).
    GcCorrupted,
    /// Memory allocation failed (OOM) — layout overflow or allocator exhaustion.
    OutOfMemory,
    /// Host CPU lacks required x86-64-v3 feature set (AVX2/FMA).
    UnsupportedCpuArchitecture,
}

/// Convenience type alias for `NamErrorCode`.
pub type NamError = NamErrorCode;

impl NamErrorCode {
    /// Returns the numeric code in "Exxxx" format.
    pub fn code(self) -> &'static str {
        match self {
            Self::FileNotFound => "E1100",
            Self::FileReadError => "E1101",
            Self::UnknownExtension => "E1102",
            Self::NamJsonParseError => "E1200",
            Self::NamJsonWeightsExceedLimit => "E1206",
            Self::NamJsonTrainingTooLarge => "E1207",
            Self::NamJsonTrainingTooDeep => "E1208",
            Self::NamJsonSubmodelsExceedLimit => "E1209",
            Self::NamJsonSubmodelsTooDeep => "E1210",
            Self::NamJsonWeightNotFinite => "E1211",
            Self::NamJsonInvalidSampleRate => "E1214",
            Self::NamJsonUnsupportedTopology => "E1215",
            Self::NamJsonInvalidVersionFormat => "E1216",
            Self::NamJsonUnsupportedVersion => "E1217",
            Self::NamJsonUnsupportedMultiChannel => "E1218",
            Self::InvalidMetadata => "E1219",
            Self::NambNonFiniteWeight => "E1212",
            Self::NambInvalidHeaderField => "E1213",
            Self::NambCrc32Mismatch => "E1201",
            Self::NambCrc32Missing => "E1205",
            Self::NambInvalidMagic => "E1202",
            Self::NambUnsupportedVersion => "E1203",
            Self::NambTruncated => "E1204",
            Self::UnsupportedArchitecture => "E1300",
            Self::TopologyDetectionFailed => "E1301",
            Self::WeightCountMismatch => "E1302",
            Self::ModelBuildFailed => "E1303",
            Self::ModelTooLarge => "E1304",
            Self::InvalidModelTopology => "E1305",
            Self::AudioInitFailed => "E2100",
            Self::StreamError => "E2101",
            Self::ResamplerBuildFailed => "E2200",
            Self::ResamplerChannelFull => "E2201",
            Self::RtPriorityDenied => "E2300",
            Self::CpuAffinityFailed => "E2301",
            Self::ProcessingOverload => "E2001",
            Self::ParamChannelFull => "E3100",
            Self::GcOverflow => "E3101",
            Self::GcCorrupted => "E3102",
            Self::InvalidGainValue => "E4100",
            Self::UnknownCommand => "E4101",
            Self::CtrlCHandlerFailed => "E4102",
            Self::IrLoadFailed => "E4103",
            Self::OutOfMemory => "E5000",
            Self::UnsupportedCpuArchitecture => "E5001",
        }
    }

    /// Returns a human-readable message for GUI display.
    pub fn message(self) -> &'static str {
        match self {
            Self::FileNotFound => "File not found",
            Self::FileReadError => "File read error",
            Self::UnknownExtension => "Unknown extension",
            Self::NamJsonParseError => "Invalid JSON format",
            Self::NamJsonWeightsExceedLimit => "JSON weights exceed limit",
            Self::NamJsonTrainingTooLarge => "Training metadata too large",
            Self::NamJsonTrainingTooDeep => "Training metadata too deeply nested",
            Self::NamJsonSubmodelsExceedLimit => "Submodels count exceeds limit (max 8)",
            Self::NamJsonSubmodelsTooDeep => "Container nesting too deep",
            Self::NamJsonWeightNotFinite => "JSON weight is non-finite (NaN/Inf)",
            Self::NamJsonInvalidSampleRate => {
                "JSON sample_rate is invalid (must be finite and > 0)"
            }
            Self::NamJsonUnsupportedTopology => {
                "JSON topology exceeds OOM safety limits (MAX_LAYERS or MAX_HIDDEN_SIZE)"
            }
            Self::NamJsonInvalidVersionFormat => "JSON version string is not parseable as SemVer",
            Self::NamJsonUnsupportedVersion => {
                "JSON version is outside the supported range (min 0.5.0, max 0.7.x)"
            }
            Self::NamJsonUnsupportedMultiChannel => {
                "LSTM model has multi-channel I/O (only mono supported)"
            }
            Self::InvalidMetadata => {
                "Model metadata float is non-finite or outside the plausible range"
            }
            Self::NambNonFiniteWeight => "NAMB weight is non-finite (NaN/Inf)",
            Self::NambInvalidHeaderField => "NAMB header field is invalid",
            Self::NambCrc32Mismatch => "CRC32 checksum mismatch",
            Self::NambCrc32Missing => "CRC32 integrity metadata missing",
            Self::NambInvalidMagic => "Invalid signature",
            Self::NambUnsupportedVersion => "Unsupported version",
            Self::NambTruncated => "Corrupted/truncated file",
            Self::UnsupportedArchitecture => "Unsupported architecture",
            Self::TopologyDetectionFailed => "Topology detection failed",
            Self::WeightCountMismatch => "Weight count mismatch",
            Self::ModelBuildFailed => "Model build failed",
            Self::ModelTooLarge => "Model file too large",
            Self::InvalidModelTopology => "Invalid slimmable model topology",
            Self::AudioInitFailed => "Audio backend initialization failed",
            Self::StreamError => "Audio stream error",
            Self::ResamplerBuildFailed => "Resampler build failed",
            Self::ResamplerChannelFull => "Resampler channel full",
            Self::RtPriorityDenied => "Real-time priority denied",
            Self::CpuAffinityFailed => "CPU affinity setting failed",
            Self::ProcessingOverload => "Processing deadline exceeded",
            Self::ParamChannelFull => "Parameter channel full",
            Self::GcOverflow => "Garbage collection overflow",
            Self::GcCorrupted => "Garbage collection corruption",
            Self::InvalidGainValue => "Invalid gain value",
            Self::UnknownCommand => "Unknown command",
            Self::CtrlCHandlerFailed => "Ctrl-C handler setup failed",
            Self::IrLoadFailed => "Cab-sim IR load failed",
            Self::OutOfMemory => "Out of memory",
            Self::UnsupportedCpuArchitecture => {
                "Host CPU lacks required x86-64-v3 feature set (AVX2/FMA)"
            }
        }
    }

    /// Returns the human-readable mnemonic name (SCREAMING_SNAKE_CASE) of the code.
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::FileNotFound => "FILE_NOT_FOUND",
            Self::FileReadError => "FILE_READ_ERROR",
            Self::UnknownExtension => "UNKNOWN_EXTENSION",
            Self::NamJsonParseError => "NAM_JSON_PARSE_ERROR",
            Self::NamJsonWeightsExceedLimit => "NAM_JSON_WEIGHTS_EXCEED_LIMIT",
            Self::NamJsonTrainingTooLarge => "NAM_JSON_TRAINING_TOO_LARGE",
            Self::NamJsonTrainingTooDeep => "NAM_JSON_TRAINING_TOO_DEEP",
            Self::NamJsonSubmodelsExceedLimit => "NAM_JSON_SUBMODELS_EXCEED_LIMIT",
            Self::NamJsonSubmodelsTooDeep => "NAM_JSON_SUBMODELS_TOO_DEEP",
            Self::NamJsonWeightNotFinite => "NAM_JSON_WEIGHT_NOT_FINITE",
            Self::NamJsonInvalidSampleRate => "NAM_JSON_INVALID_SAMPLE_RATE",
            Self::NamJsonUnsupportedTopology => "NAM_JSON_UNSUPPORTED_TOPOLOGY",
            Self::NamJsonInvalidVersionFormat => "NAM_JSON_INVALID_VERSION_FORMAT",
            Self::NamJsonUnsupportedVersion => "NAM_JSON_UNSUPPORTED_VERSION",
            Self::NamJsonUnsupportedMultiChannel => "NAM_JSON_UNSUPPORTED_MULTI_CHANNEL",
            Self::InvalidMetadata => "INVALID_METADATA",
            Self::NambNonFiniteWeight => "NAMB_NON_FINITE_WEIGHT",
            Self::NambInvalidHeaderField => "NAMB_INVALID_HEADER_FIELD",
            Self::NambCrc32Mismatch => "NAMB_CRC32_MISMATCH",
            Self::NambCrc32Missing => "NAMB_CRC32_MISSING",
            Self::NambInvalidMagic => "NAMB_INVALID_MAGIC",
            Self::NambUnsupportedVersion => "NAMB_UNSUPPORTED_VERSION",
            Self::NambTruncated => "NAMB_TRUNCATED",
            Self::UnsupportedArchitecture => "UNSUPPORTED_ARCHITECTURE",
            Self::TopologyDetectionFailed => "TOPOLOGY_DETECTION_FAILED",
            Self::WeightCountMismatch => "WEIGHT_COUNT_MISMATCH",
            Self::ModelBuildFailed => "MODEL_BUILD_FAILED",
            Self::ModelTooLarge => "MODEL_TOO_LARGE",
            Self::InvalidModelTopology => "INVALID_MODEL_TOPOLOGY",
            Self::AudioInitFailed => "AUDIO_INIT_FAILED",
            Self::StreamError => "STREAM_ERROR",
            Self::ResamplerBuildFailed => "RESAMPLER_BUILD_FAILED",
            Self::ResamplerChannelFull => "RESAMPLER_CHANNEL_FULL",
            Self::RtPriorityDenied => "RT_PRIORITY_DENIED",
            Self::CpuAffinityFailed => "CPU_AFFINITY_FAILED",
            Self::ProcessingOverload => "PROCESSING_OVERLOAD",
            Self::ParamChannelFull => "PARAM_CHANNEL_FULL",
            Self::GcOverflow => "GC_OVERFLOW",
            Self::GcCorrupted => "GC_CORRUPTED",
            Self::InvalidGainValue => "INVALID_GAIN_VALUE",
            Self::UnknownCommand => "UNKNOWN_COMMAND",
            Self::CtrlCHandlerFailed => "CTRL_C_HANDLER_FAILED",
            Self::IrLoadFailed => "IR_LOAD_FAILED",
            Self::OutOfMemory => "OUT_OF_MEMORY",
            Self::UnsupportedCpuArchitecture => "UNSUPPORTED_CPU_ARCHITECTURE",
        }
    }
}

impl std::error::Error for NamErrorCode {}

impl fmt::Display for NamErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{} {}] {}",
            self.code(),
            self.mnemonic(),
            self.message()
        )
    }
}
