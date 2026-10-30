// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Cross-platform parity vector protocol and verification engine (R-07, R-13, S5-T01, S5-T02).
//!
//! Provides a strictly versioned, endian-explicit, bit-exact serialization format
//! and verification pipeline for neural inference outputs across target platforms
//! (x86-64-v3, aarch64 Linux, Apple Silicon macOS, and opt-in AVX-512).
//!
//! # Protocol Guarantees
//! - **Strict Binary Layout:** Magic `NAMPARV1`, 32-bit LE version, length-prefixed JSON
//!   metadata header, 32-byte SHA-256 payload checksum, and IEEE-754 `f32` sample array.
//! - **Bit-Exact Tracking:** Every sample is compared via `f32::to_bits()`. `+0.0` and `-0.0`
//!   have distinct bit representations and are reported as mismatches.
//! - **Non-Normalizing FP Policy:** NaNs and subnormals are never normalized or flushed
//!   silently during serialization or comparison. Raw bit patterns are preserved and reported.
//! - **Strict Compatibility Verification:** Different sample rates, block sizes, models, signals,
//!   or channel layouts are formally rejected as incompatible, never evaluated as equal.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{self, Read, Write};

/// 8-byte magic identifier for NeuralAmpModeler-rs parity vectors.
pub const PARITY_VECTOR_MAGIC: [u8; 8] = *b"NAMPARV1";

/// Current supported vector protocol format version.
pub const PARITY_VECTOR_VERSION: u32 = 1;

/// Operating DSP mode under which inference was captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DspMode {
    /// Live performance mode with minimum latency.
    Live,
    /// High-quality offline rendering mode.
    Hq,
}

/// Channel layout mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelMode {
    /// Single-channel mono processing.
    Mono,
    /// Dual-channel independent dual-mono processing.
    DualMono,
}

/// Initial state of the model before running benchmark/parity inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InitialState {
    /// Zero/cold initialized state.
    Cold,
    /// Prewarmed state with prior inference passes to stabilize recurrent state.
    Prewarmed,
}

/// Header describing complete provenance and test parameters of a parity vector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityVectorHeader {
    /// 8-byte magic signature (`NAMPARV1`).
    pub magic: [u8; 8],
    /// Protocol schema version.
    pub protocol_version: u32,
    /// Byte endianness of raw payload samples (`little`).
    pub endianness: String,
    /// Logical identifier or filename of the neural model.
    pub model_id: String,
    /// Hex-encoded SHA-256 hash of the `.nam` model file.
    pub model_sha256: String,
    /// Identifier of the test input signal.
    pub signal_name: String,
    /// Hex-encoded SHA-256 hash of the input signal samples.
    pub signal_sha256: String,
    /// Deterministic seed used to synthesize the input signal.
    pub signal_seed: u64,
    /// Total number of samples in the input test signal.
    pub signal_len: u64,
    /// DSP operational mode (Live or HQ).
    pub dsp_mode: DspMode,
    /// Audio channel layout mode (Mono or DualMono).
    pub channel_mode: ChannelMode,
    /// Audio sample rate in Hertz (e.g. 48000).
    pub sample_rate: u32,
    /// Audio block quantum in samples (e.g. 64).
    pub block_size: u32,
    /// Initial model state (Cold or Prewarmed).
    pub initial_state: InitialState,
    /// Number of audio channels (1 for Mono, 2 for DualMono).
    pub num_channels: u16,
    /// Number of samples recorded per channel.
    pub num_samples_per_channel: u64,
    /// Host compiler and toolchain identification.
    pub toolchain: String,
    /// Compilation flags and optimization profiles.
    pub compile_flags: String,
    /// Target triple of the compilation host.
    pub target_triple: String,
    /// Active instruction set architecture and SIMD extensions.
    pub instruction_set: String,
    /// Floating-point environment description (FTZ/DAZ, rounding modes).
    pub fp_env: String,
    /// Optional unique execution/run identifier for tracing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
}

/// Complete parity vector containing metadata and raw audio samples.
#[derive(Debug, Clone, PartialEq)]
pub struct ParityVector {
    /// Header containing execution metadata and parameters.
    pub header: ParityVectorHeader,
    /// Interleaved audio samples (`[s0_c0, s0_c1, ...]`).
    pub samples: Vec<f32>,
}

/// Errors occurring during parity vector serialization, deserialization, or parsing.
#[derive(Debug, thiserror::Error)]
pub enum VectorProtocolError {
    /// Underlying I/O error during stream reading or writing.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// Stream does not begin with the expected `NAMPARV1` magic header.
    #[error("Invalid magic: expected NAMPARV1, found {0:?}")]
    InvalidMagic([u8; 8]),
    /// Protocol version is unsupported by this engine build.
    #[error("Unsupported protocol version: {0} (expected {PARITY_VECTOR_VERSION})")]
    UnsupportedVersion(u32),
    /// Deserialization of the JSON header metadata failed.
    #[error("Header JSON decode error: {0}")]
    HeaderDecode(#[from] serde_json::Error),
    /// Raw sample stream was truncated before reading the expected sample count.
    #[error("Payload truncated: expected {expected} samples, got {actual}")]
    TruncatedPayload {
        /// Expected sample count.
        expected: usize,
        /// Actual sample count retrieved before EOF.
        actual: usize,
    },
    /// SHA-256 checksum of payload samples does not match header declaration.
    #[error("Payload checksum mismatch: data corrupted")]
    CorruptedPayloadChecksum,
    /// Encoded header length exceeds reasonable safety limits.
    #[error("Header size exceeds safe limit: {0} bytes")]
    HeaderTooLarge(u32),
}

/// Reasons why two parity vectors cannot be compared as equivalent runs.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VectorIncompatibilityReason {
    /// Vector protocol versions differ.
    #[error("Protocol version mismatch: ref={ref_val}, test={test_val}")]
    MismatchedProtocolVersion {
        /// Reference protocol version.
        ref_val: u32,
        /// Test protocol version.
        test_val: u32,
    },
    /// Model identifier names differ.
    #[error("Model ID mismatch: ref='{ref_val}', test='{test_val}'")]
    MismatchedModelId {
        /// Reference model ID.
        ref_val: String,
        /// Test model ID.
        test_val: String,
    },
    /// Model SHA-256 cryptographic digests differ.
    #[error("Model SHA-256 mismatch: ref={ref_val}, test={test_val}")]
    MismatchedModelSha256 {
        /// Reference model SHA-256.
        ref_val: String,
        /// Test model SHA-256.
        test_val: String,
    },
    /// Input signal names differ.
    #[error("Signal name mismatch: ref='{ref_val}', test='{test_val}'")]
    MismatchedSignalName {
        /// Reference signal name.
        ref_val: String,
        /// Test signal name.
        test_val: String,
    },
    /// Input signal SHA-256 digests differ.
    #[error("Signal SHA-256 mismatch: ref={ref_val}, test={test_val}")]
    MismatchedSignalSha256 {
        /// Reference signal SHA-256.
        ref_val: String,
        /// Test signal SHA-256.
        test_val: String,
    },
    /// Seed used to generate input signals differs.
    #[error("Signal seed mismatch: ref={ref_val}, test={test_val}")]
    MismatchedSignalSeed {
        /// Reference signal seed.
        ref_val: u64,
        /// Test signal seed.
        test_val: u64,
    },
    /// Input signal sample lengths differ.
    #[error("Signal length mismatch: ref={ref_val}, test={test_val}")]
    MismatchedSignalLength {
        /// Reference signal length.
        ref_val: u64,
        /// Test signal length.
        test_val: u64,
    },
    /// Operational DSP modes differ.
    #[error("DSP mode mismatch: ref={ref_val:?}, test={test_val:?}")]
    MismatchedDspMode {
        /// Reference DSP mode.
        ref_val: DspMode,
        /// Test DSP mode.
        test_val: DspMode,
    },
    /// Channel layout modes differ.
    #[error("Channel mode mismatch: ref={ref_val:?}, test={test_val:?}")]
    MismatchedChannelMode {
        /// Reference channel mode.
        ref_val: ChannelMode,
        /// Test channel mode.
        test_val: ChannelMode,
    },
    /// Total channel counts differ.
    #[error("Channel count mismatch: ref={ref_val}, test={test_val}")]
    MismatchedChannelCount {
        /// Reference channel count.
        ref_val: u16,
        /// Test channel count.
        test_val: u16,
    },
    /// Sampling rates differ.
    #[error("Sample rate mismatch: ref={ref_val} Hz, test={test_val} Hz")]
    MismatchedSampleRate {
        /// Reference sample rate.
        ref_val: u32,
        /// Test sample rate.
        test_val: u32,
    },
    /// Block quanta differ.
    #[error("Block size mismatch: ref={ref_val}, test={test_val}")]
    MismatchedBlockSize {
        /// Reference block size.
        ref_val: u32,
        /// Test block size.
        test_val: u32,
    },
    /// Initial model preparation states differ.
    #[error("Initial state mismatch: ref={ref_val:?}, test={test_val:?}")]
    MismatchedInitialState {
        /// Reference initial state.
        ref_val: InitialState,
        /// Test initial state.
        test_val: InitialState,
    },
    /// Total captured sample counts per channel differ.
    #[error("Sample count mismatch: ref={ref_val}, test={test_val}")]
    MismatchedSampleCount {
        /// Reference sample count per channel.
        ref_val: u64,
        /// Test sample count per channel.
        test_val: u64,
    },
}

/// Detailed diagnostic of the first bitwise mismatch detected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FirstMismatch {
    /// Sample index along the channel where divergence was detected.
    pub sample_index: usize,
    /// Channel index (0 for Mono/Left, 1 for Right).
    pub channel: usize,
    /// Raw integer bit representation of reference sample (`f32::to_bits()`).
    pub ref_bits: u32,
    /// Raw integer bit representation of test sample (`f32::to_bits()`).
    pub test_bits: u32,
    /// Formatted hexadecimal representation of reference bits.
    pub ref_bits_hex: String,
    /// Formatted hexadecimal representation of test bits.
    pub test_bits_hex: String,
    /// Float value of the reference sample.
    pub ref_value: f32,
    /// Float value of the test sample.
    pub test_value: f32,
    /// Absolute difference between float representations (`|ref - test|`).
    pub abs_diff: f32,
}

/// Comprehensive report of cross-platform parity comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorComparisonReport {
    /// Whether the vectors have matching test configurations and are comparable.
    pub is_compatible: bool,
    /// Detailed reason if vectors are incompatible.
    pub incompatibility_reason: Option<String>,
    /// Number of audio samples per channel in each vector.
    pub total_samples_per_channel: usize,
    /// Total number of channels.
    pub channels: usize,
    /// Total number of individual float samples compared across all channels.
    pub total_samples_compared: usize,
    /// Count of samples where bitwise IEEE-754 representations differ.
    pub mismatch_count: usize,
    /// True if and only if all compared samples share identical bit representations.
    pub bitwise_identical: bool,
    /// Diagnostic details of the first mismatch, if any.
    pub first_mismatch: Option<FirstMismatch>,
    /// Energy-to-Signal Ratio between reference and test outputs.
    pub esr: f64,
    /// Signal-to-Noise Ratio in decibels.
    pub snr_db: f64,
    /// Maximum absolute sample-wise difference observed across the buffer.
    pub max_abs_diff: f32,
    /// Mean Squared Error between vectors.
    pub mse: f64,
    /// Count of NaN values observed in reference vector.
    pub nan_count_ref: usize,
    /// Count of NaN values observed in test vector.
    pub nan_count_test: usize,
    /// Count of subnormal values observed in reference vector.
    pub subnormal_count_ref: usize,
    /// Count of subnormal values observed in test vector.
    pub subnormal_count_test: usize,
    /// Target triple where reference vector was generated.
    pub ref_target: String,
    /// Target triple where test vector was generated.
    pub test_target: String,
    /// Instruction set architecture used for reference execution.
    pub ref_isa: String,
    /// Instruction set architecture used for test execution.
    pub test_isa: String,
}

impl ParityVector {
    /// Serializes vector to binary stream according to the NAMPARV1 specification.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), VectorProtocolError> {
        writer.write_all(&self.header.magic)?;
        writer.write_all(&self.header.protocol_version.to_le_bytes())?;

        let header_json = serde_json::to_vec(&self.header)?;
        let json_len = header_json.len() as u32;
        writer.write_all(&json_len.to_le_bytes())?;
        writer.write_all(&header_json)?;

        // Compute SHA-256 of raw sample bytes
        let mut hasher = Sha256::new();
        for &s in &self.samples {
            hasher.update(s.to_bits().to_le_bytes());
        }
        let checksum: [u8; 32] = hasher.finalize().into();
        writer.write_all(&checksum)?;

        // Write raw Little-Endian samples
        for &s in &self.samples {
            writer.write_all(&s.to_bits().to_le_bytes())?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Deserializes vector from binary stream, validating checksum and integrity fail-closed.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, VectorProtocolError> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if magic != PARITY_VECTOR_MAGIC {
            return Err(VectorProtocolError::InvalidMagic(magic));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != PARITY_VECTOR_VERSION {
            return Err(VectorProtocolError::UnsupportedVersion(version));
        }

        let mut json_len_bytes = [0u8; 4];
        reader.read_exact(&mut json_len_bytes)?;
        let json_len = u32::from_le_bytes(json_len_bytes);
        const MAX_HEADER_SIZE: u32 = 1024 * 1024; // 1 MiB limit for header JSON
        if json_len > MAX_HEADER_SIZE {
            return Err(VectorProtocolError::HeaderTooLarge(json_len));
        }

        let mut header_json = vec![0u8; json_len as usize];
        reader.read_exact(&mut header_json)?;
        let header: ParityVectorHeader = serde_json::from_slice(&header_json)?;

        let mut expected_checksum = [0u8; 32];
        if let Err(e) = reader.read_exact(&mut expected_checksum) {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                return Err(VectorProtocolError::TruncatedPayload {
                    expected: 0,
                    actual: 0,
                });
            }
            return Err(VectorProtocolError::Io(e));
        }

        let total_samples = (header.num_channels as usize)
            .checked_mul(header.num_samples_per_channel as usize)
            .ok_or(VectorProtocolError::TruncatedPayload {
                expected: usize::MAX,
                actual: 0,
            })?;

        let mut samples = Vec::with_capacity(total_samples);
        let mut sample_bytes = [0u8; 4];
        let mut hasher = Sha256::new();

        for i in 0..total_samples {
            match reader.read_exact(&mut sample_bytes) {
                Ok(()) => {
                    hasher.update(sample_bytes);
                    let bits = u32::from_le_bytes(sample_bytes);
                    samples.push(f32::from_bits(bits));
                }
                Err(_) => {
                    return Err(VectorProtocolError::TruncatedPayload {
                        expected: total_samples,
                        actual: i,
                    });
                }
            }
        }

        let actual_checksum: [u8; 32] = hasher.finalize().into();
        if actual_checksum != expected_checksum {
            return Err(VectorProtocolError::CorruptedPayloadChecksum);
        }

        Ok(Self { header, samples })
    }
}

/// Verifies that two parity vectors have compatible test configurations.
pub fn check_compatibility(
    ref_hdr: &ParityVectorHeader,
    test_hdr: &ParityVectorHeader,
) -> Result<(), VectorIncompatibilityReason> {
    if ref_hdr.protocol_version != test_hdr.protocol_version {
        return Err(VectorIncompatibilityReason::MismatchedProtocolVersion {
            ref_val: ref_hdr.protocol_version,
            test_val: test_hdr.protocol_version,
        });
    }
    if ref_hdr.model_id != test_hdr.model_id {
        return Err(VectorIncompatibilityReason::MismatchedModelId {
            ref_val: ref_hdr.model_id.clone(),
            test_val: test_hdr.model_id.clone(),
        });
    }
    if ref_hdr.model_sha256 != test_hdr.model_sha256 {
        return Err(VectorIncompatibilityReason::MismatchedModelSha256 {
            ref_val: ref_hdr.model_sha256.clone(),
            test_val: test_hdr.model_sha256.clone(),
        });
    }
    if ref_hdr.signal_name != test_hdr.signal_name {
        return Err(VectorIncompatibilityReason::MismatchedSignalName {
            ref_val: ref_hdr.signal_name.clone(),
            test_val: test_hdr.signal_name.clone(),
        });
    }
    if ref_hdr.signal_sha256 != test_hdr.signal_sha256 {
        return Err(VectorIncompatibilityReason::MismatchedSignalSha256 {
            ref_val: ref_hdr.signal_sha256.clone(),
            test_val: test_hdr.signal_sha256.clone(),
        });
    }
    if ref_hdr.signal_seed != test_hdr.signal_seed {
        return Err(VectorIncompatibilityReason::MismatchedSignalSeed {
            ref_val: ref_hdr.signal_seed,
            test_val: test_hdr.signal_seed,
        });
    }
    if ref_hdr.signal_len != test_hdr.signal_len {
        return Err(VectorIncompatibilityReason::MismatchedSignalLength {
            ref_val: ref_hdr.signal_len,
            test_val: test_hdr.signal_len,
        });
    }
    if ref_hdr.dsp_mode != test_hdr.dsp_mode {
        return Err(VectorIncompatibilityReason::MismatchedDspMode {
            ref_val: ref_hdr.dsp_mode,
            test_val: test_hdr.dsp_mode,
        });
    }
    if ref_hdr.channel_mode != test_hdr.channel_mode {
        return Err(VectorIncompatibilityReason::MismatchedChannelMode {
            ref_val: ref_hdr.channel_mode,
            test_val: test_hdr.channel_mode,
        });
    }
    if ref_hdr.num_channels != test_hdr.num_channels {
        return Err(VectorIncompatibilityReason::MismatchedChannelCount {
            ref_val: ref_hdr.num_channels,
            test_val: test_hdr.num_channels,
        });
    }
    if ref_hdr.sample_rate != test_hdr.sample_rate {
        return Err(VectorIncompatibilityReason::MismatchedSampleRate {
            ref_val: ref_hdr.sample_rate,
            test_val: test_hdr.sample_rate,
        });
    }
    if ref_hdr.block_size != test_hdr.block_size {
        return Err(VectorIncompatibilityReason::MismatchedBlockSize {
            ref_val: ref_hdr.block_size,
            test_val: test_hdr.block_size,
        });
    }
    if ref_hdr.initial_state != test_hdr.initial_state {
        return Err(VectorIncompatibilityReason::MismatchedInitialState {
            ref_val: ref_hdr.initial_state,
            test_val: test_hdr.initial_state,
        });
    }
    if ref_hdr.num_samples_per_channel != test_hdr.num_samples_per_channel {
        return Err(VectorIncompatibilityReason::MismatchedSampleCount {
            ref_val: ref_hdr.num_samples_per_channel,
            test_val: test_hdr.num_samples_per_channel,
        });
    }

    Ok(())
}

/// Compares two parity vectors bitwise and numerically.
///
/// Returns a complete report including compatibility status, bitwise mismatch count,
/// first divergent bit details, ESR, SNR, MSE, and floating-point anomaly counts.
pub fn compare_vectors(reference: &ParityVector, test: &ParityVector) -> VectorComparisonReport {
    let channels = reference.header.num_channels as usize;
    let samples_per_channel = reference.header.num_samples_per_channel as usize;
    let total_samples = reference.samples.len();

    let ref_target = reference.header.target_triple.clone();
    let test_target = test.header.target_triple.clone();
    let ref_isa = reference.header.instruction_set.clone();
    let test_isa = test.header.instruction_set.clone();

    if let Err(incompat) = check_compatibility(&reference.header, &test.header) {
        return VectorComparisonReport {
            is_compatible: false,
            incompatibility_reason: Some(incompat.to_string()),
            total_samples_per_channel: samples_per_channel,
            channels,
            total_samples_compared: 0,
            mismatch_count: 0,
            bitwise_identical: false,
            first_mismatch: None,
            esr: f64::NAN,
            snr_db: f64::NAN,
            max_abs_diff: f32::NAN,
            mse: f64::NAN,
            nan_count_ref: 0,
            nan_count_test: 0,
            subnormal_count_ref: 0,
            subnormal_count_test: 0,
            ref_target,
            test_target,
            ref_isa,
            test_isa,
        };
    }

    let mut mismatch_count = 0;
    let mut first_mismatch = None;
    let mut sum_diff_sq = 0.0f64;
    let mut sum_ref_sq = 0.0f64;
    let mut max_abs_diff = 0.0f32;
    let mut nan_count_ref = 0;
    let mut nan_count_test = 0;
    let mut subnormal_count_ref = 0;
    let mut subnormal_count_test = 0;

    for (idx, (&r, &t)) in reference
        .samples
        .iter()
        .zip(test.samples.iter())
        .enumerate()
    {
        if r.is_nan() {
            nan_count_ref += 1;
        }
        if t.is_nan() {
            nan_count_test += 1;
        }
        if r.is_subnormal() {
            subnormal_count_ref += 1;
        }
        if t.is_subnormal() {
            subnormal_count_test += 1;
        }

        let r_bits = r.to_bits();
        let t_bits = t.to_bits();

        if r_bits != t_bits {
            let channel = if channels > 1 { idx % channels } else { 0 };
            let sample_index = if channels > 1 { idx / channels } else { idx };
            let abs_diff = (r - t).abs();

            if first_mismatch.is_none() {
                first_mismatch = Some(FirstMismatch {
                    sample_index,
                    channel,
                    ref_bits: r_bits,
                    test_bits: t_bits,
                    ref_bits_hex: format!("0x{r_bits:08X}"),
                    test_bits_hex: format!("0x{t_bits:08X}"),
                    ref_value: r,
                    test_value: t,
                    abs_diff,
                });
            }
            mismatch_count += 1;
        }

        let diff = (r as f64) - (t as f64);
        sum_diff_sq += diff * diff;
        sum_ref_sq += (r as f64) * (r as f64);
        let abs_diff = (r - t).abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
    }

    let n = total_samples.max(1) as f64;
    let mse = sum_diff_sq / n;
    let esr = if sum_ref_sq > 1e-12 {
        sum_diff_sq / sum_ref_sq
    } else if sum_diff_sq == 0.0 {
        0.0
    } else {
        1.0
    };

    let snr_db = if sum_diff_sq <= 1e-18 {
        f64::INFINITY
    } else if sum_ref_sq <= 1e-18 {
        -f64::INFINITY
    } else {
        10.0 * (sum_ref_sq / sum_diff_sq).log10()
    };

    let bitwise_identical = mismatch_count == 0 && nan_count_ref == 0 && nan_count_test == 0;

    VectorComparisonReport {
        is_compatible: true,
        incompatibility_reason: None,
        total_samples_per_channel: samples_per_channel,
        channels,
        total_samples_compared: total_samples,
        mismatch_count,
        bitwise_identical,
        first_mismatch,
        esr,
        snr_db,
        max_abs_diff,
        mse,
        nan_count_ref,
        nan_count_test,
        subnormal_count_ref,
        subnormal_count_test,
        ref_target,
        test_target,
        ref_isa,
        test_isa,
    }
}

#[cfg(test)]
#[path = "vector_protocol_test.rs"]
mod tests;
