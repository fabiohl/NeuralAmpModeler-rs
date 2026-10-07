// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Strict RIFF/WAVE impulse response parser and converter to Linear model architecture.
//!
//! Provides a safe, allocation-bounded parser for mono and stereo WAV audio files
//! intended for use as impulse response (IR) models in the NAM engine.
//!
//! Formats supported:
//! - PCM 16-bit, 24-bit, 32-bit (signed integer, scaled by `2^(bits-1)`)
//! - IEEE Float 32-bit
//! - `WAVE_FORMAT_EXTENSIBLE` (format code 65534 / `0xFFFE`) with standard PCM / Float GUIDs
//! - Mono (1 channel) and Stereo (2 channels)
//!
//! Invariants:
//! - Zero `unsafe` code.
//! - Bounds checks before any buffer allocation (DoS / OOM protection).
//! - Transposition from interleaved WAV frames to contiguous per-channel Linear weights.
//! - Conversion to [`crate::loader::nam_json::NamModelData`] with `architecture = "Linear"`.

use crate::common::diagnostics::NamErrorCode;
use crate::loader::nam_json::{NamConfig, NamModelData, WeightsLayout};
use log::info;

/// Strongly-typed error variants for WAV impulse response parsing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum WavError {
    /// File size is smaller than the minimum 12-byte RIFF/WAVE header.
    #[error("Missing RIFF/WAVE header (file too small)")]
    MissingHeader,

    /// Missing "RIFF" or "WAVE" four-character code identifiers.
    #[error("Expected RIFF/WAVE identifier")]
    ExpectedRiffWave,

    /// Declared RIFF chunk size is invalid or exceeds file boundaries.
    #[error("Invalid RIFF size: declared end {0} exceeds available byte buffer")]
    InvalidRiffSize(u64),

    /// Incomplete chunk header encountered before end of RIFF container.
    #[error("Incomplete chunk header at byte offset {0}")]
    IncompleteChunkHeader(u64),

    /// A chunk's declared size (including odd-byte padding) exceeds RIFF container bounds.
    #[error("Chunk '{id}' exceeds RIFF bounds (padded size {padded_size}, remaining {remaining})")]
    ChunkExceedsBounds {
        /// Chunk four-character code.
        id: String,
        /// Chunk payload size plus odd-byte padding.
        padded_size: u64,
        /// Remaining bytes in the RIFF container.
        remaining: u64,
    },

    /// The format (`fmt `) chunk is missing, truncated (< 16 bytes), or duplicated.
    #[error("Invalid or duplicate format chunk")]
    InvalidOrDuplicateFmt,

    /// Channel count is not mono (1) or stereo (2).
    #[error("Only mono (1) or stereo (2) impulse responses are supported, found {0} channels")]
    UnsupportedChannelCount(u16),

    /// Extensible format chunk (`0xFFFE`) is truncated or has invalid extension sizes.
    #[error("Invalid extensible format header")]
    InvalidExtensibleFormat,

    /// Extensible `valid_bits_per_sample` field is zero or exceeds container bit depth.
    #[error("Invalid valid-bits field in extensible format: {valid_bits} (container bits: {bits})")]
    InvalidValidBits {
        /// Declared valid bits per sample.
        valid_bits: u16,
        /// Container bit depth.
        bits: u16,
    },

    /// Extensible subtype GUID does not match standard PCM or IEEE float.
    #[error("Unsupported extensible subtype GUID")]
    UnsupportedExtensibleSubtype,

    /// Extensible floating-point format specifies valid bits not equal to container bit depth.
    #[error(
        "Invalid floating-point valid-bits field in extensible format: {valid_bits} (expected {bits})"
    )]
    InvalidFloatValidBits {
        /// Declared valid bits per sample.
        valid_bits: u16,
        /// Container bit depth.
        bits: u16,
    },

    /// Format code or bit depth is not supported by the engine.
    #[error(
        "Unsupported format {format} with bit depth {bits} (supported: PCM 16/24/32-bit and IEEE float 32-bit)"
    )]
    UnsupportedFormat {
        /// Audio format code (1 = PCM, 3 = IEEE float, 65534 = Extensible).
        format: u32,
        /// Bit depth per sample.
        bits: u16,
    },

    /// Sample rate is zero, alignment is inconsistent with channels/bits, or byte rate mismatch.
    #[error("Invalid sample rate ({rate} Hz) or block alignment ({alignment})")]
    InvalidRateOrAlignment {
        /// Declared sample rate in Hz.
        rate: u32,
        /// Block alignment in bytes.
        alignment: u16,
        /// Declared byte rate in bytes/sec.
        byte_rate: u32,
    },

    /// The `data` chunk appeared before `fmt ` or appeared more than once.
    #[error("Missing format or duplicate data chunk")]
    MissingFmtOrDuplicateData,

    /// The `data` chunk is empty or its size is not a multiple of block alignment.
    #[error(
        "Missing, empty, or incomplete sample data (size: {data_size}, alignment: {alignment})"
    )]
    InvalidDataSize {
        /// Data chunk size in bytes.
        data_size: u32,
        /// Expected block alignment in bytes.
        alignment: u16,
    },

    /// Sample count exceeds 32-bit signed integer limits.
    #[error("Impulse response is too long: {0} samples")]
    TooLong(usize),

    /// A floating-point sample in the audio payload is non-finite (NaN or Inf).
    #[error("Non-finite sample detected in WAV impulse response")]
    NonFiniteSample,

    /// File truncated while attempting to read sample bytes.
    #[error("Truncated payload in WAV file")]
    TruncatedPayload,
}

impl WavError {
    /// Maps this error variant to the standardized [`NamErrorCode`].
    pub fn error_code(&self) -> NamErrorCode {
        match self {
            Self::UnsupportedFormat { .. }
            | Self::UnsupportedChannelCount(_)
            | Self::InvalidExtensibleFormat
            | Self::InvalidValidBits { .. }
            | Self::UnsupportedExtensibleSubtype
            | Self::InvalidFloatValidBits { .. }
            | Self::InvalidRateOrAlignment { .. } => NamErrorCode::WavInvalidFormat,

            Self::MissingHeader
            | Self::ExpectedRiffWave
            | Self::InvalidRiffSize(_)
            | Self::IncompleteChunkHeader(_)
            | Self::ChunkExceedsBounds { .. }
            | Self::InvalidOrDuplicateFmt
            | Self::MissingFmtOrDuplicateData
            | Self::InvalidDataSize { .. }
            | Self::TooLong(_)
            | Self::NonFiniteSample
            | Self::TruncatedPayload => NamErrorCode::WavInvalidFile,
        }
    }
}

/// Decoded impulse response data parsed from a WAV container.
#[derive(Debug, Clone, PartialEq)]
pub struct WavIrData {
    /// Audio sample rate in Hz from the `fmt ` header.
    pub sample_rate: f32,
    /// Number of output audio channels (1 = mono, 2 = stereo).
    pub channels: usize,
    /// Number of FIR taps per channel (receptive field).
    pub receptive_field: usize,
    /// Decoded filter weights in channel-contiguous layout.
    ///
    /// For mono: `[Ch0_tap0, Ch0_tap1, ..., Ch0_tapN]`.
    /// For stereo: `[Ch0_tap0, ..., Ch0_tapN, Ch1_tap0, ..., Ch1_tapN]`.
    pub weights: Vec<f32>,
}

#[inline]
fn uint_le(bytes: &[u8]) -> u32 {
    let mut val = 0u32;
    for (i, &b) in bytes.iter().enumerate().take(4) {
        val |= (b as u32) << (8 * i);
    }
    val
}

/// Standard GUID tail for `KSDATAFORMAT_SUBTYPE_PCM` and `KSDATAFORMAT_SUBTYPE_IEEE_FLOAT`:
/// `{0x00000000, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}}`
const EXTENSIBLE_GUID_TAIL: [u8; 12] = [0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71];

/// Parses a byte slice containing a RIFF/WAVE file into decoded impulse response weights.
///
/// Implements strict little-endian decoding and boundary checking matching upstream
/// `NeuralAmpModelerCore/NAM/wav.cpp`.
pub fn parse_wav_ir(bytes: &[u8]) -> Result<WavIrData, WavError> {
    if bytes.len() < 12 {
        return Err(WavError::MissingHeader);
    }

    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WavError::ExpectedRiffWave);
    }

    let riff_size = uint_le(&bytes[4..8]) as u64;
    let end = riff_size + 8;
    if end < 12 || end > bytes.len() as u64 {
        return Err(WavError::InvalidRiffSize(end));
    }
    let end = end as usize;

    let mut position = 12usize;
    let mut have_format = false;
    let mut have_data = false;
    let mut format = 0u32;
    let mut bits = 0u16;
    let mut rate = 0u32;
    let mut alignment = 0u16;
    let mut channels = 0u16;
    let mut data_position = 0usize;
    let mut data_size = 0u32;

    while position < end {
        if end - position < 8 {
            return Err(WavError::IncompleteChunkHeader(position as u64));
        }

        let chunk_id = &bytes[position..position + 4];
        let size = uint_le(&bytes[position + 4..position + 8]);
        let padded_size = (size as u64) + ((size as u64) % 2);
        position += 8;

        if padded_size > (end - position) as u64 {
            let id_str = String::from_utf8_lossy(chunk_id).to_string();
            return Err(WavError::ChunkExceedsBounds {
                id: id_str,
                padded_size,
                remaining: (end - position) as u64,
            });
        }

        if chunk_id == b"fmt " {
            if have_format || size < 16 {
                return Err(WavError::InvalidOrDuplicateFmt);
            }

            let fmt_len = (size as usize).min(40);
            if position + fmt_len > end {
                return Err(WavError::TruncatedPayload);
            }
            let fmt = &bytes[position..position + fmt_len];

            format = uint_le(&fmt[0..2]);
            channels = uint_le(&fmt[2..4]) as u16;
            if channels != 1 && channels != 2 {
                return Err(WavError::UnsupportedChannelCount(channels));
            }

            rate = uint_le(&fmt[4..8]);
            let byte_rate = uint_le(&fmt[8..12]);
            alignment = uint_le(&fmt[12..14]) as u16;
            bits = uint_le(&fmt[14..16]) as u16;

            if format == 65534 {
                if size < 40 {
                    return Err(WavError::InvalidExtensibleFormat);
                }
                let cb_size = uint_le(&fmt[16..18]);
                if cb_size < 22 || (cb_size as u64) + 18 > size as u64 {
                    return Err(WavError::InvalidExtensibleFormat);
                }
                let valid_bits = uint_le(&fmt[18..20]) as u16;
                if valid_bits == 0 || valid_bits > bits {
                    return Err(WavError::InvalidValidBits { valid_bits, bits });
                }
                if fmt[28..40] != EXTENSIBLE_GUID_TAIL {
                    return Err(WavError::UnsupportedExtensibleSubtype);
                }
                format = uint_le(&fmt[24..28]);
                if format == 3 && valid_bits != bits {
                    return Err(WavError::InvalidFloatValidBits { valid_bits, bits });
                }
            }

            let is_pcm = format == 1 && (bits == 16 || bits == 24 || bits == 32);
            let is_float = format == 3 && bits == 32;
            if !is_pcm && !is_float {
                return Err(WavError::UnsupportedFormat { format, bits });
            }

            let expected_align = channels * (bits / 8);
            let expected_byte_rate = (rate as u64) * (expected_align as u64);
            if rate == 0 || alignment != expected_align || (byte_rate as u64) != expected_byte_rate
            {
                return Err(WavError::InvalidRateOrAlignment {
                    rate,
                    alignment,
                    byte_rate,
                });
            }

            have_format = true;
        } else if chunk_id == b"data" {
            if !have_format || have_data {
                return Err(WavError::MissingFmtOrDuplicateData);
            }
            data_position = position;
            data_size = size;
            have_data = true;
        }

        position += padded_size as usize;
    }

    if !have_data || data_size == 0 || !data_size.is_multiple_of(alignment as u32) {
        return Err(WavError::InvalidDataSize {
            data_size,
            alignment,
        });
    }

    let count = (data_size / (alignment as u32)) as usize;
    if count > i32::MAX as usize {
        return Err(WavError::TooLong(count));
    }

    let channels_usize = channels as usize;
    let total_samples = count
        .checked_mul(channels_usize)
        .ok_or(WavError::TooLong(count))?;
    let mut samples = vec![0.0f32; total_samples];
    let bytes_per_sample = (bits / 8) as usize;

    if data_position + total_samples * bytes_per_sample > bytes.len() {
        return Err(WavError::TruncatedPayload);
    }

    for i in 0..total_samples {
        let frame = i / channels_usize;
        let ch = i % channels_usize;
        let sample_dest_idx = ch * count + frame;

        let sample_offset = data_position + i * bytes_per_sample;
        let raw = uint_le(&bytes[sample_offset..sample_offset + bytes_per_sample]);

        let sample_val = if format == 3 {
            let f = f32::from_bits(raw);
            if !f.is_finite() {
                return Err(WavError::NonFiniteSample);
            }
            f
        } else {
            let sign_bit = 1u32 << (bits - 1);
            let signed_sample: i64 = if (raw & sign_bit) != 0 {
                (raw as i64) - (1i64 << bits)
            } else {
                raw as i64
            };
            let scale = (1u64 << (bits - 1)) as f64;
            (signed_sample as f64 / scale) as f32
        };

        samples[sample_dest_idx] = sample_val;
    }

    info!(
        "[Loader] Parsed WAV impulse response: sample_rate={:.0} Hz, channels={}, taps={}",
        rate, channels, count
    );

    Ok(WavIrData {
        sample_rate: rate as f32,
        channels: channels_usize,
        receptive_field: count,
        weights: samples,
    })
}

/// Converts decoded [`WavIrData`] into the engine's canonical [`NamModelData`] with Linear architecture.
///
/// Configures `in_channels = 1`, `out_channels = wav.channels` (1 or 2), `bias = false`,
/// and `version = "0.7.0"`, integrating directly with `LinearModel` and `LinearOneToMany`.
pub fn wav_ir_to_model_data(wav: WavIrData) -> NamModelData {
    NamModelData {
        version: Some("0.7.0".to_string()),
        architecture: "Linear".to_string(),
        config: NamConfig {
            in_channels: Some(1),
            out_channels: Some(wav.channels),
            receptive_field: Some(wav.receptive_field),
            bias: Some(false),
            ..Default::default()
        },
        weights: wav.weights,
        sample_rate: Some(wav.sample_rate),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

#[cfg(test)]
#[path = "wav_test.rs"]
mod wav_test;
