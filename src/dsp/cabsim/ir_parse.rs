// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WAV/IR parsing: RIFF chunk scanning, header validation, sample reading and validation.
//!
//! Extracted from `loader.rs` (E-RF4).

use log::debug;

use std::io;

/// Maximum IR file size to guard against malformed/malicious WAVs (1 GiB).
pub(crate) const MAX_IR_FILE_SIZE: u64 = 1_073_741_824;

/// Maximum IR length in samples to guard against OOM (~4s @ 48kHz).
pub(crate) const MAX_IR_LENGTH: usize = 192_000;

/// Minimum size for a valid WAV header (44 bytes: RIFF + fmt + data).
pub(crate) const WAV_HEADER_MIN: usize = 44;

// RIFF chunk IDs.
const RIFF_ID: [u8; 4] = *b"RIFF";
const WAVE_ID: [u8; 4] = *b"WAVE";
const FMT_ID: [u8; 4] = *b"fmt ";
const DATA_ID: [u8; 4] = *b"data";

// WAV format tags.
pub(crate) const WAV_FORMAT_PCM: u16 = 1;
pub(crate) const WAV_FORMAT_IEEE_FLOAT: u16 = 3;
pub(crate) const WAV_FORMAT_EXTENSIBLE: u16 = 65534; // 0xFFFE

/// Minimum IR sample rate to guard against catastrophic upsampling and OOM (4 kHz).
pub(crate) const MIN_IR_SAMPLE_RATE: u32 = 4_000;

/// Maximum IR sample rate for stability and reasonable mem usage (384 kHz).
pub(crate) const MAX_IR_SAMPLE_RATE: u32 = 384_000;

/// Safety cap on the number of RIFF chunks scanned by [`find_chunk`] (F-20).
///
/// Zero-sized junk chunks advance the scan by only 8 bytes each, so a hostile
/// file (up to [`MAX_IR_FILE_SIZE`]) could otherwise force ~134 M scan
/// iterations (DoS). Beyond this cap the scan aborts with `None`, bounding
/// parse time even for malformed streams with thousands of junk chunks.
pub(crate) const MAX_CHUNKS_SCANNED: usize = 1024;

/// Reads and validates the file, guarding against oversized inputs.
///
/// Uses a single file-open to avoid TOCTOU races between metadata check and read.
pub(crate) fn read_file(path: &std::path::Path) -> io::Result<Vec<u8>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();

    if file_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("IR WAV file is empty: {}", path.display()),
        ));
    }

    if file_size > MAX_IR_FILE_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "IR WAV file too large ({} bytes, max {})",
                file_size, MAX_IR_FILE_SIZE
            ),
        ));
    }

    if file_size < WAV_HEADER_MIN as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "IR WAV file too small ({} bytes, min {})",
                file_size, WAV_HEADER_MIN
            ),
        ));
    }

    let mut data = Vec::with_capacity(file_size as usize);
    file.read_to_end(&mut data)?;
    debug!("[Loader] IR file read: {} bytes", data.len());
    Ok(data)
}

/// Parses a WAV byte buffer, returning (samples, sample_rate).
///
/// Handles PCM16, PCM24, and IEEE float32 in mono.
/// Scans for `"fmt "` and `"data"` chunks rather than assuming fixed offsets.
pub(crate) fn parse_wav(data: &[u8]) -> io::Result<(Vec<f32>, u32)> {
    if data[0..4] != RIFF_ID || data[8..12] != WAVE_ID {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IR WAV: invalid RIFF/WAVE header",
        ));
    }

    let (fmt_offset, _fmt_size) = find_chunk(data, &FMT_ID, 12).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "IR WAV: 'fmt ' chunk not found")
    })?;

    if fmt_offset + 16 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IR WAV: 'fmt ' chunk too small",
        ));
    }

    let raw_audio_format = u16::from_le_bytes([data[fmt_offset], data[fmt_offset + 1]]);
    let num_channels = u16::from_le_bytes([data[fmt_offset + 2], data[fmt_offset + 3]]);
    let sample_rate = u32::from_le_bytes([
        data[fmt_offset + 4],
        data[fmt_offset + 5],
        data[fmt_offset + 6],
        data[fmt_offset + 7],
    ]);
    let bits_per_sample = u16::from_le_bytes([data[fmt_offset + 14], data[fmt_offset + 15]]);

    let audio_format = if raw_audio_format == WAV_FORMAT_EXTENSIBLE {
        if fmt_offset + 26 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IR WAV: 'fmt ' chunk too small for extensible format",
            ));
        }
        u16::from_le_bytes([data[fmt_offset + 24], data[fmt_offset + 25]])
    } else {
        raw_audio_format
    };

    if num_channels != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("IR WAV must be mono (found {} channels)", num_channels),
        ));
    }

    if !(MIN_IR_SAMPLE_RATE..=MAX_IR_SAMPLE_RATE).contains(&sample_rate) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "IR WAV: sample rate {} out of range ({}-{})",
                sample_rate, MIN_IR_SAMPLE_RATE, MAX_IR_SAMPLE_RATE
            ),
        ));
    }

    let (data_start, data_size) = find_data_chunk(data)?;

    let bytes_per_sample = match (audio_format, bits_per_sample) {
        (WAV_FORMAT_PCM, 16) => 2usize,
        (WAV_FORMAT_PCM, 24) => 3,
        (WAV_FORMAT_PCM, 32) => 4,
        (WAV_FORMAT_IEEE_FLOAT, 32) => 4,
        (fmt, bits) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "IR WAV: unsupported format (audio_format={}, bits={})",
                    fmt, bits
                ),
            ));
        }
    };
    let num_samples = data_size as usize / bytes_per_sample;
    if num_samples > MAX_IR_LENGTH {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "IR WAV: too many samples ({} samples, max is {})",
                num_samples, MAX_IR_LENGTH
            ),
        ));
    }

    let mut samples = match audio_format {
        WAV_FORMAT_PCM => match bits_per_sample {
            16 => read_pcm16(&data[data_start..], data_size),
            24 => read_pcm24(&data[data_start..], data_size),
            32 => read_pcm32(&data[data_start..], data_size),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "IR WAV: unsupported PCM bit depth {} (only 16, 24, and 32 supported)",
                        bits_per_sample
                    ),
                ));
            }
        },
        WAV_FORMAT_IEEE_FLOAT if bits_per_sample == 32 => {
            read_float32(&data[data_start..], data_size)
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "IR WAV: unsupported format (audio_format={}, bits={})",
                    audio_format, bits_per_sample
                ),
            ));
        }
    };

    let fmt_label = match (raw_audio_format, audio_format, bits_per_sample) {
        (WAV_FORMAT_EXTENSIBLE, WAV_FORMAT_PCM, 16) => "extensible-PCM16",
        (WAV_FORMAT_EXTENSIBLE, WAV_FORMAT_PCM, 24) => "extensible-PCM24",
        (WAV_FORMAT_EXTENSIBLE, WAV_FORMAT_PCM, 32) => "extensible-PCM32",
        (WAV_FORMAT_EXTENSIBLE, WAV_FORMAT_IEEE_FLOAT, 32) => "extensible-float32",
        (_, WAV_FORMAT_PCM, 16) => "PCM16",
        (_, WAV_FORMAT_PCM, 24) => "PCM24",
        (_, WAV_FORMAT_PCM, 32) => "PCM32",
        (_, WAV_FORMAT_IEEE_FLOAT, 32) => "float32",
        _ => "unknown",
    };
    debug!(
        "[Loader] IR WAV parsed: {} channels, {} Hz, fmt={}, {} samples",
        num_channels, sample_rate, fmt_label, num_samples
    );

    if samples.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IR WAV: no audio samples found",
        ));
    }

    validate_samples(&mut samples)?;

    Ok((samples, sample_rate))
}

/// Locates a RIFF chunk by its 4-byte ID, starting search at `start_offset`.
///
/// Returns `Some((data_offset, data_size))` where `data_offset` points to the
/// chunk's payload, or `None` if not found or if the scan exceeded
/// [`MAX_CHUNKS_SCANNED`] iterations (junk-chunk DoS guard — F-20).
pub(crate) fn find_chunk(
    data: &[u8],
    chunk_id: &[u8; 4],
    start_offset: usize,
) -> Option<(usize, u32)> {
    let mut pos = start_offset;
    let mut scanned = 0usize;
    while pos + 8 <= data.len() {
        if scanned >= MAX_CHUNKS_SCANNED {
            return None;
        }
        scanned += 1;
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
        if id == chunk_id {
            return Some((pos + 8, size));
        }
        let padded_size = (size as usize + 1) & !1;
        pos = pos.saturating_add(8 + padded_size);
    }
    None
}

/// Locates the "data" chunk in a WAV file, returning (offset, size_in_bytes).
pub(crate) fn find_data_chunk(data: &[u8]) -> io::Result<(usize, u32)> {
    find_chunk(data, &DATA_ID, 12)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "IR WAV: 'data' chunk not found"))
}

/// Validates samples are finite and flushes denormals (subnormals) to zero.
///
/// Catches NaN/Inf from corrupt float32 WAVs, and sanitizes denormals that
/// would degrade performance in SIMD DSP paths downstream.
pub(crate) fn validate_samples(samples: &mut [f32]) -> io::Result<()> {
    for s in samples.iter_mut() {
        if !s.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IR WAV: samples contain NaN or Infinity",
            ));
        }
        if !s.is_normal() && *s != 0.0 {
            *s = 0.0;
        }
    }
    Ok(())
}

/// Reads PCM16 mono samples as f32 in [-1.0, 1.0).
pub(crate) fn read_pcm16(data: &[u8], data_size: u32) -> Vec<f32> {
    let num_samples = (data_size as usize) / 2;
    let available = data.len() / 2;
    let n = num_samples.min(available);
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 2;
        let raw = i16::from_le_bytes([data[offset], data[offset + 1]]);
        let f = if raw >= 0 {
            raw as f32 / 32767.0
        } else {
            raw as f32 / 32768.0
        };
        samples.push(f);
    }
    samples
}

/// Reads PCM24 mono samples as f32 in [-1.0, 1.0).
pub(crate) fn read_pcm24(data: &[u8], data_size: u32) -> Vec<f32> {
    let num_samples = (data_size as usize) / 3;
    let available = data.len() / 3;
    let n = num_samples.min(available);
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 3;
        let b0 = data[offset] as i32;
        let b1 = data[offset + 1] as i32;
        let b2 = data[offset + 2] as i32;
        let raw = (b2 << 16) | (b1 << 8) | b0;
        let raw = if raw & 0x0080_0000 != 0 {
            raw | 0xFF00_0000u32 as i32
        } else {
            raw
        };
        let f = if raw >= 0 {
            raw as f32 / 8_388_607.0
        } else {
            raw as f32 / 8_388_608.0
        };
        samples.push(f);
    }
    samples
}

/// Reads PCM32 mono samples as f32 in [-1.0, 1.0).
pub(crate) fn read_pcm32(data: &[u8], data_size: u32) -> Vec<f32> {
    let num_samples = (data_size as usize) / 4;
    let available = data.len() / 4;
    let n = num_samples.min(available);
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 4;
        let raw = i32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let f = raw as f32 / 2_147_483_648.0;
        samples.push(f);
    }
    samples
}

/// Reads IEEE float32 mono samples directly (no conversion).
pub(crate) fn read_float32(data: &[u8], data_size: u32) -> Vec<f32> {
    let num_samples = (data_size as usize) / 4;
    let available = data.len() / 4;
    let n = num_samples.min(available);
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 4;
        let f = f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        samples.push(f);
    }
    samples
}

#[cfg(test)]
#[path = "ir_parse_test.rs"]
mod ir_parse_test;
