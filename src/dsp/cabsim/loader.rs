// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Robust WAV IR loader: mono PCM16, PCM24, and IEEE float32.
//!
//! Loading, resampling, and normalization happen **outside** the audio thread.
//! The resulting `CabSimIr` is transferred to the RT callback via lock-free SPSC,
//! following the same pattern as `NamResampler` swapping.

use crate::dsp::pipeline::MAX_RESAMP_BUF;
use crate::dsp::resampler::NamResampler;
use crate::dsp::sinc_kernel::TAPS_PER_PHASE;
use log::{debug, info};

use std::io;
use std::io::Read;
use std::path::Path;

/// Maximum IR file size to guard against malformed/malicious WAVs (1 GiB).
const MAX_IR_FILE_SIZE: u64 = 1_073_741_824;

/// Maximum IR length in samples to guard against OOM (~4s @ 48kHz).
const MAX_IR_LENGTH: usize = 192_000;

/// Minimum size for a valid WAV header (44 bytes: RIFF + fmt + data).
const WAV_HEADER_MIN: usize = 44;

// RIFF chunk IDs.
const RIFF_ID: [u8; 4] = *b"RIFF";
const WAVE_ID: [u8; 4] = *b"WAVE";
const FMT_ID: [u8; 4] = *b"fmt ";
const DATA_ID: [u8; 4] = *b"data";

// WAV format tags.
const WAV_FORMAT_PCM: u16 = 1;
const WAV_FORMAT_IEEE_FLOAT: u16 = 3;
const WAV_FORMAT_EXTENSIBLE: u16 = 65534; // 0xFFFE

/// Minimum IR sample rate to guard against catastrophic upsampling and OOM (4 kHz).
const MIN_IR_SAMPLE_RATE: u32 = 4_000;

/// Maximum IR sample rate for stability and reasonable mem usage (384 kHz).
const MAX_IR_SAMPLE_RATE: u32 = 384_000;

/// A loaded impulse response ready for convolution.
///
/// All memory is pre-allocated at load time (outside the audio thread).
/// The struct holds the resampled, normalized mono samples alongside
/// the original metadata needed for convolution setup.
pub struct CabSimIr {
    /// Mono IR samples (f32).
    pub samples: Vec<f32>,
    /// Sample rate of the IR after resampling (i.e., the effective rate of `samples`).
    pub sample_rate: u32,
    /// Sample rate of the original WAV file (before resampling, if applicable).
    pub original_rate: u32,
    /// Whether the IR was normalized (peak = 1.0).
    pub normalized: bool,
}

impl CabSimIr {
    /// Loads a mono WAV impulse response from `path`.
    ///
    /// Supported formats: PCM16, PCM24, IEEE float32.
    ///
    /// # Parameters
    /// - `path`: path to the `.wav` file.
    /// - `target_rate`: desired sample rate after resampling. If 0, no resampling.
    /// - `normalize`: if true, normalize samples so peak amplitude = 1.0.
    ///
    /// # Errors
    /// Returns `io::Error` on I/O failures, invalid headers, unsupported formats,
    /// or resampling errors. Never panics.
    #[cold]
    pub fn load(path: &Path, target_rate: u32, normalize: bool) -> io::Result<Box<Self>> {
        info!(
            "[Loader] Loading IR from \"{}\" (target_rate={} Hz, normalize={})",
            path.display(),
            target_rate,
            normalize
        );
        let data = Self::read_file(path)?;
        let (samples, original_rate) = Self::parse_wav(&data)?;

        let mut samples = if target_rate != 0 && target_rate != original_rate {
            info!(
                "[Loader] IR resampling: {} Hz -> {} Hz",
                original_rate, target_rate
            );
            Self::resample(&samples, original_rate, target_rate)?
        } else {
            samples
        };

        let effective_rate = if target_rate != 0 && target_rate != original_rate {
            target_rate
        } else {
            original_rate
        };

        let normalized = if normalize {
            let was_normalized = Self::normalize_in_place(&mut samples);
            if was_normalized {
                info!("[Loader] IR normalized to peak 1.0");
            } else {
                debug!("[Loader] IR normalization skipped (peak already ~1.0 or zero)");
            }
            was_normalized
        } else {
            false
        };

        info!(
            "[Loader] IR loaded: {} samples, {} Hz, normalized={}",
            samples.len(),
            effective_rate,
            normalized
        );

        Ok(Box::new(Self {
            samples,
            sample_rate: effective_rate,
            original_rate,
            normalized,
        }))
    }

    /// Reads and validates the file, guarding against oversized inputs.
    ///
    /// Uses a single file-open to avoid TOCTOU races between metadata check and read.
    fn read_file(path: &Path) -> io::Result<Vec<u8>> {
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
    fn parse_wav(data: &[u8]) -> io::Result<(Vec<f32>, u32)> {
        if data[0..4] != RIFF_ID || data[8..12] != WAVE_ID {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IR WAV: invalid RIFF/WAVE header",
            ));
        }

        let (fmt_offset, _fmt_size) = Self::find_chunk(data, &FMT_ID, 12).ok_or_else(|| {
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

        let (data_start, data_size) = Self::find_data_chunk(data)?;

        // Reject oversized IRs early — compute sample count from data chunk
        // before any allocation, preventing OOM for malformed/huge WAVs.
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
                16 => Self::read_pcm16(&data[data_start..], data_size),
                24 => Self::read_pcm24(&data[data_start..], data_size),
                32 => Self::read_pcm32(&data[data_start..], data_size),
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
                Self::read_float32(&data[data_start..], data_size)
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

        Self::validate_samples(&mut samples)?;

        Ok((samples, sample_rate))
    }

    /// Locates a RIFF chunk by its 4-byte ID, starting search at `start_offset`.
    ///
    /// Returns `Some((data_offset, data_size))` where `data_offset` points to the
    /// chunk's payload, or `None` if not found.
    fn find_chunk(data: &[u8], chunk_id: &[u8; 4], start_offset: usize) -> Option<(usize, u32)> {
        let mut pos = start_offset;
        while pos + 8 <= data.len() {
            let id = &data[pos..pos + 4];
            let size =
                u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
            if id == chunk_id {
                return Some((pos + 8, size));
            }
            let padded_size = (size as usize + 1) & !1;
            pos = pos.saturating_add(8 + padded_size);
        }
        None
    }

    /// Locates the "data" chunk in a WAV file, returning (offset, size_in_bytes).
    fn find_data_chunk(data: &[u8]) -> io::Result<(usize, u32)> {
        Self::find_chunk(data, &DATA_ID, 12).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "IR WAV: 'data' chunk not found")
        })
    }

    /// Validates samples are finite and flushes denormals (subnormals) to zero.
    ///
    /// Catches NaN/Inf from corrupt float32 WAVs, and sanitizes denormals that
    /// would degrade performance in SIMD DSP paths downstream.
    fn validate_samples(samples: &mut [f32]) -> io::Result<()> {
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
    fn read_pcm16(data: &[u8], data_size: u32) -> Vec<f32> {
        let num_samples = (data_size as usize) / 2;
        let available = data.len() / 2;
        let n = num_samples.min(available);
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * 2;
            let raw = i16::from_le_bytes([data[offset], data[offset + 1]]);
            // Normalize to [-1.0, 1.0)
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
    fn read_pcm24(data: &[u8], data_size: u32) -> Vec<f32> {
        let num_samples = (data_size as usize) / 3;
        let available = data.len() / 3;
        let n = num_samples.min(available);
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * 3;
            // Little-endian 24-bit signed -> i32
            let b0 = data[offset] as i32;
            let b1 = data[offset + 1] as i32;
            let b2 = data[offset + 2] as i32;
            let raw = (b2 << 16) | (b1 << 8) | b0;
            // Sign-extend from 24 bits
            let raw = if raw & 0x0080_0000 != 0 {
                raw | 0xFF00_0000u32 as i32
            } else {
                raw
            };
            // Normalize to [-1.0, 1.0)
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
    fn read_pcm32(data: &[u8], data_size: u32) -> Vec<f32> {
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
    fn read_float32(data: &[u8], data_size: u32) -> Vec<f32> {
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

    /// Resamples `input` from `input_rate` to `output_rate` using the polyphase resampler.
    ///
    /// This is a batch/offline operation: feeds the entire IR through `NamResampler`
    /// and pads with zeros to flush the filter’s delay line.
    pub fn resample(input: &[f32], input_rate: u32, output_rate: u32) -> io::Result<Vec<f32>> {
        let mut resampler =
            NamResampler::new(input_rate, output_rate, MAX_RESAMP_BUF).map_err(|e| {
                io::Error::other(format!(
                    "IR resample failed ({} Hz → {} Hz): {:#}",
                    input_rate, output_rate, e
                ))
            })?;

        // Estimate output length: input_len * (output_rate / input_rate) + safety margin
        let est_len =
            ((input.len() as f64 * output_rate as f64 / input_rate as f64).ceil() as usize) + 256;
        let mut output = Vec::with_capacity(est_len);

        let mut in_buf = vec![0.0f32; MAX_RESAMP_BUF];
        let mut out_l = vec![0.0f32; MAX_RESAMP_BUF];
        let mut out_r = vec![0.0f32; MAX_RESAMP_BUF];

        let mut pos = 0usize;
        while pos < input.len() {
            let chunk = (input.len() - pos).min(MAX_RESAMP_BUF);
            in_buf[..chunk].copy_from_slice(&input[pos..pos + chunk]);
            let written = resampler.process_input_mono(&in_buf[..chunk], &mut out_l, &mut out_r);
            output.extend_from_slice(&out_l[..written]);
            pos += chunk;
        }

        // Flush the resampler's delay line with zeros. Each iteration feeds
        // MAX_RESAMP_BUF zeros; TAPS_PER_PHASE × 2 iterations ensure the longer
        // ringdown with the increased tap count is fully drained.
        in_buf.fill(0.0);
        let flush_iters = TAPS_PER_PHASE * 2;
        for _ in 0..flush_iters {
            let written = resampler.process_input_mono(&in_buf, &mut out_l, &mut out_r);
            if written == 0 {
                break;
            }
            output.extend_from_slice(&out_l[..written]);
        }

        if output.is_empty() {
            return Err(io::Error::other("IR resample produced no output samples"));
        }

        Ok(output)
    }

    /// Normalizes samples in-place so the absolute peak becomes 1.0.
    ///
    /// Returns `true` if normalization was applied (peak > 0).
    /// If all samples are zero, they are left unchanged and returns `false`.
    fn normalize_in_place(samples: &mut [f32]) -> bool {
        let peak = samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));

        if peak <= 0.0 || peak >= 1.0 && (peak - 1.0).abs() < f32::EPSILON {
            return false;
        }

        let gain = 1.0 / peak;
        for s in samples.iter_mut() {
            *s *= gain;
        }
        true
    }
}

#[cfg(test)]
#[path = "loader_test.rs"]
mod loader_test;
