// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Minimalist helpers for IEEE float32 mono WAV read/write.
//!
//! Fixed format: 1 channel, 32-bit IEEE float, little-endian.
//! No external crate — just `std::fs` and `std::io`.
// Public helpers used by integration tests (cpp_parity) and binary crates (wav_to_golden, gen_stress).

#![allow(dead_code)]

use std::fs;
use std::io;
use std::path::Path;

const WAV_HEADER_SIZE: usize = 44;
const FMT_CHUNK_SIZE: u32 = 16;
const AUDIO_FORMAT_IEEE_FLOAT: u16 = 3;

/// Writes a mono IEEE float32 WAV file.
///
/// Uses fast bulk-copy on little-endian systems instead of per-sample iteration.
pub fn write_wav_f32(path: &Path, samples: &[f32], sample_rate: u32) -> io::Result<()> {
    let num_samples = samples.len() as u32;
    let data_size = num_samples * 4;
    let file_size = WAV_HEADER_SIZE as u32 + data_size;
    let byte_rate = sample_rate * 4;
    let block_align: u16 = 4;
    let bits_per_sample: u16 = 32;

    let mut buf = Vec::with_capacity(file_size as usize);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&FMT_CHUNK_SIZE.to_le_bytes());
    buf.extend_from_slice(&AUDIO_FORMAT_IEEE_FLOAT.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    #[cfg(target_endian = "little")]
    {
        // SAFETY: `f32` has no padding bytes and is repr(transparent) over 4 bytes.
        // On little-endian systems, the in-memory byte order of f32 matches the
        // WAV format requirement (IEEE 754 LE). The source slice is valid for
        // `samples.len() * 4` bytes because it was allocated as `&[f32]`.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(samples.as_ptr() as *const u8, samples.len() * 4) };
        buf.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
    }

    fs::write(path, &buf)
}

/// Reads a mono IEEE float32 WAV file.
///
/// Returns `(samples, sample_rate)`.
pub fn read_wav_f32(path: &Path) -> io::Result<(Vec<f32>, u32)> {
    let data = fs::read(path)?;

    if data.len() < WAV_HEADER_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("WAV file too small: {} bytes", data.len()),
        ));
    }

    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid RIFF/WAVE header",
        ));
    }

    let audio_format = u16::from_le_bytes([data[20], data[21]]);
    let num_channels = u16::from_le_bytes([data[22], data[23]]);
    let sr = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);

    if audio_format != AUDIO_FORMAT_IEEE_FLOAT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Audio format is not IEEE float32: {audio_format}"),
        ));
    }
    if num_channels != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("WAV is not mono (channels={num_channels})"),
        ));
    }

    let mut pos = 36;
    let mut data_size: u32 = 0;
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
        if chunk_id == b"data" {
            data_size = chunk_size;
            pos += 8;
            break;
        }
        pos = pos
            .checked_add(8usize.checked_add(chunk_size as usize).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("WAV chunk size overflow: {chunk_size}"),
                )
            })?)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("WAV position overflow at chunk size: {chunk_size}"),
                )
            })?;
    }
    if data_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "'data' chunk not found in WAV",
        ));
    }

    let num_samples = (data_size as usize) / 4;
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let offset = pos + i * 4;
        if offset + 4 > data.len() {
            break;
        }
        let f = f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        samples.push(f);
    }

    Ok((samples, sr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    fn temp_wav_path(name: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push("nam_wav_test");
        let _ = std::fs::create_dir_all(&dir);
        dir.push(name);
        dir
    }

    #[test]
    fn read_wav_f32_valid() {
        let path = temp_wav_path("test.wav");
        let samples = vec![0.5f32; 128];
        write_wav_f32(&path, &samples, 48000).expect("write");
        let (read_samples, sr) = read_wav_f32(&path).expect("read");
        assert_eq!(sr, 48000);
        assert_eq!(read_samples.len(), 128);
        for (&w, &r) in samples.iter().zip(read_samples.iter()) {
            assert!((w - r).abs() < 1e-6);
        }
    }

    #[test]
    fn read_wav_f32_malicious_chunk_size() {
        // Craft a WAV where a non-data chunk has chunk_size = u32::MAX,
        // causing overflow or graceful error instead of infinite loop.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&0u32.to_le_bytes()); // filesize (placeholder)
        buf.extend_from_slice(b"WAVE");
        // fmt chunk (valid, minimal)
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&48000u32.to_le_bytes()); // sr
        buf.extend_from_slice(&(48000u32 * 4).to_le_bytes()); // byte_rate
        buf.extend_from_slice(&4u16.to_le_bytes()); // block_align
        buf.extend_from_slice(&32u16.to_le_bytes()); // bits_per_sample
        // Malicious junk chunk with u32::MAX size (should not panic or loop forever)
        buf.extend_from_slice(b"JUNK");
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        // No 'data' chunk — should fail with 'data' chunk not found
        let path = temp_wav_path("malicious.wav");
        std::fs::write(&path, &buf).expect("write");
        let result = read_wav_f32(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("'data' chunk not found") || msg.contains("overflow"),
            "unexpected error: {msg}"
        );
    }
}
