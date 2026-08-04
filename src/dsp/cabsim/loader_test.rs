// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Helper function to create a programmatically generated mock Cab IR WAV file for testing.
fn create_temp_cab_ir_wav(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("nam_rs_test_fat_cab_{}.wav", name));

    // Generate some mock IR samples (exponential decay sine wave)
    let samples: Vec<f32> = (0..512)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * (-100.0 * t).exp()
        })
        .collect();

    // Write PCM16 mono WAV (which is standard and supported by CabSimIr::load)
    write_pcm16_wav(&path, &samples, 48_000).expect("failed to write mock IR WAV");
    path
}

/// Loads the mock Cab IR WAV and verifies all metadata.
#[test]
fn test_load_fat_cab_wav() {
    let path = create_temp_cab_ir_wav("fat_cab");
    let ir = CabSimIr::load(&path, 0, false).expect("failed to load mock Cab IR WAV");

    assert_eq!(ir.sample_rate, 48_000, "expected 48 kHz effective rate");
    assert_eq!(ir.original_rate, 48_000, "expected 48 kHz original rate");
    assert!(!ir.samples.is_empty(), "IR samples must not be empty");
    assert!(!ir.normalized, "normalization should be off");
    assert!(
        ir.samples.iter().any(|&s| s.abs() > 0.0),
        "IR must contain non-zero samples"
    );
    std::fs::remove_file(path).ok();
}

/// Loads with resampling to 44.1 kHz.
#[test]
fn test_load_with_resample() {
    let path = create_temp_cab_ir_wav("resample");
    let ir = CabSimIr::load(&path, 44_100, false).expect("failed to load with resample");
    assert_eq!(ir.sample_rate, 44_100, "effective rate should be target");
    assert_eq!(ir.original_rate, 48_000);
    assert!(!ir.samples.is_empty());
    std::fs::remove_file(path).ok();
}

/// Loads with normalization enabled.
#[test]
fn test_load_with_normalize() {
    let path = create_temp_cab_ir_wav("normalize");
    let ir = CabSimIr::load(&path, 0, true).expect("failed to load with normalize");
    assert!(ir.normalized, "normalization flag should be true");
    assert!(!ir.samples.is_empty());

    let peak = ir.samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
    assert!(
        (peak - 1.0).abs() < 1e-6,
        "normalized peak should be ~1.0, got {}",
        peak
    );
    std::fs::remove_file(path).ok();
}

/// Loading a non-existent file must return an error, never panic.
#[test]
fn test_load_nonexistent_file() {
    let result = CabSimIr::load(std::path::Path::new("tests/__nonexistent__.wav"), 0, false);
    assert!(result.is_err(), "nonexistent file must error");
}

/// Loading an empty file must return an error.
#[test]
fn test_load_empty_file() {
    let path = std::path::Path::new("/tmp/nam_rs_test_empty.wav");
    std::fs::write(path, []).ok();
    let result = CabSimIr::load(path, 0, false);
    assert!(result.is_err(), "empty file must error");
    std::fs::remove_file(path).ok();
}

/// Loading a file too small to be a valid WAV header must error.
#[test]
fn test_load_too_small() {
    let path = std::path::Path::new("/tmp/nam_rs_test_small.wav");
    std::fs::write(path, [0u8; 30]).ok();
    let result = CabSimIr::load(path, 0, false);
    assert!(result.is_err(), "file too small must error");
    std::fs::remove_file(path).ok();
}

/// PCM16 mono WAV round-trip via the testing wav writer.
#[test]
fn test_load_pcm16_wav() {
    let samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
    let path = std::path::Path::new("/tmp/nam_rs_test_pcm16.wav");

    // Write PCM16 using the format
    write_pcm16_wav(path, &samples, 48_000).expect("failed to write PCM16 WAV");

    let ir = CabSimIr::load(path, 0, false).expect("failed to load PCM16 WAV");
    assert_eq!(ir.sample_rate, 48_000);
    assert!(!ir.samples.is_empty());
    std::fs::remove_file(path).ok();
}

/// Float32 mono WAV via the testing wav writer.
#[test]
fn test_load_float32_wav() {
    let samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.002).sin()).collect();
    let path = std::path::Path::new("/tmp/nam_rs_test_f32.wav");

    crate::testing::wav::write_wav_f32(path, &samples, 48_000).expect("failed to write f32 WAV");

    let ir = CabSimIr::load(path, 0, false).expect("failed to load f32 WAV");
    assert_eq!(ir.sample_rate, 48_000);
    assert!(!ir.samples.is_empty());
    std::fs::remove_file(path).ok();
}

/// Normalization is a no-op when peak is already 1.0.
#[test]
fn test_normalize_noop() {
    let mut samples = vec![1.0f32, -1.0f32, 0.5];
    let applied = CabSimIr::normalize_in_place(&mut samples);
    assert!(!applied, "already normalized — noop expected");
}

/// Normalization scales correctly.
#[test]
fn test_normalize_scale() {
    let mut samples = vec![0.25f32, -0.5, 0.125];
    let applied = CabSimIr::normalize_in_place(&mut samples);
    assert!(applied);
    let peak = samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
    assert!((peak - 1.0).abs() < 1e-6);
}

/// Creates a minimal 44-byte WAV header with the given sample rate.
/// Uses PCM16 mono format; byte_rate and block_align are zero (not validated by the parser).
fn make_wav_header(sample_rate: u32) -> Vec<u8> {
    let mut buf = vec![0u8; 44];
    buf[0..4].copy_from_slice(b"RIFF");
    buf[4..8].copy_from_slice(&36u32.to_le_bytes()); // file_size - 8
    buf[8..12].copy_from_slice(b"WAVE");
    buf[12..16].copy_from_slice(b"fmt ");
    buf[16..20].copy_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    buf[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
    buf[22..24].copy_from_slice(&1u16.to_le_bytes()); // mono
    buf[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    buf[28..32].copy_from_slice(&0u32.to_le_bytes()); // byte_rate (unchecked)
    buf[32..34].copy_from_slice(&0u16.to_le_bytes()); // block_align (unchecked)
    buf[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits_per_sample
    buf[36..40].copy_from_slice(b"data");
    buf[40..44].copy_from_slice(&0u32.to_le_bytes()); // 0 data size
    buf
}

/// Rejects sample rates outside [4_000, 384_000] with InvalidData.
#[test]
fn test_reject_ir_extreme_sample_rate() {
    // 1 Hz — below MIN_IR_SAMPLE_RATE
    let buf = make_wav_header(1);
    let path = std::path::Path::new("/tmp/nam_rs_test_rate_1.wav");
    std::fs::write(path, &buf).unwrap();
    match CabSimIr::load(path, 0, false) {
        Err(err) => {
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
            assert!(
                err.to_string().contains("out of range"),
                "expected 'out of range' in: {}",
                err
            );
        }
        Ok(_) => panic!("should reject 1 Hz sample rate"),
    }
    std::fs::remove_file(path).ok();

    // u32::MAX — above MAX_IR_SAMPLE_RATE
    let buf = make_wav_header(u32::MAX);
    let path = std::path::Path::new("/tmp/nam_rs_test_rate_max.wav");
    std::fs::write(path, &buf).unwrap();
    match CabSimIr::load(path, 0, false) {
        Err(err) => {
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
            assert!(
                err.to_string().contains("out of range"),
                "expected 'out of range' in: {}",
                err
            );
        }
        Ok(_) => panic!("should reject u32::MAX sample rate"),
    }
    std::fs::remove_file(path).ok();
}

/// Writes a raw float32 mono WAV with arbitrary samples for denormal testing.
fn write_f32_wav_raw(
    path: &std::path::Path,
    samples: &[f32],
    sample_rate: u32,
) -> std::io::Result<()> {
    let num_samples = samples.len() as u32;
    let data_size = num_samples * 4;
    let file_size: u32 = 36 + data_size;
    let byte_rate = sample_rate * 4;
    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&4u16.to_le_bytes()); // block_align
    buf.extend_from_slice(&32u16.to_le_bytes()); // bits_per_sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, &buf)
}

/// Subnormal (denormal) samples are flushed to zero during validation.
#[test]
fn test_flush_denormal_ir_samples() {
    let denormal = f32::from_bits(0x0000_0001); // smallest positive subnormal
    let neg_denormal = f32::from_bits(0x8000_0001); // smallest negative subnormal
    let samples = vec![0.5f32, denormal, -0.3f32, neg_denormal, denormal];
    let path = std::path::Path::new("/tmp/nam_rs_test_denormal.wav");
    write_f32_wav_raw(path, &samples, 48_000).unwrap();

    let ir = CabSimIr::load(path, 0, false)
        .expect("loading WAV with denormals should succeed (flushed, not rejected)");
    assert_eq!(ir.samples[0], 0.5);
    assert_eq!(
        ir.samples[1], 0.0,
        "positive subnormal should be flushed to 0.0"
    );
    assert_eq!(ir.samples[2], -0.3);
    assert_eq!(
        ir.samples[3], 0.0,
        "negative subnormal should be flushed to 0.0"
    );
    assert_eq!(
        ir.samples[4], 0.0,
        "second positive subnormal should be flushed to 0.0"
    );
    std::fs::remove_file(path).ok();
}

/// Writes a minimal PCM16 mono WAV file for testing.
fn write_pcm16_wav(
    path: &std::path::Path,
    samples: &[f32],
    sample_rate: u32,
) -> std::io::Result<()> {
    let num_samples = samples.len() as u32;
    let data_size = num_samples * 2;
    let file_size: u32 = 36 + data_size;
    let byte_rate = sample_rate * 2;
    let block_align: u16 = 2;
    let bits_per_sample: u16 = 16;

    let mut buf = Vec::with_capacity(44 + data_size as usize);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &s in samples {
        let clamped = s.clamp(-1.0, 0.999969);
        let raw = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&raw.to_le_bytes());
    }

    std::fs::write(path, &buf)
}

/// Helper function to write WAVE_FORMAT_EXTENSIBLE (0xFFFE / 65534) mono WAV files for testing.
fn write_extensible_wav(
    path: &std::path::Path,
    samples_f32: &[f32],
    sample_rate: u32,
    subformat_pcm: bool,
    bits_per_sample: u16,
) -> std::io::Result<()> {
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let data_size = (samples_f32.len() * bytes_per_sample) as u32;
    let fmt_chunk_size = 40u32; // Standard WAVEFORMATEXTENSIBLE size
    let file_size: u32 = 12 + (8 + fmt_chunk_size) + (8 + data_size);
    let byte_rate = sample_rate * (bits_per_sample as u32 / 8);
    let block_align = bits_per_sample / 8;

    let mut buf = Vec::with_capacity(file_size as usize + 8);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(file_size - 8).to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&fmt_chunk_size.to_le_bytes());
    buf.extend_from_slice(&65534u16.to_le_bytes()); // WAV_FORMAT_EXTENSIBLE
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(&22u16.to_le_bytes()); // cbSize (extension size)
    buf.extend_from_slice(&bits_per_sample.to_le_bytes()); // wValidBitsPerSample
    buf.extend_from_slice(&0u32.to_le_bytes()); // dwChannelMask

    // SubFormat GUID: first 2 bytes are subformat code (1 for PCM, 3 for IEEE Float)
    let subformat_code: u16 = if subformat_pcm { 1 } else { 3 };
    buf.extend_from_slice(&subformat_code.to_le_bytes());
    // Remainder of KSDATAFORMAT_SUBTYPE_PCM / IEEE_FLOAT GUID (14 bytes):
    // 00-00-00-00-10-80-00-00-AA-00-38-9B-71-00
    buf.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x00, 0x10, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71, 0x00,
    ]);

    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    if subformat_pcm && bits_per_sample == 24 {
        for &s in samples_f32 {
            let clamped = s.clamp(-1.0, 0.999999);
            let raw = if clamped >= 0.0 {
                (clamped * 8_388_607.0) as i32
            } else {
                (clamped * 8_388_608.0) as i32
            };
            let b0 = (raw & 0xFF) as u8;
            let b1 = ((raw >> 8) & 0xFF) as u8;
            let b2 = ((raw >> 16) & 0xFF) as u8;
            buf.extend_from_slice(&[b0, b1, b2]);
        }
    } else if !subformat_pcm && bits_per_sample == 32 {
        for &s in samples_f32 {
            buf.extend_from_slice(&s.to_le_bytes());
        }
    }

    std::fs::write(path, &buf)
}

/// Tests loading a 24-bit PCM WAVE_FORMAT_EXTENSIBLE IR (format 65534).
#[test]
fn test_load_extensible_pcm24_wav() {
    let samples: Vec<f32> = (0..512).map(|i| (i as f32 * 0.005).sin()).collect();
    let path = std::path::Path::new("/tmp/nam_rs_test_ext_pcm24.wav");

    write_extensible_wav(path, &samples, 48_000, true, 24)
        .expect("failed to write extensible PCM24 WAV");

    let ir = CabSimIr::load(path, 0, false).expect("failed to load extensible PCM24 WAV");
    assert_eq!(ir.sample_rate, 48_000);
    assert_eq!(ir.samples.len(), 512);

    for (orig, read) in samples.iter().zip(ir.samples.iter()) {
        assert!(
            (orig - read).abs() < 1e-4,
            "sample mismatch: orig={}, read={}",
            orig,
            read
        );
    }

    std::fs::remove_file(path).ok();
}

/// Tests loading a 32-bit Float WAVE_FORMAT_EXTENSIBLE IR (format 65534).
#[test]
fn test_load_extensible_float32_wav() {
    let samples: Vec<f32> = (0..512).map(|i| (i as f32 * 0.005).cos()).collect();
    let path = std::path::Path::new("/tmp/nam_rs_test_ext_float32.wav");

    write_extensible_wav(path, &samples, 48_000, false, 32)
        .expect("failed to write extensible Float32 WAV");

    let ir = CabSimIr::load(path, 0, false).expect("failed to load extensible Float32 WAV");
    assert_eq!(ir.sample_rate, 48_000);
    assert_eq!(ir.samples.len(), 512);

    for (orig, read) in samples.iter().zip(ir.samples.iter()) {
        assert!((orig - read).abs() < 1e-6);
    }

    std::fs::remove_file(path).ok();
}
