// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use std::io::Cursor;

fn sample_header() -> ParityVectorHeader {
    ParityVectorHeader {
        magic: PARITY_VECTOR_MAGIC,
        protocol_version: PARITY_VECTOR_VERSION,
        endianness: "little".into(),
        model_id: "BossWN-standard".into(),
        model_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
        signal_name: "stress_v1".into(),
        signal_sha256: "d41d8cd98f00b204e9800998ecf8427e".into(),
        signal_seed: 42,
        signal_len: 128,
        dsp_mode: DspMode::Live,
        channel_mode: ChannelMode::Mono,
        sample_rate: 48000,
        block_size: 64,
        initial_state: InitialState::Prewarmed,
        num_channels: 1,
        num_samples_per_channel: 128,
        toolchain: "rustc 1.99.0".into(),
        compile_flags: "-C opt-level=3".into(),
        target_triple: "x86_64-unknown-linux-gnu".into(),
        instruction_set: "x86-64-v3".into(),
        fp_env: "FTZ=1,DAZ=1".into(),
        run_id: Some("run-test-01".into()),
    }
}

#[test]
fn roundtrip_serialization_deserialization() {
    let header = sample_header();
    let samples: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let original = ParityVector {
        header: header.clone(),
        samples: samples.clone(),
    };

    let mut buf = Vec::new();
    original.write_to(&mut buf).expect("write failed");

    let mut cursor = Cursor::new(buf);
    let decoded = ParityVector::read_from(&mut cursor).expect("read failed");

    assert_eq!(decoded.header, original.header);
    assert_eq!(decoded.samples.len(), original.samples.len());
    for (a, b) in decoded.samples.iter().zip(original.samples.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn reject_invalid_magic() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"BADMAGIC");
    buf.extend_from_slice(&1u32.to_le_bytes());

    let mut cursor = Cursor::new(buf);
    let err = ParityVector::read_from(&mut cursor).unwrap_err();
    match err {
        VectorProtocolError::InvalidMagic(m) => assert_eq!(&m, b"BADMAGIC"),
        other => panic!("Expected InvalidMagic, got: {other:?}"),
    }
}

#[test]
fn reject_truncated_payload() {
    let header = sample_header();
    let samples: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let vector = ParityVector { header, samples };

    let mut buf = Vec::new();
    vector.write_to(&mut buf).expect("write failed");

    // Truncate 20 bytes from the sample payload (128 samples = 512 bytes)
    buf.truncate(buf.len() - 20);

    let mut cursor = Cursor::new(buf);
    let err = ParityVector::read_from(&mut cursor).unwrap_err();
    match err {
        VectorProtocolError::TruncatedPayload { expected, actual } => {
            assert_eq!(expected, 128);
            assert_eq!(actual, 128 - (20 / 4));
        }
        other => panic!("Expected TruncatedPayload, got: {other:?}"),
    }
}

#[test]
fn reject_corrupted_payload_checksum() {
    let header = sample_header();
    let samples: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let vector = ParityVector { header, samples };

    let mut buf = Vec::new();
    vector.write_to(&mut buf).expect("write failed");

    // Corrupt one byte of sample data at the end
    let last = buf.len() - 1;
    buf[last] ^= 0xFF;

    let mut cursor = Cursor::new(buf);
    let err = ParityVector::read_from(&mut cursor).unwrap_err();
    match err {
        VectorProtocolError::CorruptedPayloadChecksum => (),
        other => panic!("Expected CorruptedPayloadChecksum, got: {other:?}"),
    }
}

#[test]
fn compatibility_check_detects_mismatched_sample_rate() {
    let h1 = sample_header();
    let mut h2 = sample_header();
    h2.sample_rate = 96000;

    let err = check_compatibility(&h1, &h2).unwrap_err();
    match err {
        VectorIncompatibilityReason::MismatchedSampleRate { ref_val, test_val } => {
            assert_eq!(ref_val, 48000);
            assert_eq!(test_val, 96000);
        }
        other => panic!("Expected MismatchedSampleRate, got: {other:?}"),
    }
}

#[test]
fn compatibility_check_detects_mismatched_model_sha() {
    let h1 = sample_header();
    let mut h2 = sample_header();
    h2.model_sha256 = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into();

    let err = check_compatibility(&h1, &h2).unwrap_err();
    match err {
        VectorIncompatibilityReason::MismatchedModelSha256 { .. } => (),
        other => panic!("Expected MismatchedModelSha256, got: {other:?}"),
    }
}

#[test]
fn compare_identical_vectors_passes_bitwise() {
    let h1 = sample_header();
    let mut h2 = sample_header();
    // Provenance may differ across platforms
    h2.target_triple = "aarch64-unknown-linux-gnu".into();
    h2.instruction_set = "aarch64+neon".into();

    let samples: Vec<f32> = vec![0.0, 0.5, -0.25, 0.125];
    let v1 = ParityVector {
        header: h1,
        samples: samples.clone(),
    };
    let v2 = ParityVector {
        header: h2,
        samples,
    };

    let report = compare_vectors(&v1, &v2);
    assert!(report.is_compatible);
    assert!(report.bitwise_identical);
    assert_eq!(report.mismatch_count, 0);
    assert_eq!(report.esr, 0.0);
    assert_eq!(report.first_mismatch, None);
}

#[test]
fn compare_vectors_detects_first_mismatched_sample() {
    let h1 = sample_header();
    let h2 = sample_header();

    let s1 = vec![0.1f32; 10];
    let mut s2 = vec![0.1f32; 10];
    // Alter sample index 4
    s2[4] = 0.10000005f32;

    let v1 = ParityVector {
        header: h1,
        samples: s1,
    };
    let v2 = ParityVector {
        header: h2,
        samples: s2,
    };

    let report = compare_vectors(&v1, &v2);
    assert!(report.is_compatible);
    assert!(!report.bitwise_identical);
    assert_eq!(report.mismatch_count, 1);
    let first = report.first_mismatch.expect("Should have first mismatch");
    assert_eq!(first.sample_index, 4);
    assert_ne!(first.ref_bits, first.test_bits);
    assert!(report.max_abs_diff > 0.0);
}

#[test]
fn signed_zeros_treated_as_bitwise_mismatch() {
    let h1 = sample_header();
    let h2 = sample_header();

    let s1 = vec![0.0f32];
    let s2 = vec![-0.0f32];

    assert_eq!(s1[0].to_bits(), 0x0000_0000);
    assert_eq!(s2[0].to_bits(), 0x8000_0000);

    let v1 = ParityVector {
        header: h1,
        samples: s1,
    };
    let v2 = ParityVector {
        header: h2,
        samples: s2,
    };

    let report = compare_vectors(&v1, &v2);
    assert!(report.is_compatible);
    assert!(!report.bitwise_identical);
    assert_eq!(report.mismatch_count, 1);
    let first = report.first_mismatch.unwrap();
    assert_eq!(first.ref_bits_hex, "0x00000000");
    assert_eq!(first.test_bits_hex, "0x80000000");
}

#[test]
fn nans_detected_and_reported() {
    let h1 = sample_header();
    let h2 = sample_header();

    let s1 = vec![0.5f32, f32::NAN];
    let s2 = vec![0.5f32, f32::NAN];

    let v1 = ParityVector {
        header: h1,
        samples: s1,
    };
    let v2 = ParityVector {
        header: h2,
        samples: s2,
    };

    let report = compare_vectors(&v1, &v2);
    assert_eq!(report.nan_count_ref, 1);
    assert_eq!(report.nan_count_test, 1);
    assert!(
        !report.bitwise_identical,
        "Vectors with NaNs must not be marked bitwise identical"
    );
}
