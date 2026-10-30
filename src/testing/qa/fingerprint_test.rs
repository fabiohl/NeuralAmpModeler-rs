// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for fingerprint serialization and comparison (R-12, R-23, S6-T03).

use super::*;

fn mock_fingerprint(hw_class: &str) -> Fingerprint {
    Fingerprint {
        cpu_model: "AMD Ryzen 7 5700U with Radeon Graphics".to_string(),
        cpu_microarchitecture: "x86-64-v3 (AVX2/FMA/F16C/BMI)".to_string(),
        physical_cores: 8,
        rustc_version: "rustc 1.99.0".to_string(),
        target_triple: "x86_64-unknown-linux-gnu".to_string(),
        rustflags: String::new(),
        build_profile: "release".to_string(),
        frequency_governor: "performance".to_string(),
        git_commit: "abcdef1234567890".to_string(),
        bench_core: "8".to_string(),
        git_dirty: Some(false),
        cargo_lock_sha256: Some("1234".to_string()),
        hardware_class: Some(hw_class.to_string()),
    }
}

#[test]
fn test_fingerprint_compare_matching_hardware_class() {
    let baseline = mock_fingerprint("x86_zen3_laptop");
    let current = mock_fingerprint("x86_zen3_laptop");
    assert!(current.compare(&baseline).is_ok());
}

#[test]
fn test_fingerprint_compare_mismatched_hardware_class_fails() {
    let baseline = mock_fingerprint("x86_zen3_laptop");
    let current = mock_fingerprint("aarch64_apple_silicon");
    let res = current.compare(&baseline);
    assert!(res.is_err());
    match res.unwrap_err() {
        FingerprintError::IncomparableEnvironment {
            field,
            baseline,
            current,
        } => {
            assert_eq!(field, FIELD_HARDWARE_CLASS);
            assert_eq!(baseline, "x86_zen3_laptop");
            assert_eq!(current, "aarch64_apple_silicon");
        }
        other => panic!("expected IncomparableEnvironment, got {other:?}"),
    }
}

#[test]
fn test_fingerprint_roundtrip_json() {
    let fp = mock_fingerprint("x86_zen3_laptop");
    let json = fp.to_json_pretty().expect("serialize failed");
    let deserialized = Fingerprint::from_json_str(&json).expect("deserialize failed");
    assert_eq!(fp, deserialized);
}
