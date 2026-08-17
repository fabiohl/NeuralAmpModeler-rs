// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests for the performance-baseline fingerprint (S3.T1): serde roundtrip
//! with hostile quoting, the typed field-by-field comparison, the bash
//! byte-compatible JSON schema, and file I/O.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::testing::qa::env::{EnvProbe, ISA_X86_64_V3};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unique temp path (no two tests collide on the same file).
fn temp_path() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("nam-fingerprint-{}-{n}", std::process::id()))
}

/// A fully populated, comparable fingerprint (governor `performance`).
fn fp() -> Fingerprint {
    Fingerprint {
        cpu_model: "AMD Ryzen 9 5900X 12-Core Processor".to_string(),
        cpu_microarchitecture: ISA_X86_64_V3.to_string(),
        physical_cores: 12,
        rustc_version: "rustc 1.88.0 (4f5a3a9 2025-06-10)".to_string(),
        target_triple: "x86_64-unknown-linux-gnu".to_string(),
        rustflags: String::new(),
        build_profile: DEFAULT_BUILD_PROFILE.to_string(),
        frequency_governor: "performance".to_string(),
        git_commit: "abc123".to_string(),
        bench_core: "6".to_string(),
    }
}

/// Asserts the comparison failed with the exact typed reason triple.
fn assert_mismatch(
    result: Result<(), FingerprintError>,
    field: &'static str,
    baseline: &str,
    current: &str,
) {
    match result {
        Err(FingerprintError::IncomparableEnvironment {
            field: f,
            baseline: b,
            current: c,
        }) => {
            assert_eq!(f, field);
            assert_eq!(b, baseline);
            assert_eq!(c, current);
        }
        other => panic!("expected IncomparableEnvironment({field}), got {other:?}"),
    }
}

/// Acceptance (S3.T1): a rustc banner or `RUSTFLAGS` containing `"` must be
/// JSON-escaped and round-trip — serde replaces the fragile heredoc/`sed`
/// pair that corrupted the file on such values.
#[test]
fn quoted_rustc_and_rustflags_roundtrip_without_corrupting_json() {
    let mut f = fp();
    f.rustc_version = "rustc 1.99.0 (Custom \"banner\")".to_string();
    f.rustflags = "-C target-cpu=\"x86-64-v3\"".to_string();

    let json = f.to_json_pretty().unwrap();
    assert!(
        json.contains("\\\"banner\\\""),
        "quotes must be escaped: {json}"
    );
    assert!(
        json.contains("\\\"x86-64-v3\\\""),
        "quotes must be escaped: {json}"
    );

    let parsed = Fingerprint::from_json_str(&json).unwrap();
    assert_eq!(parsed, f);

    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["rustc_version"], f.rustc_version);
    assert_eq!(value["rustflags"], f.rustflags);
}

/// Acceptance (S3.T1): a current governor != `performance` is incomparable
/// even when the baseline recorded the same non-performance governor.
#[test]
fn non_performance_governor_is_incomparable() {
    let baseline = fp();
    let mut current = fp();
    current.frequency_governor = "powersave".to_string();
    assert_mismatch(
        current.compare(&baseline),
        FIELD_FREQUENCY_GOVERNOR,
        "performance",
        "powersave",
    );

    let mut both_powersave = fp();
    both_powersave.frequency_governor = "powersave".to_string();
    assert_mismatch(
        current.compare(&both_powersave),
        FIELD_FREQUENCY_GOVERNOR,
        "powersave",
        "powersave",
    );
}

/// A `performance` current governor must still match a recorded baseline
/// value (bash `elif` branch).
#[test]
fn performance_governor_must_match_a_recorded_baseline_value() {
    let baseline = fp();
    let current = fp();
    assert!(current.compare(&baseline).is_ok());

    let mut schedutil_baseline = fp();
    schedutil_baseline.frequency_governor = "schedutil".to_string();
    assert_mismatch(
        current.compare(&schedutil_baseline),
        FIELD_FREQUENCY_GOVERNOR,
        "schedutil",
        "performance",
    );
}

/// The six always-compared fields each produce the typed reason carrying the
/// canonical field name and both values.
#[test]
fn always_compared_fields_report_field_baseline_and_current() {
    let baseline = fp();
    let mut current = fp();
    current.cpu_model = "Intel(R) Xeon(R) Gold".to_string();
    assert_mismatch(
        current.compare(&baseline),
        FIELD_CPU_MODEL,
        &baseline.cpu_model,
        &current.cpu_model,
    );

    let mut current = fp();
    current.cpu_microarchitecture = "x86-64 (base)".to_string();
    assert_mismatch(
        current.compare(&baseline),
        FIELD_CPU_MICROARCHITECTURE,
        &baseline.cpu_microarchitecture,
        &current.cpu_microarchitecture,
    );

    let mut current = fp();
    current.rustc_version = "rustc 2.0.0".to_string();
    assert_mismatch(
        current.compare(&baseline),
        FIELD_RUSTC_VERSION,
        &baseline.rustc_version,
        &current.rustc_version,
    );

    let mut current = fp();
    current.target_triple = "aarch64-unknown-linux-gnu".to_string();
    assert_mismatch(
        current.compare(&baseline),
        FIELD_TARGET_TRIPLE,
        &baseline.target_triple,
        &current.target_triple,
    );

    let mut current = fp();
    current.rustflags = "-C target-cpu=native".to_string();
    assert_mismatch(
        current.compare(&baseline),
        FIELD_RUSTFLAGS,
        &baseline.rustflags,
        &current.rustflags,
    );

    let mut current = fp();
    current.build_profile = "dev".to_string();
    assert_mismatch(
        current.compare(&baseline),
        FIELD_BUILD_PROFILE,
        &baseline.build_profile,
        &current.build_profile,
    );
}

/// First mismatch wins, in the bash comparison order (`cpu_model` first).
#[test]
fn first_mismatch_in_bash_order_wins() {
    let baseline = fp();
    let mut current = fp();
    current.cpu_model = "Different CPU".to_string();
    current.rustc_version = "rustc 2.0.0".to_string();
    assert_mismatch(
        current.compare(&baseline),
        FIELD_CPU_MODEL,
        &baseline.cpu_model,
        &current.cpu_model,
    );
}

/// `bench_core` and `physical_cores` are compared only when the baseline
/// recorded them (bash `[ -n ... ]` guards).
#[test]
fn optional_fields_are_only_compared_when_recorded() {
    let baseline = fp();

    let mut repinned = fp();
    repinned.bench_core = "3".to_string();
    assert_mismatch(repinned.compare(&baseline), FIELD_BENCH_CORE, "6", "3");

    let mut unpinned = fp();
    unpinned.bench_core = String::new();
    let mut pinned = fp();
    pinned.bench_core = "4".to_string();
    assert!(pinned.compare(&unpinned).is_ok());

    let mut fewer_cores = fp();
    fewer_cores.physical_cores = 8;
    assert_mismatch(
        fewer_cores.compare(&baseline),
        FIELD_PHYSICAL_CORES,
        "12",
        "8",
    );

    let mut unknown_cores = fp();
    unknown_cores.physical_cores = 0;
    assert!(fewer_cores.compare(&unknown_cores).is_ok());
}

/// `git_commit` is provenance only — the bash never compares it, and the
/// Rust port preserves that.
#[test]
fn git_commit_drift_is_provenance_only() {
    let mut current = fp();
    current.git_commit = "fffffff".to_string();
    assert!(current.compare(&fp()).is_ok());
}

/// Identical fingerprints compare `Ok` (the `--check` green path).
#[test]
fn identical_fingerprints_are_comparable() {
    assert!(fp().compare(&fp()).is_ok());
}

/// The JSON keys and value types are byte-compatible with the bash heredoc —
/// old baselines stay readable and new files stay greppable by the old `sed`.
#[test]
fn json_keys_match_the_bash_schema() {
    let json = fp().to_json_pretty().unwrap();
    for key in [
        "cpu_model",
        "cpu_microarchitecture",
        "physical_cores",
        "rustc_version",
        "target_triple",
        "rustflags",
        "build_profile",
        "frequency_governor",
        "git_commit",
        "bench_core",
    ] {
        assert!(
            json.contains(&format!("\"{key}\"")),
            "missing key {key} in {json}"
        );
    }
    assert!(
        json.contains("\"physical_cores\": 12"),
        "physical_cores must stay a JSON number: {json}"
    );
    assert!(json.contains("\"build_profile\": \"release\""));
}

/// A fingerprint file written by the bash heredoc parses as-is.
#[test]
fn reads_baseline_files_written_by_the_bash_heredoc() {
    let legacy = r#"{
  "cpu_model": "AMD Ryzen 9 5900X 12-Core Processor",
  "cpu_microarchitecture": "x86-64-v3 (AVX2/FMA/F16C/BMI)",
  "physical_cores": 12,
  "rustc_version": "rustc 1.88.0 (4f5a3a9 2025-06-10)",
  "target_triple": "x86_64-unknown-linux-gnu",
  "rustflags": "",
  "build_profile": "release",
  "frequency_governor": "performance",
  "git_commit": "abc123",
  "bench_core": "6"
}"#;
    assert_eq!(Fingerprint::from_json_str(legacy).unwrap(), fp());
}

/// Reading a nonexistent baseline file yields the typed `MissingBaseline`
/// reason (the perf gate's `MISSING_BASELINE`).
#[test]
fn missing_baseline_file_yields_missing_baseline() {
    let absent = temp_path();
    let err = Fingerprint::read_from_path(&absent).unwrap_err();
    assert!(
        matches!(err, FingerprintError::MissingBaseline),
        "got {err:?}"
    );
    let err = fp().compare_against_path(&absent).unwrap_err();
    assert!(
        matches!(err, FingerprintError::MissingBaseline),
        "got {err:?}"
    );
}

/// `write_to_path` + `read_from_path` round-trip, including quoted values,
/// and the written file is valid JSON.
#[test]
fn write_then_read_roundtrips_even_with_quotes() {
    let mut f = fp();
    f.rustflags = "-C target-cpu=\"x86-64-v3\"".to_string();
    let path = temp_path();
    f.write_to_path(&path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str::<serde_json::Value>(&text).unwrap();
    let read_back = Fingerprint::read_from_path(&path).unwrap();
    assert_eq!(read_back, f);
    let _ = std::fs::remove_file(&path);
}

/// `write_to_path` creates missing parent directories.
#[test]
fn write_to_path_creates_parent_directories() {
    let path = temp_path().join("nested").join("baseline-fingerprint.json");
    fp().write_to_path(&path).unwrap();
    assert!(Fingerprint::read_from_path(&path).is_ok());
    let _ = std::fs::remove_file(&path);
}

/// `compare_against_path` fails on drift with the typed field reason.
#[test]
fn compare_against_path_reports_drift_from_a_written_baseline() {
    let baseline_path = temp_path();
    fp().write_to_path(&baseline_path).unwrap();

    let mut current = fp();
    current.cpu_microarchitecture = "x86-64 (base)".to_string();
    assert_mismatch(
        current.compare_against_path(&baseline_path),
        FIELD_CPU_MICROARCHITECTURE,
        &fp().cpu_microarchitecture,
        &current.cpu_microarchitecture,
    );
    let _ = std::fs::remove_file(&baseline_path);
}

/// `from_env_probe` maps every `EnvProbe` field onto the fingerprint shape —
/// the single probe of R-09 feeds the baseline, receipts and dashboard.
#[test]
fn from_env_probe_maps_every_field() {
    let probe = EnvProbe {
        cpu_model: "AMD Ryzen 9 5900X 12-Core Processor".to_string(),
        effective_isa: ISA_X86_64_V3,
        physical_cores: 12,
        rustc_version: "rustc 1.88.0 (4f5a3a9 2025-06-10)".to_string(),
        host_triple: "x86_64-unknown-linux-gnu".to_string(),
        rustflags: "-C target-cpu=x86-64-v3".to_string(),
        frequency_governor: "performance".to_string(),
        git_commit: "deadbeef".to_string(),
        git_dirty: false,
    };
    let f = Fingerprint::from_env_probe(&probe, "7");
    assert_eq!(f.cpu_model, probe.cpu_model);
    assert_eq!(f.cpu_microarchitecture, ISA_X86_64_V3);
    assert_eq!(f.physical_cores, 12);
    assert_eq!(f.rustc_version, probe.rustc_version);
    assert_eq!(f.target_triple, probe.host_triple);
    assert_eq!(f.rustflags, probe.rustflags);
    assert_eq!(f.build_profile, DEFAULT_BUILD_PROFILE);
    assert_eq!(f.frequency_governor, probe.frequency_governor);
    assert_eq!(f.git_commit, probe.git_commit);
    assert_eq!(f.bench_core, "7");
}

/// `compare_against_path` on a nonexistent baseline yields `MissingBaseline`
/// even when the current fingerprint itself is perfectly formed.
#[test]
fn compare_against_path_without_baseline_is_missing_baseline() {
    let err = fp().compare_against_path(&temp_path()).unwrap_err();
    assert!(
        matches!(err, FingerprintError::MissingBaseline),
        "got {err:?}"
    );
}
