// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tempdir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("nam-freshness-{n}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_file(dir: &Path, rel: &str, content: &[u8]) -> PathBuf {
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let mut f = fs::File::create(&path).unwrap();
    f.write_all(content).unwrap();
    path
}

fn make_manifest(dir: &Path, extra: &str) {
    let manifest = format!("# Golden freshness manifest — test fixture\n{extra}\n");
    write_file(
        dir,
        "tests/fixtures/.golden_manifest.sha256",
        manifest.as_bytes(),
    );
}

fn sha_of(content: &[u8]) -> String {
    Sha256::digest(content)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn test_missing_manifest_is_error() {
    let dir = tempdir();
    let err = check_freshness(&dir, FreshnessMode::HardFail).unwrap_err();
    assert!(err.to_string().contains("manifest not found"));
}

#[test]
fn test_consistent_manifest_passes() {
    let dir = tempdir();
    let content = b"model-a\n";
    let sha = sha_of(content);
    write_file(&dir, "tests/fixtures/models/model_a.nam", content);
    make_manifest(
        &dir,
        &format!(
            "{sha} 0000000000000000000000000000000000000000000000000000000000000000 model_a.nam golden_a.bin\n# MODEL-REGISTRY: model_a.nam"
        ),
    );
    let outcome = check_freshness(&dir, FreshnessMode::HardFail).unwrap();
    assert!(outcome.is_ok());
    assert_eq!(outcome.reason, FreshnessReason::Ok);
}

#[test]
fn test_model_hash_drift_is_stale() {
    let dir = tempdir();
    write_file(&dir, "tests/fixtures/models/model_a.nam", b"model-a\n");
    let wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000";
    make_manifest(
        &dir,
        &format!(
            "{wrong_sha} 0000000000000000000000000000000000000000000000000000000000000000 model_a.nam golden_a.bin\n# MODEL-REGISTRY: model_a.nam"
        ),
    );
    let outcome = check_freshness(&dir, FreshnessMode::ArtifactsHard).unwrap();
    assert!(!outcome.is_ok());
    assert_eq!(outcome.reason, FreshnessReason::StaleFixtures);
    assert_eq!(outcome.stale, vec![PathBuf::from("model_a.nam")]);
}

#[test]
fn test_missing_expected_golden_is_missing() {
    let dir = tempdir();
    let content = b"model-a\n";
    let sha = sha_of(content);
    write_file(&dir, "tests/fixtures/models/model_a.nam", content);
    make_manifest(
        &dir,
        &format!(
            "{sha} 0000000000000000000000000000000000000000000000000000000000000000 model_a.nam golden_a.bin\n# EXPECTED: missing_golden.bin\n# MODEL-REGISTRY: model_a.nam"
        ),
    );
    let outcome = check_freshness(&dir, FreshnessMode::ArtifactsHard).unwrap();
    assert!(!outcome.is_ok());
    assert_eq!(outcome.reason, FreshnessReason::MissingFixtures);
    assert_eq!(outcome.missing, vec![PathBuf::from("missing_golden.bin")]);
}

#[test]
fn test_orphan_model_is_detected() {
    let dir = tempdir();
    let content = b"model-a\n";
    let sha = sha_of(content);
    write_file(&dir, "tests/fixtures/models/model_a.nam", content);
    write_file(&dir, "tests/fixtures/models/orphan.nam", b"orphan\n");
    make_manifest(
        &dir,
        &format!(
            "{sha} 0000000000000000000000000000000000000000000000000000000000000000 model_a.nam golden_a.bin\n# MODEL-REGISTRY: model_a.nam"
        ),
    );
    let outcome = check_freshness(&dir, FreshnessMode::HardFail).unwrap();
    assert!(!outcome.is_ok());
    assert_eq!(outcome.reason, FreshnessReason::OrphanFixture);
    assert_eq!(outcome.orphans, vec![PathBuf::from("orphan.nam")]);
}

#[test]
fn test_generator_drift_is_non_blocking_in_artifacts_hard() {
    let dir = tempdir();
    let content = b"model-a\n";
    let sha = sha_of(content);
    let gen_content = b"generator\n";
    let gen_sha = sha_of(gen_content);
    write_file(&dir, "tests/fixtures/models/model_a.nam", content);
    write_file(&dir, "gen.sh", gen_content);
    make_manifest(
        &dir,
        &format!(
            "{sha} 0000000000000000000000000000000000000000000000000000000000000000 model_a.nam golden_a.bin\n# MODEL-REGISTRY: model_a.nam\n# GENERATORS\n{gen_sha} gen.sh"
        ),
    );
    // Generator matches: outcome OK.
    let ok = check_freshness(&dir, FreshnessMode::ArtifactsHard).unwrap();
    assert!(ok.is_ok());

    // Change generator and recompute manifest sha... actually we can just mutate the file.
    let mut f = fs::File::create(dir.join("gen.sh")).unwrap();
    f.write_all(b"generator modified\n").unwrap();
    let outcome = check_freshness(&dir, FreshnessMode::ArtifactsHard).unwrap();
    assert!(outcome.artifact_integrity_ok);
    assert!(!outcome.generator_provenance_ok);
    assert_eq!(outcome.reason, FreshnessReason::Ok);

    let outcome = check_freshness(&dir, FreshnessMode::HardFail).unwrap();
    assert_eq!(outcome.reason, FreshnessReason::StaleFixtures);
}

#[test]
fn test_warn_only_always_returns_ok_reason() {
    let dir = tempdir();
    write_file(&dir, "tests/fixtures/models/model_a.nam", b"model-a\n");
    let wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000";
    make_manifest(
        &dir,
        &format!(
            "{wrong_sha} 0000000000000000000000000000000000000000000000000000000000000000 model_a.nam golden_a.bin\n# MODEL-REGISTRY: model_a.nam"
        ),
    );
    let outcome = check_freshness(&dir, FreshnessMode::WarnOnly).unwrap();
    assert!(!outcome.is_ok());
    assert_eq!(outcome.reason, FreshnessReason::Ok);
}

#[test]
fn test_invalid_catalog_line_errors() {
    let dir = tempdir();
    write_file(&dir, "tests/fixtures/models/model_a.nam", b"model-a\n");
    make_manifest(&dir, "this is an invalid catalog line with too many words");
    let err = check_freshness(&dir, FreshnessMode::HardFail).unwrap_err();
    assert!(err.to_string().contains("invalid manifest line"));
}

#[test]
fn test_toolchain_fingerprint_parsing() {
    let text = r#"
# TOOLCHAIN: cxx: g++ 15.2.0
# TOOLCHAIN: cmake: cmake 4.2.3
# TOOLCHAIN: glibc: ldd 2.43
# TOOLCHAIN: os: 7.0.0-29-generic
# TOOLCHAIN: cxx-flags: -O3
"#;
    let fp = ToolchainFingerprint::from_manifest(text);
    assert_eq!(fp.cxx.as_deref(), Some("g++ 15.2.0"));
    assert_eq!(fp.cmake.as_deref(), Some("cmake 4.2.3"));
    assert_eq!(fp.glibc.as_deref(), Some("ldd 2.43"));
    assert_eq!(fp.os.as_deref(), Some("7.0.0-29-generic"));
    assert_eq!(fp.cxx_flags.as_deref(), Some("-O3"));
}

#[test]
fn test_toolchain_drift_detection() {
    let manifest = ToolchainFingerprint {
        cxx: Some("g++ 15.2.0".to_string()),
        cmake: Some("cmake 4.2.3".to_string()),
        glibc: Some("ldd 2.43".to_string()),
        os: Some("7.0.0-29-generic".to_string()),
        cxx_flags: None,
    };
    let current = ToolchainFingerprint {
        cxx: Some("g++ 16.0.0".to_string()),
        cmake: Some("cmake 4.2.3".to_string()),
        glibc: Some("ldd 2.43".to_string()),
        os: Some("7.0.0-29-generic".to_string()),
        cxx_flags: None,
    };
    let drift = manifest.drift_against(&current);
    assert_eq!(drift.len(), 1);
    assert!(drift[0].contains("compiler changed"));
}

#[test]
fn test_mtime_detects_stale_artifact() {
    let dir = tempdir();
    let artifact = write_file(&dir, "target/artifact.so", b"artifact");
    thread::sleep(Duration::from_millis(50));
    let src = write_file(&dir, "src/lib.rs", b"source");

    let stale = check_artifact_freshness_mtime(&[src], &[artifact]).unwrap();
    assert_eq!(stale.len(), 1);
}

#[test]
fn test_mtime_passes_when_artifact_is_fresh() {
    let dir = tempdir();
    let src = write_file(&dir, "src/lib.rs", b"source");
    thread::sleep(Duration::from_millis(50));
    let artifact = write_file(&dir, "target/artifact.so", b"artifact");

    let stale = check_artifact_freshness_mtime(&[src], &[artifact]).unwrap();
    assert!(stale.is_empty());
}
