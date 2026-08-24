// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration gate for the Rust freshness verifier.
//!
//! Exercises `testing::freshness` end-to-end with synthetic manifests, ensuring
//! the same contract expected by the `nam_freshness` CLI (`check_freshness`).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use neural_amp_modeler_rs::testing::freshness::{
    FreshnessMode, FreshnessReason, check_artifact_freshness_mtime, check_freshness,
};
use sha2::{Digest, Sha256};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tempdir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("nam-freshness-guard-{n}"));
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

fn sha_of(content: &[u8]) -> String {
    Sha256::digest(content)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn make_manifest(dir: &Path, body: &str) {
    let text = format!("# Golden freshness manifest — test fixture\n{body}\n");
    write_file(
        dir,
        "tests/fixtures/.golden_manifest.sha256",
        text.as_bytes(),
    );
}

#[test]
fn consistent_manifest_passes_artifacts_hard() {
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
    let outcome = check_freshness(&dir, FreshnessMode::ArtifactsHard).unwrap();
    assert!(outcome.is_ok());
    assert_eq!(outcome.reason, FreshnessReason::Ok);
}

#[test]
fn stale_model_hash_returns_stale_fixtures() {
    let dir = tempdir();
    write_file(&dir, "tests/fixtures/models/model_a.nam", b"model-a\n");
    make_manifest(
        &dir,
        "0000000000000000000000000000000000000000000000000000000000000000 0000000000000000000000000000000000000000000000000000000000000000 model_a.nam golden_a.bin\n# MODEL-REGISTRY: model_a.nam",
    );
    let outcome = check_freshness(&dir, FreshnessMode::ArtifactsHard).unwrap();
    assert_eq!(outcome.reason, FreshnessReason::StaleFixtures);
    assert!(outcome.stale.iter().any(|p| p == "model_a.nam"));
}

#[test]
fn missing_expected_golden_returns_missing_fixtures() {
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
    assert_eq!(outcome.reason, FreshnessReason::MissingFixtures);
}

#[test]
fn orphan_model_returns_orphan_fixture() {
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
    assert_eq!(outcome.reason, FreshnessReason::OrphanFixture);
}

#[test]
fn generator_drift_is_non_blocking_in_artifacts_hard() {
    let dir = tempdir();
    let content = b"model-a\n";
    let sha = sha_of(content);
    let generator = b"generator\n";
    let gen_sha = sha_of(generator);
    write_file(&dir, "tests/fixtures/models/model_a.nam", content);
    write_file(&dir, "gen.sh", generator);
    make_manifest(
        &dir,
        &format!(
            "{sha} 0000000000000000000000000000000000000000000000000000000000000000 model_a.nam golden_a.bin\n# MODEL-REGISTRY: model_a.nam\n# GENERATORS\n{gen_sha} gen.sh"
        ),
    );

    let fresh = check_freshness(&dir, FreshnessMode::ArtifactsHard).unwrap();
    assert!(fresh.is_ok());

    let mut f = fs::File::create(dir.join("gen.sh")).unwrap();
    f.write_all(b"generator modified\n").unwrap();

    let artifacts_hard = check_freshness(&dir, FreshnessMode::ArtifactsHard).unwrap();
    assert!(artifacts_hard.artifact_integrity_ok);
    assert!(!artifacts_hard.generator_provenance_ok);
    assert_eq!(artifacts_hard.reason, FreshnessReason::Ok);

    let hard_fail = check_freshness(&dir, FreshnessMode::HardFail).unwrap();
    assert!(!hard_fail.is_ok());
    assert_eq!(hard_fail.reason, FreshnessReason::StaleFixtures);
}

#[test]
fn warn_only_absorbs_failures() {
    let dir = tempdir();
    write_file(&dir, "tests/fixtures/models/model_a.nam", b"model-a\n");
    make_manifest(
        &dir,
        "0000000000000000000000000000000000000000000000000000000000000000 0000000000000000000000000000000000000000000000000000000000000000 model_a.nam golden_a.bin\n# MODEL-REGISTRY: model_a.nam",
    );
    let outcome = check_freshness(&dir, FreshnessMode::WarnOnly).unwrap();
    assert_eq!(outcome.reason, FreshnessReason::Ok);
}

#[test]
fn mtime_detects_stale_artifact() {
    let dir = tempdir();
    let artifact = write_file(&dir, "target/artifact.so", b"artifact");
    thread::sleep(Duration::from_millis(50));
    let source = write_file(&dir, "src/lib.rs", b"source");
    let stale = check_artifact_freshness_mtime(&[source], &[artifact]).unwrap();
    assert_eq!(stale.len(), 1);
}

#[test]
fn mtime_passes_fresh_artifact() {
    let dir = tempdir();
    let source = write_file(&dir, "src/lib.rs", b"source");
    thread::sleep(Duration::from_millis(50));
    let artifact = write_file(&dir, "target/artifact.so", b"artifact");
    let stale = check_artifact_freshness_mtime(&[source], &[artifact]).unwrap();
    assert!(stale.is_empty());
}
