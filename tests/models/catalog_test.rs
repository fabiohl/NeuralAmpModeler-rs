// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration test suite for the model catalog and SHA-256 deduplication registry.
//!
//! Validates catalog invariants:
//! - Exactly 51 unique SHA-256 model identities
//! - Exactly 61 catalog file paths mapped
//! - Exactly 10 redundant file aliases identified
//! - Exactly 45 supported models and 6 unsupported models (3 intentional negative, 3 known gaps)
//! - Disk-level SHA-256 verification when fixture files exist on disk

use std::collections::HashSet;
use std::path::PathBuf;

use neural_amp_modeler_rs::testing::catalog::{
    ModelSupportKind, alias_count, catalog_entries, compute_sha256_file, find_by_path,
    find_by_sha256, intentional_negative_count, known_gap_count, supported_count,
    total_catalog_paths, unique_sha_count, unsupported_count,
};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn test_catalog_counts_and_invariants() {
    assert_eq!(
        unique_sha_count(),
        51,
        "Catalog must contain exactly 51 unique SHA-256 identities"
    );
    assert_eq!(
        total_catalog_paths(),
        61,
        "Catalog must map exactly 61 total file paths"
    );
    assert_eq!(
        alias_count(),
        10,
        "Catalog must register exactly 10 redundant aliases"
    );
    assert_eq!(
        supported_count(),
        46,
        "Catalog must classify exactly 46 supported models"
    );
    assert_eq!(
        unsupported_count(),
        5,
        "Catalog must classify exactly 5 unsupported models"
    );
    assert_eq!(
        intentional_negative_count(),
        2,
        "Catalog must classify exactly 2 intentional negative fixtures"
    );
    assert_eq!(
        known_gap_count(),
        3,
        "Catalog must classify exactly 3 known architectural gaps"
    );
}

#[test]
fn test_catalog_uniqueness_and_no_dangling_shas() {
    let entries = catalog_entries();
    let mut seen_shas = HashSet::new();
    let mut seen_paths = HashSet::new();

    for entry in entries {
        // SHA-256 string format validation
        assert_eq!(
            entry.sha256.len(),
            64,
            "SHA-256 string must be 64 characters hex for path {}",
            entry.canonical_path
        );
        assert!(
            entry.sha256.chars().all(|c| c.is_ascii_hexdigit()),
            "SHA-256 must be hex string"
        );

        // Deduplication invariant
        assert!(
            seen_shas.insert(entry.sha256),
            "Duplicate SHA-256 found in catalog entries: {}",
            entry.sha256
        );

        // Path uniqueness
        assert!(
            seen_paths.insert(entry.canonical_path),
            "Duplicate canonical path in catalog: {}",
            entry.canonical_path
        );

        for alias in entry.aliases {
            assert!(
                seen_paths.insert(alias),
                "Duplicate alias path in catalog: {}",
                alias
            );
        }

        // Lookup consistency
        let found_sha = find_by_sha256(entry.sha256);
        assert!(
            found_sha.is_some(),
            "find_by_sha256 failed for {}",
            entry.sha256
        );
        assert_eq!(found_sha.unwrap().sha256, entry.sha256);

        let found_canonical = find_by_path(std::path::Path::new(entry.canonical_path));
        assert!(
            found_canonical.is_some(),
            "find_by_path failed for canonical path {}",
            entry.canonical_path
        );
        assert_eq!(found_canonical.unwrap().sha256, entry.sha256);

        for alias in entry.aliases {
            let found_alias = find_by_path(std::path::Path::new(alias));
            assert!(
                found_alias.is_some(),
                "find_by_path failed for alias path {}",
                alias
            );
            assert_eq!(found_alias.unwrap().sha256, entry.sha256);
        }
    }
}

#[test]
fn test_disk_sha256_verification() {
    let manifest = manifest_dir();
    let workspace_root = manifest.parent().unwrap_or(&manifest);
    let entries = catalog_entries();
    let mut verified_files = 0;

    let resolve_path = |p_str: &str| -> Option<PathBuf> {
        let p = std::path::Path::new(p_str);
        let cand1 = manifest.join(p);
        if cand1.exists() {
            return Some(cand1);
        }
        let cand2 = workspace_root.join(p);
        if cand2.exists() {
            return Some(cand2);
        }
        None
    };

    for entry in entries {
        // Check canonical file
        if let Some(canonical_full) = resolve_path(entry.canonical_path) {
            let computed = compute_sha256_file(&canonical_full).expect("Failed to compute SHA-256");
            assert_eq!(
                computed, entry.sha256,
                "Disk file SHA-256 mismatch for canonical path {}",
                entry.canonical_path
            );
            verified_files += 1;
        }

        // Check alias files
        for alias in entry.aliases {
            if let Some(alias_full) = resolve_path(alias) {
                let computed = compute_sha256_file(&alias_full).expect("Failed to compute SHA-256");
                assert_eq!(
                    computed, entry.sha256,
                    "Disk file SHA-256 mismatch for alias path {}",
                    alias
                );
                verified_files += 1;
            }
        }
    }

    eprintln!("Verified SHA-256 checksums on disk for {verified_files} existing fixture files.");
    assert!(
        verified_files > 0,
        "At least one catalog model file must exist on disk for testing"
    );
}

#[test]
fn test_support_classification_contract() {
    let entries = catalog_entries();
    for entry in entries {
        match entry.support {
            ModelSupportKind::Supported => {
                assert!(
                    !entry.description.is_empty(),
                    "Supported entry must have non-empty description"
                );
            }
            ModelSupportKind::IntentionalNegative => {
                assert!(
                    entry.description.contains("Intentional negative"),
                    "Intentional negative fixture description missing required marker: {}",
                    entry.description
                );
            }
            ModelSupportKind::KnownGap => {
                assert!(
                    entry.description.contains("Known architectural gap"),
                    "Known gap fixture description missing required marker: {}",
                    entry.description
                );
            }
        }
    }
}
