// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// The 24 validated V2 golden names (single source of truth snapshot).
const EXPECTED_V2_GOLDEN_NAMES: &[&str] = &[
    "golden_wavenet_standard",
    "golden_wavenet_lite",
    "golden_wavenet_feather",
    "golden_wavenet_nano",
    "golden_wavenet_a1_standard",
    "golden_wavenet_official",
    "golden_lstm_1x16",
    "golden_lstm_2x8",
    "golden_lstm_official",
    "golden_wavenet_a2_full",
    "golden_wavenet_a2_lite",
    "golden_wavenet_condition_dsp",
    "golden_wavenet_condition_lstm",
    "golden_a2_example",
    "golden_wavenet_app_evh",
    "golden_wavenet_boss_bd2",
    "golden_wavenet_slammin_marshall",
    "golden_lstm_1x10",
    "golden_lstm_2x24",
    "golden_lstm_3x8",
    "golden_convnet_nobn",
    "golden_convnet_relu",
    "golden_convnet_silu",
    "golden_linear_nobias",
];

#[test]
fn test_v2_catalog_subset_is_complete_and_unique() {
    let v2 = v2_catalog_entries();
    assert_eq!(v2.len(), EXPECTED_V2_GOLDEN_NAMES.len());

    let mut names: Vec<&str> = v2.iter().map(|e| e.golden_name).collect();
    names.sort_unstable();
    let mut expected = EXPECTED_V2_GOLDEN_NAMES.to_vec();
    expected.sort_unstable();
    assert_eq!(names, expected);

    let mut seen = std::collections::HashSet::new();
    for entry in GOLDEN_GEN_CATALOG {
        assert!(
            seen.insert(entry.golden_name),
            "duplicate golden_name {}",
            entry.golden_name
        );
        assert!(!entry.nam_file.is_empty() && !entry.label.is_empty());
    }
    assert_eq!(GOLDEN_GEN_CATALOG.len(), 39);
}

#[test]
fn test_v2_sample_rates_match_scope() {
    for entry in v2_catalog_entries() {
        let rates = v2_sample_rates_for(entry.nam_file);
        let expected = match entry.v2_scope {
            V2GenScope::AllRates => V2_ALL_SAMPLE_RATES,
            V2GenScope::Exclude192k => V2_EX_192K_SAMPLE_RATES,
            _ => V2_48K_SAMPLE_RATES,
        };
        assert_eq!(rates, expected, "scope mismatch for {}", entry.nam_file);
    }
    assert_eq!(
        v2_sample_rates_for("no_such_model.nam"),
        V2_48K_SAMPLE_RATES
    );
}

#[test]
fn test_emitted_catalog_lines_reparse_to_entries() {
    let catalog_text = golden_gen_catalog_lines();
    let lines: Vec<&str> = catalog_text.lines().collect();
    assert_eq!(lines.len(), GOLDEN_GEN_CATALOG.len());

    for (line, entry) in lines.iter().zip(GOLDEN_GEN_CATALOG.iter()) {
        let fields: Vec<&str> = line.splitn(6, ':').collect();
        assert_eq!(fields[0], entry.nam_file, "nam_file mismatch on {line}");
        assert_eq!(
            fields[1], entry.golden_name,
            "golden_name mismatch on {line}"
        );
        assert_eq!(fields[2], entry.label, "label mismatch on {line}");

        let scope_col = match entry.v2_scope {
            V2GenScope::NoV2 => "none",
            V2GenScope::AllRates => "all",
            V2GenScope::Exclude192k => "all:192000",
            V2GenScope::Sr48kOnly => "48k_only",
        };
        let mut expected_tail = String::from(scope_col);
        if entry.skip_reason.is_some() {
            expected_tail.push_str("::");
            expected_tail.push_str(entry.skip_reason.unwrap_or_default());
        }
        let actual_tail: String = line.splitn(4, ':').nth(3).unwrap_or_default().to_string();
        assert_eq!(actual_tail, expected_tail, "scope tail mismatch on {line}");
    }
}

#[test]
fn test_skip_reason_entries_carry_review_dates() {
    for entry in GOLDEN_GEN_CATALOG {
        if let Some(reason) = entry.skip_reason {
            assert!(
                reason.contains("(20") && reason.contains('-'),
                "skip_reason for {} lacks a (YYYY-MM-DD) review date: {reason}",
                entry.golden_name
            );
        }
    }
}

#[test]
fn test_validate_v2_catalog_ok_on_committed_fixtures() {
    let status = validate_v2_catalog().expect("validation must run in a crate checkout");
    // Every required fixture is committed; optional community models may
    // or may not exist locally, but nothing required may be missing.
    assert!(
        status.is_ok(),
        "unexpected missing required V2 fixtures: {:?}",
        status.missing_required
    );
    assert_eq!(status.entries_checked, EXPECTED_V2_GOLDEN_NAMES.len());
}

/// The 13 v1 golden files (single source of truth snapshot) — mirrors the
/// former bash lists REQUIRED_GOLDEN_MODELS / NONDIST_GOLDEN_MODELS /
/// REQUIRED_CABSIM_GOLDENS in utils/tests-long.sh (removed, Sprint S6-T01).
const EXPECTED_V1_GOLDEN_FILES: &[&str] = &[
    "golden_wavenet_standard.bin",
    "golden_wavenet_feather.bin",
    "golden_wavenet_nano.bin",
    "golden_wavenet_a1_standard.bin",
    "golden_wavenet_a2_full.bin",
    "golden_wavenet_a2_lite.bin",
    "golden_lstm_1x16.bin",
    "golden_lstm_2x8.bin",
    "golden_lstm_official.bin",
    "golden_wavenet_lite.bin",
    "golden_cabsim_cpp_short.bin",
    "golden_cabsim_cpp_medium.bin",
    "golden_cabsim_cpp_long.bin",
];

#[test]
fn test_v1_golden_catalog_is_complete_and_unique() {
    assert_eq!(V1_GOLDEN_CATALOG.len(), EXPECTED_V1_GOLDEN_FILES.len());

    let mut files: Vec<&str> = V1_GOLDEN_CATALOG.iter().map(|e| e.golden_file).collect();
    files.sort_unstable();
    let mut expected = EXPECTED_V1_GOLDEN_FILES.to_vec();
    expected.sort_unstable();
    assert_eq!(files, expected);

    let mut seen = std::collections::HashSet::new();
    for entry in V1_GOLDEN_CATALOG {
        assert!(
            seen.insert(entry.golden_file),
            "duplicate golden_file {}",
            entry.golden_file
        );
        assert!(!entry.description.is_empty());
    }
    assert_eq!(
        V1_GOLDEN_CATALOG
            .iter()
            .filter(|e| e.distribution == V2Distribution::RequiredLocal)
            .count(),
        12,
        "12 RequiredLocal (9 DistributedCore model goldens + 3 CabSim)"
    );
    assert_eq!(
        V1_GOLDEN_CATALOG
            .iter()
            .filter(|e| e.distribution == V2Distribution::OptionalExternal)
            .count(),
        1,
        "1 OptionalExternal (WaveNet Lite)"
    );
}

#[test]
fn test_validate_v1_goldens_ok_on_committed_fixtures() {
    let status = validate_v1_goldens().expect("validation must run in a crate checkout");
    // Every required v1 golden is committed; the WaveNet Lite golden is
    // optional (non-distributable) and may be absent, but nothing required
    // may be missing.
    assert!(
        status.is_ok(),
        "unexpected missing required v1 goldens: {:?}",
        status.missing_required
    );
    assert_eq!(status.entries_checked, EXPECTED_V1_GOLDEN_FILES.len());
}
