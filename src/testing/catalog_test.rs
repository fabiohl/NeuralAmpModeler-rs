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
/// REQUIRED_CABSIM_GOLDENS in utils/tests-long.sh.
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

// =============================================================================
// Reference Architecture Manifesto Tests [F-INOV-02]
// =============================================================================

#[test]
fn test_reference_architectures_covers_all_families_fail_closed() {
    let specs = reference_architectures();
    assert!(
        !specs.is_empty(),
        "reference_architectures() must not be empty"
    );

    // Fail-closed anti-drift guard: every single family defined in ArchitectureFamily
    // MUST have at least one registered reference fixture in REFERENCE_ARCHITECTURES.
    for &family in ArchitectureFamily::ALL {
        let matching: Vec<_> = specs.iter().filter(|s| s.family == family).collect();
        assert!(
            !matching.is_empty(),
            "Architecture drift detected: family {family:?} ({family}) has zero \
             reference fixtures in reference_architectures()! \
             Every supported architecture family must have >=1 canonical fixture.",
        );

        let resolved = reference_architecture_for(family);
        assert!(
            resolved.is_some(),
            "reference_architecture_for({family:?}) returned None despite matching entries"
        );
        assert_eq!(resolved.unwrap().family, family);
    }
}

#[test]
fn test_reference_architectures_fixtures_exist_and_load() {
    let specs = reference_architectures();
    let sys = crate::SystemSnapshot::capture();

    for spec in specs {
        assert!(
            spec.exists(),
            "Canonical reference fixture {:?} ({}) does not exist on disk at {:?}!",
            spec.nam_file,
            spec.family,
            spec.resolve_path(),
        );

        assert!(
            spec.suggested_sample_rate >= 44100,
            "suggested_sample_rate {} for {:?} must be a standard pro-audio rate",
            spec.suggested_sample_rate,
            spec.nam_file
        );
        assert!(
            spec.suggested_quantum > 0 && spec.suggested_quantum.is_power_of_two(),
            "suggested_quantum {} for {:?} must be a power-of-two block size",
            spec.suggested_quantum,
            spec.nam_file
        );
        assert!(
            !spec.description.is_empty(),
            "spec {:?} lacks a description",
            spec.nam_file
        );

        // Verify model parses, builds, and prewarms successfully with the engine
        let path = spec.resolve_path();
        let pair = crate::loader::load_and_build_model(
            &path,
            &sys,
            false,
            crate::loader::LoadOptions::default(),
        )
        .unwrap_or_else(|e| {
            panic!(
                "Failed to load reference architecture fixture {:?} ({}): {e}",
                spec.nam_file, spec.family
            )
        });

        assert!(
            pair.model_l.is_some(),
            "Reference architecture {:?} loaded with None model_l",
            spec.nam_file
        );
    }
}

#[test]
fn test_architecture_family_roundtrip_and_display() {
    use std::str::FromStr;

    for &family in ArchitectureFamily::ALL {
        let ident = family.as_str();
        assert!(!ident.is_empty());
        let display = family.display_name();
        assert!(!display.is_empty());

        // Display trait uses display_name
        assert_eq!(format!("{family}"), display);

        // FromStr parses machine ident
        let parsed = ArchitectureFamily::from_str(ident).expect("ident must parse");
        assert_eq!(parsed, family);

        // FromStr parses case-insensitively with dashes
        let dashed = ident.replace('_', "-");
        let parsed_dashed = ArchitectureFamily::from_str(&dashed).expect("dashed must parse");
        assert_eq!(parsed_dashed, family);

        // Serde JSON roundtrip
        let json = serde_json::to_string(&family).expect("must serialize");
        assert_eq!(json, format!("\"{ident}\""));
        let de: ArchitectureFamily = serde_json::from_str(&json).expect("must deserialize");
        assert_eq!(de, family);
    }

    // Invalid string parsing returns error
    let err = ArchitectureFamily::from_str("nonexistent_arch").unwrap_err();
    assert!(err.to_string().contains("nonexistent_arch"));
}

#[test]
fn test_anti_drift_engine_models_match_architecture_families() {
    // Structural mapping audit: every model family instantiated by the engine
    // aligns with exactly one ArchitectureFamily variant.
    use crate::models::StaticModel;

    // Helper checking mapping classification logic
    fn classify_static_model(model: &StaticModel) -> ArchitectureFamily {
        match model {
            StaticModel::WavenetStandard(_)
            | StaticModel::WavenetLite(_)
            | StaticModel::WavenetFeather(_)
            | StaticModel::WavenetNano(_)
            | StaticModel::WavenetDyn(_) => ArchitectureFamily::WaveNetA1,

            StaticModel::WavenetA2Full(_)
            | StaticModel::WavenetA2Lite(_)
            | StaticModel::WavenetA2Dyn(_)
            | StaticModel::WavenetA2Cascade(_) => ArchitectureFamily::WaveNetA2,

            StaticModel::Lstm1x3(_)
            | StaticModel::Lstm1x8(_)
            | StaticModel::Lstm1x12(_)
            | StaticModel::Lstm1x16(_)
            | StaticModel::Lstm1x24(_)
            | StaticModel::Lstm2x8(_)
            | StaticModel::Lstm2x12(_)
            | StaticModel::Lstm2x16(_)
            | StaticModel::Lstm1x40(_)
            | StaticModel::Lstm2x24(_)
            | StaticModel::LstmDyn(_) => ArchitectureFamily::Lstm,

            StaticModel::ConvNet(_) => ArchitectureFamily::ConvNet,

            StaticModel::Linear(_) => ArchitectureFamily::Linear,

            StaticModel::Container(_) => {
                // SlimmableContainer bundles WaveNet / A2 models
                ArchitectureFamily::WaveNetA2
            }
        }
    }

    // Load one fixture of each family to confirm classify_static_model produces the expected family
    let sys = crate::SystemSnapshot::capture();
    for spec in reference_architectures() {
        let pair = crate::loader::load_and_build_model(
            &spec.resolve_path(),
            &sys,
            false,
            crate::loader::LoadOptions::default(),
        )
        .expect("load must succeed");

        let model = pair.model_l.expect("model_l must exist");
        let classified = classify_static_model(&model);
        assert_eq!(
            classified, spec.family,
            "Classified family mismatch for {:?}",
            spec.nam_file
        );
    }
}
