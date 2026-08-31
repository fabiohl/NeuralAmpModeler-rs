// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn quick_parity_alias_resolves_to_golden_vectors_label() {
    assert_eq!(
        resolve_fidelity_alias("Quick ConvNet @48000 Live"),
        Some("ConvNet Test @48000 Live")
    );
}

#[test]
fn canonical_and_unknown_labels_have_no_alias() {
    assert_eq!(resolve_fidelity_alias("ConvNet Test @48000 Live"), None);
    assert_eq!(resolve_fidelity_alias("no-such-model"), None);
    assert_eq!(resolve_fidelity_alias(""), None);
}

/// The label↔fixture table is a full projection of `GOLDEN_GEN_CATALOG`:
/// same labels, same fixtures, same cardinality — the bash map is never
/// a source.
#[test]
fn fixture_label_table_is_a_catalog_projection() {
    assert_eq!(
        FIXTURE_LABEL_TABLE.len(),
        GOLDEN_GEN_CATALOG.len(),
        "table must mirror every golden catalog entry"
    );
    for (label, fixture) in FIXTURE_LABEL_TABLE {
        let in_catalog = GOLDEN_GEN_CATALOG
            .iter()
            .any(|entry| entry.label == *label && entry.nam_file == *fixture);
        assert!(in_catalog, "({label}, {fixture}) not in GOLDEN_GEN_CATALOG");
    }
    for entry in GOLDEN_GEN_CATALOG {
        assert_eq!(
            resolve_fixture_by_label(entry.label),
            Some(entry.nam_file),
            "catalog label must resolve to its fixture"
        );
    }
}

#[test]
fn fixture_label_lookup_covers_known_and_unknown_labels() {
    assert_eq!(
        resolve_fixture_by_label("WaveNet Standard (CH=16)"),
        Some("BossWN-standard.nam")
    );
    assert_eq!(
        resolve_fixture_by_label("LSTM 1×10 (uncat.)"),
        Some("lstm_1x10.nam")
    );
    assert_eq!(resolve_fixture_by_label("no-such-model"), None);
    assert_eq!(resolve_fixture_by_label(""), None);
}

/// All 19 bench labels of `benches/regression_gate.rs` are registered.
#[test]
fn rt_table_covers_every_regression_gate_bench() {
    let bench_labels: Vec<&str> = RT_BENCH_TABLE.iter().map(|e| e.bench_label).collect();
    assert_eq!(bench_labels.len(), 19);
    for expected in [
        "RT_WaveNet_Std_CH16",
        "RT_WaveNet_Feather_CH8",
        "RT_WaveNet_Lite_CH12",
        "RT_WaveNet_Nano_CH4",
        "RT_A2_Full_CH8",
        "RT_A2_Lite_CH3",
        "RT_LSTM_1x16",
        "RT_LSTM_2x8",
        "RT_Linear",
        "RT_ConvNet",
        "RT_WaveNet_Dyn_Free",
        "RT_LSTM_Dyn_1x7",
        "RT_A2_Dyn_Gated_CH8",
        "RT_A2_Dyn_Blended_CH3",
        "RT_DSP_Resampler_44k1_to_48k",
        "RT_DSP_Resampler_96k_to_48k",
        "RT_DSP_CabSim_IR_Medium",
        "RT_DSP_Pipeline_Base_NoOS",
        "RT_DSP_Pipeline_HQ_4xOS",
    ] {
        assert!(
            bench_labels.contains(&expected),
            "{expected} missing from RT_BENCH_TABLE"
        );
    }
}

/// Bench labels that differ from their contract id are mapped explicitly.
#[test]
fn rt_aliases_map_bench_label_to_contract_id() {
    assert_eq!(
        resolve_rt_contract_id("RT_Linear"),
        Some("RT_Linear_RF2048")
    );
    assert_eq!(
        resolve_rt_contract_id("RT_DSP_Resampler_44k1_to_48k"),
        Some("RT_DSP_Resampler_44k_to_48k")
    );
    assert_eq!(
        resolve_rt_contract_id("RT_DSP_Pipeline_Base_NoOS"),
        Some("RT_DSP_Pipeline_Base")
    );
    assert_eq!(
        resolve_rt_contract_id("RT_DSP_Pipeline_HQ_4xOS"),
        Some("RT_DSP_Pipeline_HQ")
    );
    assert_eq!(
        resolve_rt_contract_id("RT_WaveNet_Std_CH16"),
        Some("RT_WaveNet_Std_CH16")
    );
    assert_eq!(resolve_rt_contract_id("RT_Unknown"), None);
}

/// Model benches resolve to a catalog fixture; DSP benches resolve to
/// `None` (they exercise components, not model fixtures).
#[test]
fn rt_fixtures_split_models_from_dsp_benches() {
    assert_eq!(
        resolve_rt_fixture("RT_WaveNet_Std_CH16"),
        Some("BossWN-standard.nam")
    );
    assert_eq!(resolve_rt_fixture("RT_Linear"), Some("linear_test.nam"));
    assert_eq!(resolve_rt_fixture("RT_DSP_CabSim_IR_Medium"), None);
    assert_eq!(resolve_rt_fixture("RT_DSP_Pipeline_HQ_4xOS"), None);
}

/// Every contract fidelity entry with a finite `esr_f64` baseline must
/// resolve to an f64-oracle fixture through its label family — the ingest
/// join would otherwise leave the verify key unmeasured and the contract
/// check would fail with `ESR_F64 missing from report`.
#[test]
fn f64_oracle_fixture_table_covers_every_contract_esr_f64_entry() {
    use crate::testing::qa::QualityContract;
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/quality-contract.json");
    let content = std::fs::read_to_string(&path).expect("read docs/quality-contract.json");
    let contract = QualityContract::from_json_str(&content)
        .expect("contract must validate against the schema");
    let mut covered = 0;
    for entry in &contract.fidelity {
        if let Some(baseline) = entry.esr_f64.filter(|v| v.is_finite()) {
            let family = fidelity_label_family(&entry.label);
            let fixture = resolve_f64_oracle_fixture(family);
            assert!(
                fixture.is_some(),
                "contract esr_f64 entry '{}' (family '{family}', baseline {baseline:e}) \
                 has no f64-oracle fixture — ingest cannot join esr_f64",
                entry.label
            );
            covered += 1;
        }
    }
    assert!(
        covered > 0,
        "contract must carry at least one finite esr_f64 baseline"
    );
}

/// The family stripping must match the canonical label shapes: v1 golden
/// (`@48000 Live`), v2 golden (`@<sr> (v2) Live`) and cpp-parity
/// (`@<sr> Live`) all resolve to the same family prefix.
#[test]
fn fidelity_label_family_strips_the_rate_suffix() {
    assert_eq!(
        fidelity_label_family("BossWN-standard @48000 Live"),
        "BossWN-standard"
    );
    assert_eq!(
        fidelity_label_family("Quick A2-Full v2 @48000 Live"),
        "Quick A2-Full v2"
    );
    assert_eq!(
        fidelity_label_family("LSTM-Dyn 1×7 (dynamic path) C++ cross-reference @48000 Live"),
        "LSTM-Dyn 1×7 (dynamic path) C++ cross-reference"
    );
    assert_eq!(
        resolve_f64_oracle_fixture(fidelity_label_family("Quick WaveNet CH16 @48000 Live")),
        Some("BossWN-standard.nam")
    );
    assert_eq!(
        resolve_f64_oracle_fixture(fidelity_label_family(
            "WaveNet A2-Lite (CH=3) C++ cross-reference @48000 Live"
        )),
        Some("wavenet_a2_lite.nam")
    );
    assert_eq!(resolve_f64_oracle_fixture("no-such-family"), None);
}
