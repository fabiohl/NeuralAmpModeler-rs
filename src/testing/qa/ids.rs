// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Label↔canonical-entity alias tables of the QA engine.
//!
//! Old reports may carry labels that predate the canonical `id`/`label`
//! space of `docs/quality-contract.json` (e.g. quick_parity labels mapped
//! onto the golden_vectors key space by the bash `_LMAP` remap of
//! `parse_jsonl_fidelity`). The verify engine resolves those aliases **only**
//! through this table — never with ad-hoc string matching in `verify.rs`.
//!
//! S2.T3 grew this module with the catalog label↔fixture map (projection of
//! `catalog.rs` `GOLDEN_GEN_CATALOG` — the bash `ESR_F64_MODEL_MAP` is **not**
//! a source) and the explicit `RT_*` benchmark table of
//! `benches/regression_gate.rs` (bench label → contract performance id +
//! fixture). Both tables are guarded by consistency tests against their Rust
//! sources, so they can never silently drift from the catalog or the bench.
//!
//! S3 added the contract-label-family → f64-oracle fixture table: the ingest
//! join of `esr_f64` onto the fidelity records that `verify_contract` reads.
//! Unlike the catalog table, this one **is** a port of the legacy bash
//! `ESR_F64_MODEL_MAP` (pre-`nam_quality` `verify_contract`), because no Rust
//! source predates it — the join key is the family part of the contract
//! fidelity label, which neither the catalog nor the bench table carries.

#[cfg(test)]
use super::super::catalog::GOLDEN_GEN_CATALOG;

/// Resolves an old-report fidelity label to its canonical contract label.
///
/// Ports the bash `_LMAP` remap of `parse_jsonl_fidelity`
/// (`quality-dashboard.sh:569-578`). Returns `None` for labels that are
/// already canonical or unknown.
pub fn resolve_fidelity_alias(label: &str) -> Option<&'static str> {
    match label {
        "Quick ConvNet @48000 Live" => Some("ConvNet Test @48000 Live"),
        _ => None,
    }
}

/// Golden-catalog label ↔ fixture file projection.
///
/// Sourced **from `catalog.rs`** (`GOLDEN_GEN_CATALOG`), not from the bash
/// `ESR_F64_MODEL_MAP`: every entry is mechanically derived from the Rust
/// catalog and `table_is_catalog_projection` enforces the equality. Used by
/// the future oracle-pairing lookups of the verify engine.
pub static FIXTURE_LABEL_TABLE: &[(&str, &str)] = &[
    ("WaveNet Standard (CH=16)", "BossWN-standard.nam"),
    ("WaveNet Lite (CH=12)", "EVH-5150-Lite.nam"),
    ("WaveNet Feather (CH=8)", "BossWN-feather.nam"),
    ("WaveNet Nano (CH=4)", "BossWN-nano.nam"),
    ("WaveNet A1 Standard (Official)", "wavenet_a1_standard.nam"),
    ("WaveNet Official (CH=3 free geom)", "wavenet_official.nam"),
    ("LSTM 1×16", "BossLSTM-1x16.nam"),
    ("LSTM 2×8", "BossLSTM-2x8.nam"),
    ("LSTM Official", "lstm.nam"),
    ("A2-Full (CH=8)", "wavenet_a2_full.nam"),
    ("A2-Lite (CH=3)", "wavenet_a2_lite.nam"),
    ("Condition DSP (CH=3, cond=3)", "wavenet_condition_dsp.nam"),
    (
        "Condition DSP LSTM (CH=3, cond=3, LSTM)",
        "wavenet_condition_lstm.nam",
    ),
    ("SlimmableContainer A2 Example (CH=3→6)", "a2_example.nam"),
    ("APP EVH Stealth 100", "APP-EVH-Stealth100-Dialled-xSTD.nam"),
    ("Boss BD-2 H2O Mod", "Boss BD-2 H2O Mod T-12_00 G-12_00.nam"),
    (
        "SLAMMIN MARSHALL J45",
        "SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam",
    ),
    ("WaveNetDyn Free-Shape (CH=7/4)", "wavenet_dyn_free.nam"),
    ("LSTM-Dyn 1×7", "lstm_dyn_test.nam"),
    ("ConvNet Test (CH=8, 6 blocks)", "convnet_test.nam"),
    (
        "WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1)",
        "wavenet_a2_max.nam",
    ),
    ("A2 Dynamic Gated (CH=8)", "a2_dynamic_gated_ch8.nam"),
    ("A2 Dynamic Blended (CH=3)", "a2_dynamic_blended_ch3.nam"),
    ("A2-FiLM Lite (CH=3)", "wavenet_a2_film_lite.nam"),
    ("A2-FiLM Full (CH=8)", "wavenet_a2_film_full.nam"),
    (
        "A2-FiLM Chaos Stress (CH=3)",
        "wavenet_a2_film_chaos_stress.nam",
    ),
    (
        "A2-FiLM InputMixinPre (CH=3)",
        "wavenet_a2_film_input_mixin_pre.nam",
    ),
    ("Linear FFT RF=320", "linear_fft_rf320.nam"),
    ("Linear FFT RF=2048", "linear_fft_rf2048.nam"),
    ("Linear FFT RF=4096", "linear_fft_rf4096.nam"),
    ("Linear FFT RF=8192", "linear_fft_rf8192.nam"),
    ("LSTM 1×10 (uncat.)", "lstm_1x10.nam"),
    ("LSTM 2×24 (uncat.)", "lstm_2x24.nam"),
    ("LSTM 3×8", "lstm_3x8.nam"),
    ("ConvNet No BatchNorm", "convnet_nobn.nam"),
    ("ConvNet ReLU", "convnet_relu.nam"),
    ("ConvNet SiLU", "convnet_silu.nam"),
    ("Linear No Bias", "linear_nobias.nam"),
    (
        "WaveNet A1 Secondary Activation Rejection",
        "wavenet_a1_secondary_act.nam",
    ),
];

/// Resolves a golden-catalog label to its fixture file, if registered.
pub fn resolve_fixture_by_label(label: &str) -> Option<&'static str> {
    FIXTURE_LABEL_TABLE
        .iter()
        .find(|(catalog_label, _)| *catalog_label == label)
        .map(|(_, fixture)| *fixture)
}

/// One explicit benchmark entry of `benches/regression_gate.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtBenchEntry {
    /// Criterion `bench_function` label of `benches/regression_gate.rs`.
    pub bench_label: &'static str,
    /// Canonical performance id of `docs/quality-contract.json`
    /// (differs from the bench label for `RT_Linear` and the DSP benches).
    pub contract_id: &'static str,
    /// Model fixture file loaded by the bench (`None` for DSP benches).
    pub fixture: Option<&'static str>,
}

/// Explicit `RT_*` table of `benches/regression_gate.rs` — the contract ids
/// are the ones of the current `docs/quality-contract.json` performance
/// section, kept in sync by `rt_contract_ids_match_contract_docs`.
pub static RT_BENCH_TABLE: &[RtBenchEntry] = &[
    RtBenchEntry {
        bench_label: "RT_WaveNet_Std_CH16",
        contract_id: "RT_WaveNet_Std_CH16",
        fixture: Some("BossWN-standard.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_WaveNet_Feather_CH8",
        contract_id: "RT_WaveNet_Feather_CH8",
        fixture: Some("BossWN-feather.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_WaveNet_Lite_CH12",
        contract_id: "RT_WaveNet_Lite_CH12",
        fixture: Some("BossWN-lite.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_WaveNet_Nano_CH4",
        contract_id: "RT_WaveNet_Nano_CH4",
        fixture: Some("BossWN-nano.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_A2_Full_CH8",
        contract_id: "RT_A2_Full_CH8",
        fixture: Some("wavenet_a2_full.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_A2_Lite_CH3",
        contract_id: "RT_A2_Lite_CH3",
        fixture: Some("wavenet_a2_lite.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_LSTM_1x16",
        contract_id: "RT_LSTM_1x16",
        fixture: Some("BossLSTM-1x16.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_LSTM_2x8",
        contract_id: "RT_LSTM_2x8",
        fixture: Some("BossLSTM-2x8.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_Linear",
        contract_id: "RT_Linear_RF2048",
        fixture: Some("linear_test.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_ConvNet",
        contract_id: "RT_ConvNet",
        fixture: Some("convnet_test.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_WaveNet_Dyn_Free",
        contract_id: "RT_WaveNet_Dyn_Free",
        fixture: Some("wavenet_dyn_free.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_LSTM_Dyn_1x7",
        contract_id: "RT_LSTM_Dyn_1x7",
        fixture: Some("lstm_dyn_test.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_A2_Dyn_Gated_CH8",
        contract_id: "RT_A2_Dyn_Gated_CH8",
        fixture: Some("a2_dynamic_gated_ch8.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_A2_Dyn_Blended_CH3",
        contract_id: "RT_A2_Dyn_Blended_CH3",
        fixture: Some("a2_dynamic_blended_ch3.nam"),
    },
    RtBenchEntry {
        bench_label: "RT_DSP_Resampler_44k1_to_48k",
        contract_id: "RT_DSP_Resampler_44k_to_48k",
        fixture: None,
    },
    RtBenchEntry {
        bench_label: "RT_DSP_Resampler_96k_to_48k",
        contract_id: "RT_DSP_Resampler_96k_to_48k",
        fixture: None,
    },
    RtBenchEntry {
        bench_label: "RT_DSP_CabSim_IR_Medium",
        contract_id: "RT_DSP_CabSim_IR_Medium",
        fixture: None,
    },
    RtBenchEntry {
        bench_label: "RT_DSP_Pipeline_Base_NoOS",
        contract_id: "RT_DSP_Pipeline_Base",
        fixture: None,
    },
    RtBenchEntry {
        bench_label: "RT_DSP_Pipeline_HQ_4xOS",
        contract_id: "RT_DSP_Pipeline_HQ",
        fixture: None,
    },
];

/// Resolves a Criterion bench label of `regression_gate.rs` to its canonical
/// contract performance id.
pub fn resolve_rt_contract_id(bench_label: &str) -> Option<&'static str> {
    RT_BENCH_TABLE
        .iter()
        .find(|entry| entry.bench_label == bench_label)
        .map(|entry| entry.contract_id)
}

/// Contract fidelity-label family → f64-oracle fixture projection.
///
/// Port of the legacy bash `ESR_F64_MODEL_MAP` of the pre-`nam_quality`
/// `verify_contract` (removed in the S2 refactor, which lost the `esr_f64`
/// join). The key is the *family* part of a contract fidelity label — the
/// label minus the ` @<rate> Live` suffix (e.g. `BossWN-standard` of
/// `BossWN-standard @48000 Live`); the value is the fixture file whose
/// prewarm-paired ESR the `reference_oracle_f64` phase measures. The ingest
/// joins that value onto the fidelity record as `esr_f64`, restoring the
/// verify key that `verify_contract` reads.
///
/// Guarded by `f64_oracle_fixture_table_covers_every_contract_esr_f64_entry`
/// against `docs/quality-contract.json`, so it can never silently drift from
/// the contract.
pub static F64_ORACLE_FIXTURE_TABLE: &[(&str, &str)] = &[
    ("BossWN-standard", "BossWN-standard.nam"),
    ("BossWN-feather", "BossWN-feather.nam"),
    ("BossWN-nano", "BossWN-nano.nam"),
    ("EVH-5150-Lite", "EVH-5150-Lite.nam"),
    ("wavenet_a1_standard (Official)", "wavenet_a1_standard.nam"),
    (
        "WaveNet Condition DSP (CH=3, cond=3, dynamic path) C++ cross-reference",
        "wavenet_condition_dsp.nam",
    ),
    (
        "WaveNet Official (CH=3, dynamic path) C++ cross-reference",
        "wavenet_official.nam",
    ),
    (
        "WaveNetDyn Free-Shape (CH=7→4, dynamic path) C++ cross-reference",
        "wavenet_dyn_free.nam",
    ),
    ("BossLSTM-1x16", "BossLSTM-1x16.nam"),
    ("BossLSTM-2x8", "BossLSTM-2x8.nam"),
    ("lstm (Official)", "lstm.nam"),
    (
        "LSTM-Dyn 1×7 (dynamic path) C++ cross-reference",
        "lstm_dyn_test.nam",
    ),
    (
        "WaveNet A2-Full (CH=8) C++ cross-reference",
        "wavenet_a2_full.nam",
    ),
    (
        "WaveNet A2-Lite (CH=3) C++ cross-reference",
        "wavenet_a2_lite.nam",
    ),
    (
        "Container A2-Full (CH=8) C++ cross-reference",
        "wavenet_a2_full.nam",
    ),
    (
        "Container A2-Lite (CH=3) C++ cross-reference",
        "wavenet_a2_lite.nam",
    ),
    (
        "Container File A2-Lite (CH=3) C++ cross-reference",
        "wavenet_a2_lite.nam",
    ),
    (
        "Container File A2-Full (CH=8) C++ cross-reference",
        "wavenet_a2_full.nam",
    ),
    (
        "SlimmableContainer A2 Example (CH=3→6) C++ cross-reference",
        // Deliberate: the contract's `esr_f64` baselines for the
        // SlimmableContainer A2 Example and the A2-Lite family are IDENTICAL
        // (1.82e-14) — the legacy bash `ESR_F64_MODEL_MAP` used the A2-Lite
        // fixture for this family, and `--save` recorded its value. The
        // golden catalog's `a2_example.nam` is a different fixture (CH=8)
        // that the f64-oracle phase does not measure.
        "wavenet_a2_lite.nam",
    ),
    (
        "WaveNet A2 Dynamic Gated (CH=8, gated layers 3/23) C++ cross-reference",
        "a2_dynamic_gated_ch8.nam",
    ),
    (
        "WaveNet A2 Dynamic Blended (CH=3, blended layers 2/23) C++ cross-reference",
        "a2_dynamic_blended_ch3.nam",
    ),
    (
        "WaveNet A2-FiLM-Lite (CH=3, FiLM active) C++ cross-reference",
        "wavenet_a2_film_lite.nam",
    ),
    (
        "WaveNet A2-FiLM Chaos Stress (CH=3, FiLM active) C++ cross-reference",
        "wavenet_a2_film_chaos_stress.nam",
    ),
    (
        "WaveNet A2-FiLM-Full (CH=8, FiLM active) C++ cross-reference",
        "wavenet_a2_film_full.nam",
    ),
    (
        "WaveNet A2-FiLM-InputMixinPre (CH=3, input_mixin_pre_film) C++ cross-reference",
        "wavenet_a2_film_input_mixin_pre.nam",
    ),
    ("ConvNet Test", "convnet_test.nam"),
    ("Quick LSTM 1×16", "BossLSTM-1x16.nam"),
    ("Quick WaveNet CH16", "BossWN-standard.nam"),
    ("Quick A2-Full", "wavenet_a2_full.nam"),
];

/// Family part of a contract fidelity label — everything before the
/// ` @<rate> Live` suffix (v1 golden, v2 golden and cpp-parity shapes all
/// end with ` @… Live`). Labels without the suffix resolve to themselves.
pub fn fidelity_label_family(label: &str) -> &str {
    label
        .split_once(" @")
        .map(|(family, _)| family)
        .unwrap_or(label)
}

/// Resolves the family part of a contract fidelity label to the f64-oracle
/// fixture file that measures its prewarm-paired ESR.
pub fn resolve_f64_oracle_fixture(family: &str) -> Option<&'static str> {
    F64_ORACLE_FIXTURE_TABLE
        .iter()
        .find(|(label_family, _)| *label_family == family)
        .map(|(_, fixture)| *fixture)
}

/// Resolves a Criterion bench label to the fixture file it loads.
///
/// Returns `None` for DSP-infrastructure benches, which exercise components
/// instead of model fixtures.
pub fn resolve_rt_fixture(bench_label: &str) -> Option<&'static str> {
    RT_BENCH_TABLE
        .iter()
        .find(|entry| entry.bench_label == bench_label)
        .and_then(|entry| entry.fixture)
}

#[cfg(test)]
mod tests {
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
}
