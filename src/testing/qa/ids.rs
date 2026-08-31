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
#[path = "ids_test.rs"]
mod ids_test;
