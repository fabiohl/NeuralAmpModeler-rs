// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! V2 multi-sample-rate golden generation catalog.

// =============================================================================
// V2 Golden Catalog — single source of truth
// =============================================================================
//
// This registry defines the canonical V2 multi-SR golden matrix: which models
// participate, which sample rates each model requires, which fixture files are
// expected on disk, and the distribution policy for each file. It is the ONLY
// definition — `utils/tests-long.sh` no longer carries bash arrays and
// `tests/fixtures/golden_gen_build.sh` sources its catalog from here (via the
// `nam_golden_catalog` binary), so model lists never appear twice in shell.
//
// The v2_scope column semantics preserved from the former generator CATALOG:
//   all       — v2 golden at all 5 sample rates (44100, 48000, 88200, 96000, 192000)
//   all:192000 — all rates except 192 kHz (LSTM recurrent drift over the 5s stress)
//   48k_only  — only 48000 Hz (model declares expected_sample_rate=48000, or the
//                C++ render tool rejects other SRs)
//   none      — no v2 golden generation for this model
//
// Models with `in_v2_catalog == true` (24) are the validated V2 subset checked
// by `validate_v2_catalog`/`catalog_preflight`. The remaining generator
// registry entries (15) are v1-only models (v2_scope=none) or v1 models that
// carry an incidental unrequired v2@48k golden (KB-A2-MAX + dyn/convnet v1
// fixtures): they keep generating but are never preflight gates.

/// Sample-rate scope of a golden catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2GenScope {
    /// No v2 golden generation for this model (v1-only).
    NoV2,
    /// All 5 sample rates (44100, 48000, 88200, 96000, 192000).
    AllRates,
    /// All rates except 192 kHz (LSTM recurrent drift over the 5s stress).
    Exclude192k,
    /// Only 48000 Hz (model declares `expected_sample_rate=48000`).
    Sr48kOnly,
}

/// Distribution policy for a V2 catalog file (model fixture or golden binary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2Distribution {
    /// Must exist on disk — preflight hard-fail when absent.
    RequiredLocal,
    /// Non-distributable community model — skip gracefully when absent.
    OptionalExternal,
}

/// Single entry of the golden generation registry.
///
/// Fields map 1:1 to the former colon-format entries of
/// `tests/fixtures/golden_gen_build.sh`'s bash CATALOG (which no longer
/// exists): `nam_file:golden_name:label:v2_scope[:skip_srs[:skip_reason]]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldenGenEntry {
    /// Model fixture basename (resolved via `fixtures::model_path`).
    pub nam_file: &'static str,
    /// Golden name prefix used to derive `golden_{name}[_v2_{sr}].bin`.
    pub golden_name: &'static str,
    /// Human-readable model label (generator echo output).
    pub label: &'static str,
    /// Sample-rate scope for v2 golden generation.
    pub v2_scope: V2GenScope,
    /// Whether this entry is part of the validated V2 catalog (preflight gate).
    pub in_v2_catalog: bool,
    /// Distribution policy of the model fixture file itself.
    pub model_distribution: V2Distribution,
    /// Distribution policy of the v2 golden binary files.
    pub golden_distribution: V2Distribution,
    /// Non-empty when the model is intentionally skipped in generation loops
    /// (upstream limitation / rejection fixture). Must carry a `(YYYY-MM-DD)`
    /// review date — enforced by `threshold_calibration::test_catalog_anti_placebo_audit`.
    pub skip_reason: Option<&'static str>,
}

/// All 5 sample rates of the v2 multi-SR matrix.
pub const V2_ALL_SAMPLE_RATES: &[u32] = &[44100, 48000, 88200, 96000, 192000];

/// The 4 sample rates of the `Exclude192k` scope.
pub const V2_EX_192K_SAMPLE_RATES: &[u32] = &[44100, 48000, 88200, 96000];

/// The single 48 kHz sample rate of the `Sr48kOnly` scope.
pub const V2_48K_SAMPLE_RATES: &[u32] = &[48000];

/// Canonical golden generation registry — 39 entries (24 validated V2 + 15 v1-only).
pub static GOLDEN_GEN_CATALOG: &[GoldenGenEntry] = &[
    GoldenGenEntry {
        nam_file: "BossWN-standard.nam",
        golden_name: "golden_wavenet_standard",
        label: "WaveNet Standard (CH=16)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "EVH-5150-Lite.nam",
        golden_name: "golden_wavenet_lite",
        label: "WaveNet Lite (CH=12)",
        v2_scope: V2GenScope::AllRates,
        in_v2_catalog: true,
        model_distribution: V2Distribution::OptionalExternal,
        golden_distribution: V2Distribution::OptionalExternal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "BossWN-feather.nam",
        golden_name: "golden_wavenet_feather",
        label: "WaveNet Feather (CH=8)",
        v2_scope: V2GenScope::AllRates,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "BossWN-nano.nam",
        golden_name: "golden_wavenet_nano",
        label: "WaveNet Nano (CH=4)",
        v2_scope: V2GenScope::AllRates,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a1_standard.nam",
        golden_name: "golden_wavenet_a1_standard",
        label: "WaveNet A1 Standard (Official)",
        v2_scope: V2GenScope::AllRates,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_official.nam",
        golden_name: "golden_wavenet_official",
        label: "WaveNet Official (CH=3 free geom)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "BossLSTM-1x16.nam",
        golden_name: "golden_lstm_1x16",
        label: "LSTM 1×16",
        v2_scope: V2GenScope::Exclude192k,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "BossLSTM-2x8.nam",
        golden_name: "golden_lstm_2x8",
        label: "LSTM 2×8",
        v2_scope: V2GenScope::Exclude192k,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "lstm.nam",
        golden_name: "golden_lstm_official",
        label: "LSTM Official",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_full.nam",
        golden_name: "golden_wavenet_a2_full",
        label: "A2-Full (CH=8)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_lite.nam",
        golden_name: "golden_wavenet_a2_lite",
        label: "A2-Lite (CH=3)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_condition_dsp.nam",
        golden_name: "golden_wavenet_condition_dsp",
        label: "Condition DSP (CH=3, cond=3)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_condition_lstm.nam",
        golden_name: "golden_wavenet_condition_lstm",
        label: "Condition DSP LSTM (CH=3, cond=3, LSTM)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: Some(
            "Concordant rejection: Rust fails closed by policy (LSTM condition_dsp \
             not supported); C++ NAMCore rejects the same fixture because its \
             condition_size (3) does not match the LSTM condition DSP output \
             channels (1, the upstream default out_channels) — no golden binary \
             can be generated (2026-10-15)",
        ),
    },
    GoldenGenEntry {
        nam_file: "a2_example.nam",
        golden_name: "golden_a2_example",
        label: "SlimmableContainer A2 Example (CH=3→6)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "APP-EVH-Stealth100-Dialled-xSTD.nam",
        golden_name: "golden_wavenet_app_evh",
        label: "APP EVH Stealth 100",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::OptionalExternal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "Boss BD-2 H2O Mod T-12_00 G-12_00.nam",
        golden_name: "golden_wavenet_boss_bd2",
        label: "Boss BD-2 H2O Mod",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::OptionalExternal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam",
        golden_name: "golden_wavenet_slammin_marshall",
        label: "SLAMMIN MARSHALL J45",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::OptionalExternal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // ── v1-only models (in_v2_catalog = false) ────────────────────────────
    // wavenet_dyn_free / lstm_dyn_test / convnet_test carry an incidental
    // unrequired v2@48k golden; wavenet_a2_max carries an unrequired v2@48k
    // golden (parity verified, not a preflight gate).
    GoldenGenEntry {
        nam_file: "wavenet_dyn_free.nam",
        golden_name: "golden_wavenet_dyn_free",
        label: "WaveNetDyn Free-Shape (CH=7/4)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "lstm_dyn_test.nam",
        golden_name: "golden_lstm_dyn_test",
        label: "LSTM-Dyn 1×7",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "convnet_test.nam",
        golden_name: "golden_convnet_test",
        label: "ConvNet Test (CH=8, 6 blocks)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_max.nam",
        golden_name: "golden_wavenet_a2_max",
        label: "WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // Dynamic/FiLM models: v2_scope=none — C++ a2_fast render path rejects
    // FiLM-conditioned models and the generic Eigen engine does not support
    // multi-SR FiLM rendering; dynamic fallback coverage is a superset at test
    // time (see rationale comment in tests/fixtures/golden_gen_build.sh §7).
    GoldenGenEntry {
        nam_file: "a2_dynamic_gated_ch8.nam",
        golden_name: "golden_a2_dynamic_gated_ch8",
        label: "A2 Dynamic Gated (CH=8)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "a2_dynamic_blended_ch3.nam",
        golden_name: "golden_a2_dynamic_blended_ch3",
        label: "A2 Dynamic Blended (CH=3)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_film_lite.nam",
        golden_name: "golden_wavenet_a2_film_lite",
        label: "A2-FiLM Lite (CH=3)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_film_full.nam",
        golden_name: "golden_wavenet_a2_film_full",
        label: "A2-FiLM Full (CH=8)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_film_chaos_stress.nam",
        golden_name: "golden_wavenet_a2_film_chaos_stress",
        label: "A2-FiLM Chaos Stress (CH=3)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_film_input_mixin_pre.nam",
        golden_name: "golden_wavenet_a2_film_input_mixin_pre",
        label: "A2-FiLM InputMixinPre (CH=3)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "linear_fft_rf320.nam",
        golden_name: "golden_linear_fft_rf320",
        label: "Linear FFT RF=320",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "linear_fft_rf2048.nam",
        golden_name: "golden_linear_fft_rf2048",
        label: "Linear FFT RF=2048",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "linear_fft_rf4096.nam",
        golden_name: "golden_linear_fft_rf4096",
        label: "Linear FFT RF=4096",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "linear_fft_rf8192.nam",
        golden_name: "golden_linear_fft_rf8192",
        label: "Linear FFT RF=8192",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // LSTM uncatalogued hidden sizes and 3-layer topology
    GoldenGenEntry {
        nam_file: "lstm_1x10.nam",
        golden_name: "golden_lstm_1x10",
        label: "LSTM 1×10 (uncat.)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "lstm_2x24.nam",
        golden_name: "golden_lstm_2x24",
        label: "LSTM 2×24 (uncat.)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "lstm_3x8.nam",
        golden_name: "golden_lstm_3x8",
        label: "LSTM 3×8",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // ConvNet variants (nobn, ReLU, SiLU)
    GoldenGenEntry {
        nam_file: "convnet_nobn.nam",
        golden_name: "golden_convnet_nobn",
        label: "ConvNet No BatchNorm",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "convnet_relu.nam",
        golden_name: "golden_convnet_relu",
        label: "ConvNet ReLU",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "convnet_silu.nam",
        golden_name: "golden_convnet_silu",
        label: "ConvNet SiLU",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // Linear without bias
    GoldenGenEntry {
        nam_file: "linear_nobias.nam",
        golden_name: "golden_linear_nobias",
        label: "Linear No Bias",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // Rejection test fixture — model parses, golden generation intentionally skipped
    GoldenGenEntry {
        nam_file: "wavenet_a1_secondary_act.nam",
        golden_name: "golden_wavenet_a1_secondary_act",
        label: "WaveNet A1 Secondary Activation Rejection",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: Some("Rejection fixture for non-null secondary_activation (2026-08-03)"),
    },
];

/// Returns the full golden generation registry (39 entries).
pub fn golden_gen_entries() -> &'static [GoldenGenEntry] {
    GOLDEN_GEN_CATALOG
}

/// Returns the validated V2 catalog subset (24 entries).
pub fn v2_catalog_entries() -> Vec<&'static GoldenGenEntry> {
    GOLDEN_GEN_CATALOG
        .iter()
        .filter(|e| e.in_v2_catalog)
        .collect()
}

/// Looks up the sample-rate slice for a model from the canonical V2 catalog.
///
/// Falls back to 48 kHz only with a warning if the model is not registered.
pub fn v2_sample_rates_for(nam_file: &str) -> &'static [u32] {
    for entry in GOLDEN_GEN_CATALOG {
        if entry.in_v2_catalog && entry.nam_file == nam_file {
            return match entry.v2_scope {
                V2GenScope::AllRates => V2_ALL_SAMPLE_RATES,
                V2GenScope::Exclude192k => V2_EX_192K_SAMPLE_RATES,
                _ => V2_48K_SAMPLE_RATES,
            };
        }
    }
    eprintln!(
        "WARNING: {nam_file} not in V2 golden catalog — defaulting to 48 kHz only. \
         Add the model to GOLDEN_GEN_CATALOG in src/testing/catalog/golden.rs."
    );
    V2_48K_SAMPLE_RATES
}

/// Serializes the golden generation registry in the shell catalog line format.
///
/// One line per entry, colon-separated:
/// `nam_file:golden_name:label:v2_scope[:skip_srs[:skip_reason]]`.
/// Consumed by `tests/fixtures/golden_gen_build.sh` (via the
/// `nam_golden_catalog` binary), replacing the former static bash CATALOG array.
pub fn golden_gen_catalog_lines() -> String {
    let mut out = String::new();
    for entry in GOLDEN_GEN_CATALOG {
        out.push_str(entry.nam_file);
        out.push(':');
        out.push_str(entry.golden_name);
        out.push(':');
        out.push_str(entry.label);
        out.push(':');
        match entry.v2_scope {
            V2GenScope::NoV2 => out.push_str("none"),
            V2GenScope::AllRates => out.push_str("all"),
            V2GenScope::Exclude192k => out.push_str("all:192000"),
            V2GenScope::Sr48kOnly => out.push_str("48k_only"),
        }
        if let Some(reason) = entry.skip_reason {
            out.push_str("::");
            out.push_str(reason);
        }
        out.push('\n');
    }
    out
}
