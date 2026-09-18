// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! One-shot transcription of `docs/quality-contract.txt` → `docs/quality-contract.json`.
//!
//! Dev tool only. The production code never parses the
//! ASCII snapshot; this test holds the reviewed hardcoded table and prints the
//! canonical JSON through the typed schema of this module.
//!
//! Regenerate with:
//!
//! ```text
//! cargo test --features testing --lib qa::transcription -- --ignored --nocapture \
//!     > docs/quality-contract.json
//! ```
//!
//! Source snapshot: dashboard run (release, clean tree), measured 2026-09-17
//! 22:04:30 -03, commit `7576d49305bb` (clean), run `1789692961050429696-14880`.
//! commit `0e22ea4ec247` (dirty), run `1786537203076204151-15755`.

use super::*;

fn fid(
    id: &str,
    label: &str,
    esr_namcore: f64,
    esr_f64: Option<f64>,
    snr_db: Option<f64>,
    mrstft: f64,
) -> FidelityEntry {
    FidelityEntry {
        id: id.into(),
        label: label.into(),
        esr_namcore,
        esr_f64,
        snr_db,
        mrstft,
        optional: false,
    }
}

fn fid_optional(
    id: &str,
    label: &str,
    esr_namcore: f64,
    esr_f64: Option<f64>,
    snr_db: Option<f64>,
    mrstft: f64,
) -> FidelityEntry {
    FidelityEntry {
        id: id.into(),
        label: label.into(),
        esr_namcore,
        esr_f64,
        snr_db,
        mrstft,
        optional: true,
    }
}

fn perf(id: &str, label: &str, median_latency_us: f64) -> PerformanceEntry {
    PerformanceEntry {
        id: id.into(),
        label: label.into(),
        median_latency_us,
        batch_factor: None,
        unit: None,
    }
}

/// Micro-batch perf entry: total time of `batch_factor` blocks per Criterion
/// sample, reported in per-block contract units.
fn perf_micro_batch(
    id: &str,
    label: &str,
    median_latency_us: f64,
    batch_factor: u64,
) -> PerformanceEntry {
    PerformanceEntry {
        batch_factor: Some(batch_factor),
        unit: Some("per_block_us".into()),
        ..perf(id, label, median_latency_us)
    }
}

#[test]
#[ignore]
fn transcribe_quality_contract_to_json() {
    let contract = QualityContract {
        schema_version: SCHEMA_VERSION,
        schema_notes: Some(
            "Benchmarks com `batch_factor > 1` reportam tempo total de \
             `batch_factor` blocos; o dashboard divide por `batch_factor` \
             antes de comparar ao threshold."
                .into(),
        ),
        generated_at: "2026-09-17T22:04:30-03:00".into(),
        provenance: Provenance {
            git_commit: "7576d49305bb079ec6435516d97b0e415e524466".into(),
            git_dirty: false,
            run_id: "1789692961050429696-14880".into(),
            effective_isa: "x86-64-v3 (AVX2/FMA/F16C/BMI)".into(),
            cpu_model: "AMD Ryzen 7 5700U with Radeon Graphics".into(),
            rustc: "rustc 1.98.1 (48a229cea 2026-09-01)".into(),
            cargo_profile: "release".into(),
        },
        envelopes: Envelopes::policy_v1(),
        fidelity: vec![
            fid(
                "bosslstm-1x16@48000:live",
                "BossLSTM-1x16 @48000 Live",
                8.51e-12,
                Some(9.08e-13),
                Some(110.7),
                2.82e-05,
            ),
            fid(
                "bosslstm-2x8@48000:live",
                "BossLSTM-2x8 @48000 Live",
                1.00e-11,
                Some(5.78e-13),
                Some(110.0),
                1.65e-05,
            ),
            fid(
                "bosswn-feather@48000:live",
                "BossWN-feather @48000 Live",
                4.74e-14,
                Some(2.00e-14),
                Some(133.2),
                8.86e-06,
            ),
            fid(
                "bosswn-nano@48000:live",
                "BossWN-nano @48000 Live",
                6.43e-14,
                Some(3.05e-14),
                Some(131.9),
                7.67e-06,
            ),
            fid(
                "bosswn-standard@48000:live",
                "BossWN-standard @48000 Live",
                2.31e-14,
                Some(9.05e-15),
                Some(136.4),
                6.46e-06,
            ),
            fid(
                "convnet-test@48000:live",
                "ConvNet Test @48000 Live",
                6.10e-16,
                Some(4.93e-16),
                Some(152.1),
                4.99e-07,
            ),
            fid_optional(
                "evh-5150-lite@48000:live",
                "EVH-5150-Lite @48000 Live",
                7.87e-13,
                Some(2.64e-13),
                Some(121.0),
                4.31e-06,
            ),
            fid(
                "lstm-dyn-1x7@48000:live",
                "LSTM-Dyn 1×7 (dynamic path) C++ cross-reference @48000 Live",
                3.67e-15,
                Some(3.02e-15),
                Some(144.4),
                1.46e-06,
            ),
            fid(
                "linear-fft-rf2048@48000",
                "Linear FFT RF=2048 (C++ golden) @48000",
                1.70e-14,
                None,
                Some(137.7),
                2.17e-06,
            ),
            fid(
                "linear-fft-rf4096@48000",
                "Linear FFT RF=4096 (C++ golden) @48000",
                1.62e-14,
                None,
                Some(137.9),
                4.09e-06,
            ),
            fid(
                "linear-fft-rf8192@48000",
                "Linear FFT RF=8192 (C++ golden) @48000",
                1.69e-14,
                None,
                Some(137.7),
                5.20e-06,
            ),
            fid(
                "slim-a2-example@48000:live",
                "SlimmableContainer A2 Example (CH=3→6) C++ cross-reference @48000 Live",
                7.28e-14,
                Some(1.82e-14),
                Some(131.4),
                1.73e-05,
            ),
            fid(
                "wavenet-a2-dyn-blended@48000:live",
                "WaveNet A2 Dynamic Blended (CH=3, blended layers 2/23) C++ cross-reference @48000 Live",
                5.35e-14,
                Some(2.65e-14),
                Some(132.7),
                9.97e-06,
            ),
            fid(
                "wavenet-a2-dyn-gated@48000:live",
                "WaveNet A2 Dynamic Gated (CH=8, gated layers 3/23) C++ cross-reference @48000 Live",
                5.03e-11,
                Some(1.00e-10),
                Some(103.0),
                6.64e-05,
            ),
            fid(
                "wavenet-a2-film-chaos@48000:live",
                "WaveNet A2-FiLM Chaos Stress (CH=3, FiLM active) C++ cross-reference @48000 Live",
                1.26e-14,
                Some(1.03e-14),
                Some(139.0),
                7.00e-06,
            ),
            fid(
                "wavenet-a2-film-full@48000:live",
                "WaveNet A2-FiLM-Full (CH=8, FiLM active) C++ cross-reference @48000 Live",
                1.16e-14,
                Some(6.42e-15),
                Some(139.4),
                7.85e-06,
            ),
            fid(
                "wavenet-a2-film-input-mixin-pre@48000:live",
                "WaveNet A2-FiLM-InputMixinPre (CH=3, input_mixin_pre_film) C++ cross-reference @48000 Live",
                3.44e-14,
                Some(2.21e-14),
                Some(134.6),
                6.92e-06,
            ),
            fid(
                "wavenet-a2-film-lite@48000:live",
                "WaveNet A2-FiLM-Lite (CH=3, FiLM active) C++ cross-reference @48000 Live",
                3.82e-13,
                Some(1.61e-13),
                Some(124.2),
                1.69e-05,
            ),
            fid(
                "wavenet-a2-full@48000:live",
                "WaveNet A2-Full (CH=8) C++ cross-reference @48000 Live",
                1.46e-13,
                Some(7.83e-14),
                Some(128.3),
                1.68e-05,
            ),
            fid(
                "wavenet-a2-full-poly-simd@48000:live",
                "WaveNet A2-Full polynomial SIMD (regression gate) @48000 Live",
                1.46e-13,
                None,
                Some(128.3),
                1.68e-05,
            ),
            fid(
                "wavenet-a2-lite@48000:live",
                "WaveNet A2-Lite (CH=3) C++ cross-reference @48000 Live",
                8.36e-14,
                Some(1.82e-14),
                Some(130.8),
                9.54e-06,
            ),
            fid(
                "wavenet-condition-dsp@48000:live",
                "WaveNet Condition DSP (CH=3, cond=3, dynamic path) C++ cross-reference @48000 Live",
                1.11e-14,
                Some(6.33e-15),
                Some(139.6),
                3.59e-06,
            ),
            fid(
                "wavenet-official@48000:live",
                "WaveNet Official (CH=3, dynamic path) C++ cross-reference @48000 Live",
                9.03e-14,
                Some(6.13e-14),
                Some(130.4),
                1.66e-05,
            ),
            fid(
                "wavenet-std-poly-simd@48000:live",
                "WaveNet Standard polynomial SIMD (regression gate) @48000 Live",
                2.31e-14,
                None,
                Some(136.4),
                6.46e-06,
            ),
            fid(
                "wavenetdyn-free-shape@48000:live",
                "WaveNetDyn Free-Shape (CH=7→4, dynamic path) C++ cross-reference @48000 Live",
                4.10e-13,
                Some(1.06e-12),
                Some(123.9),
                2.58e-05,
            ),
            fid(
                "convnet-nobn@48000:live",
                "convnet_nobn @48000 Live",
                3.17e-14,
                None,
                Some(135.0),
                5.74e-06,
            ),
            fid(
                "convnet-relu@48000:live",
                "convnet_relu @48000 Live",
                9.33e-16,
                None,
                Some(150.3),
                8.14e-07,
            ),
            fid(
                "convnet-silu@48000:live",
                "convnet_silu @48000 Live",
                3.24e-13,
                None,
                None,
                1.10e-05,
            ),
            fid(
                "linear-nobias@48000:live",
                "linear_nobias @48000 Live",
                3.89e-15,
                None,
                Some(144.1),
                1.64e-06,
            ),
            fid(
                "lstm-official@48000:live",
                "lstm (Official) @48000 Live",
                7.86e-13,
                Some(2.71e-12),
                Some(121.0),
                3.08e-05,
            ),
            fid(
                "lstm-1x10@48000:live",
                "lstm_1x10 @48000 Live",
                4.08e-15,
                None,
                Some(143.9),
                1.19e-06,
            ),
            fid(
                "lstm-2x24@48000:live",
                "lstm_2x24 @48000 Live",
                2.83e-14,
                None,
                Some(135.5),
                3.70e-06,
            ),
            fid(
                "lstm-3x8@48000:live",
                "lstm_3x8 @48000 Live",
                3.70e-15,
                None,
                Some(144.3),
                5.79e-07,
            ),
            fid(
                "wavenet-a1-standard@48000:live",
                "wavenet_a1_standard (Official) @48000 Live",
                1.20e-13,
                Some(1.05e-13),
                Some(129.2),
                2.26e-06,
            ),
            fid(
                "container-a2-full@48000:live",
                "Container A2-Full (CH=8) C++ cross-reference @48000 Live",
                1.46e-13,
                Some(7.83e-14),
                Some(128.3),
                1.68e-05,
            ),
            fid(
                "container-a2-lite@48000:live",
                "Container A2-Lite (CH=3) C++ cross-reference @48000 Live",
                8.36e-14,
                Some(1.82e-14),
                Some(130.8),
                9.54e-06,
            ),
            fid(
                "container-file-a2-full@48000:live",
                "Container File A2-Full (CH=8) C++ cross-reference @48000 Live",
                1.46e-13,
                Some(7.83e-14),
                Some(128.3),
                1.68e-05,
            ),
            fid(
                "container-file-a2-lite@48000:live",
                "Container File A2-Lite (CH=3) C++ cross-reference @48000 Live",
                8.36e-14,
                Some(1.82e-14),
                Some(130.8),
                9.54e-06,
            ),
            fid(
                "quick-a2-full@48000:live",
                "Quick A2-Full @48000 Live",
                1.46e-13,
                Some(7.83e-14),
                Some(128.3),
                1.68e-05,
            ),
            fid(
                "quick-a2-full-v2@48000:live",
                "Quick A2-Full v2 @48000 Live",
                1.57e-13,
                None,
                Some(128.0),
                2.76e-05,
            ),
            fid(
                "quick-convnet-nobn@48000:live",
                "Quick ConvNet No BatchNorm @48000 Live",
                3.17e-14,
                None,
                Some(135.0),
                5.74e-06,
            ),
            fid(
                "quick-convnet-relu@48000:live",
                "Quick ConvNet ReLU @48000 Live",
                9.33e-16,
                None,
                Some(150.3),
                8.14e-07,
            ),
            fid(
                "quick-convnet-silu@48000:live",
                "Quick ConvNet SiLU @48000 Live",
                3.24e-13,
                None,
                None,
                1.10e-05,
            ),
            fid(
                "quick-lstm-1x10@48000:live",
                "Quick LSTM 1×10 @48000 Live",
                4.08e-15,
                None,
                Some(143.9),
                1.19e-06,
            ),
            fid(
                "quick-lstm-1x16@48000:live",
                "Quick LSTM 1×16 @48000 Live",
                8.19e-12,
                Some(9.08e-13),
                Some(110.9),
                3.01e-05,
            ),
            fid(
                "quick-lstm-2x24@48000:live",
                "Quick LSTM 2×24 @48000 Live",
                2.83e-14,
                None,
                Some(135.5),
                3.70e-06,
            ),
            fid(
                "quick-lstm-3x8@48000:live",
                "Quick LSTM 3×8 @48000 Live",
                3.69e-15,
                None,
                Some(144.3),
                5.79e-07,
            ),
            fid(
                "quick-linear-nobias@48000:live",
                "Quick Linear No Bias @48000 Live",
                3.89e-15,
                None,
                Some(144.1),
                1.64e-06,
            ),
            fid(
                "quick-slim-a2-v2@48000:live",
                "Quick SlimmableContainer A2 Example v2 @48000 Live",
                8.26e-14,
                None,
                Some(130.8),
                1.58e-04,
            ),
            fid(
                "quick-wavenet-ch16@48000:live",
                "Quick WaveNet CH16 @48000 Live",
                2.31e-14,
                Some(9.05e-15),
                Some(136.4),
                6.46e-06,
            ),
            fid(
                "quick-wavenet-std-v2@48000:live",
                "Quick WaveNet Standard v2 @48000 Live",
                9.92e-14,
                None,
                Some(130.0),
                4.31e-05,
            ),
        ],
        monitoring_alerts: Some(MonitoringAlerts {
            // JSON cannot carry comments; the caveat lives here so every
            // regeneration preserves the "trend aid, never a CI gate"
            // semantics. Values track the most recent long-suite run
            // (worst ESR parity and worst P99.9 under contention), refreshed
            // alongside the contract snapshot.
            notes: "Trend-monitoring only — never a blocking gate. \
                    Alert when the next long-suite run exceeds alert_above."
                .into(),
            worst_esr_parity: MonitoringAlertF64 {
                measured: 5.03e-11,
                threshold: 1.0e-10,
                alert_above: 8.0e-11,
            },
            rt_p99_9_under_contention_us: MonitoringAlertUs {
                measured_us: 131,
                threshold_us: 1_330,
                alert_above_us: 200,
            },
        }),
        performance: vec![
            // ── Model Inference Core — 14 entries ──────────────────────────
            perf("RT_WaveNet_Std_CH16", "WaveNet Standard CH16", 44.89),
            perf("RT_WaveNet_Feather_CH8", "WaveNet Feather CH8", 19.96),
            perf("RT_WaveNet_Lite_CH12", "WaveNet Lite CH12", 58.76),
            perf("RT_WaveNet_Nano_CH4", "WaveNet Nano CH4", 18.0),
            perf("RT_A2_Full_CH8", "A2 Full CH8", 25.74),
            perf("RT_A2_Lite_CH3", "A2 Lite CH3", 20.96),
            perf("RT_LSTM_1x16", "LSTM 1x16", 6.69),
            perf("RT_LSTM_2x8", "LSTM 2x8", 7.26),
            perf("RT_Linear", "Linear RF=2048", 0.26),
            perf("RT_ConvNet", "ConvNet", 8.69),
            perf("RT_WaveNet_Dyn_Free", "WaveNet Dyn Free", 21.56),
            perf("RT_LSTM_Dyn_1x7", "LSTM Dyn 1x7", 8.06),
            perf("RT_A2_Dyn_Gated_CH8", "A2 Dyn Gated CH8", 176.96),
            perf("RT_A2_Dyn_Blended_CH3", "A2 Dyn Blended CH3", 129.61),
            // ── DSP Infrastructure — 5 entries ─────────────────────────────
            perf_micro_batch(
                "RT_DSP_Resampler_44k1_to_48k",
                "DSP Resampler 44.1k->48k",
                1.24,
                64,
            ),
            perf_micro_batch(
                "RT_DSP_Resampler_96k_to_48k",
                "DSP Resampler 96k->48k",
                0.62,
                64,
            ),
            perf_micro_batch("RT_DSP_CabSim_IR_Medium", "DSP CabSim IR Medium", 1.23, 64),
            perf(
                "RT_DSP_Pipeline_Base_NoOS",
                "DSP Pipeline Base (No OS)",
                45.62,
            ),
            perf("RT_DSP_Pipeline_HQ_4xOS", "DSP Pipeline HQ (4x OS)", 185.59),
        ],
    };

    let mut ids = std::collections::HashSet::new();
    for id in contract
        .fidelity
        .iter()
        .map(|f| &f.id)
        .chain(contract.performance.iter().map(|p| &p.id))
    {
        assert!(ids.insert(id), "duplicate id in transcription: {id}");
    }
    assert_eq!(
        contract.fidelity.len(),
        51,
        "fidelity count must match snapshot"
    );
    assert_eq!(
        contract.performance.len(),
        19,
        "performance count must match snapshot"
    );
    assert_eq!(
        contract.fidelity.iter().filter(|f| f.optional).count(),
        1,
        "optional:true only on EVH-5150-Lite"
    );

    println!("{}", contract.to_json_pretty().unwrap());
}
