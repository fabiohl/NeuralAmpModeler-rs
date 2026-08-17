// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! One-shot transcription of `docs/quality-contract.txt` → `docs/quality-contract.json`.
//!
//! Dev tool only (S1.T3, finding R-01). The production code never parses the
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
//! Source snapshot: `docs/quality-contract.txt`, measured 2026-08-12 09:20:03 -03,
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
    }
}

#[test]
#[ignore]
fn transcribe_quality_contract_to_json() {
    let contract = QualityContract {
        schema_version: SCHEMA_VERSION,
        generated_at: "2026-08-12T09:20:03-03:00".into(),
        provenance: Provenance {
            git_commit: "0e22ea4ec247".into(),
            git_dirty: true,
            run_id: "1786537203076204151-15755".into(),
            effective_isa: "x86-64-v3 (AVX2/FMA/F16C/BMI)".into(),
            cpu_model: "AMD Ryzen 7 5700U with Radeon Graphics".into(),
            rustc: "rustc 1.97.1 (8bab26f4f 2026-07-14)".into(),
            cargo_profile: "release".into(),
        },
        envelopes: Envelopes::policy_v1(),
        fidelity: vec![
            // ── Canonical Fidelity (golden_vectors) — 34 entries ──────────
            fid(
                "bosslstm-1x16@48000:live",
                "BossLSTM-1x16 @48000 Live",
                8.50e-12,
                Some(8.90e-13),
                Some(110.7),
                2.80e-05,
            ),
            fid(
                "bosslstm-2x8@48000:live",
                "BossLSTM-2x8 @48000 Live",
                1.00e-11,
                Some(5.68e-13),
                Some(110.0),
                1.57e-05,
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
                4.23e-15,
                Some(3.57e-15),
                Some(143.7),
                1.17e-06,
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
                3.70e-15,
                Some(2.86e-15),
                Some(144.3),
                1.45e-06,
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
                3.23e-14,
                None,
                Some(134.9),
                5.62e-06,
            ),
            fid(
                "convnet-relu@48000:live",
                "convnet_relu @48000 Live",
                6.84e-15,
                None,
                Some(141.6),
                2.27e-06,
            ),
            fid(
                "convnet-silu@48000:live",
                "convnet_silu @48000 Live",
                2.58e-13,
                None,
                None,
                8.67e-06,
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
                4.01e-15,
                None,
                Some(144.0),
                1.25e-06,
            ),
            fid(
                "lstm-2x24@48000:live",
                "lstm_2x24 @48000 Live",
                2.71e-14,
                None,
                Some(135.7),
                3.50e-06,
            ),
            fid(
                "lstm-3x8@48000:live",
                "lstm_3x8 @48000 Live",
                3.66e-15,
                None,
                Some(144.4),
                5.55e-07,
            ),
            fid(
                "wavenet-a1-standard@48000:live",
                "wavenet_a1_standard (Official) @48000 Live",
                1.20e-13,
                Some(1.05e-13),
                Some(129.2),
                2.26e-06,
            ),
            // ── Additional Coverage (quick_parity, containers, regression gates) — 17 entries ──
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
                1.12e-13,
                Some(7.83e-14),
                Some(129.5),
                1.49e-05,
            ),
            fid(
                "quick-a2-full-v2@48000:live",
                "Quick A2-Full v2 @48000 Live",
                1.20e-13,
                None,
                Some(129.2),
                2.40e-05,
            ),
            fid(
                "quick-convnet-nobn@48000:live",
                "Quick ConvNet No BatchNorm @48000 Live",
                3.14e-14,
                None,
                Some(135.0),
                5.54e-06,
            ),
            fid(
                "quick-convnet-relu@48000:live",
                "Quick ConvNet ReLU @48000 Live",
                6.76e-15,
                None,
                Some(141.7),
                2.25e-06,
            ),
            fid(
                "quick-convnet-silu@48000:live",
                "Quick ConvNet SiLU @48000 Live",
                5.26e-13,
                None,
                Some(122.8),
                1.42e-05,
            ),
            fid(
                "quick-lstm-1x10@48000:live",
                "Quick LSTM 1×10 @48000 Live",
                4.08e-15,
                None,
                Some(143.9),
                1.35e-06,
            ),
            fid(
                "quick-lstm-1x16@48000:live",
                "Quick LSTM 1×16 @48000 Live",
                1.45e-11,
                Some(8.90e-13),
                Some(108.4),
                3.20e-05,
            ),
            fid(
                "quick-lstm-2x24@48000:live",
                "Quick LSTM 2×24 @48000 Live",
                2.78e-14,
                None,
                Some(135.6),
                3.73e-06,
            ),
            fid(
                "quick-lstm-3x8@48000:live",
                "Quick LSTM 3×8 @48000 Live",
                3.61e-15,
                None,
                Some(144.4),
                6.22e-07,
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
                4.08e-14,
                None,
                Some(133.9),
                9.17e-05,
            ),
            fid(
                "quick-wavenet-ch16@48000:live",
                "Quick WaveNet CH16 @48000 Live",
                2.46e-14,
                Some(9.05e-15),
                Some(136.1),
                6.89e-06,
            ),
            fid(
                "quick-wavenet-std-v2@48000:live",
                "Quick WaveNet Standard v2 @48000 Live",
                9.96e-14,
                None,
                Some(130.0),
                4.29e-05,
            ),
        ],
        performance: vec![
            // ── Model Inference Core — 14 entries ──────────────────────────
            perf("RT_WaveNet_Std_CH16", "WaveNet Standard CH16", 36.9),
            perf("RT_WaveNet_Feather_CH8", "WaveNet Feather CH8", 19.4),
            perf("RT_WaveNet_Lite_CH12", "WaveNet Lite CH12", 52.6),
            perf("RT_WaveNet_Nano_CH4", "WaveNet Nano CH4", 17.4),
            perf("RT_A2_Full_CH8", "A2 Full CH8", 27.6),
            perf("RT_A2_Lite_CH3", "A2 Lite CH3", 18.4),
            perf("RT_LSTM_1x16", "LSTM 1x16", 7.5),
            perf("RT_LSTM_2x8", "LSTM 2x8", 7.6),
            perf("RT_Linear_RF2048", "Linear RF=2048", 0.3),
            perf("RT_ConvNet", "ConvNet", 10.2),
            perf("RT_WaveNet_Dyn_Free", "WaveNet Dyn Free", 22.5),
            perf("RT_LSTM_Dyn_1x7", "LSTM Dyn 1x7", 8.2),
            perf("RT_A2_Dyn_Gated_CH8", "A2 Dyn Gated CH8", 170.8),
            perf("RT_A2_Dyn_Blended_CH3", "A2 Dyn Blended CH3", 136.2),
            // ── DSP Infrastructure — 5 entries ─────────────────────────────
            perf(
                "RT_DSP_Resampler_44k_to_48k",
                "DSP Resampler 44.1k->48k",
                1.3,
            ),
            perf("RT_DSP_Resampler_96k_to_48k", "DSP Resampler 96k->48k", 0.7),
            perf("RT_DSP_CabSim_IR_Medium", "DSP CabSim IR Medium", 1.3),
            perf("RT_DSP_Pipeline_Base", "DSP Pipeline Base (No OS)", 37.2),
            perf("RT_DSP_Pipeline_HQ", "DSP Pipeline HQ (4x OS)", 150.6),
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
