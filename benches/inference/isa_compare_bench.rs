// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Cross-ISA comparison benchmarks (`isa_compare`): measures the latency delta
//! between AVX2 baseline and AVX-512 upward dispatch on the same hardware.
//!
//! # Purpose & Gate Criteria (Epic C / Task C2.T3)
//!
//! To justify adding or retaining dedicated AVX-512 / VL256 kernel implementations,
//! the full end-to-end `NamModel::process()` inference must demonstrate a statistically
//! significant speedup of at least **12%** (`p < 0.05`) over the AVX2 baseline on the
//! same machine at `N=64 @ 48 kHz`.
//!
//! Smoke RT latency checks at `N=1` and `N=8` verify that low-buffer per-sample
//! processing is not slower on AVX-512 than on AVX2.
//!
//! # Target SKUs in Canonical Matrix
//!
//! | SKU / Kernel           | Benchmark ID                           | Gate Criterion                  |
//! | ---------------------- | -------------------------------------- | ------------------------------- |
//! | LSTM 4-gate            | `LSTM_2x16_64samp_48kHz`               | $\ge 12\%$ vs AVX2 (same host)  |
//! | LSTM 1x16              | `LSTM_1x16_64samp_48kHz`               | $\ge 12\%$ vs AVX2 (same host)  |
//! | A2-Full (CH=8)         | `A2Full_CH8_64samp_48kHz`              | $\ge 12\%$ vs AVX2 (same host)  |
//! | A2-Lite (CH=3)         | `A2Lite_CH3_64samp_48kHz`              | $\ge 12\%$ vs AVX2 (same host)  |
//! | WaveNet Standard CH16  | `WaveNet_Standard_CH16_64samp_48kHz`   | $\ge 12\%$ vs AVX2 (same host)  |
//!
//! The Criterion group names are namespaced `ISA_Compare_<canonical>` because
//! the canonical IDs themselves are already flat `bench_function` names in the
//! same `inference_bench` binary (`lstm_bench.rs`, `a2_bench.rs`,
//! `wavenet_bench.rs`) — sharing a Criterion report root would merge the two
//! benches. The mapping to the ROI matrix of `docs/benchmarks.md` is:
//!
//! | Criterion group                     | Canonical ROI ID                    |
//! | ----------------------------------- | ----------------------------------- |
//! | `ISA_Compare_LSTM_2x16_64samp_48kHz` | `LSTM_2x16_64samp_48kHz`            |
//! | `ISA_Compare_LSTM_1x16_64samp_48kHz` | `LSTM_1x16_64samp_48kHz`            |
//! | `ISA_Compare_A2Full_CH8_64samp_48kHz` | `A2Full_CH8_64samp_48kHz`          |
//! | `ISA_Compare_A2Lite_CH3_64samp_48kHz` | `A2Lite_CH3_64samp_48kHz`          |
//! | `ISA_Compare_WaveNet_Std_CH16_64samp_48kHz` | `WaveNet_Standard_CH16_64samp_48kHz` |
//!
//! The N=1/N=8 smoke sizes follow the same scheme (`…_1samp_48kHz`,
//! `…_8samp_48kHz`), each group containing the `AVX2` and `AVX512` members.

use criterion::Criterion;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::isa_guard::ForceAvx2Guard;
#[cfg(feature = "avx512")]
use neural_amp_modeler_rs::testing::isa_guard::ForceAvx512Guard;

use super::common::{generate_sine_440hz, load_and_prewarm, make_lstm_data};

/// Compares AVX2 vs AVX-512 inference for LSTM 2x16 across buffer sizes (64, 1, 8).
pub fn bench_isa_compare_lstm_2x16(c: &mut Criterion) {
    let data = make_lstm_data(2, 16);
    let mut model = build_model(&data).expect("Dispatcher failed for LSTM 2x16 benchmark");
    model.prewarm(2048);

    #[cfg(feature = "avx512")]
    let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");

    for &size in &[64usize, 1, 8] {
        let group_name = format!("ISA_Compare_LSTM_2x16_{size}samp_48kHz");
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(50);

        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];

        group.bench_function("AVX2", |b| {
            let _guard = ForceAvx2Guard::new();
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });

        #[cfg(feature = "avx512")]
        if has_avx512 {
            group.bench_function("AVX512", |b| {
                let _guard = ForceAvx512Guard::new();
                b.iter(|| {
                    model.process(&input, &mut output);
                });
            });
        }

        group.finish();
    }
}

/// Compares AVX2 vs AVX-512 inference for LSTM 1x16 across buffer sizes (64, 1, 8).
pub fn bench_isa_compare_lstm_1x16(c: &mut Criterion) {
    let data = make_lstm_data(1, 16);
    let mut model = build_model(&data).expect("Dispatcher failed for LSTM 1x16 benchmark");
    model.prewarm(2048);

    #[cfg(feature = "avx512")]
    let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");

    for &size in &[64usize, 1, 8] {
        let group_name = format!("ISA_Compare_LSTM_1x16_{size}samp_48kHz");
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(50);

        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];

        group.bench_function("AVX2", |b| {
            let _guard = ForceAvx2Guard::new();
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });

        #[cfg(feature = "avx512")]
        if has_avx512 {
            group.bench_function("AVX512", |b| {
                let _guard = ForceAvx512Guard::new();
                b.iter(|| {
                    model.process(&input, &mut output);
                });
            });
        }

        group.finish();
    }
}

/// Compares AVX2 vs AVX-512 inference for A2-Full (CH=8) across buffer sizes (64, 1, 8).
pub fn bench_isa_compare_a2_full(c: &mut Criterion) {
    let mut model = match load_and_prewarm("wavenet_a2_full.nam") {
        Some(m) => m,
        None => return,
    };

    #[cfg(feature = "avx512")]
    let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");

    for &size in &[64usize, 1, 8] {
        let group_name = format!("ISA_Compare_A2Full_CH8_{size}samp_48kHz");
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(50);

        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];

        group.bench_function("AVX2", |b| {
            let _guard = ForceAvx2Guard::new();
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });

        #[cfg(feature = "avx512")]
        if has_avx512 {
            group.bench_function("AVX512", |b| {
                let _guard = ForceAvx512Guard::new();
                b.iter(|| {
                    model.process(&input, &mut output);
                });
            });
        }

        group.finish();
    }
}

/// Compares AVX2 vs AVX-512 inference for A2-Lite (CH=3) across buffer sizes (64, 1, 8).
pub fn bench_isa_compare_a2_lite(c: &mut Criterion) {
    let mut model = match load_and_prewarm("wavenet_a2_lite.nam") {
        Some(m) => m,
        None => return,
    };

    #[cfg(feature = "avx512")]
    let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");

    for &size in &[64usize, 1, 8] {
        let group_name = format!("ISA_Compare_A2Lite_CH3_{size}samp_48kHz");
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(50);

        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];

        group.bench_function("AVX2", |b| {
            let _guard = ForceAvx2Guard::new();
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });

        #[cfg(feature = "avx512")]
        if has_avx512 {
            group.bench_function("AVX512", |b| {
                let _guard = ForceAvx512Guard::new();
                b.iter(|| {
                    model.process(&input, &mut output);
                });
            });
        }

        group.finish();
    }
}

/// Compares AVX2 vs AVX-512 inference for WaveNet Standard (CH=16) across buffer sizes (64, 1, 8).
pub fn bench_isa_compare_wavenet_standard(c: &mut Criterion) {
    let mut model = match load_and_prewarm("BossWN-standard.nam") {
        Some(m) => m,
        None => return,
    };

    #[cfg(feature = "avx512")]
    let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");

    for &size in &[64usize, 1, 8] {
        let group_name = format!("ISA_Compare_WaveNet_Std_CH16_{size}samp_48kHz");
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(50);

        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];

        group.bench_function("AVX2", |b| {
            let _guard = ForceAvx2Guard::new();
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });

        #[cfg(feature = "avx512")]
        if has_avx512 {
            group.bench_function("AVX512", |b| {
                let _guard = ForceAvx512Guard::new();
                b.iter(|| {
                    model.process(&input, &mut output);
                });
            });
        }

        group.finish();
    }
}
