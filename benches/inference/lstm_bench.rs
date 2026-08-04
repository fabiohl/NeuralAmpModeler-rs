// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! LSTM inference benchmarks: recurrent processing, buffer-size scaling,
//! scalar-vs-SIMD comparison, and dynamic free-geometry fallback.

use criterion::Criterion;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::StaticModel;

use super::common::{generate_sine_440hz, make_lstm_data};

/// Measures the processing time of a 2x16 LSTM recurrent network.
/// LSTMs are known for their high sequential computational load,
/// making them the ideal stress test to verify feedback loop latency.
pub fn bench_lstm_2x16_process(c: &mut Criterion) {
    let data = make_lstm_data(2, 16);
    let mut model = build_model(&data).expect("Dispatcher failed for LSTM benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("LSTM_2x16_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Compares performance between Scalar (baseline) and SIMD implementations for LSTM 1x8.
/// This benchmark validates the performance gain achieved with the Fused Gates kernel,
/// where the 4 LSTM gates are computed simultaneously in AVX2 registers.
pub fn bench_lstm_1x8_comparison(c: &mut Criterion) {
    let data = make_lstm_data(1, 8);
    let mut model_simd = build_model(&data).expect("Dispatcher failed for LSTM 1x8 benchmark");
    let mut model_scalar = build_model(&data).expect("Dispatcher failed for LSTM 1x8 benchmark");
    model_simd.prewarm(1024);
    model_scalar.prewarm(1024);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    let mut group = c.benchmark_group("LSTM_1x8_Comparison");

    group.bench_function("SIMD_Fused_T3", |b| {
        b.iter(|| {
            model_simd.process(&input, &mut output);
        });
    });

    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        StaticModel::Lstm1x8(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm1x8"),
    });
    group.finish();
}

/// Comparative benchmark: LSTM 2x16 Scalar vs SIMD (Fused Gates).
pub fn bench_lstm_2x16_comparison(c: &mut Criterion) {
    let data = make_lstm_data(2, 16);
    let mut model_simd = build_model(&data).expect("Dispatcher failed for LSTM 2x16 benchmark");
    let mut model_scalar = build_model(&data).expect("Dispatcher failed for LSTM 2x16 benchmark");
    model_simd.prewarm(1024);
    model_scalar.prewarm(1024);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    let mut group = c.benchmark_group("LSTM_2x16_Comparison");
    group.bench_function("SIMD_Fused_T3", |b| {
        b.iter(|| {
            model_simd.process(&input, &mut output);
        });
    });

    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        StaticModel::Lstm2x16(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm2x16"),
    });
    group.finish();
}

/// Evaluates LSTM scaling with different buffer sizes.
/// Unlike WaveNet, the LSTM is purely sequential (sample by sample),
/// so per-sample overhead tends to be more constant regardless of block size.
pub fn bench_lstm_2x16_block_sizes(c: &mut Criterion) {
    let data = make_lstm_data(2, 16);
    let mut model = build_model(&data).expect("Dispatcher failed");
    model.prewarm(2048);
    for &size in &[32, 128, 256, 512] {
        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];
        c.bench_function(&format!("LSTM_2x16_{}samp_48kHz", size), |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

/// Measures the LSTM 2x16 prewarm cost (2048 samples).
pub fn bench_prewarm_lstm_2x16(c: &mut Criterion) {
    let data = make_lstm_data(2, 16);
    c.bench_function("Prewarm_LSTM_2x16_2048samp", |b| {
        b.iter_with_setup(
            || build_model(&data).expect("Dispatcher failed"),
            |mut model| {
                model.prewarm(std::hint::black_box(2048));
            },
        );
    });
}

/// Measures the processing time of a 1x40 LSTM recurrent network.
pub fn bench_lstm_1x40_process(c: &mut Criterion) {
    let data = make_lstm_data(1, 40);
    let mut model = build_model(&data).expect("Dispatcher failed for LSTM benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("LSTM_1x40_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures the processing time of a 2x24 LSTM recurrent network.
pub fn bench_lstm_2x24_process(c: &mut Criterion) {
    let data = make_lstm_data(2, 24);
    let mut model = build_model(&data).expect("Dispatcher failed for LSTM benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("LSTM_2x24_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Comparative benchmark: LSTM 1x40 Scalar vs SIMD (Fused Gates).
pub fn bench_lstm_1x40_comparison(c: &mut Criterion) {
    let data = make_lstm_data(1, 40);
    let mut model_simd = build_model(&data).expect("Dispatcher failed for LSTM 1x40 benchmark");
    let mut model_scalar = build_model(&data).expect("Dispatcher failed for LSTM 1x40 benchmark");
    model_simd.prewarm(1024);
    model_scalar.prewarm(1024);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    let mut group = c.benchmark_group("LSTM_1x40_Comparison");
    group.bench_function("SIMD_Fused_T3", |b| {
        b.iter(|| {
            model_simd.process(&input, &mut output);
        });
    });

    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        StaticModel::Lstm1x40(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm1x40"),
    });
    group.finish();
}

/// Comparative benchmark: LSTM 2x24 Scalar vs SIMD (Fused Gates).
pub fn bench_lstm_2x24_comparison(c: &mut Criterion) {
    let data = make_lstm_data(2, 24);
    let mut model_simd = build_model(&data).expect("Dispatcher failed for LSTM 2x24 benchmark");
    let mut model_scalar = build_model(&data).expect("Dispatcher failed for LSTM 2x24 benchmark");
    model_simd.prewarm(1024);
    model_scalar.prewarm(1024);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    let mut group = c.benchmark_group("LSTM_2x24_Comparison");
    group.bench_function("SIMD_Fused_T3", |b| {
        b.iter(|| {
            model_simd.process(&input, &mut output);
        });
    });

    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        StaticModel::Lstm2x24(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm2x24"),
    });
    group.finish();
}
