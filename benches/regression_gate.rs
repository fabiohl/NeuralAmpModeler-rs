// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Performance regression gate — Criterion benches with adequate statistical power
//! for CI regression detection against persisted baselines.
//!
//! ## Running
//!
//! ```sh
//! # Save a baseline (first run, or after verified optimization):
//! taskset -c 0 cargo bench --bench regression_gate -- --save-baseline ci-baseline
//!
//! # CI run: compare against saved baseline, fail on regression:
//! taskset -c 0 cargo bench --bench regression_gate -- --baseline ci-baseline
//! ```
//!
//! ## Bench layout
//!
//! Each bench function loads the real model file, warms up, and measures
//! `process()` for a single 64-sample block. The Criterion framework
//! handles statistical comparison against the saved baseline.
//!
//! ## Model coverage
//!
//! WaveNet Std/Feather/Nano/Lite, A2-Full/Lite, LSTM 1x16/2x8, Linear, ConvNet.

#[path = "common.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::models::NamModel;

macro_rules! regression_bench {
    ($c:expr, $label:expr, $file:expr) => {
        if let Some(mut model) = common::load_and_prewarm($file) {
            let input = common::generate_sine_440hz(64);
            let mut output = vec![0.0f32; 64];
            $c.bench_function($label, |b| {
                b.iter(|| model.process(&input, &mut output));
            });
        }
    };
}

fn bench_wavenet_standard(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Std_CH16", "BossWN-standard.nam");
}

fn bench_wavenet_feather(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Feather_CH8", "BossWN-feather.nam");
}

fn bench_wavenet_lite(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Lite_CH12", "BossWN-lite.nam");
}

fn bench_wavenet_nano(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Nano_CH4", "BossWN-nano.nam");
}

fn bench_a2_full(c: &mut Criterion) {
    regression_bench!(c, "RT_A2_Full_CH8", "wavenet_a2_full.nam");
}

fn bench_a2_lite(c: &mut Criterion) {
    regression_bench!(c, "RT_A2_Lite_CH3", "wavenet_a2_lite.nam");
}

fn bench_lstm_1x16(c: &mut Criterion) {
    regression_bench!(c, "RT_LSTM_1x16", "BossLSTM-1x16.nam");
}

fn bench_lstm_2x8(c: &mut Criterion) {
    regression_bench!(c, "RT_LSTM_2x8", "BossLSTM-2x8.nam");
}

fn bench_linear(c: &mut Criterion) {
    regression_bench!(c, "RT_Linear", "linear_test.nam");
}

fn bench_convnet(c: &mut Criterion) {
    regression_bench!(c, "RT_ConvNet", "convnet_test.nam");
}

criterion_group!(
    name = regression_gates;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1))
        .noise_threshold(0.02);
    targets =
        bench_wavenet_standard,
        bench_wavenet_feather,
        bench_wavenet_lite,
        bench_wavenet_nano,
        bench_a2_full,
        bench_a2_lite,
        bench_lstm_1x16,
        bench_lstm_2x8,
        bench_linear,
        bench_convnet,
);

criterion_main!(regression_gates);
