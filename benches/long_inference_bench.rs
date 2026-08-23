// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Long-running (soak) inference benchmarks for the NeuralAmpModeler-rs engine.
//!
//! These benchmarks use extended measurement times (35 s) with large buffer
//! sizes to validate CPU thermal stability and detect performance jitter /
//! throttling over time.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --features standalone,long_bench --bench long_inference_bench
//! ```
//!
//! Without the `long_bench` feature this binary compiles to a no-op so that
//! `cargo bench` (default pass) does not re-run the long soak benchmarks.

#[cfg(feature = "long_bench")]
mod common;

#[cfg(feature = "long_bench")]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(feature = "long_bench")]
use neural_amp_modeler_rs::loader::dispatcher::build_model;
#[cfg(feature = "long_bench")]
use neural_amp_modeler_rs::models::NamModel;

#[cfg(feature = "long_bench")]
fn bench_wavenet_long_run(c: &mut Criterion) {
    let mut model = match common::load_and_prewarm("BossWN-standard.nam") {
        Some(m) => m,
        None => return,
    };
    let size = 4096;
    let input = common::generate_sine_440hz(size);
    let mut output = vec![0.0f32; size];
    let mut group = c.benchmark_group("Long_Run_WaveNet");
    group.measurement_time(std::time::Duration::from_secs(35));
    group.sample_size(100);
    group.bench_function("Long_WaveNet_Standard_CH16_4096samp", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
    group.finish();
}

#[cfg(feature = "long_bench")]
fn bench_lstm_long_run(c: &mut Criterion) {
    let data = common::make_lstm_data(2, 16);
    let mut model = build_model(&data).expect("Dispatcher failed");
    model.prewarm(4096);
    let size = 4096;
    let input = common::generate_sine_440hz(size);
    let mut output = vec![0.0f32; size];
    let mut group = c.benchmark_group("Long_Run_LSTM");
    group.measurement_time(std::time::Duration::from_secs(35));
    group.sample_size(100);
    group.bench_function("Long_LSTM_2x16_4096samp", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
    group.finish();
}

#[cfg(feature = "long_bench")]
fn bench_resampler_long_run(c: &mut Criterion) {
    use neural_amp_modeler_rs::dsp::resampler::NamResampler;
    let size = 4096;
    let mut rs = NamResampler::new(44_100, 48_000, size).unwrap();
    let in_l = vec![0.0f32; size];
    let in_r = vec![0.0f32; size];
    let mut out_l = vec![0.0f32; size * 2];
    let mut out_r = vec![0.0f32; size * 2];
    let mut group = c.benchmark_group("Long_Run_Resampler");
    group.measurement_time(std::time::Duration::from_secs(35));
    group.sample_size(100);
    group.bench_function("Long_Resampler_44100_to_48000_4096samp", |b| {
        b.iter(|| {
            rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.finish();
}

#[cfg(feature = "long_bench")]
fn bench_a2_full_long_run(c: &mut Criterion) {
    let mut model = match common::load_and_prewarm("wavenet_a2_full.nam") {
        Some(m) => m,
        None => return,
    };
    let size = 4096;
    let input = common::generate_sine_440hz(size);
    let mut output = vec![0.0f32; size];
    let mut group = c.benchmark_group("Long_Run_A2Full");
    group.measurement_time(std::time::Duration::from_secs(35));
    group.sample_size(100);
    group.bench_function("Long_A2Full_CH8_4096samp", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
    group.finish();
}

#[cfg(feature = "long_bench")]
fn bench_a2_lite_long_run(c: &mut Criterion) {
    let mut model = match common::load_and_prewarm("wavenet_a2_lite.nam") {
        Some(m) => m,
        None => return,
    };
    let size = 4096;
    let input = common::generate_sine_440hz(size);
    let mut output = vec![0.0f32; size];
    let mut group = c.benchmark_group("Long_Run_A2Lite");
    group.measurement_time(std::time::Duration::from_secs(35));
    group.sample_size(100);
    group.bench_function("Long_A2Lite_CH3_4096samp", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
    group.finish();
}

/// Measures CabSim 16384-tap convolution throughput under thermal soak
/// (35 s measurement). Uses the same synthetic impulse response as the
/// `cabsim_bench` suite for consistency.
#[cfg(feature = "long_bench")]
fn bench_cabsim_long_run(c: &mut Criterion) {
    use neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine;
    let ir = common::synth_ir(16384, 440.0, 10.0);
    let mut engine = ConvEngine::new(&ir, 64).expect("bench ConvEngine allocation failed");

    for _ in 0..engine.num_partitions().max(1) {
        let buf_in = vec![0.0f32; 64];
        let mut buf_out = vec![0.0f32; 64];
        engine.process(&buf_in, &mut buf_out, None);
    }

    let mut input = vec![0.0f32; 4096];
    let mut output = vec![0.0f32; 4096];

    let mut group = c.benchmark_group("Cabsim_LongRun");
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(35));
    group.bench_function("4096samp_block", |b| {
        b.iter(|| {
            for (j, v) in input.iter_mut().enumerate() {
                *v = (j as f32 * 0.01).sin();
            }
            for chunk in 0..(4096 / 64) {
                let start = chunk * 64;
                engine.process(
                    std::hint::black_box(&input[start..start + 64]),
                    std::hint::black_box(&mut output[start..start + 64]),
                    None,
                );
            }
            std::hint::black_box(&output);
        });
    });
    group.finish();
}

#[cfg(feature = "long_bench")]
criterion_group!(
    name = long_benches;
    config = Criterion::default();
    targets = bench_wavenet_long_run, bench_lstm_long_run, bench_resampler_long_run, bench_a2_full_long_run, bench_a2_lite_long_run, bench_cabsim_long_run
);

#[cfg(feature = "long_bench")]
criterion_main!(long_benches);

#[cfg(not(feature = "long_bench"))]
fn main() {}
