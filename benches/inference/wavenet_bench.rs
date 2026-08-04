// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet inference benchmarks: Standard model, buffer-size scaling, dynamic
//! free-geometry fallback, and cataloged-vs-dynamic comparison.

use criterion::Criterion;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::models::NamModel;

use super::common::{
    generate_sine_440hz, load_and_prewarm, load_model_data, make_wavenet_dyn_data,
};

/// Measures the processing time of a real WaveNet model ("Standard").
/// This benchmark is the most representative for common guitar use,
/// testing the effectiveness of dilated convolutions and optimized SIMD kernels.
pub fn bench_wavenet_standard_process(c: &mut Criterion) {
    let mut model = match load_and_prewarm("BossWN-standard.nam") {
        Some(m) => m,
        None => return,
    };
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("WaveNet_Standard_CH16_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Benchmarks WaveNet Standard inference at small RT buffer sizes (1, 16, 64
/// samples) covering the full RT range: 1 (per-sample minimum), 16 (small plugin
/// buffer), 64 (common CLAP/JACK buffer).
///
/// Records throughput (elem/s) and latency (ns) for each size.
pub fn bench_wavenet_p10_small_block_sizes(c: &mut Criterion) {
    let mut model = match load_and_prewarm("BossWN-standard.nam") {
        Some(m) => m,
        None => return,
    };

    for &size in &[1usize, 16, 64] {
        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];
        c.bench_function(&format!("WaveNet_P10_{}", size), |b| {
            b.iter(|| model.process(&input, &mut output));
        });
    }
}

/// Evaluates how WaveNet scales with different DSP buffer sizes.
/// Larger buffers allow better cache utilization and prefetching,
/// but increase the total latency perceived by the musician.
pub fn bench_wavenet_standard_block_sizes(c: &mut Criterion) {
    let mut model = match load_and_prewarm("BossWN-standard.nam") {
        Some(m) => m,
        None => return,
    };

    for &size in &[32, 128, 256, 512] {
        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];
        c.bench_function(&format!("WaveNet_Standard_CH16_{}samp_48kHz", size), |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

/// Measures the time spent in the `prewarm` function.
/// Although prewarm runs outside the audio thread, it must be fast enough
/// so that model switching during a live performance is imperceptible.
pub fn bench_prewarm_wavenet_standard(c: &mut Criterion) {
    let model_data = match load_model_data("BossWN-standard.nam") {
        Some(d) => d,
        None => return,
    };
    c.bench_function("Prewarm_WaveNet_Standard_2048samp", |b| {
        b.iter_with_setup(
            || build_model(&model_data).expect("Dispatcher failed"),
            |mut model| {
                model.prewarm(std::hint::black_box(2048));
            },
        );
    });
}

/// Measures the processing time of a free-geometry WaveNet Dynamic model (CH=5, K=3, COND=3).
///
/// CH=5 and condition_size=3 force the dispatcher to route to `WaveNetModelDyn`
/// instead of a const-generic SKU. This benchmark covers the dynamic hot-path
/// that is exercised when a user loads a model outside the catalog.
pub fn bench_wavenet_dynamic_process(c: &mut Criterion) {
    let data = make_wavenet_dyn_data();
    let mut model = build_model(&data).expect("Dispatcher failed for WaveNet Dynamic benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("WaveNet_Dynamic_CH5_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Comparative benchmark: WaveNet Standard (CH=16, cataloged) vs WaveNet Dynamic
/// (CH=5, fallback) side by side. Measures the dispatch overhead and performance
/// delta between the const-generic fast path and the dynamic free-geometry fallback.
pub fn bench_wavenet_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("WaveNet_Comparison");
    group.sample_size(50);

    if let Some(mut model) = load_and_prewarm("BossWN-standard.nam") {
        let input = generate_sine_440hz(64);
        let mut output = vec![0.0f32; 64];

        group.bench_function("Standard_CH16_cataloged", |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }

    let dyn_data = make_wavenet_dyn_data();
    let mut dyn_model =
        build_model(&dyn_data).expect("Dispatcher failed for WaveNet Dynamic benchmark");
    dyn_model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    group.bench_function("Dynamic_CH5_fallback", |b| {
        b.iter(|| {
            dyn_model.process(&input, &mut output);
        });
    });

    group.finish();
}
