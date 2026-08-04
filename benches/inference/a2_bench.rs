// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 (Channel-Aligned WaveNet) inference benchmarks: Full (CH=8) and Lite
//! (CH=3) variants, prewarm cost, and cross-variant comparison.

use criterion::Criterion;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::models::NamModel;

use super::common::{
    generate_sine_440hz, load_and_prewarm, load_model_data, make_wavenet_a2_dyn_data,
};

// ── A2-Full (CH=8) inference benchmarks ──

/// Measures the processing time of an A2-Full (CH=8) WaveNet model.
/// A2-Full is the Criterion variant with 8 channels,
/// using AVX2 T=4 broadcast-FMA convolution for maximum throughput.
pub fn bench_a2_full_process(c: &mut Criterion) {
    let mut model = match load_and_prewarm("wavenet_a2_full.nam") {
        Some(m) => m,
        None => return,
    };
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("A2Full_CH8_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures A2-Full (CH=8) scaling across different DSP buffer sizes.
/// A2 benefit from SIMD col-major layout which keeps 8 weights
/// contiguous per (tap, input) pair, maximizing L1 cache locality.
pub fn bench_a2_full_block_sizes(c: &mut Criterion) {
    let mut model = match load_and_prewarm("wavenet_a2_full.nam") {
        Some(m) => m,
        None => return,
    };

    for &size in &[32, 128, 256, 512] {
        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];
        c.bench_function(&format!("A2Full_CH8_{}samp_48kHz", size), |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

/// Measures A2-Full (CH=8) prewarm cost.
/// Prewarm fills the causal history buffers — essential to measure
/// so model switching latency is known for live performance.
pub fn bench_prewarm_a2_full(c: &mut Criterion) {
    let model_data = match load_model_data("wavenet_a2_full.nam") {
        Some(d) => d,
        None => return,
    };
    c.bench_function("Prewarm_A2Full_CH8_2048samp", |b| {
        b.iter_with_setup(
            || build_model(&model_data).expect("Dispatcher failed"),
            |mut model| {
                model.prewarm(std::hint::black_box(2048));
            },
        );
    });
}

// ── A2-Lite (CH=3) inference benchmarks ──

/// Measures the processing time of an A2-Lite (CH=3) WaveNet model.
/// A2-Lite is the CPU-efficient variant with 3 channels, designed
/// for reduced computational load while preserving timbre character.
pub fn bench_a2_lite_process(c: &mut Criterion) {
    let mut model = match load_and_prewarm("wavenet_a2_lite.nam") {
        Some(m) => m,
        None => return,
    };
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("A2Lite_CH3_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures A2-Lite (CH=3) scaling across different DSP buffer sizes.
/// With fewer channels and the u16 interleaved kernel path, A2-Lite
/// targets minimal CPU usage for low-power / high-polyphony scenarios.
pub fn bench_a2_lite_block_sizes(c: &mut Criterion) {
    let mut model = match load_and_prewarm("wavenet_a2_lite.nam") {
        Some(m) => m,
        None => return,
    };

    for &size in &[32, 128, 256, 512] {
        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];
        c.bench_function(&format!("A2Lite_CH3_{}samp_48kHz", size), |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

/// Measures A2-Lite (CH=3) prewarm cost.
pub fn bench_prewarm_a2_lite(c: &mut Criterion) {
    let model_data = match load_model_data("wavenet_a2_lite.nam") {
        Some(d) => d,
        None => return,
    };
    c.bench_function("Prewarm_A2Lite_CH3_2048samp", |b| {
        b.iter_with_setup(
            || build_model(&model_data).expect("Dispatcher failed"),
            |mut model| {
                model.prewarm(std::hint::black_box(2048));
            },
        );
    });
}

/// Comparative benchmark: A2-Full (CH=8), A2-Lite (CH=3), and A2-Dyn (CH=4, gated)
/// side by side. Validates the performance spread across the static const-generic
/// fast paths and the dynamic free-geometry fallback of the A2 architecture.
pub fn bench_a2_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("A2_Comparison");
    group.sample_size(50);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    if let Some(mut model) = load_and_prewarm("wavenet_a2_full.nam") {
        group.bench_function("Full_CH8", |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }

    if let Some(mut model) = load_and_prewarm("wavenet_a2_lite.nam") {
        group.bench_function("Lite_CH3", |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }

    let dyn_data = make_wavenet_a2_dyn_data();
    let mut dyn_model = build_model(&dyn_data).expect("Dispatcher failed for A2 Dynamic benchmark");
    dyn_model.prewarm(2048);

    group.bench_function("Dynamic_CH4_gated", |b| {
        b.iter(|| {
            dyn_model.process(&input, &mut output);
        });
    });

    group.finish();
}
