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
        // A2 `process` clamps the frame count to the negotiated `max_buffer_size`
        // (constructor default = 64). Without this call, `128/256/512` would all
        // measure only the first 64 frames — the flat-scaling artefact of F-01.
        model
            .set_max_buffer_size(size)
            .expect("A2 set_max_buffer_size must accept the bench block size");
        model.prewarm(2048);

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
/// With fewer channels and the f32 col-major/interleaved kernel path, A2-Lite
/// targets minimal CPU usage for low-power / high-polyphony scenarios.
pub fn bench_a2_lite_block_sizes(c: &mut Criterion) {
    let mut model = match load_and_prewarm("wavenet_a2_lite.nam") {
        Some(m) => m,
        None => return,
    };

    for &size in &[32, 128, 256, 512] {
        // A2 `process` clamps the frame count to the negotiated `max_buffer_size`
        // (constructor default = 64). Without this call, `128/256/512` would all
        // measure only the first 64 frames — the flat-scaling artefact of F-01.
        model
            .set_max_buffer_size(size)
            .expect("A2 set_max_buffer_size must accept the bench block size");
        model.prewarm(2048);

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

/// Direct grouped/depthwise Conv1D kernels, isolated from the rest of the A2
/// model. Covers the two shapes the S05 tap-major repack targets: a general
/// grouped layer (`groups=3`, 3 input channels → 12 output channels) and a
/// depthwise layer (`groups == in_ch == out_ch = 8`).
pub fn bench_a2_grouped_conv(c: &mut Criterion) {
    use neural_amp_modeler_rs::models::a2::A2GroupedConv1d;

    fn raw_weights(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25
            })
            .collect()
    }

    let mut group = c.benchmark_group("A2_GroupedConv");
    group.sample_size(50);

    // General grouped: in=3, out=12, groups=3, k=2, d=1.
    let in_ch = 3;
    let out_ch = 12;
    let kernel = 2;
    let groups = 3;
    let w = raw_weights((out_ch / groups) * in_ch * kernel, 0xA2_05);
    let bias = vec![0.0f32; out_ch];
    let conv = A2GroupedConv1d::new(&w, &bias, true, 1, in_ch, out_ch, kernel, groups)
        .expect("general grouped bench conv must construct");
    let frame_idx = 64;
    let layer_buffer = vec![0.125f32; (frame_idx + kernel + 1) * in_ch];
    let mut out = vec![0.0f32; out_ch];
    group.bench_function("general_groups3_3x12_k2", |b| {
        b.iter(|| {
            // SAFETY: `layer_buffer` covers `frame_idx`'s full lookback and
            // `out` has `out_ch` elements.
            unsafe {
                conv.process_single_frame(&layer_buffer, &mut out, frame_idx, None);
            }
            std::hint::black_box(&out);
        });
    });

    // Depthwise: in=out=8, groups=8, k=2, d=1.
    let channels = 8;
    let w = raw_weights(channels * kernel, 0xD3);
    let bias = vec![0.0f32; channels];
    let conv = A2GroupedConv1d::new(&w, &bias, true, 1, channels, channels, kernel, channels)
        .expect("depthwise grouped bench conv must construct");
    let layer_buffer = vec![0.125f32; (frame_idx + kernel + 1) * channels];
    let mut out = vec![0.0f32; channels];
    group.bench_function("depthwise_8ch_k2", |b| {
        b.iter(|| {
            // SAFETY: `layer_buffer` covers `frame_idx`'s full lookback and
            // `out` has `out_ch` elements.
            unsafe {
                conv.process_single_frame(&layer_buffer, &mut out, frame_idx, None);
            }
            std::hint::black_box(&out);
        });
    });

    group.finish();
}
