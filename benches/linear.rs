// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Performance benchmarks for the Linear model convolution modes.
//!
//! Compares Direct (time-domain dot product) vs FFT (partitioned overlap-save)
//! across receptive field sizes from 128 to 8192 taps, at the standard 64-sample
//! DSP block size.
//!
//! ## Purpose
//!
//! These benchmarks validate the optimal auto-selection threshold
//! (`FFT_AUTO_THRESHOLD = 256`). The threshold is correct if:
//! - Direct is faster for RF < 256 (FFT overhead dominates)
//! - FFT is faster for RF ≥ 256 (crossing point)
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench linear
//! ```
//!
//! ## Interpreting results
//!
//! | Metric               | Meaning                                          |
//! |----------------------|--------------------------------------------------|
//! | `per block`          | Time to process 64 samples (1 DSP block)         |
//! | `per sample`         | Amortized time per individual sample             |
//! | Real-time deadline   | 1.33 ms at 48 kHz with 64-sample buffer          |

use criterion::{Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::loader::nam_json::LinearImplementation;
use neural_amp_modeler_rs::models::linear::{LinearMode, LinearModel};

mod common;

/// Receptive field sizes to benchmark.
const RF_SIZES: &[usize] = &[128, 256, 512, 1024, 2048, 4096, 8192];

/// DSP block size (64 samples at 48 kHz = 1.33 ms deadline).
const BLOCK_SIZE: usize = 64;

/// Benchmarks Direct vs FFT per-block processing time for each receptive
/// field size. Each RF size gets a comparison group with two entries:
/// `Direct` and `FFT`.
fn bench_direct_vs_fft_per_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_Direct_vs_FFT_per_block");

    for &rf in RF_SIZES {
        let ir = common::synth_ir(rf, 880.0, 8.0);

        let mut model_direct =
            LinearModel::new(ir.clone(), 0.1, LinearImplementation::Direct).unwrap();
        model_direct.prewarm(4096);

        let mut model_fft = LinearModel::new(ir, 0.1, LinearImplementation::Fft).unwrap();
        model_fft.prewarm(4096);

        let input = common::generate_sine_440hz(BLOCK_SIZE);
        let mut output_direct = vec![0.0f32; BLOCK_SIZE];
        let mut output_fft = vec![0.0f32; BLOCK_SIZE];

        group.bench_function(format!("Direct_RF{rf}"), |b| {
            b.iter(|| unsafe {
                model_direct.process(&input, &mut output_direct);
            });
        });

        group.bench_function(format!("FFT_RF{rf}"), |b| {
            b.iter(|| unsafe {
                model_fft.process(&input, &mut output_fft);
            });
        });
    }

    group.finish();
}

/// Benchmarks Direct vs FFT per-sample time by processing 1024 samples
/// in 1-sample blocks. This measures the per-sample dispatch overhead
/// including block boundary detection in the FFT path.
fn bench_direct_vs_fft_per_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_Direct_vs_FFT_per_sample");

    for &rf in RF_SIZES {
        let ir = common::synth_ir(rf, 880.0, 8.0);

        let mut model_direct =
            LinearModel::new(ir.clone(), 0.1, LinearImplementation::Direct).unwrap();
        model_direct.prewarm(4096);

        let mut model_fft = LinearModel::new(ir, 0.1, LinearImplementation::Fft).unwrap();
        model_fft.prewarm(4096);

        let inputs: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
            .collect();

        let mut sample_idx = 0usize;

        group.bench_function(format!("Direct_RF{rf}"), |b| {
            b.iter(|| {
                let x = inputs[sample_idx % inputs.len()];
                sample_idx = sample_idx.wrapping_add(1);
                let mut out = 0.0f32;
                unsafe { model_direct.process(&[x], std::slice::from_mut(&mut out)) };
                out
            });
        });

        group.bench_function(format!("FFT_RF{rf}"), |b| {
            b.iter(|| {
                let x = inputs[sample_idx % inputs.len()];
                sample_idx = sample_idx.wrapping_add(1);
                let mut out = 0.0f32;
                unsafe { model_fft.process(&[x], std::slice::from_mut(&mut out)) };
                out
            });
        });
    }

    group.finish();
}

/// Benchmarks the FFT prewarm cost across RF sizes.
/// Prewarm zeroes history and resets the FFT state (allocation-free,
/// but involves large zero-fill operations).
fn bench_fft_prewarm(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_FFT_Prewarm");

    for &rf in RF_SIZES {
        let ir = common::synth_ir(rf, 880.0, 8.0);

        group.bench_function(format!("Prewarm_RF{rf}"), |b| {
            b.iter_with_setup(
                || LinearModel::new(ir.clone(), 0.1, LinearImplementation::Fft).unwrap(),
                |mut model| {
                    model.prewarm(std::hint::black_box(4096));
                },
            );
        });
    }

    group.finish();
}

/// Measures FFT tail block processing time in isolation.
///
/// This isolates the cost of `process_tail_block` (FFT + SIMD complex MAC +
/// IFFT) from the per-sample head convolution, showing how the FFT cost
/// scales with partition count.
fn bench_fft_tail_block(c: &mut Criterion) {
    let group_name = "Linear_FFT_TailBlock";
    let mut group = c.benchmark_group(group_name);

    for &rf in RF_SIZES {
        let ir = common::synth_ir(rf, 880.0, 8.0);
        let mut model = LinearModel::new(ir, 0.1, LinearImplementation::Fft).unwrap();
        model.prewarm(4096);

        if let LinearMode::Fft(ref mut state) = model.mode {
            let p = state.p;
            let window: Vec<f32> = (0..(2 * p)).map(|i| (i as f32 * 0.1).sin()).collect();

            group.bench_function(format!("TailBlock_RF{rf}_P{p}"), |b| {
                b.iter(|| {
                    state.process_tail_block(&window);
                });
            });
        }
    }

    group.finish();
}

/// Measures the processing cost of a large block across RF sizes.
/// A 4096-sample block exercises block boundary crossings and cache
/// behavior, providing a stress test perspective.
fn bench_large_block_4096(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_LargeBlock_4096samp");

    for &rf in RF_SIZES {
        let ir = common::synth_ir(rf, 880.0, 8.0);

        let mut model_direct =
            LinearModel::new(ir.clone(), 0.1, LinearImplementation::Direct).unwrap();
        model_direct.prewarm(4096);

        let mut model_fft = LinearModel::new(ir, 0.1, LinearImplementation::Fft).unwrap();
        model_fft.prewarm(4096);

        let input = common::generate_sine_440hz(4096);
        let mut output_direct = vec![0.0f32; 4096];
        let mut output_fft = vec![0.0f32; 4096];

        group.bench_function(format!("Direct_RF{rf}"), |b| {
            b.iter(|| unsafe {
                model_direct.process(&input, &mut output_direct);
            });
        });

        group.bench_function(format!("FFT_RF{rf}"), |b| {
            b.iter(|| unsafe {
                model_fft.process(&input, &mut output_fft);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = linear_benches;
    config = Criterion::default();
    targets = bench_direct_vs_fft_per_block, bench_direct_vs_fft_per_sample, bench_fft_prewarm, bench_fft_tail_block, bench_large_block_4096
);

criterion_main!(linear_benches);
