// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Benchmarks for dot_product kernels (4x, 8x, and 16x SIMD variants).
//!
//! Measures throughput of interleaved dot products for state sizes
//! representative of LSTM and WaveNet topologies, now including the
//! wider 8-channel (AVX2/FMA) and 16-channel (AVX-512) f32 kernels.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench dot_4x_bench
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::math::common::half::f32_to_f16_bits;
use neural_amp_modeler_rs::math::common::scalar_ref;
use neural_amp_modeler_rs::math::gemm::dot_4x;
use neural_amp_modeler_rs::math::gemm::dot_product_8x_f32_avx2;
use neural_amp_modeler_rs::math::gemm::dot_product_8x_f32_scalar;
use neural_amp_modeler_rs::math::gemm::dot_product_16x_f32_avx512;
use neural_amp_modeler_rs::math::gemm::dot_product_16x_f32_scalar;

fn generate_test_data(len: usize) -> (Vec<[u16; 4]>, Vec<f32>, Vec<f32>) {
    let weights: Vec<[u16; 4]> = (0..len)
        .map(|i| {
            let v = (i as f32 * 0.1).sin() * 0.5 + 0.5;
            let bits = f32_to_f16_bits(v);
            [bits; 4]
        })
        .collect();
    let state_f0: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).sin()).collect();
    let state_f1: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07 + 1.0).sin()).collect();
    (weights, state_f0, state_f1)
}

fn generate_f32_test_data<const N: usize>(len: usize) -> (Vec<[f32; N]>, Vec<f32>) {
    let weights: Vec<[f32; N]> = (0..len)
        .map(|i| {
            let v = (i as f32 * 0.1).sin() * 0.5 + 0.5;
            [v; N]
        })
        .collect();
    let state: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).sin()).collect();
    (weights, state)
}

fn bench_dot_4x_interleaved_avx512(c: &mut Criterion) {
    let sizes = [16, 64, 256, 1024, 4096];
    let mut group = c.benchmark_group("dot_4x_interleaved_avx512");

    for &size in &sizes {
        let (weights, state, _) = generate_test_data(size);

        group.bench_function(format!("fallback_{}", size), |b| {
            b.iter(|| unsafe { scalar_ref::dot_product_4x_interleaved_fallback(&weights, &state) })
        });

        group.bench_function(format!("avx2_{}", size), |b| {
            b.iter(|| unsafe { dot_4x::dot_product_4x_interleaved_avx2(&weights, &state) })
        });

        if std::is_x86_feature_detected!("avx512f") {
            group.bench_function(format!("avx512_{}", size), |b| {
                b.iter(|| unsafe { dot_4x::dot_product_4x_interleaved_avx512(&weights, &state) })
            });
        }
    }
    group.finish();
}

fn bench_dot_4x_interleaved_dual_frame_avx512(c: &mut Criterion) {
    let sizes = [16, 64, 256, 1024, 4096];
    let mut group = c.benchmark_group("dot_4x_dual_frame_avx512");

    for &size in &sizes {
        let (weights, state_f0, state_f1) = generate_test_data(size);

        group.bench_function(format!("fallback_{}", size), |b| {
            b.iter(|| unsafe {
                scalar_ref::dot_product_4x_interleaved_dual_frame_fallback(
                    &weights, &state_f0, &state_f1,
                )
            })
        });

        group.bench_function(format!("avx2_{}", size), |b| {
            b.iter(|| unsafe {
                dot_4x::dot_product_4x_interleaved_dual_frame_avx2(&weights, &state_f0, &state_f1)
            })
        });

        if std::is_x86_feature_detected!("avx512f") {
            group.bench_function(format!("avx512_{}", size), |b| {
                b.iter(|| unsafe {
                    dot_4x::dot_product_4x_interleaved_dual_frame_avx512(
                        &weights, &state_f0, &state_f1,
                    )
                })
            });
        }
    }
    group.finish();
}

fn bench_dot_8x_f32_avx2(c: &mut Criterion) {
    let sizes = [16, 64, 256, 1024, 4096];
    let mut group = c.benchmark_group("dot_8x_f32");

    for &size in &sizes {
        let (weights, state) = generate_f32_test_data::<8>(size);

        group.bench_function(format!("scalar_{}", size), |b| {
            b.iter(|| unsafe { dot_product_8x_f32_scalar(&weights, &state) })
        });

        group.bench_function(format!("avx2_{}", size), |b| {
            b.iter(|| unsafe { dot_product_8x_f32_avx2(&weights, &state) })
        });
    }
    group.finish();
}

fn bench_dot_16x_f32_avx512(c: &mut Criterion) {
    let sizes = [16, 64, 256, 1024, 4096];
    let mut group = c.benchmark_group("dot_16x_f32");

    for &size in &sizes {
        let (weights, state) = generate_f32_test_data::<16>(size);

        group.bench_function(format!("scalar_{}", size), |b| {
            b.iter(|| unsafe { dot_product_16x_f32_scalar(&weights, &state) })
        });

        if std::is_x86_feature_detected!("avx512f") {
            group.bench_function(format!("avx512_{}", size), |b| {
                b.iter(|| unsafe { dot_product_16x_f32_avx512(&weights, &state) })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dot_4x_interleaved_avx512,
    bench_dot_4x_interleaved_dual_frame_avx512,
    bench_dot_8x_f32_avx2,
    bench_dot_16x_f32_avx512,
);
criterion_main!(benches);
