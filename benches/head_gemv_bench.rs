// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Micro-benchmarks for the batched f32 GEMV head paths.
//!
//! `head_gemv_12x6_masked` isolates the previous masked tail (`out_len=6 < 8`)
//! against the clean 8-wide path. The `out1`/`out4`/`out12` cases exercise the
//! mask-free tail and the 4-frame `out_len == 1` tile over a realistic batch.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench head_gemv_bench
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::math::gemm::gemv::gemv_no_bias_f32_avx2;

mod common;

fn batch_data(in_len: usize, out_len: usize, num_frames: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let in_frames: Vec<f32> = (0..in_len * num_frames)
        .map(|i| (i as f32 * 0.17).sin())
        .collect();
    let weights: Vec<f32> = (0..in_len * out_len)
        .map(|i| (i as f32 * 0.13).sin() * 0.5)
        .collect();
    let out_frames = vec![0.0f32; out_len * num_frames];
    (in_frames, weights, out_frames)
}

fn bench_head_gemv_12x6(c: &mut Criterion) {
    let (in_frames, weights, mut out_frames) = common::make_f32_test_data(12, 6);
    c.bench_function("head_gemv_12x6_masked", |b| {
        b.iter(|| unsafe {
            gemv_no_bias_f32_avx2(&in_frames, &weights, &mut out_frames, 1);
        });
    });
}

fn bench_head_gemv_12x8(c: &mut Criterion) {
    let (in_frames, weights, mut out_frames) = common::make_f32_test_data(12, 8);
    c.bench_function("head_gemv_12x8_clean", |b| {
        b.iter(|| unsafe {
            gemv_no_bias_f32_avx2(&in_frames, &weights, &mut out_frames, 1);
        });
    });
}

fn bench_gemv_out1_in64_batch64(c: &mut Criterion) {
    let (in_frames, weights, mut out_frames) = batch_data(64, 1, 64);
    c.bench_function("gemv_out1_in64_batch64", |b| {
        b.iter(|| unsafe {
            gemv_no_bias_f32_avx2(&in_frames, &weights, &mut out_frames, 64);
        });
    });
}

fn bench_gemv_out4_in64_batch64(c: &mut Criterion) {
    let (in_frames, weights, mut out_frames) = batch_data(64, 4, 64);
    c.bench_function("gemv_out4_in64_batch64", |b| {
        b.iter(|| unsafe {
            gemv_no_bias_f32_avx2(&in_frames, &weights, &mut out_frames, 64);
        });
    });
}

fn bench_gemv_out12_in64_batch64(c: &mut Criterion) {
    let (in_frames, weights, mut out_frames) = batch_data(64, 12, 64);
    c.bench_function("gemv_out12_in64_batch64", |b| {
        b.iter(|| unsafe {
            gemv_no_bias_f32_avx2(&in_frames, &weights, &mut out_frames, 64);
        });
    });
}

criterion_group!(
    name = head_gemv;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1))
        .noise_threshold(0.05);
    targets = bench_head_gemv_12x6, bench_head_gemv_12x8,
        bench_gemv_out1_in64_batch64, bench_gemv_out4_in64_batch64,
        bench_gemv_out12_in64_batch64
);

criterion_main!(head_gemv);
