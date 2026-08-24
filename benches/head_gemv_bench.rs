// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Micro-benchmark for the Lite CH12 head GEMV masked path (in=12, out=6).
//!
//! Isolates `gemv_no_bias_f32_avx2(12, 6, 1)` to measure the cost of the
//! masked tail path (`out_len=6 < 8`) vs the full 8-wide path.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench head_gemv_bench
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::math::gemm::gemv::gemv_no_bias_f32_avx2;

mod common;

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

criterion_group!(
    name = head_gemv;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1))
        .noise_threshold(0.05);
    targets = bench_head_gemv_12x6, bench_head_gemv_12x8
);

criterion_main!(head_gemv);
