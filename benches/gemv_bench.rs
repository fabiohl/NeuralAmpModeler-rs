// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Micro-benchmarks for GEMV kernels.
//!
//! Isolates measurement of `fused_add_gemv_avx2` (generic) vs fully-unrolled
//! specialized prototypes for the dimensions listed below:
//! 1×4, 4×4, 4×6, 8×4, 8×6, 8×8 (Out × In).
//!
//! All kernels operate on f16c-quantized weights and f32 inputs/outputs.
//! The fused variant (`fused_add_gemv`) is used because it is the dominant
//! hot-path call in WaveNet and LSTM inference.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench gemv_bench
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::math::common::scalar_ref::fused_add_gemv_fallback;
use neural_amp_modeler_rs::math::gemm::gemv::fused_add_gemv_avx2;

#[path = "gemv/kernels.rs"]
mod kernels;

// ── Synthetic test data ────────────────────────────────────────────────────────

struct GemvTestData {
    in_frame: Vec<f32>,
    weights: Vec<u16>,
    weights_f32: Vec<f32>,
    bias: Vec<f32>,
    out_frame: Vec<f32>,
}

/// Creates deterministic test data for a given (in_len, out_len) pair.
/// Weights are derived from a sinusoidal pattern to avoid degenerate values.
///
/// The f32→f16 quantization here uses intentional truncation (no rounding),
/// producing a weight bit-pattern that diverges from the library's
/// `f32_to_f16_bits` (which uses round-to-nearest-even). Do NOT replace this
/// with the library helper — doing so would change the measured weights and
/// invalidate the benchmark workload.
fn make_test_data(in_len: usize, out_len: usize) -> GemvTestData {
    let in_frame: Vec<f32> = (0..in_len).map(|i| (i as f32 * 0.17).sin()).collect();
    let bias: Vec<f32> = (0..out_len)
        .map(|i| (i as f32 * 0.31).cos() * 0.1)
        .collect();
    let weights: Vec<u16> = (0..in_len * out_len)
        .map(|i| {
            let v = (i as f32 * 0.13).sin() * 0.5;
            let u = v.to_bits();
            let sign = (u >> 16) & 0x8000;
            let exp = (u >> 23) & 0xFF;
            let frac = (u & 0x7F_FFFF) >> 13;
            if exp < 112 {
                0
            } else if exp > 142 {
                (sign | 0x7BFF) as u16
            } else {
                (sign | ((exp - 112) << 10) | (frac & 0x3FF)) as u16
            }
        })
        .collect();
    let weights_f32: Vec<f32> = (0..in_len * out_len)
        .map(|i| (i as f32 * 0.13).sin() * 0.5)
        .collect();
    let out_frame = vec![0.0; out_len];
    GemvTestData {
        in_frame,
        weights,
        weights_f32,
        bias,
        out_frame,
    }
}

// ── Benchmarks ─────────────────────────────────────────────────────────────────

macro_rules! bench_dim {
    ($c:expr, $name:literal, $in_len:expr, $out_len:expr, $specialized:path) => {{
        let mut group = $c.benchmark_group($name);
        let data = make_test_data($in_len, $out_len);

        group.bench_function("generic_avx2", |b| {
            b.iter(|| {
                let mut out = data.out_frame.clone();
                unsafe {
                    fused_add_gemv_avx2(
                        &data.in_frame,
                        &data.weights_f32,
                        &data.bias,
                        &mut out,
                        true,
                    );
                }
                out
            })
        });

        group.bench_function("specialized_avx2", |b| {
            b.iter(|| {
                let mut out = data.out_frame.clone();
                unsafe {
                    $specialized(&data.in_frame, &data.weights, &data.bias, &mut out, true);
                }
                out
            })
        });

        group.bench_function("scalar_fallback", |b| {
            b.iter(|| {
                let mut out = data.out_frame.clone();
                unsafe {
                    fused_add_gemv_fallback(
                        &data.in_frame,
                        &data.weights_f32,
                        &data.bias,
                        &mut out,
                        true,
                    );
                }
                out
            })
        });

        group.finish();
    }};
}

fn bench_gemv_1x4(c: &mut Criterion) {
    bench_dim!(c, "gemv_1x4", 1, 4, kernels::gemv_specialized_1x4);
}

fn bench_gemv_4x4(c: &mut Criterion) {
    bench_dim!(c, "gemv_4x4", 4, 4, kernels::gemv_specialized_4x4);
}

fn bench_gemv_4x6(c: &mut Criterion) {
    bench_dim!(c, "gemv_4x6", 4, 6, kernels::gemv_specialized_4x6);
}

fn bench_gemv_8x4(c: &mut Criterion) {
    bench_dim!(c, "gemv_8x4", 8, 4, kernels::gemv_specialized_8x4);
}

fn bench_gemv_8x6(c: &mut Criterion) {
    bench_dim!(c, "gemv_8x6", 8, 6, kernels::gemv_specialized_8x6);
}

fn bench_gemv_8x8(c: &mut Criterion) {
    bench_dim!(c, "gemv_8x8", 8, 8, kernels::gemv_specialized_8x8);
}

criterion_group!(
    gemv_benches,
    bench_gemv_1x4,
    bench_gemv_4x4,
    bench_gemv_4x6,
    bench_gemv_8x4,
    bench_gemv_8x6,
    bench_gemv_8x8,
);
criterion_main!(gemv_benches);
