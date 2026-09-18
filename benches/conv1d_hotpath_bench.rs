// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Micro-calibration bench for two WaveNet Conv1D hot-path policies.
//!
//! Recalibration protocol: docs/benchmarks.md §"Conv1D Hotpath Bench — Calibration Protocol"
//!
//! ## `prefetch_guard` — software-prefetch dilation threshold
//!
//! Measures the static single-frame causal kernel
//! (`Conv1d::process_single_frame_with_mixin`, K=3, 64-frame block) for the
//! four catalog channel geometries (CH16/CH8/CH12/CH4) at dilations 1..32.
//! The causal tap-copy loop is where `prefetch_strategy_simple` /
//! `prefetch_strategy_2stage` fire (`src/math/common/ops.rs`); comparing
//! medians per dilation between code states (guard on/off in
//! `prefetch_strategy_simple`) keeps the simple strategy's dilation
//! threshold calibrated on the current toolchain/CPU.
//!
//! ```sh
//! taskset -c 4 cargo bench --bench conv1d_hotpath_bench -- --save-baseline prefetch-a
//! # ...toggle the dilation guard in prefetch_strategy_simple...
//! taskset -c 4 cargo bench --bench conv1d_hotpath_bench -- --baseline prefetch-a
//! ```
//!
//! ## `ch12_lane_route` — padded 16-lane vs 8+4 split (WaveNet Lite)
//!
//! Kernel-level A/B for the WaveNet Lite CH12 geometry (IN=12, K=3 → 36 taps
//! per layer call). The padded route (`select_interleave_width(12) = 16`)
//! runs the 16-wide YMM kernel over zero-padded lanes 12..15; the 8+4 split
//! runs the 8-wide kernel on lanes 0..8 plus the 4-wide kernel on lanes
//! 8..12 over the same taps. Both order per-lane accumulation identically
//! (same tap order, same unroll/tree reduction), so the 12 useful lanes are
//! bit-equal — asserted once in setup below.

use criterion::{Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::math::common::{Avx2Math, SimdMath};
use neural_amp_modeler_rs::models::wavenet::Conv1d;
use std::hint::black_box;
use std::time::Duration;

/// Kernel size of the catalog WaveNet models (`BossWN-*.nam`).
const K: usize = 3;
/// Frames per block, matching the canonical 64-sample RT block.
const N_FRAMES: usize = 64;

/// Dilations to calibrate; the 2-stage strategy owns dilation >= 128.
const DILATIONS: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// Builds interleaved weights `[block][k][in_ch][W]` (zero-padded lanes) from
/// raw row-major `[out][in][k]` weights, mirroring the loader transposes.
fn interleaved_weights<const W: usize>(in_ch: usize, out_ch: usize, k: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..out_ch * in_ch * k)
        .map(|i| (i as f32 * 0.13).sin() * 0.25 + 0.15)
        .collect();
    let num_blocks = out_ch.div_ceil(W);
    let mut w = vec![0.0f32; num_blocks * W * in_ch * k];
    for (b, out_c) in (0..out_ch).enumerate() {
        let block = b / W;
        let lane = b % W;
        for kt in 0..k {
            for in_c in 0..in_ch {
                let target = block * (k * in_ch * W) + kt * (in_ch * W) + in_c * W + lane;
                w[target] = raw[(out_c * in_ch + in_c) * k + kt];
            }
        }
    }
    w
}

fn bench_prefetch_guard(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetch_guard");
    group.warm_up_time(Duration::from_millis(1200));
    group.measurement_time(Duration::from_secs(3));

    for &(in_ch, out_ch) in &[(16usize, 16usize), (8, 8), (12, 12), (4, 4)] {
        for &dilation in &DILATIONS {
            // Per-geometry dispatch mirrors the catalog SKUs; the literal
            // channel pairs keep the const-generic monomorphization explicit.
            match (in_ch, out_ch) {
                (16, 16) => bench_prefetch_config::<16, 16, 16>(&mut group, dilation),
                (8, 8) => bench_prefetch_config::<8, 8, 8>(&mut group, dilation),
                (12, 12) => bench_prefetch_config::<12, 12, 16>(&mut group, dilation),
                (4, 4) => bench_prefetch_config::<4, 4, 4>(&mut group, dilation),
                _ => unreachable!("catalog geometry table drifted"),
            }
        }
    }
    group.finish();
}

fn bench_prefetch_config<const IN: usize, const OUT: usize, const W: usize>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    dilation: usize,
) {
    let raw = interleaved_weights::<W>(IN, OUT, K);
    let weights = neural_amp_modeler_rs::math::common::AlignedVec::from_vec(raw)
        .expect("bench weight allocation failed");
    let bias = neural_amp_modeler_rs::math::common::AlignedVec::from_vec(vec![0.0; OUT])
        .expect("bench bias allocation failed");
    let conv: Conv1d<IN, OUT, K> = Conv1d {
        weights,
        bias,
        do_bias: false,
        dilation,
    };
    let mixin = [0.0f32; OUT];

    // Mirrored-style history buffer: deep enough for the oldest causal tap of
    // the last frame, so tap strides exercise the real dilation distances.
    let buf_frames = dilation * (K - 1) + N_FRAMES + 16;
    let input: Vec<f32> = (0..buf_frames * IN)
        .map(|i| (i as f32 * 0.07 + 0.3).sin() * 0.8)
        .collect();
    let start_frame = dilation * (K - 1);

    group.bench_function(format!("dil{dilation}_{IN}x{OUT}"), |b| {
        b.iter(|| {
            let mut out = [0.0f32; OUT];
            for f in 0..N_FRAMES {
                // SAFETY: `input` spans `buf_frames * IN` f32s and
                // `start_frame >= dilation * (K - 1)`, satisfying the causal
                // receptive-field contract of the kernel.
                unsafe {
                    conv.process_single_frame_with_mixin::<Avx2Math>(
                        &input,
                        &mut out,
                        start_frame + f,
                        &mixin,
                    );
                }
            }
            black_box(out)
        })
    });
}

fn bench_ch12_lane_route(c: &mut Criterion) {
    const IN: usize = 12;
    const TAPS: usize = IN * K;

    // Raw row-major [out=12][in=12][k=3] weights; zero-padded to 16 lanes for
    // the padded route, split 8+4 for the composite route.
    let raw: Vec<f32> = (0..12 * IN * K)
        .map(|i| (i as f32 * 0.13).sin() * 0.25 + 0.15)
        .collect();
    let lane_weight = |tap: usize, out_c: usize| -> f32 {
        let k = tap / IN;
        let in_c = tap % IN;
        raw[(out_c * IN + in_c) * K + k]
    };

    let w16: Vec<[f32; 16]> = (0..TAPS)
        .map(|t| {
            let mut row = [0.0f32; 16];
            for (lane, cell) in row.iter_mut().enumerate() {
                *cell = if lane < 12 { lane_weight(t, lane) } else { 0.0 };
            }
            row
        })
        .collect();
    let w8: Vec<[f32; 8]> = (0..TAPS)
        .map(|t| core::array::from_fn(|lane| lane_weight(t, lane)))
        .collect();
    let w4: Vec<[f32; 4]> = (0..TAPS)
        .map(|t| core::array::from_fn(|lane| lane_weight(t, 8 + lane)))
        .collect();

    // Taps = one layer call worth of causal input (K=3, IN=12).
    let state: Vec<f32> = (0..TAPS)
        .map(|t| (t as f32 * 0.07 + 0.3).sin() * 0.8)
        .collect();
    let init16: [f32; 16] =
        core::array::from_fn(|lane| if lane < 12 { 0.5 + lane as f32 } else { 0.0 });
    let init8: [f32; 8] = core::array::from_fn(|lane| init16[lane]);
    let init4: [f32; 4] = core::array::from_fn(|lane| init16[8 + lane]);

    // Bit-parity of the 12 useful lanes (padded-16 vs 8+4 composite).
    let r16 = unsafe { Avx2Math::dot_product_16x_f32_accumulate(&w16, &state, &init16) };
    let (r8, r4) = unsafe {
        (
            Avx2Math::dot_product_8x_f32_accumulate(&w8, &state, &init8),
            Avx2Math::dot_product_4x_f32_accumulate(&w4, &state, &init4),
        )
    };
    assert!(
        r16[..12]
            == [
                r8[0], r8[1], r8[2], r8[3], r8[4], r8[5], r8[6], r8[7], r4[0], r4[1], r4[2], r4[3]
            ]
            && r16[12..].iter().all(|&v| v == 0.0),
        "8+4 composite diverges from padded-16 route"
    );

    let mut group = c.benchmark_group("ch12_lane_route");
    group.warm_up_time(Duration::from_millis(1200));
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("lane16_padded", |b| {
        b.iter(|| {
            // SAFETY: AVX2 dispatched impl; weights.len() == state.len() == TAPS.
            let r = unsafe { Avx2Math::dot_product_16x_f32_accumulate(&w16, &state, &init16) };
            black_box(r)
        })
    });
    group.bench_function("lane8_plus_4", |b| {
        b.iter(|| {
            // SAFETY: AVX2 dispatched impl; weights.len() == state.len() == TAPS.
            let r8 = unsafe { Avx2Math::dot_product_8x_f32_accumulate(&w8, &state, &init8) };
            let r4 = unsafe { Avx2Math::dot_product_4x_f32_accumulate(&w4, &state, &init4) };
            black_box((r8, r4))
        })
    });
    group.finish();
}

criterion_group!(benches, bench_prefetch_guard, bench_ch12_lane_route);
criterion_main!(benches);
