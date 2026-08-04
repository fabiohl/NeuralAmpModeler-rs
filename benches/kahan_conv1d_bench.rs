// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Micro-benchmarks for Kahan-compensated vs plain accumulation in Conv1d.
//!
//! Isolates the inner-loop overhead of Kahan summation against naive
//! accumulation across kernel sizes and channel configurations, using the
//! AVX2 `dot_product_4x_interleaved` primitive.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench kahan_conv1d_bench
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::math::common::half::f32_to_f16_bits;
use neural_amp_modeler_rs::math::common::{AlignedVec, Avx2Math, SimdMath, kahan_add};
use neural_amp_modeler_rs::models::wavenet::{Conv1d, Conv1dDyn};
use std::hint::black_box;

fn generate_weights(in_ch: usize, out_ch: usize, k: usize) -> Vec<u16> {
    let total = out_ch * k * in_ch * 4;
    (0..total)
        .map(|i| {
            let v = (i as f32 * 0.13).sin() * 0.25 + 0.15;
            f32_to_f16_bits(v)
        })
        .collect()
}

fn generate_weights_f32(in_ch: usize, out_ch: usize, k: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..out_ch * in_ch * k)
        .map(|i| (i as f32 * 0.13).sin() * 0.25 + 0.15)
        .collect();
    let num_blocks = out_ch.div_ceil(4);
    let total = num_blocks * k * in_ch * 4;
    let mut weights = vec![0.0f32; total];
    for b in 0..num_blocks {
        for kt in 0..k {
            for in_c in 0..in_ch {
                for lane in 0..4 {
                    let out_c = b * 4 + lane;
                    let target_idx = b * (k * in_ch * 4) + kt * (in_ch * 4) + in_c * 4 + lane;
                    if out_c < out_ch {
                        let raw_idx = (out_c * in_ch + in_c) * k + kt;
                        weights[target_idx] = raw[raw_idx];
                    }
                }
            }
        }
    }
    weights
}

fn generate_input(in_ch: usize, kernel: usize, n_frames: usize) -> Vec<f32> {
    let total = (n_frames + kernel - 1) * in_ch;
    (0..total)
        .map(|i| (i as f32 * 0.07 + 0.3).sin() * 0.8)
        .collect()
}

fn bench_kahan_inner_loop_isolated(c: &mut Criterion) {
    let in_ch = 16usize;
    let ks = [1usize, 2, 3, 6, 15, 32];
    let mut group = c.benchmark_group("kahan_inner_loop_isolated");

    for &k in &ks {
        let raw = generate_weights(in_ch, 4, k);
        let mut w_taps: Vec<&[[u16; 4]]> = Vec::with_capacity(k);
        for tap in 0..k {
            let start = tap * in_ch * 4;
            let ptr = &raw[start] as *const u16 as *const [u16; 4];
            unsafe {
                w_taps.push(core::slice::from_raw_parts(ptr, in_ch));
            }
        }

        let input: Vec<f32> = (0..k * in_ch)
            .map(|i| (i as f32 * 0.07 + 0.3).sin() * 0.8)
            .collect();
        let mut in_taps: Vec<&[f32]> = Vec::with_capacity(k);
        for tap in 0..k {
            let start = tap * in_ch;
            in_taps.push(&input[start..start + in_ch]);
        }

        group.bench_function(format!("kahan_K{}_{}ch", k, in_ch), |b| {
            b.iter(|| {
                let mut r = [0.0f32; 4];
                let mut c = [0.0f32; 4];
                for tap in 0..k {
                    let dot =
                        unsafe { Avx2Math::dot_product_4x_interleaved(w_taps[tap], in_taps[tap]) };
                    let (s, comp) = kahan_add(r[0], c[0], dot[0]);
                    r[0] = s;
                    c[0] = comp;
                    let (s, comp) = kahan_add(r[1], c[1], dot[1]);
                    r[1] = s;
                    c[1] = comp;
                    let (s, comp) = kahan_add(r[2], c[2], dot[2]);
                    r[2] = s;
                    c[2] = comp;
                    let (s, comp) = kahan_add(r[3], c[3], dot[3]);
                    r[3] = s;
                    c[3] = comp;
                }
                black_box(r)
            })
        });

        group.bench_function(format!("plain_K{}_{}ch", k, in_ch), |b| {
            b.iter(|| {
                let mut r = [0.0f32; 4];
                for tap in 0..k {
                    let dot =
                        unsafe { Avx2Math::dot_product_4x_interleaved(w_taps[tap], in_taps[tap]) };
                    r[0] += dot[0];
                    r[1] += dot[1];
                    r[2] += dot[2];
                    r[3] += dot[3];
                }
                black_box(r)
            })
        });
    }
    group.finish();
}

fn make_static_conv<const IN: usize, const OUT: usize, const K: usize>(
    raw: &[f32],
) -> Conv1d<IN, OUT, K> {
    let weights = AlignedVec::from_vec(raw.to_vec()).expect("bench allocation failed");
    Conv1d {
        weights,
        bias: AlignedVec::from_vec(vec![0.0; OUT]).expect("bench allocation failed"),
        do_bias: false,
        dilation: 1,
    }
}

fn conv1d_dyn_from_raw(raw: &[f32], in_ch: usize, out_ch: usize, kernel: usize) -> Conv1dDyn {
    Conv1dDyn {
        weights: AlignedVec::from_vec(raw.to_vec()).expect("bench allocation failed"),
        bias: AlignedVec::from_vec(vec![0.0; out_ch]).expect("bench allocation failed"),
        do_bias: false,
        dilation: 1,
        in_ch,
        out_ch,
        num_blocks: out_ch.div_ceil(4),
        interleave_width: 4,
        kernel,
    }
}

macro_rules! bench_conv1d_config {
    ($group:expr, $in_ch:literal, $out_ch:literal, $k:literal, $label:expr, $raw:expr, $input:expr, $n_frames:expr) => {{
        let static_conv = make_static_conv::<$in_ch, $out_ch, $k>($raw);
        let mixin = [0.0f32; $out_ch];
        let start_frame = $k - 1;

        $group.bench_function(format!("static_kahan_{}", $label), |b| {
            b.iter(|| {
                let mut out = [0.0f32; $out_ch];
                for f in 0..$n_frames {
                    unsafe {
                        static_conv.process_single_frame_with_mixin::<Avx2Math>(
                            $input,
                            &mut out,
                            start_frame + f,
                            &mixin,
                        );
                    }
                }
                black_box(out)
            })
        });
    }};
}

fn bench_full_conv1d_kahan_vs_nokahan(c: &mut Criterion) {
    let configs: [(usize, usize, usize, &str); 4] = [
        (8, 8, 3, "IN08_OUT08_K3"),
        (8, 16, 3, "IN08_OUT16_K3"),
        (16, 16, 3, "IN16_OUT16_K3"),
        (12, 12, 3, "IN12_OUT12_K3"),
    ];
    let n_frames = 64usize;

    let mut group = c.benchmark_group("conv1d_kahan_full");

    for &(in_ch, out_ch, k, label) in &configs {
        let raw_f32 = generate_weights_f32(in_ch, out_ch, k);
        let input = generate_input(in_ch, k, n_frames + 64);
        let dyn_conv = conv1d_dyn_from_raw(&raw_f32, in_ch, out_ch, k);
        let start_frame = k - 1;

        match (in_ch, out_ch, k) {
            (8, 8, 3) => {
                bench_conv1d_config!(&mut group, 8, 8, 3, label, &raw_f32, &input, n_frames)
            }
            (8, 16, 3) => {
                bench_conv1d_config!(&mut group, 8, 16, 3, label, &raw_f32, &input, n_frames)
            }
            (16, 16, 3) => {
                bench_conv1d_config!(&mut group, 16, 16, 3, label, &raw_f32, &input, n_frames)
            }
            (12, 12, 3) => {
                bench_conv1d_config!(&mut group, 12, 12, 3, label, &raw_f32, &input, n_frames)
            }
            _ => {}
        }

        let mut out_dyn = vec![0.0f32; out_ch];

        group.bench_function(format!("dyn_nokahan_{}", label), |b| {
            b.iter(|| {
                for f in 0..n_frames {
                    unsafe {
                        dyn_conv.process_single_frame::<Avx2Math>(
                            &input,
                            &mut out_dyn,
                            start_frame + f,
                            None,
                        );
                    }
                }
                black_box(out_dyn[0])
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_kahan_inner_loop_isolated,
    bench_full_conv1d_kahan_vs_nokahan,
);
criterion_main!(benches);
