// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Dynamic stage-level off-RT micro-benchmarks.
//!
//! Isolates per-stage computation cost for the A2 Dynamic model:
//!   1. Dilated convolution (`process_single_frame`)
//!   2. Mixin GEMV (condition × mixin_w)
//!   3. Activation / Gating / Blending
//!   4. Head 1x1 projection / accumulation
//!   5. L1x1 residual sum
//!
//! Each benchmark approximates the stage workload with synthetic data
//! at dimensions representative of real A2Dyn models (CH=8 bottleneck=8).

use criterion::Criterion;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::models::NamModel;

use std::hint::black_box;

use super::common::{generate_sine_440hz, make_wavenet_a2_dyn_data};

// ── Synthetic stage workloads (exact arithmetic patterns) ──────────────

/// Stage 1: Dilated convolution (K=3, bottleneck=8, CH=8, groups=1).
fn stage_conv() {
    const CH: usize = 8;
    const BOTTLENECK: usize = 8;
    const K: usize = 3;

    let history = [0.1f32; CH * (K + 1)];
    let mut z = [0.0f32; BOTTLENECK];
    let frame_idx = K;
    let weights = vec![0.01f32; BOTTLENECK * K * CH];

    for (oc, z_val) in z.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for tap in 0..K {
            let hist_start = (frame_idx - tap) * CH;
            let w_start = (oc * K + tap) * CH;
            for c in 0..CH {
                sum += weights[w_start + c] * history[hist_start + c];
            }
        }
        *z_val = sum;
    }
    black_box(&z);
}

/// Stage 2: Mixin GEMV — matrix-vector multiply (z_out_ch × cond_size) @ cond.
fn stage_mixin_gemv(z_out_ch: usize, cond_size: usize) {
    let mixin_w = vec![0.01f32; z_out_ch * cond_size];
    let cond = vec![0.2f32; cond_size];
    let mut mixin_scratch = vec![0.0f32; z_out_ch];

    for (c, scratch_val) in mixin_scratch.iter_mut().enumerate() {
        let base = c * cond_size;
        let mut sum = 0.0;
        for k in 0..cond_size {
            sum += mixin_w[base + k] * cond[k];
        }
        *scratch_val = sum;
    }
    black_box(&mixin_scratch);
}

/// Stage 3: Activation / Gating / Blending — element-wise on z_scratch.
fn stage_activation_gated(z_out_ch: usize) {
    let mut z = vec![0.3f32; z_out_ch];
    let half = z_out_ch / 2;
    for i in 0..half {
        let gate = 1.0 / (1.0 + (-z[i]).exp());
        let activated = z[i + half].max(0.0) + 0.01 * z[i + half].min(0.0);
        z[i] = gate * activated;
    }
    black_box(&z);
}

/// Stage 4: Head 1x1 projection + accumulation.
fn stage_head1x1(bottleneck: usize, head_accum_size: usize) {
    let h1_in = bottleneck;
    let head1x1_w = vec![0.01f32; head_accum_size * h1_in];
    let head1x1_b = vec![0.0f32; head_accum_size];
    let z_scratch = vec![0.2f32; bottleneck];
    let mut head = vec![1.0f32; head_accum_size];

    for (oc, head_val) in head.iter_mut().enumerate() {
        let mut sum = head1x1_b[oc];
        let b_start = oc * h1_in;
        for ic in 0..h1_in {
            sum += head1x1_w[b_start + ic] * z_scratch[ic];
        }
        *head_val += sum;
    }
    black_box(&head);
}

/// Stage 5: L1x1 residual sum — bottleneck × channels projection with bias.
fn stage_l1x1(bottleneck: usize, channels: usize) {
    let l1x1_w = vec![0.01f32; bottleneck * channels];
    let l1x1_b = vec![0.0f32; channels];
    let z_scratch = vec![0.2f32; bottleneck];
    let mut layer_in = vec![0.0f32; channels];

    for (oc, li_val) in layer_in.iter_mut().enumerate() {
        let mut sum = l1x1_b[oc];
        for ic in 0..bottleneck {
            sum += l1x1_w[ic * channels + oc] * z_scratch[ic];
        }
        *li_val += sum;
    }
    black_box(&layer_in);
}

/// End-to-end A2Dyn process (64 samples) — reference baseline.
fn bench_a2dyn_full(c: &mut Criterion) {
    let dyn_data = make_wavenet_a2_dyn_data();
    let mut model = build_model(&dyn_data).expect("Dispatcher failed for A2 Dynamic benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("A2Dyn_Stage_Full_64samp", |b| {
        b.iter(|| {
            model.process(black_box(&input), black_box(&mut output));
        });
    });
}

/// Stage 1: Dilated convolution (K=3, bottleneck=8, CH=8, groups=1).
fn bench_stage_conv(c: &mut Criterion) {
    c.bench_function("A2Dyn_Stage_Conv_B8_CH8_K3", |b| {
        b.iter(stage_conv);
    });
}

/// Stage 2: Mixin GEMV (bottleneck=8, cond=1 — flat path).
fn bench_stage_mixin_flat(c: &mut Criterion) {
    c.bench_function("A2Dyn_Stage_MixinGEMV_B8_C1", |b| {
        b.iter(|| stage_mixin_gemv(8, 1));
    });
}

/// Stage 2: Mixin GEMV (z_out_ch=8, cond=3 — typical multi-channel condition).
fn bench_stage_mixin_multi(c: &mut Criterion) {
    c.bench_function("A2Dyn_Stage_MixinGEMV_B8_C3", |b| {
        b.iter(|| stage_mixin_gemv(8, 3));
    });
}

/// Stage 3: Gated activation (z_out_ch=16 — typical gated 2×bottleneck).
fn bench_stage_gating(c: &mut Criterion) {
    c.bench_function("A2Dyn_Stage_Gating_B8", |b| {
        b.iter(|| stage_activation_gated(16));
    });
}

/// Stage 4: Head 1x1 projection (bottleneck=8, accum_size=8).
fn bench_stage_head1x1(c: &mut Criterion) {
    c.bench_function("A2Dyn_Stage_Head1x1_B8_A8", |b| {
        b.iter(|| stage_head1x1(8, 8));
    });
}

/// Stage 5: L1x1 residual (bottleneck=8, channels=8).
fn bench_stage_l1x1(c: &mut Criterion) {
    c.bench_function("A2Dyn_Stage_L1x1_B8_CH8", |b| {
        b.iter(|| stage_l1x1(8, 8));
    });
}

// ── Higher-capacity variant (bottleneck=16, CH=16) ───────────────────

/// Stage 1: Conv (K=3, bottleneck=16, CH=16).
fn bench_stage_conv_large(c: &mut Criterion) {
    c.bench_function("A2Dyn_Stage_Conv_B16_CH16_K3", |b| {
        b.iter(|| {
            const CH: usize = 16;
            const BOTTLENECK: usize = 16;
            const K: usize = 3;
            let history = [0.1f32; CH * (K + 1)];
            let mut z = [0.0f32; BOTTLENECK];
            let weights = vec![0.01f32; BOTTLENECK * K * CH];
            for (oc, z_val) in z.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for tap in 0..K {
                    let hist_start = (K - tap) * CH;
                    let w_start = (oc * K + tap) * CH;
                    for c in 0..CH {
                        sum += weights[w_start + c] * history[hist_start + c];
                    }
                }
                *z_val = sum;
            }
            black_box(&z);
        });
    });
}

/// Stage 2: Mixin GEMV (z_out_ch=16, cond=1).
fn bench_stage_mixin_large(c: &mut Criterion) {
    c.bench_function("A2Dyn_Stage_MixinGEMV_B16_C1", |b| {
        b.iter(|| stage_mixin_gemv(16, 1));
    });
}

/// Stage 4: Head 1x1 (bottleneck=16, accum=16).
fn bench_stage_head1x1_large(c: &mut Criterion) {
    c.bench_function("A2Dyn_Stage_Head1x1_B16_A16", |b| {
        b.iter(|| stage_head1x1(16, 16));
    });
}

/// Stage 5: L1x1 (bottleneck=16, channels=16).
fn bench_stage_l1x1_large(c: &mut Criterion) {
    c.bench_function("A2Dyn_Stage_L1x1_B16_CH16", |b| {
        b.iter(|| stage_l1x1(16, 16));
    });
}

// ── Combined: all stages in sequence (1 frame, no buffer mgmt) ────────

fn bench_all_stages_combined(c: &mut Criterion) {
    c.bench_function("A2Dyn_Stage_Combined_B8_CH8", |b| {
        b.iter(|| {
            stage_conv();
            stage_mixin_gemv(8, 1);
            stage_activation_gated(16);
            stage_head1x1(8, 8);
            stage_l1x1(8, 8);
        });
    });
}

// ── Criterion harness ─────────────────────────────────────────────────

pub fn bench_a2dyn_stages(c: &mut Criterion) {
    bench_a2dyn_full(c);
    bench_stage_conv(c);
    bench_stage_mixin_flat(c);
    bench_stage_mixin_multi(c);
    bench_stage_gating(c);
    bench_stage_head1x1(c);
    bench_stage_l1x1(c);
    bench_stage_conv_large(c);
    bench_stage_mixin_large(c);
    bench_stage_head1x1_large(c);
    bench_stage_l1x1_large(c);
    bench_all_stages_combined(c);
}
