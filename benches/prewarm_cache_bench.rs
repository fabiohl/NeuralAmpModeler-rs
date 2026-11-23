// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Prewarm/reset cost characterization for the prewarm-cache decision (DEC-05).
//!
//! Compares, per model family, the four stabilization paths a consumer can
//! choose between on a reset:
//!
//! - `reset_integral`: `reset()` = clear state + integral prewarm (full reset).
//! - `clear_state`: the zero phase alone (what the RT thread runs inline).
//! - `prewarm_only`: the integral stabilization without the clear phase.
//! - `reset_split_1024`: `prewarm_reset` + `prewarm_step(1024)` until converged
//!   (the amortized path interactive hosts use).
//!
//! A cached steady-state snapshot would only be worth its memory and RT risk
//! if the stabilization columns are a meaningful share of the reset budget;
//! see [`docs/benchmarks.md`](../docs/benchmarks.md) for the recorded run and
//! the non-adoption rationale.
//!
//! ```sh
//! # Build under the host cgroup guard, then run unconstrained on the bench core:
//! cargo bench --no-run --bench prewarm_cache_bench
//! taskset -c "$NAM_BENCH_CORE" <binary> --bench 'S11_' --noplot
//! ```

#[path = "common.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::models::{NamModel, StaticModel};

const SAMPLE_RATE: u32 = 48_000;
const BLOCK: usize = 64;
const SPLIT_STEP: usize = 1024;

/// Fixtures covering every stabilization strategy: A1 backfill, A2 zero-feed,
/// LSTM recurrence, Linear FFT ring, ConvNet, and the canonical post-stack head.
const FIXTURES: &[&str] = &[
    "BossWN-standard.nam",
    "wavenet_a2_full.nam",
    "wavenet_a2_lite.nam",
    "lstm_2x24.nam",
    "linear_fft_rf2048.nam",
    "convnet_relu.nam",
    "wavenet_post_head_multistage.nam",
];

fn load(name: &str) -> Option<Box<StaticModel>> {
    let data = common::load_model_data(name)?;
    let mut model = build_model(&data).ok()?;
    model.prewarm(2048);
    Some(model)
}

fn bench_fixture(c: &mut Criterion, name: &str) {
    let Some(mut model) = load(name) else {
        return;
    };
    let label = name.trim_end_matches(".nam");
    let pending = model.prewarm_samples();
    eprintln!("S11/{label}: prewarm_samples={pending}");

    c.bench_function(&format!("S11_reset_integral/{label}"), |b| {
        b.iter(|| {
            model
                .reset(SAMPLE_RATE, BLOCK)
                .expect("integral reset must not fail");
        });
    });

    c.bench_function(&format!("S11_clear_state/{label}"), |b| {
        b.iter(|| {
            model.clear_state();
        });
    });

    c.bench_function(&format!("S11_prewarm_only/{label}"), |b| {
        b.iter(|| {
            model.prewarm(std::hint::black_box(2048));
        });
    });

    c.bench_function(&format!("S11_reset_split_1024/{label}"), |b| {
        b.iter(|| {
            model.prewarm_reset();
            while model.prewarm_step(SPLIT_STEP) > 0 {}
        });
    });
}

fn bench_prewarm_reset(c: &mut Criterion) {
    for name in FIXTURES {
        bench_fixture(c, name);
    }
}

criterion_group!(benches, bench_prewarm_reset);
criterion_main!(benches);
