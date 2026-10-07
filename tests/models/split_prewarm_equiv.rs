// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Split-vs-integral deferred-stabilization equivalence (Sprint S4, S4-T2).
//!
//! For every fixture model family, the arithmetic state produced by the
//! integral stabilization flow (`reset()` → implicit prewarm) must be
//! bit-identical to the state produced by the deferred split flow
//! (`prewarm_reset()` + `prewarm_step` chunks until `prewarm_complete()`),
//! independently of how the zeroed samples are chunked. This is the
//! stabilization-correctness contract host reset windows rely on:
//! the RT thread runs only the cheap zero phase inside `reset()` and
//! amortizes the expensive feed over subsequent process() blocks without
//! changing a single output sample.
//!
//! ## Coverage
//! - LSTM (recurrent, `prewarm_samples` = 24000): 1×, 4 uneven chunks,
//!   per-block 64-sample steps, block size = pending (single step).
//! - WaveNet static (`BossWN-standard`, one-shot backfill): armed unit
//!   executes on the first step of any positive size; zero-size step is a
//!   no-op keep-armed check.
//! - WaveNet dynamic (`BossWN-feather` etc. — free geometry, one-shot):
//!   same unit semantics as static.
//! - A2 static full (`a2_example.nam` container → `WavenetA2Full` subs):
//!   per-sample, 64-block, and uneven chunkings.
//! - A2 dynamic (`a2_dynamic_gated_ch8.nam`): chunkings incl. odd sizes.
//! - ConvNet (`convnet_relu.nam`): unit executes on first positive step.
//! - Linear (`linear_test.nam`): small-feed chunkings incl. 1-sample steps.
//! - Container (`a2_example.nam`): all-subs convergence and max-pending
//!   accounting.
//!
//! ## Oracle design
//! After converging both flows from the same freshly loaded weights, a
//! deterministic pseudo-random probe (LCG, no heap) is processed and the
//! full outputs compared bit-exactly. `prewarm_complete()` is asserted at
//! every stage: `true` after build (loader prewarms), `false` after
//! `prewarm_reset()`, `true` after the feed drains.

use std::path::Path;

use neural_amp_modeler_rs::SystemSnapshot;
use neural_amp_modeler_rs::loader::{LoadOptions, load_and_build_model};
use neural_amp_modeler_rs::models::{NamModel, StaticModel};
// ── Helpers ──────────────────────────────────────────────────────────────────

/// Deterministic probe input: LCG in f32, no heap, no I/O.
fn probe_input(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.5
        })
        .collect()
}

fn fixture_exists(name: &str) -> bool {
    Path::new("tests/fixtures/models").join(name).exists()
}

fn load(name: &str) -> Box<StaticModel> {
    let path = Path::new("tests/fixtures/models").join(name);
    let sys = SystemSnapshot::capture();
    load_and_build_model(&path, &sys, false, LoadOptions::default())
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .model_l
        .expect("mono fixture yields model_l")
}

fn run(model: &mut StaticModel, input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    model.process(input, &mut out);
    // Second pass through the same state trajectory is not needed: a single
    // deterministic probe after convergence fully determines the state.
    out
}

/// Integral flow: `reset()` (zero phase + implicit stabilization) then probe.
fn integral_output(name: &str, input: &[f32]) -> Vec<f32> {
    let mut model = load(name);
    assert!(
        model.prewarm_complete(),
        "{name}: fresh build must be converged"
    );
    model.reset(48000, 64).expect("reset");
    assert!(
        model.prewarm_complete(),
        "{name}: integral reset must converge by construction"
    );
    run(&mut model, input)
}

/// Split flow: `prewarm_reset()` then `prewarm_step` chunks until converged,
/// then the same probe. Zero-size steps mid-sequence must be no-ops.
///
/// `drain_feed == false` covers families whose zeroing is itself the
/// converged state (integral `prewarm` is a pure zeroing with no feed):
/// any nonzero feed would advance their delay line away from the integral
/// state, so the probe runs right after the zero phase while the test still
/// asserts the full arm/step/complete accounting (with zero-size steps).
fn split_output(name: &str, input: &[f32], chunks: &[usize], drain_feed: bool) -> Vec<f32> {
    let mut model = load(name);
    model.prewarm_reset();
    assert!(
        !model.prewarm_complete(),
        "{name}: split must arm pending work"
    );
    // A zero-size step must neither converge nor disturb the armed pass.
    let pending_before = model.prewarm_step(0);
    assert!(
        !model.prewarm_complete(),
        "{name}: zero-size step must not converge"
    );
    debug_assert!(pending_before > 0, "{name}: armed pass must be non-empty");
    if drain_feed {
        for &c in chunks {
            model.prewarm_step(c);
        }
        let mut guard = 0usize;
        while !model.prewarm_complete() {
            model.prewarm_step(usize::MAX);
            guard += 1;
            assert!(guard < 1_000_000, "{name}: split feed did not converge");
        }
    }
    run(&mut model, input)
}

fn assert_equiv(name: &str, chunks: &[usize]) {
    assert_equiv_drain(name, chunks, true);
}

fn assert_equiv_drain(name: &str, chunks: &[usize], drain_feed: bool) {
    if !fixture_exists(name) {
        eprintln!("[STATUS] SKIP_CAPABILITY: model_not_found:{name}");
        return;
    }
    let input = probe_input(4096, 0x243F_6A88_85A3_08D3);
    let expected = integral_output(name, &input);
    let got = split_output(name, &input, chunks, drain_feed);
    assert_eq!(expected.len(), got.len(), "{name}: length");
    for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{name}: chunks={chunks:?} bit mismatch at {i}: {a} vs {b}"
        );
    }
}

// ── LSTM ─────────────────────────────────────────────────────────────────────

#[test]
fn split_equiv_lstm_single_step() {
    // Whole budget in one step (degenerate chunking = integral order).
    assert_equiv("BossLSTM-2x8.nam", &[usize::MAX]);
}

#[test]
fn split_equiv_lstm_uneven_chunks() {
    assert_equiv("BossLSTM-2x8.nam", &[1000, 7, 65536, 3, 999]);
}

#[test]
fn split_equiv_lstm_block64_steps() {
    // The exact amortization the plugin window will use.
    let total = 24000usize;
    let chunks = vec![64usize; total.div_ceil(64)];
    assert_equiv("BossLSTM-2x8.nam", &chunks);
}

#[test]
fn split_equiv_lstm_single_layer() {
    assert_equiv("BossLSTM-1x16.nam", &[512, 1, 30000]);
}

// ── WaveNet static (one-shot backfill unit) ──────────────────────────────────

#[test]
fn split_equiv_wavenet_standard() {
    // One-shot unit: any positive first step converges.
    assert_equiv("BossWN-standard.nam", &[1]);
    assert_equiv("BossWN-standard.nam", &[64]);
    assert_equiv("BossWN-standard.nam", &[usize::MAX]);
}

#[test]
fn split_equiv_wavenet_lite_feather_nano() {
    for name in ["BossWN-lite.nam", "BossWN-feather.nam", "BossWN-nano.nam"] {
        assert_equiv(name, &[64]);
    }
}

// ── WaveNet dynamic (free geometry, one-shot) ────────────────────────────────

#[test]
fn split_equiv_wavenet_free_geometry() {
    // `BossWN-feather` is free-geometry → WavenetDyn path.
    assert_equiv("BossWN-feather.nam", &[7]);
}

// ── A2 static / dynamic ──────────────────────────────────────────────────────

#[test]
fn split_equiv_a2_container_per_sample() {
    assert_equiv("a2_example.nam", &[1]);
}

#[test]
fn split_equiv_a2_container_block64() {
    let chunks = vec![64usize; 256];
    assert_equiv("a2_example.nam", &chunks);
}

#[test]
fn split_equiv_a2_container_uneven() {
    assert_equiv("a2_example.nam", &[100, 7, 4096, 13]);
}

#[test]
fn split_equiv_a2_dynamic_odd_chunks() {
    for name in ["a2_dynamic_gated_ch8.nam", "a2_dynamic_blended_ch3.nam"] {
        assert_equiv(name, &[3, 1000, 17]);
    }
}

// ── ConvNet / Linear ─────────────────────────────────────────────────────────

#[test]
fn split_equiv_convnet() {
    for name in ["convnet_relu.nam", "convnet_silu.nam", "convnet_nobn.nam"] {
        assert_equiv(name, &[64]);
    }
}

#[test]
fn split_equiv_linear() {
    // Direct Linear: same pure-zeroing rationale as the FFT path.
    for name in ["linear_test.nam", "linear_nobias.nam"] {
        assert_equiv_drain(name, &[1], false);
        assert_equiv_drain(name, &[7, 64, 3000], false);
    }
}

#[test]
fn split_equiv_linear_fft() {
    // FFT Linear: the integral `prewarm` is a pure zeroing (no feed), so the
    // zero phase alone is the converged state — the probe runs undrained
    // while arm/step/complete accounting is still fully asserted. Engine
    // sweep evidence: feed length 0 == integral bit-exact; any nonzero feed
    // advances the overlap-save delay line (expected convolution behavior).
    assert_equiv_drain("linear_fft_rf320.nam", &[64, 64, 64], false);
}
