// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Reset idempotence and history-independence matrix.
//!
//! The C++ NAMcore `DSP::Reset` (`dsp.cpp:133-140`) always clears every
//! temporal state buffer via `SetMaxBufferSize` before the optional prewarm.
//! A reset performed with `prewarm_on_reset == false` must therefore return a
//! model to its freshly-built condition: any probe driven afterwards must be
//! bit-identical to the same probe driven through a brand-new model.
//!
//! This module proves that invariant generically: instead of hard-coding the
//! family list, it walks the supported identities of [`MODEL_CATALOG`] and
//! applies one parametric test to each. New families added to the catalog are
//! covered automatically — the matrix cannot silently miss a topology.
//!
//! Hardening rules enforced here (`a evidência falta = falha`):
//! * The oracle is the literal `reset(polluted) == fresh` comparison against a
//!   cold-built model (`prewarm: Some(false)`); the previous
//!   `reset(fresh)` fallback is gone (T5.4-5 made cold container construction
//!   prewarm-free).
//! * The number of exercised distributed fixtures is checked **exactly**.
//! * A missing *distributed* fixture, or a build/reset failure of any present
//!   supported fixture, is a failure — never a silent skip.
//! * Only two explicit, justified exclusions remain: LSTM-governed resets
//!   (DEC-03) and absent non-distributable community fixtures.
//!
//! A second, focused test drives the `NamModel::reset` path (not just the raw
//! block state) on a ConvNet 1→1 regression: a kernel-2, dilation-1 block whose
//! only non-zero weight sits on the previous (delay-1) tap. After processing a
//! `1`, a reset-without-prewarm followed by a `0` must read `0 + bias` (the
//! stale tap must not survive the reset).

use std::path::{Path, PathBuf};

use neural_amp_modeler_rs::SystemSnapshot;
use neural_amp_modeler_rs::loader::{LoadOptions, load_and_build_model};
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::catalog::{ModelSupportKind, catalog_entries};

/// Probe length for the reset matrix (impulse + deterministic noise).
const PROBE_LEN: usize = 4096;
/// Negotiated maximum block size used by both the reset and the fresh model.
const MAX_BLOCK: usize = 64;
/// Light anti-vacuity floor; the exact count is asserted separately.
const MIN_EXPECTED_RUNS: usize = 1;

/// Deterministic probe input: LCG in `f32`, no heap, no I/O. The first sample
/// is an impulse so causal delay lines are guaranteed to see non-trivial data.
fn probe_input(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = ((x >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.5;
        out.push(if i == 0 { 1.0 } else { noise });
    }
    out
}

fn fixture_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn load_skip_prewarm(path: &Path) -> Option<Box<neural_amp_modeler_rs::models::StaticModel>> {
    let sys = SystemSnapshot::capture();
    load_and_build_model(
        path,
        &sys,
        false,
        LoadOptions::new().with_prewarm(Some(false)),
    )
    .ok()
    .and_then(|pair| pair.model_l)
}

/// Runs `input` through `model` in `MAX_BLOCK` frames per call (the
/// negotiated maximum), returning the produced samples.
fn run(model: &mut neural_amp_modeler_rs::models::StaticModel, input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    let mut pos = 0usize;
    while pos < input.len() {
        let n = (input.len() - pos).min(MAX_BLOCK);
        model.process(&input[pos..pos + n], &mut out[pos..pos + n]);
        pos += n;
    }
    out
}

/// Bit-exact history-independence check for one fixture.
///
/// The oracle is the literal `reset(polluted) == fresh`, where `fresh` is a
/// model built with `prewarm: Some(false)`. Since T5.4-5 the container loader
/// honours `prewarm: false`, so no family needs the old `reset(fresh)`
/// fallback.
///
/// Returns `Ok(true)` when the fixture was exercised, `Ok(false)` when the
/// model is LSTM-governed (top-level, or nested inside a Container/Sequential)
/// and thus out of scope while DEC-03 stands, and `Err` when the fixture
/// cannot be built in this environment.
fn assert_reset_is_fresh(path: &Path, label: &str) -> Result<bool, ()> {
    // A: build cold, pollute the state, reset without prewarm, probe silence.
    let mut polluted = load_skip_prewarm(path).ok_or(())?;
    if polluted.is_lstm_family() {
        // LSTM reset does not clear its input history (DEC-03 / T3.2): the
        // matrix excludes it rather than asserting an oracle we do not own yet.
        // `is_lstm_family` recurses through Containers and Sequential chains.
        return Ok(false);
    }
    let _ = polluted.set_max_buffer_size(MAX_BLOCK);
    let history = probe_input(PROBE_LEN + 1, 0x9E37_79B9_7F4A_7C15);
    let _ = run(&mut polluted, &history);

    polluted.set_prewarm_on_reset(false);
    polluted
        .reset(48000, MAX_BLOCK)
        .expect("reset must not fail for a supported fixture");
    let silent = vec![0.0f32; PROBE_LEN];
    let after_reset = run(&mut polluted, &silent);

    // B: build cold (same sizing), probe silence directly.
    let mut fresh = load_skip_prewarm(path).ok_or(())?;
    let _ = fresh.set_max_buffer_size(MAX_BLOCK);
    let after_fresh = run(&mut fresh, &silent);

    assert_eq!(
        after_reset.len(),
        after_fresh.len(),
        "{label}: probe length mismatch"
    );
    for (i, (a, b)) in after_reset.iter().zip(after_fresh.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{label}: reset-without-prewarm is not history-independent at \
             sample {i}: reset={a} ({:#010x}) vs fresh={b} ({:#010x})",
            a.to_bits(),
            b.to_bits(),
        );
    }
    Ok(true)
}

/// Parametric reset matrix over every supported catalog identity.
///
/// Every supported, non-LSTM, *distributed* fixture must be present and must
/// run: a missing distributed fixture is a hard failure, and so is any build
/// or reset error on a present fixture. The only tolerated gaps are the two
/// explicit exclusions (LSTM resets under DEC-03; absent non-distributable
/// community fixtures), and the exercised count is asserted exactly.
#[test]
fn reset_clears_history_for_all_catalog_families() {
    let mut ran_required = 0usize;
    let mut present_distributed = 0usize;
    let mut lstm_runtime_excluded = 0usize;
    let mut exclusions: Vec<String> = Vec::new();
    let mut required_missing: Vec<String> = Vec::new();
    let mut unexpected_failures: Vec<String> = Vec::new();

    for entry in catalog_entries() {
        if entry.support != ModelSupportKind::Supported {
            continue;
        }
        let label = entry
            .canonical_path
            .rsplit('/')
            .next()
            .unwrap_or(entry.canonical_path);
        // Community/vendor models live under `third-party/` and are not
        // distributed; their absence is an explicit, justified exclusion.
        let distributed = !entry.canonical_path.starts_with("third-party/");

        // LSTM is excluded from the T3.1 matrix while DEC-03 stands: its reset
        // contract (and the Rust↔C++ divergence recorded under T3.2) is tracked
        // separately, so requiring it to be bit-fresh here would encode the
        // wrong oracle. Nested LSTM is caught at runtime via `is_lstm_family`.
        if entry.architecture.contains("LSTM") {
            exclusions.push(format!("{label}: LSTM reset out of scope (DEC-03)"));
            continue;
        }

        let path = fixture_path(entry.canonical_path);
        if !path.exists() {
            if distributed {
                required_missing.push(label.to_string());
            } else {
                exclusions.push(format!(
                    "{label}: non-distributable community fixture absent"
                ));
            }
            continue;
        }
        if distributed {
            present_distributed += 1;
        }

        match assert_reset_is_fresh(&path, label) {
            Ok(true) => {
                if distributed {
                    ran_required += 1;
                }
            }
            Ok(false) => {
                // LSTM nested inside a Container/Sequential: same DEC-03 carve-out.
                if distributed {
                    lstm_runtime_excluded += 1;
                }
                exclusions.push(format!("{label}: runtime LSTM family (DEC-03)"));
            }
            Err(()) => unexpected_failures.push(label.to_string()),
        }
    }

    assert!(
        required_missing.is_empty(),
        "distributed fixtures are missing: {required_missing:?} — a missing \
         required fixture is a failure"
    );
    assert!(
        unexpected_failures.is_empty(),
        "supported fixtures failed to build or reset: {unexpected_failures:?}"
    );
    assert!(
        ran_required + lstm_runtime_excluded == present_distributed,
        "reset matrix ran {ran_required} (+{lstm_runtime_excluded} LSTM-excluded) \
         of {present_distributed} present distributed supported fixtures — \
         unexpected drift in the catalog matrix"
    );
    assert!(
        ran_required >= MIN_EXPECTED_RUNS,
        "reset matrix ran {ran_required} fixtures; matrix is vacuous"
    );

    if !exclusions.is_empty() {
        eprintln!(
            "[STATUS] reset-matrix justified exclusions ({}): {exclusions:?}",
            exclusions.len()
        );
    }
}

/// ConvNet 1→1 reset regression through the `NamModel::reset` path.
///
/// A kernel-2, dilation-1 block with its only weight on the previous (delay-1)
/// tap and a positive bias. With `ReLU`, `relu(0 + bias) == bias`, so a reset
/// that truly clears history makes the follow-up `0` read exactly the bias.
/// The model is driven through `NamModel::reset` (not the raw block
/// `state.clear()`), matching the production reset contract exercised by the
/// matrix above.
#[test]
fn convnet_reset_clears_previous_tap() {
    use neural_amp_modeler_rs::loader::dispatcher::build_model;
    use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;

    let bias = 0.5f32;
    // Flat ConvNet 1→1, kernel 2 (C++ hardcodes kernel_size=2), no batchnorm.
    // Weight layout: conv_w[2] (k0=previous, k1=current) + conv_b + head_w[1]
    // + head_b. Only the previous tap passes through, the head is the identity.
    let json = format!(
        r#"{{
            "version": "0.5.4",
            "architecture": "ConvNet",
            "config": {{
                "channels": 1,
                "dilations": [1],
                "batchnorm": false,
                "activation": "ReLU"
            }},
            "weights": [1.0, 0.0, {bias}, 1.0, 0.0],
            "sample_rate": 48000
        }}"#
    );

    let data = parse_nam_json(&json).expect("parse ConvNet regression json");
    let mut model = build_model(&data).expect("ConvNet regression model builds");

    // Sanity: without a reset the delay-1 tap keeps the previous `1` alive.
    let mut live = build_model(&data).expect("build live model");
    let mut out = [0.0f32; 1];
    live.process(&[1.0], &mut out);
    live.process(&[0.0], &mut out);
    assert!(
        (out[0] - (1.0 + bias)).abs() < 1e-5,
        "stale previous tap should survive without reset, got {}",
        out[0]
    );

    // Through `NamModel::reset`: the delay line is emptied and the `0` reads
    // `0 + bias`.
    model.process(&[1.0], &mut out);
    model.set_prewarm_on_reset(false);
    model.reset(48000, 1).expect("ConvNet reset must succeed");
    model.process(&[0.0], &mut out);
    assert!(
        (out[0] - bias).abs() < 1e-6,
        "NamModel::reset must clear the previous tap: expected 0 + bias = {bias}, got {}",
        out[0]
    );
}
