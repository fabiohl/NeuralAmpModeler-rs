// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Live C++ v0.6.0 cross-validation of the `Sequential` architecture (NC-3.4).
//
//  ## What is validated
//  1. Whole-chain engine parity: Rust `SequentialModel` vs the C++ NAMcore
//     `render` tool (v0.6.0 vendor mirror) over the committed deterministic
//     fixtures — `sequential_linear2.nam` (Linear→Linear), `sequential_nested.nam`
//     (recursion), `sequential_multichannel.nam` (1→2→1 interior, NC-2), and
//     the WaveNet/Linear and double-LSTM chains of the NC-3.4 harness.
//  2. The f64 ideal composition: the chain is replayed through the double
//     precision per-child oracles (`oracle_forward`) with a single chain-wide
//     zero-feed of the Rust-model stabilization count, mirroring the engines'
//     single-prewarm operational semantics.
//  3. The single-prewarm invariant via the **initial transient** (first
//     [`TRANSIENT_FRAMES`] frames of the real signal), not just steady state:
//     the C++ `SequentialModel::Reset` (sequential.cpp L152-180) disables the
//     child prewarm flags, resets children, and runs exactly one chain-wide
//     `DSP::prewarm` pass; the Rust chain mirrors it. A regression that loses
//     the chain pass (no warm) must surface as a strongly different first
//     transient on stateful children (LSTM), which the positive control below
//     proves is discriminative.
//
//  ## Prewarm count arithmetic (why the transients are expected identical)
//  C++ counts: Linear = 0 (`dsp.h` default); WaveNet = 1 + Σ array receptive
//  fields (wavenet/model.cpp:656-660); LSTM = 0.5·declared rate
//  (lstm.cpp:127-133). The Rust chain mirror reproduces them (RF-sum for
//  WaveNet; declared-rate 0.5·rate for LSTM) and sums them exactly. C++
//  `DSP::prewarm` feeds whole 64-frame chunks (dsp.cpp:95-99), rounding the
//  total up; the Rust chain feeds the exact count (declared divergence of
//  `models/sequential.rs`). Either feed settled state is still the same
//  response stream:
//  - stateless children (Linear) — the zero response is the bias constant
//    (ring tails identical for any feeding depth ≥ RF);
//  - WaveNet — dilated tap lines keep the last RF fed values, likewise
//    identical for any depth ≥ RF;
//  - LSTM — zero-feed converges geometrically to the zero-input fixed point
//    (a 24000-sample warm leaves ≈ 0 residual between the engines' counts),
//    and `sequential_double_lstm.nam` additionally sums to 48000 = 750·64,
//    an exact multiple of the C++ chunk (no overshoot at all at 48 kHz).
//
//  ## Gates (floor calibration: `// Measured:` per test)
//  - Linear/WaveNet chains: ESR < 1e-12, SNR > 120 dB (task-proposed full
//    floors; measured margins recorded in the test bodies and in
//    docs/cpp_parity_map.md §6.6).
//  - LSTM chain: calibrated with margin over the LSTM C++ parity band
//    (tests/parity/cpp_parity.rs table: standalone LSTM SNR 50–97 dB).
//  - Transient window: max |diff| ≤ calibrated floor + windowed SNR gate.
//
//  ## Execution profiles
//  - All tests are non-ignored: they run in targeted invocations and in the
//    quick phase-2 measurement-oracle pass, compiling the C++ `render` tool
//    on demand (idempotent, cached in `build/namcore_render`).
//  - Multi-SR: all-unknown-rate fixtures sweep
//    44.1/48/88.2/96/192 kHz; the declared-rate double-LSTM fixture is
//    48k-only (the C++ render rejects other input rates by DEC-02 — covered
//    by an explicit rejection test).

use super::common;
use super::cpp_parity::{cpp_render_available, cpp_render_bin};
use common::*;

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::math::activations::ActivationPrecision;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::reference_oracle::{
    PrecisionConfig, oracle_forward, oracle_linear_multichannel,
};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ_WAV_SEQ: AtomicU64 = AtomicU64::new(0);

/// Frames of the real signal compared against the C++ output for the
/// single-prewarm transient gates (starting at the true signal onset).
const TRANSIENT_FRAMES: usize = 256;

/// Calibrated gate bundle for one fixture × sample rate.
///
/// `max_*` are upper bounds for ESR (energy ratio of noise to signal,
/// 0 = bit-exact); `min_*` lower bounds for SNR.
struct SequentialParityLimits {
    max_esr_cpp: f64,
    min_snr_cpp: f64,
    max_esr_oracle: f64,
    min_snr_oracle: f64,
    /// Max |Rust−C++| over the first [`TRANSIENT_FRAMES`] signal frames.
    transient_max_diff: f64,
    /// SNR floor over the same transient window.
    transient_min_snr: f64,
}

/// Mono Linear-chain floors: the chain kernel is the direct convolution in
/// both engines, so parity sits at the f32 bit-neighborhood.
///
/// Measured (NC-3.4 calibration run, release; see docs/cpp_parity_map.md §6.6):
/// - Rust vs C++ ESR ≤ 3.7e-15, SNR ≥ 144.3 dB (linear_chain/nested/1→2→1)
///   and ≤ 3.27e-16, SNR ≥ 154.9 dB (linear2 over the 5-rate sweep);
/// - Rust vs f64 ≤ 3.9e-15, SNR ≥ 144.0 dB;
/// - transient[256] max |diff| ≤ 1.49e-8, SNR ≥ 141.9 dB.
///
/// The floors below are the task proposal with ≥ 20 dB / ≥ 260× margins.
const MONO_LINEAR_LIMITS: SequentialParityLimits = SequentialParityLimits {
    max_esr_cpp: 1.0e-12,
    min_snr_cpp: 120.0,
    max_esr_oracle: 1.0e-12,
    min_snr_oracle: 120.0,
    transient_max_diff: 5.0e-7,
    transient_min_snr: 110.0,
};

/// Floors for chains with a WaveNet child (multi-rate sweep): the small
/// standard A1 WaveNet accumulates activation-kernel f32 noise, settled by
/// the chain warm (RF-count arithmetic in the module header).
///
/// Measured (NC-3.4 calibration run, release): Rust vs C++ ESR ≈ 3.46e-15
/// (all rates), SNR = 144.6 dB; Rust vs f64 ≈ 3.86e-15, 144.1 dB;
/// C++ vs f64 ≈ 1.42e-15, 148.5 dB; transient[256] max |diff| =
/// 2.328e-10, window SNR = inf (max_diff at the f32 ULP of the chain
/// constant). Windows keep ~2 decades of margin.
const MONO_WAVENET_LIMITS: SequentialParityLimits = SequentialParityLimits {
    max_esr_cpp: 1.0e-12,
    min_snr_cpp: 120.0,
    max_esr_oracle: 1.0e-12,
    min_snr_oracle: 120.0,
    transient_max_diff: 1.0e-8,
    transient_min_snr: 120.0,
};

/// Calibrated floors for the double-LSTM chain (recurrent accumulation):
/// the standalone-LSTM C++ parity band (cpp_parity table: 50–97 dB) uses
/// heavy trained state machines; the synthetic 1×8 chain carries much
/// lighter state noise.
///
/// Measured (NC-3.4 calibration run, release, v2 signal 5 s @ 48 kHz):
/// Rust vs C++ ESR = 1.023e-14, SNR = 139.9 dB; Rust vs f64 =
/// 1.022e-14, 139.9 dB; C++ vs f64 = 9.22e-16, 150.4 dB;
/// transient[256] max |diff| = 7.451e-9, SNR = 140.0 dB.
/// Positive control (no prewarm): transient max |diff| = 2.291e-3 — six
/// orders of magnitude above the warmed transient floor, so the
/// single-prewarm gate has detection power.
/// The floors stay at the task proposal (ESR < 1e-12 / SNR > 120 dB):
/// ≥ 98× / ≥ 20 dB margins.
const MONO_LSTM_CHAIN_LIMITS: SequentialParityLimits = SequentialParityLimits {
    max_esr_cpp: 1.0e-12,
    min_snr_cpp: 120.0,
    max_esr_oracle: 1.0e-12,
    min_snr_oracle: 120.0,
    transient_max_diff: 1.0e-7,
    transient_min_snr: 100.0,
};

/// Parses a committed fixture into the serde root document (the child JSON
/// trees feed the composed oracle).
fn read_fixture_json(fixture_filename: &str) -> serde_json::Value {
    let fixture_path = neural_amp_modeler_rs::testing::fixtures::model_path(fixture_filename);
    assert!(
        fixture_path.exists(),
        "committed fixture missing: {fixture_path:?} — run python3 tests/fixtures/\
         generate_namcore_v060_fixtures.py"
    );
    let raw = fs::read_to_string(&fixture_path).expect("read fixture");
    serde_json::from_str(&raw).expect("fixture JSON parse")
}

/// Double-precision composition oracle for a `Sequential` chain.
///
/// Mirrors the C++ `nam::get_dsp` recursion into `config.models`: each child
/// is re-parsed as a standalone envelope and nested `Sequential` children
/// recurse through the same branch (raw JSON keeps the tree); all other
/// architectures route through the shared oracle kernels —
/// [`oracle_linear_multichannel`] for `Linear` children (any channel
/// geometry, mono included), [`oracle_forward`] for the neural mono
/// children (WaveNet/LSTM/ConvNet). The signal travels as channel planes
/// (`Vec<Vec<f64>>`), matching the interior 1→N→1 plane layouts.
fn oracle_sequential_forward(
    root_models: &[serde_json::Value],
    input: &[Vec<f64>],
    cfg: &PrecisionConfig,
) -> Vec<Vec<f64>> {
    let mut signal = input.to_vec();
    for child in root_models {
        let text = serde_json::to_string(child).expect("child JSON render");
        let child_data = parse_nam_json(&text).expect("child parse");
        match child_data.architecture.as_str() {
            "Sequential" => {
                let grandchildren = child["config"]["models"].as_array().expect("models array");
                signal = oracle_sequential_forward(grandchildren, &signal, cfg);
            }
            "Linear" => signal = oracle_linear_multichannel(&child_data, &signal),
            _ => {
                // Neural children in the audited fixtures are mono; the
                // strict loader enforces their geometry at build time.
                assert_eq!(
                    signal.len(),
                    1,
                    "[oracle] neural child {:?} requires a mono input plane, got {} channels",
                    child_data.architecture,
                    signal.len()
                );
                signal = vec![oracle_forward(&child_data, &signal[0], cfg)];
            }
        }
    }
    signal
}

/// Composed-chain oracle with the single chain-wide prewarm replay:
/// `warm_zeros` zero frames stream before the real signal inside the
/// recursion — exactly like the engines' chain warm; the pre-signal slice
/// is then removed and only the real-signal segment is returned.
fn oracle_chain_with_warm(
    root_models: &[serde_json::Value],
    chain_prewarm_samples: usize,
    signal: &[f32],
    cfg: &PrecisionConfig,
) -> Vec<f32> {
    let mut stream = vec![vec![0.0f64; signal.len() + chain_prewarm_samples]];
    for (f, &x) in signal.iter().enumerate() {
        stream[0][chain_prewarm_samples + f] = x as f64;
    }
    let composed: Vec<Vec<f64>> = oracle_sequential_forward(root_models, &stream, cfg);
    assert_eq!(
        composed.len(),
        1,
        "oracle chain must exit mono for the audited fixtures, got {} channels",
        composed.len()
    );
    let plane = &composed[0];
    let start = chain_prewarm_samples.min(plane.len());
    plane[start..].iter().map(|&x| x as f32).collect()
}

/// Executes the C++ render tool for one fixture/input/output combination
/// and returns the channel-0 output samples.
fn run_cpp_render(
    fixture_path: &Path,
    input_path: &Path,
    output_path: &Path,
    label: &str,
) -> Vec<f32> {
    // Hard failure for committed v0.6.0 fixtures: the reference renderer must
    // load any chain that upstream accepts.
    let output = Command::new(cpp_render_bin())
        .arg(fixture_path.as_os_str())
        .arg(input_path.as_os_str())
        .arg(output_path.as_os_str())
        .output()
        .expect("execute NAMCore render tool");
    assert!(
        output.status.success(),
        "[{label}] C++ render failed (exit {:?}): {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let (cpp_output, _sr) = common::wav::read_wav_f32(output_path).expect("read C++ WAV");
    cpp_output
}

/// Probes whether the C++ render accepts a (fixture, signal rate) pair
/// (the DEC-02 declared-rate policy: `expectedRate > 0` with input
/// distance > 0.5 Hz ⇒ exit 1).
fn cpp_render_accepts(fixture_filename: &str, input_sr: u32) -> bool {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_dir = project_root.join("tests/fixtures/.temp_live");
    fs::create_dir_all(&temp_dir).ok();
    let model_path = neural_amp_modeler_rs::testing::fixtures::model_path(fixture_filename);
    let signal = generate_stress_signal_v1();
    let seq = SEQ_WAV_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let input_wav = temp_dir.join(format!("seq_rate_probe_in_{pid}_{seq}_{input_sr}.wav"));
    let output_wav = temp_dir.join(format!("seq_rate_probe_out_{pid}_{seq}.wav"));
    common::wav::write_wav_f32(&input_wav, &signal, input_sr).expect("write rate-probe WAV");
    let output = Command::new(cpp_render_bin())
        .arg(model_path.as_os_str())
        .arg(input_wav.as_os_str())
        .arg(output_wav.as_os_str())
        .output()
        .expect("execute render rate-probe");
    fs::remove_file(&input_wav).ok();
    fs::remove_file(&output_wav).ok();
    output.status.success()
}

/// Full mono-chain parity run: C++ render ↔ Rust engine ↔ f64 composition
/// oracle, plus fixed/irregular block-schedule invariance and the
/// single-prewarm transient gates.
///
/// `check_no_warm_positive_control` enables the positive control (see
/// module header) — meaningful only for stateful children (LSTM).
fn run_sequential_mono_parity(
    fixture_filename: &str,
    label: &str,
    sample_rate: u32,
    use_v2: bool,
    limits: &SequentialParityLimits,
    check_no_warm_positive_control: bool,
) {
    let model_status = FIXTURE_CATALOG.check(fixture_filename);
    assert!(
        model_status.is_available(),
        "[{label}] committed Sequential fixture missing: {fixture_filename}"
    );

    if !cpp_render_available() {
        if std::env::var("NAM_REQUIRE_CPP_ORACLE").as_deref() == Ok("1") {
            panic!("NAM_REQUIRE_CPP_ORACLE=1 — aborting test: {label} (render tool unavailable)");
        }
        eprintln!("[STATUS] SKIP_CAPABILITY reason=\"render_tool_unavailable\" ({label})");
        return;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_dir = project_root.join("tests/fixtures/.temp_live");
    fs::create_dir_all(&temp_dir).ok();

    let model_path = neural_amp_modeler_rs::testing::fixtures::model_path(fixture_filename);
    let json_raw = fs::read_to_string(&model_path).expect("read fixture");
    let model_data = parse_nam_json(&json_raw).expect("parse fixture JSON");

    // Sample-rate resolution shared with `run_render_comparison`: the
    // declared chain rate when present, else the caller WAV rate.
    let model_sr = model_data.sample_rate.unwrap_or(sample_rate as f32) as u32;
    assert_eq!(
        model_sr, sample_rate,
        "[{label}] sweep requires the chain to accept {sample_rate} Hz (declared {model_sr}); \
         the C++ render rejects mismatches (DEC-02)"
    );

    let signal = if use_v2 {
        generate_stress_signal_v2(sample_rate)
    } else {
        generate_stress_signal_v1()
    };
    let frames_tot = signal.len();

    let seq = SEQ_WAV_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let input_wav = temp_dir.join(format!("seq_in_{pid}_{seq}_{sample_rate}.wav"));
    let output_wav = temp_dir.join(format!("seq_out_cpp_{pid}_{seq}_{sample_rate}.wav"));
    common::wav::write_wav_f32(&input_wav, &signal, sample_rate).expect("write stress WAV");

    let cpp_output = run_cpp_render(&model_path, &input_wav, &output_wav, label);
    assert_eq!(
        cpp_output.len(),
        frames_tot,
        "[{label}] C++ rendered length mismatch"
    );
    assert!(
        cpp_output.iter().all(|x| x.is_finite()),
        "[{label}] C++ output is not finite"
    );

    // ── Rust engine chain ────────────────────────────────────────────────
    // Standard activations: both engines run the native-grade math kernels
    // (the C++ reference has no fast-math build here).
    let _precision = PrecisionGuard::new(ActivationPrecision::Standard);
    let mut model = build_model(&model_data).expect("Rust chain build failed");
    assert_eq!(
        model.class_label().split_whitespace().next(),
        Some("Sequential"),
        "[{label}] Rust must classify the fixture as Sequential"
    );
    assert_eq!(model.in_channels(), 1, "[{label}] chain input channels");
    assert_eq!(
        model.num_output_channels(),
        1,
        "[{label}] chain output channels"
    );

    // Stabilization count captured from the built chain (reset-preserving;
    // mirrors the C++ sum of children in `GetPrewarmSamples`).
    let chain_prewarm_samples = model.prewarm_samples();

    // Fixed-block schedule (mirrors C++ `render.cpp:147-197`, 64-frame wheel)
    model.reset(sample_rate, 64).expect("chain reset");
    let mut rust_out_blocks = vec![0.0f32; frames_tot];
    process_in_blocks(&mut model, &signal, &mut rust_out_blocks, 64);

    // Irregular-block schedule (1..=64): chain invariance under arbitrary
    // caller chunking, bit-equal with the fixed schedule.
    model
        .reset(sample_rate, 64)
        .unwrap_or_else(|e| panic!("[{label}] chain reset (irregular schedule) failed: {e}"));
    let mut rust_out_irregular = vec![0.0f32; frames_tot];
    let mut pos = 0;
    while pos < frames_tot {
        let take = ((pos % 64) + 1).min(frames_tot - pos);
        model.process(
            &signal[pos..pos + take],
            &mut rust_out_irregular[pos..pos + take],
        );
        pos += take;
    }
    assert!(
        rust_out_blocks == rust_out_irregular,
        "[{label}] Rust chain is chunk-schedule dependent (must be bit-invariant)"
    );

    // ── f64 composition oracle (single chain-wide warm replay) ──────────
    let root = read_fixture_json(fixture_filename);
    let children = root["config"]["models"].as_array().expect("models array");
    let cfg = PrecisionConfig::default();
    // Detail: the planar management above mirrors the reference C++ plane
    // tables kept in `oracle_linear_multichannel` (kernel rows per
    // out-channel).
    let oracle_f32 = oracle_chain_with_warm(children, chain_prewarm_samples, &signal, &cfg);

    // ── Metrics ──────────────────────────────────────────────────────────
    let esr_rust_cpp = common::metrics::compute_esr(&cpp_output, &rust_out_blocks);
    let snr_rust_cpp =
        neural_amp_modeler_rs::testing::perceptual::compute_snr_db(&cpp_output, &rust_out_blocks);
    let esr_rust_oracle = common::metrics::compute_esr(&oracle_f32, &rust_out_blocks);
    let snr_rust_oracle =
        neural_amp_modeler_rs::testing::perceptual::compute_snr_db(&oracle_f32, &rust_out_blocks);
    let esr_cpp_oracle = common::metrics::compute_esr(&oracle_f32, &cpp_output);
    let snr_cpp_oracle =
        neural_amp_modeler_rs::testing::perceptual::compute_snr_db(&oracle_f32, &cpp_output);

    // Transient window (post-prewarm onset, first real frames)
    let win = TRANSIENT_FRAMES.min(frames_tot);
    let max_diff_transient = rust_out_blocks[..win]
        .iter()
        .zip(cpp_output[..win].iter())
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);
    let snr_transient = neural_amp_modeler_rs::testing::perceptual::compute_snr_db(
        &cpp_output[..win],
        &rust_out_blocks[..win],
    );

    println!(
        "[{label} @ {sample_rate} Hz] Rust vs C++: ESR={esr_rust_cpp:.3e}, SNR={snr_rust_cpp:.1} dB | \
         Rust vs f64: ESR={esr_rust_oracle:.3e}, SNR={snr_rust_oracle:.1} dB | \
         C++ vs f64: ESR={esr_cpp_oracle:.3e}, SNR={snr_cpp_oracle:.1} dB | \
         transient[{win}]: max_diff={max_diff_transient:.3e}, SNR={snr_transient:.1} dB | \
         prewarm={chain_prewarm_samples}"
    );

    // Whole-signal gates
    assert!(
        esr_rust_cpp < limits.max_esr_cpp,
        "[{label}] Rust vs C++ ESR {esr_rust_cpp:.3e} ≥ floor {:e}",
        limits.max_esr_cpp
    );
    assert!(
        snr_rust_cpp > limits.min_snr_cpp,
        "[{label}] Rust vs C++ SNR {snr_rust_cpp:.1} dB ≤ floor {} dB",
        limits.min_snr_cpp
    );
    assert!(
        esr_rust_oracle < limits.max_esr_oracle,
        "[{label}] Rust vs f64 oracle ESR {esr_rust_oracle:.3e} ≥ floor {:e}",
        limits.max_esr_oracle
    );
    assert!(
        snr_rust_oracle > limits.min_snr_oracle,
        "[{label}] Rust vs f64 oracle SNR {snr_rust_oracle:.1} dB ≤ floor {} dB",
        limits.min_snr_oracle
    );
    // The C++ reference must also agree with the ideal composition; those
    // results show where a defect (if any) lives.
    assert!(
        esr_cpp_oracle < limits.max_esr_oracle,
        "[{label}] C++ vs f64 oracle ESR {esr_cpp_oracle:.3e} ≥ floor {:e}",
        limits.max_esr_oracle
    );

    // Single-prewarm transient gates
    assert!(
        max_diff_transient <= limits.transient_max_diff,
        "[{label}] initial transient differs from C++ (max |diff| {max_diff_transient:.3e} \
         > {:e} over the first {win} frames) — chain re-warm semantics regressed",
        limits.transient_max_diff
    );
    assert!(
        snr_transient > limits.transient_min_snr,
        "[{label}] transient-window SNR {snr_transient:.1} dB ≤ floor {} dB",
        limits.transient_min_snr
    );

    // Positive control: no-warm detection power. Losing the chain warm (child
    // flags not cleared and no chain pass) must be visible in the transient
    // of a stateful chain; for stateless chains (Linear-only) warm is
    // semantically inert, so this control is opt-in per fixture.
    if check_no_warm_positive_control {
        let mut nowarm = build_model(&model_data).expect("Rust chain build (no-warm control)");
        nowarm.set_prewarm_on_reset(false);
        nowarm.reset(sample_rate, 64).expect("no-warm chain reset");
        let mut rust_out_nowarm = vec![0.0f32; frames_tot];
        process_in_blocks(&mut nowarm, &signal, &mut rust_out_nowarm, 64);
        let max_diff_nowarm = rust_out_nowarm[..win]
            .iter()
            .zip(cpp_output[..win].iter())
            .map(|(&a, &b)| (a - b).abs() as f64)
            .fold(0.0f64, f64::max);
        println!(
            "[{label} @ {sample_rate} Hz] positive control (no prewarm): \
             transient max |diff| vs C++ = {max_diff_nowarm:.3e}"
        );
        assert!(
            max_diff_nowarm > limits.transient_max_diff,
            "[{label}] positive control is not discriminative: an un-warmed chain \
             (max_diff={max_diff_nowarm:.3e}) does not exceed the warm transient floor {:e} — \
             the single-prewarm gate cannot catch the regression it claims to",
            limits.transient_max_diff
        );
    }

    fs::remove_file(&input_wav).ok();
    fs::remove_file(&output_wav).ok();
}

// =============================================================================
// Linear-only chains — f32-neighborhood parity, 44.1/48/88.2/96/192 kHz sweeps
// =============================================================================

#[test]
fn test_sequential_linear2_multi_sr_cpp_parity() {
    // All-unknown rate chain (DEC-01): accepted by the C++ render at every
    // WAV input rate; windowed at the f32 kernel noise floor.
    //
    // Measured (release, v2 signal 5 s @ each rate) — Rust vs C++:
    //   44.1k ESR=3.235e-16, SNR=154.9 | 48k ESR=3.188e-16, SNR=155.0 |
    //   88.2k ESR=3.249e-16, SNR=154.9 | 96k ESR=3.267e-16, SNR=154.9 |
    //   192k ESR=3.217e-16, SNR=154.9 | Rust vs f64 ≤ 4.53e-16 (153.4 dB) |
    //   C++ vs f64 ≤ 3.94e-16 (154.1 dB) | transient max_diff ≤ 7.451e-9.
    for &sr in &[44_100, 48_000, 88_200, 96_000, 192_000] {
        run_sequential_mono_parity(
            "sequential_linear2.nam",
            "Sequential Linear2",
            sr,
            true,
            &MONO_LINEAR_LIMITS,
            false,
        );
    }
}

#[test]
fn test_sequential_linear_chain_48k_cpp_parity() {
    // Committed 48k-declared two-Linear chain (NC-3.1 reference fixture).
    //
    // Measured (release, v2 signal 5s): Rust vs C++ ESR=1.572e-15,
    // SNR=148.0 dB | Rust vs f64 ESR=1.997e-15, SNR=147.0 dB |
    // C++ vs f64 ESR=1.470e-15, SNR=148.3 dB | transient max_diff=3.725e-9
    // (SNR 149.8 dB).
    run_sequential_mono_parity(
        "sequential_linear_chain.nam",
        "Sequential Linear Chain",
        48_000,
        false,
        &MONO_LINEAR_LIMITS,
        false,
    );
}

#[test]
fn test_sequential_nested_48k_cpp_parity() {
    // Recursive composition: root[Linear, Sequential[Linear]] — validates the
    // C++ `get_dsp` recursion and the Rust chain-of-chain equivalence.
    //
    // Measured (release, v2 signal 5s): Rust vs C++ ESR=1.572e-15,
    // SNR=148.0 dB | Rust vs f64 ESR=1.997e-15, SNR=147.0 dB |
    // C++ vs f64 ESR=1.470e-15, SNR=148.3 dB | transient max_diff=3.725e-9
    // (SNR 149.8 dB).
    run_sequential_mono_parity(
        "sequential_nested.nam",
        "Sequential Nested",
        48_000,
        false,
        &MONO_LINEAR_LIMITS,
        false,
    );
}

// =============================================================================
// 1→2→1 interior multichannel chain (NC-2 geometries inside Sequential)
// =============================================================================

#[test]
fn test_sequential_multichannel_interior_48k_cpp_parity() {
    // Exterior is mono (1 in → 1 out); the interior crosses a 2-channel 1→2
    // boundary and recombines 2→1: both engines' interior channel-pointer
    // tables and boundary planes process every frame; channel-0 parity is a
    // witness of the full interior composition.
    //
    // Measured (release, v2 signal 5s): Rust vs C++ ESR=3.749e-15,
    // SNR=144.3 dB | Rust vs f64 ESR=3.945e-15, SNR=144.0 dB |
    // C++ vs f64 ESR=3.480e-15, SNR=144.6 dB | transient
    // max_diff=1.490e-8 (SNR 141.9 dB).
    run_sequential_mono_parity(
        "sequential_multichannel.nam",
        "Sequential 1→2→1",
        48_000,
        false,
        &MONO_LINEAR_LIMITS,
        false,
    );
}

// =============================================================================
// WaveNet child chain — multi-rate sweep with a stateful neural child
// =============================================================================

#[test]
fn test_sequential_linear_wavenet_multi_sr_cpp_parity() {
    // The WaveNet child contributes 1 + ΣRF stabilization frames (count
    // arithmetic in the module header); the chain warm replays it exactly.
    //
    // Measured (release, v2 signal 5 s @ each rate) — Rust vs C++:
    //   ESR ≈ 3.46e-15, SNR = 144.6 dB (all rates: 44.1/48/88.2/96/192 kHz);
    //   Rust vs f64 ≤ 3.88e-15 (144.1 dB); C++ vs f64 ≤ 1.43e-15 (148.5 dB);
    //   transient max_diff = 2.328e-10, window SNR = inf; prewarm=6.
    for &sr in &[44_100, 48_000, 88_200, 96_000, 192_000] {
        run_sequential_mono_parity(
            "sequential_linear_wavenet.nam",
            "Sequential Linear→WaveNet",
            sr,
            true,
            &MONO_WAVENET_LIMITS,
            false,
        );
    }
}

// =============================================================================
// Double-LSTM chain — stateful children; single-prewarm transient proof
// =============================================================================

#[test]
fn test_sequential_double_lstm_single_prewarm_cpp_parity() {
    // Declared 48 kHz chain (SR48kOnly live scope): the transient identity
    // pins the C++ Reset semantics — child flags disabled, ONE chain-wide
    // zero-feed of 48000 (= 750·64) frames. The positive control proves the
    // gate has detection power (an un-warmed chain diverges hard).
    //
    // Measured (release, v2 signal 5 s @ 48 kHz): Rust vs C++
    // ESR=1.023e-14, SNR=139.9 dB | Rust vs f64 ESR=1.022e-14, SNR=139.9 dB |
    // C++ vs f64 ESR=9.217e-16, SNR=150.4 dB | transient max_diff=7.451e-9
    // (SNR 140.0 dB) | prewarm=48000 | positive control (no prewarm)
    // transient max |diff| = 2.291e-3 (≈ 3e5× the warmed floor).
    run_sequential_mono_parity(
        "sequential_double_lstm.nam",
        "Sequential double LSTM 1×8",
        48_000,
        true,
        &MONO_LSTM_CHAIN_LIMITS,
        true,
    );
}

#[test]
fn test_sequential_double_lstm_cross_rate_rejection() {
    // DEC-02 mirror (NC-5.3 pattern): the declared-rate chain leaves the C++
    // render exit 1 for a non-48k input WAV, while the Rust engine accepts
    // the reconfiguration and runs clean at the operational rate (48k wired
    // behavior recorded in docs/cpp_parity_map.md §6.6 and NC-5.3).
    if !cpp_render_available() {
        if std::env::var("NAM_REQUIRE_CPP_ORACLE").as_deref() == Ok("1") {
            panic!("NAM_REQUIRE_CPP_ORACLE=1 — aborting test: double-LSTM cross-rate probe");
        }
        eprintln!("[STATUS] SKIP_CAPABILITY reason=\"render_tool_unavailable\"");
        return;
    }
    assert!(
        !cpp_render_accepts("sequential_double_lstm.nam", 44_100),
        "C++ render must reject the declared-48k chain at 44.1 kHz input (DEC-02)"
    );
    assert!(
        cpp_render_accepts("sequential_double_lstm.nam", 48_000),
        "C++ render must accept the declared-48k chain at 48 kHz"
    );

    // Rust side: accepts and processes cleanly at the operational rate.
    let _precision = PrecisionGuard::new(ActivationPrecision::Standard);
    let model_data = {
        let json_raw = fs::read_to_string(neural_amp_modeler_rs::testing::fixtures::model_path(
            "sequential_double_lstm.nam",
        ))
        .expect("read fixture");
        parse_nam_json(&json_raw).expect("parse fixture")
    };
    let mut model = build_model(&model_data).expect("Rust chain build");
    model.reset(44_100, 64).expect("operational-rate reset");
    let mut out = vec![0.0f32; 256];
    let input = vec![0.35f32; 256];
    let mut pos = 0;
    while pos < out.len() {
        let take = 64.min(out.len() - pos);
        model.process(&input[pos..pos + take], &mut out[pos..pos + take]);
        pos += take;
    }
    assert!(
        out.iter().all(|s| s.is_finite()),
        "operational-rate chain run must stay finite"
    );
}
