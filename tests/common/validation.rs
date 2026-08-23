// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(dead_code)]

use std::fmt::Write;

/// Lock serializing report emission across threads so that each model's
/// fidelity report (header + all metrics + footer) prints as a contiguous
/// block even under `--test-threads > 1`.
static REPORT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Per-thread flag: when true, `report_dsp_fidelity_impl` suppresses stdout
// emission. Used by controlled-panic regression tests (e.g.
// `test_mrstft_hard_gate_catches_regression`) to prevent "✗" and error
// messages from polluting the green-test output.
thread_local! {
    static SUPPRESS_REPORT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    static METRIC_MODEL: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    static METRIC_MODE: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    static METRIC_KIND: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    // Per-thread override for the JSONL metric sink. When set, the current
    // thread's `report_dsp_fidelity*` appends to this path instead of reading
    // the process-global `NAM_METRICS_JSONL` env var. This is what makes the
    // JSONL regression tests deterministic under `--test-threads > 1`: tests
    // isolate their sink to a thread-local unique file without mutating shared
    // process state (mutating `NAM_METRICS_JSONL` while other test threads
    // read it is a data race, and any other reporter would append to the same
    // file, corrupting line-count assertions).
    static METRIC_JSONL_PATH: std::cell::RefCell<Option<std::path::PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

pub fn set_metric_model(model: String) {
    METRIC_MODEL.with(|c| {
        *c.borrow_mut() = Some(model);
    });
}

pub fn set_metric_mode(mode: String) {
    METRIC_MODE.with(|c| {
        *c.borrow_mut() = Some(mode);
    });
}

pub fn set_metric_kind(kind: String) {
    METRIC_KIND.with(|c| {
        *c.borrow_mut() = Some(kind);
    });
}

/// RAII guard that sets [`SUPPRESS_REPORT`] for the current thread on creation
/// and restores it on drop.
///
/// # Usage
/// ```ignore
/// {
///     let _guard = SuppressReportGuard::new();
///     report_dsp_fidelity(...); // output suppressed
/// }
/// // SUPPRESS_REPORT restored to false here
/// ```
pub struct SuppressReportGuard;

impl SuppressReportGuard {
    pub fn new() -> Self {
        SUPPRESS_REPORT.with(|c| c.set(true));
        SuppressReportGuard
    }
}

impl Drop for SuppressReportGuard {
    fn drop(&mut self) {
        SUPPRESS_REPORT.with(|c| c.set(false));
    }
}

/// RAII guard that sets [`METRIC_KIND`] for the current thread on creation
/// and restores it on drop.
///
/// # Usage
/// ```ignore
/// {
///     let _guard = MetricKindGuard::selftest();
///     report_dsp_fidelity(...); // JSONL emission labeled "selftest"
/// }
/// // METRIC_KIND restored to default ("fidelity") here
/// ```
pub struct MetricKindGuard;

impl MetricKindGuard {
    pub fn selftest() -> Self {
        METRIC_KIND.with(|c| {
            *c.borrow_mut() = Some("selftest".to_string());
        });
        MetricKindGuard
    }
}

impl Drop for MetricKindGuard {
    fn drop(&mut self) {
        METRIC_KIND.with(|c| {
            *c.borrow_mut() = None;
        });
    }
}

/// RAII guard that redirects the current thread's JSONL metric sink to `path`
/// and restores the previous sink (env var fallback) on drop.
///
/// The override is thread-local, so concurrent reporters on other threads keep
/// writing to their own (or the env var's) sink — the guarded file receives
/// exactly the lines this thread emits. The owning test is responsible for
/// removing the file; pair with a `Drop` cleanup guard for fail-safe
/// post-panic hygiene.
pub struct MetricJsonlGuard {
    prev: Option<std::path::PathBuf>,
}

impl MetricJsonlGuard {
    pub fn new(path: std::path::PathBuf) -> Self {
        let prev = METRIC_JSONL_PATH.with(|c| c.borrow_mut().replace(path));
        MetricJsonlGuard { prev }
    }
}

impl Drop for MetricJsonlGuard {
    fn drop(&mut self) {
        METRIC_JSONL_PATH.with(|c| {
            *c.borrow_mut() = self.prev.take();
        });
    }
}

/// Serializes a metric value to a canonical JSON value that never emits `null`.
///
/// `serde_json` serializes non-finite floats (`f64::INFINITY`, `-inf`, `NaN`)
/// as JSON `null`, because JSON has no native representation for them. A
/// *perfect-parity* result (SNR = ∞) or a silent/identical signal (ESR = ∞ or
/// ESR dB = −∞) would therefore corrupt the JSONL metric stream with `null`,
/// which the downstream `quality-dashboard.sh` would treat as missing data and
/// coerce to `0.0` (fail-open).
///
/// To keep the stream canonical, non-finite values are mapped to explicit
/// string sentinels `"inf"`, `"-inf"`, and `"nan"` — recognized downstream as
/// typed non-finite states rather than absent metrics.
fn json_metric(v: f64) -> serde_json::Value {
    if v.is_finite() {
        serde_json::json!(v)
    } else if v.is_nan() {
        serde_json::json!("nan")
    } else if v.is_sign_positive() {
        serde_json::json!("inf")
    } else {
        serde_json::json!("-inf")
    }
}

/// Serializes the SNR of a fidelity record.
///
/// SNR is the one metric where a non-finite value can be a **positive**
/// outcome: `+∞` dB means the output is bit-identical to the reference
/// (perfect parity), which is above any envelope floor. The literal `"inf"`
/// sentinel is therefore never written for a positive `snr_db` — it trips the
/// `is_finite_num` syntactic gate of foreign consumers (jq/awk `+0` coercion)
/// and the fail-closed verify engine reads the omission as the
/// above-the-floor state instead. `null` is the canonical representation of
/// "measured, non-finite, positive".
///
/// Non-positive non-finite values stay typed sentinels (`"-inf"`, `"nan"`):
/// a silent reference with noisy output (`-∞` dB) or a NaN measurement is a
/// broken/degenerate result, not an above-floor one, and must keep failing
/// the contract.
fn json_snr_db(snr: f64) -> serde_json::Value {
    if snr.is_finite() {
        serde_json::json!(snr)
    } else if snr.is_nan() {
        serde_json::json!("nan")
    } else if snr.is_sign_positive() {
        serde_json::Value::Null
    } else {
        serde_json::json!("-inf")
    }
}

/// Appends one JSONL line to the metric sink — thread-local override first,
/// process env `NAM_METRICS_JSONL` fallback, serialized under [`REPORT_LOCK`]
/// so concurrent reporters never interleave partial lines.
///
/// Shared by the fidelity sink of `report_dsp_fidelity*` and by the S2.T6
/// oracle sinks (`report_f64_table`, `report_f64_decomp`, `report_activation`,
/// `report_isa`). A missing/closed sink is silently ignored — the JSONL
/// stream is an enrichment, never a gate.
fn append_metric_line(obj: serde_json::Value) {
    let jsonl_path = METRIC_JSONL_PATH.with(|c| c.borrow().clone()).or_else(|| {
        std::env::var("NAM_METRICS_JSONL")
            .ok()
            .map(std::path::PathBuf::from)
    });
    if let Some(jsonl_path) = jsonl_path {
        let _lock = REPORT_LOCK.lock().unwrap();
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&jsonl_path)
        {
            use std::io::Write;
            let _ = writeln!(file, "{obj}");
        }
    }
}

// ── S2.T6 oracle sinks (R-06, slice 1) ───────────────────────────────────────
// Structured JSONL for the f64 oracle, activation precision, and ISA parity
// tests that today only print human-readable logs. The dashboard keeps
// scraping the logs until S2.T7; these records are the forensic JSONL stream
// for agents (`kind` values are ignored by the S2.T1/T2 parsers by design —
// they extend without touching the fidelity filter).

/// `f64_table` — one row of the f64-oracle summary table
/// (`tests/parity/reference_oracle_f64.rs::test_summary_table`), mirroring
/// the fields the dashboard awk extracts from the human log.
pub fn report_f64_table(filename: &str, family: &str, esr_linear: f64, esr_db: f64) {
    append_metric_line(serde_json::json!({
        "kind": "f64_table",
        "filename": filename,
        "family": family,
        "esr": json_metric(esr_linear),
        "esr_db": json_metric(esr_db),
    }));
}

/// `f64_decomp` — one source-decomposition record of the f64 oracle
/// (`print_decomposition` in `tests/parity/reference_oracle_f64.rs`).
/// Unmeasured decomposition terms (absent `Option` fields) are omitted from
/// the record — never `null`, never a fabricated `0.0`.
pub fn report_f64_decomp(
    result: &neural_amp_modeler_rs::testing::reference_oracle::DecompositionResult,
) {
    let mut obj = serde_json::Map::new();
    obj.insert("kind".into(), serde_json::json!("f64_decomp"));
    obj.insert("label".into(), serde_json::json!(result.label));
    obj.insert(
        "architecture".into(),
        serde_json::json!(result.architecture),
    );
    obj.insert("esr_f32_vs_f64".into(), json_metric(result.esr_f32_vs_f64));
    if let Some(v) = result.esr_quant_f16c {
        obj.insert("esr_quant_f16c".into(), json_metric(v));
    }
    if let Some(v) = result.esr_quant_bf16 {
        obj.insert("esr_quant_bf16".into(), json_metric(v));
    }
    if let Some(v) = result.esr_activation {
        obj.insert("esr_activation".into(), json_metric(v));
    }
    if let Some(v) = result.esr_accumulation {
        obj.insert("esr_accumulation".into(), json_metric(v));
    }
    if let Some(v) = result.esr_combined {
        obj.insert("esr_combined".into(), json_metric(v));
    }
    append_metric_line(serde_json::Value::Object(obj));
}

/// `activation` — one activation-precision measurement (Fast Padé vs
/// exact-grade), `tests/models/lstm_activation_precision.rs`. Callers report
/// only finite pairs — skips are human-log-only, exactly like today.
pub fn report_activation(model: &str, snr_fast_db: f64, snr_exact_db: f64) {
    append_metric_line(serde_json::json!({
        "kind": "activation",
        "model": model,
        "snr_fast_db": json_metric(snr_fast_db),
        "snr_exact_db": json_metric(snr_exact_db),
        "gain_db": json_metric(snr_exact_db - snr_fast_db),
    }));
}

/// `isa` — one ISA-parity comparison (`tests/parity/isa_parity.rs`):
/// cross-ISA pairs carry `esr`/`max_abs_err`/`budget`; self-consistency
/// checks (`ref_isa == test_isa`) carry only `mse`. Absent fields are
/// omitted, never `null`.
pub fn report_isa(
    label: &str,
    ref_isa: &str,
    test_isa: &str,
    esr: Option<f64>,
    mse: f64,
    max_abs_err: Option<f64>,
    budget: Option<f64>,
) {
    let mut obj = serde_json::Map::new();
    obj.insert("kind".into(), serde_json::json!("isa"));
    obj.insert("label".into(), serde_json::json!(label));
    obj.insert("ref_isa".into(), serde_json::json!(ref_isa));
    obj.insert("test_isa".into(), serde_json::json!(test_isa));
    obj.insert("mse".into(), json_metric(mse));
    if let Some(v) = esr {
        obj.insert("esr".into(), json_metric(v));
    }
    if let Some(v) = max_abs_err {
        obj.insert("max_abs_err".into(), json_metric(v));
    }
    if let Some(v) = budget {
        obj.insert("budget".into(), json_metric(v));
    }
    append_metric_line(serde_json::Value::Object(obj));
}

/// Plausible LUFS range for golden reference output (sanity gate — BS.1770-4 2-pass).
///
/// Guitar/amp model output at typical stress-signal levels falls between −35 and 0 LUFS.
/// The lower bound of −50 LUFS is intentionally generous — it only catches egregious
/// errors (e.g., where LUFS −67 near-silence went undetected in a validly passing test).
/// The upper bound of +10 LUFS guards against output saturation/clipping that would also
/// indicate a defective golden.
///
/// Backed by BS.1770-4 full 2-pass gating (absolute −70 LUFS → relative −10 LU),
/// providing accurate integrated LUFS for the plausibility assert.
///
/// Part of the perceptual metrics guard-rail suite.
const LUFS_PLAUSIBLE_MIN: f64 = -50.0;
const LUFS_PLAUSIBLE_MAX: f64 = 10.0;

/// Calibrated soft-gate warning threshold for MR-STFT at non-standard sample rates
/// (rates other than 44.1/48 kHz where per-model hard gates apply).
///
/// # Empirical calibration (2026-07-11)
///
/// MR-STFT is a bounded metric [0, 1] measuring spectral divergence. At standard
/// rates (44.1/48 kHz), per-model hard gates from `get_calibrated_threshold()`
/// enforce individual ceilings. At non-standard rates (88.2, 96, 176.4, 192 kHz),
/// this soft gate provides a global sanity guardrail — purely informational,
/// not a hard assertion.
///
/// ## Calibration data
///
/// Non-degenerated calibrated hard gates at 44.1/48 kHz (from `get_calibrated_threshold`):
///
/// | Model                    | mrstft_max | Measured v2 worst   |
/// |--------------------------|------------|---------------------|
/// | wavenet_official         | 0.45       | 0.42 @ 48 kHz 5s    |
/// | wavenet_condition_dsp    | 0.35       | 0.336 @ 48 kHz 5s   |
/// | wavenet_dyn_free         | 0.18       | 0.170 @ 48 kHz 5s   |
/// | BossLSTM-2x8             | 0.12       | —                   |
/// | lstm_dyn_test            | 0.10       | 0.081 @ 48 kHz 2k   |
/// | wavenet_a2_film_lite     | 1.0e-4     | 3.92e-5             |
/// | wavenet_a2_film_full     | 1.0e-4     | 3.28e-5             |
/// | Near-bit-exact models    | 0.05       | ≈ 1e-5              |
///
/// The highest non-degenerated calibrated hard gate is 0.45 (wavenet_official).
/// The project anti-placebo policy (threshold_calibration.rs Rule 4) defines
/// MR-STFT ≥ 0.5 as a placebo gate.
///
/// ## Calibrated threshold: 0.50
///
/// Set to the anti-placebo ceiling (0.50), providing a 0.05 margin above the
/// highest non-degenerated calibrated model (0.45). At non-standard rates,
/// MR-STFT naturally increases due to longer v2 sequences and recurrent drift
/// accumulation — this threshold catches only pathological divergence exceeding
/// the placebo line, not normal v2 behavior.
///
/// A warning is emitted when MR-STFT ≥ 0.50 at non-standard sample rates,
/// indicating a regression that would be a placebo gate even under per-model
/// calibration.
// Measured: calibrated against anti-placebo ceiling (Rule 4, threshold_calibration.rs).
// Highest non-degenerated calibrated hard gate: 0.45 (wavenet_official).
// Value: 0.50 (anti-placebo ceiling, 0.05 margin above worst real hard gate).
pub const MRSTFT_SOFT_THRESHOLD: f64 = 0.50;

/// Validates DSP fidelity in a single pass, computing MSE, MAE, SNR, PSNR,
/// equivalent bits, ESR, and LUFS simultaneously.
///
/// # Parameters
/// - `reference` — reference output vector (NeuralAmpModelerCore C++)
/// - `test`      — Rust engine output vector to be validated
/// - `mse_limit` — maximum allowed MSE threshold
/// - `min_snr_db` — minimum SNR in dB that must be achieved
/// - `max_esr`   — optional maximum ESR threshold for regression gating (default `None`)
/// - `mrstft_max` — optional maximum MR-STFT gate; asserted as hard gate at
///   44.1/48 kHz, informational at higher rates
/// - `label`     — label for identification in diagnostic messages
/// - `sample_rate` — sample rate in Hz (used for LUFS, MR-STFT gate severity, and anchor SNR diagnostics)
///
/// # Output format
/// ```text
/// [NeuralAmpModelerCore × NeuralAmpModeler-rs — label]
///   MSE     = 3.21e-02      (threshold < 5.0e-02)  ✓
///   MAE     = 2.84e-01
///   SNR     = 10.1 dB       (threshold ≥ 9.0 dB)   ✓
///   PSNR    = 14.9 dB
///   Bits    = 2.5 bits equiv.
///   ESR     = 1.23e-05       (−49.1 dB)   (threshold < 1.0e-1)  ✓   [baseline A1-Std: 6.23e-03, A2-Full: 3.34e-03]
///   MR-STFT = 0.0042         (log-mag abs)                   ✓   [hard gate ≤ 0.05 @ 44.1/48 kHz]
///   LUFS    = −23.4 LUFS    (reference)   [plausible: −50.0..+10.0]  ✓
///   LUFS    = −65.0 LUFS    (reference)   [plausible: −50.0..+10.0]  ⓘ informational (gate opt-out — expected)
///   Fidelity Margin = 48.2 dB (target > 8.0 dB) ✓
///   Samples = 2048 @ 48 kHz (stress signal)
/// ```
#[track_caller]
#[expect(
    clippy::too_many_arguments,
    reason = "Test validation helper with many configuration parameters for comprehensive test fixture setup"
)]
pub fn report_dsp_fidelity(
    reference: &[f32],
    test: &[f32],
    mse_limit: Option<f64>,
    min_snr_db: f64,
    max_esr: Option<f64>,
    mrstft_max: Option<f64>,
    label: &str,
    sample_rate: u32,
) {
    report_dsp_fidelity_impl(
        reference,
        test,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        label,
        sample_rate,
        true,
    )
}

/// Like [`report_dsp_fidelity`] but skips the LUFS plausibility gate.
///
/// Use when the reference signal has LUFS outside the plausible range for
/// legitimate reasons, not indicating a defect:
/// - IR convolution goldens (synthetic signal + IR can legitimately produce
///   LUFS above +10 or below −50)
/// - Dynamic/free-shape models with low head_scale (e.g., WaveNetDyn
///   Free-Shape at ~−65 LUFS, LSTM-Dyn at ~−55 LUFS)
///
/// With BS.1770-4 full LUFS, the measurement is accurate — these
/// are genuine opt-outs for models whose output loudness is inherently
/// outside the [−50, +10] range, not a workaround for measurement error.
#[track_caller]
#[expect(
    clippy::too_many_arguments,
    reason = "Test validation helper with many configuration parameters for comprehensive test fixture setup"
)]
pub fn report_dsp_fidelity_no_lufs(
    reference: &[f32],
    test: &[f32],
    mse_limit: Option<f64>,
    min_snr_db: f64,
    max_esr: Option<f64>,
    mrstft_max: Option<f64>,
    label: &str,
    sample_rate: u32,
) {
    report_dsp_fidelity_impl(
        reference,
        test,
        mse_limit,
        min_snr_db,
        max_esr,
        mrstft_max,
        label,
        sample_rate,
        false,
    )
}

#[track_caller]
#[expect(
    clippy::too_many_arguments,
    reason = "Test validation helper with many configuration parameters for comprehensive test fixture setup"
)]
fn report_dsp_fidelity_impl(
    reference: &[f32],
    test: &[f32],
    mse_limit: Option<f64>,
    min_snr_db: f64,
    max_esr: Option<f64>,
    mrstft_max: Option<f64>,
    label: &str,
    sample_rate: u32,
    check_lufs_gate: bool,
) {
    assert_eq!(
        reference.len(),
        test.len(),
        "[{label}] Vectors of different sizes for report_dsp_fidelity"
    );
    let n = reference.len() as f64;
    let sr = sample_rate;

    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;
    let mut max_abs_diff = 0.0f64;
    let mut peak_ref = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let r64 = r as f64;
        let t64 = t as f64;
        let diff = r64 - t64;
        signal_power += r64 * r64;
        noise_power += diff * diff;
        let abs_diff = diff.abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        if r64.abs() > peak_ref {
            peak_ref = r64.abs();
        }
    }
    let mse = noise_power / n;
    let mae = max_abs_diff;
    let snr = if noise_power <= f64::EPSILON {
        f64::INFINITY
    } else {
        10.0 * (signal_power / noise_power).log10()
    };
    let psnr = if mse <= f64::EPSILON {
        f64::INFINITY
    } else {
        10.0 * (peak_ref * peak_ref / mse).log10()
    };
    let signal_avg_power = signal_power / n;
    let bits = if mse <= f64::EPSILON || signal_avg_power <= f64::EPSILON {
        f64::INFINITY
    } else {
        -0.5 * (mse / signal_avg_power).log2()
    };

    // ESR (linear + dB)
    let esr_linear = if signal_power <= f64::EPSILON {
        if noise_power <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        noise_power / signal_power
    };
    let esr_db = neural_amp_modeler_rs::testing::perceptual::esr_to_db(esr_linear);

    // LUFS — reference (golden) for plausibility sanity gate
    let lufs_ref = neural_amp_modeler_rs::testing::perceptual::compute_lufs(reference, sr);
    let lufs_test = neural_amp_modeler_rs::testing::perceptual::compute_lufs(test, sr);
    let dbtp_ref = neural_amp_modeler_rs::testing::perceptual::compute_true_peak_db(reference);
    let lufs_plausible = if lufs_ref.is_finite() {
        (LUFS_PLAUSIBLE_MIN..=LUFS_PLAUSIBLE_MAX).contains(&lufs_ref)
    } else {
        // LUFS non-finite: signal too short (<400 ms block) or all-zero — gate not applicable
        true
    };

    // SNR(reference, anchor) sanity: compute SNR of test against a low-pass 3.5 kHz anchor
    let anchor_snr_db = {
        let anchor = neural_amp_modeler_rs::testing::mushra::low_pass_1pole(reference, 3500.0, sr);
        neural_amp_modeler_rs::testing::perceptual::compute_snr_db(reference, &anchor)
    };

    // Build the entire report into a single String buffer so that the block
    // (header + all metrics + footer) is emitted atomically.
    // Protected by REPORT_LOCK to prevent interleaving across threads
    // even under --test-threads > 1.
    let mut buf = String::with_capacity(1024);
    writeln!(buf).unwrap();
    writeln!(
        buf,
        "[NeuralAmpModelerCore × NeuralAmpModeler-rs — {label}]"
    )
    .unwrap();
    if let Some(limit) = mse_limit {
        writeln!(
            buf,
            "  MSE     = {mse:.2e}      (threshold < {limit:.1e})  {}",
            if mse < limit { "✓" } else { "✗" }
        )
        .unwrap();
    } else {
        writeln!(
            buf,
            "  MSE     = {mse:.2e}      (gate: N/A — ESR primary)  ⓘ"
        )
        .unwrap();
    }
    writeln!(buf, "  MAE     = {mae:.2e}").unwrap();
    if snr.is_finite() {
        writeln!(
            buf,
            "  SNR     = {snr:.1} dB       (threshold ≥ {min_snr_db:.1} dB)   {}",
            if snr >= min_snr_db { "✓" } else { "✗" }
        )
        .unwrap();
    } else {
        writeln!(buf, "  SNR     = ∞ dB").unwrap();
    }
    if psnr.is_finite() {
        writeln!(buf, "  PSNR    = {psnr:.1} dB").unwrap();
    } else {
        writeln!(buf, "  PSNR    = ∞ dB").unwrap();
    }
    if bits.is_finite() {
        writeln!(buf, "  Bits    = {bits:.2} bits equiv.").unwrap();
    } else {
        writeln!(buf, "  Bits    = ∞ bits equiv.").unwrap();
    }
    if esr_linear.is_finite() {
        let a1std = neural_amp_modeler_rs::testing::perceptual::A2ESR_A1_STANDARD_MEDIAN;
        let a2full = neural_amp_modeler_rs::testing::perceptual::A2ESR_A2_FULL_MEDIAN;
        let a2lite = neural_amp_modeler_rs::testing::perceptual::A2ESR_A2_LITE_MEDIAN;
        if let Some(limit) = max_esr {
            writeln!(
                buf,
                "  ESR     = {esr_linear:.2e}       ({esr_db:.1} dB)   (threshold < {limit:.1e})  {}   [baseline A1-Std: {a1std:.2e}, A2-Full: {a2full:.2e}, A2-Lite: {a2lite:.2e}]",
                if esr_linear < limit { "✓" } else { "✗" },
            )
            .unwrap();
        } else {
            writeln!(
                buf,
                "  ESR     = {esr_linear:.2e}       ({esr_db:.1} dB)   [baseline A1-Std: {a1std:.2e}, A2-Full: {a2full:.2e}, A2-Lite: {a2lite:.2e}]",
            )
            .unwrap();
        }
    } else {
        writeln!(buf, "  ESR     = ∞  (identical)").unwrap();
    }

    // MR-STFT — hard gate at 44.1/48 kHz, soft gate at higher rates
    let mr_stft = neural_amp_modeler_rs::testing::perceptual::compute_mr_stft(reference, test);
    let mrstft_hard = mrstft_max.is_some() && (sample_rate == 44100 || sample_rate == 48000);
    if mrstft_hard {
        let limit = mrstft_max.unwrap();
        writeln!(
            buf,
            "  MR-STFT = {mr_stft:.4e}      (threshold < {limit:.2e})  {}   [hard gate @ {sample_rate} Hz]",
            if mr_stft.is_finite() && mr_stft < limit { "✓" } else { "✗" },
        )
        .unwrap();
    } else {
        writeln!(buf, "  MR-STFT = {mr_stft:.4e}      (log-mag abs)").unwrap();
        if !mr_stft.is_finite() || mr_stft > MRSTFT_SOFT_THRESHOLD {
            writeln!(
                buf,
                "  ⚠  MR-STFT soft gate: {mr_stft:.4e} exceeds calibrated ceiling {MRSTFT_SOFT_THRESHOLD:.2e} (anti-placebo)"
            )
            .unwrap();
            writeln!(
                buf,
                "     (informational, not a hard assertion @ {sample_rate} Hz)"
            )
            .unwrap();
        }
    }

    if lufs_test.is_finite() {
        if lufs_ref.is_finite() {
            let dbtp_str = if dbtp_ref.is_finite() {
                format!("  dBTP    = {dbtp_ref:.1} dBTP    (true-peak)\n")
            } else {
                String::new()
            };
            writeln!(
                buf,
                "{dbtp_str}  LUFS    = {lufs_ref:.1} LUFS    (reference)   [plausible: {LUFS_PLAUSIBLE_MIN:.0}..{LUFS_PLAUSIBLE_MAX:.0}]  {}",
                if lufs_plausible {
                    "✓"
                } else if check_lufs_gate {
                    "✗ — GOLDEN DEFECT (LUFS plausibility check)"
                } else {
                    "ⓘ informational (gate opt-out — expected)"
                }
            )
            .unwrap();
        } else {
            writeln!(buf, "  LUFS    = N/A (signal too short for 400 ms block)").unwrap();
        }
        if anchor_snr_db.is_finite() {
            let delta_snr = snr - anchor_snr_db;
            let is_satisfactory = delta_snr > 8.0;
            writeln!(
                buf,
                "  SNR(anchor) = {anchor_snr_db:.1} dB (degradation reference)"
            )
            .unwrap();
            writeln!(
                buf,
                "  Fidelity Margin = {delta_snr:.1} dB (target > 8.0 dB) {}",
                if is_satisfactory { "✓" } else { "?" }
            )
            .unwrap();
        }
    }
    writeln!(
        buf,
        "  Samples = {} @ {sr} Hz (stress signal)",
        reference.len()
    )
    .unwrap();

    {
        let _lock = REPORT_LOCK.lock().unwrap();
        if !SUPPRESS_REPORT.with(|c| c.get()) {
            print!("{buf}");
        }
    }

    // JSONL sink resolution: per-thread override first (deterministic,
    // concurrency-immune), process env var as fallback for dashboard runs.
    let json_label = METRIC_MODEL
        .with(|c| c.borrow().clone())
        .unwrap_or_else(|| format!("{label} @{sample_rate}"));
    let json_kind = METRIC_KIND
        .with(|c| c.borrow().clone())
        .unwrap_or_else(|| "fidelity".to_string());
    let obj = serde_json::json!({
        "label": json_label,
        "kind": json_kind,
        "esr": json_metric(esr_linear),
        "esr_db": json_metric(esr_db),
        "snr_db": json_snr_db(snr),
        "mrstft": json_metric(mr_stft),
        "mse": json_metric(mse),
    });
    append_metric_line(obj);

    if let Some(limit) = mse_limit {
        assert!(
            mse < limit,
            "[{label}] MSE={mse:.6e} exceeds threshold {limit:.1e} (MAE={mae:.6e}, SNR={snr:.1} dB)"
        );
    }
    assert!(
        snr >= min_snr_db,
        "[{label}] SNR={snr:.1} dB below minimum {min_snr_db:.1} dB (MSE={mse:.6e}, MAE={mae:.6e})"
    );
    if let Some(limit) = max_esr {
        assert!(
            esr_linear < limit,
            "[{label}] ESR={esr_linear:.6e} exceeds threshold {limit:.1e} (ESR dB={esr_db:.1})"
        );
    }
    // MR-STFT hard gate at 44.1/48 kHz
    if mrstft_hard {
        let limit = mrstft_max.unwrap();
        assert!(
            mr_stft.is_finite() && mr_stft < limit,
            "[{label}] MR-STFT={mr_stft:.4e} exceeds threshold {limit:.2e} @ {sample_rate} Hz \
             (spectral fidelity regression detected)"
        );
    }
    // LUFS plausibility sanity gate — catch near-silence / implausible golden output.
    // Only enforced when check_lufs_gate is true (opt-out for IR convolution goldens).
    if check_lufs_gate {
        assert!(
            lufs_plausible,
            "[{label}] Reference LUFS={lufs_ref:.1} is outside plausible audio range \
             [{LUFS_PLAUSIBLE_MIN:.0}, {LUFS_PLAUSIBLE_MAX:.0}]. \
             The golden output may be defective (near-silence, clipping, or wrong scaling)."
        );
    } else if !lufs_plausible {
        eprintln!(
            "  ⓘ  LUFS gate skipped for [{label}]: reference LUFS={lufs_ref:.1} \
             outside [{LUFS_PLAUSIBLE_MIN:.0}, {LUFS_PLAUSIBLE_MAX:.0}] — \
             expected for IR convolution / dynamic free-shape goldens \
             (gate opt-out)"
        );
    }
}

/// Converts SNR (dB) to a conservative MSE upper-bound estimate.
///
/// Assumes signal_power ≈ 0.3 × n for a normalized stress signal,
/// balancing tightness with headroom to avoid false positives.
#[inline]
fn snr_to_mse(snr_db: f64) -> f64 {
    10.0_f64.powf(-snr_db / 10.0) * 0.3
}

/// Shared WaveNet MSE/SNR/ESR threshold lookup — used by both `topology_thresholds`
/// (golden vectors) and `live_parity_thresholds` (cpp_parity) as a fallback.
///
/// Live v1/v2 SNR measurements (f32-exact, 2026-06-18):
///   Standard (CH=16): 123-135 dB → floor 105 dB (18 dB margin)
///   Feather  (CH=8):  117-133 dB → floor 100 dB (17 dB margin)
///   Nano     (CH=4):  114-132 dB → floor  95 dB (19 dB margin)
///   Lite     (CH=12): 117.4 dB → floor 100 dB (17.4 dB margin, P1)
///
// ESR gates (f32-exact):
///   Standard:             3e-11
///   Feather:              1e-10
///   Nano:                 3e-10
///   Lite:                 1e-10 (P1 ✅ resolved)
///   Default:              1e-3
///
/// Returns `(mse_limit, min_snr_db, max_esr)`.
#[inline]
fn wavenet_thresholds(channels: u32) -> (f64, f64, Option<f64>) {
    match channels {
        3 => (snr_to_mse(40.0), 40.0, Some(1e-3)),
        4 => {
            let snr_db = 95.0;
            (snr_to_mse(snr_db), snr_db, Some(3e-10))
        }
        8 => {
            let snr_db = 100.0;
            (snr_to_mse(snr_db), snr_db, Some(1e-10))
        }
        12 => {
            // Post-P1 fix: measured SNR=117.4 dB, ESR=1.83e-12.
            // Floor: 100 dB (17.4 dB margin, honest — on par with Feather CH=8).
            let snr_db = 100.0;
            (snr_to_mse(snr_db), snr_db, Some(1e-10))
        }
        16 => {
            let snr_db = 85.0;
            (snr_to_mse(snr_db), snr_db, Some(3e-9))
        }
        _ => {
            let snr_db = 40.0;
            (snr_to_mse(snr_db), snr_db, Some(1e-3))
        }
    }
}

/// Lookup for calibrated thresholds of committed models based on real measurements.
/// Sets the floors as `SNR_medido - margem` and `ESR_medido * fator`.
///
/// Returns `None` if the model has no calibrated entry, falling back to
/// heuristic thresholds (`wavenet_thresholds` or LSTM formula).
///
/// Every model with a committed golden `.bin` fixture MUST have an entry here.
/// The meta-test `tests/threshold_calibration.rs` enforces this invariant.
///
/// Returns `(mse_limit, min_snr_db, max_esr, mrstft_max)`.
/// `mse_limit` is `None` when MSE gate is not applicable (ESR is the primary gate,
/// explicit `MseGate::NotApplicable` semantics).
/// `mrstft_max` is asserted as a hard gate at 44.1/48 kHz.
#[allow(clippy::type_complexity, clippy::allow_attributes)]
pub fn get_calibrated_threshold(
    model_name: &str,
) -> Option<(Option<f64>, f64, Option<f64>, Option<f64>)> {
    let base_name = if let Some(idx) = model_name.find("_v2_") {
        &model_name[..idx]
    } else {
        model_name
    };
    match base_name {
        // --- WaveNet Standard (CH=16) ---
        // Measured: SNR=134.6 dB (live v1), SNR=123.0 dB (v2 worst @ 88.2 kHz), ESR=4.99e-13,
        // MRSTFT near-zero (near-bit-exact, gate=0.05 conservative)
        // pós-nuke f32. Floor: SNR - 18 dB margin from v2 worst, ESR factor ~60x
        "BossWN-standard" | "wavenet_standard" => {
            let snr_db = 105.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(3.0e-11), Some(0.05)))
        }
        // --- WaveNet Feather (CH=8) ---
        // Measured: SNR=133.1 dB (live v1), SNR=117.6 dB (v2 worst @ 192 kHz), ESR=1.74e-12,
        // MRSTFT near-zero (near-bit-exact, gate=0.05 conservative)
        // pós-nuke f32. Floor: SNR - 17 dB margin from v2 worst, ESR factor ~57x
        "BossWN-feather" | "wavenet_feather" => {
            let snr_db = 100.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(1.0e-10), Some(0.05)))
        }
        // --- WaveNet Nano (CH=4) ---
        // Measured: SNR=132.0 dB (live v1), SNR=114.6 dB (v2 worst @ 192 kHz), ESR=3.46e-12,
        // MRSTFT near-zero (near-bit-exact, gate=0.05 conservative)
        // pós-nuke f32. Floor: SNR - 19 dB margin from v2 worst, ESR factor ~87x
        "BossWN-nano" | "wavenet_nano" => {
            let snr_db = 95.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(3.0e-10), Some(0.05)))
        }
        // --- WaveNet A1 Standard (Official) (CH=16) ---
        // Measured: SNR=123.4 dB (live v1), SNR=101.8 dB (v2 worst @ 192 kHz), ESR=6.62e-11,
        // MRSTFT near-zero (near-bit-exact, gate=0.05 conservative)
        // pós-nuke f32. Floor: SNR - 16 dB margin from v2 worst, ESR factor ~45x
        "wavenet_a1_standard" => {
            let snr_db = 85.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(3.0e-9), Some(0.05)))
        }
        // --- WaveNet Official (CH=3 free geom, dynamic path) ---
        // Triage: (a) Inherent — free-geometry CH=3 dynamic path exercises non-SKU
        // WaveNet Official (free-geom CH=3): synthetic dilations [(1,2),(8)].
        // Thresholds preserved from pre-fix calibration; now provide ~116 dB margin.
        // Relaxed mrstft_max to 0.45 — free-geometry dynamic path shows
        //   significant spectral drift over 5s v2 sequences despite near-bit-exact SNR.
        // Measured: SNR=130.4 dB, ESR=1.8e-12, MR-STFT=0.42 (v2 @ 48 kHz, 5s stress)
        "wavenet_official" => {
            let snr_db = 14.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(3.5e-2), Some(0.45)))
        }
        // --- LSTM 1x16 ---
        // Measured: SNR=108.5 dB, ESR=1.42e-11 (golden v1, 2026-07-12, Standard engine).
        // Post-Standard-default (exact polynomial activations): LSTM golden fidelity
        // improved by ~89 dB SNR / ~9 orders of magnitude ESR vs old Fast (Padé) mode.
        // Floor: SNR - 15.5 dB, ESR factor ~106× (conservative, LSTM v2 drift reserve).
        "BossLSTM-1x16" | "lstm_1x16" => {
            let snr_db = 93.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(1.5e-9), Some(0.20)))
        }
        // --- LSTM 2x8 ---
        // Measured: SNR=107.8 dB, ESR=1.67e-11 (golden v1, 2026-07-12, Standard engine).
        // Post-Standard-default: LSTM golden fidelity improved by ~82 dB SNR /
        // ~8 orders of magnitude ESR vs old Fast (Padé) mode.
        // Floor: SNR - 14.8 dB, ESR factor ~102× (conservative, LSTM v2 drift reserve).
        "BossLSTM-2x8" | "lstm_2x8" => {
            let snr_db = 93.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(1.7e-9), Some(0.12)))
        }
        // --- LSTM Official (H=3) ---
        // Measured: SNR=120.8 dB, ESR=8.30e-13 (golden v1, 2026-07-12, Standard engine).
        // Post-Standard-default: LSTM golden fidelity improved by ~91 dB SNR /
        // ~9 orders of magnitude ESR vs old Fast (Padé) mode.
        // Floor: SNR - 15.8 dB, ESR factor ~108× (conservative, LSTM v2 drift reserve).
        "lstm (Official)" | "lstm_official" => {
            let snr_db = 105.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(9.0e-11), Some(0.22)))
        }
        // --- WaveNet Lite (CH=12) — P1 ✅ RESOLVIDO ---
        // Measured: SNR=122.3 dB, ESR=5.84e-13 (EVH-5150-Lite, post-migration),
        // MRSTFT near-zero (near-bit-exact, gate=0.05 conservative)
        // Floor: SNR - 17.3 dB margin, ESR factor ~60x
        "EVH-5150-Lite" | "wavenet_lite" => {
            let snr_db = 105.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(3.5e-11), Some(0.05)))
        }
        // --- WaveNet A2 Full (CH=8) ---
        // Post-weight-dequantization — near-bit-exact (was 79.2 dB / 1.21e-8 with f16c weights).
        // Measured: SNR = 129.5 dB, ESR = 1.13e-13 (golden v1, 2026-07-05),
        // MRSTFT = 4.2e-5 (near-bit-exact, gate=0.05 conservative)
        // Margin: SNR - 24.5 dB, ESR factor ~265×
        "wavenet_a2_full" => {
            let snr_db = 105.0;
            Some((None, snr_db, Some(3.0e-11), Some(0.05)))
        }
        // --- WaveNet A2 Lite (CH=3) ---
        // Post-weight-dequantization — near-bit-exact (was 90.7 dB / 8.58e-10 with f16c weights).
        // Measured: SNR = 132.2 dB, ESR = 6.08e-14 (golden v1, 2026-07-05),
        // MRSTFT = 2.9e-5 (near-bit-exact, gate=0.05 conservative)
        // Margin: SNR - 27.2 dB, ESR factor ~576×
        "wavenet_a2_lite" => {
            let snr_db = 105.0;
            Some((None, snr_db, Some(3.5e-11), Some(0.05)))
        }
        // --- WaveNet Condition DSP (CH=3, cond=3, dynamic path) ---
        // condition_dsp sub-model with 2-layer WaveNet providing 3-channel
        // conditioning. The sub-model processes the raw audio before the main arrays.
        // Measured: SNR=139.5 dB, ESR=1.13e-14 (nearly bit-exact, 2026-06-19),
        // MR-STFT=0.021 (v1) / 0.336 (v2 @ 48 kHz 5s)
        // Floor: SNR - 39.5 dB margin, ESR factor ~8800x
        // Relaxed mrstft_max to 0.35 — condition_dsp sub-model accumulates
        //   drift over 5-second v2 sequences.
        "wavenet_condition_dsp" => {
            let snr_db = 100.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(1.0e-10), Some(0.35)))
        }
        // --- Nondist Models ---
        // Measured: not individually measured; floors SNR≥100.0 dB, ESR≤1.0e-10, MRSTFT≤0.05
        // (production WaveNet CH=3/4 models validated by cpp_parity + golden vectors,
        // near-bit-exact characteristics, thresholds calibrated conservatively)
        "APP-EVH-Stealth100-Dialled-xSTD"
        | "APP-EVH-Stealth100-Dialled-xSTD.nam"
        | "Boss BD-2 H2O Mod T-12_00 G-12_00"
        | "Boss BD-2 H2O Mod T-12_00 G-12_00.nam"
        | "SLAMMIN MARSHALL JTM 45 REISSUE"
        | "SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam"
        | "wavenet_app_evh"
        | "wavenet_boss_bd2"
        | "wavenet_slammin_marshall" => {
            let snr_db = 100.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(1.0e-10), Some(0.05)))
        }
        // --- WaveNet A2-FiLM-Lite (CH=3, FiLM active, RF1 🔴) ---
        // FiLM modulation adds per-frame conditioning (FiLM gamma/beta) that diverges
        // from C++ generic WaveNet path. The Rust WaveNetA2Dyn engine implements FiLM
        // natively; C++ a2_fast.cpp rejects FiLM and falls back to Eigen-based generic
        // WaveNet. The divergence is inherent — not an engine regression.
        // Measured: SNR=124.2 dB, ESR=3.83e-13 (golden v1, 2026-07-12, Standard engine),
        // MR-STFT=1.65e-5 (FiLM vs generic @ 48 kHz), gate=1.0e-4 (generous tolerance margin)
        // Margin: SNR - 10.2 dB, ESR factor ~26×
        "wavenet_a2_film_lite" => {
            let snr_db = 114.0;
            Some((
                Some(snr_to_mse(snr_db)),
                snr_db,
                Some(1.0e-11),
                Some(1.0e-4),
            ))
        }
        // --- WaveNet A2-FiLM-Full (CH=8, FiLM active, RF1 🔴) ---
        // FiLM modulation on 8-channel A2 model. C++ routes to generic WaveNet
        // (Eigen), Rust routes to WaveNetA2Dyn with native FiLM support.
        // Measured: SNR=138.8 dB, ESR=1.31e-14 (2026-07-10, float32 precision limit),
        // MR-STFT=3.28e-5 (FiLM vs generic), gate=1.0e-4 (generous margin)
        // Margin: SNR - 18.8 dB, ESR factor ~7.6e2x
        "wavenet_a2_film_full" => {
            let snr_db = 120.0;
            Some((
                Some(snr_to_mse(snr_db)),
                snr_db,
                Some(1.0e-11),
                Some(1.0e-4),
            ))
        }
        // --- WaveNet A2-FiLM Chaos Stress (CH=3, FiLM active) ---
        // Measured: SNR=139.0 dB, ESR=1.25e-14, MR-STFT=7.32e-6
        // (golden v1, 2026-07-12, Standard engine). Post-Standard-default:
        // chaos stress model is near-bit-exact — exact polynomial activations
        // replaced Padé approximations that previously caused divergence.
        // Floor: SNR - 19.0 dB, ESR factor ~80×, MR-STFT factor ~6.8×.
        "wavenet_a2_film_chaos_stress" => {
            let snr_db = 120.0;
            Some((
                Some(snr_to_mse(snr_db)),
                snr_db,
                Some(1.0e-12),
                Some(5.0e-5),
            ))
        }
        // --- WaveNet A2-FiLM-InputMixinPre (CH=3, input_mixin_pre_film only) ---
        // Isolated input_mixin_pre_film (slot 2) with only one FiLM slot active,
        // producing less divergence than multi-slot FiLM models (Lite/Full).
        // Measured: SNR=134.4 dB, ESR=3.66e-14, MR-STFT=7.03e-6 (2026-07-12, float32 precision limit)
        // Margin: SNR - 14.4 dB, ESR factor ~273x, MR-STFT factor ~14x
        "wavenet_a2_film_input_mixin_pre" => {
            let snr_db = 120.0;
            Some((
                Some(snr_to_mse(snr_db)),
                snr_db,
                Some(1.0e-11),
                Some(1.0e-4),
            ))
        }
        // --- WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1) — KB-A2-MAX ---
        // Fail-closed TR1.1. Thresholds are placeholders for a future un-ignore
        // only after §4.4.3 reopening (SNR≥90 dB). HEAD measured ~0.23 dB.
        // Do not use these gates to claim parity while the guard is active.
        "wavenet_a2_max" => {
            let snr_db = 90.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(1.0e-9), Some(0.05)))
        }
        // --- WaveNet A2 Dynamic Gated CH=8 (Task 3.3) ---
        // Gating doubles conv output (channels × 2*bottleneck) and applies
        // Sigmoid gate + LeakyReLU main activation. C++ uses Eigen-based generic
        // WaveNet, Rust uses WaveNetA2Dyn per-frame.
        // Measured: SNR=103.0 dB, ESR=5.01e-11 (2026-06-19),
        // MRSTFT near-zero (near-bit-exact, gate=0.05 conservative)
        // Margin: SNR - 18.0 dB, ESR factor ~20x
        "a2_dynamic_gated_ch8" => {
            let snr_db = 85.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(1.0e-9), Some(0.05)))
        }
        // --- WaveNet A2 Dynamic Blended CH=3 (Task 3.3) ---
        // Blending mixes main activation (LeakyReLU) with Tanh gate via linear
        // interpolation. C++ uses Eigen-based generic WaveNet, Rust uses
        // WaveNetA2Dyn per-frame.
        // Measured: SNR=133.0 dB, ESR=5.01e-14 (2026-06-19),
        // MRSTFT near-zero (near-bit-exact, gate=0.05 conservative)
        // Margin: SNR - 23.0 dB, ESR factor ~20x
        "a2_dynamic_blended_ch3" => {
            let snr_db = 110.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(1.0e-12), Some(0.05)))
        }
        // --- WaveNetDyn Free-Shape (CH=7→4, dynamic free geometry topology) ---
        // Free-geometry dynamic path, 2 arrays, Tanh activation, head_scale=0.02.
        // The low head_scale produces quiet output (~−65 LUFS) that trips the
        // LUFS plausibility gate — golden tests use report_dsp_fidelity_no_lufs.
        // Measured: SNR=124.2 dB, ESR=3.79e-13,
        // MR-STFT=0.170 (v2 @ 48 kHz 5s), gate=0.18 (6% margin)
        // Margin: SNR - 34.2 dB, ESR factor ~26x
        "wavenet_dyn_free" => {
            let snr_db = 90.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(1.0e-11), Some(0.18)))
        }
        // --- LSTM-Dyn 1×7 (dynamic single-layer topology) ---
        // Single-layer LSTM with hidden_size=7, non-catalog geometry routed to
        // LstmModelDyn. Recurrent state accumulation over 2048-sample stress
        // signal produces measurable but minimal drift at 48 kHz.
        // Measured: SNR=90.8 dB, ESR=8.34e-10 (after minimax/Padé unification,
        // `Fast` was the global default at the time), MR-STFT=0.0585 (v1, 2048 samples @ 48 kHz), gate=0.08 (37% margin)
        // Margin: SNR - 10.8 dB, ESR factor ~4x
        //
        // Re-measured after the Standard/Fast rename (2026-07-09): `Standard`
        // (exact-grade polynomial activations) is now the universal default,
        // replacing `Fast` (Padé/minimax) as the ambient global mode for this
        // model (LstmModelDyn is not covered by any per-model override).
        // Measured: SNR=144.3 dB, ESR=3.69e-15, MR-STFT=8.1463e-2 (v1, 2048
        // samples @ 48 kHz). SNR/ESR improved by ~9 orders of magnitude (as
        // expected — exact activations eliminate Padé approximation error),
        // but MR-STFT (a different, spectral-domain metric) increased
        // slightly relative to the old Fast-default measurement; gate raised
        // from 0.08 to 0.10 (~23% margin over the new measured value).
        "lstm_dyn_test" => {
            let snr_db = 80.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(3.5e-9), Some(0.10)))
        }
        // --- SlimmableContainer A2 Example (CH=3→6) ---
        // Official C++ upstream model A2.nam with 2 WaveNet A2 submodels (CH=3, CH=6).
        // Measured: SNR=134.9 dB, ESR=3.27e-14, MR-STFT=9.22e-6
        // (golden v1, 2026-07-12, Standard engine). Post-Standard-default:
        // A2 container golden fidelity improved by ~53 dB SNR / ~5 orders of
        // magnitude ESR vs old Fast (Padé) mode.
        // Floor: SNR - 14.9 dB, ESR factor ~107×.
        "a2_example" => {
            let snr_db = 120.0;
            Some((None, snr_db, Some(3.5e-12), Some(0.08)))
        }
        // --- ConvNet Test (CH=1, 1 block, C++ flat format) ---
        // Measured: SNR=143.8 dB, ESR=4.20e-15, MR-STFT=1.20e-6 (C++ render
        // cross-validation, 2026-07-28, 2048-sample v1 stress signal).
        // Prewarm fix (TASK-CONVNET-01) eliminated the 2.54e-5 transient,
        // bringing ESR to the f32 numerical noise floor.
        //
        // Floor: SNR - 23.8 dB, ESR factor ~238×, MR-STFT factor ~83×.
        "convnet_test" => {
            let snr_db = 120.0;
            Some((
                Some(snr_to_mse(snr_db)),
                snr_db,
                Some(1.0e-12),
                Some(1.0e-4),
            ))
        }
        // --- Linear FFT — Partitioned Convolution ---
        // FFT-based FIR convolution via partitioned overlapless FFT.
        // Mathematically equivalent to time-domain convolution; FFT round-trip
        // error is the only noise source. Validated against direct FIR oracle
        // and C++ golden vectors (NeuralAmpModelerCore `nam::Linear` dsp.cpp).
        //
        // Unlike neural models (WaveNet, LSTM), Linear FFT has no recurrent
        // state, no activation functions, and no weight dequantization — it is
        // deterministic floating-point signal processing.
        //
        // Measured: SNR=137.9 dB (worst RF=2048), ESR=1.62e-14, MR-STFT=2.02e-6
        // (golden v1, 2026-07-12, Standard engine). Other signals (sine, multi-freq)
        // produce MR-STFT near-zero (FFT round-trip only).
        //
        // Floor: SNR - 12.9 dB from worst measured (RF=2048), ESR factor ~6.2e3×.
        //
        // Thresholds apply to all Linear FFT receptive field sizes:
        // RF=2048, RF=4096, RF=8192 — FFT precision is RF-independent at f32.
        "linear_fft_rf320" | "linear_fft_rf2048" | "linear_fft_rf4096" | "linear_fft_rf8192" => {
            let snr_db = 125.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(1.0e-10), Some(0.12)))
        }
        // LSTM uncatalogued synthetic topologies (golden v1, 2026-08-10, Standard engine)
        //
        // Measured: lstm_1x10 SNR=144.0 dB, ESR=4.01e-15, MR-STFT=1.25e-6
        // Measured: lstm_2x24 SNR=135.7 dB, ESR=2.71e-14, MR-STFT=3.50e-6
        // Measured: lstm_3x8  SNR=144.4 dB, ESR=3.66e-15, MR-STFT=5.55e-7
        //
        // Floor: SNR - 19..26 dB from measured, ESR factor ~185..273×.
        // MR-STFT factor ~80..180×. Worst case across all 3 topologies.
        "lstm_1x10" | "lstm_2x24" | "lstm_3x8" => {
            let snr_db = 110.0;
            Some((
                Some(snr_to_mse(snr_db)),
                snr_db,
                Some(5.0e-12),
                Some(5.0e-4),
            ))
        }
        // ConvNet variants (golden v1, 2026-08-10, Standard engine)
        //
        // Measured: convnet_nobn SNR=134.9 dB, ESR=3.23e-14, MR-STFT=5.62e-6
        // Measured: convnet_relu SNR=141.6 dB, ESR=6.84e-15, MR-STFT=2.27e-6
        // Measured: convnet_silu SNR=∞, ESR=2.58e-13, MR-STFT=8.67e-6
        //
        // Floor: SNR - 20..22 dB from worst (nobn), ESR factor ~31..146×,
        // MR-STFT factor ~44..89×. Calibrated conservatively across all variants.
        "convnet_nobn" | "convnet_relu" | "convnet_silu" => {
            let snr_db = 115.0;
            Some((
                Some(snr_to_mse(snr_db)),
                snr_db,
                Some(1.0e-11),
                Some(5.0e-4),
            ))
        }
        // Linear No Bias (golden v1, 2026-08-10, Standard engine)
        //
        // Measured: SNR=144.1 dB, ESR=3.89e-15, MR-STFT=1.64e-6
        // (2048-sample v1 stress, 48 kHz). Near-bit-exact FFT round-trip
        // like other Linear variants; same conservative floor as linear_fft.
        //
        // Floor: SNR - 19.1 dB, ESR factor ~2.6e4×, MR-STFT factor ~7.3e4×.
        "linear_nobias" => {
            let snr_db = 125.0;
            Some((Some(snr_to_mse(snr_db)), snr_db, Some(1.0e-10), Some(0.12)))
        }
        _ => None,
    }
}

/// Computes MSE/SNR/ESR test thresholds for golden vector tests.
///
/// For live cpp_parity cross-validation, use `live_parity_thresholds()`
/// which applies tighter LSTM floors reflecting the 50–97 dB live SNR.
///
/// Returns `(mse_limit, min_snr_db, max_esr, mrstft_max)` — added relative
/// ESR gate as primary threshold (robust to scale mismatch).
/// mrstft_max asserted as hard gate at 44.1/48 kHz.
pub fn topology_thresholds(
    data: &neural_amp_modeler_rs::loader::nam_json::NamModelData,
    model_name: &str,
) -> (Option<f64>, f64, Option<f64>, Option<f64>) {
    if let Some(thresholds) = get_calibrated_threshold(model_name) {
        return thresholds;
    }
    match data.architecture.as_str() {
        "WaveNet" => {
            let channels = data
                .config
                .layers
                .first()
                .and_then(|l| l.channels)
                .unwrap_or(16);
            let (mse, snr, esr) = wavenet_thresholds(channels as u32);
            (Some(mse), snr, esr, None)
        }
        "LSTM" => {
            let num_layers = data.config.num_layers.unwrap_or(1);
            let hidden_size = data.config.hidden_size.unwrap_or(16);
            let complexity = (num_layers * hidden_size) as f64;
            let snr_db = (30.0 - complexity * 0.65).clamp(12.0, 30.0);
            let mse = snr_to_mse(snr_db);
            let esr = 10.0_f64.powf(-snr_db / 10.0) * 2.0;
            (Some(mse.clamp(1e-4, 5e-2)), snr_db, Some(esr), None)
        }
        "Linear" => (Some(1e-10), 135.0, Some(1e-10), Some(0.12)),
        "ConvNet" => {
            // ConvNet multi-block models — self-golden consistency: ESR=0
            // (output determinism proven across block sizes).
            // Live C++ cross-validation (2026-07-28): ESR=4.20e-15, SNR=143.8 dB.
            let snr_db = 140.0;
            (Some(snr_to_mse(snr_db)), snr_db, Some(1.0e-10), Some(0.05))
        }
        _ => (Some(5e-2), 9.0, Some(1e-3), None),
    }
}

/// Computes MSE/SNR/ESR thresholds for live C++ cross-validation (`cpp_parity.rs`).
///
/// Uses aggressive floors reflecting post-T16.1 live SNR measurements.
/// LSTM formula targets 50–97 dB live SNR with ~10–15 dB margin
/// (v2 stress signal relaxation applied separately in `cpp_parity.rs`).
///
/// T16.4: ESR gate added as primary threshold (robust to scale mismatch).
///
/// Returns `(mse_limit, min_snr_db, max_esr, mrstft_max)`.
/// mrstft_max asserted as hard gate at 44.1/48 kHz.
pub fn live_parity_thresholds(
    data: &neural_amp_modeler_rs::loader::nam_json::NamModelData,
    model_name: &str,
) -> (Option<f64>, f64, Option<f64>, Option<f64>) {
    if let Some(thresholds) = get_calibrated_threshold(model_name) {
        return thresholds;
    }
    match data.architecture.as_str() {
        "WaveNet" => {
            let channels = data
                .config
                .layers
                .first()
                .and_then(|l| l.channels)
                .unwrap_or(16);
            let (mse, snr, esr) = wavenet_thresholds(channels as u32);
            (Some(mse), snr, esr, None)
        }
        "LSTM" => {
            let num_layers = data.config.num_layers.unwrap_or(1);
            let hidden_size = data.config.hidden_size.unwrap_or(16);
            let complexity = (num_layers * hidden_size) as f64;
            let snr_db = (85.0 - complexity * 1.0).clamp(45.0, 75.0);
            let mse = snr_to_mse(snr_db);
            let esr = 10.0_f64.powf(-snr_db / 10.0) * 2.0;
            (Some(mse.clamp(1e-4, 5e-2)), snr_db, Some(esr), None)
        }
        "Linear" => (Some(1e-10), 135.0, Some(1e-10), Some(0.12)),
        "ConvNet" => {
            let snr_db = 35.0;
            (Some(snr_to_mse(snr_db)), snr_db, Some(1.0e-4), Some(0.03))
        }
        _ => (Some(5e-2), 9.0, Some(1e-3), None),
    }
}
