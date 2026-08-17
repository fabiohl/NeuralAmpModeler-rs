// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Sanitization regression test for the JSONL metric generator.
//!
//! Guards the invariant of Tarefa 1.1: `report_dsp_fidelity*` must never emit
//! JSON `null` in the fundamental numeric metric fields (`esr`, `esr_db`,
//! `snr_db`). `serde_json` serializes non-finite floats (`f64::INFINITY`,
//! `-inf`, `NaN`) as `null`, so a perfect-parity result (SNR = ∞) or a silent
//! signal (ESR = ∞, ESR dB = −∞) would otherwise corrupt the metric stream that
//! `quality-dashboard.sh` consumes — where `null` is coerced to `0.0` (fail-open).
//!
//! The generator maps non-finite values to canonical string sentinels `"inf"`,
//! `"-inf"`, and `"nan"` instead.
//!
//! # Concurrency (S6-T02 / RES-07)
//!
//! The sink is isolated per test thread via [`MetricJsonlGuard`] over a unique
//! temp path, and the process-global `NAM_METRICS_JSONL` env var is never
//! touched. Under `--test-threads > 1` the `--test models` binary runs many
//! reporters (`golden_vectors`, …) on parallel threads; mutating the shared env
//! var would race with their reads and make them append to this test's file,
//! corrupting the exact-two-lines assertion. The guard keeps the file
//! thread-exclusive and removes it on drop — fail-safe even on panic.

use std::path::PathBuf;

use super::common;

use common::validation::{
    MetricJsonlGuard, MetricKindGuard, SuppressReportGuard, report_dsp_fidelity_no_lufs,
};

/// Unique, process-scoped temp path for the JSONL sink. The timestamp suffix
/// keeps it distinct across repeated runs; the PID keeps it distinct across
/// concurrent `cargo test` processes.
fn unique_jsonl_path() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "nam_metrics_jsonl_{}_{nanos}.jsonl",
        std::process::id()
    ))
}

/// RAII guard that scopes the JSONL sink to a fresh unique file and removes it
/// on drop. Drop runs during unwind, so a panic anywhere in the test body
/// cannot leak the temp file or leave a stale sink behind.
struct JsonlFileGuard {
    path: PathBuf,
    _sink: MetricJsonlGuard,
}

impl JsonlFileGuard {
    fn new() -> Self {
        let path = unique_jsonl_path();
        let _ = std::fs::remove_file(&path);
        JsonlFileGuard {
            _sink: MetricJsonlGuard::new(path.clone()),
            path,
        }
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Drop for JsonlFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[test]
fn metrics_jsonl_never_emits_null() {
    let _report_guard = SuppressReportGuard::new();
    let _kind_guard = MetricKindGuard::selftest();

    let n = 2048usize;
    let reference: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let noisy: Vec<f32> = reference
        .iter()
        .enumerate()
        .map(|(i, &r)| r + (i as f32 * 0.0013).cos() * 1e-4)
        .collect();

    let jsonl = JsonlFileGuard::new();
    let tmp = jsonl.path().clone();

    // Perfect parity (identity): snr_db = ∞, esr = 0, esr_db = −∞.
    report_dsp_fidelity_no_lufs(
        &reference,
        &reference,
        None,
        0.0,
        None,
        None,
        "identity-parity-selftest",
        48000,
    );
    // Finite divergence: every fundamental metric must remain a JSON number.
    report_dsp_fidelity_no_lufs(
        &reference,
        &noisy,
        None,
        0.0,
        None,
        None,
        "finite-divergence-selftest",
        48000,
    );

    // The guard is still alive here: the file is read while the sink is
    // thread-exclusive, then removed by `JsonlFileGuard::drop`.
    let content = std::fs::read_to_string(&tmp).expect("JSONL output file must exist");

    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2, "expected two JSONL lines, got:\n{content}");

    let identity: serde_json::Value =
        serde_json::from_str(lines[0]).expect("identity line must be valid JSON");
    for field in ["esr", "esr_db", "snr_db", "mrstft", "mse"] {
        assert!(
            !identity[field].is_null(),
            "identity `{field}` must not be null in: {}",
            lines[0]
        );
    }
    assert_eq!(identity["snr_db"], "inf");
    assert_eq!(identity["esr_db"], "-inf");
    assert_eq!(identity["esr"].as_f64().unwrap(), 0.0);

    let finite: serde_json::Value =
        serde_json::from_str(lines[1]).expect("finite line must be valid JSON");
    for field in ["esr", "esr_db", "snr_db", "mrstft", "mse"] {
        assert!(
            finite[field].is_number() && finite[field].as_f64().unwrap().is_finite(),
            "finite `{field}` must remain a finite JSON number in: {}",
            lines[1]
        );
    }
}

// ── S2.T6 oracle sinks (R-06, slice 1) ──────────────────────────────────────
// One test per `report_*` kind: each emits a serde-valid JSONL line with the
// canonical `kind` and finite numeric fields (non-finite → string sentinels,
// never `null`), following the `metrics_jsonl_never_emits_null` invariant.

#[test]
fn f64_table_sink_emits_valid_jsonl() {
    let jsonl = JsonlFileGuard::new();
    common::validation::report_f64_table(
        "BossWN-standard.nam",
        "BossWN-standard",
        2.31e-14,
        -136.4,
    );

    let content = std::fs::read_to_string(jsonl.path()).expect("JSONL output file must exist");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1, "expected one line, got:\n{content}");
    let obj: serde_json::Value =
        serde_json::from_str(lines[0]).expect("f64_table line must be valid JSON");
    assert_eq!(obj["kind"], "f64_table");
    assert_eq!(obj["filename"], "BossWN-standard.nam");
    assert_eq!(obj["family"], "BossWN-standard");
    assert_eq!(obj["esr"].as_f64().unwrap(), 2.31e-14);
    assert_eq!(obj["esr_db"].as_f64().unwrap(), -136.4);
}

#[test]
fn f64_decomp_sink_emits_valid_jsonl() {
    use neural_amp_modeler_rs::testing::reference_oracle::DecompositionResult;

    let jsonl = JsonlFileGuard::new();
    let result = DecompositionResult {
        label: "BossLSTM-1x16 @48000 Live".to_string(),
        architecture: "LSTM".to_string(),
        esr_f32_vs_f64: 1.02e-14,
        esr_quant_f16c: Some(8.5e-15),
        esr_quant_bf16: None,
        esr_activation: Some(1.2e-16),
        esr_accumulation: None,
        esr_combined: Some(1.05e-14),
    };
    common::validation::report_f64_decomp(&result);

    let content = std::fs::read_to_string(jsonl.path()).expect("JSONL output file must exist");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1, "expected one line, got:\n{content}");
    let obj: serde_json::Value =
        serde_json::from_str(lines[0]).expect("f64_decomp line must be valid JSON");
    assert_eq!(obj["kind"], "f64_decomp");
    assert_eq!(obj["label"], "BossLSTM-1x16 @48000 Live");
    assert_eq!(obj["architecture"], "LSTM");
    assert_eq!(obj["esr_f32_vs_f64"].as_f64().unwrap(), 1.02e-14);
    assert_eq!(obj["esr_quant_f16c"].as_f64().unwrap(), 8.5e-15);
    assert_eq!(obj["esr_activation"].as_f64().unwrap(), 1.2e-16);
    assert_eq!(obj["esr_combined"].as_f64().unwrap(), 1.05e-14);
    assert!(
        obj.get("esr_quant_bf16").is_none() && obj.get("esr_accumulation").is_none(),
        "unmeasured decomposition terms must be omitted, not null: {obj}"
    );
}

#[test]
fn activation_sink_emits_valid_jsonl() {
    let jsonl = JsonlFileGuard::new();
    common::validation::report_activation("LSTM 1×16", 92.5, 96.1);

    let content = std::fs::read_to_string(jsonl.path()).expect("JSONL output file must exist");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1, "expected one line, got:\n{content}");
    let obj: serde_json::Value =
        serde_json::from_str(lines[0]).expect("activation line must be valid JSON");
    assert_eq!(obj["kind"], "activation");
    assert_eq!(obj["model"], "LSTM 1×16");
    assert_eq!(obj["snr_fast_db"].as_f64().unwrap(), 92.5);
    assert_eq!(obj["snr_exact_db"].as_f64().unwrap(), 96.1);
    assert!(
        (obj["gain_db"].as_f64().unwrap() - 3.6).abs() < 1e-12,
        "gain must be snr_exact − snr_fast: {obj}"
    );
}

#[test]
fn isa_sink_emits_valid_jsonl() {
    let jsonl = JsonlFileGuard::new();
    // Cross-ISA pair: full field set.
    common::validation::report_isa(
        "BossWN-standard @ 48000 Hz",
        "AVX2",
        "AVX-512",
        Some(1.4e-15),
        9.9e-10,
        Some(2.1e-5),
        Some(1e-13),
    );
    // Self-consistency: only `mse`; optional fields must be omitted.
    common::validation::report_isa(
        "BossWN-standard @ 48000 Hz",
        "AVX2",
        "AVX2",
        None,
        0.0,
        None,
        None,
    );

    let content = std::fs::read_to_string(jsonl.path()).expect("JSONL output file must exist");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2, "expected two lines, got:\n{content}");

    let cross: serde_json::Value =
        serde_json::from_str(lines[0]).expect("isa line must be valid JSON");
    assert_eq!(cross["kind"], "isa");
    assert_eq!(cross["label"], "BossWN-standard @ 48000 Hz");
    assert_eq!(cross["ref_isa"], "AVX2");
    assert_eq!(cross["test_isa"], "AVX-512");
    assert_eq!(cross["esr"].as_f64().unwrap(), 1.4e-15);
    assert_eq!(cross["mse"].as_f64().unwrap(), 9.9e-10);
    assert_eq!(cross["max_abs_err"].as_f64().unwrap(), 2.1e-5);
    assert_eq!(cross["budget"].as_f64().unwrap(), 1e-13);

    let self_consistency: serde_json::Value =
        serde_json::from_str(lines[1]).expect("isa self-consistency line must be valid JSON");
    assert_eq!(self_consistency["ref_isa"], "AVX2");
    assert_eq!(self_consistency["test_isa"], "AVX2");
    assert_eq!(self_consistency["mse"].as_f64().unwrap(), 0.0);
    for field in ["esr", "max_abs_err", "budget"] {
        assert!(
            self_consistency.get(field).is_none(),
            "self-consistency must omit `{field}`, not null: {self_consistency}"
        );
    }
}
