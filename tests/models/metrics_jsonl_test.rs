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

use super::common;

use common::validation::{MetricKindGuard, SuppressReportGuard, report_dsp_fidelity_no_lufs};

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

    let tmp = std::env::temp_dir().join(format!("nam_metrics_jsonl_{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    unsafe {
        std::env::set_var("NAM_METRICS_JSONL", &tmp);
    }

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

    unsafe {
        std::env::remove_var("NAM_METRICS_JSONL");
    }

    let content = std::fs::read_to_string(&tmp).expect("JSONL output file must exist");
    let _ = std::fs::remove_file(&tmp);

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
