// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests for the human-facing dashboard renderer (S6.T1).
//!
//! - `render_plain_matches_golden`: snapshot test — the plain render of
//!   `tests/fixtures/qa/report.jsonl` must match
//!   `tests/fixtures/qa/render_plain.golden` byte-for-byte (the fixture is a
//!   full report built from the 2026-08-12 snapshot numbers of
//!   `docs/quality-contract.json`).
//! - `regenerate_render_fixture_and_golden` (`#[ignore]`): dev tool — rebuilds
//!   both files from the committed contract. Run with:
//!   `cargo test --features testing --lib qa::render -- --ignored --nocapture`.
//! - `performance_not_verified_is_never_green`: the S6 invariant — a
//!   `regression_gate != PASS` report renders the performance section without
//!   any green escape sequence.

use std::fs;
use std::path::PathBuf;

use serde_json::{Value, json};

use super::*;
use crate::testing::qa::ids;
use crate::testing::qa::verify::{LatencyRecord, PhaseRecord};
use crate::testing::qa::{FidelityEntry, QualityContract};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qa")
        .join(name)
}

fn contract_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/quality-contract.json")
}

fn load_contract() -> QualityContract {
    let content = fs::read_to_string(contract_path()).expect("read docs/quality-contract.json");
    QualityContract::from_json_str(&content).expect("contract must validate")
}

fn fidelity_line(entry: &FidelityEntry) -> String {
    let mut m = serde_json::Map::new();
    m.insert("kind".into(), json!("fidelity"));
    m.insert("label".into(), json!(entry.label));
    m.insert("esr".into(), json!(entry.esr_namcore));
    if let Some(v) = entry.esr_f64 {
        m.insert("esr_f64".into(), json!(v));
    }
    if let Some(v) = entry.snr_db {
        m.insert("snr_db".into(), json!(v));
    }
    m.insert("mrstft".into(), json!(entry.mrstft));
    Value::Object(m).to_string()
}

fn phase_line(phase_id: &str, status: &str) -> String {
    format!(r#"{{"phase_id":"{phase_id}","status":"{status}"}}"#)
}

/// Builds the full report JSONL fixture from the committed contract plus a
/// representative set of extra-kind records (snapshot numbers).
fn build_report_fixture() -> String {
    let contract = load_contract();

    let mut lines: Vec<String> = Vec::new();

    // Provenance header.
    lines.push(
        r#"{"kind":"build_metadata","cargo_profile":"release","target_triple":"x86_64-unknown-linux-gnu","rustflags":"","rustc_version":"rustc 1.97.1 (8bab26f4f 2026-07-14)","git_commit":"0e22ea4ec247","git_dirty_state":true,"run_id":"1786537203076204151-15755","effective_isa":"x86-64-v3 (AVX2/FMA/F16C/BMI)"}"#
            .to_string(),
    );

    // All mandatory phases PASS (performance verified — green allowed).
    // T3.2: the local ISA phase is `isa_self_consistency` (AVX2 vs AVX2) and the
    // cross-ISA matrix is a declared SKIP_CAPABILITY gap on the local runner.
    for (phase, status) in [
        ("golden_vectors", "PASS"),
        ("reference_oracle_f64", "PASS"),
        ("isa_self_consistency", "PASS"),
        ("isa_parity_cross_isa", "SKIP_CAPABILITY"),
        ("spectral_fidelity", "PASS"),
        ("lstm_activation_precision", "PASS"),
        ("quick_parity", "PASS"),
        ("regression_gate", "PASS"),
    ] {
        lines.push(phase_line(phase, status));
    }

    // Fidelity records (all contract entries).
    for entry in &contract.fidelity {
        lines.push(fidelity_line(entry));
    }

    // Latency records keyed by the Criterion bench id (RT_*) of the real
    // `write_latency_stream`, mapped to the contract median latencies via
    // `ids::RT_BENCH_TABLE` (bench id → contract id).
    let median_by_id: std::collections::HashMap<&str, f64> = contract
        .performance
        .iter()
        .map(|e| (e.id.as_str(), e.median_latency_us))
        .collect();
    for entry in ids::RT_BENCH_TABLE {
        if let Some(latency) = median_by_id.get(entry.contract_id) {
            lines.push(format!(
                r#"{{"kind":"latency","label":"{}","median_latency_us":{}}}"#,
                entry.bench_label, latency
            ));
        }
    }

    // f64-oracle summary table rows.
    lines.push(
        r#"{"kind":"f64_table","filename":"BossWN-standard.nam","family":"BossWN-standard","esr":9.05e-15,"esr_db":-140.4}"#
            .to_string(),
    );
    lines.push(
        r#"{"kind":"f64_table","filename":"BossLSTM-1x16.nam","family":"BossLSTM-1x16","esr":8.9e-13,"esr_db":-120.5}"#
            .to_string(),
    );

    // f64-oracle decomposition blocks (cold-start).
    lines.push(
        r#"{"kind":"f64_decomp","label":"BossLSTM-1x16 @48000 Live","architecture":"LSTM","esr_f32_vs_f64":1.02e-14,"esr_quant_f16c":8.5e-15,"esr_activation":1.2e-16,"esr_combined":1.05e-14}"#
            .to_string(),
    );
    lines.push(
        r#"{"kind":"f64_decomp","label":"BossWN-standard @48000 Live","architecture":"WaveNet","esr_f32_vs_f64":9.05e-15,"esr_quant_f16c":6.0e-15,"esr_combined":8.5e-15}"#
            .to_string(),
    );

    // Activation precision rows.
    lines.push(
        r#"{"kind":"activation","model":"LSTM 1×16","snr_fast_db":92.5,"snr_exact_db":96.1,"gain_db":3.6}"#
            .to_string(),
    );
    lines.push(
        r#"{"kind":"activation","model":"LSTM 2×8","snr_fast_db":91.0,"snr_exact_db":94.0,"gain_db":3.0}"#
            .to_string(),
    );

    // ISA parity rows (one cross-ISA pair + one self-consistency).
    lines.push(
        r#"{"kind":"isa","label":"BossWN-standard @ 48000 Hz","ref_isa":"AVX2","test_isa":"AVX-512","esr":1.4e-15,"mse":9.9e-10,"max_abs_err":2.1e-5,"budget":1e-13}"#
            .to_string(),
    );
    lines.push(
        r#"{"kind":"isa","label":"BossWN-standard @ 48000 Hz","ref_isa":"AVX2","test_isa":"AVX2","mse":0.0}"#
            .to_string(),
    );

    // Governance records.
    let f64_oracle = contract
        .fidelity
        .iter()
        .filter(|e| e.esr_f64.is_some())
        .count();
    lines.push(format!(
        r#"{{"kind":"coverage_matrix","namcore_parity":{},"f64_oracle":{},"isa_optimizations":1,"spectral_baselines":{},"rt_performance":{}}}"#,
        contract.fidelity.len(),
        f64_oracle,
        contract.fidelity.len(),
        contract.performance.len()
    ));
    lines.push(
        r#"{"kind":"test_counts","passed":7,"failed":0,"ignored":0,"filtered":0,"skip_capability":0}"#
            .to_string(),
    );

    let mut out = lines.join("\n");
    out.push('\n');
    out
}

// ── Snapshot test ───────────────────────────────────────────────────────────

#[test]
fn render_plain_matches_golden() {
    let fixture = fs::read_to_string(fixture_path("report.jsonl"))
        .expect("report.jsonl fixture must exist (regenerate with the ignored test)");
    let golden = fs::read_to_string(fixture_path("render_plain.golden"))
        .expect("render_plain.golden must exist (regenerate with the ignored test)");

    let report = parse_quality_report(&fixture).expect("fixture report must parse");
    let rendered = render_quality_report(&report, RenderStyle::Plain);

    assert_eq!(
        rendered, golden,
        "plain render drifted from the golden snapshot — run the ignored \
         `regenerate_render_fixture_and_golden` test to refresh both files"
    );
}

/// Dev tool: rebuilds `report.jsonl` + `render_plain.golden` from the
/// committed contract. Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn regenerate_render_fixture_and_golden() {
    let fixture = build_report_fixture();
    let report = parse_quality_report(&fixture).expect("fixture must parse");
    let rendered = render_quality_report(&report, RenderStyle::Plain);

    fs::write(fixture_path("report.jsonl"), &fixture).expect("write report.jsonl");
    fs::write(fixture_path("render_plain.golden"), &rendered).expect("write render_plain.golden");
    eprintln!(
        "regenerated report.jsonl ({} bytes) and render_plain.golden ({} bytes)",
        fixture.len(),
        rendered.len()
    );
}

// ── S6 invariants ───────────────────────────────────────────────────────────

#[test]
fn performance_not_verified_is_never_green() {
    let report = QualityReport {
        header: ReportHeader::default(),
        phases: vec![PhaseRecord {
            phase_id: "regression_gate".into(),
            status: "NOT_VERIFIED".into(),
            observed_records: 0,
            expected_records: 10,
        }],
        fidelity: Vec::new(),
        latency: vec![LatencyRecord {
            label: "RT_LSTM_1x16".into(),
            median_latency_us: 7.5,
        }],
        f64_table: Vec::new(),
        f64_decomp: Vec::new(),
        activation: Vec::new(),
        isa: Vec::new(),
        coverage: None,
        test_counts: None,
    };

    let rendered = render_quality_report(&report, RenderStyle::Ansi);
    assert!(rendered.contains("NOT_VERIFIED"), "{rendered}");

    // The performance section (from its title to the ISA section) must carry
    // no green escape sequence.
    let start = rendered.find("PERFORMANCE — Block Latency").unwrap();
    let end = rendered.find("ISA SELF-CONSISTENCY").unwrap();
    let perf_section = &rendered[start..end];
    assert!(
        !perf_section.contains("\x1b[0;32m"),
        "performance section must never be green when NOT_VERIFIED: {perf_section}"
    );
}

#[test]
fn parse_routes_every_kind() {
    let fixture = build_report_fixture();
    let report = parse_quality_report(&fixture).expect("fixture must parse");

    assert_eq!(
        report.phases.len(),
        8,
        "mandatory phases + declared cross-ISA gap"
    );
    assert_eq!(report.fidelity.len(), 51, "all contract fidelity entries");
    assert_eq!(report.latency.len(), 19, "all RT_* latency records");
    assert_eq!(report.f64_table.len(), 2);
    assert_eq!(report.f64_decomp.len(), 2);
    assert_eq!(report.activation.len(), 2);
    assert_eq!(report.isa.len(), 2);
    assert!(report.coverage.is_some());
    assert!(report.test_counts.is_some());

    assert_eq!(report.header.git_commit, "0e22ea4ec247");
    assert!(report.header.git_dirty);
    assert_eq!(report.header.effective_isa, "x86-64-v3 (AVX2/FMA/F16C/BMI)");
    assert_eq!(report.phase_status("regression_gate"), "PASS");
    assert!(!report.performance_not_verified());
}

fn sample_report_with_decomp(decomp: Vec<F64Decomp>) -> QualityReport {
    QualityReport {
        header: ReportHeader::default(),
        phases: Vec::new(),
        fidelity: Vec::new(),
        latency: Vec::new(),
        f64_table: Vec::new(),
        f64_decomp: decomp,
        activation: Vec::new(),
        isa: Vec::new(),
        coverage: None,
        test_counts: None,
    }
}

#[test]
fn f64_decomp_transient_emits_notice_without_yellow_violation() {
    let report = sample_report_with_decomp(vec![F64Decomp {
        label: "WaveNet-official".to_string(),
        architecture: "WaveNet".to_string(),
        esr_f32_vs_f64: MetricValue::Raw("2.68e-5".to_string()),
        esr_quant_f16c: None,
        esr_quant_bf16: None,
        esr_activation: Some(MetricValue::Raw("2.12e-13".to_string())),
        esr_accumulation: Some(MetricValue::Raw("2.21e-13".to_string())),
        esr_combined: Some(MetricValue::Raw("2.12e-13".to_string())),
    }]);

    let palette = Palette::new(RenderStyle::Ansi);
    let rendered = render_f64_decomposition(&report, &palette);

    assert!(
        rendered.contains("Rule 5 notice:"),
        "must emit notice for WaveNet cold-start: {rendered}"
    );
    assert!(
        rendered.contains("expected cold-start buffer fill-in transient"),
        "must explain cold-start buffer fill-in: {rendered}"
    );
    assert!(
        !rendered.contains("violated"),
        "transient must not say 'violated': {rendered}"
    );
    assert!(
        !rendered.contains("\x1b[1;33m"),
        "transient must not use yellow alert color: {rendered}"
    );
}

#[test]
fn f64_decomp_steady_state_emits_yellow_violation() {
    let report = sample_report_with_decomp(vec![F64Decomp {
        label: "LSTM-1x10".to_string(),
        architecture: "LSTM".to_string(),
        esr_f32_vs_f64: MetricValue::Raw("1.0e-5".to_string()),
        esr_quant_f16c: None,
        esr_quant_bf16: None,
        esr_activation: Some(MetricValue::Raw("1.0e-8".to_string())),
        esr_accumulation: Some(MetricValue::Raw("1.0e-8".to_string())),
        esr_combined: Some(MetricValue::Raw("1.0e-8".to_string())),
    }]);

    let palette = Palette::new(RenderStyle::Ansi);
    let rendered = render_f64_decomposition(&report, &palette);

    assert!(
        rendered.contains("Rule 5 (Σ sources ≈ total, within 10x) violated:"),
        "must emit violation for LSTM steady-state discrepancy: {rendered}"
    );
    assert!(
        rendered.contains("\x1b[1;33m"),
        "violation must use yellow alert color: {rendered}"
    );
}
