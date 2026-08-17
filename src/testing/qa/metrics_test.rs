// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests for the JSONL fidelity metrics ingest (F-27).
//!
//! The primary fixture is the canonical block of `utils/tests-long.sh:833-837`,
//! verbatim; asserts mirror the `expect_str` of `:840-856`.

use super::*;

/// Canonical fixture of `utils/tests-long.sh:833-837`, verbatim.
const CANONICAL_JSONL: &str = r#"{"kind":"fidelity","label":"Model A @48000 Live","esr":null,"esr_db":"","snr_db":"1.5e2","mse":"1.2e-5","mrstft":"inf"}
{"kind":"fidelity","label":"Model B","esr":"","esr_db":null,"snr_db":"-inf","mse":null,"mrstft":"nan"}
{"label":"Model C @44100","esr":"0.0001","esr_db":"-40.0","snr_db":"50.0","mse":"3.0e-7","mrstft":"0.001"}
{"label":null,"esr":"1","esr_db":"2","snr_db":"3","mse":"4","mrstft":"5"}"#;

#[test]
fn canonical_fixture_matches_tests_long_expect_str() {
    let records = parse_fidelity_jsonl(CANONICAL_JSONL).expect("canonical JSONL must parse");
    assert_eq!(
        records.len(),
        3,
        "null-label record must be dropped (3 labels)"
    );

    let model_a = &records[0];
    assert_eq!(model_a.label, "Model A @48000 Live");
    assert_eq!(model_a.kind.as_deref(), Some("fidelity"));
    assert_eq!(model_a.esr, MetricValue::Na, "null esr -> N/A");
    assert_eq!(model_a.esr_db, MetricValue::Na, "empty esr_db -> N/A");
    assert_eq!(
        model_a.snr_db,
        MetricValue::Raw("1.5e2".into()),
        "e-notation string preserved"
    );
    assert_eq!(model_a.mse, MetricValue::Raw("1.2e-5".into()));
    assert_eq!(
        model_a.mrstft,
        MetricValue::Raw("inf".into()),
        "non-finite sentinel preserved"
    );
    assert_eq!(
        model_a.esr_f64,
        MetricValue::Na,
        "sink lines carry no esr_f64 (S2.T6)"
    );

    let model_b = &records[1];
    assert_eq!(model_b.label, "Model B");
    assert_eq!(model_b.kind.as_deref(), Some("fidelity"));
    assert_eq!(model_b.esr, MetricValue::Na, "empty esr -> N/A");
    assert_eq!(model_b.esr_db, MetricValue::Na, "null esr_db -> N/A");
    assert_eq!(model_b.snr_db, MetricValue::Raw("-inf".into()));
    assert_eq!(model_b.mse, MetricValue::Na);
    assert_eq!(model_b.mrstft, MetricValue::Raw("nan".into()));

    let model_c = &records[2];
    assert_eq!(model_c.label, "Model C @44100");
    assert_eq!(model_c.kind, None, "record without kind is accepted");
    assert_eq!(model_c.esr, MetricValue::Raw("0.0001".into()));
    assert_eq!(model_c.esr_db, MetricValue::Raw("-40.0".into()));
    assert_eq!(model_c.snr_db, MetricValue::Raw("50.0".into()));
    assert_eq!(model_c.mse, MetricValue::Raw("3.0e-7".into()));
    assert_eq!(model_c.mrstft, MetricValue::Raw("0.001".into()));
}

#[test]
fn is_finite_num_accepts_canonical_forms() {
    for value in [
        "0",
        "0.0",
        "0.5",
        ".5",
        "1.",
        "1.5e-3",
        "-1.5E3",
        "+3.14",
        "3.14e2",
        "42",
        "12345678901234567890",
    ] {
        assert!(is_finite_num(value), "expected '{value}' to be accepted");
    }
}

#[test]
fn is_finite_num_rejects_sentinels_and_garbage() {
    for value in [
        "",
        " ",
        "inf",
        "-inf",
        "+inf",
        "Infinity",
        "-infinity",
        "nan",
        "-nan",
        "NaN",
        "null",
        "N/A",
        "abc",
        "1.2.3",
        "0x10",
        "1e",
        "e5",
        ".",
    ] {
        assert!(!is_finite_num(value), "expected '{value}' to be rejected");
    }
}

#[test]
fn verify_path_is_fail_closed_on_non_finite() {
    let records = parse_fidelity_jsonl(CANONICAL_JSONL).unwrap();
    let model_a = &records[0];
    assert!(
        model_a.mrstft.as_finite().is_none(),
        "'inf' sentinel from JSONL must be rejected"
    );
    assert!(model_a.esr.as_finite().is_none(), "N/A must be rejected");
    assert_eq!(
        model_a.snr_db.as_finite(),
        Some("1.5e2"),
        "e-notation from JSONL must be accepted"
    );
    assert_eq!(
        model_a.snr_db.as_raw(),
        Some("1.5e2"),
        "raw text is preserved verbatim"
    );
}

#[test]
fn non_fidelity_kinds_are_skipped_until_s2_t6() {
    let input = r#"{"kind":"latency","label":"RT_WaveNet_Std_CH16","median_latency_us":36.9}
{"kind":"activation","label":"act","esr":"1"}
{"kind":5,"label":"Weird","esr":"1"}
{"kind":"","label":"EmptyKind","esr":"1"}
{"label":"NoKind","esr":"1"}"#;
    let records = parse_fidelity_jsonl(input).unwrap();
    assert_eq!(records.len(), 1, "only the kindless record survives");
    assert_eq!(records[0].label, "NoKind");
}

#[test]
fn number_metrics_render_like_tsv() {
    let input = r#"{"label":"Numeric","esr":1,"esr_db":1.5,"snr_db":2,"mse":0,"mrstft":7}"#;
    let records = parse_fidelity_jsonl(input).unwrap();
    let record = &records[0];
    assert_eq!(record.esr, MetricValue::Raw("1".into()));
    assert_eq!(record.esr_db, MetricValue::Raw("1.5".into()));
    assert_eq!(record.snr_db, MetricValue::Raw("2".into()));
    assert_eq!(record.mse, MetricValue::Raw("0".into()));
    assert_eq!(record.mrstft, MetricValue::Raw("7".into()));
    assert!(record.mrstft.as_finite().is_some());
}

#[test]
fn missing_metric_fields_normalize_to_na() {
    let input = r#"{"label":"Minimal"}"#;
    let records = parse_fidelity_jsonl(input).unwrap();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.esr, MetricValue::Na);
    assert_eq!(record.esr_db, MetricValue::Na);
    assert_eq!(record.snr_db, MetricValue::Na);
    assert_eq!(record.mse, MetricValue::Na);
    assert_eq!(record.mrstft, MetricValue::Na);
}

#[test]
fn malformed_json_line_fails_closed() {
    let input = "{\"label\":\"Ok\",\"esr\":\"1\"}\nnot json at all\n";
    match parse_fidelity_jsonl(input).unwrap_err() {
        MetricsError::MalformedLine { line, .. } => assert_eq!(line, 2),
        other => panic!("expected MalformedLine, got {other:?}"),
    }
}

#[test]
fn blank_lines_are_ignored() {
    let input = "\n  \n{\"label\":\"Only\"}\n\n";
    let records = parse_fidelity_jsonl(input).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].label, "Only");
}

#[test]
fn literal_null_string_label_is_dropped() {
    let input = r#"{"label":"null","esr":"1"}
{"label":"","esr":"1"}
{"label":"Real","esr":"1"}"#;
    let records = parse_fidelity_jsonl(input).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].label, "Real");
}

#[test]
fn esr_f64_field_is_parsed() {
    let input = r#"{"kind":"fidelity","label":"M","esr":"1","esr_f64":"9.05e-15"}"#;
    let records = parse_fidelity_jsonl(input).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].esr_f64, MetricValue::Raw("9.05e-15".into()));
    assert_eq!(
        records[0].esr_f64.as_finite(),
        Some("9.05e-15"),
        "e-notation f64 ESR must pass the fail-closed accessor"
    );
}

#[test]
fn empty_stream_parses_to_empty() {
    assert!(parse_fidelity_jsonl("").unwrap().is_empty());
}

#[test]
fn file_ingest_round_trip_and_fail_closed() {
    let path = std::env::temp_dir().join(format!("nam_qa_metrics_{}.jsonl", std::process::id()));
    std::fs::write(&path, CANONICAL_JSONL).unwrap();
    let records = parse_fidelity_jsonl_file(&path).unwrap();
    assert_eq!(records.len(), 3);
    let _ = std::fs::remove_file(&path);
    assert!(parse_fidelity_jsonl_file(&path).is_err());
}
