// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for contract verification and hardware compatibility checks (R-12, R-23, S6-T03).

use super::*;
use crate::testing::qa::{Envelopes, EsrNamcoreEnvelope, Provenance};

fn mock_contract_with_hardware(hw_class: Option<&str>) -> QualityContract {
    QualityContract {
        schema_version: 1,
        schema_notes: None,
        generated_at: "2026-10-29T00:00:00Z".to_string(),
        provenance: Provenance {
            git_commit: "abcdef".to_string(),
            git_dirty: false,
            run_id: "test-run".to_string(),
            effective_isa: "x86-64-v3 (AVX2/FMA/F16C/BMI)".to_string(),
            cpu_model: "AMD Ryzen 7 5700U with Radeon Graphics".to_string(),
            rustc: "rustc 1.99.0".to_string(),
            cargo_profile: "release".to_string(),
            hardware_class: hw_class.map(|s| s.to_string()),
        },
        envelopes: Envelopes {
            esr_namcore: EsrNamcoreEnvelope {
                noise_mult: 3.0,
                noise_floor_abs: 5e-14,
                safety_mult: 10.0,
                safety_floor_abs: 1e-12,
            },
            snr_db_drop: 6.0,
            mrstft_mult: 10.0,
            latency_mult: 1.1,
            latency_floor_us: 0.05,
        },
        fidelity: Vec::new(),
        performance: vec![PerformanceEntry {
            id: "test_bench".to_string(),
            label: "Test Bench".to_string(),
            median_latency_us: 10.0,
            batch_factor: None,
            unit: None,
        }],
        monitoring_alerts: None,
    }
}

fn mock_report_with_meta(
    reg_gate_status: &str,
    target_triple: &str,
    effective_isa: &str,
    hw_class: Option<&str>,
) -> VerifyReport {
    VerifyReport {
        phases: vec![
            PhaseRecord {
                phase_id: "golden_vectors".to_string(),
                status: "PASS".to_string(),
                observed_records: 1,
                expected_records: 1,
            },
            PhaseRecord {
                phase_id: "reference_oracle_f64".to_string(),
                status: "PASS".to_string(),
                observed_records: 1,
                expected_records: 1,
            },
            PhaseRecord {
                phase_id: "quick_parity".to_string(),
                status: "PASS".to_string(),
                observed_records: 1,
                expected_records: 1,
            },
            PhaseRecord {
                phase_id: REGRESSION_GATE_PHASE.to_string(),
                status: reg_gate_status.to_string(),
                observed_records: 1,
                expected_records: 1,
            },
        ],
        fidelity: Vec::new(),
        latency: vec![LatencyRecord {
            label: "Test Bench".to_string(),
            median_latency_us: 9.5,
        }],
        build_metadata: Some(BuildMetadataRecord {
            target_triple: target_triple.to_string(),
            effective_isa: effective_isa.to_string(),
            git_commit: "abcdef".to_string(),
            run_id: "test-run".to_string(),
            cargo_profile: "release".to_string(),
            hardware_class: hw_class.map(|s| s.to_string()),
        }),
    }
}

#[test]
fn test_verify_contract_hardware_class_compatible_ok() {
    let contract = mock_contract_with_hardware(Some("x86_zen3_laptop"));
    let report = mock_report_with_meta(
        "PASS",
        "x86_64-unknown-linux-gnu",
        "x86-64-v3 (AVX2/FMA/F16C/BMI)",
        Some("x86_zen3_laptop"),
    );
    let outcome = verify_contract(&contract, &report);
    assert_eq!(outcome.performance, PerformanceVerdict::Ok);
}

#[test]
fn test_verify_contract_hardware_class_mismatch_yields_not_verified() {
    let contract = mock_contract_with_hardware(Some("x86_zen3_laptop"));
    let report = mock_report_with_meta(
        "PASS",
        "x86_64-unknown-linux-gnu",
        "x86-64-v3 (AVX2/FMA/F16C/BMI)",
        Some("aarch64_apple_silicon"),
    );
    let outcome = verify_contract(&contract, &report);
    assert_eq!(outcome.performance, PerformanceVerdict::NotVerified);
}

#[test]
fn test_verify_contract_isa_mismatch_yields_not_verified() {
    let contract = mock_contract_with_hardware(Some("x86_zen3_laptop"));
    let report = mock_report_with_meta(
        "PASS",
        "aarch64-unknown-linux-gnu",
        "aarch64 (NEON)",
        Some("aarch64_linux_server"),
    );
    let outcome = verify_contract(&contract, &report);
    assert_eq!(outcome.performance, PerformanceVerdict::NotVerified);
}

#[test]
fn test_verify_contract_missing_build_metadata_yields_not_verified() {
    let contract = mock_contract_with_hardware(Some("x86_zen3_laptop"));
    let mut report = mock_report_with_meta(
        "PASS",
        "x86_64-unknown-linux-gnu",
        "x86-64-v3 (AVX2/FMA/F16C/BMI)",
        Some("x86_zen3_laptop"),
    );
    report.build_metadata = None;
    let outcome = verify_contract(&contract, &report);
    assert_eq!(outcome.performance, PerformanceVerdict::NotVerified);
}

fn with_freshness(mut report: VerifyReport, status: &str) -> VerifyReport {
    report.phases.push(PhaseRecord {
        phase_id: PROVENANCE_PHASE.to_string(),
        status: status.to_string(),
        observed_records: 1,
        expected_records: 1,
    });
    report
}

#[test]
fn test_verify_contract_freshness_pass_reports_provenance_ok() {
    let contract = mock_contract_with_hardware(None);
    let report = with_freshness(
        mock_report_with_meta(
            "PASS",
            "x86_64-unknown-linux-gnu",
            "x86-64-v3 (AVX2/FMA/F16C/BMI)",
            None,
        ),
        "PASS",
    );
    let outcome = verify_contract(&contract, &report);
    assert_eq!(outcome.provenance, ProvenanceVerdict::Ok);
    // The mock contract carries no fidelity entries, so the fidelity domain is
    // independently red — but provenance must not be the reason.
    assert_ne!(outcome.fidelity, FidelityVerdict::NotVerified);
}

#[test]
fn test_verify_contract_freshness_fail_is_provenance_not_fidelity() {
    let contract = mock_contract_with_hardware(None);
    let report = with_freshness(
        mock_report_with_meta(
            "PASS",
            "x86_64-unknown-linux-gnu",
            "x86-64-v3 (AVX2/FMA/F16C/BMI)",
            None,
        ),
        "FAIL",
    );
    let outcome = verify_contract(&contract, &report);
    assert_eq!(outcome.provenance, ProvenanceVerdict::Fail);
    assert_eq!(outcome.fidelity, FidelityVerdict::NotVerified);
    assert!(!outcome.fidelity.is_ok());
}

#[test]
fn test_verify_contract_freshness_fail_masks_fidelity_fail() {
    let contract = mock_contract_with_hardware(None);
    let mut report = mock_report_with_meta(
        "PASS",
        "x86_64-unknown-linux-gnu",
        "x86-64-v3 (AVX2/FMA/F16C/BMI)",
        None,
    );
    report.phases[0].status = "FAIL".to_string();
    let outcome = verify_contract(&contract, &with_freshness(report, "FAIL"));
    assert_eq!(outcome.provenance, ProvenanceVerdict::Fail);
    assert_eq!(outcome.fidelity, FidelityVerdict::NotVerified);
}

#[test]
fn test_verify_contract_absent_freshness_is_not_run() {
    let contract = mock_contract_with_hardware(None);
    let report = mock_report_with_meta(
        "PASS",
        "x86_64-unknown-linux-gnu",
        "x86-64-v3 (AVX2/FMA/F16C/BMI)",
        None,
    );
    let outcome = verify_contract(&contract, &report);
    assert_eq!(outcome.provenance, ProvenanceVerdict::NotRun);
    assert_ne!(outcome.fidelity, FidelityVerdict::NotVerified);
}
