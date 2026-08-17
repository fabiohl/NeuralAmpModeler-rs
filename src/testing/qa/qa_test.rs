// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Smoke tests for the typed quality-contract schema.
//!
//! The primary fixture is the illustrative example payload, kept verbatim
//! (including the abbreviated commit and the placeholder run id) so the
//! round trip proves the schema matches the agreed contract.

use super::*;

/// Example quality-contract payload, verbatim.
const R01_EXAMPLE: &str = r#"{
  "schema_version": 1,
  "generated_at": "2026-08-12T12:20:03-03:00",
  "provenance": {
    "git_commit": "0e22ea4ec247…",
    "git_dirty": true,
    "run_id": "…",
    "effective_isa": "x86-64-v3 (AVX2/FMA/F16C/BMI)",
    "cpu_model": "AMD Ryzen 7 5700U with Radeon Graphics",
    "rustc": "rustc 1.97.1 (8bab26f4f 2026-07-14)",
    "cargo_profile": "release"
  },
  "envelopes": {
    "esr_namcore": { "noise_mult": 3.0, "noise_floor_abs": 5e-14, "safety_mult": 10.0, "safety_floor_abs": 1e-12 },
    "snr_db_drop": 6.0,
    "mrstft_mult": 10.0,
    "latency_mult": 1.10,
    "latency_floor_us": 0.05
  },
  "fidelity": [
    { "id": "bosswn-standard@48000:live", "label": "BossWN-standard @48000 Live", "esr_namcore": 2.31e-14, "esr_f64": 9.05e-15, "snr_db": 136.4, "mrstft": 6.46e-06, "optional": false }
  ],
  "performance": [
    { "id": "RT_WaveNet_Std_CH16", "label": "WaveNet Standard CH16", "median_latency_us": 36.9 }
  ]
}"#;

#[test]
fn round_trip_r01_example() {
    let contract = QualityContract::from_json_str(R01_EXAMPLE).expect("R-01 example must parse");
    assert_eq!(contract.schema_version, SCHEMA_VERSION);
    assert_eq!(contract.provenance.git_commit, "0e22ea4ec247…");
    assert!(contract.provenance.git_dirty);
    assert_eq!(contract.provenance.run_id, "…");
    assert_eq!(
        contract.provenance.effective_isa,
        "x86-64-v3 (AVX2/FMA/F16C/BMI)"
    );
    assert_eq!(
        contract.provenance.cpu_model,
        "AMD Ryzen 7 5700U with Radeon Graphics"
    );
    assert_eq!(
        contract.provenance.rustc,
        "rustc 1.97.1 (8bab26f4f 2026-07-14)"
    );
    assert_eq!(contract.provenance.cargo_profile, "release");

    let env = &contract.envelopes;
    assert_eq!(env.esr_namcore.noise_mult, 3.0);
    assert_eq!(env.esr_namcore.noise_floor_abs, 5e-14);
    assert_eq!(env.esr_namcore.safety_mult, 10.0);
    assert_eq!(env.esr_namcore.safety_floor_abs, 1e-12);
    assert_eq!(env.snr_db_drop, 6.0);
    assert_eq!(env.mrstft_mult, 10.0);
    assert_eq!(env.latency_mult, 1.10);
    assert_eq!(env.latency_floor_us, 0.05);

    assert_eq!(contract.fidelity.len(), 1);
    let fidelity = &contract.fidelity[0];
    assert_eq!(fidelity.id, "bosswn-standard@48000:live");
    assert_eq!(fidelity.label, "BossWN-standard @48000 Live");
    assert_eq!(fidelity.esr_namcore, 2.31e-14);
    assert_eq!(fidelity.esr_f64, Some(9.05e-15));
    assert_eq!(fidelity.snr_db, Some(136.4));
    assert_eq!(fidelity.mrstft, 6.46e-06);
    assert!(!fidelity.optional);

    assert_eq!(contract.performance.len(), 1);
    assert_eq!(contract.performance[0].id, "RT_WaveNet_Std_CH16");
    assert_eq!(contract.performance[0].label, "WaveNet Standard CH16");
    assert_eq!(contract.performance[0].median_latency_us, 36.9);

    let pretty = contract
        .to_json_pretty()
        .expect("pretty serialization must succeed");
    assert!(pretty.starts_with('{'));
    assert!(pretty.contains("\n  \"schema_version\": 1,"));
    assert!(pretty.contains("\n    \"esr_namcore\": {"));
    assert!(
        serde_json::from_str::<serde_json::Value>(&pretty).is_ok(),
        "pretty output must remain valid JSON"
    );
    let reparsed = QualityContract::from_json_str(&pretty).expect("round trip must parse");
    assert_eq!(reparsed, contract);
}

#[test]
fn missing_and_null_optionals_are_tolerated() {
    let json = r#"{
  "schema_version": 1,
  "generated_at": "2026-08-12T12:20:03-03:00",
  "provenance": {
    "git_commit": "c",
    "git_dirty": false,
    "run_id": "r",
    "effective_isa": "isa",
    "cpu_model": "cpu",
    "rustc": "rustc 1.97.1",
    "cargo_profile": "release"
  },
  "envelopes": {
    "esr_namcore": { "noise_mult": 3.0, "noise_floor_abs": 5e-14, "safety_mult": 10.0, "safety_floor_abs": 1e-12 },
    "snr_db_drop": 6.0,
    "mrstft_mult": 10.0,
    "latency_mult": 1.10,
    "latency_floor_us": 0.05
  },
  "fidelity": [
    { "id": "a", "label": "A", "esr_namcore": 1.0, "esr_f64": null, "snr_db": null, "mrstft": 1.0 }
  ],
  "performance": []
}"#;
    let contract = QualityContract::from_json_str(json).expect("null optionals must parse");
    let fidelity = &contract.fidelity[0];
    assert_eq!(fidelity.esr_f64, None);
    assert_eq!(fidelity.snr_db, None);
    assert!(
        !fidelity.optional,
        "missing `optional` must default to false"
    );
}

/// Parses an awk `%.17g` reference literal (exact double round-trip).
fn awk_ref(literal: &str) -> f64 {
    literal.parse().expect("awk reference literal must parse")
}

#[test]
fn policy_v1_is_the_measured_policy_not_the_stale_comments() {
    let policy = Envelopes::policy_v1();
    let esr = &policy.esr_namcore;
    assert_eq!(
        esr.noise_mult, 3.0,
        "noise is 3x, not the 10x of the stale bash comment"
    );
    assert_eq!(esr.noise_floor_abs, 5e-14);
    assert_eq!(
        esr.safety_mult, 10.0,
        "safety is 10x, not the 100x of the stale bash comment"
    );
    assert_eq!(esr.safety_floor_abs, 1e-12);
    assert_eq!(policy.snr_db_drop, 6.0);
    assert_eq!(policy.mrstft_mult, 10.0);
    assert_eq!(policy.latency_mult, 1.10);
    assert_eq!(policy.latency_floor_us, 0.05);
}

#[test]
fn policy_v1_esr_envelopes_match_awk_reference() {
    let policy = Envelopes::policy_v1();
    // Reference values generated with the exact awk of verify_contract
    // (utils/quality-dashboard.sh:2241-2250), printed at %.17g:
    //   noise:  awk -v c="$b" 'BEGIN { lim = c*3.0; if (lim < c+5.0e-14) lim = c+5.0e-14; printf "%.17g", lim }'
    //   safety: awk -v c="$b" 'BEGIN { lim = c*10.0; if (lim < 1.0e-12) lim = 1.0e-12; printf "%.17g", lim }'
    // Baselines are rows of docs/quality-contract.txt (snapshot 2026-08-12).
    let cases: &[(f64, &str, &str, &str)] = &[
        (
            2.31e-14,
            "BossWN-standard @48000 Live (NAMCore)",
            "7.3099999999999999e-14",
            "9.9999999999999998e-13",
        ),
        (
            9.05e-15,
            "BossWN-standard @48000 Live (f64)",
            "5.9049999999999998e-14",
            "9.9999999999999998e-13",
        ),
        (
            1.20e-13,
            "wavenet_a1_standard (NAMCore)",
            "3.5999999999999998e-13",
            "1.1999999999999999e-12",
        ),
        (
            5.03e-11,
            "WaveNet A2 Dynamic Gated (NAMCore)",
            "1.5089999999999999e-10",
            "5.0300000000000002e-10",
        ),
        (
            1.00e-10,
            "WaveNet A2 Dynamic Gated (f64)",
            "3e-10",
            "1.0000000000000001e-09",
        ),
        (
            8.90e-13,
            "LSTM 1x16 (f64)",
            "2.6700000000000001e-12",
            "8.8999999999999996e-12",
        ),
        (
            3.66e-15,
            "lstm_3x8 (NAMCore)",
            "5.3660000000000002e-14",
            "9.9999999999999998e-13",
        ),
    ];
    for (baseline, row, noise_ref, safety_ref) in cases {
        let baseline = *baseline;
        let noise = (baseline * policy.esr_namcore.noise_mult)
            .max(baseline + policy.esr_namcore.noise_floor_abs);
        let safety =
            (baseline * policy.esr_namcore.safety_mult).max(policy.esr_namcore.safety_floor_abs);
        assert_eq!(noise, awk_ref(noise_ref), "ESR noise envelope: {row}");
        assert_eq!(safety, awk_ref(safety_ref), "ESR safety envelope: {row}");
    }
}

#[test]
fn policy_v1_snr_mrstft_latency_limits_match_awk_reference() {
    let policy = Envelopes::policy_v1();
    // SNR: awk 'BEGIN { if (cur+0 < ctr-6.0) print "1" }' (quality-dashboard.sh:2345).
    let snr_cases: &[(f64, &str, &str)] = &[
        (136.4, "BossWN-standard @48000 Live", "130.40000000000001"),
        (110.7, "BossLSTM-1x16 @48000 Live", "104.7"),
        (144.4, "lstm_3x8 @48000 Live", "138.40000000000001"),
    ];
    for (baseline, row, limit_ref) in snr_cases {
        assert_eq!(
            *baseline - policy.snr_db_drop,
            awk_ref(limit_ref),
            "SNR limit: {row}"
        );
    }

    // MR-STFT: awk 'BEGIN { if (cur+0 > ctr*10.0) print "1" }' (quality-dashboard.sh:2367).
    let mrstft_cases: &[(f64, &str, &str)] = &[
        (
            6.46e-06,
            "BossWN-standard @48000 Live",
            "6.4599999999999998e-05",
        ),
        (
            6.64e-05,
            "WaveNet A2 Dynamic Gated",
            "0.00066399999999999999",
        ),
        (5.55e-07, "lstm_3x8 @48000 Live", "5.5499999999999994e-06"),
    ];
    for (baseline, row, limit_ref) in mrstft_cases {
        assert_eq!(
            *baseline * policy.mrstft_mult,
            awk_ref(limit_ref),
            "MR-STFT limit: {row}"
        );
    }

    // Latency: awk 'BEGIN { limit = ctr*1.10; if (limit < ctr+0.05) limit = ctr+0.05; if (cur+0 > limit) print "1" }'
    // (quality-dashboard.sh:2425).
    let latency_cases: &[(f64, &str, &str)] = &[
        (36.9, "WaveNet Standard CH16", "40.590000000000003"),
        (52.6, "WaveNet Lite CH12", "57.860000000000007"),
        (150.6, "DSP Pipeline HQ (4x OS)", "165.66"),
        (0.3, "Linear RF=2048", "0.34999999999999998"),
    ];
    for (baseline, row, limit_ref) in latency_cases {
        let limit = (*baseline * policy.latency_mult).max(*baseline + policy.latency_floor_us);
        assert_eq!(limit, awk_ref(limit_ref), "latency limit: {row}");
    }
}

#[test]
fn rejects_unsupported_schema_version() {
    let json = r#"{
  "schema_version": 2,
  "generated_at": "2026-08-12T12:20:03-03:00",
  "provenance": {
    "git_commit": "c",
    "git_dirty": false,
    "run_id": "r",
    "effective_isa": "isa",
    "cpu_model": "cpu",
    "rustc": "rustc 1.97.1",
    "cargo_profile": "release"
  },
  "envelopes": {
    "esr_namcore": { "noise_mult": 3.0, "noise_floor_abs": 5e-14, "safety_mult": 10.0, "safety_floor_abs": 1e-12 },
    "snr_db_drop": 6.0,
    "mrstft_mult": 10.0,
    "latency_mult": 1.10,
    "latency_floor_us": 0.05
  },
  "fidelity": [],
  "performance": []
}"#;
    match QualityContract::from_json_str(json) {
        Err(QualityContractError::UnsupportedSchemaVersion { actual, expected }) => {
            assert_eq!(actual, 2);
            assert_eq!(expected, SCHEMA_VERSION);
        }
        other => panic!("expected UnsupportedSchemaVersion, got {other:?}"),
    }
}

#[test]
fn committed_quality_contract_json_loads_and_matches_snapshot_counts() {
    let json = std::fs::read_to_string("docs/quality-contract.json")
        .expect("docs/quality-contract.json must exist (S1.T3 transcription)");
    let contract = QualityContract::from_json_str(&json)
        .expect("committed contract must validate against the schema");
    assert_eq!(
        contract.fidelity.len(),
        51,
        "fidelity count (34 canonical + 17 coverage)"
    );
    assert_eq!(
        contract.performance.len(),
        19,
        "latency count (14 core + 5 DSP)"
    );
    let optional: Vec<&FidelityEntry> = contract.fidelity.iter().filter(|f| f.optional).collect();
    assert_eq!(optional.len(), 1, "optional:true only on EVH-5150-Lite");
    assert!(optional[0].id.contains("evh-5150-lite"));
    assert_eq!(contract.schema_version, SCHEMA_VERSION);
    assert_eq!(contract.provenance.git_commit, "0e22ea4ec247");
    assert!(contract.provenance.git_dirty);
}
