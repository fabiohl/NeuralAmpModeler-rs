// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_ln_gamma() {
    // Gamma(1) = 1 => ln(Gamma(1)) = 0
    assert!((ln_gamma(1.0)).abs() < 1e-7);
    // Gamma(2) = 1 => ln(Gamma(2)) = 0
    assert!((ln_gamma(2.0)).abs() < 1e-7);
    // Gamma(3) = 2 => ln(Gamma(3)) = ln(2)
    assert!((ln_gamma(3.0) - 2.0f64.ln()).abs() < 1e-7);
    // Gamma(4) = 6 => ln(Gamma(4)) = ln(6)
    assert!((ln_gamma(4.0) - 6.0f64.ln()).abs() < 1e-7);
}

#[test]
fn test_student_t_p_value() {
    // For high t value and large df, p-value should be extremely small (< 0.001)
    let p = student_t_two_tailed_p_value(5.0, 50.0);
    assert!(p < 0.001, "Expected p < 0.001, got {p}");

    // For t = 0, p-value should be 1.0
    let p_zero = student_t_two_tailed_p_value(0.0, 50.0);
    assert!(
        (p_zero - 1.0).abs() < 1e-5,
        "Expected p ≈ 1.0, got {p_zero}"
    );

    // Known critical value: for df=50, t ≈ 2.008 gives p ≈ 0.05
    let p_crit = student_t_two_tailed_p_value(2.008, 50.0);
    assert!(
        (p_crit - 0.05).abs() < 0.01,
        "Expected p ≈ 0.05, got {p_crit}"
    );
}

#[test]
fn test_welch_t_test() {
    let s1 = SampleStats {
        mean_ns: 10000.0,
        std_dev_ns: 200.0,
        sample_count: 50,
        variance_ns2: 40000.0,
    };
    let s2 = SampleStats {
        mean_ns: 8000.0, // 20% speedup
        std_dev_ns: 200.0,
        sample_count: 50,
        variance_ns2: 40000.0,
    };
    let p = welch_t_test_p_value(&s1, &s2);
    assert!(
        p < 0.0001,
        "Expected highly significant difference, got p = {p}"
    );
}

#[test]
fn test_parse_sample_json() {
    let json = r#"{"iters":[100,100,100],"times":[10000.0,10200.0,9800.0]}"#;
    let stats = parse_sample_json(json).expect("Failed to parse sample json");
    assert_eq!(stats.sample_count, 3);
    assert!((stats.mean_ns - 100.0).abs() < 1e-6);
}
