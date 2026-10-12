// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::fs;
use std::path::{Path, PathBuf};

use super::*;

fn make_temp_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "nam-test-criterion-{test_name}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_mock_estimates(dir: &Path, median_ns: f64, mean_ns: f64, slope_ns: f64) -> String {
    let json = serde_json::json!({
        "mean": {
            "confidence_interval": {
                "confidence_level": 0.95,
                "lower_bound": mean_ns * 0.99,
                "upper_bound": mean_ns * 1.01
            },
            "point_estimate": mean_ns,
            "standard_error": 1.0
        },
        "median": {
            "confidence_interval": {
                "confidence_level": 0.95,
                "lower_bound": median_ns * 0.99,
                "upper_bound": median_ns * 1.01
            },
            "point_estimate": median_ns,
            "standard_error": 1.0
        },
        "slope": {
            "confidence_interval": {
                "confidence_level": 0.95,
                "lower_bound": slope_ns * 0.99,
                "upper_bound": slope_ns * 1.01
            },
            "point_estimate": slope_ns,
            "standard_error": 1.0
        }
    });
    let s = json.to_string();
    fs::write(dir.join("estimates.json"), &s).expect("write estimates.json");
    s
}

#[test]
fn test_synthetic_distinct_estimates_extracts_median_only() {
    // Distinct statistics:
    // median = 30_000 ns (30.0 µs)
    // mean   = 35_000 ns (35.0 µs)
    // slope  = 40_000 ns (40.0 µs)
    let json_str = serde_json::json!({
        "mean": { "point_estimate": 35_000.0 },
        "median": { "point_estimate": 30_000.0 },
        "slope": { "point_estimate": 40_000.0 }
    })
    .to_string();

    let extracted_ns = parse_criterion_median_ns(&json_str, "synthetic").unwrap();
    assert_eq!(extracted_ns, 30_000.0);
    let us = extracted_ns / 1_000.0;
    assert_eq!(us, 30.0);
    assert_ne!(us, 35.0, "must not extract mean");
    assert_ne!(us, 40.0, "must not extract slope");
}

#[test]
fn test_standard_case_reproduction() {
    // Real Standard values observed in target/criterion/RT_WaveNet_Std_CH16/new/estimates.json:
    // median = 41809.7858 ns (41.8097858 µs)
    // mean   = 42102.9353 ns (42.1029353 µs)
    // slope  = 42675.2513 ns (42.6752513 µs)
    let json_str = serde_json::json!({
        "mean": { "point_estimate": 42102.9353 },
        "median": { "point_estimate": 41809.7858 },
        "slope": { "point_estimate": 42675.2513 }
    })
    .to_string();

    let extracted_ns = parse_criterion_median_ns(&json_str, "standard").unwrap();
    let us = extracted_ns / 1_000.0;
    // Before rounding, produces 41.8097858 µs
    assert!((us - 41.8097858).abs() < 1e-6);
    // Rounded to 4 decimal places, this is 41.8098 µs
    assert_eq!(format!("{us:.4}"), "41.8098");
}

#[test]
fn test_micro_batch_factor_division() {
    // 64 blocks batched, total median = 64_000 ns (64.0 µs)
    // per-block median should be 1.0 µs
    let json_str = serde_json::json!({
        "median": { "point_estimate": 64_000.0 }
    })
    .to_string();

    let extracted_ns = parse_criterion_median_ns(&json_str, "resampler").unwrap();
    let batch_factor = resolve_batch_factor("RT_DSP_Resampler_44k1_to_48k", 64, None).unwrap();
    assert_eq!(batch_factor, 64);

    let per_block_us = (extracted_ns / 1_000.0) / (batch_factor as f64);
    assert!((per_block_us - 1.0).abs() < 1e-9);
}

#[test]
fn test_missing_field_and_null_errors() {
    let missing_median = serde_json::json!({
        "mean": { "point_estimate": 1000.0 }
    })
    .to_string();
    assert_eq!(
        parse_criterion_median_ns(&missing_median, "test_missing"),
        Err(CriterionExtractionError::MissingMedian {
            path: "test_missing".to_string()
        })
    );

    let null_point = serde_json::json!({
        "median": { "point_estimate": null }
    })
    .to_string();
    assert_eq!(
        parse_criterion_median_ns(&null_point, "test_null"),
        Err(CriterionExtractionError::MissingMedian {
            path: "test_null".to_string()
        })
    );
}

#[test]
fn test_non_finite_median_errors() {
    let nan_val = serde_json::json!({
        "median": { "point_estimate": f64::NAN }
    })
    .to_string();
    // In serde_json, NaN becomes null, which triggers MissingMedian or NonFinite
    let res = parse_criterion_median_ns(&nan_val, "test_nan");
    assert!(res.is_err());

    let inf_str = r#"{"median":{"point_estimate":1e999}}"#;
    let res_inf = parse_criterion_median_ns(inf_str, "test_inf");
    assert!(res_inf.is_err());
}

#[test]
fn test_missing_file_error() {
    let temp = make_temp_dir("missing_file");
    let err = extract_criterion_latencies(&temp, 64, None).unwrap_err();
    match err {
        CriterionExtractionError::MissingFile { path } => {
            assert!(path.contains("RT_WaveNet_Std_CH16"));
        }
        other => panic!("expected MissingFile error, got {other:?}"),
    }
    let _ = fs::remove_dir_all(&temp);
}

#[test]
fn test_full_table_extraction() {
    let temp = make_temp_dir("full_table");
    for bench in RT_BENCH_TABLE {
        let bench_dir = temp.join(bench.bench_label).join("new");
        fs::create_dir_all(&bench_dir).unwrap();
        write_mock_estimates(&bench_dir, 10_000.0, 11_000.0, 12_000.0);
    }

    let entries = extract_criterion_latencies(&temp, 64, None).unwrap();
    assert_eq!(entries.len(), RT_BENCH_TABLE.len());

    for entry in &entries {
        assert_eq!(entry.kind, "latency");
        if MICRO_BATCH_BENCH_LABELS.contains(&entry.label.as_str()) {
            // 10_000 ns = 10 µs / 64
            assert!((entry.median_latency_us - (10.0 / 64.0)).abs() < 1e-6);
        } else {
            // 10_000 ns = 10 µs / 1
            assert!((entry.median_latency_us - 10.0).abs() < 1e-6);
        }
    }

    let jsonl = criterion_latencies_to_jsonl(&entries).unwrap();
    assert_eq!(jsonl.lines().count(), RT_BENCH_TABLE.len());

    let _ = fs::remove_dir_all(&temp);
}

#[test]
fn test_parse_bench_constants_env() {
    let content = "# generated by build.rs\nDSP_MICRO_BATCH=64\nOTHER=123\n";
    assert_eq!(parse_bench_constants_env(content), Some(64));

    let quoted = "DSP_MICRO_BATCH=\"32\"\n";
    assert_eq!(parse_bench_constants_env(quoted), Some(32));

    let invalid = "DSP_MICRO_BATCH=0\n";
    assert_eq!(parse_bench_constants_env(invalid), None);

    let empty = "";
    assert_eq!(parse_bench_constants_env(empty), None);
}
