// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Machine regression verdict command over Criterion comparison artifacts.

use std::fs;
use std::path::Path;
use std::process::exit;

use serde::Serialize;

use neural_amp_modeler_rs::testing::qa::coverage::expected_bench_ids_for_suite;
use neural_amp_modeler_rs::testing::qa::verdict::{
    VerdictBlind, VerdictFinding, VerdictOptions, regression_verdict_with_options,
};

use crate::cli::{
    HELP_REQUEST, default_criterion_root, parse_f64, parse_flags_full, parse_noise, print_help,
    required, run_error, usage_error,
};

// ── verdict ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct VerdictJsonReport {
    schema_version: &'static str,
    executed_count: usize,
    noise_band_relative: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    abs_noise_floor_ns: Option<f64>,
    verdict: &'static str,
    summary: VerdictJsonSummary,
    findings: Vec<VerdictJsonFinding>,
}

#[derive(Debug, Serialize)]
struct VerdictJsonSummary {
    regressions: usize,
    unstable_baselines: usize,
    sub_threshold_noise: usize,
    clean: usize,
}

#[derive(Debug, Serialize)]
struct VerdictJsonFinding {
    id: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    relative_mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ci_95: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta_ns: Option<f64>,
    reason: String,
}

/// Prints the machine regression verdict over Criterion's persisted
/// `change/estimates.json` artifacts (R-3/B2).
///
/// Exit codes (Section 4):
/// - 0: SUCCESS (no regressions, sub-threshold noise ignored, clean run);
/// - 1: REGRESSION_DETECTED (at least one machine regression above noise & floor);
/// - 2: UNSTABLE_BASELINE (mean improvement exceeds stability band under --strict-stability);
/// - 3: REGRESSION_BLIND (missing/unparseable log or missing Criterion comparison estimates).
pub(crate) fn cmd_verdict(args: &[String]) {
    let flags = match parse_flags_full(
        args,
        &[
            "log",
            "root",
            "noise",
            "suite",
            "manifest",
            "abs-noise-floor-ns",
        ],
        &[
            "check-stability",
            "strict-stability",
            "confirm-regressions",
            "json",
        ],
    ) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_help();
            exit(0);
        }
        Err(e) => usage_error("verdict", &e),
    };
    let log_path = match required(&flags, "log") {
        Ok(v) => v,
        Err(e) => usage_error("verdict", &e),
    };
    let criterion_root = flags
        .get("root")
        .map(str::to_string)
        .unwrap_or_else(default_criterion_root);
    let noise = parse_noise(&flags, "noise", "verdict");
    let strict_stability = flags.get_bool("strict-stability");
    let check_stability = strict_stability || flags.get_bool("check-stability");
    let abs_noise_floor_ns = parse_f64(&flags, "abs-noise-floor-ns", "verdict");
    let confirm_regressions = flags.get_bool("confirm-regressions");
    let json_output = flags.get_bool("json");

    // An absent/unreadable log maps to empty text — the same blind gate as
    // the bash `grep` on a missing file (fail-closed, R-3).
    let log_text = fs::read_to_string(log_path).unwrap_or_default();

    let expected_ids = if let Some(suite) = flags.get("suite") {
        let manifest_path = flags.get("manifest").map(Path::new);
        let ids = match expected_bench_ids_for_suite(suite, manifest_path) {
            Ok(ids) => ids,
            Err(e) => run_error(e),
        };
        if suite != "regression_gate" && !json_output {
            println!("(alternative scope: {suite})");
        }
        Some(ids)
    } else {
        None
    };

    let options = VerdictOptions {
        noise,
        expected_ids,
        check_stability,
        abs_noise_floor_ns,
    };

    let outcome_result =
        regression_verdict_with_options(&log_text, Path::new(&criterion_root), &options);

    let outcome = match outcome_result {
        Err(VerdictBlind) => {
            if json_output {
                let report = VerdictJsonReport {
                    schema_version: "1.1.0",
                    executed_count: 0,
                    noise_band_relative: noise,
                    abs_noise_floor_ns,
                    verdict: "FAILED",
                    summary: VerdictJsonSummary {
                        regressions: 0,
                        unstable_baselines: 0,
                        sub_threshold_noise: 0,
                        clean: 0,
                    },
                    findings: vec![VerdictJsonFinding {
                        id: String::new(),
                        status: "REGRESSION_BLIND",
                        relative_mean: None,
                        ci_95: None,
                        delta_ns: None,
                        reason: format!("no executed benchmark could be parsed from {log_path}"),
                    }],
                };
                if let Ok(json) = serde_json::to_string_pretty(&report) {
                    println!("{json}");
                }
            } else {
                println!("REGRESSION_BLIND no executed benchmark could be parsed from {log_path}");
                println!("The machine regression verdict is blind: nothing passes unverified.");
            }
            exit(3);
        }
        Ok(o) => o,
    };

    let flagged_file = Path::new(&criterion_root).join("flagged_regressions.txt");
    let mut flakes_cleared = Vec::new();
    if confirm_regressions {
        if flagged_file.exists() {
            let previously_flagged: std::collections::HashSet<String> =
                fs::read_to_string(&flagged_file)
                    .unwrap_or_default()
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(String::from)
                    .collect();
            let _ = fs::remove_file(&flagged_file);

            let current_reg_ids: std::collections::HashSet<String> =
                outcome.regressions().map(|f| f.id().to_string()).collect();
            for prev_id in &previously_flagged {
                if !current_reg_ids.contains(prev_id) {
                    flakes_cleared.push(prev_id.clone());
                }
            }
        } else {
            let reg_ids: Vec<&str> = outcome.regressions().map(|f| f.id()).collect();
            if !reg_ids.is_empty() {
                let text = reg_ids.join("\n") + "\n";
                let _ = fs::write(&flagged_file, text);
                if !json_output {
                    println!(
                        "[CONFIRMATION] Flagged {} regression(s) to {}",
                        reg_ids.len(),
                        flagged_file.display()
                    );
                }
            }
        }
    }

    let blind_count = outcome.blind_findings().count();
    let regression_count = outcome.regressions().count();
    let unstable_count = outcome.unstable_baselines().count();
    let noise_ignored_count = outcome.sub_threshold_noise().count();

    let has_failed =
        blind_count > 0 || regression_count > 0 || (strict_stability && unstable_count > 0);
    let verdict_str = if has_failed { "FAILED" } else { "PASSED" };
    let exit_code = if blind_count > 0 {
        3
    } else if regression_count > 0 {
        1
    } else if strict_stability && unstable_count > 0 {
        2
    } else {
        0
    };

    if json_output {
        let json_findings: Vec<VerdictJsonFinding> = outcome
            .findings
            .iter()
            .map(|f| match f {
                VerdictFinding::MissingExpectedBenchmark { id } => VerdictJsonFinding {
                    id: id.clone(),
                    status: "REGRESSION_BLIND",
                    relative_mean: None,
                    ci_95: None,
                    delta_ns: None,
                    reason: format!("expected benchmark {id} was not executed in {log_path}"),
                },
                VerdictFinding::MissingEstimates { id } => VerdictJsonFinding {
                    id: id.clone(),
                    status: "REGRESSION_BLIND",
                    relative_mean: None,
                    ci_95: None,
                    delta_ns: None,
                    reason: format!(
                        "missing comparison artifact at {criterion_root}/{id}/change/estimates.json"
                    ),
                },
                VerdictFinding::Regression {
                    id,
                    point_estimate,
                    lower_bound,
                    upper_bound,
                    delta_ns,
                    ..
                } => VerdictJsonFinding {
                    id: id.clone(),
                    status: "REGRESSION_DETECTED",
                    relative_mean: Some(*point_estimate),
                    ci_95: Some([*lower_bound, *upper_bound]),
                    delta_ns: *delta_ns,
                    reason: "Relative mean change lies entirely above noise band".to_string(),
                },
                VerdictFinding::UnstableBaseline {
                    id,
                    point_estimate,
                    lower_bound,
                    upper_bound,
                    delta_ns,
                    ..
                } => VerdictJsonFinding {
                    id: id.clone(),
                    status: "UNSTABLE_BASELINE",
                    relative_mean: Some(*point_estimate),
                    ci_95: Some([*lower_bound, *upper_bound]),
                    delta_ns: *delta_ns,
                    reason: "Mean improvement exceeds bilateral stability band".to_string(),
                },
                VerdictFinding::NoiseIgnored {
                    id,
                    point_estimate,
                    lower_bound,
                    upper_bound,
                    delta_ns,
                    floor_ns,
                } => VerdictJsonFinding {
                    id: id.clone(),
                    status: "NOISE_IGNORED",
                    relative_mean: Some(*point_estimate),
                    ci_95: Some([*lower_bound, *upper_bound]),
                    delta_ns: Some(*delta_ns),
                    reason: format!("Absolute delta within {:.1}ns noise floor", floor_ns),
                },
            })
            .collect();

        let summary = VerdictJsonSummary {
            regressions: regression_count,
            unstable_baselines: unstable_count,
            sub_threshold_noise: noise_ignored_count,
            clean: outcome
                .executed
                .saturating_sub(regression_count + unstable_count + noise_ignored_count),
        };
        let report = VerdictJsonReport {
            schema_version: "1.1.0",
            executed_count: outcome.executed,
            noise_band_relative: noise,
            abs_noise_floor_ns,
            verdict: verdict_str,
            summary,
            findings: json_findings,
        };
        match serde_json::to_string_pretty(&report) {
            Ok(s) => println!("{s}"),
            Err(e) => run_error(format!("cannot serialize verdict json: {e}")),
        }
        exit(exit_code);
    }

    for id in &flakes_cleared {
        println!("FLAKE_CLEARED {id} cleared upon re-execution");
    }

    for finding in &outcome.findings {
        match finding {
            VerdictFinding::MissingExpectedBenchmark { id } => {
                println!("REGRESSION_BLIND expected benchmark {id} was not executed in {log_path}");
            }
            VerdictFinding::MissingEstimates { id } => {
                println!(
                    "REGRESSION_BLIND {id} has no readable Criterion comparison \
                     artifact at {criterion_root}/{id}/change/estimates.json"
                );
            }
            VerdictFinding::Regression {
                id,
                point_estimate,
                lower_bound,
                upper_bound,
                noise,
                ..
            } => {
                println!(
                    "REGRESSION_DETECTED {id} relative mean {:+}% \
                     (CI [{:+}%, {:+}%]) lies entirely above the +{}% noise band",
                    point_estimate * 100.0,
                    lower_bound * 100.0,
                    upper_bound * 100.0,
                    noise * 100.0
                );
            }
            VerdictFinding::UnstableBaseline {
                id,
                point_estimate,
                lower_bound,
                upper_bound,
                noise,
                ..
            } => {
                println!(
                    "UNSTABLE_BASELINE {id} relative mean {:+}% \
                     (CI [{:+}%, {:+}%]) exceeds stability threshold (±{:.2}%)",
                    point_estimate * 100.0,
                    lower_bound * 100.0,
                    upper_bound * 100.0,
                    noise * 100.0
                );
            }
            VerdictFinding::NoiseIgnored {
                id,
                point_estimate,
                delta_ns,
                floor_ns,
                ..
            } => {
                println!(
                    "NOISE_IGNORED {id} relative mean {:+}% within absolute noise floor ({:.2} ns <= {:.2} ns)",
                    point_estimate * 100.0,
                    delta_ns,
                    floor_ns
                );
            }
        }
    }

    if exit_code == 0 {
        println!(
            "verdict ok: no machine regression among {} executed benchmark(s) \
             (noise band ±{:.2}%)",
            outcome.executed,
            noise * 100.0
        );
    }
    exit(exit_code);
}
