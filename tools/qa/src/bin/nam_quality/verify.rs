// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Contract verification, report rendering, and regression classification commands.

use std::fs;
use std::io::{self, IsTerminal};
use std::process::exit;

use neural_amp_modeler_rs::testing::qa::QualityContract;
use neural_amp_modeler_rs::testing::qa::classify::{
    classify_fresh_regression, classify_regression_outcome,
};
use neural_amp_modeler_rs::testing::qa::render::{
    RenderStyle, parse_quality_report_file, render_quality_report,
};
use neural_amp_modeler_rs::testing::qa::verify::{
    FidelityOutcome, FidelityVerdict, MalformedReason, Metric as MetricId, MetricOutcome,
    PROVENANCE_PHASE, PerfResult, PerformanceVerdict, ProvenanceVerdict, VerifyOutcome,
    parse_verify_report_file, verify_contract,
};

use crate::cli::{HELP_REQUEST, parse_flags, required, run_error, usage_error};

/// Mandatory fidelity phases of the dashboard suite — a FAIL here refuses
/// `save` and fails the fidelity domain of `verify`. This is deliberately a
/// subset of the long-suite [`neural_amp_modeler_rs::testing::receipt::FIDELITY_PHASE_IDS`]:
/// the two suites cover different phase rosters, so the identifiers are not
/// interchangeable.
const FIDELITY_PHASES: [&str; 3] = ["golden_vectors", "reference_oracle_f64", "quick_parity"];

pub(crate) fn cmd_verify(args: &[String]) {
    let flags = match parse_flags(
        args,
        &["contract", "report", "fidelity-only"],
        &["fidelity-only"],
    ) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_verify_help();
            exit(0);
        }
        Err(e) => usage_error("verify", &e),
    };
    let contract_path = match required(&flags, "contract") {
        Ok(v) => v,
        Err(e) => usage_error("verify", &e),
    };
    let report_path = match required(&flags, "report") {
        Ok(v) => v,
        Err(e) => usage_error("verify", &e),
    };
    let fidelity_only = flags.get("fidelity-only").is_some();
    let exit_code = execute_verify(contract_path, report_path, fidelity_only);
    exit(exit_code);
}

pub(crate) fn execute_verify(contract_path: &str, report_path: &str, fidelity_only: bool) -> i32 {
    let contract = match fs::read_to_string(contract_path)
        .map_err(|e| format!("cannot read contract {contract_path}: {e}"))
        .and_then(|s| QualityContract::from_json_str(&s).map_err(|e| e.to_string()))
    {
        Ok(c) => c,
        Err(e) => run_error(e),
    };
    let report = match parse_verify_report_file(report_path) {
        Ok(r) => r,
        Err(e) => run_error(format!("cannot parse verify report {report_path}: {e}")),
    };

    let outcome = verify_contract(&contract, &report);
    print!(
        "{}",
        render_outcome(&contract, &report, &outcome, fidelity_only)
    );
    if fidelity_only {
        outcome.fidelity_exit_code()
    } else {
        outcome.exit_code()
    }
}

fn print_verify_help() {
    println!("Usage: nam_quality verify --contract <path> --report <path> [--fidelity-only]");
    println!();
    println!("Runs the literal port of the bash `verify_contract` against");
    println!("docs/quality-contract.json (JSON-only) and a JSONL report produced");
    println!("by `nam_quality ingest` (or by the test fixtures).");
    println!();
    println!("Options:");
    println!("  --fidelity-only   verifies only the fidelity domain (mandatory phases &");
    println!("                    metric envelopes); unverified performance does not fail.");
    println!();
    println!("Prints the verdict lines (`FIDELITY: OK/FAIL`, `PERFORMANCE: OK/");
    println!("FAIL/NOT_VERIFIED`, `CONTRACT VIOLATED`) and exits 0 only when the active");
    println!("domains pass and no oracle review is pending; exit 1 otherwise.");
}

/// Renders the verify outcome as plain text, mirroring the dashboard verdict
/// strings (`FIDELITY: OK`, `PERFORMANCE: NOT_VERIFIED`, `CONTRACT VIOLATED`,
/// `REVIEW_REQUIRED`) that scripts and operators grep for.
fn render_outcome(
    contract: &QualityContract,
    report: &neural_amp_modeler_rs::testing::qa::verify::VerifyReport,
    outcome: &VerifyOutcome,
    fidelity_only: bool,
) -> String {
    let mut out = String::new();
    out.push_str("QUALITY CONTRACT VERIFICATION\n");
    if fidelity_only {
        out.push_str("  [MODE] FIDELITY-ONLY VERIFICATION (latency checks skipped)\n");
    }
    out.push('\n');

    for phase in &report.phases {
        out.push_str(&format!("  phase {}: {}\n", phase.phase_id, phase.status));
    }
    for phase_id in FIDELITY_PHASES {
        let status = report
            .phases
            .iter()
            .rev()
            .find(|p| p.phase_id == phase_id)
            .map(|p| p.status.as_str())
            .unwrap_or("NOT_RUN");
        if status != "PASS" {
            out.push_str(&format!(
                "  PHASE_FAILED {phase_id}: status={status} (requires PASS)\n"
            ));
        }
    }
    for phase in &report.phases {
        if phase.status == "FAIL"
            && !FIDELITY_PHASES.contains(&phase.phase_id.as_str())
            && phase.phase_id != "regression_gate"
            && phase.phase_id != PROVENANCE_PHASE
        {
            out.push_str(&format!(
                "  PHASE_FAILED {}: status=FAIL (observed={}, expected={})\n",
                phase.phase_id, phase.observed_records, phase.expected_records
            ));
        }
    }
    out.push('\n');

    out.push_str(&format!(
        "  FIDELITY — {} model(s) in contract\n",
        contract.fidelity.len()
    ));
    for check in &outcome.fidelity_checks {
        match &check.outcome {
            FidelityOutcome::OptionalSkipped => {
                out.push_str(&format!(
                    "    ok {}: OPTIONAL_SKIPPED (absent from report)\n",
                    check.label
                ));
            }
            FidelityOutcome::MissingLabel => {
                out.push_str(&format!(
                    "    FAIL {}: MISSING_LABEL — mandatory contract entry not found in report\n",
                    check.label
                ));
            }
            FidelityOutcome::Measured(metrics) => {
                for m in metrics {
                    match &m.outcome {
                        MetricOutcome::Ok => {}
                        MetricOutcome::SafetyCeiling {
                            current,
                            limit,
                            baseline,
                        } => out.push_str(&format!(
                            "    FAIL {}: {} above safety ceiling (current={current:.3e}, limit={limit:.3e}, baseline={baseline:.3e})\n",
                            check.label,
                            metric_name(m.metric)
                        )),
                        MetricOutcome::NoiseEnvelope {
                            current,
                            limit,
                            baseline,
                        } => out.push_str(&format!(
                            "    FAIL {}: {} above noise envelope (current={current:.3e}, limit={limit:.3e}, baseline={baseline:.3e})\n",
                            check.label,
                            metric_name(m.metric)
                        )),
                        MetricOutcome::Envelope {
                            current,
                            limit,
                            baseline,
                        } => out.push_str(&format!(
                            "    FAIL {}: {} out of envelope (current={current:.3e}, limit={limit:.3e}, baseline={baseline:.3e})\n",
                            check.label,
                            metric_name(m.metric)
                        )),
                        MetricOutcome::Malformed(reason) => out.push_str(&format!(
                            "    FAIL {}: {} malformed in report ({})\n",
                            check.label,
                            metric_name(m.metric),
                            malformed_reason(reason)
                        )),
                        MetricOutcome::Missing => out.push_str(&format!(
                            "    FAIL {}: {} missing from report\n",
                            check.label,
                            metric_name(m.metric)
                        )),
                    }
                }
            }
        }
    }
    out.push('\n');

    out.push_str(&format!(
        "  PERFORMANCE — {} benchmark(s) in contract\n",
        contract.performance.len()
    ));
    if fidelity_only {
        out.push_str("    [SKIPPED] performance checks skipped in fidelity-only mode\n");
    } else {
        for check in &outcome.perf_checks {
            match &check.result {
                PerfResult::Ok { median_us } => {
                    out.push_str(&format!(
                        "    ok {}: median={median_us:.3} µs\n",
                        check.label
                    ));
                }
                PerfResult::Regressed {
                    median_us,
                    limit_us,
                    baseline_us,
                } => {
                    out.push_str(&format!(
                        "    FAIL {}: latency regressed (current={median_us:.3} µs, limit={limit_us:.3} µs, baseline={baseline_us:.3} µs)\n",
                        check.label
                    ));
                }
                PerfResult::MissingLabel => {
                    out.push_str(&format!(
                        "    FAIL {}: MISSING_LABEL — contract entry not found in report\n",
                        check.label
                    ));
                }
                PerfResult::MissingLatency => {
                    out.push_str(&format!(
                        "    FAIL {}: MISSING_LATENCY — benchmark carried no latency data\n",
                        check.label
                    ));
                }
            }
        }
    }
    out.push('\n');

    match outcome.provenance {
        ProvenanceVerdict::Ok => out.push_str("  PROVENANCE: OK\n"),
        ProvenanceVerdict::Fail => out.push_str("  PROVENANCE: FAIL\n"),
        ProvenanceVerdict::NotRun => out.push_str("  PROVENANCE: NOT_RUN\n"),
    }
    match outcome.fidelity {
        FidelityVerdict::Ok => out.push_str("  FIDELITY: OK\n"),
        FidelityVerdict::NotVerified => out.push_str("  FIDELITY: NOT_VERIFIED\n"),
        FidelityVerdict::Fail { violations } => {
            out.push_str(&format!("  FIDELITY: FAIL ({violations} violation(s))\n"));
            if outcome.review_required > 0 {
                out.push_str(&format!(
                    "  [GOVERNANCE] REVIEW_REQUIRED — NAMCore vs f64 divergence detected on {} model(s).\n",
                    outcome.review_required
                ));
            }
        }
    }
    if fidelity_only {
        out.push_str("  PERFORMANCE: NOT_VERIFIED (skipped in fidelity-only mode)\n");
    } else {
        match outcome.performance {
            PerformanceVerdict::Ok => out.push_str("  PERFORMANCE: OK\n"),
            PerformanceVerdict::NotVerified => {
                out.push_str("  PERFORMANCE: NOT_VERIFIED\n");
            }
            PerformanceVerdict::Fail { violations } => {
                out.push_str(&format!(
                    "  PERFORMANCE: FAIL ({violations} violation(s))\n"
                ));
            }
        }
    }

    if outcome.fidelity.is_ok()
        && (outcome.performance.is_ok() || fidelity_only)
        && outcome.review_required > 0
    {
        out.push_str(
            "  CONTRACT UNDER REVIEW — numeric metrics OK, but oracle divergence needs investigation.\n",
        );
        out.push('\n');
    } else {
        let is_violated = if fidelity_only {
            outcome.fidelity_exit_code() != 0
        } else {
            outcome.exit_code() != 0
        };
        if is_violated {
            out.push_str("  CONTRACT VIOLATED\n");
            out.push('\n');
        }
    }
    out
}

fn metric_name(metric: MetricId) -> &'static str {
    match metric {
        MetricId::EsrNamcore => "ESR_NAMCORE",
        MetricId::EsrF64 => "ESR_F64",
        MetricId::SnrDb => "SNR_DB",
        MetricId::Mrstft => "MRSTFT",
    }
}

fn malformed_reason(reason: &MalformedReason) -> String {
    match reason {
        MalformedReason::Missing => "N/A or empty in report".to_string(),
        MalformedReason::NonFinite(raw) => format!("non-finite literal {raw}"),
        MalformedReason::Inconsistent(detail) => format!("inconsistent metrics: {detail}"),
    }
}

// ── render ──────────────────────────────────────────────────────────────────

pub(crate) fn cmd_render(args: &[String]) {
    let flags = match parse_flags(args, &["report", "ansi", "plain"], &["ansi", "plain"]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_render_help();
            exit(0);
        }
        Err(e) => usage_error("render", &e),
    };
    let report_path = match required(&flags, "report") {
        Ok(v) => v,
        Err(e) => usage_error("render", &e),
    };
    let style = match (flags.get("ansi").is_some(), flags.get("plain").is_some()) {
        (true, true) => usage_error("render", "--ansi and --plain are mutually exclusive"),
        (_, true) => RenderStyle::Plain,
        (true, _) => RenderStyle::Ansi,
        _ if io::stdout().is_terminal() => RenderStyle::Ansi,
        _ => RenderStyle::Plain,
    };

    let report = match parse_quality_report_file(report_path) {
        Ok(r) => r,
        Err(e) => run_error(format!("cannot parse report {report_path}: {e}")),
    };
    print!("{}", render_quality_report(&report, style));
    exit(0);
}

fn print_render_help() {
    println!("Usage: nam_quality render --report <path> [--ansi | --plain]");
    println!();
    println!("Renders the human-facing quality dashboard from a JSONL report");
    println!("(the output of `nam_quality ingest`). The report is typed and");
    println!("never re-derived — the renderer only formats it.");
    println!();
    println!("  --report <path>  verify report (JSONL) produced by `ingest` — required.");
    println!("  --ansi           ANSI color escape sequences (default on a TTY).");
    println!("  --plain          plain text (default when stdout is not a TTY).");
}

// ── classify ────────────────────────────────────────────────────────────────

pub(crate) fn cmd_classify(args: &[String]) {
    let flags = match parse_flags(args, &["status", "reason", "reg-exit", "run-id-match"], &[]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_classify_help();
            exit(0);
        }
        Err(e) => usage_error("classify", &e),
    };
    let status = match required(&flags, "status") {
        Ok(v) => v,
        Err(e) => usage_error("classify", &e),
    };
    let reason = flags.get("reason").unwrap_or("");

    // Fresh-run classification — `--reg-exit <n>` and
    // `--run-id-match <0|1>` make the classifier stale-receipt-proof. When
    // omitted (legacy callers) the 3-way F-08 classifier is used.
    let reg_exit = flags.get("reg-exit");
    let run_id_match = flags.get("run-id-match");
    let outcome = match (reg_exit, run_id_match) {
        (Some(exit_str), Some(match_str)) => {
            let exit_code: i32 = match exit_str.parse() {
                Ok(n) => n,
                Err(_) => usage_error("classify", "--reg-exit must be an integer"),
            };
            let matches = match match_str {
                "1" | "true" => true,
                "0" | "false" => false,
                _ => usage_error("classify", "--run-id-match must be 0 or 1"),
            };
            classify_fresh_regression(exit_code, status, reason, matches)
        }
        (None, None) => classify_regression_outcome(status, reason),
        _ => usage_error(
            "classify",
            "--reg-exit and --run-id-match must be provided together",
        ),
    };
    println!("{}", outcome.as_str());
    exit(0);
}

fn print_classify_help() {
    println!("Usage: nam_quality classify --status <STATUS> --reason <text>");
    println!("                           [--reg-exit <n> --run-id-match <0|1>]");
    println!();
    println!("Classifies a regression receipt into the single performance status");
    println!("(`qa::classify`, F-08): PASS / NOT_VERIFIED / FAIL.");
    println!();
    println!("  --status <STATUS>  receipt status (e.g. PASS, FAIL, NOT_RUN).");
    println!("  --reason <text>    receipt reason (e.g. MISSING_BASELINE,");
    println!("                      INCOMPARABLE_ENVIRONMENT, REGRESSION_DETECTED).");
    println!("  --reg-exit <n>     exit code of the regression runner.");
    println!("  --run-id-match     whether the receipt's run_id equals the current");
    println!("                     RUN_ID (0|1). Together they make the classifier");
    println!("                     stale-receipt-proof: a PASS from a previous run");
    println!("                     never validates a failed/aborted current run.");
}
