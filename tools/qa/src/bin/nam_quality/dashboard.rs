// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Quality dashboard orchestrator command.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

use serde_json::Value;

use neural_amp_modeler_rs::math::common::SIMD_MATH;
use neural_amp_modeler_rs::testing::qa::classify::{RegressionOutcome, classify_fresh_regression};
use neural_amp_modeler_rs::testing::qa::env::EnvProbe;
use neural_amp_modeler_rs::testing::qa::ids;
use neural_amp_modeler_rs::testing::qa::render::{
    RenderStyle, parse_quality_report_file, render_quality_report,
};

use crate::cli::{HELP_REQUEST, parse_flags, run_error, usage_error};
use crate::ingest::{read_jsonl_lines, run_ingest};
use crate::receipt::{PhaseReceiptRecord, append_line};
use crate::save::save_dashboard_contract;
use crate::third_party::{
    check_third_party, detect_active_isa, host_has_full_avx512, run_dashboard_phase,
};
use crate::verify::execute_verify;

pub(crate) fn cmd_dashboard(args: &[String]) {
    // `--check` without a value means the canonical contract, so the shell
    // wrapper stays a pure `exec` (constitution §4.1).
    let mut args = args.to_vec();
    if let Some(i) = args.iter().position(|a| a == "--check")
        && args.get(i + 1).is_none_or(|v| v.starts_with("--"))
    {
        args.insert(i + 1, "docs/quality-contract.json".to_string());
    }
    if let Some(i) = args.iter().position(|a| a == "--save")
        && let Some(next) = args.get(i + 1)
        && !next.starts_with("--")
    {
        let contract = args.remove(i + 1);
        args.push("--contract".to_string());
        args.push(contract);
    }
    let flags = match parse_flags(
        &args,
        &[
            "check",
            "save",
            "contract",
            "fidelity-only",
            "bench-core",
            "run-id",
            "log-dir",
            "receipt",
            "metrics",
            "criterion-dir",
            "ansi",
            "plain",
        ],
        &["save", "fidelity-only", "ansi", "plain"],
    ) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_dashboard_help();
            exit(0);
        }
        Err(e) => usage_error("dashboard", &e),
    };
    if flags.get("ansi").is_some() && flags.get("plain").is_some() {
        usage_error("dashboard", "--ansi and --plain are mutually exclusive");
    }

    let contract_path = flags
        .get("check")
        .or_else(|| flags.get("contract"))
        .unwrap_or("docs/quality-contract.json");
    let log_dir_str = flags.get("log-dir").unwrap_or("target/logs");
    let log_dir = PathBuf::from(log_dir_str);
    if let Err(e) = fs::create_dir_all(&log_dir) {
        run_error(format!("cannot create log dir {}: {e}", log_dir.display()));
    }
    let receipt_path = flags.get("receipt").map(PathBuf::from).unwrap_or_else(|| {
        let p = log_dir.join("dashboard_phase_receipt.jsonl");
        let _ = fs::remove_file(&p);
        p
    });
    let metrics_path = flags.get("metrics").map(PathBuf::from).unwrap_or_else(|| {
        let p = log_dir.join("fidelity_metrics.jsonl");
        let _ = fs::remove_file(&p);
        p
    });
    let report_path = log_dir.join("quality_report.jsonl");
    let _ = fs::remove_file(&report_path);

    let run_id = flags
        .get("run-id")
        .map(String::from)
        .or_else(|| env::var("NAM_RUN_ID").ok())
        .unwrap_or_else(|| {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            format!("{ts}-{}", std::process::id())
        });

    let receipt_path_str = receipt_path.to_string_lossy().to_string();
    let metrics_path_str = metrics_path.to_string_lossy().to_string();
    let report_path_str = report_path.to_string_lossy().to_string();

    let existing_receipt_lines = read_jsonl_lines(&receipt_path_str).unwrap_or_default();
    let mut existing_phases: HashMap<String, String> = HashMap::new();
    for line in &existing_receipt_lines {
        if let Ok(v) = serde_json::from_str::<Value>(line)
            && v.get("run_id").and_then(Value::as_str) == Some(&run_id)
            && let (Some(pid), Some(st)) = (
                v.get("phase_id").and_then(Value::as_str),
                v.get("status").and_then(Value::as_str),
            )
        {
            existing_phases.insert(pid.to_string(), st.to_string());
        }
    }

    // ── Phase 0: freshness & third-party integrity ──
    if !existing_phases.contains_key("freshness") {
        let freshness_log = log_dir.join("freshness.log");
        run_dashboard_phase(
            "freshness",
            &[(
                &[
                    "cargo",
                    "run",
                    "--quiet",
                    "--locked",
                    "-p",
                    "nam-qa",
                    "--bin",
                    "nam_freshness",
                    "--",
                    "hard-fail",
                ],
                &[],
            )],
            &freshness_log,
            &receipt_path_str,
            &run_id,
            1,
            1,
        );
    }
    if !existing_phases.contains_key("third_party") {
        let check = check_third_party();
        let rec = PhaseReceiptRecord {
            phase_id: "third_party",
            status: check.status,
            exit_code: 0,
            observed_records: check.observed_records,
            expected_records: 1,
            reason: &check.reason,
            run_id: &run_id,
        };
        let _ = append_line(&receipt_path_str, &serde_json::to_string(&rec).unwrap());
    }

    // ── Phase 1: golden_vectors ──
    if !existing_phases.contains_key("golden_vectors") {
        let gv_log = log_dir.join("golden_vectors.log");
        run_dashboard_phase(
            "golden_vectors",
            &[(
                &[
                    "cargo",
                    "test",
                    "--release",
                    "--features",
                    "testing",
                    "--test",
                    "models",
                    "golden_vectors",
                    "--",
                    "--include-ignored",
                    "--test-threads=1",
                    "--nocapture",
                ],
                &[("NAM_METRICS_JSONL", &metrics_path_str)],
            )],
            &gv_log,
            &receipt_path_str,
            &run_id,
            50,
            1,
        );
    }

    // ── Phase 2: reference_oracle_f64 ──
    if !existing_phases.contains_key("reference_oracle_f64") {
        let f64_log = log_dir.join("reference_oracle_f64.log");
        run_dashboard_phase(
            "reference_oracle_f64",
            &[
                (
                    &[
                        "cargo",
                        "test",
                        "--release",
                        "--features",
                        "testing",
                        "--test",
                        "parity",
                        "reference_oracle_f64",
                        "--",
                        "--test-threads=1",
                        "--nocapture",
                    ],
                    &[("NAM_METRICS_JSONL", &metrics_path_str)],
                ),
                (
                    &[
                        "cargo",
                        "test",
                        "--release",
                        "--features",
                        "testing",
                        "--test",
                        "parity",
                        "--",
                        "--include-ignored",
                        "--exact",
                        "reference_oracle_f64::test_summary_table",
                        "--nocapture",
                    ],
                    &[("NAM_METRICS_JSONL", &metrics_path_str)],
                ),
            ],
            &f64_log,
            &receipt_path_str,
            &run_id,
            10,
            10,
        );
    }

    // ── Phase 3: isa_self_consistency ──
    if !existing_phases.contains_key("isa_self_consistency") {
        let isa_log = log_dir.join("isa_self_consistency.log");
        run_dashboard_phase(
            "isa_self_consistency",
            &[
                (
                    &[
                        "cargo",
                        "test",
                        "--release",
                        "--features",
                        "testing",
                        "--test",
                        "parity",
                        "isa_parity::isa_self_consistency",
                        "--",
                        "--include-ignored",
                        "--test-threads=1",
                        "--nocapture",
                    ],
                    &[("NAM_METRICS_JSONL", &metrics_path_str)],
                ),
                (
                    &[
                        "cargo",
                        "test",
                        "--release",
                        "--features",
                        "testing",
                        "--test",
                        "parity",
                        "isa_parity::isa_hf_self_consistency",
                        "--",
                        "--include-ignored",
                        "--test-threads=1",
                        "--nocapture",
                    ],
                    &[("NAM_METRICS_JSONL", &metrics_path_str)],
                ),
            ],
            &isa_log,
            &receipt_path_str,
            &run_id,
            10,
            10,
        );
    }
    if !existing_phases.contains_key("isa_parity_cross_isa") {
        if host_has_full_avx512() {
            let cross_log = log_dir.join("isa_parity_cross_isa.log");
            run_dashboard_phase(
                "isa_parity_cross_isa",
                &[(
                    &[
                        "cargo",
                        "test",
                        "--release",
                        "--features",
                        "testing,avx512",
                        "--test",
                        "parity",
                        "avx2_vs_avx512",
                        "--",
                        "--include-ignored",
                        "--test-threads=1",
                        "--nocapture",
                    ],
                    &[("NAM_METRICS_JSONL", &metrics_path_str)],
                )],
                &cross_log,
                &receipt_path_str,
                &run_id,
                10,
                10,
            );
        } else {
            let cross_rec = PhaseReceiptRecord {
                phase_id: "isa_parity_cross_isa",
                status: "SKIP_CAPABILITY",
                exit_code: 0,
                observed_records: 0,
                expected_records: 0,
                reason: neural_amp_modeler_rs::testing::qa::phases::CROSS_ISA_GAP_REASON,
                run_id: &run_id,
            };
            let _ = append_line(
                &receipt_path_str,
                &serde_json::to_string(&cross_rec).unwrap(),
            );
        }
    }

    // ── Phase 4: spectral_fidelity ──
    if !existing_phases.contains_key("spectral_fidelity") {
        let spec_log = log_dir.join("spectral_fidelity.log");
        run_dashboard_phase(
            "spectral_fidelity",
            &[
                (
                    &[
                        "cargo",
                        "test",
                        "--release",
                        "--features",
                        "testing",
                        "--test",
                        "models",
                        "spectral_fidelity",
                        "--",
                        "--nocapture",
                    ],
                    &[],
                ),
                (
                    &[
                        "cargo",
                        "test",
                        "--release",
                        "--features",
                        "testing",
                        "--test",
                        "models",
                        "spectral_fidelity::model_baselines::baseline_",
                        "--",
                        "--ignored",
                        "--nocapture",
                    ],
                    &[],
                ),
            ],
            &spec_log,
            &receipt_path_str,
            &run_id,
            5,
            5,
        );
    }

    // ── Phase 5: lstm_activation_precision ──
    // Runs both `structural` and the two SNR precision characterizations (`gain`
    // against NAMCore goldens and `gain_stress_v2` against the f64 oracle).
    if !existing_phases.contains_key("lstm_activation_precision") {
        let lstm_log = log_dir.join("lstm_activation_precision.log");
        run_dashboard_phase(
            "lstm_activation_precision",
            &[(
                &[
                    "cargo",
                    "test",
                    "--release",
                    "--features",
                    "testing",
                    "--test",
                    "models",
                    "lstm_activation_precision",
                    "--",
                    "--include-ignored",
                    "--nocapture",
                ],
                &[("NAM_METRICS_JSONL", &metrics_path_str)],
            )],
            &lstm_log,
            &receipt_path_str,
            &run_id,
            3,
            3,
        );
    }

    // ── Phase 6: quick_parity ──
    if !existing_phases.contains_key("quick_parity") {
        let tp_exists = Path::new("third-party/NeuralAmpModelerCore").is_dir();
        if !tp_exists {
            let qp_rec = PhaseReceiptRecord {
                phase_id: "quick_parity",
                status: "SKIP_CAPABILITY",
                exit_code: 0,
                observed_records: 0,
                expected_records: 1,
                reason: "third_party_absent",
                run_id: &run_id,
            };
            let _ = append_line(&receipt_path_str, &serde_json::to_string(&qp_rec).unwrap());
        } else {
            let qp_log = log_dir.join("quick_parity.log");
            run_dashboard_phase(
                "quick_parity",
                &[(
                    &[
                        "cargo",
                        "test",
                        "--release",
                        "--features",
                        "testing",
                        "--test",
                        "parity",
                        "quick_parity",
                        "--",
                        "--include-ignored",
                        "--test-threads=1",
                        "--nocapture",
                    ],
                    &[("NAM_METRICS_JSONL", &metrics_path_str)],
                )],
                &qp_log,
                &receipt_path_str,
                &run_id,
                1,
                1,
            );
        }
    }

    // ── Build metadata ──
    let receipt_content = fs::read_to_string(&receipt_path).unwrap_or_default();
    if !receipt_content.contains("\"kind\":\"build_metadata\"") {
        let probe = EnvProbe::probe();
        // `EnvProbe::effective_isa` classifies the host CPU flags; the actual
        // engine dispatch depends on compile-time features (`avx512` opt-in).
        let active_isa = detect_active_isa(&probe);
        let meta = serde_json::json!({
            "kind": "build_metadata",
            "cargo_profile": "mixed (audio=release, tools=dev)",
            "audio_profile": "release",
            "tool_profile": "dev",
            "target_triple": probe.host_triple,
            "rustflags": probe.rustflags,
            "rustc_version": probe.rustc_version,
            "git_commit": probe.git_commit,
            "git_dirty_state": probe.git_dirty,
            "run_id": run_id,
            "effective_isa": active_isa,
            "host_isa": probe.effective_isa,
            "engine_backend": SIMD_MATH.name,
            "avx512_feature_active": SIMD_MATH.is_avx512,
            "hardware_class": probe.hardware_class.as_str(),
        });
        let _ = append_line(&receipt_path_str, &meta.to_string());
    }

    // ── Phase 7: regression_gate ──
    let fidelity_only = flags.get("fidelity-only").is_some();
    let mut perf_passed = false;
    if !fidelity_only && !existing_phases.contains_key("regression_gate") {
        println!("-> Running regression_gate (nam_perf_gate run --check)...");
        let reg_receipt_file = log_dir.join("regression_phase_receipt.jsonl");
        let reg_log_file = log_dir.join("regression_gate.log");
        let mut cmd = Command::new("cargo");
        cmd.args([
            "run",
            "--quiet",
            "--locked",
            "--features",
            "testing",
            "--bin",
            "nam_perf_gate",
            "--",
            "run",
            "--check",
            "--receipt",
            &reg_receipt_file.to_string_lossy(),
            "--log",
            &reg_log_file.to_string_lossy(),
            "--run-id",
            &run_id,
        ]);
        if let Some(core) = flags.get("bench-core") {
            cmd.args(["--bench-core", core]);
        }
        let output = cmd.output();
        let reg_exit = output
            .as_ref()
            .map(|o| o.status.code().unwrap_or(1))
            .unwrap_or(1);

        let mut rec_status = "FAIL".to_string();
        let mut rec_reason = "missing_receipt".to_string();
        let mut run_id_match = false;

        if let Ok(lines) = read_jsonl_lines(&reg_receipt_file.to_string_lossy()) {
            for line in lines.iter().rev() {
                if let Ok(v) = serde_json::from_str::<Value>(line)
                    && v.get("phase_id").and_then(Value::as_str) == Some("regression_check")
                {
                    if let Some(s) = v.get("status").and_then(Value::as_str) {
                        rec_status = s.to_string();
                    }
                    if let Some(r) = v.get("reason").and_then(Value::as_str) {
                        rec_reason = r.to_string();
                    }
                    if let Some(rid) = v.get("run_id").and_then(Value::as_str) {
                        run_id_match = rid == run_id;
                    }
                    break;
                }
            }
        }

        let outcome = classify_fresh_regression(reg_exit, &rec_status, &rec_reason, run_id_match);
        perf_passed = outcome == RegressionOutcome::Pass;
        let (final_status, final_reason, observed) = match outcome {
            RegressionOutcome::Pass => ("PASS", "", 10),
            RegressionOutcome::NotVerified => ("NOT_VERIFIED", rec_reason.as_str(), 0),
            RegressionOutcome::Fail => (
                "FAIL",
                if rec_reason.is_empty() {
                    "Performance regression check failed"
                } else {
                    rec_reason.as_str()
                },
                0,
            ),
        };

        let record = PhaseReceiptRecord {
            phase_id: "regression_gate",
            status: final_status,
            exit_code: if perf_passed { 0 } else { reg_exit as u32 },
            observed_records: observed,
            expected_records: 10,
            reason: final_reason,
            run_id: &run_id,
        };
        let _ = append_line(&receipt_path_str, &serde_json::to_string(&record).unwrap());
        println!(
            "   {} regression_gate ({final_status})",
            if perf_passed { "ok" } else { "NOT_OK" }
        );
    } else if existing_phases.get("regression_gate").map(String::as_str) == Some("PASS") {
        perf_passed = true;
    }

    // ── Ingest report ──
    let target_root = env::var("CARGO_TARGET_DIR")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "target".to_string());
    let default_criterion_dir = format!("{target_root}/criterion");
    let criterion_candidate = flags
        .get("criterion-dir")
        .unwrap_or(default_criterion_dir.as_str());
    let criterion_available = {
        let mut ok = Path::new(criterion_candidate).is_dir();
        if ok {
            for bench in ids::RT_BENCH_TABLE {
                let est = Path::new(criterion_candidate)
                    .join(bench.bench_label)
                    .join("new")
                    .join("estimates.json");
                if !est.is_file() {
                    ok = false;
                    break;
                }
            }
        }
        ok
    };

    let criterion_dir = if criterion_available && (perf_passed || fidelity_only) {
        Some(criterion_candidate)
    } else {
        None
    };

    // The build script exports the canonical micro-bench batch size here; feed
    // it to ingest so latency normalization never depends on a hard-coded copy.
    let target_bench_constants = format!("{target_root}/bench_constants.env");
    let bench_constants_env = if Path::new(&target_bench_constants).is_file() {
        Some(target_bench_constants.as_str())
    } else if Path::new("target/bench_constants.env").is_file() {
        Some("target/bench_constants.env")
    } else {
        None
    };

    if let Err(e) = run_ingest(
        &receipt_path_str,
        if metrics_path.is_file() {
            Some(&metrics_path_str)
        } else {
            None
        },
        None,
        criterion_dir,
        bench_constants_env,
        Some(contract_path),
        Some(&report_path_str),
    ) {
        run_error(format!("failed to ingest quality report: {e}"));
    }

    // ── Render ──
    let style = match (flags.get("ansi").is_some(), flags.get("plain").is_some()) {
        (_, true) => RenderStyle::Plain,
        (true, _) => RenderStyle::Ansi,
        _ if io::stdout().is_terminal() => RenderStyle::Ansi,
        _ => RenderStyle::Plain,
    };
    if let Ok(parsed_report) = parse_quality_report_file(&report_path_str) {
        print!("{}", render_quality_report(&parsed_report, style));
    }

    // ── Check / Save ──
    if flags.get("save").is_some() {
        save_dashboard_contract(
            contract_path,
            &receipt_path_str,
            &report_path_str,
            &run_id,
            fidelity_only,
            perf_passed,
        );
    }

    if flags.get("check").is_some() {
        let exit_code = execute_verify(contract_path, &report_path_str, fidelity_only);
        exit(exit_code);
    }

    let receipt_lines = read_jsonl_lines(&receipt_path_str).unwrap_or_default();
    let mut had_failure = false;
    for line in receipt_lines {
        if let Ok(v) = serde_json::from_str::<Value>(&line)
            && v.get("status").and_then(Value::as_str) == Some("FAIL")
        {
            had_failure = true;
            break;
        }
    }
    exit(if had_failure { 1 } else { 0 });
}

fn print_dashboard_help() {
    println!("Usage: nam_quality dashboard [--check <contract>] [--save]");
    println!("                               [--contract <path>] [--fidelity-only]");
    println!("                               [--bench-core <core>] [--run-id <id>]");
    println!("                               [--log-dir <dir>] [--criterion-dir <dir>]");
    println!("                               [--ansi | --plain]");
    println!();
    println!("Self-contained quality dashboard orchestrator: executes missing fidelity");
    println!("phases, invokes the performance regression check, ingests the verified");
    println!("report, renders the human-facing table, and verifies or saves the contract.");
    println!();
    println!("  --fidelity-only  verify only the fidelity domain (skips performance).");
    println!("  --ansi           ANSI color escape sequences (default on a TTY).");
    println!("  --plain          plain text (default when stdout is not a TTY).");
}
