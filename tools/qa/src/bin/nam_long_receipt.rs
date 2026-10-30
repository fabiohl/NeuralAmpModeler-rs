// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Structured JSONL receipt emitter for the long-duration audit suite.
//!
//! `utils/tests-long.sh` appends one machine-readable line per completed phase
//! to `target/logs/long-audit-receipt.jsonl` — schema: `phase_id`, `name`,
//! `status`, `duration_ms`, `tests_executed`, `gaps`, `timestamp`. All JSON
//! generation is done here (serde); the shell script never hand-serializes.
//!
//! Subcommands:
//! - `append`:  validate one phase outcome and append its JSONL line.
//! - `summary`: read the receipt, derive the suite-level `overall` line and
//!   append it (verdict: PASSED | FAILED | COMPLETED_WITH_GAPS).
//! - `validate`: fail-closed semantic check that every line is valid JSON with
//!   the receipt schema, that the canonical roster is complete and
//!   duplicate-free, and that the declared `overall` verdict is consistent with
//!   the phase/gap evidence (never a substring search). `--strict` additionally
//!   rejects any declared gap or failure.
//! - `count-log`: print how many tests/benchmarks a phase log proves executed
//!   (F-21 counter behind `_lib.sh::assert_ran_tests`).
//!
//! Exit codes:
//! - 0: success.
//! - 1: I/O or validation error (receipt file missing/corrupt, write failure).
//! - 2: usage error (unknown subcommand/flag, malformed argument).
//! - 4: `validate` only — receipt is valid but the audit verdict is FAILED.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};
use std::str::FromStr;

use neural_amp_modeler_rs::testing::receipt::{
    CANONICAL_ROSTER, LongAuditReceipt, LongPhaseReceipt, LongPhaseStatus, LongReason,
    PREFLIGHT_PHASE_IDS, count_tests_executed_from_log, detect_gap_markers, is_preflight_id,
    now_iso8601,
};

const DEFAULT_OUT: &str = "target/logs/long-audit-receipt.jsonl";

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        exit(2);
    }

    match args[0].as_str() {
        "append" => cmd_append(&args[1..]),
        "summary" => cmd_summary(&args[1..]),
        "validate" => cmd_validate(&args[1..]),
        "count-log" => cmd_count_log(&args[1..]),
        "-h" | "--help" => {
            print_help();
            exit(0);
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            print_help();
            exit(2);
        }
    }
}

fn print_help() {
    println!("NeuralAmpModeler-rs Long-Audit Receipt Emitter");
    println!("Usage:");
    println!("  nam_long_receipt append --phase-id <id> --name <name> --status <STATUS>");
    println!("                        --duration-ms <ms> [--tests-executed <n>] [--log <path>]");
    println!("                        [--gaps <a,b,...>] [--out <file>]");
    println!("                        [--oracle-sha256 <hex>] [--fixture-sha256 <hex>]");
    println!("                        [--artifact-sha256 <hex>] [--reason <REASON>]");
    println!("  nam_long_receipt summary [--out <file>] [--only <a,b,...>]");
    println!("  nam_long_receipt validate [--out <file>] [--strict] [--only <a,b,...>]");
    println!("  nam_long_receipt count-log --log <path>");
    println!();
    println!(
        "Status values: PASSED, FAILED, SKIPPED, INCONCLUSIVE, SKIP_CAPABILITY, NOT_RUN, SIMULATED"
    );
    println!(
        "Canonical preflight ids: {}",
        PREFLIGHT_PHASE_IDS.join(", ")
    );
    println!(
        "Reasons: GENERATOR_DRIFT, MISSING_BASELINE, TEST_FAILURE, TIMEOUT, PREFLIGHT_FAILURE"
    );
    println!("Default output file: {DEFAULT_OUT}");
    println!(
        "validate: fail-closed semantic verification (canonical roster + OVERALL consistency)."
    );
    println!("validate --strict: additionally rejects any declared gap or failure.");
}

/// Flags that do not take a value (boolean switches).
const VALUE_LESS_FLAGS: &[&str] = &["strict"];

fn parse_flags(args: &[String]) -> Result<std::collections::HashMap<String, String>, String> {
    let mut flags = std::collections::HashMap::new();
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        let name = flag.strip_prefix("--").unwrap_or(flag.as_str()).to_string();
        if name.is_empty() {
            return Err(format!("{flag}: expected a flag name"));
        }
        if VALUE_LESS_FLAGS.contains(&name.as_str()) {
            flags.insert(name, String::new());
            continue;
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        flags.insert(name, value.clone());
    }
    Ok(flags)
}

fn required<'a>(
    flags: &'a std::collections::HashMap<String, String>,
    name: &str,
) -> Result<&'a String, String> {
    flags
        .get(name)
        .ok_or_else(|| format!("missing required flag --{name}"))
}

fn cmd_append(args: &[String]) {
    let flags = match parse_flags(args) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("append: {e}");
            exit(2);
        }
    };

    let phase_id = match required(&flags, "phase-id") {
        Ok(v) => v.clone(),
        Err(e) => die_usage("append", &e),
    };
    if is_preflight_id(&phase_id) && !PREFLIGHT_PHASE_IDS.contains(&phase_id.as_str()) {
        die_usage(
            "append",
            &format!(
                "unknown preflight identifier '{phase_id}' (canonical: {})",
                PREFLIGHT_PHASE_IDS.join(", ")
            ),
        );
    }
    let name = match required(&flags, "name") {
        Ok(v) => v.clone(),
        Err(e) => die_usage("append", &e),
    };
    let status = match required(&flags, "status").and_then(|s| LongPhaseStatus::from_str(s)) {
        Ok(s) => s,
        Err(e) => die_usage("append", &e),
    };
    let duration_ms: u64 = match required(&flags, "duration-ms") {
        Ok(v) => match v.parse() {
            Ok(n) => n,
            Err(_) => die_usage("append", "--duration-ms must be a non-negative integer"),
        },
        Err(e) => die_usage("append", &e),
    };
    let out = flags
        .get("out")
        .cloned()
        .unwrap_or_else(|| DEFAULT_OUT.to_string());

    let log_path = flags.get("log").map(PathBuf::from);
    let tests_executed = match flags.get("tests-executed") {
        Some(v) => match v.parse::<u64>() {
            Ok(n) => n,
            Err(_) => die_usage("append", "--tests-executed must be a non-negative integer"),
        },
        None => log_path
            .as_deref()
            .map(count_tests_executed_from_log)
            .unwrap_or(0),
    };

    let mut gaps: Vec<String> = Vec::new();
    let mut optional_gaps: Vec<String> = Vec::new();
    if let Some(list) = flags.get("gaps") {
        for g in list.split(',') {
            let g = g.trim();
            if !g.is_empty() {
                gaps.push(g.to_string());
            }
        }
    }
    if let Some(id) = status.gap_id() {
        gaps.push(id.to_string());
    }
    if let Some(path) = log_path.as_deref() {
        for marker in detect_gap_markers(path) {
            let is_optional = marker == "skip_optional" || marker.starts_with("skip_optional:");
            let target = if is_optional {
                &mut optional_gaps
            } else {
                &mut gaps
            };
            if !target.contains(&marker) {
                target.push(marker);
            }
        }
    }

    let reason = match flags.get("reason") {
        Some(value) => match LongReason::from_str(value) {
            Ok(r) => Some(r),
            Err(e) => die_usage("append", &e),
        },
        None => None,
    };

    let (git_sha, git_dirty) = git_provenance();
    let (oracle_core_commit, oracle_compiler) = if phase_id == "preflight-render" {
        render_oracle_provenance()
    } else {
        (None, None)
    };
    let receipt = LongPhaseReceipt {
        phase_id,
        name,
        status,
        duration_ms,
        tests_executed,
        gaps,
        optional_gaps,
        timestamp: now_iso8601(),
        oracle_sha256: flags.get("oracle-sha256").cloned(),
        oracle_core_commit,
        oracle_compiler,
        fixture_sha256: flags.get("fixture-sha256").cloned(),
        artifact_sha256: flags.get("artifact-sha256").cloned(),
        run_id: env::var("NAM_RUN_ID").ok().filter(|v| !v.is_empty()),
        git_sha,
        git_dirty,
        reason,
    };

    if let Err(e) = append_line(&out, &receipt.render_jsonl_line()) {
        eprintln!("append: failed to write receipt to {out}: {e}");
        exit(1);
    }
    eprintln!(
        "✓ long-audit receipt: {} {} ({} tests, {} ms)",
        receipt.phase_id, receipt.status, receipt.tests_executed, receipt.duration_ms
    );
    exit(0);
}

fn cmd_summary(args: &[String]) {
    let flags = match parse_flags(args) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("summary: {e}");
            exit(2);
        }
    };
    let out = flags
        .get("out")
        .cloned()
        .unwrap_or_else(|| DEFAULT_OUT.to_string());
    let only_list: Option<Vec<String>> = flags.get("only").map(|s| {
        s.split(',')
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect()
    });

    let mut audit = match LongAuditReceipt::parse_jsonl_file(Path::new(&out)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("summary: cannot build summary from {out}: {e}");
            exit(1);
        }
    };
    if audit.phases.is_empty() {
        eprintln!("summary: no phase entries in {out} — nothing to summarize");
        exit(1);
    }
    audit.push_summary_with_only(only_list.as_deref());
    let overall = audit
        .phases
        .last()
        .expect("push_summary always appends the overall line");
    if let Err(e) = write_file(&out, &audit.render_jsonl()) {
        eprintln!("summary: failed to write overall line to {out}: {e}");
        exit(1);
    }
    eprintln!(
        "✓ long-audit receipt: overall {} ({} tests, {} ms, {} gaps)",
        overall.status,
        overall.tests_executed,
        overall.duration_ms,
        overall.gaps.len()
    );
    // S5: the human summary — WARNING/ERROR alarms + verdict lines, echoed
    // verbatim by utils/tests-long.sh (which maps `OVERALL:` to its exit
    // code). The forensic data stays in the JSONL.
    for line in audit.human_summary_lines_with_only(None, only_list.as_deref()) {
        println!("{line}");
    }
    exit(0);
}

fn cmd_validate(args: &[String]) {
    let flags = match parse_flags(args) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("validate: {e}");
            exit(2);
        }
    };
    let out = flags
        .get("out")
        .cloned()
        .unwrap_or_else(|| DEFAULT_OUT.to_string());
    let strict = flags.contains_key("strict");
    let only_list: Option<Vec<String>> = flags.get("only").map(|s| {
        s.split(',')
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect()
    });

    if !Path::new(&out).exists() {
        eprintln!("validate: receipt file not found: {out}");
        exit(1);
    }
    let audit = match LongAuditReceipt::parse_jsonl_file(Path::new(&out)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("validate: invalid receipt at {out}: {e}");
            exit(1);
        }
    };
    // Semantic verification is always fail-closed — never a substring search and
    // never a mere parse check: the canonical roster must be complete and
    // duplicate-free, timed phases must have run real tests, and the declared
    // `overall` verdict must match the verdict derived from the phase/gap
    // evidence (a forged/injected OVERALL line is rejected here).
    if let Err(e) = audit.semantic_verdict_with_only(&CANONICAL_ROSTER, only_list.as_deref()) {
        eprintln!("validate: {e}");
        exit(1);
    }
    let mut counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    let mut preflight_count = 0u64;
    for phase in &audit.phases {
        *counts.entry(phase.status.as_str().to_string()).or_default() += 1;
        if is_preflight_id(&phase.phase_id) {
            preflight_count += 1;
        }
    }
    let mut summary = format!(
        "VALID: {} receipt line(s) in {out} ({} tests, {} ms)",
        audit.phases.len(),
        audit.tests_executed_total(),
        audit.duration_ms_total()
    );
    for (status, count) in &counts {
        summary.push_str(&format!(" | {status}: {count}"));
    }
    summary.push_str(&format!(" | preflight: {preflight_count}"));
    println!("{summary}");

    if strict {
        // Every FAILED line must name its typed root cause; a bare FAILED makes
        // the audit unreproducible for the operator.
        let missing: Vec<&str> = audit
            .phase_entries()
            .filter(|p| p.status == LongPhaseStatus::Failed && p.reason.is_none())
            .map(|p| p.phase_id.as_str())
            .collect();
        if !missing.is_empty() {
            eprintln!(
                "STRICT: FAIL — FAILED phase(s) without reason: {}",
                missing.join(", ")
            );
            exit(1);
        }
        // Provenance is part of the certification claim: every phase line must
        // carry run_id/git_sha/git_dirty, come from a clean tree and share the
        // same run and commit.
        let mut run_id: Option<&str> = None;
        let mut git_sha: Option<&str> = None;
        for p in audit.phase_entries() {
            let (Some(run), Some(sha), Some(dirty)) =
                (p.run_id.as_deref(), p.git_sha.as_deref(), p.git_dirty)
            else {
                eprintln!(
                    "STRICT: FAIL — phase '{}' lacks provenance (run_id/git_sha/git_dirty)",
                    p.phase_id
                );
                exit(1);
            };
            if dirty {
                eprintln!(
                    "STRICT: FAIL — phase '{}' was emitted from a dirty tree",
                    p.phase_id
                );
                exit(1);
            }
            if *run_id.get_or_insert(run) != run || *git_sha.get_or_insert(sha) != sha {
                eprintln!("STRICT: FAIL — mixed run/commit provenance across receipt lines");
                exit(1);
            }
        }
    }

    // A structurally valid receipt whose audit verdict is FAILED is still a
    // rejected audit: report it with the typed code 4 (1 stays reserved for a
    // corrupt/invalid receipt). The runner already has the root causes.
    if audit
        .phases
        .last()
        .is_some_and(|p| p.status == LongPhaseStatus::Failed)
    {
        eprintln!("validate: valid receipt, but overall audit FAILED (audit rejected)");
        exit(4);
    }

    if strict {
        // Fail-closed strict-pre-release — any declared gap (or failure)
        // rejects the receipt with a non-zero exit code.
        match audit.campaign_strict_verdict_with_only(&CANONICAL_ROSTER, only_list.as_deref()) {
            Ok(()) => {
                println!("STRICT: PASSED (no gaps, no failures)");
                exit(0);
            }
            Err(reason) => {
                eprintln!("STRICT: FAIL — {reason}");
                exit(1);
            }
        }
    }
    exit(0);
}

fn cmd_count_log(args: &[String]) {
    let flags = match parse_flags(args) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("count-log: {e}");
            exit(2);
        }
    };
    let log = match required(&flags, "log") {
        Ok(v) => PathBuf::from(v),
        Err(e) => die_usage("count-log", &e),
    };
    println!("{}", count_tests_executed_from_log(&log));
    exit(0);
}

/// C++ oracle provenance recorded by `utils/ensure_namcore_render.sh` in
/// `build/namcore_render/.build_provenance`. Read here so the shell runner
/// stays a thin orchestrator and the receipt owns the structured capture.
fn render_oracle_provenance() -> (Option<String>, Option<String>) {
    let path = Path::new("build/namcore_render/.build_provenance");
    let Ok(text) = fs::read_to_string(path) else {
        return (None, None);
    };
    let field = |key: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(key).map(str::trim))
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    (
        field("core_commit="),
        field("compiler=").map(|c| {
            let version = field("compiler_version=").unwrap_or_default();
            if version.is_empty() {
                c
            } else {
                format!("{c}: {version}")
            }
        }),
    )
}

fn git_provenance() -> (Option<String>, Option<bool>) {
    let sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty());
    (sha, dirty)
}

fn append_line(out: &str, line: &str) -> std::io::Result<()> {
    if let Some(parent) = Path::new(out).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(out)?;
    file.write_all(line.as_bytes())?;
    file.write_all(b"\n")
}

fn write_file(out: &str, content: &str) -> std::io::Result<()> {
    if let Some(parent) = Path::new(out).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, content)
}

fn die_usage(subcommand: &str, message: &str) -> ! {
    eprintln!("{subcommand}: {message}");
    eprintln!("Run `nam_long_receipt --help` for usage.");
    exit(2);
}
