// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Benchmark prebuild, execution, and end-to-end regression gate runner (`run`).

use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use neural_amp_modeler_rs::testing::qa::baseline_store::{
    DEFAULT_BASELINE_NAME, persist_baseline, restore_baseline,
};
use neural_amp_modeler_rs::testing::qa::coverage::{
    BaselineCoverageGap, executed_bench_ids, expected_bench_ids_for_suite,
    missing_baseline_coverage, missing_expected_benchmarks,
};
use neural_amp_modeler_rs::testing::qa::fingerprint::{Fingerprint, FingerprintError};
use neural_amp_modeler_rs::testing::qa::pick_core::resolve_bench_core;
use neural_amp_modeler_rs::testing::qa::verdict::{
    DEFAULT_NOISE_THRESHOLD, VerdictBlind, VerdictOptions, regression_verdict_with_options,
};

use crate::cli::{
    DEFAULT_BASELINE_DIR, DEFAULT_RECEIPT, HELP_REQUEST, default_criterion_root, parse_f64,
    parse_flags_full, run_error, usage_error,
};
use crate::receipt::append_phase_receipt;

fn format_duration_ms(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        let s = ms / 1000;
        let rem = (ms % 1000) / 100;
        format!("{s}.{rem}s")
    }
}

fn current_utc_iso() -> String {
    neural_amp_modeler_rs::common::diagnostics::format::timestamp()
}

/// Pre-compiles the Criterion benchmark binary on all cores before thermal cooldown
/// and returns its executable path so the pinned run can bypass Cargo invocation.
fn prebuild_benchmark(suite_name: &str) -> io::Result<Option<PathBuf>> {
    eprintln!("  Pre-compiling benchmark '{suite_name}' prior to cooldown (all cores)...");
    let output = Command::new("cargo")
        .args([
            "bench",
            "--bench",
            suite_name,
            "--features",
            "testing",
            "--no-run",
            "--message-format=json",
        ])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "cargo bench pre-compilation failed with exit status {}",
            output.status
        )));
    }

    // Parse Cargo's JSON stream to locate the compiled benchmark artifact path.
    let mut candidate_exe: Option<PathBuf> = None;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            let reason = val.get("reason").and_then(|v| v.as_str());
            let target_name = val
                .get("target")
                .and_then(|t| t.get("name"))
                .and_then(|v| v.as_str());
            let is_bench = val
                .get("target")
                .and_then(|t| t.get("kind"))
                .and_then(|v| v.as_array())
                .map(|kinds| kinds.iter().any(|k| k.as_str() == Some("bench")))
                .unwrap_or(false);

            if reason == Some("compiler-artifact")
                && target_name == Some(suite_name)
                && is_bench
                && let Some(exe) = val.get("executable").and_then(|v| v.as_str())
            {
                candidate_exe = Some(PathBuf::from(exe));
            }
        }
    }

    // Fallback scan of `target/release/deps` when Cargo omits the JSON executable field.
    if candidate_exe.is_none() {
        let deps_dir = Path::new("target/release/deps");
        if deps_dir.is_dir() {
            let prefix = format!("{suite_name}-");
            if let Ok(entries) = fs::read_dir(deps_dir) {
                let mut matches: Vec<(SystemTime, PathBuf)> = Vec::new();
                for entry in entries.flatten() {
                    let path = entry.path();
                    let file_name = path.file_name().and_then(|s| s.to_str());
                    let is_match = file_name
                        .map(|name| {
                            let stem = if cfg!(windows) {
                                match name.strip_suffix(".exe") {
                                    Some(s) => s,
                                    None => return false,
                                }
                            } else {
                                name
                            };
                            stem.starts_with(&prefix) && !stem.contains('.')
                        })
                        .unwrap_or(false);

                    if is_match && let Ok(meta) = entry.metadata() {
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::PermissionsExt;
                            if meta.permissions().mode() & 0o111 != 0 {
                                let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
                                matches.push((mtime, path));
                            }
                        }
                        #[cfg(not(unix))]
                        {
                            let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
                            matches.push((mtime, path));
                        }
                    }
                }
                matches.sort_by_key(|(mtime, _)| *mtime);
                if let Some((_, newest)) = matches.pop() {
                    candidate_exe = Some(newest);
                }
            }
        }
    }

    // Sanity-check that the resolved binary responds to Criterion's `--bench --list`.
    if let Some(ref exe) = candidate_exe {
        let test_out = Command::new(exe).args(["--bench", "--list"]).output();
        match test_out {
            Ok(out) if out.status.success() && !out.stdout.is_empty() => {
                eprintln!("  ✓ Benchmark executable ready: {}", exe.display());
                Ok(Some(exe.clone()))
            }
            _ => {
                eprintln!(
                    "  WARN: candidate executable {} failed criterion list validation",
                    exe.display()
                );
                Ok(None)
            }
        }
    } else {
        eprintln!(
            "  WARN: Could not isolate standalone benchmark binary — fallback to cargo bench"
        );
        Ok(None)
    }
}

fn apply_thermal_cooldown(cooldown: u64) {
    if cooldown > 0 {
        eprintln!("\n[INFO] Thermal cooldown: sleeping {cooldown}s before microbenchmarks...");
        std::thread::sleep(Duration::from_secs(cooldown));
    } else {
        eprintln!("\n[INFO] Thermal cooldown: bypassed (NAM_THERMAL_COOLDOWN_S=0)");
    }
}

fn build_bench_command(
    bench_exe: Option<&Path>,
    suite: &str,
    bench_core: Option<u32>,
    extra_args: &[&str],
) -> Command {
    if let Some(core) = bench_core {
        let mut c = Command::new("taskset");
        c.arg("-c").arg(core.to_string());
        if let Some(exe) = bench_exe {
            c.arg(exe).arg("--bench").args(extra_args);
        } else {
            c.args([
                "cargo",
                "bench",
                "--bench",
                suite,
                "--features",
                "testing",
                "--",
            ])
            .args(extra_args);
        }
        return c;
    }
    if let Some(exe) = bench_exe {
        let mut c = Command::new(exe);
        c.arg("--bench").args(extra_args);
        c
    } else {
        let mut c = Command::new("cargo");
        c.args(["bench", "--bench", suite, "--features", "testing", "--"])
            .args(extra_args);
        c
    }
}

fn fail_check_with_receipt(
    receipt_path: &str,
    phase_id: &str,
    exit_code: i32,
    observed: u32,
    expected: u32,
    reason: &str,
    run_id: &str,
) -> ! {
    let _ = append_phase_receipt(
        receipt_path,
        phase_id,
        "FAIL",
        exit_code as u32,
        observed,
        expected,
        reason,
        run_id,
    );
    exit(exit_code);
}

/// Spawns the benchmark binary (optionally pinned with `taskset`) and tees stdout/stderr to `log_path`.
fn run_benchmark_process(
    bench_exe: Option<&Path>,
    suite: &str,
    bench_core: Option<u32>,
    extra_args: &[&str],
    log_path: &str,
    baseline_name: &str,
    run_id: &str,
) -> io::Result<i32> {
    if let Some(parent) = Path::new(log_path).parent() {
        fs::create_dir_all(parent)?;
    }
    let date_str = current_utc_iso();
    let exe_display = bench_exe
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| format!("cargo bench --bench {suite}"));
    let header = format!(
        "# [NAM_REGRESSION_GATE] bench_exe={exe_display} baseline={baseline_name} run_id={run_id} date={date_str}\n"
    );
    fs::write(log_path, header)?;

    let mut cmd = build_bench_command(bench_exe, suite, bench_core, extra_args);

    // Fall back to unpinned execution if `taskset` is unavailable on the host.
    let mut child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(child) => child,
        Err(e) if bench_core.is_some() && e.kind() == io::ErrorKind::NotFound => {
            eprintln!("WARN: taskset not found in PATH, running without CPU pinning");
            build_bench_command(bench_exe, suite, None, extra_args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?
        }
        Err(e) => return Err(e),
    };

    // Stream stdout and stderr concurrently to avoid pipe deadlocks and mirror to the log file.
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("no child stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("no child stderr"))?;

    let log_file = Arc::new(Mutex::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?,
    ));

    let log_out = Arc::clone(&log_file);
    let t_out = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            println!("{line}");
            if let Ok(mut f) = log_out.lock() {
                let _ = writeln!(f, "{line}");
            }
        }
    });

    let log_err = Arc::clone(&log_file);
    let t_err = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            eprintln!("{line}");
            if let Ok(mut f) = log_err.lock() {
                let _ = writeln!(f, "{line}");
            }
        }
    });

    let _ = t_out.join();
    let _ = t_err.join();
    let status = child.wait()?;
    Ok(status.code().unwrap_or(1))
}

// ── run ──────────────────────────────────────────────────────────────────────

pub(crate) fn cmd_run(args: &[String]) {
    let flags = match parse_flags_full(
        args,
        &[
            "baseline",
            "baseline-dir",
            "fingerprint",
            "criterion-root",
            "bench-core",
            "sysfs",
            "suite",
            "manifest",
            "cooldown",
            "receipt",
            "log",
            "run-id",
            "noise",
            "abs-noise-floor-ns",
            "bench-exe",
        ],
        &[
            "check",
            "bootstrap-baseline",
            "check-stability",
            "strict-stability",
            "confirm-regressions",
            "strict",
        ],
    ) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_run_help();
            exit(0);
        }
        Err(e) => usage_error("run", &e),
    };

    let is_bootstrap = flags.get_bool("bootstrap-baseline");
    let is_check = flags.get_bool("check");
    if is_bootstrap && is_check {
        usage_error(
            "run",
            "cannot specify both --check and --bootstrap-baseline",
        );
    }
    let mode_bootstrap = is_bootstrap;

    let baseline_dir = flags
        .get("baseline-dir")
        .unwrap_or(DEFAULT_BASELINE_DIR)
        .to_string();
    let baseline = flags
        .get("baseline")
        .map(str::to_string)
        .unwrap_or_else(|| {
            env::var("NAM_BASELINE_NAME").unwrap_or_else(|_| DEFAULT_BASELINE_NAME.to_string())
        });
    let criterion_root = flags
        .get("criterion-root")
        .map(str::to_string)
        .unwrap_or_else(default_criterion_root);
    let suite = flags.get("suite").unwrap_or("regression_gate").to_string();
    let manifest = flags.get("manifest").map(str::to_string);
    let receipt_path = flags.get("receipt").unwrap_or(DEFAULT_RECEIPT).to_string();
    let log_path = flags
        .get("log")
        .unwrap_or("target/logs/regression-check.log")
        .to_string();
    let run_id = flags.get("run-id").map(str::to_string).unwrap_or_else(|| {
        env::var("NAM_RUN_ID").unwrap_or_else(|_| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            format!("{}-{}", now, std::process::id())
        })
    });
    let cooldown_secs = match flags.get("cooldown") {
        Some(v) => v.parse::<u64>().unwrap_or_else(|_| {
            eprintln!("\n[WARN] Invalid --cooldown='{v}', defaulting to 0s");
            0
        }),
        None => match env::var("NAM_THERMAL_COOLDOWN_S") {
            Ok(v) => v.parse::<u64>().unwrap_or_else(|_| {
                eprintln!("\n[WARN] Invalid NAM_THERMAL_COOLDOWN_S='{v}', defaulting to 0s");
                0
            }),
            Err(_) => 0,
        },
    };

    let is_strict = flags.get_bool("strict");
    let explicit_flag = flags.get("bench-core");
    let env_core = env::var("NAM_BENCH_CORE").ok();
    let explicit_str = explicit_flag.or(env_core.as_deref());
    let sysfs = flags.get("sysfs").map(Path::new);
    let resolved_core = resolve_bench_core(explicit_str, sysfs);

    if is_strict && resolved_core.is_none() {
        eprintln!("run: strict mode enabled and no bench core available (GAP)");
        exit(1);
    }

    let fingerprint_path = flags
        .get("fingerprint")
        .map(String::from)
        .unwrap_or_else(|| format!("{baseline_dir}/baseline-fingerprint.json"));

    // Baseline presence is a state decision (A2): decide it here, before the
    // benchmark pre-compilation (L-06). A missing baseline is a GAP in
    // non-strict mode (exit 3) and a typed FAIL in strict mode (exit 4),
    // never a performance regression.
    if !mode_bootstrap && !Path::new(&fingerprint_path).is_file() {
        println!("MISSING_BASELINE no baseline fingerprint at {fingerprint_path}");
        println!("Re-bootstrap the baseline with `--bootstrap-baseline` (human operation).");
        if is_strict {
            fail_check_with_receipt(
                &receipt_path,
                "regression_check",
                4,
                0,
                1,
                "MISSING_BASELINE",
                &run_id,
            );
        }
        println!("GAP perf-baseline-missing (non-strict mode)");
        exit(3);
    }

    let bench_exe: Option<PathBuf> = if let Some(exe) = flags.get("bench-exe") {
        let p = PathBuf::from(exe);
        if !p.exists() {
            run_error(format!(
                "specified --bench-exe does not exist: {}",
                p.display()
            ));
        }
        Some(p)
    } else {
        match prebuild_benchmark(&suite) {
            Ok(exe) => exe,
            Err(e) => {
                eprintln!("Pre-compilation of benchmark '{suite}' failed: {e}");
                if !mode_bootstrap {
                    let _ = append_phase_receipt(
                        &receipt_path,
                        "regression_check",
                        "FAIL",
                        1,
                        0,
                        1,
                        "BENCHMARK_PREBUILD_FAILED",
                        &run_id,
                    );
                }
                exit(1);
            }
        }
    };

    // ── Bootstrap mode: human-only baseline creation ─────────────────────────
    if mode_bootstrap {
        println!("\n[BOOTSTRAP] Creating new performance baseline...");
        println!("  ⚠ This operation must be performed by a human operator.");
        println!("  ⚠ Automated/CI/agent-driven execution is prohibited.\n");

        apply_thermal_cooldown(cooldown_secs);

        let t0 = Instant::now();
        let extra = ["--save-baseline", &baseline];
        let bench_rc = match run_benchmark_process(
            bench_exe.as_deref(),
            &suite,
            resolved_core,
            &extra,
            &log_path,
            &baseline,
            &run_id,
        ) {
            Ok(rc) => rc,
            Err(e) => run_error(format!("Failed to execute benchmark: {e}")),
        };
        if bench_rc != 0 {
            run_error(format!(
                "Benchmark execution failed during baseline bootstrap (rc={bench_rc})"
            ));
        }
        let dur_ms = t0.elapsed().as_millis() as u64;
        let dur_str = format_duration_ms(dur_ms);

        // Fail closed if Criterion produced zero series under the target baseline name.
        match persist_baseline(
            Path::new(&baseline_dir),
            Path::new(&criterion_root),
            &baseline,
        ) {
            Ok(0) => run_error(format!(
                "no '{baseline}' baseline series found under {criterion_root} — nothing persisted (fail-closed)"
            )),
            Ok(count) => {
                eprintln!("  ✓ Persisted {count} baseline series to {baseline_dir}");
            }
            Err(e) => run_error(format!("cannot persist baseline: {e}")),
        }

        let core_str = resolved_core.map(|c| c.to_string()).unwrap_or_default();
        let fingerprint = Fingerprint::probe(&core_str);
        if let Err(e) = fingerprint.write_to_path(Path::new(&fingerprint_path)) {
            run_error(format!(
                "cannot write fingerprint to {fingerprint_path}: {e}"
            ));
        }

        if let Err(e) = append_phase_receipt(
            &receipt_path,
            "regression_baseline_created",
            "PASS",
            0,
            1,
            1,
            "",
            &run_id,
        ) {
            run_error(format!("cannot append receipt to {receipt_path}: {e}"));
        }

        eprintln!("  ✓ Baseline '{baseline}' created and persisted in {dur_str}.");
        println!(
            "\n================================================================================"
        );
        println!("  Performance Bootstrap Artifacts saved:");
        println!("    - Fingerprint:   {fingerprint_path}");
        println!("    - Baseline Data: {baseline_dir}");
        println!("    - Receipt:       {receipt_path}");
        println!(
            "================================================================================\n"
        );
        exit(0);
    }

    // ── Check mode ───────────────────────────────────────────────────────────
    println!("\n[CHECK] Comparing against CI baseline...");

    let core_str = resolved_core.map(|c| c.to_string()).unwrap_or_default();
    let current_fingerprint = Fingerprint::probe(&core_str);

    let baseline_fingerprint = match Fingerprint::read_from_path(Path::new(&fingerprint_path)) {
        Ok(b) => b,
        Err(FingerprintError::MissingBaseline) => {
            println!("MISSING_BASELINE no baseline fingerprint at {fingerprint_path}");
            println!("Re-bootstrap the baseline with `--bootstrap-baseline` (human operation).");
            if is_strict {
                fail_check_with_receipt(
                    &receipt_path,
                    "regression_check",
                    4,
                    0,
                    1,
                    "MISSING_BASELINE",
                    &run_id,
                );
            }
            exit(3);
        }
        Err(e) => {
            eprintln!("cannot read baseline fingerprint {fingerprint_path}: {e}");
            fail_check_with_receipt(
                &receipt_path,
                "regression_check",
                1,
                0,
                1,
                "FINGERPRINT_ERROR",
                &run_id,
            );
        }
    };

    // Verify host/ISA/toolchain fingerprint compatibility before restoring Criterion baselines.
    if let Err(e) = current_fingerprint.compare(&baseline_fingerprint) {
        println!("{e}");
        println!("Performance comparison aborted: environment incompatible with baseline.");
        fail_check_with_receipt(
            &receipt_path,
            "regression_check",
            1,
            0,
            1,
            "INCOMPARABLE_ENVIRONMENT",
            &run_id,
        );
    }

    // Copy persisted baseline series into `target/criterion/` so Criterion compares against them.
    if let Err(e) = restore_baseline(
        Path::new(&baseline_dir),
        Path::new(&criterion_root),
        &baseline,
    ) {
        eprintln!("cannot restore baseline: {e}");
        fail_check_with_receipt(
            &receipt_path,
            "regression_check",
            1,
            0,
            1,
            "BASELINE_RESTORE_FAILED",
            &run_id,
        );
    }

    apply_thermal_cooldown(cooldown_secs);

    let t0 = Instant::now();
    let extra = ["--baseline", &baseline];
    let bench_rc = match run_benchmark_process(
        bench_exe.as_deref(),
        &suite,
        resolved_core,
        &extra,
        &log_path,
        &baseline,
        &run_id,
    ) {
        Ok(rc) => rc,
        Err(e) => run_error(format!("Failed to execute benchmark: {e}")),
    };
    let dur_ms = t0.elapsed().as_millis() as u64;
    let dur_str = format_duration_ms(dur_ms);

    if bench_rc != 0 {
        eprintln!("\n❌ Benchmark run failed (status={bench_rc})");
        fail_check_with_receipt(
            &receipt_path,
            "regression_check",
            1,
            0,
            1,
            "Benchmark run failed",
            &run_id,
        );
    }

    // Reconcile executed benchmarks against the expected manifest and saved baseline series.
    let log_text = fs::read_to_string(&log_path).unwrap_or_default();

    let manifest_p = manifest.as_deref().map(Path::new);
    let expected_ids = match expected_bench_ids_for_suite(&suite, manifest_p) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("\n❌ {e}");
            fail_check_with_receipt(
                &receipt_path,
                "regression_check",
                1,
                0,
                1,
                "EXPECTED_IDS_ERROR",
                &run_id,
            );
        }
    };

    match missing_expected_benchmarks(&log_text, &expected_ids) {
        Err(BaselineCoverageGap) => {
            eprintln!(
                "\n❌ BASELINE_COVERAGE_GAP: no executed benchmark could be parsed from {log_path}"
            );
            fail_check_with_receipt(
                &receipt_path,
                "regression_check",
                1,
                0,
                1,
                "BASELINE_COVERAGE_GAP",
                &run_id,
            );
        }
        Ok(missing) if !missing.is_empty() => {
            eprintln!(
                "\n❌ BASELINE_COVERAGE_GAP: expected benchmark(s) were not executed in {log_path}:"
            );
            for id in &missing {
                eprintln!("  {id}");
            }
            fail_check_with_receipt(
                &receipt_path,
                "regression_check",
                1,
                0,
                1,
                "BASELINE_COVERAGE_GAP",
                &run_id,
            );
        }
        Ok(_) => {}
    }

    match missing_baseline_coverage(&log_text, Path::new(&criterion_root), &baseline) {
        Err(BaselineCoverageGap) => {
            eprintln!(
                "\n❌ BASELINE_COVERAGE_GAP: no executed benchmark could be parsed from {log_path}"
            );
            fail_check_with_receipt(
                &receipt_path,
                "regression_check",
                1,
                0,
                1,
                "BASELINE_COVERAGE_GAP",
                &run_id,
            );
        }
        Ok(missing) if !missing.is_empty() => {
            eprintln!(
                "\n❌ BASELINE_COVERAGE_GAP: executed benchmark(s) without a saved baseline series under {criterion_root}/<id>/{baseline}/:"
            );
            for id in &missing {
                eprintln!("  {id}");
            }
            fail_check_with_receipt(
                &receipt_path,
                "regression_check",
                1,
                0,
                1,
                "BASELINE_COVERAGE_GAP",
                &run_id,
            );
        }
        Ok(_) => {}
    }

    let noise = parse_f64(&flags, "noise", "run").unwrap_or(DEFAULT_NOISE_THRESHOLD);
    let abs_noise_floor_ns = parse_f64(&flags, "abs-noise-floor-ns", "run");
    let check_stability = flags.get_bool("check-stability") || flags.get_bool("strict-stability");
    let strict_stability = flags.get_bool("strict-stability");
    let opts = VerdictOptions {
        noise,
        expected_ids: Some(expected_ids.clone()),
        check_stability,
        abs_noise_floor_ns,
    };

    // Evaluate statistical CI bounds from Criterion's `change/estimates.json` artifacts.
    let outcome = match regression_verdict_with_options(
        &log_text,
        Path::new(&criterion_root),
        &opts,
    ) {
        Err(VerdictBlind) => {
            eprintln!(
                "\n❌ REGRESSION VERDICT BLIND — Criterion comparison artifacts missing/unreadable"
            );
            fail_check_with_receipt(
                &receipt_path,
                "regression_check",
                1,
                0,
                1,
                "REGRESSION_BLIND",
                &run_id,
            );
        }
        Ok(out) => out,
    };

    let regressions: Vec<_> = outcome.regressions().collect();
    if !regressions.is_empty() {
        for f in &regressions {
            eprintln!("  ❌ Regression: {:?}", f);
        }
        eprintln!("\n❌ PERFORMANCE REGRESSION DETECTED");
        fail_check_with_receipt(
            &receipt_path,
            "regression_check",
            1,
            0,
            1,
            "REGRESSION_DETECTED",
            &run_id,
        );
    }

    let unstable: Vec<_> = outcome.unstable_baselines().collect();
    if strict_stability && !unstable.is_empty() {
        for f in &unstable {
            eprintln!("  ⚠️ Unstable baseline: {:?}", f);
        }
        eprintln!("\n❌ BILATERAL BASELINE INSTABILITY DETECTED (--strict-stability)");
        fail_check_with_receipt(
            &receipt_path,
            "regression_check",
            1,
            0,
            1,
            "UNSTABLE_BASELINE",
            &run_id,
        );
    }

    let executed = executed_bench_ids(&log_text);
    let executed_count = executed.len();
    if executed_count < 1 {
        eprintln!("\n❌ NO BENCHMARK EXECUTED — refusing to certify observed==expected");
        fail_check_with_receipt(
            &receipt_path,
            "regression_check",
            1,
            0,
            1,
            "NO_BENCHMARK_EXECUTED",
            &run_id,
        );
    }

    let expected_count = expected_ids.len();
    if executed_count != expected_count {
        eprintln!(
            "\n❌ BENCHMARK COVERAGE MISMATCH — observed {executed_count} != expected {expected_count}"
        );
        let _ = append_phase_receipt(
            &receipt_path,
            "regression_baseline_coverage",
            "FAIL",
            1,
            executed_count as u32,
            expected_count as u32,
            "EXPECTED_BENCHMARK_MISSING",
            &run_id,
        );
        exit(1);
    }

    let _ = append_phase_receipt(
        &receipt_path,
        "regression_baseline_coverage",
        "PASS",
        0,
        executed_count as u32,
        expected_count as u32,
        "",
        &run_id,
    );

    eprintln!(
        "  ✓ Coverage cross-check: all {executed_count} executed benchmark(s) have baseline series."
    );
    eprintln!("  ✓ No performance regression detected in {dur_str} (machine verdict).");

    let _ = append_phase_receipt(
        &receipt_path,
        "regression_check",
        "PASS",
        0,
        1,
        1,
        "",
        &run_id,
    );

    println!("\n================================================================================");
    println!("  Performance Gate Artifacts saved:");
    println!("    - Benchmark Log: {log_path}");
    println!("    - Receipt:       {receipt_path}");
    println!("================================================================================\n");

    exit(0);
}

fn print_run_help() {
    println!("Usage:");
    println!("  nam_perf_gate run [--check | --bootstrap-baseline] [--baseline <name>]");
    println!("                    [--baseline-dir <dir>] [--criterion-root <dir>]");
    println!("                    [--bench-core <core>] [--sysfs <dir>] [--suite <name>]");
    println!("                    [--manifest <path>] [--cooldown <seconds>]");
    println!("                    [--receipt <path>] [--log <path>] [--run-id <id>]");
    println!("                    [--noise <fraction>] [--abs-noise-floor-ns <nanoseconds>]");
    println!("                    [--check-stability] [--strict-stability]");
    println!("                    [--confirm-regressions] [--bench-exe <path>]");
    println!("                    [--strict]");
}
