// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Third-party mirror verification, ISA detection, and phase command runner.

use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

use neural_amp_modeler_rs::math::common::SIMD_MATH;
use neural_amp_modeler_rs::testing::qa::command::run_command_with_timeout;
use neural_amp_modeler_rs::testing::qa::env::{EnvProbe, ISA_AVX512, ISA_X86_64_V3};
use neural_amp_modeler_rs::testing::qa::verify::PROVENANCE_PHASE;
use neural_amp_modeler_rs::testing::receipt::tests_passed_from_output;

use crate::receipt::{PhaseReceiptRecord, append_line};

/// Returns `true` when the x86_64 host CPU exposes the full AVX-512 feature
/// matrix (`F + VL + BW + DQ`) required by the cross-ISA parity suite.
#[cfg(target_arch = "x86_64")]
pub(crate) fn host_has_full_avx512() -> bool {
    neural_amp_modeler_rs::math::common::avx512_capability_complete(
        is_x86_feature_detected!("avx512f"),
        is_x86_feature_detected!("avx512vl"),
        is_x86_feature_detected!("avx512bw"),
        is_x86_feature_detected!("avx512dq"),
    )
}

/// Returns the ISA actually active for engine dispatch (downscales AVX-512
/// host classification to x86-64-v3 when the `avx512` feature is disabled).
pub(crate) fn detect_active_isa(probe: &EnvProbe) -> &'static str {
    if cfg!(target_arch = "x86_64") && !SIMD_MATH.is_avx512 && probe.effective_isa == ISA_AVX512 {
        ISA_X86_64_V3
    } else {
        probe.effective_isa
    }
}

/// Non-x86_64 hosts never execute the x86 AVX2-vs-AVX-512 matrix.
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn host_has_full_avx512() -> bool {
    false
}

pub(crate) type CommandStep<'a> = (&'a [&'a str], &'a [(&'a str, &'a str)]);

// ── third-party provenance ──────────────────────────────────────────────────

const THIRD_PARTY_CORE: &str = "third-party/NeuralAmpModelerCore";
const THIRD_PARTY_PLUGIN: &str = "third-party/NeuralAmpModelerPlugin";

pub(crate) struct ThirdPartyCheck {
    pub(crate) status: &'static str,
    pub(crate) observed_records: u32,
    pub(crate) reason: String,
}

impl ThirdPartyCheck {
    fn fail(reason: String) -> Self {
        Self {
            status: "FAIL",
            observed_records: 0,
            reason,
        }
    }

    fn skip(reason: &str) -> Self {
        Self {
            status: "SKIP_CAPABILITY",
            observed_records: 0,
            reason: reason.to_string(),
        }
    }
}

/// Reads a pin from `variables.env` (`NAME=value`, quotes stripped) without
/// sourcing shell code — the same file is the single source of truth for
/// `utils/setup-third-party.sh`.
fn pinned_commit(name: &str) -> Option<String> {
    let text = fs::read_to_string("variables.env").ok()?;
    text.lines().find_map(|line| {
        let (key, value) = line.trim().split_once('=')?;
        (key.trim() == name).then(|| value.trim().trim_matches('"').to_string())
    })
}

/// First stdout line of a successful `git -C <dir> …` (`None` when git or the
/// command fails, which stays fail-closed at the call site).
fn git_line(dir: &str, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn verify_mirror(dir: &str, pin_name: &str, label: &str) -> Result<(), ThirdPartyCheck> {
    let Some(pin) = pinned_commit(pin_name) else {
        return Err(ThirdPartyCheck::fail(format!("pin_missing:{pin_name}")));
    };
    let Some(head) = git_line(dir, &["rev-parse", "HEAD"]) else {
        return Err(ThirdPartyCheck::fail(format!("{label}_not_git:{dir}")));
    };
    if head != pin {
        return Err(ThirdPartyCheck::fail(format!(
            "{label}_commit_mismatch:{head}"
        )));
    }
    match git_line(dir, &["status", "--porcelain"]) {
        Some(status) if status.is_empty() => {}
        Some(_) => return Err(ThirdPartyCheck::fail(format!("{label}_dirty"))),
        None => {
            return Err(ThirdPartyCheck::fail(format!("{label}_status_unavailable")));
        }
    }
    Ok(())
}

pub(crate) fn check_third_party() -> ThirdPartyCheck {
    if !Path::new(THIRD_PARTY_CORE).is_dir() {
        return ThirdPartyCheck::skip("third_party_absent");
    }
    if let Err(failure) = verify_mirror(THIRD_PARTY_CORE, "NAM_CORE_COMMIT", "third_party_core") {
        return failure;
    }
    // Only the core submodules back the render oracle (eigen, AudioDSPTools).
    // The plugin also pins IDE-only submodules that setup leaves uninitialized.
    let Some(submodules) = git_line(THIRD_PARTY_CORE, &["submodule", "status"]) else {
        return ThirdPartyCheck::fail("third_party_core_submodule_status_unavailable".to_string());
    };
    if let Some(bad) = submodules
        .lines()
        .find(|line| matches!(line.chars().next(), Some('-' | '+' | 'U')))
    {
        return ThirdPartyCheck::fail(format!("third_party_core_submodule:{bad}"));
    }
    if !Path::new(THIRD_PARTY_PLUGIN).is_dir() {
        return ThirdPartyCheck::skip("third_party_plugin_absent");
    }
    if let Err(failure) = verify_mirror(
        THIRD_PARTY_PLUGIN,
        "NAM_PLUGIN_COMMIT",
        "third_party_plugin",
    ) {
        return failure;
    }
    ThirdPartyCheck {
        status: "PASS",
        observed_records: 1,
        reason: String::new(),
    }
}

struct SequenceResult {
    exit_code: u32,
    tests_passed: u32,
    saw_test_result: bool,
    output_len: usize,
}

fn run_command_sequence(commands: &[CommandStep<'_>], log_path: &Path) -> SequenceResult {
    let timeout_s: u64 = env::var("NAM_PHASE_TIMEOUT_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let timeout = std::time::Duration::from_secs(timeout_s);

    let mut all_output = Vec::new();
    let mut final_exit = 0;

    for (args, envs) in commands {
        let mut cmd = Command::new(args[0]);
        cmd.args(&args[1..]);
        for (k, v) in *envs {
            cmd.env(k, v);
        }

        let result = run_command_with_timeout(&mut cmd, timeout);
        all_output.extend_from_slice(&result.output);
        if result.timed_out {
            let msg = format!(
                "command {} timed out after {}s (process tree killed)\n",
                args[0], timeout_s
            );
            all_output.extend_from_slice(msg.as_bytes());
        }
        if result.exit_code != 0 {
            final_exit = result.exit_code;
            break;
        }
    }

    if let Err(e) = fs::write(log_path, &all_output) {
        let msg = format!("failed to write log file {}: {e}\n", log_path.display());
        eprintln!("{msg}");
        all_output.extend_from_slice(msg.as_bytes());
        if final_exit == 0 {
            final_exit = 1;
        }
    }

    let (tests_passed, saw_test_result) = tests_passed_from_output(&all_output);
    SequenceResult {
        exit_code: final_exit,
        tests_passed,
        saw_test_result,
        output_len: all_output.len(),
    }
}

pub(crate) fn run_dashboard_phase(
    phase_id: &str,
    commands: &[CommandStep<'_>],
    log_path: &Path,
    receipt_path: &str,
    run_id: &str,
    min_observed: u32,
    expected_records: u32,
) -> bool {
    println!("-> Running {phase_id}...");
    let result = run_command_sequence(commands, log_path);
    let exit_code = result.exit_code;
    // `freshness` is a provenance probe, not a libtest run: one successful
    // non-empty observation is its record. Every other phase must report a
    // canonical `test result:` line — silence is never a pass.
    let observed = if phase_id == PROVENANCE_PHASE {
        u32::from(exit_code == 0 && result.output_len > 0)
    } else {
        result.tests_passed
    };
    let (status, reason) = if exit_code != 0 {
        if exit_code == 124 {
            ("FAIL", "phase command timed out")
        } else if phase_id == PROVENANCE_PHASE {
            if exit_code == 3 || exit_code == 5 {
                ("FAIL", "generator_drift")
            } else {
                ("FAIL", "freshness_failed")
            }
        } else {
            ("FAIL", "test command failed")
        }
    } else if phase_id != PROVENANCE_PHASE && !result.saw_test_result {
        ("FAIL", "no_test_result")
    } else if observed < min_observed && min_observed > 0 {
        ("FAIL", "insufficient test cases executed")
    } else {
        ("PASS", "")
    };

    let rec = PhaseReceiptRecord {
        phase_id,
        status,
        exit_code,
        observed_records: observed,
        expected_records: expected_records.max(min_observed),
        reason,
        run_id,
    };
    let rec_str = match serde_json::to_string(&rec) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to serialize receipt record: {e}");
            return false;
        }
    };
    if let Err(e) = append_line(receipt_path, &rec_str) {
        eprintln!("failed to append line to receipt {receipt_path}: {e}");
        return false;
    }
    if status == "PASS" {
        println!("   ok {phase_id} (exit={exit_code}, {observed} cases)");
    } else {
        println!(
            "   FAIL {phase_id} (exit={exit_code}, {observed}/{} cases, reason={reason})",
            expected_records.max(min_observed)
        );
    }
    status == "PASS"
}
