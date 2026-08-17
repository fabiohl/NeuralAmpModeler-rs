// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Single environment probe (R-09) — the one place that classifies the host
//! for QA receipts and fingerprints.
//!
//! Ports the duplicated heuristics of `quality-dashboard.sh:103-130` and
//! `tests-performance-regression.sh:42-77` into one `EnvProbe`: cpuinfo
//! (`model name`, `flags` → canonical ISA string, `cpu cores`), toolchain
//! (`rustc --version`, host triple, `RUSTFLAGS`), frequency governor and git
//! state (`rev-parse HEAD`, dirty). S3 reuses this type for the fingerprint.
//!
//! CPU parsing is pure (`parse_cpuinfo`, `classify_isa`) and covered by the
//! fixtures of `tests/fixtures/qa/cpuinfo_*.txt`; only `EnvProbe::probe()`
//! touches the live system.

use std::fs;
use std::process::Command;

/// Canonical ISA string — AVX-512 class.
pub const ISA_AVX512: &str = "AVX-512";
/// Canonical ISA string — full x86-64-v3 feature set.
pub const ISA_X86_64_V3: &str = "x86-64-v3 (AVX2/FMA/F16C/BMI)";
/// Canonical ISA string — AVX2 present but the v3 set incomplete.
pub const ISA_AVX2_INCOMPLETE: &str = "AVX2 (incompleto / unsupported)";
/// Canonical ISA string — no AVX2.
pub const ISA_X86_64_BASE: &str = "x86-64 (base)";

/// Path of the live CPU information file.
pub const CPUINFO_PATH: &str = "/proc/cpuinfo";
/// Path of the live frequency governor of CPU 0.
pub const GOVERNOR_PATH: &str = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor";

/// CPU-related fields parsed from a `/proc/cpuinfo` text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuInfo {
    /// First `model name` line (empty when absent).
    pub model_name: String,
    /// Canonical ISA string classified from the first `flags` line.
    pub effective_isa: &'static str,
    /// First `cpu cores` field (0 when absent — the caller applies fallback).
    pub physical_cores: u32,
}

/// Parses a `/proc/cpuinfo` text — literal port of the bash greps/sed/awk of
/// `detect_cpu_model` / `detect_cpu_microarch` / `detect_physical_cores`.
pub fn parse_cpuinfo(text: &str) -> CpuInfo {
    let mut out = CpuInfo {
        model_name: String::new(),
        effective_isa: ISA_X86_64_BASE,
        physical_cores: 0,
    };
    for line in text.lines() {
        if line.starts_with("model name") && out.model_name.is_empty() {
            if let Some((_, value)) = line.split_once(':') {
                out.model_name = value.trim().to_string();
            }
        } else if line.starts_with("flags") {
            out.effective_isa = classify_isa(line);
        } else if line.starts_with("cpu cores")
            && out.physical_cores == 0
            && let Some(field) = line.split_whitespace().nth(3)
        {
            out.physical_cores = field.parse::<u32>().unwrap_or(0);
        }
    }
    out
}

/// Classifies a cpuinfo `flags` line into the one canonical ISA string.
///
/// Literal port of `detect_isa` (`quality-dashboard.sh:103-123`, mirrored at
/// `tests-performance-regression.sh:46-65`): `avx512f` wins; the full v3 set
/// (`avx`, `avx2`, `bmi1`, `bmi2`, `f16c`, `fma`, `lzcnt`|`abm`, `movbe`)
/// yields the v3 label; any bare `avx2` yields the incomplete label; else
/// base. Matching is whole-word, like `grep -w`.
pub fn classify_isa(flags_line: &str) -> &'static str {
    let flags: Vec<&str> = flags_line.split_whitespace().collect();
    let has = |flag: &str| flags.contains(&flag);
    if has("avx512f") {
        ISA_AVX512
    } else if has("avx")
        && has("avx2")
        && has("bmi1")
        && has("bmi2")
        && has("f16c")
        && has("fma")
        && (has("lzcnt") || has("abm"))
        && has("movbe")
    {
        ISA_X86_64_V3
    } else if has("avx2") {
        ISA_AVX2_INCOMPLETE
    } else {
        ISA_X86_64_BASE
    }
}

/// Full environment snapshot used by QA receipts and fingerprints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvProbe {
    /// First `model name` of `/proc/cpuinfo` (`unknown` when unavailable).
    pub cpu_model: String,
    /// Canonical ISA string — the single string of future receipts (R-09).
    pub effective_isa: &'static str,
    /// Physical cores (`cpu cores`), falling back to `nproc`, then 1.
    pub physical_cores: u32,
    /// First line of `rustc --version` (`unknown` when unavailable).
    pub rustc_version: String,
    /// `host:` triple of `rustc -vV` (`unknown` when unavailable).
    pub host_triple: String,
    /// `RUSTFLAGS` environment value (empty when unset).
    pub rustflags: String,
    /// `scaling_governor` of CPU 0 (`unknown` when unavailable).
    pub frequency_governor: String,
    /// `git rev-parse HEAD` of the working directory (`unknown` when not a repo).
    pub git_commit: String,
    /// Whether `git status --porcelain` produced output (dirty tree).
    pub git_dirty: bool,
}

impl EnvProbe {
    /// Probes the live host: `/proc/cpuinfo`, `rustc`, `RUSTFLAGS`, governor
    /// and the git state of the current working directory.
    pub fn probe() -> Self {
        let cpuinfo_text = fs::read_to_string(CPUINFO_PATH).unwrap_or_default();
        Self::probe_from_cpuinfo(&cpuinfo_text)
    }

    /// Probes with the CPU portion supplied (fixture-driven tests) and the
    /// toolchain/environment portion live.
    pub fn probe_from_cpuinfo(cpuinfo_text: &str) -> Self {
        let cpu = parse_cpuinfo(cpuinfo_text);
        let physical_cores = if cpu.physical_cores == 0 {
            probe_nproc().unwrap_or(1)
        } else {
            cpu.physical_cores
        };
        EnvProbe {
            cpu_model: if cpu.model_name.is_empty() {
                "unknown".to_string()
            } else {
                cpu.model_name
            },
            effective_isa: cpu.effective_isa,
            physical_cores,
            rustc_version: first_line_of_command(&["rustc", "--version"])
                .unwrap_or_else(|| "unknown".to_string()),
            host_triple: rustc_host_triple().unwrap_or_else(|| "unknown".to_string()),
            rustflags: std::env::var("RUSTFLAGS").unwrap_or_default(),
            frequency_governor: read_trimmed(GOVERNOR_PATH)
                .unwrap_or_else(|| "unknown".to_string()),
            git_commit: first_line_of_command(&["git", "rev-parse", "HEAD"])
                .unwrap_or_else(|| "unknown".to_string()),
            git_dirty: git_is_dirty(),
        }
    }
}

/// First line of the stdout of a command, trimmed — the same probe shape of
/// the bash `$(...)` assignments with `|| echo 'unknown'` fallbacks.
fn first_line_of_command(parts: &[&str]) -> Option<String> {
    let output = Command::new(parts[0]).args(&parts[1..]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .map(|line| line.trim().to_string())
}

/// `host:` triple of `rustc -vV` — the bash `sed -n 's/^host: //p'`.
fn rustc_host_triple() -> Option<String> {
    let output = Command::new("rustc").arg("-vV").output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| {
            line.strip_prefix("host: ")
                .map(|value| value.trim().to_string())
        })
}

/// Number of processors as reported by `nproc`.
fn probe_nproc() -> Option<u32> {
    let count = first_line_of_command(&["nproc"])?;
    count.parse::<u32>().ok().filter(|n| *n > 0)
}

/// Contents of a small status file, trimmed; `None` when unreadable/empty.
fn read_trimmed(path: &str) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Whether the working tree has uncommitted changes — the bash
/// `[ -n "$(git status --porcelain 2>/dev/null)" ]` of `write_build_metadata`.
fn git_is_dirty() -> bool {
    let output = Command::new("git").args(["status", "--porcelain"]).output();
    matches!(output, Ok(out) if out.status.success() && !out.stdout.is_empty())
}

#[cfg(test)]
#[path = "env_test.rs"]
mod env_test;
