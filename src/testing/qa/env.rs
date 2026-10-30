// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Single environment probe (R-09) — the one place that classifies the host
//! for QA receipts and fingerprints.
//!
//! Ports the duplicated heuristics of `quality-dashboard.sh:103-130` and
//! o script de performance legado (removido) into one `EnvProbe`: cpuinfo
//! (`model name`, `flags` → canonical ISA string, `cpu cores`), toolchain
//! (`rustc --version`, host triple, `RUSTFLAGS`), frequency governor and git
//! state (`rev-parse HEAD`, dirty). S3 reuses this type for the fingerprint.
//!
//! CPU parsing is pure (`parse_cpuinfo`, `classify_isa`) and covered by the
//! fixtures of `tests/fixtures/qa/cpuinfo_*.txt`; only `EnvProbe::probe()`
//! touches the live system.

use std::fs;
use std::process::Command;

use serde::{Deserialize, Serialize};

use super::pick_core;

/// Canonical ISA string — AVX-512 class.
pub const ISA_AVX512: &str = "AVX-512";
/// Canonical ISA string — full x86-64-v3 feature set.
pub const ISA_X86_64_V3: &str = "x86-64-v3 (AVX2/FMA/F16C/BMI)";
/// Canonical ISA string — AVX2 present but the v3 set incomplete.
pub const ISA_AVX2_INCOMPLETE: &str = "AVX2 (incompleto / unsupported)";
/// Canonical ISA string — no AVX2.
pub const ISA_X86_64_BASE: &str = "x86-64 (base)";
/// Canonical ISA string — 64-bit ARM NEON baseline.
pub const ISA_AARCH64_NEON: &str = "aarch64 (NEON)";
/// Canonical ISA string — macOS Apple Silicon (ARM64 with NEON).
pub const ISA_AARCH64_APPLE_SILICON: &str = "aarch64 (Apple Silicon NEON)";
/// Canonical ISA string — unknown or unverified architecture.
pub const ISA_UNKNOWN: &str = "unknown";

/// Canonical hardware classes for performance benchmarking and baseline comparability (R-12, R-23, S6-T03).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareClass {
    /// AMD Zen 3 laptop (Ryzen 7 5700U with Radeon Graphics, x86-64-v3) - reference baseline.
    X86Zen3Laptop,
    /// AMD Zen server/workstation class.
    X86AmdZen,
    /// Intel Client (Core i5/i7/i9 with AVX2/FMA).
    X86IntelClient,
    /// x86-64 server with AVX-512 capability (e.g. AWS c8id / Xeon / EPYC Genoa+).
    X86Avx512Server,
    /// Generic x86-64-v3 compliant host.
    X86GenericV3,
    /// Apple Silicon M-series (macOS ARM64 with NEON).
    AArch64AppleSilicon,
    /// Linux AArch64 server (e.g. AWS Graviton 3/4, Neoverse N1/V1).
    AArch64LinuxServer,
    /// Generic AArch64 host with NEON.
    AArch64Generic,
    /// Unknown or unclassified hardware class.
    Unknown,
}

impl HardwareClass {
    /// Canonical identifier string for fingerprints, contracts, and receipts.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::X86Zen3Laptop => "x86_zen3_laptop",
            Self::X86AmdZen => "x86_amd_zen",
            Self::X86IntelClient => "x86_intel_client",
            Self::X86Avx512Server => "x86_avx512_server",
            Self::X86GenericV3 => "x86_generic_v3",
            Self::AArch64AppleSilicon => "aarch64_apple_silicon",
            Self::AArch64LinuxServer => "aarch64_linux_server",
            Self::AArch64Generic => "aarch64_generic",
            Self::Unknown => "unknown",
        }
    }

    /// Parses a hardware class from a string name.
    pub fn from_str_name(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "x86_zen3_laptop" | "zen3_laptop" => Self::X86Zen3Laptop,
            "x86_amd_zen" | "amd_zen" => Self::X86AmdZen,
            "x86_intel_client" | "intel_client" => Self::X86IntelClient,
            "x86_avx512_server" | "avx512_server" => Self::X86Avx512Server,
            "x86_generic_v3" | "generic_v3" => Self::X86GenericV3,
            "aarch64_apple_silicon" | "apple_silicon" => Self::AArch64AppleSilicon,
            "aarch64_linux_server" | "linux_server" => Self::AArch64LinuxServer,
            "aarch64_generic" => Self::AArch64Generic,
            _ => Self::Unknown,
        }
    }
}

/// Classifies host into a canonical hardware class based on CPU model, ISA, and host triple.
pub fn classify_hardware_class(
    cpu_model: &str,
    effective_isa: &str,
    host_triple: &str,
) -> HardwareClass {
    let model_lower = cpu_model.to_lowercase();
    let triple_lower = host_triple.to_lowercase();

    if triple_lower.contains("apple") || triple_lower.contains("darwin") {
        return HardwareClass::AArch64AppleSilicon;
    }

    if triple_lower.contains("aarch64") {
        if model_lower.contains("neoverse") || model_lower.contains("graviton") {
            return HardwareClass::AArch64LinuxServer;
        }
        return HardwareClass::AArch64Generic;
    }

    if effective_isa == ISA_AVX512 {
        return HardwareClass::X86Avx512Server;
    }

    if model_lower.contains("5700u") && model_lower.contains("ryzen") {
        return HardwareClass::X86Zen3Laptop;
    }

    if model_lower.contains("amd") || model_lower.contains("ryzen") || model_lower.contains("epyc")
    {
        return HardwareClass::X86AmdZen;
    }

    if model_lower.contains("intel") || model_lower.contains("core") || model_lower.contains("xeon")
    {
        return HardwareClass::X86IntelClient;
    }

    if effective_isa == ISA_X86_64_V3 {
        return HardwareClass::X86GenericV3;
    }

    HardwareClass::Unknown
}

/// Path of the live CPU information file.
pub const CPUINFO_PATH: &str = "/proc/cpuinfo";
/// Builds the `scaling_governor` path for an explicit CPU core.
///
/// Pure constructor (no I/O) — unit-tested with a mocked core id.
pub fn governor_path_for_core(core: u32) -> String {
    format!("/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor")
}

/// Effective bench core for governor probing: the single owner of the
/// precedence (`pick_core::resolve_bench_core`): `NAM_BENCH_CORE` → first
/// online isolated CPU → first permitted CPU. `None` (GAP) keeps the probe
/// fail-closed (`unknown`).
pub fn effective_bench_core() -> Option<u32> {
    pick_core::resolve_bench_core(std::env::var("NAM_BENCH_CORE").ok().as_deref(), None)
}

/// Path of the live frequency governor of the effective bench core.
pub fn effective_governor_path() -> Option<String> {
    effective_bench_core().map(governor_path_for_core)
}

/// `scaling_governor` of the effective bench core (`None` when unreadable —
/// the caller maps to `unknown`, preserving the fail-closed
/// `INCOMPARABLE_ENVIRONMENT` semantics on VMs/CI without per-core
/// `cpufreq`).
pub fn read_effective_governor() -> Option<String> {
    read_trimmed(&effective_governor_path()?)
}

/// Governor path for an explicit bench-core string (e.g. the
/// `Fingerprint::probe` argument): a parseable value wins, otherwise the
/// effective bench core. Never falls back to `cpu0` — a missing per-core
/// path stays fail-closed (`unknown`).
pub fn governor_path_for_bench_core_arg(bench_core: &str) -> Option<String> {
    let trimmed = bench_core.trim();
    if !trimmed.is_empty()
        && let Ok(core) = trimmed.parse::<u32>()
    {
        return Some(governor_path_for_core(core));
    }
    effective_governor_path()
}

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
        effective_isa: ISA_UNKNOWN,
        physical_cores: 0,
    };
    if text.is_empty() {
        return out;
    }
    let mut found_isa = false;
    for line in text.lines() {
        if (line.starts_with("model name")
            || line.starts_with("Model")
            || line.starts_with("Hardware"))
            && out.model_name.is_empty()
        {
            if let Some((_, value)) = line.split_once(':') {
                out.model_name = value.trim().to_string();
            }
        } else if line.starts_with("flags") {
            out.effective_isa = classify_isa(line);
            found_isa = true;
        } else if line.starts_with("Features") && !found_isa {
            out.effective_isa = classify_arm_features(line);
            found_isa = true;
        } else if line.starts_with("cpu cores")
            && out.physical_cores == 0
            && let Some(field) = line.split_whitespace().nth(3)
        {
            out.physical_cores = field.parse::<u32>().unwrap_or(0);
        }
    }
    out
}

/// Classifies a cpuinfo ARM `Features` line.
pub fn classify_arm_features(features_line: &str) -> &'static str {
    let flags: Vec<&str> = features_line.split_whitespace().collect();
    if flags.contains(&"asimd") || flags.contains(&"neon") {
        ISA_AARCH64_NEON
    } else {
        ISA_UNKNOWN
    }
}

/// Classifies a cpuinfo `flags` line into the one canonical ISA string.
///
/// Literal port of `detect_isa` (`quality-dashboard.sh:103-123`, mirrored at
/// o script de performance legado (removido)): `avx512f` wins; the full v3 set
/// (`avx`, `avx2`, `bmi1`, `bmi2`, `f16c`, `fma`, `lzcnt`|`abm`, `movbe`)
/// yields the v3 label; any bare `avx2` yields the incomplete label; else
/// base. Matching is whole-word, like `grep -w`.
pub fn classify_isa(flags_line: &str) -> &'static str {
    let flags: Vec<&str> = flags_line.split_whitespace().collect();
    let has = |flag: &str| flags.contains(&flag);
    if has("avx512f") && has("avx512vl") {
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvProbe {
    /// First `model name` of `/proc/cpuinfo` (`unknown` when unavailable).
    pub cpu_model: String,
    /// Canonical ISA string — the single string of future receipts (R-09).
    pub effective_isa: &'static str,
    /// Canonical hardware class for baseline comparability (R-12, R-23, S6-T03).
    pub hardware_class: HardwareClass,
    /// Physical cores (`cpu cores`), falling back to `nproc`, then 1.
    pub physical_cores: u32,
    /// First line of `rustc --version` (`unknown` when unavailable).
    pub rustc_version: String,
    /// `host:` triple of `rustc -vV` (`unknown` when unavailable).
    pub host_triple: String,
    /// `RUSTFLAGS` environment value (empty when unset).
    pub rustflags: String,
    /// `scaling_governor` of the effective bench core (`unknown` when unavailable).
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
        let mut cpu = parse_cpuinfo(cpuinfo_text);
        if cpu.model_name.is_empty() && cfg!(target_os = "macos") {
            if let Some(model) =
                first_line_of_command(&["sysctl", "-n", "machdep.cpu.brand_string"])
                    .or_else(|| first_line_of_command(&["sysctl", "-n", "hw.model"]))
            {
                cpu.model_name = model;
            }
            if cfg!(target_arch = "aarch64") {
                cpu.effective_isa = ISA_AARCH64_APPLE_SILICON;
            }
        }
        if cpu.effective_isa == ISA_UNKNOWN && cfg!(target_arch = "aarch64") {
            cpu.effective_isa = ISA_AARCH64_NEON;
        }

        let physical_cores = if cpu.physical_cores == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1)
        } else {
            cpu.physical_cores
        };
        let host_triple = rustc_host_triple().unwrap_or_else(|| "unknown".to_string());
        let cpu_model_str = if cpu.model_name.is_empty() {
            "unknown".to_string()
        } else {
            cpu.model_name
        };
        let hardware_class =
            classify_hardware_class(&cpu_model_str, cpu.effective_isa, &host_triple);

        EnvProbe {
            cpu_model: cpu_model_str,
            effective_isa: cpu.effective_isa,
            hardware_class,
            physical_cores,
            rustc_version: first_line_of_command(&["rustc", "--version"])
                .unwrap_or_else(|| "unknown".to_string()),
            host_triple,
            rustflags: std::env::var("RUSTFLAGS").unwrap_or_default(),
            // Probe the governor of the effective bench core, not a
            // fixed `cpu0` — on hybrid (P-core/E-core) systems `cpu0` may
            // report `performance` while the pinned bench core does not.
            // Missing per-core `cpufreq` (VM/CI) stays `unknown` (fail-closed).
            frequency_governor: read_effective_governor().unwrap_or_else(|| "unknown".to_string()),
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
mod tests;
