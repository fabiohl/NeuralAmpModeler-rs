// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Deterministic CPU core resolution for benchmarks and real-time execution.
//!
//! Implements the strict resolution order mandated by script and harness policy:
//! 1. Explicit core value (`NAM_BENCH_CORE` or CLI argument) if set and not "auto".
//! 2. First online isolated CPU from `/sys/devices/system/cpu/isolated`.
//! 3. First permitted CPU from process affinity mask / sysfs online CPUs.
//! 4. Unpinned fallback (`GAP`).

use std::fs;
use std::path::Path;

/// Default sysfs CPU base directory on Linux hosts.
pub const DEFAULT_SYSFS_CPU_DIR: &str = "/sys/devices/system/cpu";

/// Parses standard Linux CPU list syntax (e.g. "0", "1-3", "0-3,7,9-10")
/// into a sorted, unique vector of CPU indices.
pub fn parse_cpu_list(text: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in text.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((start_s, end_s)) = trimmed.split_once('-') {
            if let (Ok(start), Ok(end)) = (
                start_s.trim().parse::<usize>(),
                end_s.trim().parse::<usize>(),
            ) && start <= end
            {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = trimmed.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

/// Checks whether a given CPU index is currently online according to sysfs.
///
/// Convention:
/// - If `sysfs_cpu/cpu<N>/online` exists, reads trimmed value: `"1"` means online, `"0"` offline.
/// - If `sysfs_cpu/cpu<N>/online` does not exist but `sysfs_cpu/cpu<N>` directory exists,
///   it is considered online (on Linux, CPU 0 is typically non-hotpluggable and lacks the online file).
/// - If `sysfs_cpu/cpu<N>` does not exist at all, returns false.
pub fn is_cpu_online(sysfs_cpu: &Path, cpu: usize) -> bool {
    let cpu_dir = sysfs_cpu.join(format!("cpu{cpu}"));
    if !cpu_dir.is_dir() {
        return false;
    }
    let online_file = cpu_dir.join("online");
    if online_file.is_file() {
        if let Ok(content) = fs::read_to_string(&online_file) {
            return content.trim() == "1";
        }
        return false;
    }
    true
}

/// Returns the CPU indices allowed for the current process via `sched_getaffinity`.
#[cfg(target_os = "linux")]
pub fn process_allowed_cpus() -> Vec<usize> {
    let mut allowed = Vec::new();
    // SAFETY: zeroing a libc::cpu_set_t POD struct is valid initialization for cpu_set_t.
    let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: CPU_ZERO initializes the cpuset structure, and sched_getaffinity
    // queries the affinity mask of the current process (pid 0) with a valid pointer and size.
    let ok = unsafe {
        libc::CPU_ZERO(&mut cpuset);
        libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut cpuset) == 0
    };
    if ok {
        for i in 0..libc::CPU_SETSIZE as usize {
            // SAFETY: cpuset is a valid cpu_set_t and i is within bounds (0..CPU_SETSIZE).
            if unsafe { libc::CPU_ISSET(i, &cpuset) } {
                allowed.push(i);
            }
        }
    }
    allowed
}

/// Returns the CPU indices allowed for the current process on non-Linux targets (empty fallback).
#[cfg(not(target_os = "linux"))]
pub fn process_allowed_cpus() -> Vec<usize> {
    Vec::new()
}

/// Resolves the benchmark CPU core according to strict policy:
/// 1. Explicit core value (`NAM_BENCH_CORE` or CLI argument) if set and not "auto".
/// 2. First online isolated CPU from sysfs `isolated`.
/// 3. First permitted CPU from process affinity mask / sysfs `online`.
/// 4. `None` (`GAP` / unpinned).
pub fn resolve_bench_core(explicit: Option<&str>, sysfs_dir: Option<&Path>) -> Option<u32> {
    resolve_bench_core_with_affinity(explicit, sysfs_dir, None)
}

/// Resolves the benchmark CPU core with an optional affinity override (for test isolation).
pub fn resolve_bench_core_with_affinity(
    explicit: Option<&str>,
    sysfs_dir: Option<&Path>,
    allowed_override: Option<&[usize]>,
) -> Option<u32> {
    // 1. Explicit override
    if let Some(raw) = explicit {
        let trimmed = raw.trim();
        if !trimmed.is_empty()
            && trimmed != "auto"
            && let Ok(core) = trimmed.parse::<u32>()
        {
            return Some(core);
        }
    }

    let sysfs = sysfs_dir.unwrap_or_else(|| Path::new(DEFAULT_SYSFS_CPU_DIR));
    // When a custom/mock sysfs directory is supplied without explicit allowed_override,
    // we do not constrain by the host process affinity mask.
    let allowed = match allowed_override {
        Some(list) => list.to_vec(),
        None => {
            if sysfs_dir.is_some() {
                Vec::new()
            } else {
                process_allowed_cpus()
            }
        }
    };

    // 2. Inspect sysfs isolated cpus
    let isolated_file = sysfs.join("isolated");
    if let Ok(content) = fs::read_to_string(&isolated_file) {
        let candidates = parse_cpu_list(&content);
        for c in candidates {
            let is_allowed = allowed.is_empty() || allowed.contains(&c);
            if is_allowed && is_cpu_online(sysfs, c) {
                return Some(c as u32);
            }
        }
    }

    // 3. First permitted CPU from process affinity mask or sysfs online
    if !allowed.is_empty() {
        for &c in &allowed {
            let cpu_dir = sysfs.join(format!("cpu{c}"));
            if cpu_dir.is_dir() {
                if is_cpu_online(sysfs, c) {
                    return Some(c as u32);
                }
            } else {
                return Some(c as u32);
            }
        }
    }

    // Fallback: check sysfs online file
    let online_file = sysfs.join("online");
    if let Ok(content) = fs::read_to_string(&online_file) {
        let online_list = parse_cpu_list(&content);
        for c in online_list {
            if is_cpu_online(sysfs, c) {
                return Some(c as u32);
            }
        }
    }

    // Check CPU 0 if present
    if is_cpu_online(sysfs, 0) {
        return Some(0);
    }

    // 4. Sem afinidade (GAP)
    None
}
