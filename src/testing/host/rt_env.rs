// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Real-time environment calibration primitives (best-effort, never panicking).

/// Reads the process's `Cpus_allowed_list` from `/proc/self/status`.
///
/// Returns an empty list when the file is missing or the field is absent.
pub fn cpus_allowed_list() -> Vec<u32> {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
            return parse_cpu_ranges(rest.trim());
        }
    }
    Vec::new()
}

/// Parses the kernel's `/sys/devices/system/cpu/isolated` CPU ids.
///
/// Returns an empty list when the sysfs node is missing.
pub fn isolated_cpus() -> Vec<u32> {
    let s = std::fs::read_to_string("/sys/devices/system/cpu/isolated").unwrap_or_default();
    parse_cpu_ranges(s.trim())
}

/// Expands `"0,2-4"`-style CPU ranges into a flat id list.
///
/// Tolerant of stray whitespace and malformed parts (which are skipped or
/// parsed leniently), so it never panics on unexpected sysfs content.
pub fn parse_cpu_ranges(s: &str) -> Vec<u32> {
    let mut cpus = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: u32 = lo.trim().parse().unwrap_or(0);
            let hi: u32 = hi.trim().parse().unwrap_or(lo);
            cpus.extend(lo..=hi);
        } else if let Ok(c) = part.parse::<u32>() {
            cpus.push(c);
        }
    }
    cpus
}

/// Tries to set `SCHED_FIFO` with the given priority on the current thread.
///
/// Returns `false` when the caller lacks `CAP_SYS_NICE` (the common,
/// sandboxed case) instead of panicking.
pub fn try_promote_sched_fifo(priority: i32) -> bool {
    let param = libc::sched_param {
        sched_priority: priority,
    };
    // SAFETY: `param` is a fully initialized `sched_param`; the syscall only
    // reads it and targets the calling thread (`0`).
    unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) == 0 }
}

/// Whether the calling thread currently runs under `SCHED_FIFO`.
pub fn current_sched_fifo() -> bool {
    // SAFETY: `sched_getscheduler` only reads the scheduling policy of the
    // calling thread (`0`) and returns an integer; no pointers involved.
    unsafe { libc::sched_getscheduler(0) == libc::SCHED_FIFO }
}

/// Reads the cpufreq `scaling_governor` for the given CPU (or `cpu0`).
///
/// Returns `"unknown"` when the sysfs node is missing (containers, VMs).
pub fn read_cpu_governor(cpu: Option<u32>) -> String {
    let path = match cpu {
        Some(id) => format!("/sys/devices/system/cpu/cpu{id}/cpufreq/scaling_governor"),
        None => "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor".to_string(),
    };
    std::fs::read_to_string(&path)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}
