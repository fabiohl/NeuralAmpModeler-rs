// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Real-time environment calibration primitives (best-effort, never panicking).

/// Largest CPU index accepted by [`try_parse_cpu_ranges`].
///
/// Linux affinity masks are `cpu_set_t` values limited to `CPU_SETSIZE`
/// entries; `sched_getaffinity`/`CPU_SET` cannot represent an index outside
/// `[0, CPU_SETSIZE)`. Rejecting larger ids keeps every parsed target
/// pin-able by this crate instead of silently returning unreachable CPUs.
const MAX_CPU_LIST_ENTRIES: usize = libc::CPU_SETSIZE as usize;

/// Longest accepted input for [`try_parse_cpu_ranges`] (64 KiB).
const MAX_CPU_LIST_INPUT_BYTES: usize = 64 * 1024;

/// Why a CPU list could not be expanded within budget.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuListError {
    /// The input string exceeds the accepted length.
    InputTooLong {
        /// Observed length in bytes.
        len: usize,
        /// Accepted maximum.
        limit: usize,
    },
    /// A CPU id cannot be represented in a Linux affinity mask.
    CpuIndexOutOfRange {
        /// Offending id.
        cpu: u32,
        /// Exclusive upper bound.
        limit: usize,
    },
    /// The list expands to more entries than the budget allows.
    TooManyCpus {
        /// Maximum number of expanded entries.
        limit: usize,
    },
}

impl std::fmt::Display for CpuListError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputTooLong { len, limit } => {
                write!(f, "CPU list input too long ({len} > {limit} bytes)")
            }
            Self::CpuIndexOutOfRange { cpu, limit } => {
                write!(f, "CPU id {cpu} is outside [0, {limit})")
            }
            Self::TooManyCpus { limit } => {
                write!(f, "CPU list expands to more than {limit} entries")
            }
        }
    }
}

impl std::error::Error for CpuListError {}

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
/// Tolerant of stray whitespace and malformed parts (which are skipped), so it
/// never panics on unexpected sysfs content. Oversized input, ids outside
/// `[0, CPU_SETSIZE)` and lists that expand past the budget yield an empty
/// list rather than a partial selection.
pub fn parse_cpu_ranges(s: &str) -> Vec<u32> {
    try_parse_cpu_ranges(s).unwrap_or_default()
}

/// Expands `"0,2-4"`-style CPU ranges under an explicit resource budget.
///
/// Malformed parts are skipped (no fabricated ids). Budget violations are
/// reported as [`CpuListError`] and never return a partial expansion.
pub fn try_parse_cpu_ranges(s: &str) -> Result<Vec<u32>, CpuListError> {
    if s.len() > MAX_CPU_LIST_INPUT_BYTES {
        return Err(CpuListError::InputTooLong {
            len: s.len(),
            limit: MAX_CPU_LIST_INPUT_BYTES,
        });
    }
    let mut cpus = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        if let Some((lo, hi)) = part.split_once('-') {
            let (Ok(lo), Ok(hi)) = (lo.trim().parse::<u32>(), hi.trim().parse::<u32>()) else {
                continue;
            };
            if lo > hi {
                continue;
            }
            if hi as usize >= MAX_CPU_LIST_ENTRIES {
                return Err(CpuListError::CpuIndexOutOfRange {
                    cpu: hi,
                    limit: MAX_CPU_LIST_ENTRIES,
                });
            }
            let span = (hi - lo) as usize + 1;
            if span > MAX_CPU_LIST_ENTRIES - cpus.len() {
                return Err(CpuListError::TooManyCpus {
                    limit: MAX_CPU_LIST_ENTRIES,
                });
            }
            cpus.extend(lo..=hi);
        } else if let Ok(cpu) = part.parse::<u32>() {
            if cpu as usize >= MAX_CPU_LIST_ENTRIES {
                return Err(CpuListError::CpuIndexOutOfRange {
                    cpu,
                    limit: MAX_CPU_LIST_ENTRIES,
                });
            }
            if cpus.len() >= MAX_CPU_LIST_ENTRIES {
                return Err(CpuListError::TooManyCpus {
                    limit: MAX_CPU_LIST_ENTRIES,
                });
            }
            cpus.push(cpu);
        }
    }
    Ok(cpus)
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
