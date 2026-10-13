// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared `/proc` process telemetry readers (best-effort, never panicking).

/// Reads the current process resident set size in KiB from `/proc/self/status`.
///
/// Returns `0` when the file is missing or the field cannot be parsed.
pub fn read_rss_kb() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
        }
    }
    0
}

/// Reads the minor/major page-fault totals from `/proc/self/stat` (fields 10/12).
///
/// Returns `(0, 0)` when the file is missing or malformed.
pub fn read_page_faults() -> (u64, u64) {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let Some(tail) = stat.rfind(')') else {
        return (0, 0);
    };
    let fields: Vec<&str> = stat[tail + 1..].split_whitespace().collect();
    let minflt = fields
        .get(7)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let majflt = fields
        .get(9)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    (minflt, majflt)
}

/// Reads the thread count (`Threads:` line of `/proc/self/status`).
///
/// Returns `0` when the file is missing or the field cannot be parsed.
pub fn read_thread_count() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("Threads:") {
            return rest.trim().parse::<usize>().unwrap_or(0);
        }
    }
    0
}

/// Reads the number of open file descriptors (`/proc/self/fd` entries).
///
/// Returns `0` when `/proc/self/fd` is unreadable.
pub fn read_fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .map(|it| it.count())
        .unwrap_or(0)
}

/// One periodic process telemetry sample (raw values — never differenced).
#[derive(Debug, Clone, Copy)]
pub struct TelemetrySample {
    /// Resident set size in KiB.
    pub rss_kb: usize,
    /// Minor page faults since process start.
    pub minflt: u64,
    /// Major page faults since process start.
    pub majflt: u64,
    /// Live thread count.
    pub threads: usize,
    /// Open file-descriptor count.
    pub fds: usize,
}

impl TelemetrySample {
    /// Captures a fresh sample of all process counters.
    pub fn capture() -> Self {
        let (minflt, majflt) = read_page_faults();
        Self {
            rss_kb: read_rss_kb(),
            minflt,
            majflt,
            threads: read_thread_count(),
            fds: read_fd_count(),
        }
    }
}

/// Thread resource usage counters from `getrusage(RUSAGE_THREAD)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThreadRusage {
    /// Minor page faults in the thread.
    pub minflt: u64,
    /// Major page faults in the thread.
    pub majflt: u64,
    /// Voluntary context switches.
    pub nvcsw: u64,
    /// Involuntary context switches.
    pub nivcsw: u64,
}

/// Reads resource usage counters for the calling thread via `getrusage(RUSAGE_THREAD)`.
///
/// Returns an error if the kernel/libc does not support `RUSAGE_THREAD`, allowing
/// callers to fall back to `/proc/thread-self` with a declared gap marker.
pub fn read_thread_rusage() -> Result<ThreadRusage, std::io::Error> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` writes up to `size_of::<libc::rusage>()` bytes into
    // the pointer, which is valid and writable for that size.
    let ret = unsafe { libc::getrusage(libc::RUSAGE_THREAD, usage.as_mut_ptr()) };
    if ret == 0 {
        // SAFETY: `getrusage` returned 0, so it initialized the whole struct.
        let usage = unsafe { usage.assume_init() };
        return Ok(ThreadRusage {
            minflt: usage.ru_minflt as u64,
            majflt: usage.ru_majflt as u64,
            nvcsw: usage.ru_nvcsw as u64,
            nivcsw: usage.ru_nivcsw as u64,
        });
    }
    read_thread_rusage_proc()
}

/// Fallback thread-usage reader using `/proc/thread-self/{stat,status}`.
pub fn read_thread_rusage_proc() -> Result<ThreadRusage, std::io::Error> {
    let stat = std::fs::read_to_string("/proc/thread-self/stat")?;
    let tail = stat.rfind(')').ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid stat format")
    })?;
    let fields: Vec<&str> = stat[tail + 1..].split_whitespace().collect();
    let minflt = fields
        .get(7)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let majflt = fields
        .get(9)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let status = std::fs::read_to_string("/proc/thread-self/status").unwrap_or_default();
    let mut nvcsw = 0u64;
    let mut nivcsw = 0u64;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("voluntary_ctxt_switches:") {
            nvcsw = rest.trim().parse::<u64>().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("nonvoluntary_ctxt_switches:") {
            nivcsw = rest.trim().parse::<u64>().unwrap_or(0);
        }
    }

    Ok(ThreadRusage {
        minflt,
        majflt,
        nvcsw,
        nivcsw,
    })
}
