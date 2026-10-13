// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for the host probes.
//!
//! These assert the *absence tolerance* contract — a missing sysfs node, a
//! denied PMU, or a non-existent CPU must never panic — plus the pure parsing
//! helpers. Live values are environment-dependent, so only the invariants that
//! hold anywhere on Linux are checked.

use super::{pmu::PmuCounters, proc, rt_env};

#[test]
fn parse_cpu_ranges_expands_and_tolerates() {
    assert_eq!(rt_env::parse_cpu_ranges("0,2-4"), vec![0, 2, 3, 4]);
    assert_eq!(rt_env::parse_cpu_ranges(" 5 "), vec![5]);
    assert_eq!(rt_env::parse_cpu_ranges(""), Vec::<u32>::new());
    assert_eq!(rt_env::parse_cpu_ranges("3-3"), vec![3]);
    // A malformed part must not panic.
    let _ = rt_env::parse_cpu_ranges("a-b,");
}

#[test]
fn governor_of_missing_cpu_is_unknown() {
    // A CPU id far beyond any real topology has no sysfs node.
    assert_eq!(rt_env::read_cpu_governor(Some(u32::MAX)), "unknown");
    // The default path must return a string without panicking.
    let _ = rt_env::read_cpu_governor(None);
}

#[test]
fn scheduler_probes_never_panic() {
    // Promotion is expected to fail without CAP_SYS_NICE; either way it returns.
    let _ = rt_env::try_promote_sched_fifo(80);
    let _ = rt_env::current_sched_fifo();
}

#[test]
fn proc_telemetry_is_consistent() {
    let _ = proc::read_rss_kb();
    let _ = proc::read_page_faults();
    let sample = proc::TelemetrySample::capture();
    // A running process always has at least one thread.
    assert!(sample.threads >= 1, "thread count must be >= 1");
    // The fd table always contains at least stdin/stdout/stderr.
    assert!(sample.fds >= 3, "fd count must be >= 3");
}

#[test]
fn thread_rusage_reports_or_falls_back() {
    // On Linux this succeeds; if the kernel lacks RUSAGE_THREAD the fallback
    // reads /proc and still returns `Ok`. Either way it must not panic.
    let _ = proc::read_thread_rusage();
    let _ = proc::read_thread_rusage_proc();
}

#[test]
fn pmu_unavailability_is_typed_not_silent() {
    let pmu = PmuCounters::open();
    match pmu.cycles_instructions() {
        Some(_) => assert!(
            pmu.unavailable_reason().is_empty(),
            "available counters must report no reason"
        ),
        None => assert!(
            !pmu.unavailable_reason().is_empty(),
            "unavailable counters must carry a typed reason (e.g. perf_event_paranoid)"
        ),
    }
    // IPC is optional and never panics.
    let _ = pmu.ipc();
}
