// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for the host probes.
//!
//! These assert the *absence tolerance* contract — a missing sysfs node, a
//! denied PMU, or a non-existent CPU must never panic — plus the pure parsing
//! helpers. Live values are environment-dependent, so only the invariants that
//! hold anywhere on Linux are checked.

use super::{pmu::PmuCounters, proc, rt_env};
use std::io::Write;
use std::os::fd::{FromRawFd, IntoRawFd};

fn pipe_pair(payload: &[u8]) -> i32 {
    let mut fds = [0i32; 2];
    // SAFETY: `fds` is a valid two-element array for `pipe`.
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
    // SAFETY: `fds[1]` is the write end returned by `pipe`.
    let mut writer = unsafe { std::fs::File::from_raw_fd(fds[1]) };
    writer.write_all(payload).unwrap();
    drop(writer);
    fds[0]
}

fn dev_null_fd() -> i32 {
    std::fs::File::open("/dev/null").unwrap().into_raw_fd()
}

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

#[test]
fn pmu_short_read_is_unavailable_not_zero() {
    let pmu = PmuCounters::from_raw_fds(pipe_pair(&[1, 2, 3]), pipe_pair(&[0; 8]));
    assert_eq!(
        pmu.cycles_instructions(),
        None,
        "a short counter read must not be reported as a valid sample"
    );
    assert!(
        pmu.unavailable_reason().contains("short perf counter read"),
        "short read must be observable: {}",
        pmu.unavailable_reason()
    );
}

#[test]
fn pmu_ioctl_failure_is_unavailable_not_zero() {
    let pmu = PmuCounters::from_raw_fds(dev_null_fd(), dev_null_fd());
    pmu.start();
    assert_eq!(
        pmu.cycles_instructions(),
        None,
        "a failed ioctl must not leave a misleading (0, 0) sample"
    );
    assert!(!pmu.unavailable_reason().is_empty());
    pmu.stop();
}

#[test]
fn pmu_genuine_zero_remains_distinguishable() {
    let pmu = PmuCounters::from_raw_fds(pipe_pair(&[0; 8]), pipe_pair(&[0; 8]));
    assert_eq!(pmu.cycles_instructions(), Some((0, 0)));
    assert!(pmu.unavailable_reason().is_empty());
    // Zero cycles cannot produce an IPC value.
    assert_eq!(pmu.ipc(), None);
}

#[test]
fn parse_cpu_ranges_never_fabricates_cpu_zero() {
    for malformed in ["a-b", "x-5", "1-", "5-1", "-", "a"] {
        assert_eq!(
            rt_env::parse_cpu_ranges(malformed),
            Vec::<u32>::new(),
            "malformed part {malformed:?} must not yield CPU 0"
        );
    }
    assert_eq!(rt_env::parse_cpu_ranges("0,5-1,2"), vec![0, 2]);
}

#[test]
fn parse_cpu_ranges_rejects_oversized_ranges_and_ids() {
    assert!(rt_env::parse_cpu_ranges("0-10000000000").is_empty());
    assert!(rt_env::parse_cpu_ranges("0-4294967295").is_empty());
    assert!(rt_env::parse_cpu_ranges(&u32::MAX.to_string()).is_empty());
    assert!(matches!(
        rt_env::try_parse_cpu_ranges("0-4294967295"),
        Err(rt_env::CpuListError::CpuIndexOutOfRange { .. })
    ));

    assert_eq!(rt_env::parse_cpu_ranges("0-1023").len(), 1024);
    assert!(matches!(
        rt_env::try_parse_cpu_ranges("0-1024"),
        Err(rt_env::CpuListError::CpuIndexOutOfRange { .. })
    ));
    assert!(matches!(
        rt_env::try_parse_cpu_ranges("0-600,0-600"),
        Err(rt_env::CpuListError::TooManyCpus { .. })
    ));
}

#[test]
fn parse_cpu_ranges_rejects_long_input_before_expanding() {
    let long = format!("0{}", " ".repeat(64 * 1024));
    assert!(matches!(
        rt_env::try_parse_cpu_ranges(&long),
        Err(rt_env::CpuListError::InputTooLong { .. })
    ));
    assert!(rt_env::parse_cpu_ranges(&long).is_empty());
}
