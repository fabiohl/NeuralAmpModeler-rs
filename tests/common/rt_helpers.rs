// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::fs;
use std::io;

#[derive(Debug, PartialEq)]
pub enum RtPreflightStatus {
    Pass,
    Inconclusive {
        cpu_affinity_ok: bool,
        governor_ok: bool,
        background_load_ok: bool,
    },
}

#[derive(Debug)]
pub struct RtPreflightResult {
    pub status: RtPreflightStatus,
    pub cpu_affinity_ok: bool,
    pub governor_ok: bool,
    pub background_load_ok: bool,
    pub governor: String,
    pub pinned_core: Option<usize>,
    pub load_1m: Option<f64>,
    pub num_cpus: usize,
}

fn check_cpu_affinity(single_core_required: bool) -> (bool, Option<usize>) {
    #[cfg(target_os = "linux")]
    {
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        let ret =
            unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
        if ret != 0 {
            return (false, None);
        }

        let mut pinned: Vec<usize> = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if unsafe { libc::CPU_ISSET(cpu, &set) } {
                pinned.push(cpu);
            }
        }

        if single_core_required {
            if pinned.len() == 1 {
                (true, Some(pinned[0]))
            } else {
                (false, None)
            }
        } else if !pinned.is_empty() {
            if pinned.len() == 1 {
                (true, Some(pinned[0]))
            } else {
                (true, None)
            }
        } else {
            (false, None)
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        if single_core_required {
            (false, None)
        } else {
            (true, None)
        }
    }
}

/// Builds the `scaling_governor` path for an explicit CPU core.
///
/// Pure constructor (no I/O) — unit-testable with a mocked core id.
pub fn governor_path_for_core(core: u32) -> String {
    format!("/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor")
}

/// Effective bench core for governor probing.
///
/// Precedence: the pinned core observed from the process affinity mask when
/// it resolves to exactly one core (the `taskset -c <core>` case — the most
/// precise signal), then `BENCH_CORE`/`NAM_BENCH_CORE`, then `nproc / 2`
/// (integer division — the same default as
/// `utils/tests-performance-regression.sh:53-55`). Unparseable values are
/// ignored; on machines without `cpufreq` the read below fails closed to
/// `unknown` (never a hard error where a graceful skip existed before).
fn effective_bench_core() -> u32 {
    // Most precise signal first: when the process is pinned to exactly one
    // core (`taskset -c <core>`, the RT-gate invocation shape), probe that
    // core — `BENCH_CORE` may be unset in that path (tests-long.sh only
    // exports `NAM_BENCH_CORE` when the operator sets it explicitly).
    #[cfg(target_os = "linux")]
    {
        let (single_pinned, pinned) = check_cpu_affinity(true);
        if single_pinned && let Some(core) = pinned {
            return core as u32;
        }
    }
    for key in ["BENCH_CORE", "NAM_BENCH_CORE"] {
        if let Ok(raw) = std::env::var(key) {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(core) = trimmed.parse::<u32>() {
                return core;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2) as u32)
        .unwrap_or(0)
}

fn read_governor() -> Result<String, io::Error> {
    // Probe the governor of the effective bench core — the core the
    // bench is actually pinned to — instead of a fixed `cpu0`. On hybrid
    // (P-core/E-core) systems `cpu0` may report `performance` while the
    // pinned core does not.
    let path = governor_path_for_core(effective_bench_core());
    let text = fs::read_to_string(path)?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        Err(io::Error::new(io::ErrorKind::InvalidData, "empty governor"))
    } else {
        Ok(trimmed.to_string())
    }
}

fn check_governor() -> (bool, String) {
    match read_governor() {
        Ok(gov) => {
            let ok = gov == "performance";
            (ok, gov)
        }
        Err(_) => (false, "unknown".to_string()),
    }
}

fn read_load_1m() -> Result<f64, io::Error> {
    let content = fs::read_to_string("/proc/loadavg")?;
    let first = content
        .split_whitespace()
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty loadavg"))?;
    first
        .parse::<f64>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn system_cpu_count() -> usize {
    #[cfg(target_os = "linux")]
    {
        let count = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_CONF) };
        if count > 0 {
            count as usize
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

fn check_background_load(num_cpus: usize) -> (bool, Option<f64>) {
    match read_load_1m() {
        Ok(load) => {
            let scaled = load / (num_cpus as f64);
            let ok = scaled < 1.5;
            (ok, Some(load))
        }
        Err(_) => (false, None),
    }
}

/// Preflight check for deterministic single-core RT deadline verification (`rt_deadline.rs`).
///
/// Requires single-core CPU affinity, performance governor, and low background load.
pub fn rt_preflight() -> RtPreflightResult {
    rt_preflight_impl(true)
}

/// Preflight check for multi-threaded RT jitter characterization under contention (`rt_jitter.rs`).
///
/// Permits multi-core execution (does not constrain affinity mask to 1 CPU), while still
/// verifying that the performance governor and low background load conditions are met.
pub fn rt_preflight_jitter() -> RtPreflightResult {
    rt_preflight_impl(false)
}

fn rt_preflight_impl(single_core_required: bool) -> RtPreflightResult {
    let num_cpus = system_cpu_count();

    let (cpu_affinity_ok, pinned_core) = check_cpu_affinity(single_core_required);
    let (governor_ok, governor) = check_governor();
    let (background_load_ok, load_1m) = check_background_load(num_cpus);

    let status = if cpu_affinity_ok && governor_ok && background_load_ok {
        RtPreflightStatus::Pass
    } else {
        RtPreflightStatus::Inconclusive {
            cpu_affinity_ok,
            governor_ok,
            background_load_ok,
        }
    };

    RtPreflightResult {
        status,
        cpu_affinity_ok,
        governor_ok,
        background_load_ok,
        governor,
        pinned_core,
        load_1m,
        num_cpus,
    }
}

pub fn print_preflight(result: &RtPreflightResult) {
    println!(
        "[RT_PREFLIGHT] cpu_affinity={} governor={} ({}) load_1m={} {}cpus",
        if result.cpu_affinity_ok {
            if let Some(core) = result.pinned_core {
                format!("pinned_cpu{}", core)
            } else {
                "ok".to_string()
            }
        } else {
            "FAIL".to_string()
        },
        if result.governor_ok { "ok" } else { "FAIL" },
        result.governor,
        result
            .load_1m
            .map(|l| format!("{:.1}", l))
            .unwrap_or_else(|| "N/A".to_string()),
        result.num_cpus,
    );

    if result.status != RtPreflightStatus::Pass {
        println!("[RT_PREFLIGHT] INCONCLUSIVE — environment preconditions not met:");
        if !result.cpu_affinity_ok {
            println!("  - CPU affinity check failed (not pinned or affinity mask empty)");
        }
        if !result.governor_ok {
            println!(
                "  - CPU governor is '{}' (requires 'performance')",
                result.governor
            );
        }
        if !result.background_load_ok {
            println!("  - Background load too high (load_1m / ncpu >= 1.5)");
        }
    }
}
