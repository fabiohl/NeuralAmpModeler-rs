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

fn check_cpu_affinity() -> (bool, Option<usize>) {
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    #[cfg(target_os = "linux")]
    {
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        let ret =
            unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
        if ret != 0 {
            return (false, None);
        }

        let mut pinned: Vec<usize> = Vec::new();
        for cpu in 0..num_cpus.min(libc::CPU_SETSIZE as usize) {
            if unsafe { libc::CPU_ISSET(cpu, &set) } {
                pinned.push(cpu);
            }
        }

        if pinned.len() == 1 {
            (true, Some(pinned[0]))
        } else {
            (false, None)
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        (false, None)
    }
}

fn read_governor() -> Result<String, io::Error> {
    fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .map(|s| s.trim().to_string())
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

pub fn rt_preflight() -> RtPreflightResult {
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let (cpu_affinity_ok, pinned_core) = check_cpu_affinity();
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
            println!("  - CPU affinity not pinned to single core (use taskset -c <core>)");
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
