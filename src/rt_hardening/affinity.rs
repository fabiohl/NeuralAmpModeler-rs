// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! IRQ-aware CPU topology inspection and optimal core selection (off-RT only).
//!
//! Parses Linux CPU list syntax, `/proc/interrupts` load, `sched_getaffinity`
//! cpusets and sysfs topology to pin the audio thread to the most isolated,
//! highest-capacity core. These functions must be called exclusively outside
//! the RT thread, during audio host initialization. Errors are returned as
//! `Result` — never as panic.

#![cfg(all(feature = "rt-hardening", target_os = "linux"))]

use std::collections::HashMap;

/// Parses standard Linux CPU list syntax (e.g. "0-3,7,9-10") into a sorted,
/// unique vector of CPU indices.
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

/// Parses a `/proc/interrupts` table from a reader, returning total numeric
/// interrupts per CPU index.
pub fn parse_proc_interrupts<R: std::io::BufRead>(reader: R) -> HashMap<usize, u64> {
    let mut totals: HashMap<usize, u64> = HashMap::new();
    let mut lines = reader.lines();

    let Some(Ok(header)) = lines.next() else {
        return totals;
    };

    let cpu_ids: Vec<usize> = header
        .split_whitespace()
        .filter_map(|tok| tok.strip_prefix("CPU")?.parse::<usize>().ok())
        .collect();

    if cpu_ids.is_empty() {
        return totals;
    }
    for &id in &cpu_ids {
        totals.insert(id, 0);
    }

    for line_res in lines {
        let Ok(line) = line_res else { break };
        let trimmed = line.trim_start();
        let irq_end = trimmed.find(':').unwrap_or(0);
        if irq_end == 0 {
            continue;
        }

        // Only numeric interrupts (ignores NMI, LOC, etc.).
        if !trimmed[..irq_end]
            .trim()
            .bytes()
            .all(|b| b.is_ascii_digit())
        {
            continue;
        }

        let Some(after_colon) = trimmed.get(irq_end + 1..) else {
            continue;
        };

        for (&cpu_id, token) in cpu_ids.iter().zip(after_colon.split_whitespace()) {
            if let Ok(count) = token.parse::<u64>() {
                *totals.entry(cpu_id).or_insert(0) += count;
            } else {
                break;
            }
        }
    }

    totals
}

/// Parses `/proc/interrupts` to extract the interrupt load per physical CPU.
///
/// Returns an empty map when the file cannot be opened (non-Linux CI, sandbox).
pub fn parse_interrupts_per_cpu() -> HashMap<usize, u64> {
    use std::fs::File;
    use std::io::BufReader;

    let Ok(file) = File::open("/proc/interrupts") else {
        return HashMap::new();
    };
    parse_proc_interrupts(BufReader::new(file))
}

/// Returns the CPUs allowed for the current process via `sched_getaffinity`.
///
/// Respects kernel isolation (isolcpus), cgroups and affinity masks imposed
/// by the OS or the user (e.g. taskset).
pub fn get_allowed_cpus() -> Vec<usize> {
    let mut allowed = Vec::new();

    // SAFETY: on the supported Linux targets `cpu_set_t` is a C bitmask whose
    // all-zero pattern is the empty set — a valid value with no reference
    // formed over uninitialized storage.
    let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };

    // SAFETY: `CPU_ZERO` mutates the initialized mask in place;
    // `sched_getaffinity` fills the same valid object. On failure the mask is
    // left untouched and `ok` keeps the loop from reading it.
    let ok = unsafe {
        libc::CPU_ZERO(&mut cpuset);
        libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut cpuset) == 0
    };

    if ok {
        for i in 0..libc::CPU_SETSIZE as usize {
            // SAFETY: `cpuset` was filled by the successful `sched_getaffinity`
            // above; `CPU_ISSET` only reads initialized storage for `i` in
            // `[0, CPU_SETSIZE)`.
            if unsafe { libc::CPU_ISSET(i, &cpuset) } {
                allowed.push(i);
            }
        }
    }
    allowed
}

/// Abstraction over sysfs and OS affinity queries for deterministic tests.
pub trait SysfsTopologySource {
    /// Returns the discovered logical CPUs (e.g. from `/sys/devices/system/cpu/cpu*`).
    fn read_cpu_indices(&self) -> Vec<usize>;
    /// Reads a sysfs string from the given path.
    fn read_sysfs_string(&self, path: &str) -> Option<String>;
    /// Returns the allowed CPUs for the current process (`sched_getaffinity`).
    fn get_allowed_cpus(&self) -> Vec<usize>;
    /// Returns the CPU index → cumulative IRQ count mapping (`/proc/interrupts`).
    fn get_irq_counts(&self) -> HashMap<usize, u64>;
}

/// Live implementation querying `/sys` and `/proc`.
pub struct SystemSysfsSource;

impl SysfsTopologySource for SystemSysfsSource {
    fn read_cpu_indices(&self) -> Vec<usize> {
        let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") else {
            return Vec::new();
        };
        let mut cpus: Vec<usize> = entries
            .filter_map(|entry| {
                let name = entry.ok()?.file_name();
                let name = name.to_str()?;
                name.strip_prefix("cpu")?.parse::<usize>().ok()
            })
            .collect();
        cpus.sort_unstable();
        cpus
    }

    fn read_sysfs_string(&self, path: &str) -> Option<String> {
        std::fs::read_to_string(path).ok()
    }

    fn get_allowed_cpus(&self) -> Vec<usize> {
        get_allowed_cpus()
    }

    fn get_irq_counts(&self) -> HashMap<usize, u64> {
        parse_interrupts_per_cpu()
    }
}

/// Discovered topology details for a single logical CPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuCoreTopology {
    /// Logical CPU index.
    pub logical_id: usize,
    /// Physical package id (`physical_package_id`), if readable.
    pub package_id: Option<usize>,
    /// Core id (`core_id`), if readable.
    pub core_id: Option<usize>,
    /// SMT siblings (`thread_siblings_list`, fallback `core_cpus_list`).
    pub smt_siblings: Vec<usize>,
    /// Capacity (`cpu_capacity`, default 1024).
    pub capacity: u64,
    /// Cumulative numeric IRQ count.
    pub irq_count: u64,
    /// Listed in `/sys/devices/system/cpu/isolated`.
    pub is_isolated: bool,
    /// Listed in `/sys/devices/system/cpu/nohz_full`.
    pub is_nohz_full: bool,
    /// Allowed by the process cpuset.
    pub in_cpuset: bool,
}

/// Structured explanation of why a specific CPU was selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuSelectionReason {
    /// Explicitly pinned via host CLI/config.
    ExplicitCli {
        /// Requested CPU.
        cpu: usize,
        /// Whether it was allowed by the cpuset.
        in_cpuset: bool,
    },
    /// Selected core is proven isolated by the kernel.
    FullyIsolated {
        /// Selected CPU.
        cpu: usize,
        /// Physical package id.
        package_id: Option<usize>,
        /// Core id.
        core_id: Option<usize>,
        /// SMT siblings.
        smt_siblings: Vec<usize>,
        /// Tickless (`nohz_full`).
        nohz_full: bool,
    },
    /// Selected core was chosen via conservative heuristics.
    ConservativeHeuristic {
        /// Selected CPU.
        cpu: usize,
        /// Capacity used for ranking.
        capacity: u64,
        /// IRQ load used for ranking.
        irq_count: u64,
        /// Static explanation of the heuristic.
        explanation: &'static str,
    },
}

/// Typed receipt of CPU selection and isolation proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuSelectionReceipt {
    /// Selected CPU index.
    pub selected_cpu: usize,
    /// Proven dedicated (kernel-isolated or explicitly pinned).
    pub is_dedicated: bool,
    /// Physical package id.
    pub package_id: Option<usize>,
    /// Core id.
    pub core_id: Option<usize>,
    /// SMT siblings.
    pub smt_siblings: Vec<usize>,
    /// Kernel-isolated.
    pub is_isolated: bool,
    /// Tickless.
    pub is_nohz_full: bool,
    /// Why this CPU was selected.
    pub reason: CpuSelectionReason,
    /// CPUs reserved for housekeeping / I/O workers.
    pub housekeeping_cpus: Vec<usize>,
    /// Full discovered topology.
    pub topology: Vec<CpuCoreTopology>,
}

/// Selects the optimal CPU core given an explicit request and a topology source.
pub fn select_cpu_with_source<S: SysfsTopologySource>(
    requested_cpu: Option<usize>,
    source: &S,
) -> CpuSelectionReceipt {
    let isolated_set = source
        .read_sysfs_string("/sys/devices/system/cpu/isolated")
        .map(|s| parse_cpu_list(&s))
        .unwrap_or_default();

    let nohz_set = source
        .read_sysfs_string("/sys/devices/system/cpu/nohz_full")
        .map(|s| parse_cpu_list(&s))
        .unwrap_or_default();

    let irqs = source.get_irq_counts();
    let mut allowed_cpus = source.get_allowed_cpus();
    let mut online_cpus = source.read_cpu_indices();

    if online_cpus.is_empty() {
        online_cpus = if allowed_cpus.is_empty() {
            vec![0]
        } else {
            allowed_cpus.clone()
        };
    }

    if allowed_cpus.is_empty() {
        allowed_cpus = online_cpus.clone();
    }

    let mut topology = Vec::with_capacity(online_cpus.len());
    for &cpu in &online_cpus {
        let pkg_path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id");
        let package_id = source
            .read_sysfs_string(&pkg_path)
            .and_then(|s| s.trim().parse::<usize>().ok());

        let core_path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/core_id");
        let core_id = source
            .read_sysfs_string(&core_path)
            .and_then(|s| s.trim().parse::<usize>().ok());

        let siblings_path =
            format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let core_cpus_path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/core_cpus_list");
        let smt_siblings = source
            .read_sysfs_string(&siblings_path)
            .or_else(|| source.read_sysfs_string(&core_cpus_path))
            .map(|s| parse_cpu_list(&s))
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![cpu]);

        let cap_path = format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
        let capacity = source
            .read_sysfs_string(&cap_path)
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(1024);

        let irq_count = irqs.get(&cpu).copied().unwrap_or(0);
        let is_isolated = isolated_set.contains(&cpu);
        let is_nohz_full = nohz_set.contains(&cpu);
        let in_cpuset = allowed_cpus.contains(&cpu);

        topology.push(CpuCoreTopology {
            logical_id: cpu,
            package_id,
            core_id,
            smt_siblings,
            capacity,
            irq_count,
            is_isolated,
            is_nohz_full,
            in_cpuset,
        });
    }

    // 1. Explicit request honoured when allowed by the cpuset.
    if let Some(target) = requested_cpu {
        if allowed_cpus.contains(&target) {
            let target_topo = topology.iter().find(|t| t.logical_id == target);
            let is_isolated = target_topo.is_some_and(|t| t.is_isolated);
            let is_nohz = target_topo.is_some_and(|t| t.is_nohz_full);
            let pkg_id = target_topo.and_then(|t| t.package_id);
            let core_id = target_topo.and_then(|t| t.core_id);
            let siblings = target_topo.map_or_else(|| vec![target], |t| t.smt_siblings.clone());

            let housekeeping_cpus: Vec<usize> = allowed_cpus
                .iter()
                .copied()
                .filter(|&c| c != target && !siblings.contains(&c))
                .collect();
            let final_housekeeping = if housekeeping_cpus.is_empty() {
                allowed_cpus.clone()
            } else {
                housekeeping_cpus
            };

            return CpuSelectionReceipt {
                selected_cpu: target,
                is_dedicated: is_isolated,
                package_id: pkg_id,
                core_id,
                smt_siblings: siblings,
                is_isolated,
                is_nohz_full: is_nohz,
                reason: CpuSelectionReason::ExplicitCli {
                    cpu: target,
                    in_cpuset: true,
                },
                housekeeping_cpus: final_housekeeping,
                topology,
            };
        }
        log::warn!(
            "Requested CPU {target} is NOT in the process cpuset ({allowed_cpus:?}). \
             Enforcing cpuset invariant and falling back to auto-selection."
        );
    }

    // 2. Automatic selection among cpuset-allowed candidates.
    let candidates: Vec<&CpuCoreTopology> = topology.iter().filter(|t| t.in_cpuset).collect();

    if candidates.is_empty() {
        let fallback_cpu = online_cpus.first().copied().unwrap_or(0);
        log::warn!(
            "CPU selection fallback: no candidates in cpuset, defaulting to CPU {fallback_cpu}"
        );
        return CpuSelectionReceipt {
            selected_cpu: fallback_cpu,
            is_dedicated: false,
            package_id: None,
            core_id: None,
            smt_siblings: vec![fallback_cpu],
            is_isolated: false,
            is_nohz_full: false,
            reason: CpuSelectionReason::ConservativeHeuristic {
                cpu: fallback_cpu,
                capacity: 1024,
                irq_count: 0,
                explanation: "No allowed CPU found in cpuset; total fallback to core 0",
            },
            housekeeping_cpus: vec![fallback_cpu],
            topology,
        };
    }

    // Proven isolated cores first: nohz_full > SMT primary > capacity > lowest IRQ.
    let isolated_candidates: Vec<&CpuCoreTopology> = candidates
        .iter()
        .copied()
        .filter(|t| t.is_isolated)
        .collect();

    if let Some(&chosen) = isolated_candidates.iter().max_by(|a, b| {
        let a_primary = a.smt_siblings.first() == Some(&a.logical_id);
        let b_primary = b.smt_siblings.first() == Some(&b.logical_id);

        a.is_nohz_full
            .cmp(&b.is_nohz_full)
            .then_with(|| a_primary.cmp(&b_primary))
            .then_with(|| a.capacity.cmp(&b.capacity))
            .then_with(|| b.irq_count.cmp(&a.irq_count))
            .then_with(|| a.logical_id.cmp(&b.logical_id))
    }) {
        let housekeeping_cpus: Vec<usize> = allowed_cpus
            .iter()
            .copied()
            .filter(|&c| c != chosen.logical_id && !chosen.smt_siblings.contains(&c))
            .collect();
        let final_housekeeping = if housekeeping_cpus.is_empty() {
            allowed_cpus.clone()
        } else {
            housekeeping_cpus
        };

        return CpuSelectionReceipt {
            selected_cpu: chosen.logical_id,
            is_dedicated: true,
            package_id: chosen.package_id,
            core_id: chosen.core_id,
            smt_siblings: chosen.smt_siblings.clone(),
            is_isolated: true,
            is_nohz_full: chosen.is_nohz_full,
            reason: CpuSelectionReason::FullyIsolated {
                cpu: chosen.logical_id,
                package_id: chosen.package_id,
                core_id: chosen.core_id,
                smt_siblings: chosen.smt_siblings.clone(),
                nohz_full: chosen.is_nohz_full,
            },
            housekeeping_cpus: final_housekeeping,
            topology,
        };
    }

    // Conservative heuristic: capacity > SMT primary > lowest IRQ > highest id.
    let chosen = candidates
        .iter()
        .copied()
        .max_by(|a, b| {
            let a_primary = a.smt_siblings.first() == Some(&a.logical_id);
            let b_primary = b.smt_siblings.first() == Some(&b.logical_id);

            a.capacity
                .cmp(&b.capacity)
                .then_with(|| a_primary.cmp(&b_primary))
                .then_with(|| b.irq_count.cmp(&a.irq_count))
                .then_with(|| a.logical_id.cmp(&b.logical_id))
        })
        .cloned()
        .unwrap_or_else(|| candidates[0].clone());

    let housekeeping_cpus: Vec<usize> = allowed_cpus
        .iter()
        .copied()
        .filter(|&c| c != chosen.logical_id && !chosen.smt_siblings.contains(&c))
        .collect();
    let final_housekeeping = if housekeeping_cpus.is_empty() {
        allowed_cpus
    } else {
        housekeeping_cpus
    };

    CpuSelectionReceipt {
        selected_cpu: chosen.logical_id,
        is_dedicated: false,
        package_id: chosen.package_id,
        core_id: chosen.core_id,
        smt_siblings: chosen.smt_siblings.clone(),
        is_isolated: false,
        is_nohz_full: chosen.is_nohz_full,
        reason: CpuSelectionReason::ConservativeHeuristic {
            cpu: chosen.logical_id,
            capacity: chosen.capacity,
            irq_count: chosen.irq_count,
            explanation: "Highest capacity with lowest IRQ load and SMT primary preference (non-isolated)",
        },
        housekeeping_cpus: final_housekeeping,
        topology,
    }
}

/// Selects the ideal CPU core to pin the audio thread, returning the receipt.
pub fn select_optimal_cpu_with_receipt(
    requested_cpu: Option<usize>,
) -> Option<CpuSelectionReceipt> {
    let source = SystemSysfsSource;
    Some(select_cpu_with_source(requested_cpu, &source))
}

/// Selects the ideal CPU core to pin the audio thread.
///
/// Backward-compatible wrapper returning only the selected CPU index.
pub fn select_optimal_cpu() -> Option<usize> {
    select_optimal_cpu_with_receipt(None).map(|r| r.selected_cpu)
}

#[cfg(test)]
#[path = "affinity_test.rs"]
mod affinity_test;
