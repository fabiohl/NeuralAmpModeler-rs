// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//
// Union of the engine and downstream host affinity test suites.
// The host-agnostic selection/topology API is consolidated here so every
// consumer exercises one canonical set of fixtures and ranking invariants.

use super::*;
use std::io::BufReader;

/// In-memory mock of sysfs and affinity sources for deterministic testing.
struct MockSysfsSource {
    pub cpus: Vec<usize>,
    pub allowed_cpus: Vec<usize>,
    pub sysfs_files: HashMap<String, String>,
    pub irqs: HashMap<usize, u64>,
}

impl MockSysfsSource {
    fn new(cpus: Vec<usize>, allowed_cpus: Vec<usize>) -> Self {
        Self {
            cpus,
            allowed_cpus,
            sysfs_files: HashMap::new(),
            irqs: HashMap::new(),
        }
    }

    fn with_file(mut self, path: &str, content: &str) -> Self {
        self.sysfs_files
            .insert(path.to_string(), content.to_string());
        self
    }

    fn with_irq(mut self, cpu: usize, count: u64) -> Self {
        self.irqs.insert(cpu, count);
        self
    }
}

impl SysfsTopologySource for MockSysfsSource {
    fn read_cpu_indices(&self) -> Vec<usize> {
        self.cpus.clone()
    }

    fn read_sysfs_string(&self, path: &str) -> Option<String> {
        self.sysfs_files.get(path).cloned()
    }

    fn get_allowed_cpus(&self) -> Vec<usize> {
        self.allowed_cpus.clone()
    }

    fn get_irq_counts(&self) -> HashMap<usize, u64> {
        self.irqs.clone()
    }
}

#[test]
fn test_parse_cpu_list_ranges() {
    assert_eq!(parse_cpu_list("0-3,7,9-10"), vec![0, 1, 2, 3, 7, 9, 10]);
    assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
}

#[test]
fn test_parse_cpu_list_various_formats() {
    assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
    assert_eq!(parse_cpu_list("0"), vec![0]);
    assert_eq!(parse_cpu_list("0,1,2,3"), vec![0, 1, 2, 3]);
    assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
    assert_eq!(
        parse_cpu_list(" 0 - 3 , 7 , 9 - 10 \n"),
        vec![0, 1, 2, 3, 7, 9, 10]
    );
    assert_eq!(parse_cpu_list("3,2,1,0"), vec![0, 1, 2, 3]);
    assert_eq!(parse_cpu_list("1,1,2,2"), vec![1, 2]);
}

#[test]
fn test_parse_proc_interrupts_numeric_only() {
    let text = "          CPU0       CPU1\n  24: 100 200 IO-APIC\n NMI: 1 2\n";
    let map = parse_proc_interrupts(BufReader::new(text.as_bytes()));
    assert_eq!(map[&0], 100);
    assert_eq!(map[&1], 200);
}

#[test]
fn test_parse_proc_interrupts_fixture() {
    let raw = "            CPU0       CPU1       CPU2       CPU3\n\
               0:         10          0          5          0  IO-APIC   2-edge      timer\n\
               1:          0        200          0          0  IO-APIC   1-edge      i8042\n\
             LOC:       1000       1000       1000       1000  Local timer interrupts\n\
             NMI:          0          0          0          0  Non-maskable interrupts\n\
              24:          0          0          0        500  PCI-MSI 512000-edge   nvme\n";

    let totals = parse_proc_interrupts(std::io::BufReader::new(raw.as_bytes()));
    assert_eq!(totals.get(&0).copied(), Some(10));
    assert_eq!(totals.get(&1).copied(), Some(200));
    assert_eq!(totals.get(&2).copied(), Some(5));
    assert_eq!(totals.get(&3).copied(), Some(500));
}

#[test]
fn test_explicit_cpu_honoured_when_allowed() {
    let src = MockSysfsSource::new(vec![0, 1, 2, 3], vec![0, 1, 2, 3]);
    let receipt = select_cpu_with_source(Some(2), &src);
    assert_eq!(receipt.selected_cpu, 2);
    assert!(matches!(
        receipt.reason,
        CpuSelectionReason::ExplicitCli { cpu: 2, .. }
    ));
}

#[test]
fn test_isolated_core_wins_over_heuristic() {
    let src = MockSysfsSource::new(vec![0, 1], vec![0, 1])
        .with_file("/sys/devices/system/cpu/isolated", "1");
    let receipt = select_cpu_with_source(None, &src);
    assert_eq!(receipt.selected_cpu, 1);
    assert!(receipt.is_isolated);
}

#[test]
fn test_fixture_smt_topology_prefers_primary_sibling() {
    // 4 logical CPUs: Core 0 has (0, 2), Core 1 has (1, 3)
    let mock = MockSysfsSource::new(vec![0, 1, 2, 3], vec![0, 1, 2, 3])
        .with_file("/sys/devices/system/cpu/cpu0/topology/core_id", "0")
        .with_file(
            "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list",
            "0,2",
        )
        .with_file("/sys/devices/system/cpu/cpu2/topology/core_id", "0")
        .with_file(
            "/sys/devices/system/cpu/cpu2/topology/thread_siblings_list",
            "0,2",
        )
        .with_file("/sys/devices/system/cpu/cpu1/topology/core_id", "1")
        .with_file(
            "/sys/devices/system/cpu/cpu1/topology/thread_siblings_list",
            "1,3",
        )
        .with_file("/sys/devices/system/cpu/cpu3/topology/core_id", "1")
        .with_file(
            "/sys/devices/system/cpu/cpu3/topology/thread_siblings_list",
            "1,3",
        )
        .with_irq(0, 10)
        .with_irq(2, 10)
        .with_irq(1, 10)
        .with_irq(3, 10);

    let receipt = select_cpu_with_source(None, &mock);
    assert!(!receipt.is_dedicated);
    // Primary siblings are 0 and 1. Tie-breaker prefers higher logical ID -> 1
    assert_eq!(receipt.selected_cpu, 1);
    assert_eq!(receipt.smt_siblings, vec![1, 3]);
    // Housekeeping should exclude CPU 1 and sibling 3 -> [0, 2]
    assert_eq!(receipt.housekeeping_cpus, vec![0, 2]);
}

#[test]
fn test_fixture_hybrid_big_little() {
    // 4 CPUs: 0, 1 are LITTLE (capacity 512), 2, 3 are BIG (capacity 1024)
    let mock = MockSysfsSource::new(vec![0, 1, 2, 3], vec![0, 1, 2, 3])
        .with_file("/sys/devices/system/cpu/cpu0/cpu_capacity", "512")
        .with_file("/sys/devices/system/cpu/cpu1/cpu_capacity", "512")
        .with_file("/sys/devices/system/cpu/cpu2/cpu_capacity", "1024")
        .with_file("/sys/devices/system/cpu/cpu3/cpu_capacity", "1024")
        .with_irq(2, 500)
        .with_irq(3, 50); // CPU 3 has fewer interrupts among BIG cores

    let receipt = select_cpu_with_source(None, &mock);
    assert_eq!(receipt.selected_cpu, 3);
    assert!(!receipt.is_dedicated);
    assert_eq!(receipt.housekeeping_cpus, vec![0, 1, 2]);
}

#[test]
fn test_fixture_proven_isolated_core() {
    // 4 CPUs: CPU 3 is in /sys/devices/system/cpu/isolated
    let mock = MockSysfsSource::new(vec![0, 1, 2, 3], vec![0, 1, 2, 3])
        .with_file("/sys/devices/system/cpu/isolated", "3")
        .with_file("/sys/devices/system/cpu/nohz_full", "3")
        .with_file(
            "/sys/devices/system/cpu/cpu3/topology/physical_package_id",
            "0",
        )
        .with_file("/sys/devices/system/cpu/cpu3/topology/core_id", "3")
        .with_file(
            "/sys/devices/system/cpu/cpu3/topology/thread_siblings_list",
            "3",
        );

    let receipt = select_cpu_with_source(None, &mock);
    assert!(receipt.is_dedicated);
    assert!(receipt.is_isolated);
    assert!(receipt.is_nohz_full);
    assert_eq!(receipt.selected_cpu, 3);
    assert_eq!(receipt.package_id, Some(0));
    assert_eq!(receipt.core_id, Some(3));
    match receipt.reason {
        CpuSelectionReason::FullyIsolated { cpu, nohz_full, .. } => {
            assert_eq!(cpu, 3);
            assert!(nohz_full);
        }
        _ => panic!("Expected FullyIsolated reason"),
    }
}

#[test]
fn test_fixture_multi_socket_and_nohz_full_mix() {
    // Socket 0: 0, 1 (isolated, but standard tick)
    // Socket 1: 2, 3 (isolated AND nohz_full)
    let mock = MockSysfsSource::new(vec![0, 1, 2, 3], vec![0, 1, 2, 3])
        .with_file("/sys/devices/system/cpu/isolated", "1,2,3")
        .with_file("/sys/devices/system/cpu/nohz_full", "2,3")
        .with_file(
            "/sys/devices/system/cpu/cpu0/topology/physical_package_id",
            "0",
        )
        .with_file(
            "/sys/devices/system/cpu/cpu1/topology/physical_package_id",
            "0",
        )
        .with_file(
            "/sys/devices/system/cpu/cpu2/topology/physical_package_id",
            "1",
        )
        .with_file(
            "/sys/devices/system/cpu/cpu3/topology/physical_package_id",
            "1",
        )
        .with_irq(2, 100)
        .with_irq(3, 10); // CPU 3 has fewer IRQs among nohz_full cores

    let receipt = select_cpu_with_source(None, &mock);
    assert!(receipt.is_dedicated);
    assert_eq!(receipt.selected_cpu, 3);
    assert_eq!(receipt.package_id, Some(1));
    assert!(receipt.is_nohz_full);
}

#[test]
fn test_fixture_cpuset_partial_confinement() {
    // Process only allowed on CPU 1, 2 by cgroups
    let mock = MockSysfsSource::new(vec![0, 1, 2, 3], vec![1, 2])
        .with_file("/sys/devices/system/cpu/isolated", "0") // 0 is isolated but forbidden by cpuset!
        .with_irq(1, 20)
        .with_irq(2, 5);

    let receipt = select_cpu_with_source(None, &mock);
    // Invariant: Never select CPU outside cpuset! 0 must not be chosen even if isolated.
    assert_eq!(receipt.selected_cpu, 2);
    assert!(!receipt.is_dedicated); // Non-isolated heuristic because 2 is not in isolated list
    assert_eq!(receipt.housekeeping_cpus, vec![1]);
}

#[test]
fn test_fixture_explicit_cli_cpu_allowed() {
    let mock = MockSysfsSource::new(vec![0, 1, 2, 3], vec![0, 1, 2, 3])
        .with_file("/sys/devices/system/cpu/isolated", "3")
        .with_file("/sys/devices/system/cpu/cpu2/topology/core_id", "2");

    let receipt = select_cpu_with_source(Some(2), &mock);
    assert_eq!(receipt.selected_cpu, 2);
    assert_eq!(
        receipt.reason,
        CpuSelectionReason::ExplicitCli {
            cpu: 2,
            in_cpuset: true
        }
    );
    assert!(!receipt.is_dedicated); // Core 2 is not isolated
}

#[test]
fn test_fixture_explicit_cli_cpu_forbidden_by_cpuset_falls_back() {
    // User requested CPU 0, but cpuset only allows [2, 3]
    let mock = MockSysfsSource::new(vec![0, 1, 2, 3], vec![2, 3])
        .with_irq(2, 100)
        .with_irq(3, 10);

    let receipt = select_cpu_with_source(Some(0), &mock);
    // Must enforce cpuset invariant and fall back to allowed core
    assert_eq!(receipt.selected_cpu, 3);
    assert!(receipt.housekeeping_cpus.contains(&2));
}

#[test]
fn test_fixture_incomplete_sysfs_conservative_fallback() {
    // No sysfs files at all (e.g. stripped container)
    let mock = MockSysfsSource::new(vec![0, 1], vec![0, 1]);

    let receipt = select_cpu_with_source(None, &mock);
    assert!(!receipt.is_dedicated);
    assert_eq!(receipt.selected_cpu, 1);
    match receipt.reason {
        CpuSelectionReason::ConservativeHeuristic { .. } => {}
        _ => panic!("Expected conservative heuristic on incomplete sysfs"),
    }
}
