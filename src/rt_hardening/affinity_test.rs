// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use std::io::BufReader;

struct FixtureSource {
    cpus: Vec<usize>,
    allowed: Vec<usize>,
    isolated: String,
    irqs: HashMap<usize, u64>,
}

impl SysfsTopologySource for FixtureSource {
    fn read_cpu_indices(&self) -> Vec<usize> {
        self.cpus.clone()
    }
    fn read_sysfs_string(&self, path: &str) -> Option<String> {
        if path.ends_with("/isolated") {
            return Some(self.isolated.clone());
        }
        None
    }
    fn get_allowed_cpus(&self) -> Vec<usize> {
        self.allowed.clone()
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
fn test_parse_proc_interrupts_numeric_only() {
    let text = "          CPU0       CPU1\n  24: 100 200 IO-APIC\n NMI: 1 2\n";
    let map = parse_proc_interrupts(BufReader::new(text.as_bytes()));
    assert_eq!(map[&0], 100);
    assert_eq!(map[&1], 200);
}

#[test]
fn test_explicit_cpu_honoured_when_allowed() {
    let src = FixtureSource {
        cpus: vec![0, 1, 2, 3],
        allowed: vec![0, 1, 2, 3],
        isolated: String::new(),
        irqs: HashMap::new(),
    };
    let receipt = select_cpu_with_source(Some(2), &src);
    assert_eq!(receipt.selected_cpu, 2);
    assert!(matches!(
        receipt.reason,
        CpuSelectionReason::ExplicitCli { cpu: 2, .. }
    ));
}

#[test]
fn test_isolated_core_wins_over_heuristic() {
    let src = FixtureSource {
        cpus: vec![0, 1],
        allowed: vec![0, 1],
        isolated: "1".to_string(),
        irqs: HashMap::new(),
    };
    let receipt = select_cpu_with_source(None, &src);
    assert_eq!(receipt.selected_cpu, 1);
    assert!(receipt.is_isolated);
}
