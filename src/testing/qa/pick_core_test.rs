// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn parse_cpu_list_preserves_valid_lists() {
    assert_eq!(parse_cpu_list("0-3,7,9-10"), vec![0, 1, 2, 3, 7, 9, 10]);
    assert_eq!(parse_cpu_list("3,2,1,0"), vec![0, 1, 2, 3]);
    assert_eq!(parse_cpu_list("1,1,2,2"), vec![1, 2]);
    assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
    assert_eq!(parse_cpu_list("5-1"), Vec::<usize>::new());
}

#[test]
fn parse_cpu_list_is_bounded() {
    assert_eq!(parse_cpu_list("0-1023").len(), 1024);
    assert!(matches!(
        try_parse_cpu_list("0-1024"),
        Err(CpuListError::CpuIndexOutOfRange { cpu: 1024, .. })
    ));
    assert!(matches!(
        try_parse_cpu_list("0-600,0-600"),
        Err(CpuListError::TooManyCpus { .. })
    ));
    assert!(parse_cpu_list("0-10000000000").is_empty());
    assert!(parse_cpu_list(&usize::MAX.to_string()).is_empty());
}

#[test]
fn parse_cpu_list_rejects_long_input_before_expanding() {
    let long = format!("0{}", " ".repeat(64 * 1024));
    assert!(matches!(
        try_parse_cpu_list(&long),
        Err(CpuListError::InputTooLong { .. })
    ));
    assert!(parse_cpu_list(&long).is_empty());
}
