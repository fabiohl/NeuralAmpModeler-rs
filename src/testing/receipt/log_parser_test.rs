// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn sums_test_result_lines_with_nocapture_payloads() {
    let log = b"running 74 tests\n\
        test golden::model_a ... Measurement: ESR=1e-14\n\
        test golden::model_b ... ok\n\
        test result: ok. 74 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n";
    assert_eq!(tests_passed_from_output(log), (74, true));
}

#[test]
fn sums_every_command_of_a_phase() {
    let log = b"test result: ok. 22 passed; 0 failed; 0 ignored\n\
        test result: ok. 1 passed; 0 failed; 0 ignored\n";
    assert_eq!(tests_passed_from_output(log), (23, true));
}

#[test]
fn strips_ansi_before_parsing() {
    let log = b"\x1b[0;32mtest result: ok. 10 passed;\x1b[0m 0 failed; 0 ignored\n";
    assert_eq!(tests_passed_from_output(log), (10, true));
}

#[test]
fn no_test_result_is_fail_closed() {
    let log = b"running 3 tests\ntest one ... ok\ntest two ... ok\n";
    assert_eq!(tests_passed_from_output(log), (0, false));
}

#[test]
fn failed_result_line_still_reports_execution() {
    let log = b"test result: FAILED. 0 passed; 2 failed; 0 ignored\n";
    assert_eq!(tests_passed_from_output(log), (0, true));
}

#[test]
fn malformed_result_line_never_invents_a_count() {
    assert_eq!(test_result_passed("test result: ignored"), None);
    assert_eq!(test_result_passed("test result: ok. many passed"), None);
}
