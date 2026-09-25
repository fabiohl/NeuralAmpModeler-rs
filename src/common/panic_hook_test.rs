// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_shutdown_in_progress_lifecycle_and_reset() {
    // Ensure initial or reset state
    clear_shutdown_in_progress();
    assert!(
        !is_shutdown_in_progress(),
        "Shutdown latch should be false initially or after clear"
    );

    // Simulate instance teardown (0 remaining instances)
    set_shutdown_in_progress();
    assert!(
        is_shutdown_in_progress(),
        "Shutdown latch should be true after set_shutdown_in_progress"
    );

    // Simulate new instance created in the same process (0 -> 1 reload)
    clear_shutdown_in_progress();
    assert!(
        !is_shutdown_in_progress(),
        "Shutdown latch should be reset to false on new instance initialization"
    );
}

#[test]
fn test_panic_report_without_install_emits_minimal_report() {
    // Do not call install_panic_hook: the hook path must degrade, never panic.
    let written = std::panic::catch_unwind(|| {
        let mut buf = [0u8; 16384];
        let n = write_panic_report(
            &mut buf,
            "test-no-snapshot",
            "test-thread",
            "src/common/panic_hook.rs:0:0",
            "synthetic panic",
        );
        (n, buf)
    })
    .expect("panic hook path must not panic when SYSTEM_SNAPSHOT is unset");

    let (n, buf) = written;
    assert!(n > 0, "minimal report must produce output");
    assert!(n <= 16384, "report must stay within the stack buffer");
    let report = std::str::from_utf8(&buf[..n]).expect("report is utf-8");
    assert!(report.contains("NeuralAmpModeler-rs CRASH REPORT"));
    assert!(report.contains("synthetic panic"));
    assert!(report.contains("test-no-snapshot"));
    if SYSTEM_SNAPSHOT.get().is_none() {
        assert!(
            report.contains("features=<snapshot not initialized>"),
            "missing snapshot must emit the minimal system-info markers"
        );
    }
}

#[test]
fn test_format_panic_report_none_snapshot_is_minimal() {
    let mut buf = [0u8; 16384];
    let n = format_panic_report_to_buf(
        &mut buf,
        "test-component",
        "test-thread",
        "src/test.rs:1:1",
        "test panic message",
        None,
    );
    assert!(n > 0);
    assert!(n <= 16384);
    let report = std::str::from_utf8(&buf[..n]).expect("report is utf-8");
    assert!(report.starts_with("======================================="));
    assert!(report.contains("test panic message"));
    assert!(report.contains("arch=<unavailable>"));
    assert!(report.contains("os=<unavailable> kernel=<unavailable>"));
    assert!(report.contains("features=<snapshot not initialized>"));
}
