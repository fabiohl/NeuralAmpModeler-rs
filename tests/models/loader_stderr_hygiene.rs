// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Loader stderr hygiene: rejected model loads must never paint diagnostic
//! support blocks on the host process's stderr (library contract).
//!
//! `NeuralAmpModeler-rs` is a library embedded in plugins, daemons and DAWs;
//! loading failures are reported through the enriched `Result::Err(LoadError)`
//! and structured `log` records. The visual support block (`NamDiagnostic::
//! support_block()`) is rendered only by executable tools and crash hooks.
//!
//! `libtest` captures the stderr of in-process tests, so a leaking
//! `eprintln!` would be invisible to assertions. The negative load therefore
//! runs in a child process (this binary re-spawning itself with `--exact`),
//! whose raw stderr is captured verbatim and scanned for diagnostic banners.

use std::process::Command;

/// Exact name of the inner negative-load test executed by the child process.
const INNER_TEST: &str = "loader_stderr_hygiene::inner_negative_load_is_silent";

/// Banner lines painted by the former `NamDiagnostic::emit` path — none of
/// them may appear on stderr during a library load rejection.
const FORBIDDEN_STDERR_MARKERS: [&str; 3] = [
    "NeuralAmpModeler-rs Diagnostic",
    "Copy the block above when opening a support ticket.",
    "──── Runtime State ────",
];

#[test]
fn test_load_error_never_leaks_support_blocks_to_stderr() {
    let exe = std::env::current_exe().expect("test binary path must be resolvable");
    let output = Command::new(exe)
        .args(["--exact", INNER_TEST, "--nocapture", "--test-threads", "1"])
        .output()
        .expect("child test process must spawn");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "inner negative-load test must pass; child stderr:\n{stderr}"
    );
    for marker in FORBIDDEN_STDERR_MARKERS {
        assert!(
            !stderr.contains(marker),
            "loader rejection leaked diagnostic block marker {marker:?} to stderr:\n{stderr}"
        );
    }
}

/// Inner negative-load scenarios (executed only by the child process above).
///
/// Exercises the two rejection families that previously painted the visual
/// support block via `NamDiagnostic::emit`:
/// 1. unreadable file (I/O rejection), and
/// 2. malformed `.nam` JSON (parse rejection through the full loader path).
#[test]
fn inner_negative_load_is_silent() {
    use neural_amp_modeler_rs::SystemSnapshot;
    use neural_amp_modeler_rs::loader::{LoadError, LoadOptions, load_and_build_model};
    use std::path::PathBuf;

    let sys = SystemSnapshot::capture();

    // 1. Missing file → typed I/O rejection.
    let missing = std::env::temp_dir().join(format!(
        "nam_stderr_hygiene_missing_{}.nam",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&missing);
    let missing = PathBuf::from(&missing);
    match load_and_build_model(&missing, &sys, false, LoadOptions::default()) {
        Err(LoadError::Io(_)) => {}
        other => panic!("expected LoadError::Io for missing file, got: {other:?}"),
    }

    // 2. Malformed JSON payload → typed parse rejection.
    let malformed = std::env::temp_dir().join(format!(
        "nam_stderr_hygiene_malformed_{}.nam",
        std::process::id()
    ));
    std::fs::write(&malformed, "{ invalid json").expect("temp file must be writable");
    let res = load_and_build_model(&malformed, &sys, false, LoadOptions::default());
    let _ = std::fs::remove_file(&malformed);
    assert!(
        matches!(res, Err(LoadError::Json(_))),
        "malformed JSON must be rejected as LoadError::Json, got: {res:?}"
    );
}
