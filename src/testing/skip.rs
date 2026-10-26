// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Test skip reporting utility.
//!
//! Centralizes logging of skipped test capabilities or missing fixtures into
//! `target/logs/skips.jsonl` (for aggregation into GAP status in CI and quick test runner)
//! and stderr (`[STATUS] SKIP_CAPABILITY ...`).
//!
//! Panics when `NAM_REQUIRE_CPP_ORACLE=1` and the skip is related to the C++ oracle / render tool.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

static SKIP_LOG_LOCK: Mutex<()> = Mutex::new(());

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq, Eq)]
struct SkipRecord<'a> {
    test: &'a str,
    reason: &'a str,
}

/// Appends a skip record to `target/logs/skips.jsonl` and emits standard diagnostic output.
///
/// If `NAM_REQUIRE_CPP_ORACLE=1` and the test/reason involves the C++ oracle or render tool,
/// this immediately panics to fail closed.
pub fn report_skip(test: &str, reason: &str) {
    let is_oracle = reason.contains("render")
        || reason.contains("oracle")
        || reason.contains("namcore")
        || test.contains("cpp_parity")
        || test.contains("sequential_cpp");

    if is_oracle && std::env::var("NAM_REQUIRE_CPP_ORACLE").as_deref() == Ok("1") {
        panic!("NAM_REQUIRE_CPP_ORACLE=1 — aborting test: {test} ({reason})");
    }

    eprintln!("[STATUS] SKIP_CAPABILITY reason=\"{reason}\" ({test})");

    let _guard = SKIP_LOG_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let log_dir = manifest_dir.join("target/logs");
    let _ = std::fs::create_dir_all(&log_dir);
    let log_path = log_dir.join("skips.jsonl");

    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&log_path) {
        let record = SkipRecord { test, reason };
        if let Ok(line) = serde_json::to_string(&record) {
            let _ = writeln!(f, "{line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_skip_serialization() {
        let test_name = "unit_test_probe";
        let reason = "fixture_not_found";
        report_skip(test_name, reason);

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let log_path = manifest_dir.join("target/logs/skips.jsonl");
        assert!(log_path.exists());
        let content = std::fs::read_to_string(&log_path).expect("read skips.jsonl");
        assert!(content.contains(test_name));
        assert!(content.contains(reason));
        let _ = std::fs::remove_file(&log_path);
    }

    #[test]
    #[should_panic(expected = "NAM_REQUIRE_CPP_ORACLE=1")]
    fn test_report_skip_panics_on_oracle_strict() {
        // SAFETY: Test runs in isolated unit test module; modifying test-only env var.
        unsafe {
            std::env::set_var("NAM_REQUIRE_CPP_ORACLE", "1");
        }
        struct ResetEnv;
        impl Drop for ResetEnv {
            fn drop(&mut self) {
                // SAFETY: Resetting test environment variable on drop.
                unsafe {
                    std::env::remove_var("NAM_REQUIRE_CPP_ORACLE");
                }
            }
        }
        let _guard = ResetEnv;
        report_skip("cpp_parity_test", "render_tool_unavailable");
    }
}
