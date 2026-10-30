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
    #[serde(skip_serializing_if = "Option::is_none")]
    run_id: Option<&'a str>,
    pid: u32,
    test: &'a str,
    reason: &'a str,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    optional: bool,
}

/// Classifies a skip reason as an optional, non-distributable fixture.
///
/// Only the documented optional families are non-blocking under `--strict`;
/// a missing distributable fixture keeps reporting a blocking capability gap.
fn is_optional_reason(reason: &str) -> bool {
    matches!(reason, "models_nondist_absent" | "models_nondist_empty")
        || reason.starts_with("community_model_absent:")
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

    let require_cpp_oracle = std::env::var("NAM_REQUIRE_CPP_ORACLE").is_ok_and(|v| {
        let v = v.trim();
        v == "1" || v.eq_ignore_ascii_case("true")
    });
    if is_oracle && require_cpp_oracle {
        panic!("NAM_REQUIRE_CPP_ORACLE=1 — aborting test: {test} ({reason})");
    }

    let optional = is_optional_reason(reason);
    let marker = if optional {
        "SKIP_OPTIONAL"
    } else {
        "SKIP_CAPABILITY"
    };
    eprintln!("[STATUS] {marker} reason=\"{reason}\" ({test})");

    let _guard = SKIP_LOG_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let log_dir = manifest_dir.join("target/logs");
    let _ = std::fs::create_dir_all(&log_dir);
    let log_path = log_dir.join("skips.jsonl");

    let run_id = std::env::var("NAM_RUN_ID").ok();
    let pid = std::process::id();
    let record = SkipRecord {
        run_id: run_id.as_deref(),
        pid,
        test,
        reason,
        optional,
    };

    match OpenOptions::new().create(true).append(true).open(&log_path) {
        Ok(mut f) => {
            let line =
                serde_json::to_string(&record).expect("SkipRecord serialization must succeed");
            if let Err(e) = writeln!(f, "{line}") {
                panic!("Failed to write skip record to {}: {e}", log_path.display());
            }
        }
        Err(e) => {
            panic!("Failed to open skip log {}: {e}", log_path.display());
        }
    }
}
