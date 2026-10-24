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

use std::sync::{Mutex, OnceLock};

static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct EnvVarGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    key: &'static str,
    saved: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let lock = env_lock();
        let saved = std::env::var_os(key);
        // SAFETY: test-scoped env mutation serialized by ENV_LOCK.
        unsafe { std::env::set_var(key, value) };
        Self {
            _lock: lock,
            key,
            saved,
        }
    }

    fn remove(key: &'static str) -> Self {
        let lock = env_lock();
        let saved = std::env::var_os(key);
        // SAFETY: test-scoped env mutation serialized by ENV_LOCK.
        unsafe { std::env::remove_var(key) };
        Self {
            _lock: lock,
            key,
            saved,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(ref val) = self.saved {
            // SAFETY: restoring ambient pre-test env under ENV_LOCK.
            unsafe { std::env::set_var(self.key, val) };
        } else {
            // SAFETY: restoring ambient pre-test env under ENV_LOCK.
            unsafe { std::env::remove_var(self.key) };
        }
    }
}

#[test]
fn test_crash_directory_env_override() {
    let custom_path = std::path::PathBuf::from("/tmp/nam_test_custom_crash_dir");

    let _guard = EnvVarGuard::set("NAM_CRASH_DIR", &custom_path);
    assert_eq!(crash_directory(), Some(custom_path.clone()));

    // When empty, falls back to CARGO_TARGET_TMPDIR or target/test_crashes under test harness
    // SAFETY: isolated test env mutation under EnvVarGuard mutex.
    unsafe {
        std::env::set_var("NAM_CRASH_DIR", "");
    }
    let fallback = crash_directory().expect("crash directory must be set under test");
    let fallback_str = fallback.to_string_lossy();
    assert!(
        fallback_str.contains("target") || fallback_str.contains("tmp"),
        "fallback must resolve to test harness directory, got: {:?}",
        fallback
    );
}

#[test]
fn test_crash_directory_contract_avoids_user_cache() {
    // Under testing/cfg(test), if NAM_CRASH_DIR is unset or empty,
    // the directory MUST NOT start with $HOME/.cache
    let _guard = EnvVarGuard::remove("NAM_CRASH_DIR");
    let dir = crash_directory().expect("crash directory must be available in test harness");
    let dir_str = dir.to_string_lossy();
    if let Some(home) = std::env::var_os("HOME") {
        let home_cache = std::path::PathBuf::from(home).join(".cache");
        let home_cache_str = home_cache.to_string_lossy();
        assert!(
            !dir_str.starts_with(home_cache_str.as_ref()),
            "crash directory in test harness must not point to user cache, got: {dir_str}"
        );
    }
    assert!(
        dir_str.contains("target") || dir_str.contains("tmp"),
        "crash directory in test harness must resolve to CARGO_TARGET_TMPDIR or target/test_crashes, got: {dir_str}"
    );
}
