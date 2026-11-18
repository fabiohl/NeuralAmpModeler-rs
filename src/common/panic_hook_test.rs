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

#[test]
fn test_production_crash_directory_windows_and_posix_fallbacks() {
    use std::ffi::OsStr;

    // Windows prefers LOCALAPPDATA over USERPROFILE and HOME.
    let win_local = production_crash_directory_for(
        true,
        Some(OsStr::new(r"C:\Users\Alice\AppData\Local")),
        Some(OsStr::new(r"C:\Users\Alice")),
        Some(OsStr::new("/home/alice")),
    );
    assert_eq!(
        win_local,
        Some(
            PathBuf::from(r"C:\Users\Alice\AppData\Local")
                .join("neural-amp-modeler-rs")
                .join("crashes")
        )
    );

    // Windows falls back to USERPROFILE when LOCALAPPDATA is unset or empty.
    let win_profile = production_crash_directory_for(
        true,
        Some(OsStr::new("")),
        Some(OsStr::new(r"C:\Users\Alice")),
        None,
    );
    assert_eq!(
        win_profile,
        Some(
            PathBuf::from(r"C:\Users\Alice")
                .join(".cache")
                .join("neural-amp-modeler-rs")
        )
    );

    // Windows falls back to HOME if neither LOCALAPPDATA nor USERPROFILE is set.
    let win_home =
        production_crash_directory_for(true, None, None, Some(OsStr::new("/c/Users/Alice")));
    assert_eq!(
        win_home,
        Some(
            PathBuf::from("/c/Users/Alice")
                .join(".cache")
                .join("neural-amp-modeler-rs")
        )
    );

    // POSIX ignores Windows-only variables and resolves HOME.
    let posix = production_crash_directory_for(
        false,
        Some(OsStr::new(r"C:\Ignored")),
        Some(OsStr::new(r"C:\Ignored")),
        Some(OsStr::new("/home/alice")),
    );
    assert_eq!(
        posix,
        Some(
            PathBuf::from("/home/alice")
                .join(".cache")
                .join("neural-amp-modeler-rs")
        )
    );

    assert_eq!(production_crash_directory_for(true, None, None, None), None);
    assert_eq!(
        production_crash_directory_for(false, None, None, None),
        None
    );
}

#[test]
fn test_panic_report_reads_host_library_version_set_after_hook_install() {
    let mut snap = SystemSnapshot::capture();
    snap.host_library_version = None;

    crate::common::diagnostics::set_host_library_version("test-host 1.2.3".to_string());

    let mut buf = [0u8; 16384];
    let n = format_panic_report_to_buf(
        &mut buf,
        "test-component",
        "test-thread",
        "src/test.rs:1:1",
        "test panic message",
        Some(&snap),
    );
    let report = std::str::from_utf8(&buf[..n]).expect("report is utf-8");
    assert!(
        report.contains("host_library=test-host 1.2.3"),
        "expected dynamic host_library_version fallback, got:\n{report}"
    );
}

#[test]
fn test_panic_report_suppression_guard_skips_file_write() {
    let crash_dir = std::path::PathBuf::from("target/test_crashes_suppressed");
    let _ = std::fs::remove_dir_all(&crash_dir);
    std::fs::create_dir_all(&crash_dir).expect("create test crash dir");
    let _env = EnvVarGuard::set("NAM_CRASH_DIR", &crash_dir);

    install_panic_hook("test-suppression");
    assert!(!PANIC_REPORT_SUPPRESSED.with(|c| c.get()));

    {
        let _guard = suppress_panic_report_on_current_thread();
        assert!(PANIC_REPORT_SUPPRESSED.with(|c| c.get()));
        let res = std::panic::catch_unwind(|| {
            panic!("contained third-party panic");
        });
        assert!(res.is_err());
    }

    assert!(!PANIC_REPORT_SUPPRESSED.with(|c| c.get()));
    let entries: Vec<_> = std::fs::read_dir(&crash_dir)
        .expect("read test crash dir")
        .filter_map(|e| e.ok())
        .collect();
    assert!(
        entries.is_empty(),
        "expected no crash file when suppression guard is active, found: {entries:?}"
    );
    let _ = std::fs::remove_dir_all(&crash_dir);
}
