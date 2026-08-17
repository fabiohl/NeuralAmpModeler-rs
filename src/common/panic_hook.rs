// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Panic hook facility to capture and write diagnostic bundles to disk on panic.
//!
//! The hook minimizes allocations during the crash path: `SystemSnapshot` is
//! pre-captured at `install_panic_hook` time, the report is formatted into a
//! stack-allocated `[u8; 16384]` buffer via `LimitWriter`, and `RwLock` reads
//! use `try_read()` with fallback to avoid deadlocks. Remaining heap
//! allocations (`var_os`, `PathBuf`) only occur when a writable `~/.cache/nam-rs`
//! directory is found.
//!
//! Crash files are automatically pruned: when the total number of `crash-*.txt`
//! files exceeds `MAX_CRASH_FILES`, the oldest (by modification time) are removed.

use crate::common::diagnostics::NamLogger;
use crate::common::diagnostics::SystemSnapshot;
use crate::common::diagnostics::snapshot::{
    ACTIVE_MODEL_INFO, ACTIVE_MODEL_NAME, ACTIVE_SAMPLE_RATE,
};
use core::fmt::Write as FmtWrite;
use std::cell::Cell;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;
use std::time::SystemTime;

static SYSTEM_SNAPSHOT: OnceLock<SystemSnapshot> = OnceLock::new();
static SHUTDOWN_IN_PROGRESS: OnceLock<bool> = OnceLock::new();

thread_local! {
    static PANIC_HOOK_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Sets the global flag signaling that shutdown is in progress.
///
/// This bypasses the panic hook to avoid race conditions or deadlock
/// during host-initiated cleanup.
pub fn set_shutdown_in_progress() {
    let _ = SHUTDOWN_IN_PROGRESS.set(true);
}

/// Checks if a shutdown (host-initiated or signal-initiated) is in progress.
pub fn is_shutdown_in_progress() -> bool {
    crate::common::spsc::SHUTDOWN.load(std::sync::atomic::Ordering::Acquire)
        || *SHUTDOWN_IN_PROGRESS.get().unwrap_or(&false)
}

struct LimitWriter<'a> {
    buf: &'a mut [u8],
    cursor: usize,
}

impl<'a> LimitWriter<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, cursor: 0 }
    }
}

impl FmtWrite for LimitWriter<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let bytes = s.as_bytes();
        let remaining = self.buf.len().saturating_sub(self.cursor);
        let len = bytes.len().min(remaining);
        self.buf[self.cursor..self.cursor + len].copy_from_slice(&bytes[..len]);
        self.cursor += len;
        Ok(())
    }
}

/// Maximum number of crash report files retained in `~/.cache/nam-rs/`.
const MAX_CRASH_FILES: usize = 10;

/// Prunes old crash files from `~/.cache/nam-rs/` if the total exceeds
/// [`MAX_CRASH_FILES`]. Crash files are identified by the `crash-*.txt` pattern
/// and sorted by modification time (oldest first, excluding `.tmp` files).
///
/// All I/O errors are silently swallowed — this runs inside the panic hook and
/// must never escalate into a new panic.
fn prune_old_crash_files(cache_dir: &std::path::Path) {
    let mut entries: Vec<(std::path::PathBuf, std::time::SystemTime)> =
        match std::fs::read_dir(cache_dir) {
            Ok(iter) => iter
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("crash-") && n.ends_with(".txt"))
                        .unwrap_or(false)
                })
                .filter_map(|e| {
                    let mtime = e.metadata().ok()?.modified().ok()?;
                    Some((e.path(), mtime))
                })
                .collect(),
            Err(_) => return,
        };

    if entries.len() <= MAX_CRASH_FILES {
        return;
    }

    entries.sort_by_key(|(_, mtime)| *mtime);
    let excess = entries.len() - MAX_CRASH_FILES;
    for (path, _) in entries.iter().take(excess) {
        let _ = std::fs::remove_file(path);
    }
}

fn format_panic_report_to_buf(
    buf: &mut [u8],
    component: &str,
    thread_name: &str,
    location: &str,
    payload_str: &str,
    system: &SystemSnapshot,
) -> usize {
    let mut w = LimitWriter::new(buf);

    let _ = write!(
        w,
        "================================================================\n\
         NAM-rs CRASH REPORT\n\
         ================================================================\n\
         Version: {}\n\
         Component: {}\n\
         Thread: {}\n\
         Location: {}\n\
         Message: {}\n\n",
        system.version, component, thread_name, location, payload_str
    );

    let _ = writeln!(w, "──── Runtime State ─────────────────────────────");

    match ACTIVE_MODEL_NAME.try_read() {
        Ok(guard) => {
            if guard.is_empty() {
                let _ = writeln!(w, "model=none");
            } else {
                let _ = writeln!(w, "model={}", *guard);
            }
        }
        Err(_) => {
            let _ = writeln!(w, "model=<unavailable>");
        }
    }

    match ACTIVE_MODEL_INFO.try_read() {
        Ok(guard) => {
            if let Some(ref info) = *guard {
                let _ = writeln!(w, "model.arch={}", info.arch_label);
                let _ = writeln!(w, "model.topology={}", info.topology);
                let _ = writeln!(w, "model.channels={}", info.channels);
                let _ = writeln!(w, "model.receptive_field={}", info.receptive_field);
                let _ = writeln!(w, "model.sample_rate={}", info.model_sample_rate);
                let _ = writeln!(w, "model.weights_layout={}", info.weights_layout);
                let _ = writeln!(w, "model.path_basename={}", info.path_basename);
            }
        }
        Err(_) => {
            let _ = writeln!(w, "model.info=<unavailable>");
        }
    }

    let rate = ACTIVE_SAMPLE_RATE.load(Ordering::Relaxed);
    if rate != 0 {
        let _ = writeln!(w, "audio.sr={}", rate);
    }

    // Recent Log Trace (up to 30 lines, zero-alloc via try_lock)
    let _ = writeln!(w, "──── Recent Log Trace ─────────────────────────");
    if let Some(buffer) = NamLogger::log_buffer() {
        let count = buffer.try_render_trace_into(&mut w, 30);
        if count == 0 {
            let _ = writeln!(w, "<log buffer locked>");
        }
    } else {
        let _ = writeln!(w, "<log buffer not initialized>");
    }

    let unix_ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let _ = writeln!(w, "──── System Info ─────────────────────────────");
    let _ = writeln!(w, "arch={}", system.arch);
    let _ = writeln!(w, "os={} kernel={}", system.os, system.kernel);
    let _ = write!(
        w,
        "host_library={}",
        system.host_library_version.as_deref().unwrap_or("N/A")
    );
    let _ = write!(w, "\nfeatures=");
    if system.features.is_empty() {
        let _ = write!(w, "none (baseline x86-64-v3 only)");
    } else {
        for (i, f) in system.features.iter().enumerate() {
            if i > 0 {
                let _ = write!(w, ", ");
            }
            let _ = write!(w, "{}", f);
        }
    }
    let _ = writeln!(w);
    let _ = writeln!(w, "timestamp_unix={}", unix_ts);

    let _ = write!(
        w,
        "────────────────────────────────────────────────\n\
         Copy the block above when opening a support ticket.\n"
    );

    w.cursor
}

/// Public entry point for heap-audit tests to exercise `format_panic_report_to_buf`
/// with the pre-captured `SystemSnapshot` without triggering a real panic.
#[cfg(feature = "heap-audit")]
#[cfg_attr(docsrs, doc(cfg(feature = "heap-audit")))]
pub fn format_panic_report_for_audit_test(
    buf: &mut [u8],
    component: &str,
    thread_name: &str,
    location: &str,
    payload_str: &str,
) -> usize {
    let system = SYSTEM_SNAPSHOT
        .get()
        .expect("SystemSnapshot not initialized");
    format_panic_report_to_buf(buf, component, thread_name, location, payload_str, system)
}

/// Installs a panic hook that writes a zero-alloc crash report to
/// `~/.cache/nam-rs/crash-<unix_ts>-<component>.txt`.
pub fn install_panic_hook(component: &'static str) {
    SYSTEM_SNAPSHOT.get_or_init(SystemSnapshot::capture);

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if is_shutdown_in_progress() {
            prev_hook(info);
            return;
        }

        if PANIC_HOOK_ACTIVE.replace(true) {
            prev_hook(info);
            return;
        }

        let thread = std::thread::current();
        let thread_name = thread.name().unwrap_or("unknown");

        let payload = info.payload();
        let payload_str = if let Some(s) = payload.downcast_ref::<&str>() {
            *s
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.as_str()
        } else {
            "Box<dyn Any>"
        };

        let mut location_buf = [0u8; 256];
        let location_str = {
            let mut lw = LimitWriter::new(&mut location_buf);
            if let Some(loc) = info.location() {
                let _ = write!(lw, "{}:{}:{}", loc.file(), loc.line(), loc.column());
            } else {
                let _ = write!(lw, "unknown");
            }
            let len = lw.cursor;
            std::str::from_utf8(&location_buf[..len]).unwrap_or("<invalid utf-8>")
        };

        let system = SYSTEM_SNAPSHOT
            .get()
            .expect("SystemSnapshot not initialized");

        let mut report_buf = [0u8; 16384];
        let written = format_panic_report_to_buf(
            &mut report_buf,
            component,
            thread_name,
            location_str,
            payload_str,
            system,
        );

        if let Some(home_dir) = std::env::var_os("HOME") {
            let mut cache_dir = PathBuf::from(home_dir);
            cache_dir.push(".cache/nam-rs");

            if std::fs::create_dir_all(&cache_dir).is_ok() {
                #[cfg(unix)]
                {
                    use std::fs::Permissions;
                    use std::os::unix::fs::PermissionsExt;
                    let _ = std::fs::set_permissions(&cache_dir, Permissions::from_mode(0o700));
                }

                let unix_ts = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);

                let mut filename_buf = [0u8; 128];
                let filename = {
                    let mut lw = LimitWriter::new(&mut filename_buf);
                    let _ = write!(lw, "crash-{}-{}.txt", unix_ts, component);
                    let len = lw.cursor;
                    std::str::from_utf8(&filename_buf[..len]).unwrap_or("crash-unknown.txt")
                };

                let mut tmp_buf = [0u8; 128];
                let tmp_filename = {
                    let mut lw = LimitWriter::new(&mut tmp_buf);
                    let _ = write!(lw, "crash-{}-{}.tmp", unix_ts, component);
                    let len = lw.cursor;
                    std::str::from_utf8(&tmp_buf[..len]).unwrap_or("crash-unknown.tmp")
                };

                let file_path = cache_dir.join(filename);
                let tmp_file_path = cache_dir.join(tmp_filename);

                let mut options = OpenOptions::new();
                options.write(true).create(true).truncate(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    options.mode(0o600);
                }

                if let Ok(mut file) = options.open(&tmp_file_path) {
                    if file.write_all(&report_buf[..written]).is_ok() {
                        let _ = file.sync_all();
                        drop(file);
                        let _ = std::fs::rename(&tmp_file_path, &file_path);
                        prune_old_crash_files(&cache_dir);
                    } else {
                        let _ = std::fs::remove_file(&tmp_file_path);
                    }
                }
            }
        }

        PANIC_HOOK_ACTIVE.set(false);
        prev_hook(info);
    }));
}
