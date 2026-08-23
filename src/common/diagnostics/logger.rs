// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Ring buffer `LogBuffer` — in-memory circular history of formatted log
//! entries, suitable for rendering into diagnostic dumps and crash reports.
//!
//! # RT-Safety
//!
//! The `LogBuffer` is strictly off-RT. RT callers must **never** invoke `log::*`
//! and therefore never touch this buffer from the audio thread. The public API
//! uses `std::sync::Mutex` — ordinary (blocking) `lock()` for off-RT use,
//! `try_lock()` for panic-hook contexts where deadlock must be avoided.
//!
//! # NamLogger — Facade Bridge
//!
//! The `NamLogger` struct implements the `log::Log` trait, acting as the
//! global backend for the `log` crate facade. It maintains:
//! 1. A global `LogBuffer` ring buffer for diagnostic trace.
//! 2. A list of registered host-log sinks (host application instances).
//! 3. Optional stderr output for standalone/CLI mode.
//!
//! ## Multi-instance safety
//!
//! `NamLogger::init()` is protected by `OnceLock`, so the first host instance
//! installs the global logger and subsequent instances reuse it without
//! triggering `SetLoggerError`.

use log::{LevelFilter, Log, Metadata, Record, SetLoggerError};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default maximum number of log records held in the ring buffer.
const DEFAULT_CAPACITY: usize = 256;

/// Callback type for host-log sinks.
///
/// Each host instance registers a sink via `register_sink()`.
/// The callback receives (severity-as-string, formatted-message).
pub type HostLogFn = dyn for<'a, 'b> Fn(&'a str, &'b str) + Send + Sync;

/// A single log record stored in the `LogBuffer`.
#[derive(Debug, Clone)]
pub struct LogRecord {
    /// UNIX timestamp (seconds) when the log entry was emitted.
    pub timestamp_secs: u64,
    /// Log level string (`ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE`).
    pub level: String,
    /// Target / module that produced the log entry.
    pub target: String,
    /// Formatted log message.
    pub message: String,
}

impl LogRecord {
    /// Creates a new `LogRecord` with the current wall-clock timestamp.
    pub fn now(
        level: impl Into<String>,
        target: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            timestamp_secs: ts,
            level: level.into(),
            target: target.into(),
            message: message.into(),
        }
    }
}

/// Thread-safe ring buffer for retaining a bounded history of log entries.
///
/// Uses a `Mutex<VecDeque<LogRecord>>` with a fixed maximum capacity. When
/// capacity is exceeded, the oldest entry is discarded (FIFO). Designed
/// exclusively for off-RT use; the `try_push` variant is safe to call from
/// panic hooks because it falls back gracefully on lock contention.
#[derive(Debug)]
pub struct LogBuffer {
    inner: Mutex<VecDeque<LogRecord>>,
    capacity: usize,
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

impl LogBuffer {
    /// Creates a new `LogBuffer` with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "LogBuffer capacity must be > 0");
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    /// Pushes a `LogRecord` into the buffer, discarding the oldest entry if
    /// the buffer has reached its capacity.
    ///
    /// Uses a blocking `lock()` — intended for off-RT callers.
    pub fn push(&self, record: LogRecord) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.len() >= self.capacity {
            inner.pop_front();
        }
        inner.push_back(record);
    }

    /// Attempts to push without blocking. Returns `false` if the lock could
    /// not be acquired (e.g. held by a panicking thread), `true` on success.
    ///
    /// Safe to call from panic hooks and crash-report paths.
    pub fn try_push(&self, record: LogRecord) -> bool {
        let Some(mut inner) = self.inner.try_lock().ok() else {
            return false;
        };
        if inner.len() >= self.capacity {
            inner.pop_front();
        }
        inner.push_back(record);
        true
    }

    /// Returns a snapshot of all currently buffered log records (cloned).
    /// Does not mutate the original state.
    pub fn snapshot(&self) -> Vec<LogRecord> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// Attempts to take a snapshot without blocking. Returns `None` if the
    /// lock could not be acquired.
    pub fn try_snapshot(&self) -> Option<Vec<LogRecord>> {
        let inner = self.inner.try_lock().ok()?;
        Some(inner.iter().cloned().collect())
    }

    /// Renders up to `limit` most recent log entries as a formatted string
    /// suitable for embedding in diagnostic dumps and crash reports.
    ///
    /// Each line follows the format:
    /// `[timestamp_secs] LEVEL target: message`
    pub fn render_trace(&self, limit: usize) -> String {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let start = if inner.len() > limit {
            inner.len() - limit
        } else {
            0
        };
        let mut out = String::with_capacity(limit * 128);
        for record in inner.range(start..) {
            out.push_str(&format!(
                "[{}] {} {}: {}\n",
                record.timestamp_secs, record.level, record.target, record.message
            ));
        }
        out
    }

    /// Attempts to render a trace without blocking. Returns `None` if the
    /// lock could not be acquired.
    pub fn try_render_trace(&self, limit: usize) -> Option<String> {
        let inner = self.inner.try_lock().ok()?;
        let start = if inner.len() > limit {
            inner.len() - limit
        } else {
            0
        };
        let mut out = String::with_capacity(limit * 128);
        for record in inner.range(start..) {
            out.push_str(&format!(
                "[{}] {} {}: {}\n",
                record.timestamp_secs, record.level, record.target, record.message
            ));
        }
        Some(out)
    }

    /// Attempts to render a trace directly into a [`std::fmt::Write`] without
    /// blocking or allocating. Returns the number of records written (0 if the
    /// lock could not be acquired).
    ///
    /// Designed for zero-alloc use in panic hooks and crash-report paths.
    pub fn try_render_trace_into(&self, writer: &mut impl std::fmt::Write, limit: usize) -> usize {
        let inner = match self.inner.try_lock() {
            Ok(guard) => guard,
            Err(_) => return 0,
        };
        let start = if inner.len() > limit {
            inner.len() - limit
        } else {
            0
        };
        let mut count = 0;
        for record in inner.range(start..) {
            let _ = writeln!(
                writer,
                "[{}] {} {}: {}",
                record.timestamp_secs, record.level, record.target, record.message
            );
            count += 1;
        }
        count
    }

    /// Returns the current number of records in the buffer.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Returns `true` if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the fixed capacity of the ring buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// NamLogger — Global log::Log backend
// ---------------------------------------------------------------------------

/// Global `NamLogger` instance (initialized once per process).
static NAM_LOGGER: OnceLock<NamLogger> = OnceLock::new();

/// Converts a `log::Level` to a static string slice.
fn level_str(level: log::Level) -> &'static str {
    match level {
        log::Level::Error => "ERROR",
        log::Level::Warn => "WARN",
        log::Level::Info => "INFO",
        log::Level::Debug => "DEBUG",
        log::Level::Trace => "TRACE",
    }
}

fn format_wall_clock_time() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    let hours = (secs / 3600) % 24;
    let minutes = (secs / 60) % 60;
    let seconds = secs % 60;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

/// Central bridge between the `log` crate facade and NeuralAmpModeler-rs logging infra.
///
/// Implements `log::Log` and routes every log record to:
/// 1. The global `LogBuffer` ring buffer.
/// 2. stderr (if configured via `LoggerConfig::emit_stderr`).
/// 3. All registered host-log sinks (`Weak<HostLogFn>`).
///
/// ## Multi-instance
///
/// `NamLogger::init()` installs the global `log` backend exactly once
/// via `OnceLock`. Subsequent calls are no-ops, so multiple host instances
/// in the same process never trigger `SetLoggerError`.
#[derive(Debug)]
pub struct NamLogger {
    buffer: LogBuffer,
    sinks: Mutex<Vec<Weak<HostLogFn>>>,
    emit_stderr: Mutex<bool>,
    max_level: Mutex<LevelFilter>,
}

/// Configuration for `NamLogger::init()`.
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Maximum log level to record.
    pub level_filter: LevelFilter,
    /// Whether to emit log records to stderr (standalone/CLI mode).
    pub emit_stderr: bool,
}

impl NamLogger {
    /// Creates and installs the global `NamLogger` as the `log` crate backend.
    ///
    /// Safe to call multiple times — only the first call takes effect.
    /// Returns a reference to the global instance (or `None` if already
    /// initialized without setting the logger).
    pub fn init(config: LoggerConfig) -> Result<&'static NamLogger, SetLoggerError> {
        let logger = NAM_LOGGER.get_or_init(|| NamLogger {
            buffer: LogBuffer::default(),
            sinks: Mutex::new(Vec::new()),
            emit_stderr: Mutex::new(config.emit_stderr),
            max_level: Mutex::new(config.level_filter),
        });
        match log::set_logger(logger) {
            Ok(()) => {
                log::set_max_level(config.level_filter);
                Ok(logger)
            }
            Err(_) => {
                logger.set_max_level(config.level_filter);
                Ok(logger)
            }
        }
    }

    /// Returns a reference to the global `NamLogger`, if initialized.
    pub fn global() -> Option<&'static NamLogger> {
        NAM_LOGGER.get()
    }

    /// Returns a reference to the global `LogBuffer`, if the logger is initialized.
    pub fn log_buffer() -> Option<&'static LogBuffer> {
        NAM_LOGGER.get().map(|l| &l.buffer)
    }

    /// Registers a host-log sink. The sink is stored as a `Weak`
    /// reference — when the host instance is destroyed and its `Arc` drops,
    /// the `Weak` becomes dead and is automatically purged on the next log
    /// dispatch.
    pub fn register_sink(&self, sink: &Arc<HostLogFn>) {
        let mut sinks = self.sinks.lock().unwrap_or_else(|e| e.into_inner());
        sinks.push(Arc::downgrade(sink));
    }

    /// Updates the maximum log level filter at runtime.
    pub fn set_max_level(&self, level: LevelFilter) {
        *self.max_level.lock().unwrap_or_else(|e| e.into_inner()) = level;
        log::set_max_level(level);
    }
}

impl Log for NamLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        let max = *self.max_level.lock().unwrap_or_else(|e| e.into_inner());
        metadata.level() <= max
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let level = record.level();
        let target = record.target();
        let message = record.args().to_string();

        let log_record = LogRecord::now(level_str(level), target, &message);
        self.buffer.push(log_record);

        // stderr output if configured
        if *self.emit_stderr.lock().unwrap_or_else(|e| e.into_inner()) {
            eprintln!(
                "{time} {level:5} {target}: {message}",
                time = format_wall_clock_time(),
                level = level_str(level),
                target = record.target(),
                message = record.args()
            );
        }

        // Dispatch to registered host-log sinks
        let mut sinks = self.sinks.lock().unwrap_or_else(|e| e.into_inner());
        let severity = level_str(level);
        sinks.retain(|weak| {
            if let Some(sink) = weak.upgrade() {
                sink(severity, &message);
                true
            } else {
                false
            }
        });
    }

    fn flush(&self) {}
}

#[cfg(test)]
#[path = "logger_test.rs"]
mod logger_test;
