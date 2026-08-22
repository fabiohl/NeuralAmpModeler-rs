// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use log::LevelFilter;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

// =========================================================================
// LogBuffer unit tests
// =========================================================================

#[test]
fn log_buffer_default_capacity() {
    let buf = LogBuffer::default();
    assert_eq!(buf.capacity(), 256);
    assert!(buf.is_empty());
}

#[test]
fn log_buffer_custom_capacity() {
    let buf = LogBuffer::new(8);
    assert_eq!(buf.capacity(), 8);
    assert!(buf.is_empty());
}

#[test]
#[should_panic(expected = "capacity must be > 0")]
fn log_buffer_zero_capacity_panics() {
    LogBuffer::new(0);
}

#[test]
fn push_single_record() {
    let buf = LogBuffer::new(4);
    buf.push(LogRecord::now("INFO", "test", "hello"));
    assert_eq!(buf.len(), 1);

    let snap = buf.snapshot();
    assert_eq!(snap.len(), 1);
    assert_eq!(snap[0].level, "INFO");
    assert_eq!(snap[0].target, "test");
    assert_eq!(snap[0].message, "hello");
    assert!(snap[0].timestamp_secs > 0);
}

#[test]
fn push_up_to_capacity() {
    let buf = LogBuffer::new(3);
    for i in 0..3 {
        buf.push(LogRecord::now("INFO", "test", format!("msg{}", i)));
    }
    assert_eq!(buf.len(), 3);
}

#[test]
fn overflow_discards_oldest_fifo() {
    let buf = LogBuffer::new(3);
    for i in 0..5 {
        buf.push(LogRecord::now("INFO", "test", format!("msg{}", i)));
    }
    assert_eq!(buf.len(), 3);

    let snap = buf.snapshot();
    assert_eq!(snap[0].message, "msg2");
    assert_eq!(snap[1].message, "msg3");
    assert_eq!(snap[2].message, "msg4");
}

#[test]
fn snapshot_does_not_mutate_original() {
    let buf = LogBuffer::new(3);
    buf.push(LogRecord::now("INFO", "test", "a"));
    buf.push(LogRecord::now("WARN", "test", "b"));

    let snap1 = buf.snapshot();
    assert_eq!(snap1.len(), 2);

    buf.push(LogRecord::now("ERROR", "test", "c"));

    let snap2 = buf.snapshot();
    assert_eq!(snap2.len(), 3);

    assert_eq!(snap1.len(), 2);
    assert_eq!(snap1[0].message, "a");
    assert_eq!(snap1[1].message, "b");
}

#[test]
fn render_trace_within_limit() {
    let buf = LogBuffer::new(10);
    buf.push(LogRecord::now("INFO", "mod", "one"));
    buf.push(LogRecord::now("WARN", "mod", "two"));

    let trace = buf.render_trace(3);
    assert!(trace.contains("[") && trace.contains("INFO mod: one"));
    assert!(trace.contains("WARN mod: two"));
    assert_eq!(trace.matches('\n').count(), 2);
}

#[test]
fn render_trace_clips_to_limit() {
    let buf = LogBuffer::new(10);
    for i in 0..5 {
        buf.push(LogRecord::now("INFO", "t", format!("m{}", i)));
    }
    let trace = buf.render_trace(2);
    assert_eq!(trace.matches('\n').count(), 2);
    assert!(trace.contains("m3"));
    assert!(trace.contains("m4"));
    assert!(!trace.contains("m2"));
}

#[test]
fn render_trace_limit_beyond_capacity_shows_all() {
    let buf = LogBuffer::new(4);
    for i in 0..4 {
        buf.push(LogRecord::now("INFO", "t", format!("m{}", i)));
    }
    let trace = buf.render_trace(100);
    assert_eq!(trace.matches('\n').count(), 4);
}

#[test]
fn concurrent_push() {
    let buf = Arc::new(LogBuffer::new(256));
    let threads: Vec<_> = (0..8)
        .map(|t| {
            let buf = Arc::clone(&buf);
            thread::spawn(move || {
                for i in 0..100 {
                    buf.push(LogRecord::now(
                        "INFO",
                        "concurrent",
                        format!("t{}_msg{}", t, i),
                    ));
                }
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }

    assert_eq!(buf.len(), 256);
}

#[test]
fn try_push_succeeds_when_uncontested() {
    let buf = LogBuffer::new(8);
    assert!(buf.try_push(LogRecord::now("INFO", "t", "x")));
    assert_eq!(buf.len(), 1);
}

#[test]
fn try_snapshot_returns_some() {
    let buf = LogBuffer::new(4);
    buf.push(LogRecord::now("INFO", "t", "a"));
    let snap = buf.try_snapshot().unwrap();
    assert_eq!(snap.len(), 1);
}

#[test]
fn try_render_trace_returns_some() {
    let buf = LogBuffer::new(4);
    buf.push(LogRecord::now("WARN", "m", "msg"));
    let trace = buf.try_render_trace(10).unwrap();
    assert!(trace.contains("WARN m: msg"));
}

#[test]
fn try_operations_on_contended_lock_during_panic() {
    let buf = Arc::new(LogBuffer::new(32));

    let held = buf.inner.lock().unwrap();

    let buf_clone = Arc::clone(&buf);
    let handle = thread::spawn(move || {
        assert!(!buf_clone.try_push(LogRecord::now("INFO", "t", "should_fail")));
        assert!(buf_clone.try_snapshot().is_none());
        assert!(buf_clone.try_render_trace(10).is_none());
    });
    handle.join().unwrap();
    drop(held);
}

#[test]
fn len_and_is_empty() {
    let buf = LogBuffer::new(16);
    assert!(buf.is_empty());
    buf.push(LogRecord::now("INFO", "t", "x"));
    assert!(!buf.is_empty());
    assert_eq!(buf.len(), 1);

    for _ in 0..20 {
        buf.push(LogRecord::now("DEBUG", "t", "fill"));
    }
    assert_eq!(buf.len(), 16);
}

#[test]
fn concurrent_push_mixed_levels_overflow_consistent() {
    let cap: usize = 64;
    let buf = Arc::new(LogBuffer::new(cap));
    let levels = ["ERROR", "WARN", "INFO", "DEBUG"];
    let n_threads = 4;
    let per_thread = 200;

    let threads: Vec<_> = (0..n_threads)
        .map(|t| {
            let buf = Arc::clone(&buf);
            thread::spawn(move || {
                for i in 0..per_thread {
                    let level = levels[(t + i) % levels.len()];
                    buf.push(LogRecord::now(level, "multi", format!("t{t}_i{i}")));
                }
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }

    assert_eq!(buf.len(), cap);
    let snap = buf.snapshot();
    assert_eq!(snap.len(), cap);
    // Verify all records have valid fields
    for r in &snap {
        assert!(levels.contains(&r.level.as_str()));
        assert_eq!(r.target, "multi");
        assert!(!r.message.is_empty());
        assert!(r.timestamp_secs > 0);
    }
}

// =========================================================================
// NamLogger unit tests — fields accessible from within this module
// =========================================================================

fn new_nam_logger(standalone_mode: bool, level_filter: LevelFilter) -> NamLogger {
    NamLogger {
        buffer: LogBuffer::new(64),
        sinks: Mutex::new(Vec::new()),
        emit_stderr: Mutex::new(standalone_mode),
        max_level: Mutex::new(level_filter),
    }
}

// Helper to build a log::Record for testing.
// The caller passes a `format_args!()` invocation to keep the
// Arguments value alive on the caller's stack frame.
fn build_record<'a>(
    level: log::Level,
    target: &'a str,
    args: std::fmt::Arguments<'a>,
) -> log::Record<'a> {
    log::Record::builder()
        .level(level)
        .target(target)
        .args(args)
        .build()
}

#[test]
fn nam_logger_enabled_respects_max_level() {
    let logger = new_nam_logger(false, LevelFilter::Warn);

    let info_meta = log::Metadata::builder()
        .level(log::Level::Info)
        .target("test")
        .build();
    let warn_meta = log::Metadata::builder()
        .level(log::Level::Warn)
        .target("test")
        .build();

    assert!(!logger.enabled(&info_meta));
    assert!(logger.enabled(&warn_meta));
}

#[test]
fn nam_logger_set_max_level_updates_filter() {
    let logger = new_nam_logger(false, LevelFilter::Error);

    logger.set_max_level(LevelFilter::Info);

    let info_meta = log::Metadata::builder()
        .level(log::Level::Info)
        .target("test")
        .build();
    assert!(logger.enabled(&info_meta));
}

#[test]
fn nam_logger_log_writes_to_buffer() {
    let logger = new_nam_logger(false, LevelFilter::Trace);

    let record = build_record(log::Level::Warn, "my_module", format_args!("test message"));
    logger.log(&record);

    let snap = logger.buffer.snapshot();
    assert_eq!(snap.len(), 1);
    assert_eq!(snap[0].level, "WARN");
    assert_eq!(snap[0].target, "my_module");
    assert_eq!(snap[0].message, "test message");
}

#[test]
fn nam_logger_log_below_max_level_skipped() {
    let logger = new_nam_logger(false, LevelFilter::Error);

    let record = build_record(log::Level::Warn, "test", format_args!("should be skipped"));
    logger.log(&record);

    assert_eq!(logger.buffer.len(), 0);
}

#[test]
fn nam_logger_plugin_mode_no_stderr() {
    let logger = new_nam_logger(false, LevelFilter::Info);
    let record = build_record(log::Level::Info, "test", format_args!("should not print"));

    // In plugin mode, the log should not panic and should write to buffer.
    logger.log(&record);
    assert_eq!(logger.buffer.len(), 1);
}

#[test]
fn sink_registration_and_dispatch() {
    let logger = new_nam_logger(false, LevelFilter::Info);
    let call_count = Arc::new(AtomicUsize::new(0));
    let last_severity = Arc::new(Mutex::new(String::new()));
    let last_msg = Arc::new(Mutex::new(String::new()));

    {
        let cc = Arc::clone(&call_count);
        let ls = Arc::clone(&last_severity);
        let lm = Arc::clone(&last_msg);

        let sink: Arc<HostLogFn> = Arc::new(move |severity, msg| {
            cc.fetch_add(1, Ordering::SeqCst);
            *ls.lock().unwrap() = severity.to_string();
            *lm.lock().unwrap() = msg.to_string();
        });

        logger.register_sink(&sink);

        let record = build_record(log::Level::Error, "err_mod", format_args!("fatal error"));
        logger.log(&record);

        // sink and its captured Arcs drop here
    }

    assert_eq!(call_count.load(Ordering::SeqCst), 1);
    assert_eq!(*last_severity.lock().unwrap(), "ERROR");
    assert_eq!(*last_msg.lock().unwrap(), "fatal error");
    assert_eq!(logger.buffer.len(), 1);
}

#[test]
fn sink_auto_purge_on_drop() {
    let logger = new_nam_logger(false, LevelFilter::Info);
    let call_count = Arc::new(AtomicUsize::new(0));

    {
        let cc = Arc::clone(&call_count);
        let sink: Arc<HostLogFn> = Arc::new(move |_, _| {
            cc.fetch_add(1, Ordering::SeqCst);
        });
        logger.register_sink(&sink);

        let record = build_record(log::Level::Info, "t", format_args!("first"));
        logger.log(&record);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    // After drop, Weak should be dead but still in vec
    {
        let sinks = logger.sinks.lock().unwrap();
        assert_eq!(sinks.len(), 1);
        assert!(sinks[0].upgrade().is_none());
    }

    // Next log dispatch purges the dead Weak
    let record = build_record(log::Level::Info, "t", format_args!("second"));
    logger.log(&record);

    let sinks = logger.sinks.lock().unwrap();
    assert_eq!(sinks.len(), 0);
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

#[test]
fn multiple_sinks_all_receive_dispatch() {
    let logger = new_nam_logger(false, LevelFilter::Info);
    let count_a = Arc::new(AtomicUsize::new(0));
    let count_b = Arc::new(AtomicUsize::new(0));
    let count_c = Arc::new(AtomicUsize::new(0));

    let ca = Arc::clone(&count_a);
    let cb = Arc::clone(&count_b);
    let cc = Arc::clone(&count_c);

    let sink_a: Arc<HostLogFn> = Arc::new(move |_, _| {
        ca.fetch_add(1, Ordering::SeqCst);
    });
    let sink_b: Arc<HostLogFn> = Arc::new(move |_, _| {
        cb.fetch_add(1, Ordering::SeqCst);
    });
    let sink_c: Arc<HostLogFn> = Arc::new(move |_, _| {
        cc.fetch_add(1, Ordering::SeqCst);
    });

    logger.register_sink(&sink_a);
    logger.register_sink(&sink_b);
    logger.register_sink(&sink_c);

    let record = build_record(log::Level::Warn, "multi", format_args!("broadcast"));
    logger.log(&record);

    assert_eq!(count_a.load(Ordering::SeqCst), 1);
    assert_eq!(count_b.load(Ordering::SeqCst), 1);
    assert_eq!(count_c.load(Ordering::SeqCst), 1);
    assert_eq!(logger.buffer.len(), 1);
}

#[test]
fn sink_mixed_live_and_dead_purge_correctly() {
    let logger = new_nam_logger(false, LevelFilter::Info);
    let live_count = Arc::new(AtomicUsize::new(0));

    let lc = Arc::clone(&live_count);
    let live_sink: Arc<HostLogFn> = Arc::new(move |_, _| {
        lc.fetch_add(1, Ordering::SeqCst);
    });

    // Create a dead sink (dropped immediately)
    {
        let dead_sink: Arc<HostLogFn> = Arc::new(|_, _| {});
        logger.register_sink(&dead_sink);
    }

    logger.register_sink(&live_sink);

    let record = build_record(log::Level::Error, "t", format_args!("mix"));
    logger.log(&record);

    assert_eq!(live_count.load(Ordering::SeqCst), 1);

    // Dead sink purged, live sink remains
    let sinks = logger.sinks.lock().unwrap();
    assert_eq!(sinks.len(), 1);
    assert!(sinks[0].upgrade().is_some());
}

// =========================================================================
// Concurrent init idempotency test
// =========================================================================

#[test]
fn init_plugin_idempotent_from_multiple_threads() {
    // Ensure the global logger is initialized first.
    let first = NamLogger::init(LoggerConfig {
        level_filter: LevelFilter::Info,
        emit_stderr: false,
    });
    assert!(first.is_ok());

    let threads: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let result = NamLogger::init(LoggerConfig {
                    level_filter: LevelFilter::Debug,
                    emit_stderr: false,
                });
                assert!(result.is_ok());
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }

    assert!(NamLogger::global().is_some());
    assert!(NamLogger::log_buffer().is_some());
}

#[test]
fn init_standalone_returns_ok() {
    let _ = NamLogger::init(LoggerConfig {
        level_filter: LevelFilter::Warn,
        emit_stderr: true,
    });
    assert!(NamLogger::global().is_some());
}

// =========================================================================
// global() / log_buffer() accessor tests
// =========================================================================

#[test]
fn global_returns_some_after_init() {
    let result = NamLogger::init(LoggerConfig {
        level_filter: LevelFilter::Warn,
        emit_stderr: false,
    });
    assert!(result.is_ok());
    assert!(NamLogger::global().is_some());
    assert!(NamLogger::log_buffer().is_some());
}

#[test]
fn log_buffer_returns_some_after_init() {
    let _ = NamLogger::init(LoggerConfig {
        level_filter: LevelFilter::Warn,
        emit_stderr: false,
    });
    let buf = NamLogger::log_buffer();
    assert!(buf.is_some());
    assert_eq!(buf.unwrap().capacity(), DEFAULT_CAPACITY);
}

#[test]
fn global_and_log_buffer_reference_roundtrip() {
    let _ = NamLogger::init(LoggerConfig {
        level_filter: LevelFilter::Info,
        emit_stderr: false,
    });
    let global = NamLogger::global().unwrap();
    let buf = NamLogger::log_buffer().unwrap();

    let msg = "rt_msg";
    let args = format_args!("{}", msg);
    let record = build_record(log::Level::Error, "roundtrip", args);
    global.log(&record);

    let snap = buf.snapshot();
    assert!(snap.iter().any(|r| r.message == msg));
}

// =========================================================================
// Concurrent sink dispatch (multi-threaded logs through NamLogger)
// =========================================================================

#[test]
fn concurrent_log_records_all_reach_buffer_and_sink() {
    let logger = Arc::new(new_nam_logger(false, LevelFilter::Trace));
    let sink_count = Arc::new(AtomicUsize::new(0));

    let sc = Arc::clone(&sink_count);
    let sink: Arc<HostLogFn> = Arc::new(move |_, _| {
        sc.fetch_add(1, Ordering::SeqCst);
    });
    logger.register_sink(&sink);

    let n_threads = 4;
    let per_thread = 50;
    let threads: Vec<_> = (0..n_threads)
        .map(|t| {
            let logger = Arc::clone(&logger);
            thread::spawn(move || {
                for i in 0..per_thread {
                    let msg = format!("t{t}_i{i}");
                    let args = format_args!("{}", msg);
                    let record = build_record(log::Level::Info, "concurrent", args);
                    logger.log(&record);
                }
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }

    let expected = n_threads * per_thread;
    assert!(logger.buffer.len() <= 64);
    assert_eq!(logger.buffer.len(), 64usize.min(expected));
    assert_eq!(sink_count.load(Ordering::SeqCst), expected);
}

// =========================================================================
// flush is a no-op but should not panic
// =========================================================================

#[test]
fn flush_does_nothing() {
    let logger = new_nam_logger(false, LevelFilter::Info);
    logger.flush();
}

#[test]
fn flush_idempotent() {
    let logger = new_nam_logger(false, LevelFilter::Trace);
    let record = build_record(log::Level::Info, "t", format_args!("pre"));
    logger.log(&record);
    logger.flush();
    logger.flush();
    assert_eq!(logger.buffer.len(), 1);
}

// =========================================================================
// Wall-clock format and LogRecord timestamp integrity tests
// =========================================================================

#[test]
fn test_format_wall_clock_time_produces_valid_format() {
    let time_str = format_wall_clock_time();
    assert_eq!(
        time_str.len(),
        12,
        "format should be HH:MM:SS.mmm (12 chars)"
    );
    assert_eq!(&time_str[2..3], ":", "char 2 should be ':'");
    assert_eq!(&time_str[5..6], ":", "char 5 should be ':'");
    assert_eq!(&time_str[8..9], ".", "char 8 should be '.'");
    let parts: Vec<&str> = time_str.split([':', '.']).collect();
    assert_eq!(parts.len(), 4);
    for (i, part) in parts.iter().enumerate() {
        let val: u32 = part
            .parse()
            .unwrap_or_else(|_| panic!("part {i} '{part}' not numeric"));
        if i == 0 {
            assert!(val < 24, "hours must be < 24, got {val}");
        } else if i < 3 {
            assert!(val < 60, "minutes/seconds must be < 60, got {val}");
        } else {
            assert!(val < 1000, "milliseconds must be < 1000, got {val}");
        }
    }
}

#[test]
fn test_log_record_timestamp_is_u64_unix_epoch() {
    let record = LogRecord::now("INFO", "test", "message");
    assert!(
        record.timestamp_secs > 1_700_000_000,
        "timestamp_secs should be a recent UNIX epoch value (got {})",
        record.timestamp_secs
    );
}

#[test]
fn test_log_record_timestamp_is_monotonic() {
    let r1 = LogRecord::now("INFO", "a", "first");
    let r2 = LogRecord::now("INFO", "b", "second");
    assert!(
        r2.timestamp_secs >= r1.timestamp_secs,
        "LogRecord timestamps should be monotonic"
    );
}
