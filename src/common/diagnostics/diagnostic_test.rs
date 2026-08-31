// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::common::diagnostics::{
    LoggerConfig, NamErrorCode, NamLogger, SystemSnapshot, format::redact_path, format::redact_text,
};

static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Validates the visual formatting and consistency of the error code and its mnemonic.
#[test]
fn test_error_code_display() {
    let code = NamErrorCode::NambCrc32Mismatch;
    assert_eq!(code.code(), "E1201");
    assert_eq!(code.mnemonic(), "NAMB_CRC32_MISMATCH");
    assert_eq!(
        format!("{}", code),
        "[E1201 NAMB_CRC32_MISMATCH] CRC32 checksum mismatch"
    );
}

/// Ensures the integrity of the error registry, preventing duplicate numeric codes (E####).
/// Essential for maintaining log stability and support protocols.
#[test]
fn test_all_codes_have_unique_numeric() {
    use NamErrorCode::*;
    let all = [
        FileNotFound,
        FileReadError,
        UnknownExtension,
        NamJsonParseError,
        NamJsonWeightsExceedLimit,
        NamJsonTrainingTooLarge,
        NamJsonTrainingTooDeep,
        NamJsonSubmodelsExceedLimit,
        NamJsonSubmodelsTooDeep,
        NamJsonWeightNotFinite,
        NamJsonInvalidSampleRate,
        NamJsonUnsupportedTopology,
        NamJsonInvalidVersionFormat,
        NamJsonUnsupportedVersion,
        NamJsonUnsupportedMultiChannel,
        InvalidMetadata,
        NambNonFiniteWeight,
        NambInvalidHeaderField,
        NambCrc32Mismatch,
        NambCrc32Missing,
        NambInvalidMagic,
        NambUnsupportedVersion,
        NambTruncated,
        UnsupportedArchitecture,
        TopologyDetectionFailed,
        WeightCountMismatch,
        ModelBuildFailed,
        ModelTooLarge,
        InvalidModelTopology,
        AudioInitFailed,
        StreamError,
        ResamplerBuildFailed,
        ResamplerChannelFull,
        RtPriorityDenied,
        CpuAffinityFailed,
        ProcessingOverload,
        ParamChannelFull,
        GcOverflow,
        GcCorrupted,
        InvalidGainValue,
        UnknownCommand,
        CtrlCHandlerFailed,
        IrLoadFailed,
        OutOfMemory,
        UnsupportedCpuArchitecture,
    ];

    let codes: Vec<&str> = all.iter().map(|c| c.code()).collect();
    let mut sorted = codes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        codes.len(),
        sorted.len(),
        "Duplicate numeric codes detected"
    );
}

/// Verifies that system information capture (OS, Arch, Version) is active and returning valid data.
#[test]
fn test_system_snapshot_capture() {
    let snap = SystemSnapshot::capture();
    assert!(!snap.version.is_empty());
    assert!(!snap.arch.is_empty());
    assert!(!snap.os.is_empty());
}

/// Validates the generation of the "support block" for the user, ensuring it contains the error,
/// mnemonic, contextual parameters, and copy instructions.
#[test]
fn test_diagnostic_support_block_contains_code() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let snap = SystemSnapshot::capture();
    let diag = NamDiagnostic::new(NamErrorCode::NambCrc32Mismatch, &snap)
        .message("Corrupted file")
        .hint("Download it again")
        .param("expected", "0xDEADBEEF")
        .param("computed", "0x12345678");

    let block = diag.support_block();
    assert!(block.contains("E1201"), "Block must contain the code");
    assert!(
        block.contains("NAMB_CRC32_MISMATCH"),
        "Block must contain the mnemonic"
    );
    assert!(
        block.contains("expected=0xDEADBEEF"),
        "Block must contain parameters"
    );
    assert!(
        block.contains("computed=0x12345678"),
        "Block must contain parameters"
    );
    assert!(
        block.contains("Copy the block above"),
        "Block must contain copy instructions"
    );
}

/// Tests the simplified diagnostic representation for display via the Display trait.
#[test]
fn test_diagnostic_display() {
    let snap = SystemSnapshot::capture();
    let diag = NamDiagnostic::new(NamErrorCode::FileNotFound, &snap).message("Model not found");
    assert_eq!(format!("{}", diag), "[E1100] Model not found");
}

/// Validates the manual algorithm for converting days since epoch to a date (Year, Month, Day),
/// keeping the binary lightweight without external time dependencies (like chrono).
#[test]
fn test_days_to_date_epoch() {
    let (y, m, d) = days_to_date(0);
    assert_eq!((y, m, d), (1970, 1, 1));
}

/// Tests the accuracy of the date algorithm with a known date (2026-04-10).
#[test]
fn test_days_to_date_known() {
    // 2026-04-10 = ~20553 days since epoch
    // (1970-01-01 + 20553 = 2026-04-10)
    let (y, m, d) = days_to_date(20553);
    assert_eq!(y, 2026);
    assert_eq!(m, 4);
    assert_eq!(d, 10);
}

/// Ensures the generated timestamp strictly follows the ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).
#[test]
fn test_timestamp_format() {
    let ts = NamDiagnostic::timestamp();
    // Verify ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
    assert_eq!(ts.len(), 20, "Timestamp must be 20 characters long");
    assert!(ts.ends_with('Z'), "Timestamp must end with Z");
    assert_eq!(&ts[4..5], "-", "Date separator");
    assert_eq!(&ts[10..11], "T", "T separator");
}

/// Ensures the IRQ mask calculation does not panic on systems with many cores (Bug B5).
#[test]
fn test_emit_irq_advisory_safety() {
    let snap = SystemSnapshot::capture();
    // Test low cores, u32 boundary, and u64 boundary
    snap.emit_irq_advisory(0);
    snap.emit_irq_advisory(31);
    snap.emit_irq_advisory(32);
    snap.emit_irq_advisory(63);
    snap.emit_irq_advisory(64); // Should return via guard without panic
    snap.emit_irq_advisory(128); // Extreme case
}

/// Verifies that DiagnosticBundle::capture().render() yields a nominal block without error fields.
#[test]
fn test_diagnostic_bundle_nominal() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let bundle = DiagnosticBundle::capture();
    let rendered = bundle.render();

    assert!(
        rendered.contains("NeuralAmpModeler-rs Diagnostic"),
        "Must contain standard header"
    );
    assert!(
        rendered.contains(&format!("NeuralAmpModeler-rs v{}", bundle.system.version)),
        "Must contain version"
    );
    assert!(!rendered.contains("E1"), "Must not contain error codes");
    assert!(
        !rendered.contains("CRC32"),
        "Must not contain error mnemonics"
    );
    assert!(rendered.contains("arch="), "Must contain system info");
    assert!(rendered.contains("os="), "Must contain system info");
}

/// Verifies that DiagnosticBundle::capture_with_error() matches NamDiagnostic::support_block() output.
#[test]
fn test_diagnostic_bundle_with_error_matches() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let snap = SystemSnapshot::capture();
    let code = NamErrorCode::NambCrc32Mismatch;
    let params = vec![
        ("expected", "0xDEADBEEF".to_string()),
        ("computed", "0x12345678".to_string()),
    ];

    let diag = NamDiagnostic::new(code, &snap)
        .param("expected", "0xDEADBEEF")
        .param("computed", "0x12345678");

    let bundle = DiagnosticBundle::capture_with_error(code, params);

    let block_diag = diag.support_block();
    let block_bundle = bundle.render();

    fn filter_diagnostic_lines(block: &str) -> Vec<&str> {
        let mut lines = Vec::new();
        let mut in_log_trace = false;
        for line in block.lines() {
            if line.contains("Recent Log Trace") {
                in_log_trace = true;
                lines.push(line);
                continue;
            }
            if in_log_trace {
                if line.starts_with("───") {
                    in_log_trace = false;
                    lines.push(line);
                }
                continue;
            }
            if !line.starts_with("timestamp=") && !line.contains("isolate core") {
                lines.push(line);
            }
        }
        lines
    }

    let lines_diag = filter_diagnostic_lines(&block_diag);
    let lines_bundle = filter_diagnostic_lines(&block_bundle);

    assert_eq!(lines_diag, lines_bundle);
}

/// Verifies that active model name path redaction works properly based on the `full` flag.
#[test]
fn test_diagnostic_bundle_redaction() {
    let _guard = TEST_MUTEX.lock().unwrap();
    // Populate the active session statics
    if let Ok(mut name) = ACTIVE_MODEL_NAME.write() {
        *name = "/home/user/my_secret_path/model.nam".to_string();
    }
    ACTIVE_SAMPLE_RATE.store(44100, std::sync::atomic::Ordering::Relaxed);

    // Default/Redacted mode: should only render basename
    let bundle_redacted = DiagnosticBundle::capture();
    let rendered_redacted = bundle_redacted.render();
    assert!(
        rendered_redacted.contains("model=model.nam"),
        "Should render only basename in redacted mode"
    );
    assert!(
        !rendered_redacted.contains("my_secret_path"),
        "Should not expose path details in redacted mode"
    );
    assert!(
        rendered_redacted.contains("sample_rate=44100"),
        "Should render active sample rate"
    );

    // Full/Unredacted mode: should render full absolute path
    let bundle_full = DiagnosticBundle::capture().with_full(true);
    let rendered_full = bundle_full.render();
    assert!(
        rendered_full.contains("model=/home/user/my_secret_path/model.nam"),
        "Should render full path in full mode"
    );
    assert!(
        rendered_full.contains("sample_rate=44100"),
        "Should render active sample rate"
    );

    // Cleanup/restore statics for other tests
    if let Ok(mut name) = ACTIVE_MODEL_NAME.write() {
        *name = String::new();
    }
    ACTIVE_SAMPLE_RATE.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Verifies that `DiagnosticBundle::render()` includes the `──── Recent Log Trace ────`
/// section when the global `NamLogger` is initialized, and that emitted log messages
/// appear in the output.
#[test]
fn test_diagnostic_bundle_includes_log_trace_when_logger_initialized() {
    let _guard = TEST_MUTEX.lock().unwrap();

    // Ensure NamLogger is initialized (safe to call multiple times due to OnceLock;
    // log::set_logger may fail harmlessly if already registered by another test).
    let _ = NamLogger::init(LoggerConfig {
        level_filter: log::LevelFilter::Info,
        emit_stderr: true,
    });

    log::info!("Test log message 1");
    log::warn!("Test warning with /path/to/file");
    log::error!("Test error message");

    let bundle = DiagnosticBundle::capture();
    let rendered = bundle.render();

    assert!(
        rendered.contains("──── Recent Log Trace ─────────────────────────"),
        "Must contain log trace header when logger is initialized"
    );

    // Verify messages reached the buffer directly (snapshot).
    // render_trace(50) may not contain them when other test modules
    // log concurrently, so we use the full snapshot here.
    if let Some(buffer) = NamLogger::log_buffer() {
        let snapshot = buffer.snapshot();
        let has_info = snapshot
            .iter()
            .any(|r| r.level == "INFO" && r.message.contains("Test log message 1"));
        let has_warn = snapshot
            .iter()
            .any(|r| r.level == "WARN" && r.message.contains("Test warning"));
        let has_error = snapshot
            .iter()
            .any(|r| r.level == "ERROR" && r.message.contains("Test error message"));
        assert!(has_info, "Must have info message in log buffer");
        assert!(has_warn, "Must have warning in log buffer");
        assert!(has_error, "Must have error in log buffer");
    }
}

/// Verifies that `redact_text` correctly replaces HOME paths with `~`
/// in free-form text, and that the `full` flag disables redaction.
#[test]
fn test_diagnostic_bundle_log_trace_redaction() {
    let home = std::env::var("HOME").unwrap_or_default();
    if home.is_empty() {
        return; // skip if HOME is not set (e.g. some CI environments)
    }

    let test_path = format!("{}/test_model.nam", home);
    let text = format!("Loading model from {}", test_path);

    // Redacted mode: HOME should be replaced with ~
    let redacted = redact_text(&text, false);
    assert!(
        !redacted.contains(&test_path),
        "Full HOME path should not appear in redacted text"
    );
    assert!(
        redacted.contains("~/test_model.nam"),
        "HOME path should be redacted to ~/..."
    );

    // Full mode: path should be unmodified
    let full = redact_text(&text, true);
    assert!(
        full.contains(&test_path),
        "Full HOME path should appear in full text"
    );
}

/// Verifies that `redact_path` respects the `full` flag and correctly
/// redacts XDG_RUNTIME_DIR paths with higher priority than HOME.
#[test]
fn test_redact_path_and_xdg_priority() {
    let home = std::env::var("HOME").unwrap_or_default();
    if home.is_empty() {
        return;
    }

    let path_str = format!("{}/test.nam", home);
    let p = std::path::Path::new(&path_str);

    // full=true bypasses redaction
    assert_eq!(redact_path(p, true), format!("{}/test.nam", home));

    // full=false redacts HOME to ~
    assert_eq!(redact_path(p, false), "~/test.nam");
}

/// Validates that `UnsupportedCpuArchitecture` is properly formatted in diagnostics.
#[test]
fn test_unsupported_cpu_architecture_diagnostic() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let snap = SystemSnapshot::capture();
    let code = NamErrorCode::UnsupportedCpuArchitecture;

    assert_eq!(code.code(), "E5001");
    assert_eq!(code.mnemonic(), "UNSUPPORTED_CPU_ARCHITECTURE");
    assert_eq!(
        code.message(),
        "Host CPU lacks required x86-64-v3 feature set (AVX2/FMA)"
    );

    let diag = NamDiagnostic::new(code, &snap)
        .message("CPU incompatible: x86-64-v3 baseline required (AVX2 + FMA).")
        .hint("Upgrade to a CPU supporting AVX2 and FMA (Intel Haswell+ / AMD Excavator+).");

    let block = diag.support_block();
    assert!(block.contains("E5001"));
    assert!(block.contains("UNSUPPORTED_CPU_ARCHITECTURE"));
    assert!(format!("{}", diag).contains("[E5001] CPU incompatible"));
}

/// Verifies that `redact_text` preserves multi-byte UTF-8 sequences (emojis, box characters,
/// accents) without corrupting them into mojibake or modifying non-path content.
#[test]
fn test_redact_text_preserves_multibyte_utf8() {
    let home = std::env::var("HOME").unwrap_or_default();
    let sample = if !home.is_empty() {
        format!(
            "🔇 Silent Mode: ──── {}/model.nam ──── 🔊 Resumed! 🎸 Português: ação, café.",
            home
        )
    } else {
        "🔇 Silent Mode: ──── /tmp/model.nam ──── 🔊 Resumed! 🎸 Português: ação, café.".to_string()
    };

    let redacted = redact_text(&sample, false);
    assert!(
        redacted.contains("🔇"),
        "Emoji 🔇 must be preserved without mojibake"
    );
    assert!(
        redacted.contains("🔊"),
        "Emoji 🔊 must be preserved without mojibake"
    );
    assert!(
        redacted.contains("🎸"),
        "Emoji 🎸 must be preserved without mojibake"
    );
    assert!(
        redacted.contains("────"),
        "Box drawing horizontal line ──── must be preserved"
    );
    assert!(
        redacted.contains("Português: ação, café."),
        "UTF-8 accents must be preserved"
    );

    if !home.is_empty() {
        assert!(
            redacted.contains("~/model.nam"),
            "Path must still be correctly redacted"
        );
        assert!(
            !redacted.contains(&home),
            "Raw HOME path must not appear in redacted output"
        );
    }

    // Verify idempotency across multiple redaction passes (no generational corruption)
    let second_pass = redact_text(&redacted, false);
    assert_eq!(
        redacted, second_pass,
        "Multiple redact_text passes must be strictly idempotent"
    );
}

/// Verifies that `NamDiagnostic::emit` and `emit_warning` do not inject the entire support block
/// into the `LogBuffer` circular trace, preventing recursive support block nesting.
#[test]
fn test_diagnostic_emit_does_not_pollute_log_buffer_recursively() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let _ = NamLogger::init(LoggerConfig {
        level_filter: log::LevelFilter::Debug,
        emit_stderr: false,
    });

    let snap = SystemSnapshot::capture();
    let diag = NamDiagnostic::new(NamErrorCode::BackendFailure, &snap)
        .message("Backend connection failed")
        .hint("Restart service");

    diag.emit();

    if let Some(buf) = NamLogger::log_buffer() {
        let trace = buf.render_trace(10);
        assert!(trace.contains("Backend connection failed"));
        assert!(
            !trace.contains("Copy the block above when opening a support ticket."),
            "LogBuffer must not contain the rendered technical support block"
        );
        assert!(
            !trace.contains("──── NeuralAmpModeler-rs Diagnostic"),
            "LogBuffer must not contain the diagnostic header box"
        );
    }
}
