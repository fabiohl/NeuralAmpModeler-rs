// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Integration tests for DiagnosticBundle and RuntimeSnapshot.

use neural_amp_modeler_rs::common::diagnostics::{
    AudioInfo, AudioMetadata, DiagnosticBundle, HasRuntimeSnapshot, ModelInfo, RtInfo,
    RuntimeSnapshot, TelemetrySnapshot,
};
use neural_amp_modeler_rs::common::spsc::RtStatusFlags;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
static HOOK_ID: AtomicU64 = AtomicU64::new(0);

/// Generate a unique `&'static str` component name for panic hook tests
/// to prevent cross-contamination from concurrent test thread panics.
fn unique_component(base: &str) -> &'static str {
    let id = HOOK_ID.fetch_add(1, Ordering::Relaxed);
    Box::leak(format!("{}-{}", base, id).into_boxed_str())
}

struct MockSnapshotProvider {
    model: Option<ModelInfo>,
    audio: AudioInfo,
    rt: RtInfo,
    telemetry: TelemetrySnapshot,
    flags: u64,
}

impl HasRuntimeSnapshot for MockSnapshotProvider {
    fn model_info(&self) -> Option<ModelInfo> {
        self.model.clone()
    }

    fn audio_info(&self, consumer: &AudioMetadata) -> AudioInfo {
        AudioInfo {
            sample_rate: self.audio.sample_rate,
            buffer_size: self.audio.buffer_size,
            channel_count: consumer.channel_count,
            host_name: consumer.host_name.clone(),
        }
    }

    fn rt_info(&self) -> RtInfo {
        self.rt.clone()
    }

    fn telemetry_snapshot(&self) -> TelemetrySnapshot {
        self.telemetry.clone()
    }

    fn flags_seen(&self) -> u64 {
        self.flags
    }
}

#[test]
fn test_diagnostic_bundle_with_mock_provider() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let provider = MockSnapshotProvider {
        model: Some(ModelInfo {
            arch_label: "WaveNet".to_string(),
            topology: "Standard".to_string(),
            channels: 16,
            receptive_field: 2048,
            model_sample_rate: 48000,
            weights_layout: "Interleaved4WaveNet".to_string(),
            path_basename: "test_model.nam".to_string(),
        }),
        audio: AudioInfo {
            sample_rate: 48000,
            buffer_size: 256,
            channel_count: 2,
            host_name: "CLAP".to_string(),
        },
        rt: RtInfo {
            thread_priority: 90,
            scheduler: "FIFO".to_string(),
            cpu_pinned: Some(3),
            huge_pages_active: true,
            huge_page_mode: "hugetlb".to_string(),
        },
        telemetry: TelemetrySnapshot {
            p50_us: 120,
            p99_us: 250,
            p999_us: 400,
            max_us: 850,
            total_blocks: 10000,
            xruns: 2,
            drains: 5,
            input_buffer_miss: 0,
            output_buffer_miss: 0,
        },
        flags: 0x1a,
    };

    let consumer = AudioMetadata {
        channel_count: 2,
        host_name: "CLAP".into(),
    };
    let bundle = DiagnosticBundle::capture_with_runtime(&provider, &consumer);
    let rendered = bundle.render();

    // Verify model info
    assert!(rendered.contains("model.arch=WaveNet"));
    assert!(rendered.contains("model.channels=16"));
    assert!(rendered.contains("model.receptive_field=2048"));
    assert!(rendered.contains("model.weights_layout=Interleaved4WaveNet"));
    assert!(rendered.contains("model.sample_rate=48000"));
    assert!(rendered.contains("model.path_basename=test_model.nam"));

    // Verify audio info
    assert!(rendered.contains("audio.sr=48000"));
    assert!(rendered.contains("audio.buffer_size=256"));
    assert!(rendered.contains("audio.channel_count=2"));
    assert!(rendered.contains("audio.host_name=CLAP"));

    // Verify RT info
    assert!(rendered.contains("rt.prio=90"));
    assert!(rendered.contains("rt.scheduler=FIFO"));
    assert!(rendered.contains("rt.cpu_pinned=3"));
    assert!(rendered.contains("rt.huge_pages_active=true"));

    // Verify Telemetry info
    assert!(rendered.contains("telemetry.p50_us=120"));
    assert!(rendered.contains("telemetry.p99_us=250"));
    assert!(rendered.contains("telemetry.p999_us=400"));
    assert!(rendered.contains("telemetry.max_us=850"));
    assert!(rendered.contains("telemetry.total_blocks=10000"));
    assert!(rendered.contains("telemetry.xruns=2"));
    assert!(rendered.contains("telemetry.drains=5"));

    // Verify flags seen in hex format
    assert!(rendered.contains("flags_seen=0x1a"));
}

#[test]
fn test_rt_status_flags_provider_defaults() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let rt_status = RtStatusFlags::new();
    let consumer = AudioMetadata {
        channel_count: 4,
        host_name: "Standalone".into(),
    };
    let snapshot = RuntimeSnapshot::capture(&rt_status, &consumer);

    assert_eq!(snapshot.audio.sample_rate, 0);
    assert_eq!(snapshot.audio.buffer_size, 0);
    assert_eq!(snapshot.audio.host_name, "Standalone");
    assert_eq!(snapshot.rt.thread_priority, -1);
    assert_eq!(snapshot.rt.scheduler, "UNKNOWN");
    assert_eq!(snapshot.rt.cpu_pinned, None);
    assert!(!snapshot.rt.huge_pages_active);
    assert_eq!(snapshot.telemetry.xruns, 0);
    assert_eq!(snapshot.telemetry.drains, 0);
    assert_eq!(snapshot.flags_seen, 0);
}

#[test]
fn test_rt_status_flags_provider_populated() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let rt_status = RtStatusFlags::new();
    rt_status.active_rate.store(44100, Ordering::Relaxed);
    rt_status.last_n_samples.store(512, Ordering::Relaxed);
    rt_status.confirmed_priority.store(85, Ordering::Relaxed);
    rt_status
        .rt_policy
        .store(libc::SCHED_FIFO, Ordering::Relaxed);
    rt_status.rt_cpu.store(2, Ordering::Relaxed);
    rt_status.xruns.store(3, Ordering::Relaxed);
    rt_status.drains.store(7, Ordering::Relaxed);
    rt_status.flags_seen.store(0x4b, Ordering::Relaxed);

    let consumer = AudioMetadata {
        channel_count: 2,
        host_name: "Standalone".into(),
    };
    let snapshot = RuntimeSnapshot::capture(&rt_status, &consumer);

    assert_eq!(snapshot.audio.sample_rate, 44100);
    assert_eq!(snapshot.audio.buffer_size, 512);
    assert_eq!(snapshot.rt.thread_priority, 85);
    assert_eq!(snapshot.rt.scheduler, "FIFO");
    assert_eq!(snapshot.rt.cpu_pinned, Some(2));
    assert_eq!(snapshot.telemetry.xruns, 3);
    assert_eq!(snapshot.telemetry.drains, 7);
    assert_eq!(snapshot.flags_seen, 0x4b);
}

#[test]
fn test_panic_hook_behavior() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use std::fs;
    use std::path::PathBuf;

    let home = match std::env::var_os("HOME") {
        Some(h) => PathBuf::from(h),
        None => return,
    };
    let cache_dir = home.join(".cache/nam-rs");
    let _ = fs::create_dir_all(&cache_dir);

    // Part 1: Test persistence when shutdown is NOT in progress.
    let component_name = unique_component("test-panic-persistence");

    // Clear old files
    if let Ok(entries) = fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_target = path.is_file()
                && path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .map(|name| name.contains(component_name))
                    .unwrap_or(false);
            if is_target {
                let _ = fs::remove_file(path);
            }
        }
    }

    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    neural_amp_modeler_rs::common::panic_hook::install_panic_hook(component_name);

    let result = std::panic::catch_unwind(|| {
        panic!("Controlled testing panic message");
    });
    assert!(result.is_err());

    std::panic::set_hook(original_hook);

    // Verify report
    let mut found_report = None;
    if let Ok(entries) = fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_target = path.is_file()
                && path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .map(|name| name.contains(component_name) && name.ends_with(".txt"))
                    .unwrap_or(false);
            if is_target {
                found_report = Some(path);
                break;
            }
        }
    }

    let report_path = found_report.expect("Crash report file should be created");
    let content = fs::read_to_string(&report_path).expect("Should read report content");

    assert!(content.contains("NeuralAmpModeler-rs CRASH REPORT"));
    assert!(content.contains(&format!("Component: {}", component_name)));
    assert!(content.contains("Location:"));
    assert!(content.contains("Message: Controlled testing panic message"));
    assert!(content.contains("──── Runtime State ─────────────────────────────"));
    assert!(content.contains("arch="));
    assert!(content.contains("os="));

    let _ = fs::remove_file(report_path);

    // Part 2: Test bypass when shutdown IS in progress.
    let component_bypass = unique_component("test-panic-bypass");

    // Clear old files
    if let Ok(entries) = fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_target = path.is_file()
                && path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .map(|name| name.contains(component_bypass))
                    .unwrap_or(false);
            if is_target {
                let _ = fs::remove_file(path);
            }
        }
    }

    neural_amp_modeler_rs::common::panic_hook::set_shutdown_in_progress();
    assert!(neural_amp_modeler_rs::common::panic_hook::is_shutdown_in_progress());

    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    neural_amp_modeler_rs::common::panic_hook::install_panic_hook(component_bypass);

    let result = std::panic::catch_unwind(|| {
        panic!("Controlled panic during shutdown");
    });
    assert!(result.is_err());

    std::panic::set_hook(original_hook);

    // Verify NO report was created
    let mut found_bypass_report = false;
    if let Ok(entries) = fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_target = path.is_file()
                && path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .map(|name| name.contains(component_bypass))
                    .unwrap_or(false);
            if is_target {
                found_bypass_report = true;
                let _ = fs::remove_file(path);
            }
        }
    }

    assert!(
        !found_bypass_report,
        "No report should be written if shutdown is in progress"
    );
}

#[test]
fn test_diagnostic_bundle_path_redaction() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    // Save current env vars to restore later if modified, or just use them.
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/mockuser".to_string());
    let xdg = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/run/user/1000".to_string());

    // We override standard env vars if they are empty for the test
    if std::env::var("HOME").is_err() {
        unsafe {
            std::env::set_var("HOME", &home);
        }
    }
    if std::env::var("XDG_RUNTIME_DIR").is_err() {
        unsafe {
            std::env::set_var("XDG_RUNTIME_DIR", &xdg);
        }
    }

    let home_path = format!("{}/nam-rs/models/test_model.nam", home);
    let xdg_path = format!("{}/pipewire-0", xdg);

    // Mock provider with a full path in model.path_basename
    let provider = MockSnapshotProvider {
        model: Some(ModelInfo {
            arch_label: "WaveNet".to_string(),
            topology: "Standard".to_string(),
            channels: 16,
            receptive_field: 2048,
            model_sample_rate: 48000,
            weights_layout: "Interleaved4WaveNet".to_string(),
            path_basename: home_path.clone(),
        }),
        audio: AudioInfo {
            sample_rate: 48000,
            buffer_size: 256,
            channel_count: 2,
            host_name: "CLAP".to_string(),
        },
        rt: RtInfo {
            thread_priority: 90,
            scheduler: "FIFO".to_string(),
            cpu_pinned: Some(3),
            huge_pages_active: true,
            huge_page_mode: "thp".to_string(),
        },
        telemetry: TelemetrySnapshot::default(),
        flags: 0,
    };

    let consumer = AudioMetadata {
        channel_count: 2,
        host_name: "Test".into(),
    };

    // Case A: Default capture (redacted)
    let bundle_default = DiagnosticBundle::capture_with_runtime(&provider, &consumer);
    let rendered_default = bundle_default.render();

    // Verifications for default capture
    // 1. Should NOT contain raw HOME path
    assert!(!rendered_default.contains(&home_path));
    // 2. model.path_basename should be formatted to just the basename
    assert!(rendered_default.contains("model.path_basename=test_model.nam"));
    // 3. model should also print only the basename
    assert!(rendered_default.contains("model=test_model.nam"));

    // Now test parameter redaction. Let's capture with an error containing paths
    let params = vec![
        ("model_path", home_path.clone()),
        ("socket_path", xdg_path.clone()),
    ];
    let bundle_err = DiagnosticBundle::capture_with_error(
        neural_amp_modeler_rs::common::diagnostics::NamErrorCode::ModelBuildFailed,
        params,
    );
    let rendered_err_default = bundle_err.render();

    // Verifications for default error parameters
    assert!(rendered_err_default.contains("model_path=~/nam-rs/models/test_model.nam"));
    assert!(rendered_err_default.contains("socket_path=$XDG_RUNTIME_DIR/pipewire-0"));
    assert!(!rendered_err_default.contains(&home_path));
    assert!(!rendered_err_default.contains(&xdg_path));

    // Case B: Full capture (unredacted / bruto)
    let bundle_full = DiagnosticBundle::capture_with_runtime(&provider, &consumer).with_full(true);
    let rendered_full = bundle_full.render();

    // Verifications for full capture
    // 1. Should contain raw HOME path
    assert!(rendered_full.contains(&home_path));
    // 2. model.path_basename should print full path
    assert!(rendered_full.contains(&format!("model.path_basename={}", home_path)));
    // 3. model should print full path
    assert!(rendered_full.contains(&format!("model={}", home_path)));

    // Full error parameters
    let bundle_err_full = DiagnosticBundle::capture_with_error(
        neural_amp_modeler_rs::common::diagnostics::NamErrorCode::ModelBuildFailed,
        vec![
            ("model_path", home_path.clone()),
            ("socket_path", xdg_path.clone()),
        ],
    )
    .with_full(true);
    let rendered_err_full = bundle_err_full.render();

    assert!(rendered_err_full.contains(&format!("model_path={}", home_path)));
    assert!(rendered_err_full.contains(&format!("socket_path={}", xdg_path)));
}

#[test]
fn test_diagnostic_bundle_capture_nominal() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let bundle = DiagnosticBundle::capture();
    let rendered = bundle.render();

    // Test 1: Verify it is a valid diagnostic block
    assert!(rendered.contains("NeuralAmpModeler-rs Diagnostic"));
    assert!(rendered.contains("NeuralAmpModeler-rs v"));
    assert!(rendered.contains("arch="));
    assert!(rendered.contains("os="));
    assert!(rendered.contains("kernel="));
    assert!(rendered.contains("timestamp="));

    // Verify nominal state does not contain error fields or runtime model
    assert!(!rendered.contains("model.arch="));
    assert!(!rendered.contains("error_code="));
    assert!(!rendered.contains("mnemonic="));
    // Since no model is active, it should show model=none
    assert!(rendered.contains("model=none"));
}

#[test]
fn test_diagnostic_bundle_mandatory_fields() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let params = vec![("test_key", "test_value".to_string())];
    let bundle = DiagnosticBundle::capture_with_error(
        neural_amp_modeler_rs::common::diagnostics::NamErrorCode::ModelBuildFailed,
        params,
    );
    let rendered = bundle.render();

    // Test 2: Verify all mandatory fields from the contract are present:
    // 1. Error details
    assert!(rendered.contains("MODEL_BUILD_FAILED"));
    assert!(rendered.contains("E1303"));

    // 2. System and metadata fields
    assert!(rendered.contains("arch="));
    assert!(rendered.contains("os="));
    assert!(rendered.contains("kernel="));
    assert!(rendered.contains("features="));
    assert!(rendered.contains("timestamp="));
}

#[test]
fn test_diagnostic_bundle_regex_roundtrip() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let provider = MockSnapshotProvider {
        model: Some(ModelInfo {
            arch_label: "LSTM".to_string(),
            topology: "1x8".to_string(),
            channels: 8,
            receptive_field: 0,
            model_sample_rate: 48000,
            weights_layout: "GateMajorLstm".to_string(),
            path_basename: "test.nam".to_string(),
        }),
        audio: AudioInfo {
            sample_rate: 96000,
            buffer_size: 128,
            channel_count: 1,
            host_name: "PipeWire".to_string(),
        },
        rt: RtInfo {
            thread_priority: 95,
            scheduler: "FIFO".to_string(),
            cpu_pinned: Some(1),
            huge_pages_active: false,
            huge_page_mode: "off".to_string(),
        },
        telemetry: TelemetrySnapshot {
            p50_us: 10,
            p99_us: 20,
            p999_us: 30,
            max_us: 100,
            total_blocks: 500,
            xruns: 0,
            drains: 0,
            input_buffer_miss: 0,
            output_buffer_miss: 0,
        },
        flags: 0x01,
    };

    let consumer = AudioMetadata {
        channel_count: 1,
        host_name: "PipeWire".into(),
    };
    let bundle = DiagnosticBundle::capture_with_runtime(&provider, &consumer);
    let rendered = bundle.render();

    // Test 4: Custom parsing helper to check key=value lines
    let mut keys = std::collections::HashMap::new();
    for line in rendered.lines() {
        if let Some(pos) = line.find('=') {
            let key = line[..pos].trim();
            let val = line[pos + 1..].trim();
            keys.insert(key.to_string(), val.to_string());
        }
    }

    // Verify key-value round-trip matches input data
    assert_eq!(keys.get("model.arch").map(|s| s.as_str()), Some("LSTM"));
    assert_eq!(keys.get("model.channels").map(|s| s.as_str()), Some("8"));
    assert_eq!(
        keys.get("model.receptive_field").map(|s| s.as_str()),
        Some("0")
    );
    assert_eq!(
        keys.get("model.weights_layout").map(|s| s.as_str()),
        Some("GateMajorLstm")
    );
    assert_eq!(
        keys.get("model.sample_rate").map(|s| s.as_str()),
        Some("48000")
    );
    assert_eq!(
        keys.get("model.path_basename").map(|s| s.as_str()),
        Some("test.nam")
    );

    assert_eq!(keys.get("audio.sr").map(|s| s.as_str()), Some("96000"));
    assert_eq!(
        keys.get("audio.buffer_size").map(|s| s.as_str()),
        Some("128")
    );
    assert_eq!(
        keys.get("audio.channel_count").map(|s| s.as_str()),
        Some("1")
    );
    assert_eq!(
        keys.get("audio.host_name").map(|s| s.as_str()),
        Some("PipeWire")
    );

    assert_eq!(keys.get("rt.prio").map(|s| s.as_str()), Some("95"));
    assert_eq!(keys.get("rt.scheduler").map(|s| s.as_str()), Some("FIFO"));
    assert_eq!(keys.get("rt.cpu_pinned").map(|s| s.as_str()), Some("1"));
    assert_eq!(
        keys.get("rt.huge_pages_active").map(|s| s.as_str()),
        Some("false")
    );
    assert_eq!(
        keys.get("rt.huge_page_mode").map(|s| s.as_str()),
        Some("off")
    );

    assert_eq!(keys.get("telemetry.p50_us").map(|s| s.as_str()), Some("10"));
    assert_eq!(keys.get("telemetry.p99_us").map(|s| s.as_str()), Some("20"));
    assert_eq!(
        keys.get("telemetry.p999_us").map(|s| s.as_str()),
        Some("30")
    );
    assert_eq!(
        keys.get("telemetry.max_us").map(|s| s.as_str()),
        Some("100")
    );
    assert_eq!(
        keys.get("telemetry.total_blocks").map(|s| s.as_str()),
        Some("500")
    );
    assert_eq!(keys.get("telemetry.xruns").map(|s| s.as_str()), Some("0"));
    assert_eq!(keys.get("telemetry.drains").map(|s| s.as_str()), Some("0"));
    assert_eq!(keys.get("flags_seen").map(|s| s.as_str()), Some("0x1"));

    // Check formatting and presence of other dynamic system fields
    assert!(
        rendered
            .lines()
            .any(|l| l.contains("os=") && l.contains("kernel="))
    );
    assert!(rendered.lines().any(|l| l.starts_with("arch=")));
    assert!(rendered.lines().any(|l| l.starts_with("features=")));
    assert!(rendered.lines().any(|l| l.starts_with("timestamp=")));
}

#[test]
fn test_diagnostic_bundle_model_sample_rate_mismatch() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let provider = MockSnapshotProvider {
        model: Some(ModelInfo {
            arch_label: "WaveNet".to_string(),
            topology: "Lite".to_string(),
            channels: 12,
            receptive_field: 1024,
            model_sample_rate: 44100,
            weights_layout: "Interleaved4WaveNet".to_string(),
            path_basename: "wavenet_lite_44k.nam".to_string(),
        }),
        audio: AudioInfo {
            sample_rate: 48000,
            buffer_size: 256,
            channel_count: 2,
            host_name: "CLAP".to_string(),
        },
        rt: RtInfo {
            thread_priority: 90,
            scheduler: "FIFO".to_string(),
            cpu_pinned: Some(2),
            huge_pages_active: false,
            huge_page_mode: "off".to_string(),
        },
        telemetry: TelemetrySnapshot::default(),
        flags: 0,
    };

    let consumer = AudioMetadata {
        channel_count: 1,
        host_name: "Test".into(),
    };
    let bundle = DiagnosticBundle::capture_with_runtime(&provider, &consumer);
    let rendered = bundle.render();

    assert!(rendered.contains("model.sample_rate=44100"));
    assert!(rendered.contains("audio.sr=48000"));
    assert!(
        rendered.contains("model.sample_rate=44100") && rendered.contains("audio.sr=48000"),
        "Diagnostic bundle must expose model sample rate separately from host audio rate \
         so that a mismatch is visible for triage"
    );
}

#[cfg(feature = "heap-audit")]
mod heap_audit_tests {
    use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};
    use neural_amp_modeler_rs::common::spsc::RtStatusFlags;

    #[test]
    fn test_diagnostic_bundle_heap_audit() {
        let _guard = super::TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let rt_status = RtStatusFlags::new();
        rt_status.set_flag(neural_amp_modeler_rs::common::spsc::RT_STATUS_HAS_CLIPPED);
        rt_status.latency_hist.record(150);

        let allocs = {
            let _guard = TrackingGuard::new();

            rt_status.set_flag(neural_amp_modeler_rs::common::spsc::RT_STATUS_HAS_CLIPPED);
            rt_status.clear_flag(neural_amp_modeler_rs::common::spsc::RT_STATUS_HAS_CLIPPED);
            let _ =
                rt_status.check_flag(neural_amp_modeler_rs::common::spsc::RT_STATUS_HAS_CLIPPED);
            let _ = rt_status
                .check_and_clear_flag(neural_amp_modeler_rs::common::spsc::RT_STATUS_HAS_CLIPPED);
            rt_status.latency_hist.record(500);
            rt_status
                .xruns
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            rt_status
                .drains
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            get_alloc_count()
        };

        assert_eq!(
            allocs, 0,
            "RT status/telemetry operations triggered heap allocations!"
        );
    }

    #[test]
    fn test_panic_hook_zero_alloc() {
        let _guard = super::TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

        neural_amp_modeler_rs::common::panic_hook::install_panic_hook("test-panic-hook-zero-alloc");

        let allocs = {
            let _guard = TrackingGuard::new();
            let mut buf = [0u8; 4096];
            let written =
                neural_amp_modeler_rs::common::panic_hook::format_panic_report_for_audit_test(
                    &mut buf,
                    "test-component",
                    "test-thread",
                    "src/test.rs:42:5",
                    "test panic message",
                );
            assert!(written > 0, "format_panic_report should produce output");
            assert!(buf[..written].starts_with(b"======================================="));
            get_alloc_count()
        };

        assert_eq!(
            allocs, 0,
            "format_panic_report_to_buf triggered heap allocations during panic formatting!"
        );
    }
}
