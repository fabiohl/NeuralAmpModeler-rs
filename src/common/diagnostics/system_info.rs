// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Snapshot of the execution environment, captured once at startup.
//!
//! Includes static system information useful for triage:
//! NeuralAmpModeler-rs version, architecture, OS, CPU features, kernel version.

/// Binary version, injected at compile time.
const NAM_VERSION: &str = env!("CARGO_PKG_VERSION");

use std::sync::OnceLock;

/// Audio backend library version, populated by the host at startup.
static HOST_LIBRARY_VERSION: OnceLock<String> = OnceLock::new();

/// Registers the host library version for inclusion in diagnostic snapshots.
///
/// Called by the host after audio backend initialization.
pub fn set_host_library_version(version: String) {
    let _ = HOST_LIBRARY_VERSION.set(version);
}

/// Returns the registered host library version, if set.
pub(crate) fn host_library_version() -> Option<&'static str> {
    HOST_LIBRARY_VERSION.get().map(String::as_str)
}

/// Snapshot of the execution environment, captured once at startup.
///
/// Includes static system information useful for triage:
/// NeuralAmpModeler-rs version, architecture, OS, CPU features, kernel version.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SystemSnapshot {
    /// NeuralAmpModeler-rs version (e.g. "1.0.0").
    pub version: &'static str,
    /// CPU architecture (e.g. "x86_64").
    pub arch: &'static str,
    /// Operating system (e.g. "linux").
    pub os: &'static str,
    /// Linux kernel version (read from /proc/version).
    pub kernel: String,
    /// Detected CPU features (above x86-64-v3 baseline).
    pub features: Vec<String>,
    /// Host audio backend library version, populated by the standalone host (None in library-only contexts).
    pub host_library_version: Option<String>,
}

impl SystemSnapshot {
    /// Captures system information at the time of the call.
    ///
    /// Should be called once in `main()` and the result propagated
    /// to functions that emit diagnostics.
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_amp_modeler_rs::SystemSnapshot;
    ///
    /// let sys = SystemSnapshot::capture();
    /// assert!(!sys.version.is_empty());
    /// assert_eq!(sys.arch, std::env::consts::ARCH);
    /// assert_eq!(sys.os, std::env::consts::OS);
    /// ```
    pub fn capture() -> Self {
        let kernel = read_kernel_version();
        let features = detect_advanced_features();
        Self {
            version: NAM_VERSION,
            arch: std::env::consts::ARCH,
            os: std::env::consts::OS,
            kernel,
            features,
            host_library_version: HOST_LIBRARY_VERSION.get().cloned(),
        }
    }

    /// Emits an informational advisory recommending IRQ isolation from a specific
    /// CPU core, in order to minimize real-time preemption.
    pub fn emit_irq_advisory(&self, core: usize) {
        #[cfg(target_os = "linux")]
        {
            if core >= 64 {
                return;
            }
            let mask = 0xFFFFFFFFFFFFFFFFu64 ^ (1u64 << core);

            log::info!(
                "💡 For minimum latency, isolate core {} from system IRQs (as sudo): echo {:X} > /proc/irq/default_smp_affinity",
                core,
                mask
            );
        }
        #[cfg(not(target_os = "linux"))]
        let _ = core;
    }
}

/// Returns the compile-time baseline ISA label when no optional features are detected.
pub(crate) const fn baseline_isa_label() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "none (baseline x86-64-v3 only)"
    } else if cfg!(target_arch = "aarch64") {
        "none (baseline aarch64 NEON)"
    } else {
        "none (baseline)"
    }
}

/// Reads the OS/kernel version string for diagnostics (off-RT).
fn read_kernel_version() -> String {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/version")
            .ok()
            .and_then(|v| {
                // Format: "Linux version 6.8.0-generic (...)"
                v.split_whitespace().nth(2).map(String::from)
            })
            .unwrap_or_else(|| "unknown".to_string())
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        // SAFETY: `utsname` is a plain C struct; `libc::uname` initializes it when returning 0.
        let mut uts: libc::utsname = unsafe { std::mem::zeroed() };
        // SAFETY: `uts` is a valid mutable pointer to `libc::utsname`.
        if unsafe { libc::uname(&mut uts) } == 0 {
            // SAFETY: `uname` NUL-terminates `release` on success.
            let release = unsafe { std::ffi::CStr::from_ptr(uts.release.as_ptr()) };
            if let Ok(s) = release.to_str()
                && !s.is_empty()
            {
                return s.to_string();
            }
        }
        "unknown".to_string()
    }
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Wdk::System::SystemServices::RtlGetVersion;
        use windows_sys::Win32::System::SystemInformation::OSVERSIONINFOW;
        let mut info = OSVERSIONINFOW {
            dwOSVersionInfoSize: core::mem::size_of::<OSVERSIONINFOW>() as u32,
            ..Default::default()
        };
        // SAFETY: `dwOSVersionInfoSize` is set to `size_of::<OSVERSIONINFOW>()` as required by MSDN.
        if unsafe { RtlGetVersion(&raw mut info) } == 0 && info.dwMajorVersion != 0 {
            format!(
                "{}.{}.{}",
                info.dwMajorVersion, info.dwMinorVersion, info.dwBuildNumber
            )
        } else {
            "unknown".to_string()
        }
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        "unknown".to_string()
    }
}

/// Detects advanced CPU features (above x86-64-v3 baseline).
#[cfg(target_arch = "x86_64")]
fn detect_advanced_features() -> Vec<String> {
    let mut features = Vec::new();
    // Features de x86-64-v4 (AVX-512)
    if std::is_x86_feature_detected!("avx512f") {
        features.push("avx512f".to_string());
    }
    if std::is_x86_feature_detected!("avx512bw") {
        features.push("avx512bw".to_string());
    }
    if std::is_x86_feature_detected!("avx512dq") {
        features.push("avx512dq".to_string());
    }
    if std::is_x86_feature_detected!("avx512vl") {
        features.push("avx512vl".to_string());
    }
    if std::is_x86_feature_detected!("avx512vnni") {
        features.push("avx512vnni".to_string());
    }
    if std::is_x86_feature_detected!("avx512bf16") {
        features.push("avx512bf16".to_string());
    }
    features
}

/// Detects optional CPU features above the aarch64 NEON baseline.
#[cfg(target_arch = "aarch64")]
fn detect_advanced_features() -> Vec<String> {
    let mut features = Vec::new();
    for (name, present) in [
        ("fp16", std::arch::is_aarch64_feature_detected!("fp16")),
        (
            "dotprod",
            std::arch::is_aarch64_feature_detected!("dotprod"),
        ),
        ("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")),
        ("bf16", std::arch::is_aarch64_feature_detected!("bf16")),
        ("sve", std::arch::is_aarch64_feature_detected!("sve")),
        ("sve2", std::arch::is_aarch64_feature_detected!("sve2")),
    ] {
        if present {
            features.push(name.to_string());
        }
    }
    features
}
