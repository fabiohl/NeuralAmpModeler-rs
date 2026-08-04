// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Diagnostic bundle — captures and renders system/runtime snapshots
//! with optional error context for support triage.

use super::error_codes::NamErrorCode;
use super::format::{format_model_path, redact_path, redact_text, timestamp};
use super::logger::NamLogger;
use super::snapshot::{ACTIVE_MODEL_NAME, ACTIVE_SAMPLE_RATE, RuntimeSnapshot};
use super::system_info::SystemSnapshot;
use std::sync::atomic::Ordering;

/// Context of the error for the diagnostic bundle.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Typed error code.
    pub code: NamErrorCode,
    /// Contextual parameters (key=value).
    pub params: Vec<(&'static str, String)>,
}

/// Decoupled diagnostic bundle containing system information, runtime state, and optional error.
///
/// # Privacy & Redaction Policy
/// To protect user privacy when sharing support blocks publicly:
/// - By default (when `full` is false), absolute path prefixes like `$HOME` are replaced with `~`,
///   and `$XDG_RUNTIME_DIR` paths are replaced with a placeholder. Basenames of model files are shown
///   instead of full paths.
/// - When `full` is true (e.g. `--diagnose-full` or interactive `:diag --full`), paths are shown unredacted (bruto).
/// - Weights content, audio signals, and user/host names are never captured or included.
#[derive(Debug, Clone)]
pub struct DiagnosticBundle {
    /// System snapshot (OS, kernel, etc).
    pub system: SystemSnapshot,
    /// Runtime snapshot (buffers, samplerate, etc).
    pub runtime: RuntimeSnapshot,
    /// Error context if captured due to an error.
    pub error: Option<ErrorContext>,
    /// Whether to print absolute paths unredacted.
    pub full: bool,
}

impl DiagnosticBundle {
    /// Captures a nominal diagnostic bundle without error context.
    pub fn capture() -> Self {
        Self {
            system: SystemSnapshot::capture(),
            runtime: RuntimeSnapshot::default(),
            error: None,
            full: false,
        }
    }

    /// Captures a nominal diagnostic bundle, capturing dynamic runtime state via the given provider.
    pub fn capture_with_runtime(
        provider: &impl super::snapshot::HasRuntimeSnapshot,
        consumer: &super::snapshot::AudioMetadata,
    ) -> Self {
        Self {
            system: SystemSnapshot::capture(),
            runtime: RuntimeSnapshot::capture(provider, consumer),
            error: None,
            full: false,
        }
    }

    /// Captures a diagnostic bundle with the given error code and parameters.
    pub fn capture_with_error(code: NamErrorCode, params: Vec<(&'static str, String)>) -> Self {
        Self {
            system: SystemSnapshot::capture(),
            runtime: RuntimeSnapshot::default(),
            error: Some(ErrorContext { code, params }),
            full: false,
        }
    }

    /// Captures a diagnostic bundle with error and runtime provider.
    pub fn capture_with_error_and_runtime(
        code: NamErrorCode,
        params: Vec<(&'static str, String)>,
        provider: &impl super::snapshot::HasRuntimeSnapshot,
        consumer: &super::snapshot::AudioMetadata,
    ) -> Self {
        Self {
            system: SystemSnapshot::capture(),
            runtime: RuntimeSnapshot::capture(provider, consumer),
            error: Some(ErrorContext { code, params }),
            full: false,
        }
    }

    /// Builder method to specify whether to use full (unredacted) paths.
    pub fn with_full(mut self, full: bool) -> Self {
        self.full = full;
        self
    }

    /// Renders the support block as a formatted string.
    pub fn render(&self) -> String {
        let separator = "────────────────────────────────────────────────";
        let mut block = if let Some(ref err) = self.error {
            format!(
                "──── NAM-rs Diagnostic {separator}\n\
                 nam-rs v{} | {} | {}\n",
                self.system.version,
                err.code.code(),
                err.code.mnemonic(),
            )
        } else {
            format!(
                "──── NAM-rs Diagnostic {separator}\n\
                 nam-rs v{}\n",
                self.system.version,
            )
        };

        // Contextual parameters
        if let Some(err) = self.error.as_ref().filter(|e| !e.params.is_empty()) {
            let param_line: String = err
                .params
                .iter()
                .map(|(k, v)| {
                    let v_redacted = {
                        let path = std::path::Path::new(v);
                        if path.is_absolute() {
                            redact_path(path, self.full)
                        } else {
                            v.clone()
                        }
                    };
                    format!("{k}={v_redacted}")
                })
                .collect::<Vec<_>>()
                .join(" ");
            block.push_str(&param_line);
            block.push('\n');
        }

        // Runtime State
        let model_name_raw = ACTIVE_MODEL_NAME
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let model_name = if model_name_raw.is_empty() {
            self.runtime
                .model
                .as_ref()
                .map(|m| m.path_basename.clone())
                .unwrap_or_default()
        } else {
            model_name_raw
        };
        let active_rate = {
            let r = ACTIVE_SAMPLE_RATE.load(Ordering::Relaxed);
            if r == 0 {
                self.runtime.audio.sample_rate
            } else {
                r
            }
        };

        block.push_str("──── Runtime State ─────────────────────────────\n");
        let mut model_printed = false;
        if !model_name.is_empty() {
            let formatted = format_model_path(&model_name, self.full);
            block.push_str(&format!("model={}\n", formatted));
            model_printed = true;
        }
        if active_rate != 0 {
            block.push_str(&format!("sample_rate={}\n", active_rate));
        }

        if let Some(ref model) = self.runtime.model {
            let path_val = if self.full {
                model.path_basename.clone()
            } else {
                let path = std::path::Path::new(&model.path_basename);
                if let Some(file_name) = path.file_name() {
                    file_name.to_string_lossy().into_owned()
                } else {
                    model.path_basename.clone()
                }
            };
            block.push_str(&format!(
                "model.arch={}\n\
                 model.topology={}\n\
                 model.channels={}\n\
                 model.receptive_field={}\n\
                 model.sample_rate={}\n\
                 model.weights_layout={}\n\
                 model.path_basename={}\n",
                model.arch_label,
                model.topology,
                model.channels,
                model.receptive_field,
                model.model_sample_rate,
                model.weights_layout,
                path_val
            ));
            model_printed = true;
        }

        if !model_printed {
            block.push_str("model=none\n");
        }

        block.push_str(&format!(
            "audio.sr={}\n\
             audio.buffer_size={}\n\
             audio.channel_count={}\n\
             audio.host_name={}\n\
             rt.prio={}\n\
             rt.scheduler={}\n\
             rt.cpu_pinned={}\n\
             rt.huge_pages_active={}\n\
             rt.huge_page_mode={}\n\
             telemetry.p50_us={}\n\
             telemetry.p99_us={}\n\
             telemetry.p999_us={}\n\
             telemetry.max_us={}\n\
             telemetry.total_blocks={}\n\
             telemetry.xruns={}\n\
             telemetry.drains={}\n\
             flags_seen={:#x}\n",
            self.runtime.audio.sample_rate,
            self.runtime.audio.buffer_size,
            self.runtime.audio.channel_count,
            self.runtime.audio.host_name,
            self.runtime.rt.thread_priority,
            self.runtime.rt.scheduler,
            self.runtime
                .rt
                .cpu_pinned
                .map(|c| c.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.runtime.rt.huge_pages_active,
            self.runtime.rt.huge_page_mode,
            self.runtime.telemetry.p50_us,
            self.runtime.telemetry.p99_us,
            self.runtime.telemetry.p999_us,
            self.runtime.telemetry.max_us,
            self.runtime.telemetry.total_blocks,
            self.runtime.telemetry.xruns,
            self.runtime.telemetry.drains,
            self.runtime.flags_seen
        ));

        // System information
        block.push_str(&format!(
            "arch={}\n\
             os={} kernel={}\n\
             host_library={}\n\
             features={}\n\
             timestamp={}\n",
            self.system.arch,
            self.system.os,
            self.system.kernel,
            self.system.host_library_version.as_deref().unwrap_or("N/A"),
            if self.system.features.is_empty() {
                "none (baseline x86-64-v3 only)".to_string()
            } else {
                self.system.features.join(", ")
            },
            timestamp(),
        ));

        // Recent Log Trace
        if let Some(buffer) = NamLogger::log_buffer() {
            let trace = buffer.render_trace(50);
            if !trace.is_empty() {
                let trace_formatted = redact_text(&trace, self.full);
                block.push_str("──── Recent Log Trace ─────────────────────────\n");
                block.push_str(&trace_formatted);
            }
        }

        block.push_str(&format!(
            "{separator}\n\
             Copy the block above when opening a support ticket.",
        ));

        block
    }
}
