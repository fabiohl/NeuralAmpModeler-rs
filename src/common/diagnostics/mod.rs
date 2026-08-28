// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Structured diagnostics system for NeuralAmpModeler-rs.
//!
//! Provides error messages in two layers:
//! 1. **User-friendly message** — readable text for the end user.
//! 2. **Support block** — typed error code, contextual parameters,
//!    version, architecture, and timestamp for precise triage by devs/AI.
//!
//! ## Usage outside the RT thread
//!
//! All formatting and printing of diagnostics occurs **exclusively** in
//! non-RT threads (CLI, host main loop). The `process()` callback
//! continues using atomic flags (`RtStatusFlags`) for silent signaling.

pub mod bundle;
pub mod diagnostic;
pub mod error_codes;
pub mod format;
pub mod logger;
pub mod snapshot;
pub mod system_info;

pub use bundle::{DiagnosticBundle, ErrorContext};
pub use diagnostic::NamDiagnostic;
pub use error_codes::{NamError, NamErrorCode};
pub use logger::{
    HostLogFn, InstanceLogScope, LogBuffer, LogRecord, LoggerConfig, NamLogger,
    current_instance_id, scope_instance, set_current_instance_id, with_instance_id,
};
pub use snapshot::{
    ACTIVE_MODEL_INFO, ACTIVE_MODEL_NAME, ACTIVE_SAMPLE_RATE, AudioInfo, AudioMetadata,
    HasRuntimeSnapshot, ModelInfo, RtInfo, RuntimeSnapshot, TelemetrySnapshot,
};
pub use system_info::SystemSnapshot;
pub use system_info::set_host_library_version;
