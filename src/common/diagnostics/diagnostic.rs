// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Structured diagnostic for NeuralAmpModeler-rs errors and warnings.
//!
//! Combines a user-friendly message with a copyable technical
//! support block. Formatted for precise triage.

pub(crate) use super::bundle::{DiagnosticBundle, ErrorContext};
use super::error_codes::NamErrorCode;
#[cfg(test)]
pub(crate) use super::format::days_to_date;
use super::format::timestamp;
use super::snapshot::RuntimeSnapshot;
#[cfg(test)]
pub(crate) use super::snapshot::{ACTIVE_MODEL_NAME, ACTIVE_SAMPLE_RATE};
use super::system_info::SystemSnapshot;
use std::fmt;

/// Structured diagnostic for NeuralAmpModeler-rs errors and warnings.
///
/// Combines a user-friendly message with a copyable technical
/// support block. Formatted for precise triage.
#[derive(Debug)]
pub struct NamDiagnostic {
    /// Typed error code.
    code: NamErrorCode,
    /// User-friendly message (en-US).
    user_message: String,
    /// Suggested action for the user (en-US).
    user_hint: String,
    /// Contextual parameters for the support block (key=value).
    params: Vec<(&'static str, String)>,
    /// System snapshot (captured at startup).
    system: SystemSnapshot,
}

impl NamDiagnostic {
    /// Creates a new diagnostic with the error code and system snapshot.
    #[cold]
    pub fn new(code: NamErrorCode, system: &SystemSnapshot) -> Self {
        Self {
            code,
            user_message: String::new(),
            user_hint: String::new(),
            params: Vec::new(),
            system: system.clone(),
        }
    }

    /// Sets the user-friendly message.
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.user_message = msg.into();
        self
    }

    /// Sets the suggested action for the user.
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.user_hint = hint.into();
        self
    }

    /// Adds a contextual parameter to the support block.
    pub fn param(mut self, key: &'static str, value: impl fmt::Display) -> Self {
        self.params.push((key, value.to_string()));
        self
    }

    /// Generates the ISO 8601 timestamp for the current moment.
    pub fn timestamp() -> String {
        timestamp()
    }

    /// Generates the formatted technical support block for copying.
    pub(crate) fn support_block(&self) -> String {
        DiagnosticBundle {
            system: self.system.clone(),
            runtime: RuntimeSnapshot::default(),
            error: Some(ErrorContext {
                code: self.code,
                params: self.params.clone(),
            }),
            full: false,
        }
        .render()
    }

    /// Prints the complete diagnostic to stderr (user-friendly message + support block).
    pub fn emit(&self) {
        log::error!(
            "{}\n\n{}\n\n{}",
            self.user_message,
            if self.user_hint.is_empty() {
                ""
            } else {
                &self.user_hint
            },
            self.support_block()
        );
    }

    /// Prints as a warning (non-fatal) with a distinct visual prefix.
    pub fn emit_warning(&self) {
        log::warn!(
            "{}\n\n{}\n\n{}",
            self.user_message,
            if self.user_hint.is_empty() {
                ""
            } else {
                &self.user_hint
            },
            self.support_block()
        );
    }

    /// Returns the error code associated with the diagnostic.
    pub fn error_code(&self) -> NamErrorCode {
        self.code
    }
}

impl fmt::Display for NamDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code.code(), self.user_message)
    }
}

impl std::error::Error for NamDiagnostic {}

#[cfg(test)]
#[path = "diagnostic_test.rs"]
mod diagnostic_test;
