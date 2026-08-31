// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Validation guards for `.nam` format (JSON).
//!
//! Split across two submodules:
//! - [`schema`]: structural (per-field) JSON deserialization guards with
//!   parse-time constants and custom serde visitors.
//! - [`semantic`]: semantic (cross-field / topology) bounds used after
//!   deserialization by the dispatcher and topology detection.

use super::error::JsonError;
use std::cell::RefCell;

pub(crate) mod schema;
pub(crate) mod semantic;

pub(crate) use schema::*;
pub use schema::{MAX_HIDDEN_SIZE, MAX_LAYERS};
pub use semantic::*;

// Side channel for typed parse errors produced by serde visitors (T5.1).
//
// `serde::de::Error::custom` only preserves a `Display` string — the typed
// `JsonError` raised inside a visitor (e.g. `WeightNotFinite`) would be
// flattened into `JsonError::Serde`, losing the precise variant and forcing
// `build.rs` to fall back to the generic `NamJsonParseError`. The visitors
// record the typed error here immediately before returning the serde error;
// [`super::parse::parse_nam_json`] clears the slot before deserializing and
// takes it on failure, falling back to `JsonError::Serde` when no visitor
// produced a typed error.
//
// Thread-local: parsing is off-RT and each parse call clears the slot on
// entry, so concurrent parses on different threads never interfere.
thread_local! {
    static LAST_TYPED_PARSE_ERROR: RefCell<Option<JsonError>> = const { RefCell::new(None) };
}

/// Clears the typed-error slot before a fresh deserialization pass.
pub(crate) fn clear_last_typed_parse_error() {
    LAST_TYPED_PARSE_ERROR.with(|c| *c.borrow_mut() = None);
}

/// Records a typed error produced by a serde visitor.
pub(crate) fn record_last_typed_parse_error(err: JsonError) {
    LAST_TYPED_PARSE_ERROR.with(|c| *c.borrow_mut() = Some(err));
}

/// Takes the typed error recorded by the last visitor, if any.
pub(crate) fn take_last_typed_parse_error() -> Option<JsonError> {
    LAST_TYPED_PARSE_ERROR.with(|c| c.borrow_mut().take())
}
