// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Validation guards for `.nam` format (JSON).
//!
//! Split across two submodules:
//! - [`schema`]: structural (per-field) JSON deserialization guards with
//!   parse-time constants and custom serde visitors.
//! - [`semantic`]: semantic (cross-field / topology) bounds used after
//!   deserialization by the dispatcher and topology detection.

pub(crate) mod schema;
pub(crate) mod semantic;

pub(crate) use schema::*;
pub use schema::{MAX_HIDDEN_SIZE, MAX_LAYERS};
pub use semantic::*;
