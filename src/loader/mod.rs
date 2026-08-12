// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Loading module for the NAM ecosystem.
//!
//! Contains parsers for .nam (JSON) and .namb (binary) formats.
//! The entire loading process occurs **outside** the RT thread to
//! avoid any unwanted allocation during audio processing.

/// Model construction: assembles `StaticModel` instances from parsed weight data.
pub mod build;
/// Architecture-specific model dispatcher: routes parsed weights to concrete model builders.
pub mod dispatcher;
/// Struct, constants, and Debug impl for `LoadedModelPair`.
pub mod loaded_model_pair;
/// `.nam` (JSON) format parser: schema validation, topology parsing, activation detection.
pub mod nam_json;
/// `.namb` (binary) format: header parsing, layout decoding, buffer-to-model construction.
pub mod namb;
/// `.namb` encoder: serializes `NamModelData` into the binary compact profile format.
pub mod namb_encoder;
/// Weight matrix transposition utilities for interleaved memory layouts.
pub mod transpose;

pub use build::load_and_build_model;
pub use loaded_model_pair::*;

#[cfg(test)]
#[path = "loader_malformed_test.rs"]
mod loader_malformed_test;

/// Controls loading behaviour and model initialization.
///
/// Produced by the main/UI thread and consumed by the loader before passing
/// the ready-to-render model pair to the RT thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoadOptions {
    /// `None`  → use default (prewarm runs normally).
    /// `Some(false)` → skip the initial prewarm pass (fast preview / preset browsing).
    /// `Some(true)`  → force prewarm on (explicit override).
    pub prewarm: Option<bool>,
}
