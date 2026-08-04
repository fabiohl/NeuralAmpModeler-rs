// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Loading module for the NAM ecosystem.
//!
//! Contains parsers for .nam (JSON) and .namb (binary) formats.
//! The entire loading process occurs **outside** the RT thread to
//! avoid any unwanted allocation during audio processing.

pub mod build;
pub mod dispatcher;
/// Struct, constants, and Debug impl for `LoadedModelPair`.
pub mod loaded_model_pair;
pub mod nam_json;
pub mod namb;
pub mod namb_encoder;
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
