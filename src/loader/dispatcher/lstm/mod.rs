// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! LSTM model construction sub-modules.
//!
//! - `dispatch` – geometry-based dispatch to static profiles or dynamic fallback.
//! - `static_builder` – compile-time-optimized single and dual-layer builders.
//! - `dynamic_builder` – runtime-shaped builder for arbitrary geometries.
//! - `weights` – weight extraction and interleave packing utilities.

pub(crate) mod dispatch;
pub(crate) mod dynamic_builder;
pub(crate) mod static_builder;
pub(crate) mod weights;

pub(crate) use dispatch::build_lstm;
