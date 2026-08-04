// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Backward-compatible re-exports from `neural_amp_modeler_rs::testing::wav`.
//!
//! The canonical implementation lives in the library crate so that binaries
//! (`gen_stress`, `wav_to_golden`) can use it directly.

#![allow(dead_code)]
#![allow(unused_imports)]

pub use neural_amp_modeler_rs::testing::wav::*;
