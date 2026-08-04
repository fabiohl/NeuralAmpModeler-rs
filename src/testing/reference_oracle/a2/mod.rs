// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! f64 reference oracle for WaveNet A2 models.
//!
//! - `static_eval` — weight extraction and static array construction from parsed model data.
//! - `dynamic_eval` — forward-pass execution loop with FiLM conditioning and gating.

#![allow(missing_docs)]

pub(crate) mod dynamic_eval;
pub(crate) mod static_eval;

pub(crate) use dynamic_eval::oracle_a2_forward;
