// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Weight transposition routines for `.namb` v2 binary encoding.
//!
//! These functions rearrange weights from "Original" (NAM JSON) layout into
//! SIMD/SIMT-friendly formats that the processor consumes during inference.
//! The output layout must remain in lock-step with the decoder in
//! `dispatcher/lstm.rs` and `dispatcher/wavenet/`.

pub mod lstm;
pub mod wavenet;
