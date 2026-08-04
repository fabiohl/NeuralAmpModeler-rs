// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Parity test suite entry point for `NeuralAmpModeler-rs`.
//!
//! Aggregates cross-validation test modules comparing Rust DSP execution against reference
//! C++ NAMCore behavior and the double-precision (f64) ideal mathematical oracle.

mod common;

// ── C++ Upstream Parity Submodules ───────────────────────────────────────────
#[path = "parity/cabsim_cpp_parity.rs"]
mod cabsim_cpp_parity;
#[path = "parity/cpp_parity.rs"]
mod cpp_parity;
#[path = "parity/isa_parity.rs"]
mod isa_parity;

// ── Low-Level & Quantization Parity Submodules ──────────────────────────────
#[path = "parity/lstm_gate_bf16_parity.rs"]
mod lstm_gate_bf16_parity;
#[path = "parity/lstm_scalar_bf16_parity.rs"]
mod lstm_scalar_bf16_parity;
#[path = "parity/parity_primitives.rs"]
mod parity_primitives;

// ── Double-Precision Reference Oracle Submodule ──────────────────────────────
#[path = "parity/reference_oracle_f64.rs"]
mod reference_oracle_f64;
