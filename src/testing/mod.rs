// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Testing utilities for NeuralAmpModeler-rs cross-validation and signal generation.
//!
//! This module provides:
//! - Deterministic stress signal generators (v1 for fast CI, v2 for comprehensive validation)
//! - Audio primitives ported from t3k-mushra (MIT-licensed)
//! - Perceptual metrics (ESR, LUFS) calibrated against published baselines
//! - ASR (Aliasing-to-Signal Ratio) metric per Sato & Smith DAFx 2025
//! - Centralized fixture-path resolution for models, goldens, and stress signals
//! - WAV I/O helpers for test fixtures

pub mod aliasing;
pub mod bin_guard;
pub mod catalog;
pub mod diagnostics;
pub mod fixtures;
pub mod freshness;
pub mod isa_guard;
pub mod mushra;
pub mod perceptual;
#[cfg(feature = "testing")]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
pub mod qa;
pub mod receipt;
pub mod reference_oracle;
pub mod spectral;
pub mod stress;
pub mod wav;

#[cfg(feature = "avx512")]
pub use isa_guard::ForceAvx512Guard;
pub use isa_guard::{ForceAvx2Guard, IsaGuard};
