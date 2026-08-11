// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP core invariants test suite entry point for `NeuralAmpModeler-rs`.
//!
//! Validates DSP fundamental properties such as resampler block-size
//! invariance, sample-rate conversion determinism, and polyphase filter
//! stability under variable buffer fragmentation.

mod common;

// ── DSP Core Invariant Submodules ────────────────────────────────────────────
#[path = "dsp_core/resampler_invariance_test.rs"]
mod resampler_invariance_test;
