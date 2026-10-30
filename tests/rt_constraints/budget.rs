// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Single RT budget module — the one place that defines the
//! real-time processing budget for `tests/rt_constraints`.
//!
//! Previously `RT_DEADLINE_US` and `BLOCK_SIZE` were independently
//! redeclared in `rt_deadline.rs` and `rt_jitter.rs`, allowing silent skew
//! between the deadline gate (phase 4) and the jitter telemetry (phase 5).
//! Both suites now import from here.
//!
//! ## Budget formula (previously implicit)
//!
//! ```text
//! budget_us = 1_000_000 * block_size / sample_rate_hz
//! ```
//!
//! For the canonical operating point (64 samples @ 48 kHz):
//! `1_000_000 * 64 / 48_000 = 1333 µs` (truncated). The enforced gate
//! [`RT_DEADLINE_US`] is `1330 µs` — a 3 µs conservative margin below the
//! exact budget, retained verbatim so both suites behave identically
//! before/after this extraction.
//!
//! Parametrization by the real `(sample_rate, block_size)` of each scenario
//! is intentionally out of scope here (requires explicit design decision for multi-rate suites).

/// RT deadline for 64 samples at 48 kHz: 1.33 ms (see module docs for the margin).
pub const RT_DEADLINE_US: u64 = 1330;

/// DSP block size in samples (standard 48 kHz audio-host buffer).
pub const BLOCK_SIZE: usize = 64;

/// Canonical sample rate of the RT budget operating point.
pub const BUDGET_SAMPLE_RATE_HZ: u32 = 48_000;

/// Computes the exact processing budget in microseconds for a
/// `(block_size, sample_rate_hz)` operating point.
///
/// `sample_rate_hz` must be non-zero.
pub const fn budget_us(block_size: usize, sample_rate_hz: u32) -> u64 {
    // SAFETY: none — pure integer arithmetic, no unsafe.
    (1_000_000u64 * block_size as u64) / sample_rate_hz as u64
}
