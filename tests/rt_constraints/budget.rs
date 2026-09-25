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

/// DSP block size in samples (standard 48 kHz JACK/PipeWire buffer).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_values_are_frozen() {
        assert_eq!(RT_DEADLINE_US, 1330);
        assert_eq!(BLOCK_SIZE, 64);
        assert_eq!(BUDGET_SAMPLE_RATE_HZ, 48_000);
    }

    #[test]
    fn canonical_budget_matches_formula_with_conservative_margin() {
        // Exact budget at the canonical point truncates to 1333 µs; the
        // enforced gate keeps a 3 µs conservative margin below it.
        assert_eq!(budget_us(BLOCK_SIZE, BUDGET_SAMPLE_RATE_HZ), 1333);
        assert_eq!(budget_us(64, 48_000), 1333);
        assert!(
            RT_DEADLINE_US <= budget_us(BLOCK_SIZE, BUDGET_SAMPLE_RATE_HZ),
            "gate {} must not exceed exact budget {}",
            RT_DEADLINE_US,
            budget_us(BLOCK_SIZE, BUDGET_SAMPLE_RATE_HZ)
        );
        assert_eq!(
            budget_us(BLOCK_SIZE, BUDGET_SAMPLE_RATE_HZ) - RT_DEADLINE_US,
            3
        );
    }
}
