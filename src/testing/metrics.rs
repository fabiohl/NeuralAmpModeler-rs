// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Generic numeric helpers shared by engine and downstream test harnesses.
//!
//! These are intentionally free of audio/host semantics so any consumer
//! (plugin, standalone, benchmark) can compare buffers or summarize latency
//! samples without carrying a private copy. Audio-specific metrics live under
//! [`crate::testing::perceptual`].

/// Maximum absolute sample-wise difference over the common prefix of `a` and `b`.
///
/// Returns `0.0` when either input is empty. Callers should pass equal-length
/// slices; only the shared prefix is compared.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
}

/// Exact percentile of `samples` (`p` in `[0.0, 1.0]`), sorting in place.
///
/// Uses the rank `(len * p).min(len).saturating_sub(1)` (the same convention as
/// [`crate::dsp::telemetry::ExactLatencyReservoir::percentile_exact`]). Returns
/// `0` for an empty slice; `p` is clamped to `[0.0, 1.0]`.
pub fn percentile_exact(samples: &mut [u64], p: f64) -> u64 {
    let len = samples.len();
    if len == 0 {
        return 0;
    }
    samples.sort_unstable();
    let clamped = p.clamp(0.0, 1.0);
    let rank = ((len as f64 * clamped) as usize).min(len).saturating_sub(1);
    samples[rank]
}

/// Exact percentile of a borrowed `samples` slice, cloning it first.
///
/// Convenience for callers that do not own a mutable buffer.
pub fn percentile_exact_copy(samples: &[u64], p: f64) -> u64 {
    let mut copy = samples.to_vec();
    percentile_exact(&mut copy, p)
}

#[cfg(test)]
#[path = "metrics_test.rs"]
mod metrics_test;
