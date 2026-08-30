// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Real-time latency telemetry for the DSP pipeline.

use std::sync::atomic::{AtomicU64, Ordering};

/// Exponential histogram for tracking the latency distribution of the RT callback.
///
/// Has 32 bins covering powers of 2 from 2^5 ns (~32 ns) to 2^36 ns (~68 s).
/// Uses atomic operations to ensure RT-safety (lock-free, zero-allocation).
#[repr(align(128))]
pub struct LatencyHistogram {
    /// Histogram atomic bins.
    bins: [AtomicU64; 32],
    /// Exact maximum latency observed (lock-free, via fetch_max).
    exact_max: AtomicU64,
    /// Exact minimum latency observed (lock-free, via fetch_min).
    exact_min: AtomicU64,
    /// Sum of all observations in nanoseconds for computing mean.
    sum_ns: AtomicU64,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyHistogram {
    /// Creates a new zeroed histogram.
    #[cold]
    pub fn new() -> Self {
        Self {
            bins: [const { AtomicU64::new(0) }; 32],
            exact_max: AtomicU64::new(0),
            exact_min: AtomicU64::new(u64::MAX),
            sum_ns: AtomicU64::new(0),
        }
    }

    /// Records a latency observation in nanoseconds.
    ///
    /// # RT-Safety
    /// This function is safe for calls in the DSP hot-path.
    #[inline(always)]
    pub fn record(&self, duration_ns: u64) {
        self.exact_max.fetch_max(duration_ns, Ordering::Relaxed);
        self.exact_min.fetch_min(duration_ns, Ordering::Relaxed);
        self.sum_ns.fetch_add(duration_ns, Ordering::Relaxed);
        if duration_ns == 0 {
            self.bins[0].fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Computes the bin index via approximate log2 (LZCNT).
        // 2^5 ns (32ns) -> bin 0
        // 2^36 ns (68s) -> bin 31
        let msb = 63 - duration_ns.leading_zeros();
        let index = msb.saturating_sub(5).min(31) as usize;

        self.bins[index].fetch_add(1, Ordering::Relaxed);
    }

    /// Computes the approximate value of a percentile (e.g., 0.95 for P95).
    /// Returns the value in nanoseconds of the bin's upper edge.
    pub fn get_percentile(&self, p: f32) -> u64 {
        let counts: [u64; 32] = core::array::from_fn(|i| self.bins[i].load(Ordering::Relaxed));
        let total: u64 = counts.iter().sum();

        if total == 0 {
            return 0;
        }

        let target = (total as f64 * p as f64) as u64;
        let mut accum: u64 = 0;

        for (i, &count) in counts.iter().enumerate() {
            accum += count;
            if accum >= target {
                // Returns 2^(i + 5)
                return 1u64 << (i + 5);
            }
        }

        1u64 << 36
    }

    /// Returns the maximum observed value (upper edge of the highest non-empty bin).
    pub fn get_max(&self) -> u64 {
        for i in (0..32).rev() {
            if self.bins[i].load(Ordering::Relaxed) > 0 {
                return 1u64 << (i + 5);
            }
        }
        0
    }

    /// Returns the minimum observed value (lower edge of the lowest non-empty bin).
    pub fn get_min(&self) -> u64 {
        for i in 0..32 {
            if self.bins[i].load(Ordering::Relaxed) > 0 {
                return if i == 0 { 0 } else { 1u64 << (i + 4) };
            }
        }
        0
    }

    /// Returns the mean/average observed latency in nanoseconds.
    pub fn get_mean(&self) -> u64 {
        let total = self.total_count();
        self.sum_ns
            .load(Ordering::Relaxed)
            .checked_div(total)
            .unwrap_or(0)
    }

    /// Returns the total number of observations recorded since the last reset.
    pub fn total_count(&self) -> u64 {
        self.bins.iter().map(|b| b.load(Ordering::Relaxed)).sum()
    }

    /// Zeros all histogram bins and resets statistics.
    ///
    /// Uses `swap` instead of `store` to guarantee visibility via cache-coherence
    /// (RMW). Best-effort reset: concurrent `fetch_add` records may be lost
    /// during the sweep; `record` calls are lock-free and never blocked.
    pub fn reset(&self) {
        for bin in &self.bins {
            bin.swap(0, Ordering::Relaxed);
        }
        self.exact_max.swap(0, Ordering::Relaxed);
        self.exact_min.swap(u64::MAX, Ordering::Relaxed);
        self.sum_ns.swap(0, Ordering::Relaxed);
    }

    /// Returns the exact maximum latency observed via lock-free fetch_max (in nanoseconds).
    pub fn get_exact_max(&self) -> u64 {
        self.exact_max.load(Ordering::Relaxed)
    }

    /// Atomically reads and resets the exact maximum (in nanoseconds).
    pub fn take_exact_max(&self) -> u64 {
        self.exact_max.swap(0, Ordering::Relaxed)
    }

    /// Returns the exact minimum latency observed via lock-free fetch_min (in nanoseconds).
    pub fn get_exact_min(&self) -> u64 {
        let val = self.exact_min.load(Ordering::Relaxed);
        if val == u64::MAX { 0 } else { val }
    }

    /// Atomically reads and resets the exact minimum (in nanoseconds).
    pub fn take_exact_min(&self) -> u64 {
        let val = self.exact_min.swap(u64::MAX, Ordering::Relaxed);
        if val == u64::MAX { 0 } else { val }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_mapping() {
        // Organization Test: Verifies that the system stores processing times
        // in the correct "drawers" (bins), separating ultra-fast from slow.
        let hist = LatencyHistogram::new();

        hist.record(10); // bin 0 (sub-32ns)
        hist.record(40); // bin 0 (2^5 = 32)
        hist.record(100); // bin 1 (2^6 = 64)
        hist.record(1000); // bin 4 (2^9 = 512)

        assert_eq!(hist.bins[0].load(Ordering::Relaxed), 2);
        assert_eq!(hist.bins[1].load(Ordering::Relaxed), 1);
        assert_eq!(hist.bins[4].load(Ordering::Relaxed), 1);
        assert_eq!(hist.get_exact_min(), 10);
        assert_eq!(hist.get_exact_max(), 1000);
        assert_eq!(hist.get_mean(), (10 + 40 + 100 + 1000) / 4);
    }

    #[test]
    fn test_percentiles() {
        // Statistics Test: Verifies that the system can correctly identify
        // the "median" (P50) and "worst cases" (P95 and P99) in a batch of measurements.
        let hist = LatencyHistogram::new();

        // 100 samples
        for _ in 0..50 {
            hist.record(100);
        } // P50 should be in bin 1 (2^6 = 64)
        for _ in 0..45 {
            hist.record(1000);
        } // P95 should be in bin 4 (2^9 = 512)
        for _ in 0..5 {
            hist.record(10000);
        } // P99 should be in bin 8 (2^13 = 8192)

        assert!(hist.get_percentile(0.50) <= 64);
        assert!(hist.get_percentile(0.95) <= 512);
        assert!(hist.get_percentile(0.99) <= 16384);
        assert_eq!(hist.get_exact_min(), 100);
        assert_eq!(hist.get_exact_max(), 10000);
    }
}
