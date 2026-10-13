// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Real-time latency telemetry for the DSP pipeline.
//!
//! Two complementary atomic histograms coexist:
//!
//! * `AtomicHistogram` — the legacy 32-bin octave histogram. Cheap and
//!   stable; its percentile output is always a power-of-two upper edge
//!   (relative error up to 100 %).
//! * `LatencyHistogram` — the log-linear 256-bin histogram (32 octaves ×
//!   8 sub-bins, HdrHistogram style) with a bounded **≤ 12.5 %** relative
//!   error, so `p50 ≠ p99` is observable and gate thresholds can be relative
//!   to a quantum instead of snapping to the next power of two.
//!
//! The complementary `ExactLatencyReservoir` (`testing`/`heap-audit`
//! only, never production) stores exact samples for trend reports so
//! sub-bucket drift is visible without touching the gate.

use std::sync::atomic::{AtomicU64, Ordering};

/// Number of sub-bins per octave in [`LatencyHistogram`] (3 bits ⇒ 8).
const SUB_BITS: u32 = 3;
/// Number of sub-bins per octave.
const SUB_COUNT: usize = 1 << SUB_BITS;
/// Total number of atomic counters in [`LatencyHistogram`] (32 octaves × 8 sub-bins).
const LATENCY_BIN_COUNT: usize = 32 * SUB_COUNT;

/// Maps a nanosecond latency to its [`LatencyHistogram`] sub-bin index.
///
/// The index is `(msb << 3) | ((v >> (msb - 3)) & 7)` for `msb >= 3`. Values
/// below 8 ns (`msb < 3`) are clamped to their own exact bin (`index == v`) so
/// the shift never underflows. Values at or above `2^31` ns saturate into the
/// top octave; the documented ≤ 12.5 % relative-error bound of the reported
/// upper edge holds across the whole `[1, 2^31)` range, far beyond any
/// real-time audio latency.
#[inline(always)]
fn latency_bin_index(value: u64) -> usize {
    if value == 0 {
        return 0;
    }
    let msb = (63 - value.leading_zeros()).min(31);
    if msb < SUB_BITS {
        value as usize
    } else {
        (((msb as u64) << SUB_BITS) | ((value >> (msb - SUB_BITS)) & (SUB_COUNT as u64 - 1)))
            as usize
    }
}

/// Inclusive lower edge of a [`LatencyHistogram`] sub-bin.
///
/// For the clamp region (`index < 8`) the bin holds the single exact value
/// `index`. Bins `8..24` are unreachable (no value maps to them) and keep the
/// identity edge so the helper stays total; the reachable bins are strictly
/// monotone, which is what percentile scanning relies on.
#[inline]
fn latency_bin_lower_edge(index: usize) -> u64 {
    if index < SUB_COUNT {
        return index as u64;
    }
    let msb = index >> SUB_BITS;
    if msb < SUB_BITS as usize {
        return index as u64;
    }
    let sub = (index & (SUB_COUNT - 1)) as u64;
    (SUB_COUNT as u64 + sub) << (msb - SUB_BITS as usize)
}

/// Exclusive upper edge of a [`LatencyHistogram`] sub-bin.
///
/// The clamp region (`index < 8`) returns the exact value `index` (0 % error);
/// every populated log-linear bin returns its true exclusive upper edge
/// `2^msb + (sub + 1)·2^(msb-3)`, guaranteeing a relative error of at most
/// 1/8 (12.5 %) against any sample it holds.
#[inline]
fn latency_bin_upper_edge(index: usize) -> u64 {
    if index < SUB_COUNT {
        return index as u64;
    }
    let msb = index >> SUB_BITS;
    if msb < SUB_BITS as usize {
        return index as u64;
    }
    let sub = (index & (SUB_COUNT - 1)) as u64;
    (SUB_COUNT as u64 + 1 + sub) << (msb - SUB_BITS as usize)
}

/// Exponential histogram for tracking the latency distribution of the RT callback.
///
/// Has 32 bins covering powers of 2 from 2^5 ns (~32 ns) to 2^36 ns (~68 s).
/// Uses atomic operations to ensure RT-safety (lock-free, zero-allocation).
///
/// Kept alongside [`LatencyHistogram`] for consumers that rely on the coarse
/// power-of-two bucket edges; behavior is frozen.
#[repr(align(128))]
pub struct AtomicHistogram {
    /// Histogram atomic bins.
    bins: [AtomicU64; 32],
    /// Exact maximum latency observed (lock-free, via fetch_max).
    exact_max: AtomicU64,
    /// Exact minimum latency observed (lock-free, via fetch_min).
    exact_min: AtomicU64,
    /// Sum of all observations in nanoseconds for computing mean.
    sum_ns: AtomicU64,
}

impl Default for AtomicHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicHistogram {
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
    ///
    /// Uses rank `ceil(N * p)` with minimum 1 over the total observations.
    /// Returns the nanoseconds value of the matched bin's true upper edge:
    /// - Bin 0: [0, 64) ns -> upper edge 64 ns (`1 << 6`)
    /// - Bin i (1 <= i <= 30): [2^(i+5), 2^(i+6)) ns -> upper edge 2^(i+6) ns (`1 << (i + 6)`)
    /// - Bin 31: [2^36, ∞) ns -> upper edge 2^37 ns (`1 << 37`)
    ///
    /// An empty histogram, or a percentile `p` outside `[0.0, 1.0]` (or NaN),
    /// safely returns the documented sentinel `0` without panic.
    pub fn get_percentile(&self, p: f32) -> u64 {
        self.get_percentile_opt(p).unwrap_or(0)
    }

    /// Computes the approximate value of a percentile as an `Option<u64>`.
    ///
    /// Returns `None` when empty or when `p` is outside `[0.0, 1.0]`.
    pub fn get_percentile_opt(&self, p: f32) -> Option<u64> {
        if !(0.0..=1.0).contains(&p) || p.is_nan() {
            return None;
        }

        let counts: [u64; 32] = core::array::from_fn(|i| self.bins[i].load(Ordering::Relaxed));
        let total: u64 = counts.iter().sum();

        if total == 0 {
            return None;
        }

        let rank = ((total as f64 * p as f64).ceil() as u64).max(1);
        let mut accum: u64 = 0;

        for (i, &count) in counts.iter().enumerate() {
            accum += count;
            if accum >= rank {
                // Returns real upper edge 2^(i + 6)
                return Some(1u64 << (i + 6));
            }
        }

        Some(1u64 << 37)
    }

    /// Returns the maximum observed value (real upper edge of the highest non-empty bin).
    /// Returns 0 when empty.
    pub fn get_max(&self) -> u64 {
        for i in (0..32).rev() {
            if self.bins[i].load(Ordering::Relaxed) > 0 {
                return 1u64 << (i + 6);
            }
        }
        0
    }

    /// Returns the minimum observed value (real lower edge of the lowest non-empty bin).
    /// Returns 0 when empty or when the lowest non-empty bin is bin 0 ([0, 64) ns).
    pub fn get_min(&self) -> u64 {
        for i in 0..32 {
            if self.bins[i].load(Ordering::Relaxed) > 0 {
                return if i == 0 { 0 } else { 1u64 << (i + 5) };
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

/// Log-linear atomic latency histogram (HdrHistogram style) with a bounded error.
///
/// 256 `AtomicU64` counters (2 KiB) partitioned into 32 octaves × 8 sub-bins.
/// The sub-bin is derived from the three bits below the leading `1`
/// (`index = (msb << 3) | ((v >> (msb-3)) & 7)`), so each recorded sample is
/// resolved to a value whose relative error over the reported upper edge is at
/// most 12.5 % — versus the up-to-100 % error of the power-of-two
/// [`AtomicHistogram`]. The hot path cost is one `lzcnt` + `shr` + `fetch_add`,
/// identical in spirit to the octave histogram.
///
/// `record` is lock-free and zero-allocation. Percentile reads snapshot the
/// 256 counters off-RT (never from the audio callback) and follow the same
/// rank/`None`/NaN semantics as [`AtomicHistogram`].
#[repr(align(128))]
pub struct LatencyHistogram {
    /// Histogram atomic sub-bins (32 octaves × 8 sub-bins).
    bins: [AtomicU64; LATENCY_BIN_COUNT],
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
            bins: [const { AtomicU64::new(0) }; LATENCY_BIN_COUNT],
            exact_max: AtomicU64::new(0),
            exact_min: AtomicU64::new(u64::MAX),
            sum_ns: AtomicU64::new(0),
        }
    }

    /// Records a latency observation in nanoseconds.
    ///
    /// # RT-Safety
    /// Lock-free and zero-allocation: safe for the DSP hot-path.
    #[inline(always)]
    pub fn record(&self, duration_ns: u64) {
        self.exact_max.fetch_max(duration_ns, Ordering::Relaxed);
        self.exact_min.fetch_min(duration_ns, Ordering::Relaxed);
        self.sum_ns.fetch_add(duration_ns, Ordering::Relaxed);
        let index = latency_bin_index(duration_ns);
        // `index < LATENCY_BIN_COUNT` by construction; `get` keeps the hot path panic-free.
        if let Some(bin) = self.bins.get(index) {
            bin.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Computes the approximate value of a percentile (e.g., 0.99 for P99).
    ///
    /// Uses rank `ceil(N * p)` with minimum 1 over the total observations and
    /// returns the true exclusive upper edge of the matched sub-bin. An empty
    /// histogram, or `p` outside `[0.0, 1.0]` (or NaN), safely returns `0`.
    pub fn get_percentile(&self, p: f32) -> u64 {
        self.get_percentile_opt(p).unwrap_or(0)
    }

    /// Computes the approximate value of a percentile as an `Option<u64>`.
    ///
    /// Returns `None` when empty or when `p` is outside `[0.0, 1.0]`.
    pub fn get_percentile_opt(&self, p: f32) -> Option<u64> {
        if !(0.0..=1.0).contains(&p) || p.is_nan() {
            return None;
        }

        let counts: [u64; LATENCY_BIN_COUNT] =
            core::array::from_fn(|i| self.bins[i].load(Ordering::Relaxed));
        let total: u64 = counts.iter().sum();

        if total == 0 {
            return None;
        }

        let rank = ((total as f64 * p as f64).ceil() as u64).max(1);
        let mut accum: u64 = 0;

        for (i, &count) in counts.iter().enumerate() {
            accum += count;
            if accum >= rank {
                return Some(latency_bin_upper_edge(i));
            }
        }

        Some(latency_bin_upper_edge(LATENCY_BIN_COUNT - 1))
    }

    /// Returns the upper edge of the highest non-empty sub-bin; 0 when empty.
    pub fn get_max(&self) -> u64 {
        for i in (0..LATENCY_BIN_COUNT).rev() {
            if self.bins[i].load(Ordering::Relaxed) > 0 {
                return latency_bin_upper_edge(i);
            }
        }
        0
    }

    /// Returns the lower edge of the lowest non-empty sub-bin; 0 when empty.
    pub fn get_min(&self) -> u64 {
        for i in 0..LATENCY_BIN_COUNT {
            if self.bins[i].load(Ordering::Relaxed) > 0 {
                return latency_bin_lower_edge(i);
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

    /// Zeros all sub-bins and resets statistics.
    ///
    /// `swap` (RMW) guarantees visibility via cache-coherence. Best-effort
    /// reset: concurrent `fetch_add` records may be lost during the sweep;
    /// `record` calls are lock-free and never blocked.
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

/// Complementary exact-sample reservoir for trend reports.
///
/// Stores up to `N` raw latency samples (ring overwrite) so exact
/// percentiles (`p50`/`p90`/`p99`/`p99.9` via sorting) can be compared
/// against the [`LatencyHistogram`] bucket edges. Off-RT only: `record`
/// performs heap-free index arithmetic but `percentile_exact` sorts a
/// snapshot — never call it on the audio thread.
///
/// Compiled only with `testing`/`heap-audit` (never in the default or
/// production build) so the shipped codegen stays byte-identical.
///
/// Note: unit tests always compile with `cfg(test)`, so the reservoir tests
/// below exercise the reservoir in every `cargo test` invocation — but the
/// type itself is absent from non-`testing` library builds.
#[cfg(any(test, feature = "testing", feature = "heap-audit"))]
#[cfg_attr(
    docsrs,
    doc(cfg(any(test, feature = "testing", feature = "heap-audit")))
)]
#[repr(align(128))]
pub struct ExactLatencyReservoir<const N: usize = 2048> {
    /// Ring buffer of exact samples in nanoseconds.
    samples: [AtomicU64; N],
    /// Monotonic write counter (also the filled-count source via `min`).
    count: AtomicU64,
}

#[cfg(any(test, feature = "testing", feature = "heap-audit"))]
#[cfg_attr(
    docsrs,
    doc(cfg(any(test, feature = "testing", feature = "heap-audit")))
)]
impl<const N: usize> ExactLatencyReservoir<N> {
    /// Creates a new empty reservoir.
    #[cold]
    pub fn new() -> Self {
        Self {
            samples: [const { AtomicU64::new(0) }; N],
            count: AtomicU64::new(0),
        }
    }

    /// Records one exact latency sample (off-RT trend path only).
    ///
    /// Lock-free ring overwrite: the slot is `count % N`. Concurrent
    /// overwrites are benign (trend telemetry, not the binary gate).
    pub fn record(&self, duration_ns: u64) {
        let slot = self.count.fetch_add(1, Ordering::Relaxed) as usize % N;
        // SAFETY: `slot < N` by construction (`% N`); single-slot store.
        if let Some(cell) = self.samples.get(slot) {
            cell.store(duration_ns, Ordering::Relaxed);
        }
    }

    /// Number of valid samples currently held (`min(count, N)`).
    pub fn len(&self) -> usize {
        (self.count.load(Ordering::Relaxed) as usize).min(N)
    }

    /// Whether no sample has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Exact percentile of the stored samples (`p` in `[0.0, 1.0]`).
    ///
    /// Sorts an off-RT snapshot — never call on the audio thread. Returns
    /// `0` when empty; clamps `p` to `[0.0, 1.0]`.
    pub fn percentile_exact(&self, p: f64) -> u64 {
        let len = self.len();
        if len == 0 || N == 0 {
            return 0;
        }
        let mut snapshot: [u64; N] =
            core::array::from_fn(|i| self.samples[i].load(Ordering::Relaxed));
        snapshot[..len].sort_unstable();
        let clamped = p.clamp(0.0, 1.0);
        let rank = ((len as f64 * clamped) as usize).min(len).saturating_sub(1);
        snapshot[rank]
    }

    /// Zeros all slots and the write counter (off-RT only).
    pub fn reset(&self) {
        for cell in &self.samples {
            cell.store(0, Ordering::Relaxed);
        }
        self.count.store(0, Ordering::Relaxed);
    }
}

#[cfg(any(test, feature = "testing", feature = "heap-audit"))]
#[cfg_attr(
    docsrs,
    doc(cfg(any(test, feature = "testing", feature = "heap-audit")))
)]
impl<const N: usize> Default for ExactLatencyReservoir<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_mapping() {
        // Organization Test: Verifies that the system stores processing times
        // in the correct "drawers" (bins), separating ultra-fast from slow.
        let hist = AtomicHistogram::new();

        hist.record(10); // bin 0 (0..64 ns)
        hist.record(40); // bin 0 (0..64 ns)
        hist.record(100); // bin 1 (64..128 ns)
        hist.record(1000); // bin 4 (512..1024 ns)

        assert_eq!(hist.bins[0].load(Ordering::Relaxed), 2);
        assert_eq!(hist.bins[1].load(Ordering::Relaxed), 1);
        assert_eq!(hist.bins[4].load(Ordering::Relaxed), 1);
        assert_eq!(hist.get_min(), 0); // bin 0 lower edge is 0
        assert_eq!(hist.get_max(), 1024); // bin 4 upper edge is 1024
        assert_eq!(hist.get_exact_min(), 10);
        assert_eq!(hist.get_exact_max(), 1000);
        assert_eq!(hist.get_mean(), (10 + 40 + 100 + 1000) / 4);
    }

    #[test]
    fn test_percentiles() {
        // Statistics Test: Verifies that the system can correctly identify
        // the "median" (P50) and "worst cases" (P95 and P99) in a batch of measurements.
        let hist = AtomicHistogram::new();

        // 100 samples
        for _ in 0..50 {
            hist.record(100);
        } // P50 in bin 1 (upper edge 128)
        for _ in 0..45 {
            hist.record(1000);
        } // P95 in bin 4 (upper edge 1024)
        for _ in 0..5 {
            hist.record(10000);
        } // P99 in bin 8 (upper edge 16384)

        assert_eq!(hist.get_percentile(0.50), 128);
        assert_eq!(hist.get_percentile(0.95), 1024);
        assert_eq!(hist.get_percentile(0.99), 16384);
        assert_eq!(hist.get_exact_min(), 100);
        assert_eq!(hist.get_exact_max(), 10000);
    }

    #[test]
    fn test_histogram_edge_cases() {
        let hist = AtomicHistogram::new();
        // Empty population
        assert_eq!(hist.get_percentile(0.50), 0);
        assert_eq!(hist.get_percentile_opt(0.50), None);
        assert_eq!(hist.get_min(), 0);
        assert_eq!(hist.get_max(), 0);

        // Invalid p
        hist.record(64);
        assert_eq!(hist.get_percentile(-0.1), 0);
        assert_eq!(hist.get_percentile(1.1), 0);
        assert_eq!(hist.get_percentile(f32::NAN), 0);
        assert_eq!(hist.get_percentile_opt(-0.1), None);
        assert_eq!(hist.get_percentile_opt(1.1), None);
        assert_eq!(hist.get_percentile_opt(f32::NAN), None);

        // Single sample population: rank ceil(1 * p) with min 1 => rank 1 for any p in [0, 1]
        assert_eq!(hist.get_percentile(0.0), 128); // 64 ns is in bin 1 (64..128)
        assert_eq!(hist.get_percentile(1.0), 128);
        assert_eq!(hist.get_min(), 64);
        assert_eq!(hist.get_max(), 128);

        // Boundary values: 0, 31, 32, 33, 63, 64, 1000 ns
        let h2 = AtomicHistogram::new();
        for val in [0u64, 31, 32, 33, 63] {
            h2.record(val);
        }
        // All [0, 63] belong to bin 0 [0, 64)
        assert_eq!(h2.bins[0].load(Ordering::Relaxed), 5);
        assert_eq!(h2.get_min(), 0);
        assert_eq!(h2.get_max(), 64);

        h2.record(64); // bin 1 [64, 128)
        assert_eq!(h2.bins[1].load(Ordering::Relaxed), 1);
        assert_eq!(h2.get_max(), 128);

        h2.record(1000); // bin 4 [512, 1024)
        assert_eq!(h2.bins[4].load(Ordering::Relaxed), 1);
        assert_eq!(h2.get_max(), 1024);
    }

    /// Exact percentiles resolve sub-bucket drift the log2 buckets hide.
    /// Unit tests compile with `cfg(test)`, so this runs in every suite.
    #[test]
    fn test_exact_reservoir_resolves_sub_bucket_drift() {
        let hist = AtomicHistogram::new();
        let exact = ExactLatencyReservoir::<2048>::new();
        for value in [700u64, 750, 800, 950, 1050] {
            for _ in 0..400 {
                hist.record(value);
                exact.record(value);
            }
        }
        // Both bucket populations collapse toward neighboring edges while the
        // exact p50 tracks the true median sample.
        assert_eq!(exact.percentile_exact(0.50), 800);
        assert_eq!(exact.percentile_exact(0.80), 950);
        assert_eq!(exact.percentile_exact(0.99), 1050);
        assert!(
            hist.get_percentile(0.50) != exact.percentile_exact(0.50),
            "bucket edge must differ from the exact median for this mix"
        );
        assert_eq!(exact.len(), 2000);
        assert!(!exact.is_empty());
        exact.reset();
        assert!(exact.is_empty());
        assert_eq!(exact.percentile_exact(0.99), 0);
    }

    /// Ring overwrite keeps the newest `N` samples; empty reads zero.
    #[test]
    fn test_exact_reservoir_ring_overwrite_and_empty() {
        let exact = ExactLatencyReservoir::<8>::new();
        assert!(exact.is_empty());
        assert_eq!(exact.percentile_exact(0.50), 0);
        for value in 1u64..=10 {
            exact.record(value * 100);
        }
        assert_eq!(exact.len(), 8);
        // Newest 8 of 1..=10 (×100): 300..=1000.
        assert_eq!(exact.percentile_exact(0.0), 300);
        assert_eq!(exact.percentile_exact(1.0), 1000);
        assert_eq!(exact.percentile_exact(0.50), 600);
    }

    // ── Log-linear LatencyHistogram (U-01 / F-PR-07) ─────────────────────────

    #[test]
    fn test_loglinear_mapping_is_monotone_and_resolved() {
        // Values below the log region map to exact bins; beyond it, to the
        // 8-sub-bin scheme. The reported upper edge is never below the sample.
        for value in [0u64, 1, 7, 8, 15, 16, 255, 256, 3000, 1_000_000] {
            let upper = latency_bin_upper_edge(latency_bin_index(value));
            assert!(
                upper >= value,
                "upper edge {upper} must bound sample {value}"
            );
        }
        // A non-power-of-two sample resolves to a non-power-of-two edge.
        let hist = LatencyHistogram::new();
        for _ in 0..1000 {
            hist.record(3000);
        }
        assert_eq!(hist.get_percentile(0.50), 3072);
        assert_eq!(hist.get_percentile(1.0), 3072);
        assert_eq!(hist.get_exact_max(), 3000);
    }

    #[test]
    fn test_loglinear_distinguishes_p50_from_p99() {
        let hist = LatencyHistogram::new();
        for _ in 0..980 {
            hist.record(1000); // 1 µs cluster (p50)
        }
        for _ in 0..20 {
            hist.record(5000); // 5 µs tail (p99)
        }
        let p50 = hist.get_percentile(0.50);
        let p99 = hist.get_percentile(0.99);
        assert!(p50 < p99, "p50 {p50} must be below p99 {p99}");
        assert_eq!(p50, 1024); // 1000 ∈ [896, 1024)
        assert_eq!(p99, 5120); // 5000 ∈ [4608, 5120)
        assert_eq!(hist.total_count(), 1000);
    }

    #[test]
    fn test_loglinear_empty_invalid_and_reset() {
        let hist = LatencyHistogram::new();
        assert_eq!(hist.get_percentile(0.50), 0);
        assert_eq!(hist.get_percentile_opt(0.50), None);
        assert_eq!(hist.get_max(), 0);
        assert_eq!(hist.get_min(), 0);
        assert_eq!(hist.total_count(), 0);

        hist.record(1234);
        assert_eq!(hist.get_percentile(-0.1), 0);
        assert_eq!(hist.get_percentile(1.1), 0);
        assert_eq!(hist.get_percentile(f32::NAN), 0);
        assert_eq!(hist.get_percentile_opt(-0.1), None);
        assert_eq!(hist.get_percentile_opt(1.1), None);
        assert_eq!(hist.get_percentile_opt(f32::NAN), None);

        hist.reset();
        assert_eq!(hist.total_count(), 0);
        assert_eq!(hist.get_percentile(0.99), 0);
        assert_eq!(hist.get_exact_max(), 0);
        assert_eq!(hist.get_exact_min(), 0);
    }

    #[test]
    fn test_loglinear_percentile_monotonic() {
        let hist = LatencyHistogram::new();
        for value in 1u64..=2048 {
            hist.record(value);
        }
        let mut last = 0u64;
        for p in [0.0f32, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0] {
            let v = hist.get_percentile(p);
            assert!(v >= last, "percentile p={p} regressed: {v} < {last}");
            last = v;
        }
    }

    mod loglinear_properties {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// The reported sub-bin edge never undershoots a sample and keeps
            /// the relative error within the documented 12.5 % bound.
            #[test]
            fn relative_error_within_twelfth(value in 1u64..=(1u64 << 31)) {
                let edge = latency_bin_upper_edge(latency_bin_index(value));
                prop_assert!(edge >= value, "edge {edge} < sample {value}");
                // edge <= value * 9 / 8  ⇔  8·edge <= 9·value (integer, overflow-safe via u128).
                prop_assert!(
                    (edge as u128) * 8 <= (value as u128) * 9,
                    "relative error > 12.5%: edge {edge} vs sample {value}"
                );
            }

            /// Reachable log-region bins are mapped monotonically: a larger
            /// index never reports a smaller upper edge (clamp bins `0..8` are
            /// exact and below the first log bin).
            #[test]
            fn edges_are_monotone(i in (3 * SUB_COUNT)..LATENCY_BIN_COUNT, j in (3 * SUB_COUNT)..LATENCY_BIN_COUNT) {
                let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                prop_assert!(latency_bin_upper_edge(lo) <= latency_bin_upper_edge(hi));
                prop_assert!(
                    latency_bin_upper_edge(SUB_COUNT - 1) <= latency_bin_upper_edge(3 * SUB_COUNT)
                );
            }
        }
    }
}
