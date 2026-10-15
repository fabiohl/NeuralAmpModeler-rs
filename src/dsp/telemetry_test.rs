// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

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
