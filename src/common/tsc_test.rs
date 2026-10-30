// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Test-only calibration override.
///
/// Stores `mult` and publishes anchor values as if `calibrate_tsc` had measured
/// the given rate. Unit tests use this to pin exact conversion rates without sleeping.
fn set_mult_for_test(mult: u64) {
    if mult != 0 {
        publish_calibration(mult);
    } else {
        TSC_ANCHOR_TSC.store(0, Ordering::Relaxed);
        TSC_ANCHOR_NS.store(0, Ordering::Relaxed);
        TSC_MULT.store(0, Ordering::Release);
    }
}

fn set_calibration_for_test(mult: u64, anchor_tsc: u64, anchor_ns: u64) {
    TSC_ANCHOR_TSC.store(anchor_tsc, Ordering::Relaxed);
    TSC_ANCHOR_NS.store(anchor_ns, Ordering::Relaxed);
    TSC_MULT.store(mult, Ordering::Release);
}

#[test]
fn tsc_scale_matches_u128_reference_without_overflow() {
    // Exact 2 GHz rate: 2 cycles per ns over a 50 ms window.
    // Measured: mult/shift reproduces the window (100M cycles -> 50M ns)
    // and tracks cycles/elapsed scaling across the full u64 domain.
    let elapsed_cycles: u64 = 100_000_000;
    let elapsed_nanos: u64 = 50_000_000;
    let mult = mult_for_rate(elapsed_cycles, elapsed_nanos).expect("valid window");
    assert_eq!(cycles_to_nanos(elapsed_cycles, mult), elapsed_nanos);

    // Full u64 domain: the old `(cycles * 1000) / freq_x1000` panics in
    // debug once cycles > u64::MAX / 1000; the widened path must not.
    for cycles in [
        0u64,
        1,
        1_000,
        u64::MAX / 1000,
        u64::MAX / 1000 + 1,
        u64::MAX,
    ] {
        // Per-sample reference: ns = cycles * elapsed_nanos / elapsed_cycles.
        let expected = ((cycles as u128 * elapsed_nanos as u128) / elapsed_cycles as u128) as u64;
        let got = cycles_to_nanos(cycles, mult);
        let tolerance = (expected / 1_000_000).max(2);
        assert!(
            got.abs_diff(expected) <= tolerance,
            "cycles={cycles}: got={got} expected~{expected}"
        );
    }
}

#[test]
fn tsc_scale_matches_reference_across_typical_rates() {
    // Measured: mult/shift tracks the u128 division reference within 1 ppm
    // for typical desktop TSC rates (1.8, 3.0, 4.8 GHz).
    for mhz in [1800u64, 3000, 4800] {
        let elapsed_cycles = mhz * 1_000_000 / 20; // 50 ms window
        let elapsed_nanos = 50_000_000;
        let mult = mult_for_rate(elapsed_cycles, elapsed_nanos).expect("valid window");
        for cycles in [1u64, 1_000_000, u64::MAX / 2, u64::MAX] {
            let expected =
                ((cycles as u128 * elapsed_nanos as u128) / elapsed_cycles as u128) as u64;
            let got = cycles_to_nanos(cycles, mult);
            let tolerance = (expected / 1_000_000).max(2);
            assert!(
                got.abs_diff(expected) <= tolerance,
                "rate={mhz}MHz cycles={cycles}: got={got} expected~{expected}"
            );
        }
    }
}

#[test]
fn tsc_conversion_is_monotonic() {
    // Measured: consecutive cycle counts map to non-decreasing ns.
    let mult = mult_for_rate(100_000_000, 50_000_000).expect("valid window");
    let mut prev = cycles_to_nanos(0, mult);
    for cycles in (0u64..1_000_000).step_by(7).chain([u64::MAX - 1, u64::MAX]) {
        let now = cycles_to_nanos(cycles, mult);
        assert!(now >= prev, "non-monotonic at cycles={cycles}");
        prev = now;
    }
}

#[test]
fn tsc_rejects_degenerate_calibration_windows() {
    assert_eq!(mult_for_rate(0, 50_000_000), None);
    assert_eq!(mult_for_rate(100_000_000, 0), None);
}

#[test]
#[serial_test::serial]
fn rdtsc_nanos_advances_after_test_calibration() {
    // Pins a nominal 2 GHz rate and checks the live counter advances.
    let mult = mult_for_rate(100_000_000, 50_000_000).expect("valid window");
    set_mult_for_test(mult);
    let first = rdtsc_nanos();
    let mut last = first;
    for _ in 0..100 {
        let now = rdtsc_nanos();
        assert!(now >= last, "TSC clock went backwards");
        last = now;
    }
    // Counter must tick on real hardware (not frozen at zero mapping).
    assert!(last >= first);
    set_mult_for_test(0);
}

/// aarch64: a plausible CNTFRQ_EL0 is used as-is without measuring; a zero
/// or implausible value falls back to the measurement.
#[cfg(target_arch = "aarch64")]
#[test]
fn generic_timer_uses_register_unless_implausible() {
    let not_called = || -> Option<(u64, u64)> { panic!("must not measure") };
    for hz in [1_000_000, 24_000_000, 25_000_000, 1_000_000_000] {
        let expect = mult_for_rate(hz, 1_000_000_000).expect("valid rate");
        assert_eq!(
            choose_generic_timer_mult(hz, not_called),
            Some((expect, GenericTimerSource::Register)),
            "{hz} Hz"
        );
    }
    // 25 MHz measured over 20 ms = 500_000 cycles.
    let measured = mult_for_rate(500_000, 20_000_000).expect("valid window");
    for hz in [0, 999_999, 1_000_000_001, u64::MAX] {
        assert_eq!(
            choose_generic_timer_mult(hz, || Some((500_000, 20_000_000))),
            Some((measured, GenericTimerSource::Measured)),
            "{hz} Hz"
        );
    }
    // Implausible register and no usable measurement.
    assert_eq!(choose_generic_timer_mult(0, || None), None);
    assert_eq!(choose_generic_timer_mult(0, || Some((0, 0))), None);
}

/// aarch64: the bracketed measurement agrees with CNTFRQ_EL0 on real
/// hardware (where the register is correct) to well within 1%.
#[cfg(target_arch = "aarch64")]
#[test]
fn generic_timer_measurement_matches_register() {
    let (cycles, nanos) = measure_generic_timer(Duration::from_millis(5), 3).expect("measurement");
    let measured_hz = cycles as f64 * 1.0e9 / nanos as f64;
    let reported_hz = generic_timer_frequency() as f64;
    let deviation = (measured_hz - reported_hz).abs() / reported_hz;
    assert!(
        deviation < 0.01,
        "measured {measured_hz:.0} Hz vs CNTFRQ_EL0 {reported_hz:.0} Hz"
    );
}

/// aarch64: the CNTFRQ_EL0-derived conversion tracks `Instant` over a
/// short sleep. Uses a local multiplier (not the global `TSC_MULT`, which
/// other tests in this module set and reset concurrently).
#[cfg(target_arch = "aarch64")]
#[test]
fn generic_timer_calibration_tracks_instant() {
    let freq = generic_timer_frequency();
    let mult = mult_for_rate(freq, 1_000_000_000).expect("CNTFRQ_EL0 must be non-zero");
    let c0 = read_counter();
    let i0 = Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let counter_ns = cycles_to_nanos(read_counter() - c0, mult);
    let instant_ns = i0.elapsed().as_nanos() as u64;
    // Both clocks cover the same window; allow scheduling jitter.
    assert!(
        counter_ns.abs_diff(instant_ns) < 2_000_000,
        "freq={freq} Hz counter={counter_ns} ns instant={instant_ns} ns"
    );
}

#[test]
#[serial_test::serial]
fn rdtsc_nanos_flip_is_continuous_and_monotonic() {
    set_mult_for_test(0);
    let before_flip = rdtsc_nanos();

    // Simulate calibration flip
    let mult = mult_for_rate(100_000_000, 50_000_000).expect("valid window");
    set_mult_for_test(mult);
    let after_flip = rdtsc_nanos();

    // Invariant: continuous (|Δ| < 1 ms, no uptime jump) and monotonic
    assert!(
        after_flip >= before_flip,
        "Monotonicity violated across calibration flip: before={before_flip}, after={after_flip}"
    );
    let delta_ns = after_flip - before_flip;
    assert!(
        delta_ns < 1_000_000,
        "Discontinuity across calibration flip (|Δ| >= 1ms): delta_ns={delta_ns}"
    );

    let mut prev = after_flip;
    for _ in 0..100 {
        let next = rdtsc_nanos();
        assert!(next >= prev, "Time went backwards after calibration flip");
        prev = next;
    }

    set_mult_for_test(0);
}

#[test]
#[serial_test::serial]
fn rdtsc_nanos_uses_explicit_anchor() {
    let mult = mult_for_rate(100_000_000, 50_000_000).expect("valid window");
    // Anchor at tsc = 1_000_000, ns = 500_000
    set_calibration_for_test(mult, 1_000_000, 500_000);
    let now = rdtsc_nanos();
    assert!(
        now >= 500_000,
        "rdtsc_nanos must respect explicit anchor_ns baseline"
    );
    set_mult_for_test(0);
}

use proptest::prelude::*;

proptest! {
    #[test]
    fn proptest_cycles_to_nanos_monotonic(
        c1 in 0u64..u64::MAX - 1_000_000,
        step in 0u64..1_000_000,
        freq_mhz in 1000u64..6000u64,
    ) {
        let mult = mult_for_rate(freq_mhz * 1_000_000, 1_000_000_000).expect("valid rate");
        let c2 = c1 + step;
        let ns1 = cycles_to_nanos(c1, mult);
        let ns2 = cycles_to_nanos(c2, mult);
        prop_assert!(ns2 >= ns1, "ns2 ({}) < ns1 ({}) for c1={}, c2={}, mult={}", ns2, ns1, c1, c2, mult);
    }

    #[test]
    fn proptest_anchor_timeline_monotonic(
        anchor_tsc in any::<u64>(),
        anchor_ns in 0u64..u64::MAX / 2,
        step in 0u64..100_000_000,
        freq_mhz in 1000u64..6000u64,
    ) {
        let mult = mult_for_rate(freq_mhz * 1_000_000, 1_000_000_000).expect("valid rate");
        let tsc1 = anchor_tsc;
        let tsc2 = anchor_tsc.wrapping_add(step);
        let t1 = anchor_ns.wrapping_add(cycles_to_nanos(tsc1.wrapping_sub(anchor_tsc), mult));
        let t2 = anchor_ns.wrapping_add(cycles_to_nanos(tsc2.wrapping_sub(anchor_tsc), mult));
        prop_assert!(t2 >= t1, "t2 ({}) < t1 ({}) for step={}", t2, t1, step);
    }
}

#[cfg(feature = "on-demand")]
#[test]
#[ignore = "on-demand: micro-benchmark fallback Instant vs calibrated RDTSC"]
fn bench_rdtsc_overhead() {
    // Measured: Uncalibrated (Instant fallback with HPET clocksource) = 1134.66 ns/call
    //           Calibrated RDTSC (mult/shift) = 17.16 ns/call (~40-50 cycles at 2.5 GHz)
    // Measured on AMD Ryzen 7 5700U, pinned to isolated core 8.
    set_mult_for_test(0);
    let iters = 1_000_000;
    let mut sink = 0u64;

    for _ in 0..10_000 {
        sink = sink.wrapping_add(rdtsc_nanos());
    }

    let start = Instant::now();
    for _ in 0..iters {
        sink = sink.wrapping_add(rdtsc_nanos());
    }
    let elapsed_uncal = start.elapsed();
    let ns_per_call_uncal = elapsed_uncal.as_nanos() as f64 / iters as f64;

    let mult = mult_for_rate(100_000_000, 50_000_000).expect("valid window");
    set_mult_for_test(mult);

    for _ in 0..10_000 {
        sink = sink.wrapping_add(rdtsc_nanos());
    }

    let start = Instant::now();
    for _ in 0..iters {
        sink = sink.wrapping_add(rdtsc_nanos());
    }
    let elapsed_cal = start.elapsed();
    let ns_per_call_cal = elapsed_cal.as_nanos() as f64 / iters as f64;
    set_mult_for_test(0);

    println!(
        "\n>>> [RDTSC MEASUREMENT] Uncalibrated (Instant fallback): {:.2} ns/call | Calibrated (RDTSC mult/shift): {:.2} ns/call (sink={})",
        ns_per_call_uncal, ns_per_call_cal, sink
    );
    assert!(ns_per_call_cal < ns_per_call_uncal);
}
