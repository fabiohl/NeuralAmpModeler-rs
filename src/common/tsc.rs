// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Cycle-counter clock: Time Stamp Counter (RDTSC) on x86_64, the
//! architectural generic timer (`CNTVCT_EL0`) on aarch64.
//!
//! Provides low-cost time measurement (~1 ns resolution on x86_64; the
//! generic-timer tick, e.g. ~42 ns at 24 MHz on Apple Silicon, on aarch64),
//! avoiding the vDSO clock_gettime syscall in the audio hot-path.
//!
//! Conversion uses fixed-point `mult/shift` scaling in the style of the
//! Linux kernel `cyc2ns` (`ns = (cycles * mult) >> shift`), computed once
//! at startup. The hot-path performs no division and no `cycles * 1000`,
//! so large counter values cannot overflow in debug builds.
//!
//! Portability contract: x86_64 reads RDTSC (serialized with LFENCE) and
//! calibrates against the system clock; aarch64 reads `CNTVCT_EL0`
//! (serialized with ISB) and calibrates exactly from `CNTFRQ_EL0`. The public
//! names (`rdtsc_nanos`, `calibrate_tsc`) are kept on both architectures so
//! host code is portable. The `Instant` fallback covers the uncalibrated case.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Fixed shift for the `cyc2ns`-style conversion.
///
/// `32` keeps the multiplier in 32-bit range for typical TSC rates
/// (1–6 GHz) while holding relative error below ~2.4e-10.
const TSC_SHIFT: u32 = 32;
/// Calibrated multiplier for [`cycles_to_nanos`].
///
/// Zero means uncalibrated; the hot-path then uses the `Instant` fallback.
static TSC_MULT: AtomicU64 = AtomicU64::new(0);
/// Hardware TSC cycle count captured at calibration time.
///
/// Published before [`TSC_MULT`] (via Release ordering on `TSC_MULT`) to
/// anchor continuous monotonic time across the calibration transition.
static TSC_ANCHOR_TSC: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds elapsed since [`BOOT_TIME`] captured at calibration time.
///
/// Published before [`TSC_MULT`] alongside [`TSC_ANCHOR_TSC`].
static TSC_ANCHOR_NS: AtomicU64 = AtomicU64::new(0);
/// Time anchor for rdtsc fallback (monotonic).
static BOOT_TIME: OnceLock<Instant> = OnceLock::new();

/// Helper to read `CLOCK_MONOTONIC_RAW` on Linux for startup calibration validation.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn monotonic_raw_nanos() -> Option<u64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, stack-allocated `timespec` passed by mutable reference.
    // `CLOCK_MONOTONIC_RAW` is a valid POSIX clock ID on Linux. The call returns 0 on
    // success and -1 on error; we check the return value before reading `ts`.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut ts) } == 0 {
        Some((ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64))
    } else {
        None
    }
}

#[cfg(all(target_arch = "x86_64", not(target_os = "linux")))]
fn monotonic_raw_nanos() -> Option<u64> {
    None
}

/// Converts raw TSC cycles to nanoseconds via the calibrated `mult/shift`.
///
/// Overflow-free by construction: the product is widened to `u128` before
/// the shift, so the full `u64` cycle domain maps without wrapping.
#[inline(always)]
fn cycles_to_nanos(cycles: u64, mult: u64) -> u64 {
    (((cycles as u128) * (mult as u128)) >> TSC_SHIFT) as u64
}

/// Computes the calibration multiplier for a measured rate.
///
/// `mult = ns_per_cycle * 2^SHIFT = elapsed_nanos * 2^SHIFT / elapsed_cycles`.
/// Pure helper (no atomics) so unit tests can pin exact rates.
fn mult_for_rate(elapsed_cycles: u64, elapsed_nanos: u64) -> Option<u64> {
    if elapsed_cycles == 0 || elapsed_nanos == 0 {
        return None;
    }
    let mult = (((elapsed_nanos as u128) << TSC_SHIFT) / (elapsed_cycles as u128)) as u64;
    // A zero multiplier would freeze the clock (every sample maps to 0 ns).
    if mult == 0 { None } else { Some(mult) }
}

/// Publishes the conversion rate `mult` together with its continuity anchor
/// (the counter and the `BOOT_TIME`-relative nanoseconds now), so that
/// `rdtsc_nanos()` continues from the fallback clock without a jump. The
/// anchor is stored before `mult` (Release) for `rdtsc_nanos`' Acquire load.
fn publish_calibration(mult: u64) {
    let anchor_ns = BOOT_TIME.get_or_init(Instant::now).elapsed().as_nanos() as u64;
    let anchor_tsc = read_counter();
    TSC_ANCHOR_TSC.store(anchor_tsc, Ordering::Relaxed);
    TSC_ANCHOR_NS.store(anchor_ns, Ordering::Relaxed);
    TSC_MULT.store(mult, Ordering::Release);
}

/// Reads the raw cycle counter, serialized against earlier instructions.
///
/// x86_64: `LFENCE; RDTSC`. aarch64: `ISB; MRS CNTVCT_EL0`.
#[inline(always)]
fn read_counter() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `_mm_lfence` + `_rdtsc` is available on all x86-64 CPUs; it performs no
        // memory access and has no side effects, so reading it here is sound.
        unsafe {
            core::arch::x86_64::_mm_lfence();
            core::arch::x86_64::_rdtsc()
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let cycles: u64;
        // SAFETY: `isb` and `mrs cntvct_el0` only read a system register; no memory is
        // accessed (`nomem`, `nostack`). EL0 access to CNTVCT_EL0 is granted by the kernel
        // via CNTKCTL_EL1.EL0VCTEN, which Linux and macOS always set; if a kernel
        // cleared it, the read would trap (SIGILL), which aborts the process but is not
        // undefined behavior.
        unsafe {
            core::arch::asm!(
                "isb",
                "mrs {0}, cntvct_el0",
                out(reg) cycles,
                options(nomem, nostack, preserves_flags)
            );
        }
        cycles
    }
}

/// Returns the current time in nanoseconds from the serialized cycle counter
/// (RDTSC on x86_64, `CNTVCT_EL0` on aarch64).
///
/// Avoids vDSO syscalls. If the counter is not calibrated, falls back to
/// `Instant::now()`.
///
/// Continuity guarantee: when calibration completes, `rdtsc_nanos()` transitions
/// continuously and monotonically (|Δ| < 1 ms, no system uptime discontinuity)
/// by evaluating cycles elapsed since the calibration anchor point.
#[inline(always)]
pub fn rdtsc_nanos() -> u64 {
    // Acquire pairs with Release store on TSC_MULT so TSC_ANCHOR_TSC and
    // TSC_ANCHOR_NS written prior to publication are visible immediately.
    let mult = TSC_MULT.load(Ordering::Acquire);

    if mult != 0 {
        let anchor_tsc = TSC_ANCHOR_TSC.load(Ordering::Relaxed);
        let anchor_ns = TSC_ANCHOR_NS.load(Ordering::Relaxed);
        anchor_ns.wrapping_add(cycles_to_nanos(read_counter().wrapping_sub(anchor_tsc), mult))
    } else {
        BOOT_TIME.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

/// Probes the CPU for invariant TSC support via CPUID.
///
/// Invariant TSC means the counter ticks at a constant rate regardless of
/// P-state, C-state, or other CPU frequency scaling. This is critical for
/// reliable timing in the audio hot-path.
#[cfg(target_arch = "x86_64")]
fn probe_invariant_tsc() {
    let res = core::arch::x86_64::__cpuid(0x8000_0007);
    if res.edx & (1 << 8) != 0 {
        log::info!("Invariant TSC confirmed");
    } else {
        log::warn!("Non-invariant TSC detected — timing may drift under CPU scaling");
    }
}

/// Returns whether the TSC has been calibrated.
#[inline]
pub fn is_tsc_calibrated() -> bool {
    TSC_MULT.load(Ordering::Relaxed) != 0
}

/// Reads the generic-timer frequency in Hz (`CNTFRQ_EL0`).
#[cfg(target_arch = "aarch64")]
fn generic_timer_frequency() -> u64 {
    let freq: u64;
    // SAFETY: CNTFRQ_EL0 is readable from EL0; no memory is accessed.
    unsafe {
        core::arch::asm!("mrs {0}, cntfrq_el0", out(reg) freq, options(nomem, nostack, preserves_flags));
    }
    freq
}

/// Plausible generic-timer frequency range. Armv8.6+ fixes the frequency at
/// 1 GHz; earlier systems commonly use 19.2–100 MHz (24 MHz on Apple Silicon).
#[cfg(target_arch = "aarch64")]
const GENERIC_TIMER_PLAUSIBLE_HZ: std::ops::RangeInclusive<u64> = 1_000_000..=1_000_000_000;

/// Where the aarch64 calibration rate came from.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenericTimerSource {
    /// `CNTFRQ_EL0`, used as-is (exact).
    Register,
    /// Measured against `Instant` because `CNTFRQ_EL0` was implausible.
    Measured,
}

/// Chooses the calibration multiplier for the aarch64 generic timer.
///
/// A plausible `reported_hz` (`CNTFRQ_EL0`) is used as-is and `measure` is not
/// called: a single short wall-clock window is noisier than the register, and
/// on macOS `Instant` is derived from this same counter, so a measurement could
/// not detect a wrong register value anyway. Only a zero or out-of-range value
/// (misprogrammed firmware) falls back to `measure`, which returns
/// `(elapsed_cycles, elapsed_nanos)`. Returns `None` when neither is usable.
#[cfg(target_arch = "aarch64")]
fn choose_generic_timer_mult(
    reported_hz: u64,
    measure: impl FnOnce() -> Option<(u64, u64)>,
) -> Option<(u64, GenericTimerSource)> {
    if GENERIC_TIMER_PLAUSIBLE_HZ.contains(&reported_hz) {
        return mult_for_rate(reported_hz, 1_000_000_000)
            .map(|m| (m, GenericTimerSource::Register));
    }
    let (cycles, nanos) = measure()?;
    mult_for_rate(cycles, nanos).map(|m| (m, GenericTimerSource::Measured))
}

/// Measures the generic timer against `Instant` over `windows` windows of
/// `window` each and returns the `(cycles, nanos)` of the least-disturbed one.
///
/// Each counter read is bracketed by `Instant` readings taken immediately
/// before and after it, and the wall-clock interval is taken between the
/// bracket midpoints, so a preemption shows up as bracket slack instead of
/// skewing only one side. The window with the smallest total slack wins.
#[cfg(target_arch = "aarch64")]
fn measure_generic_timer(window: Duration, windows: usize) -> Option<(u64, u64)> {
    let mut best: Option<(u128, u64, u64)> = None;
    for _ in 0..windows {
        let a0 = Instant::now();
        let c0 = read_counter();
        let a1 = Instant::now();
        std::thread::sleep(window);
        let b0 = Instant::now();
        let c1 = read_counter();
        let b1 = Instant::now();

        let slack = (a1 - a0).as_nanos() + (b1 - b0).as_nanos();
        // Midpoint-to-midpoint wall time: (b0 + b1)/2 - (a0 + a1)/2.
        let nanos = ((b0 - a0).as_nanos() + (b1 - a1).as_nanos()) / 2;
        let cycles = c1.wrapping_sub(c0);
        if best.is_none_or(|(s, _, _)| slack < s) {
            best = Some((slack, cycles, nanos as u64));
        }
    }
    best.map(|(_, cycles, nanos)| (cycles, nanos))
}

/// Calibrates the cycle counter.
///
/// aarch64: from the architected generic-timer frequency (`CNTFRQ_EL0`),
/// exact and without blocking whenever the value is plausible. Only a zero or
/// implausible value (misprogrammed firmware) triggers a short measurement
/// (best of three 5 ms windows), with a warning.
///
/// Runs only once at program startup (cold-path).
#[cfg(target_arch = "aarch64")]
#[cold]
pub fn calibrate_tsc() {
    if is_tsc_calibrated() {
        return;
    }
    let reported_hz = generic_timer_frequency();
    let measure = || measure_generic_timer(Duration::from_millis(5), 3);
    match choose_generic_timer_mult(reported_hz, measure) {
        Some((mult, GenericTimerSource::Register)) => {
            publish_calibration(mult);
            log::info!(
                "Generic timer (CNTVCT_EL0) calibrated at {:.3} MHz from CNTFRQ_EL0",
                reported_hz as f64 / 1.0e6
            );
        }
        Some((mult, GenericTimerSource::Measured)) => {
            publish_calibration(mult);
            log::warn!(
                "CNTFRQ_EL0 reports an implausible {} Hz; generic timer calibrated \
                 from a measurement instead",
                reported_hz
            );
        }
        None => log::warn!(
            "Generic timer could not be calibrated (CNTFRQ_EL0 = {} Hz, no usable \
             measurement) — cycle clock left uncalibrated (Instant fallback)",
            reported_hz
        ),
    }
}

/// Calibrates the TSC (Time Stamp Counter) frequency against the system clock
/// and validates it against `CLOCK_MONOTONIC_RAW`.
///
/// This function runs only once at program startup (cold-path).
#[cfg(target_arch = "x86_64")]
#[cold]
pub fn calibrate_tsc() {
    if is_tsc_calibrated() {
        return;
    }
    use std::thread;

    // 0. PROBE: Check if the CPU supports invariant TSC.
    probe_invariant_tsc();

    // 1. WARM-UP:
    // Call the serialized instruction once and wait a bit; the result is discarded.
    let _ = read_counter();
    thread::sleep(Duration::from_millis(10));

    // 2. ZERO POINT (Start of Measurement):
    let start_raw = monotonic_raw_nanos();
    let start_inst = Instant::now();
    let start_tsc = read_counter();

    // 3. CONTROLLED WAIT:
    thread::sleep(Duration::from_millis(50));

    // 4. END POINT:
    let end_raw = monotonic_raw_nanos();
    let end_inst = Instant::now();
    let end_tsc = read_counter();

    let elapsed_nanos = end_inst.duration_since(start_inst).as_nanos() as u64;
    let elapsed_cycles = end_tsc.wrapping_sub(start_tsc);

    // 5. CONVERSION RATE CALCULATION (cyc2ns-style):
    // Computes `mult` once on this cold-path so the hot-path needs only a
    // widening multiply plus shift. Rejects degenerate windows to avoid
    // freezing the clock at zero.
    if let Some(mult) = mult_for_rate(elapsed_cycles, elapsed_nanos) {
        publish_calibration(mult);
        let freq_ghz = elapsed_cycles as f64 / elapsed_nanos as f64;

        if let (Some(s_raw), Some(e_raw)) = (start_raw, end_raw) {
            let raw_delta = e_raw.saturating_sub(s_raw);
            let tsc_calc_ns = cycles_to_nanos(elapsed_cycles, mult);
            let drift_ppm = if raw_delta > 0 {
                ((tsc_calc_ns as i64 - raw_delta as i64).abs() * 1_000_000) / raw_delta as i64
            } else {
                0
            };
            log::info!(
                "TSC calibrated at {:.3} GHz (validated against CLOCK_MONOTONIC_RAW, drift: {} ppm)",
                freq_ghz,
                drift_ppm
            );
        } else {
            log::info!("TSC calibrated at {:.3} GHz", freq_ghz);
        }
    }
}

/// Test-only calibration override.
///
/// Stores `mult` and publishes anchor values as if `calibrate_tsc` had measured
/// the given rate. Unit tests use this to pin exact conversion rates without sleeping.
#[cfg(test)]
fn set_mult_for_test(mult: u64) {
    if mult != 0 {
        publish_calibration(mult);
    } else {
        TSC_ANCHOR_TSC.store(0, Ordering::Relaxed);
        TSC_ANCHOR_NS.store(0, Ordering::Relaxed);
        TSC_MULT.store(0, Ordering::Release);
    }
}

#[cfg(test)]
fn set_calibration_for_test(mult: u64, anchor_tsc: u64, anchor_ns: u64) {
    TSC_ANCHOR_TSC.store(anchor_tsc, Ordering::Relaxed);
    TSC_ANCHOR_NS.store(anchor_ns, Ordering::Relaxed);
    TSC_MULT.store(mult, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let expected =
                ((cycles as u128 * elapsed_nanos as u128) / elapsed_cycles as u128) as u64;
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
        let (cycles, nanos) =
            measure_generic_timer(Duration::from_millis(5), 3).expect("measurement");
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

    #[test]
    #[ignore = "Empirical micro-benchmark for T-P4.2.3: fallback Instant vs calibrated RDTSC"]
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
}
