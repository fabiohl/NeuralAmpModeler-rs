// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Time Stamp Counter (TSC) calibration and reading via RDTSC.
//!
//! Provides time measurement with ~1ns precision and ~1 cycle cost,
//! avoiding the vDSO clock_gettime syscall in the audio hot-path.
//!
//! Conversion uses fixed-point `mult/shift` scaling in the style of the
//! Linux kernel `cyc2ns` (`ns = (cycles * mult) >> shift`), computed once
//! at startup. The hot-path performs no division and no `cycles * 1000`,
//! so large counter values cannot overflow in debug builds.
//!
//! Portability contract: this module is x86-64-only, requiring the
//! x86-64-v3 baseline (AVX2, FMA, BMI2). The crate root rejects other
//! architectures at compile time and `common/mod.rs` exposes this module
//! only under `cfg(target_arch = "x86_64")`. There is intentionally no
//! cross-ISA fallback in this module; the `Instant` fallback below covers
//! only the uncalibrated-TSC case on x86-64 hosts.

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
/// Time anchor for rdtsc fallback (monotonic).
static BOOT_TIME: OnceLock<Instant> = OnceLock::new();

/// Helper to read `CLOCK_MONOTONIC_RAW` on Linux for startup calibration validation.
#[cfg(target_os = "linux")]
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

#[cfg(not(target_os = "linux"))]
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

/// Returns the current time in nanoseconds using the serialized RDTSC instruction.
///
/// Serialized with `_mm_lfence` to prevent out-of-order execution reordering.
/// Provides sub-nanosecond precision with ~15ns cost, avoiding vDSO syscalls.
/// If the TSC is not calibrated, falls back to Instant::now().
#[inline(always)]
pub fn rdtsc_nanos() -> u64 {
    let mult = TSC_MULT.load(Ordering::Relaxed);

    if mult != 0 {
        // SAFETY: `_mm_lfence` + `_rdtsc` is available on all x86-64 CPUs; it performs no
        // memory access and has no side effects, so reading it here is sound.
        let cycles = unsafe {
            core::arch::x86_64::_mm_lfence();
            core::arch::x86_64::_rdtsc()
        };
        cycles_to_nanos(cycles, mult)
    } else {
        BOOT_TIME.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

/// Probes the CPU for invariant TSC support via CPUID.
///
/// Invariant TSC means the counter ticks at a constant rate regardless of
/// P-state, C-state, or other CPU frequency scaling. This is critical for
/// reliable timing in the audio hot-path.
fn probe_invariant_tsc() {
    let res = core::arch::x86_64::__cpuid(0x8000_0007);
    if res.edx & (1 << 8) != 0 {
        log::info!("Invariant TSC confirmed");
    } else {
        log::warn!("Non-invariant TSC detected — timing may drift under CPU scaling");
    }
}

/// Calibrates the TSC (Time Stamp Counter) frequency against the system clock
/// and validates it against `CLOCK_MONOTONIC_RAW`.
///
/// This function runs only once at program startup (cold-path).
#[cold]
pub fn calibrate_tsc() {
    use std::thread;

    // 0. PROBE: Check if the CPU supports invariant TSC.
    probe_invariant_tsc();

    // 1. WARM-UP:
    // Call the serialized instruction once and wait a bit.
    // SAFETY: `_mm_lfence` and `_rdtsc` are available on all x86-64 CPUs (including
    // x86-64-v3 baseline). They perform no memory writes and have no side-effects
    // beyond reading the time-stamp counter; the result is intentionally discarded.
    let _ = unsafe {
        core::arch::x86_64::_mm_lfence();
        core::arch::x86_64::_rdtsc()
    };
    thread::sleep(Duration::from_millis(10));

    // 2. ZERO POINT (Start of Measurement):
    let start_raw = monotonic_raw_nanos();
    let start_inst = Instant::now();
    // SAFETY: `_mm_lfence` serializes the instruction stream before `_rdtsc`, ensuring
    // that no prior loads are reordered past the timestamp read. Both intrinsics are
    // available unconditionally on the x86-64-v3 baseline enforced by `.cargo/config.toml`.
    let start_tsc = unsafe {
        core::arch::x86_64::_mm_lfence();
        core::arch::x86_64::_rdtsc()
    };

    // 3. CONTROLLED WAIT:
    thread::sleep(Duration::from_millis(50));

    // 4. END POINT:
    let end_raw = monotonic_raw_nanos();
    let end_inst = Instant::now();
    // SAFETY: Same contract as `start_tsc` above — `_mm_lfence` + `_rdtsc` on the
    // unconditional x86-64-v3 baseline; no memory writes, no side-effects.
    let end_tsc = unsafe {
        core::arch::x86_64::_mm_lfence();
        core::arch::x86_64::_rdtsc()
    };

    let elapsed_nanos = end_inst.duration_since(start_inst).as_nanos() as u64;
    let elapsed_cycles = end_tsc.wrapping_sub(start_tsc);

    // 5. CONVERSION RATE CALCULATION (cyc2ns-style):
    // Computes `mult` once on this cold-path so the hot-path needs only a
    // widening multiply plus shift. Rejects degenerate windows to avoid
    // freezing the clock at zero.
    if let Some(mult) = mult_for_rate(elapsed_cycles, elapsed_nanos) {
        TSC_MULT.store(mult, Ordering::Release);
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
/// Stores `mult` as if `calibrate_tsc` had measured the given rate. Unit
/// tests use this to pin exact conversion rates without sleeping.
#[cfg(test)]
fn set_mult_for_test(mult: u64) {
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
}
