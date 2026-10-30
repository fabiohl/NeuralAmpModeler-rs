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
        anchor_ns.wrapping_add(cycles_to_nanos(
            read_counter().wrapping_sub(anchor_tsc),
            mult,
        ))
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

#[cfg(test)]
#[path = "tsc_test.rs"]
mod tests;
