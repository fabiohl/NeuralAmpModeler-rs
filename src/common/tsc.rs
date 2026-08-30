// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Time Stamp Counter (TSC) calibration and reading via RDTSC.
//!
//! Provides time measurement with ~1ns precision and ~1 cycle cost,
//! avoiding the vDSO clock_gettime syscall in the audio hot-path.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Calibrated TSC frequency in GHz (cycles per nanosecond).
/// Stored as fixed-point (value * 1000) to avoid floats in the hot-path.
static TSC_FREQ_GHZ_X1000: AtomicU64 = AtomicU64::new(0);
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

/// Returns the current time in nanoseconds using the serialized RDTSC instruction.
///
/// Serialized with `_mm_lfence` to prevent out-of-order execution reordering.
/// Provides sub-nanosecond precision with ~15ns cost, avoiding vDSO syscalls.
/// If the TSC is not calibrated or unavailable, falls back to Instant::now().
#[inline(always)]
pub fn rdtsc_nanos() -> u64 {
    let freq_x1000 = TSC_FREQ_GHZ_X1000.load(Ordering::Relaxed);

    // SAFETY: Division by zero in the hot-path is fatal. If calibration failed
    // at boot, we use the system clock as a safety net.
    #[expect(
        clippy::manual_checked_ops,
        reason = "Manual bounds check used for RT-predictable assembly over checked_add"
    )]
    if freq_x1000 != 0 {
        // SAFETY: `_mm_lfence` + `_rdtsc` is available on all x86-64 CPUs; it performs no
        // memory access and has no side effects, so reading it here is sound.
        let cycles = unsafe {
            core::arch::x86_64::_mm_lfence();
            core::arch::x86_64::_rdtsc()
        };
        (cycles * 1000) / freq_x1000
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

    // 5. CONVERSION RATE CALCULATION:
    if let Some(freq_x1000) = (elapsed_cycles * 1000).checked_div(elapsed_nanos) {
        TSC_FREQ_GHZ_X1000.store(freq_x1000, Ordering::Release);

        if let (Some(s_raw), Some(e_raw)) = (start_raw, end_raw) {
            let raw_delta = e_raw.saturating_sub(s_raw);
            let tsc_calc_ns = (elapsed_cycles * 1000) / freq_x1000;
            let drift_ppm = if raw_delta > 0 {
                ((tsc_calc_ns as i64 - raw_delta as i64).abs() * 1_000_000) / raw_delta as i64
            } else {
                0
            };
            log::info!(
                "TSC calibrated at {:.3} GHz (validated against CLOCK_MONOTONIC_RAW, drift: {} ppm)",
                freq_x1000 as f64 / 1000.0,
                drift_ppm
            );
        } else {
            log::info!("TSC calibrated at {:.3} GHz", freq_x1000 as f64 / 1000.0);
        }
    }
}
