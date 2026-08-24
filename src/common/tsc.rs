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

/// Returns the current time in nanoseconds using the RDTSC instruction.
///
/// Provides ~1ns precision with ~1 cycle cost, avoiding the vDSO clock_gettime syscall.
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
        // SAFETY: `_rdtsc` is available on all x86-64 CPUs; it performs no
        // memory access and has no side effects, so reading it here is sound.
        let cycles = unsafe { core::arch::x86_64::_rdtsc() };
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

/// Calibrates the TSC (Time Stamp Counter) frequency against the system clock.
///
/// Imagine the CPU has an internal "odometer" that counts every heartbeat (cycle).
/// Since the CPU speed can vary, we need to figure out how many "ticks"
/// equal 1 real nanosecond so we can measure time with surgical precision.
///
/// This function runs only once at program startup (cold-path).
#[cold]
pub fn calibrate_tsc() {
    use std::thread;

    // 0. PROBE: Check if the CPU supports invariant TSC.
    probe_invariant_tsc();

    // 1. WARM-UP:
    // We call the instruction once and wait a bit. This ensures the CPU
    // "wakes up" from low-power states and that data is ready in the caches.
    // SAFETY: `_rdtsc` is available on all x86-64 CPUs; it performs no memory
    // access and has no side effects, so reading it here is sound.
    let _ = unsafe { core::arch::x86_64::_rdtsc() };
    thread::sleep(Duration::from_millis(10));

    // 2. ZERO POINT (Start of Measurement):
    // We simultaneously capture the system clock time (slow but reliable)
    // and the CPU cycle counter value (ultra-fast).
    let start_inst = Instant::now();
    // SAFETY: `_rdtsc` is available on all x86-64 CPUs; it performs no memory
    // access and has no side effects, so reading it here is sound.
    let start_tsc = unsafe { core::arch::x86_64::_rdtsc() };

    // 3. CONTROLLED WAIT:
    // We wait 50 milliseconds. It's a short time for a human, but allows
    // the CPU to run millions of cycles, reducing sampling errors.
    thread::sleep(Duration::from_millis(50));

    // 4. END POINT:
    // We capture both values again to calculate how much each advanced.
    let end_inst = Instant::now();
    // SAFETY: `_rdtsc` is available on all x86-64 CPUs; it performs no memory
    // access and has no side effects, so reading it here is sound.
    let end_tsc = unsafe { core::arch::x86_64::_rdtsc() };

    let elapsed_nanos = end_inst.duration_since(start_inst).as_nanos() as u64;
    let elapsed_cycles = end_tsc.wrapping_sub(start_tsc);

    // 5. CONVERSION RATE CALCULATION:
    // We compute the ratio: (Cycles * 1000) / Nanoseconds.
    // We use the 1000 multiplier to store the result as an integer
    // while maintaining 3 decimal places of precision without floating point.
    if let Some(freq_x1000) = (elapsed_cycles * 1000).checked_div(elapsed_nanos) {
        TSC_FREQ_GHZ_X1000.store(freq_x1000, Ordering::Release);

        log::info!("TSC calibrated at {:.3} GHz", freq_x1000 as f64 / 1000.0);
    }
}
