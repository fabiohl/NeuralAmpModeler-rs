// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Pure Mach absolute-time conversion shared by the macOS RT bring-up.
//!
//! Kept free of libc and system calls so the arithmetic can be exercised on
//! any host. The macOS module supplies the live `mach_timebase_info` values.

use std::io;

/// Real-time timing parameters in Mach absolute-time ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadTiming {
    /// Nominal time between callback arrivals.
    pub period_ticks: u32,
    /// CPU budget expected inside one callback.
    pub computation_ticks: u32,
    /// Maximum time from callback start to completion.
    pub constraint_ticks: u32,
    /// The same period expressed in nanoseconds (telemetry/diagnostics).
    pub period_ns: u64,
}

/// Converts a `buffer_frames`/`sample_rate` quantum into Mach ticks.
///
/// `numer` and `denom` are the values returned by `mach_timebase_info`:
/// `nanoseconds = ticks * numer / denom`, therefore
/// `ticks = nanoseconds * denom / numer`.
///
/// Returns `InvalidInput` for a zero quantum, zero sample rate, zero timebase
/// component, or a period that does not fit in a `u32` tick count.
pub fn compute_thread_timing(
    buffer_frames: u32,
    sample_rate: u32,
    numer: u32,
    denom: u32,
) -> io::Result<ThreadTiming> {
    if buffer_frames == 0 || sample_rate == 0 || numer == 0 || denom == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "buffer_frames, sample_rate, timebase numer and denom must all be non-zero",
        ));
    }

    let period_ns = (u64::from(buffer_frames) * 1_000_000_000) / u64::from(sample_rate);
    if period_ns == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "buffer quantum is shorter than one nanosecond",
        ));
    }

    let ticks = u128::from(period_ns) * u128::from(denom) / u128::from(numer);
    let ticks = u32::try_from(ticks).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "callback period does not fit in u32 Mach ticks",
        )
    })?;

    Ok(ThreadTiming {
        period_ticks: ticks,
        computation_ticks: ticks,
        constraint_ticks: ticks,
        period_ns,
    })
}

#[cfg(test)]
#[path = "timebase_test.rs"]
mod timebase_test;
