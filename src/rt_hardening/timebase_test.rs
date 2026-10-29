// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for the pure Mach timebase conversion.
//!
//! These run on any host: no macOS syscall is made.

use super::compute_thread_timing;

#[test]
fn test_64_frames_at_48k_identity_timebase() {
    let timing = compute_thread_timing(64, 48_000, 1, 1).expect("valid 64-frame quantum");
    let expected_ns = 64 * 1_000_000_000 / 48_000;
    assert_eq!(timing.period_ns, expected_ns);
    assert_eq!(timing.period_ticks, expected_ns as u32);
    assert_eq!(timing.computation_ticks, timing.period_ticks);
    assert_eq!(timing.constraint_ticks, timing.period_ticks);
}

#[test]
fn test_75_frames_at_48k_fractional_timebase() {
    // 75 / 48000 s = 1_562_500 ns; with numer=125, denom=3:
    // ticks = 1_562_500 * 3 / 125 = 37_500.
    let timing = compute_thread_timing(75, 48_000, 125, 3).expect("valid 75-frame quantum");
    assert_eq!(timing.period_ns, 1_562_500);
    assert_eq!(timing.period_ticks, 37_500);
}

#[test]
fn test_one_frame_at_384k() {
    // 2_604 ns floor with numer=1, denom=1.
    let timing = compute_thread_timing(1, 384_000, 1, 1).expect("valid single-frame quantum");
    assert_eq!(timing.period_ns, 2_604);
    assert_eq!(timing.period_ticks, 2_604);
}

#[test]
fn test_zero_inputs_are_rejected() {
    assert!(compute_thread_timing(0, 48_000, 1, 1).is_err());
    assert!(compute_thread_timing(64, 0, 1, 1).is_err());
    assert!(compute_thread_timing(64, 48_000, 0, 1).is_err());
    assert!(compute_thread_timing(64, 48_000, 1, 0).is_err());
}

#[test]
fn test_tick_overflow_is_rejected() {
    // u32::MAX frames at 48 kHz is far beyond 4.29 s of Mach ticks.
    assert!(compute_thread_timing(u32::MAX, 48_000, 1, 1).is_err());
}
