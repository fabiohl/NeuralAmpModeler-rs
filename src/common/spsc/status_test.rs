// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use core::mem::{align_of, offset_of, size_of};

#[test]
fn test_rt_status_flags_alignment_and_cache_isolation() {
    assert_eq!(align_of::<RtStatusFlags>(), 128);
    assert_eq!(size_of::<RtStatusFlags>() % 128, 0);

    // Verify RT-hot fields are in cache line 0 (offset < 64)
    assert!(offset_of!(RtStatusFlags, status_bits) < 64);
    assert!(offset_of!(RtStatusFlags, dsp_cycle_time) < 64);
    assert!(offset_of!(RtStatusFlags, last_n_samples) < 64);
    assert!(offset_of!(RtStatusFlags, dsp_overloads) < 64);
    assert!(offset_of!(RtStatusFlags, xruns) < 64);
    assert!(offset_of!(RtStatusFlags, degrade_transitions_total) < 64);
    assert!(offset_of!(RtStatusFlags, structural_deferred_total) < 64);
    assert!(offset_of!(RtStatusFlags, drains) < 64);

    // Verify latency histogram is 128-byte aligned and isolated
    let hist_offset = offset_of!(RtStatusFlags, latency_hist);
    assert_eq!(hist_offset % 128, 0);

    // Verify capture-owned miss counter is on its own cache line
    let capture_offset = offset_of!(RtStatusFlags, input_buffer_miss);
    assert_eq!(capture_offset % 64, 0);

    // Verify playback-owned miss counter is on its own cache line
    let playback_offset = offset_of!(RtStatusFlags, output_buffer_miss);
    assert_eq!(playback_offset % 64, 0);

    // Verify main-owned previous_buffer_frames is on its own cache line
    let main_offset = offset_of!(RtStatusFlags, previous_buffer_frames);
    assert_eq!(main_offset % 64, 0);

    // F-PERF-09: Ensure input_buffer_miss, output_buffer_miss, and previous_buffer_frames
    // are on strictly distinct 64-byte cache lines.
    let capture_line = capture_offset / 64;
    let playback_line = playback_offset / 64;
    let main_line = main_offset / 64;
    let rt_hot_line = offset_of!(RtStatusFlags, status_bits) / 64;
    let hist_line_start = hist_offset / 64;
    let hist_line_end =
        (hist_offset + size_of::<crate::dsp::telemetry::LatencyHistogram>() - 1) / 64;

    assert_ne!(
        capture_line, playback_line,
        "Capture and Playback must not share a cache line"
    );
    assert_ne!(
        capture_line, main_line,
        "Capture and Main must not share a cache line"
    );
    assert_ne!(
        playback_line, main_line,
        "Playback and Main must not share a cache line"
    );
    assert_ne!(
        capture_line, rt_hot_line,
        "Capture and RT hot path must not share a cache line"
    );
    assert_ne!(
        playback_line, rt_hot_line,
        "Playback and RT hot path must not share a cache line"
    );
    assert_ne!(
        main_line, rt_hot_line,
        "Main and RT hot path must not share a cache line"
    );

    assert!(capture_line < hist_line_start || capture_line > hist_line_end);
    assert!(playback_line < hist_line_start || playback_line > hist_line_end);
    assert!(main_line < hist_line_start || main_line > hist_line_end);
}

#[test]
fn test_rt_status_flags_default_values() {
    let flags = RtStatusFlags::new();
    assert_eq!(flags.status_bits.load(Ordering::Relaxed), 0);
    assert_eq!(flags.dsp_cycle_time.load(Ordering::Relaxed), 0);
    assert_eq!(flags.last_n_samples.load(Ordering::Relaxed), 0);
    assert_eq!(flags.dsp_overloads.load(Ordering::Relaxed), 0);
    assert_eq!(flags.xruns.load(Ordering::Relaxed), 0);
    assert_eq!(flags.degrade_transitions_total.load(Ordering::Relaxed), 0);
    assert_eq!(flags.structural_deferred_total.load(Ordering::Relaxed), 0);
    assert_eq!(flags.drains.load(Ordering::Relaxed), 0);

    assert_eq!(flags.active_rate.load(Ordering::Relaxed), 0);
    assert_eq!(flags.active_rate_changed.load(Ordering::Relaxed), 0);
    assert_eq!(flags.requested_host_rate.load(Ordering::Relaxed), 0);
    assert_eq!(flags.requested_nam_rate.load(Ordering::Relaxed), 48_000);
    assert_eq!(flags.requested_buffer_frames.load(Ordering::Relaxed), 0);
    assert_eq!(
        flags
            .requested_cabsim_partition_size
            .load(Ordering::Relaxed),
        0
    );
    assert_eq!(flags.requested_cabsim_host_rate.load(Ordering::Relaxed), 0);
    assert_eq!(flags.requested_slimmable_ch.load(Ordering::Relaxed), 0);
    assert_eq!(flags.requested_os_factor.load(Ordering::Relaxed), 0);
    assert_eq!(flags.requested_rate_generation.load(Ordering::Relaxed), 0);
    assert_eq!(flags.applied_rate_generation.load(Ordering::Relaxed), 0);
    assert_eq!(flags.requested_cabsim_generation.load(Ordering::Relaxed), 0);
    assert_eq!(flags.applied_cabsim_generation.load(Ordering::Relaxed), 0);
    assert_eq!(
        flags.requested_slimmable_generation.load(Ordering::Relaxed),
        0
    );
    assert_eq!(flags.requested_os_generation.load(Ordering::Relaxed), 0);
    assert_eq!(flags.applied_os_generation.load(Ordering::Relaxed), 0);

    assert_eq!(flags.input_buffer_miss.load(Ordering::Relaxed), 0);
    assert_eq!(flags.output_buffer_miss.load(Ordering::Relaxed), 0);

    assert_eq!(flags.previous_buffer_frames.load(Ordering::Relaxed), 0);
    assert_eq!(flags.structural_superseded_total.load(Ordering::Relaxed), 0);
    assert_eq!(flags.resampler_failed_generation.load(Ordering::Relaxed), 0);
    assert_eq!(flags.flags_seen.load(Ordering::Relaxed), 0);
    assert_eq!(flags.rt_priority.load(Ordering::Relaxed), -1);
    assert_eq!(flags.confirmed_priority.load(Ordering::Relaxed), -1);
    assert_eq!(flags.rt_policy.load(Ordering::Relaxed), -1);
    assert_eq!(flags.rt_cpu.load(Ordering::Relaxed), -1);
    assert_eq!(flags.rt_target_cpu.load(Ordering::Relaxed), -1);
    assert_eq!(flags.rt_affinity_err.load(Ordering::Relaxed), 0);
    assert_eq!(flags.rt_sched_err.load(Ordering::Relaxed), 0);
    assert_eq!(flags.rt_getsched_err.load(Ordering::Relaxed), 0);
    assert_eq!(flags.rt_tid.load(Ordering::Relaxed), -1);
    assert_eq!(flags.first_block_nanos.load(Ordering::Relaxed), 0);
}
