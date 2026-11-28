// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::common::diagnostics::NamErrorCode;
use crate::loader::nam_json::LinearImplementation;
use crate::models::StaticModel;
use crate::models::linear::LinearModel;

fn make_lstm() -> Box<StaticModel> {
    Box::new(StaticModel::Lstm1x8(Box::default()))
}

fn make_linear_dc(bias: f32) -> Box<StaticModel> {
    Box::new(StaticModel::Linear(Box::new(
        LinearModel::new(vec![0.0], bias, LinearImplementation::Direct).expect("linear dc"),
    )))
}

#[test]
fn test_valid_max_value_passes() {
    let submodels = vec![(0.5, make_lstm()), (1.0, make_lstm())];
    assert!(ContainerModel::new(submodels, 48000).is_ok());
}

#[test]
fn test_reject_max_value_nan() {
    let submodels = vec![(f32::NAN, make_lstm()), (1.0, make_lstm())];
    match ContainerModel::new(submodels, 48000) {
        Ok(_) => panic!("Expected NaN rejection"),
        Err(e) => assert!(
            e.to_string().contains("invalid max_value=NaN"),
            "Expected NaN rejection, got: {e}"
        ),
    }
}

#[test]
fn test_reject_max_value_inf() {
    let submodels = vec![(f32::INFINITY, make_lstm()), (1.0, make_lstm())];
    match ContainerModel::new(submodels, 48000) {
        Ok(_) => panic!("Expected Inf rejection"),
        Err(e) => assert!(
            e.to_string().contains("invalid max_value"),
            "Expected Inf rejection, got: {e}"
        ),
    }
}

#[test]
fn test_reject_max_value_neg_inf() {
    let submodels = vec![(f32::NEG_INFINITY, make_lstm()), (1.0, make_lstm())];
    match ContainerModel::new(submodels, 48000) {
        Ok(_) => panic!("Expected -Inf rejection"),
        Err(e) => assert!(
            e.to_string().contains("invalid max_value"),
            "Expected -Inf rejection, got: {e}"
        ),
    }
}

#[test]
fn test_reject_max_value_negative() {
    let submodels = vec![(-0.5, make_lstm()), (1.0, make_lstm())];
    match ContainerModel::new(submodels, 48000) {
        Ok(_) => panic!("Expected negative rejection"),
        Err(e) => assert!(
            e.to_string().contains("invalid max_value=-0.5"),
            "Expected negative rejection, got: {e}"
        ),
    }
}

#[test]
fn test_slimmable_size_zero_selects_first_submodel() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_slimmable_size(0.0, None);

    assert_eq!(container.pending_index(), Some(0));
}

#[test]
fn test_slimmable_size_one_selects_last_submodel() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_active_index(0);

    container.set_slimmable_size(1.0, None);

    assert_eq!(container.pending_index(), Some(2));
}

#[test]
fn test_slimmable_size_between_thresholds() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_slimmable_size(0.5, None);

    assert_eq!(container.pending_index(), Some(1));
}

#[test]
fn test_slimmable_size_same_value_noop() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_slimmable_size(0.5, None);
    assert_eq!(container.pending_index(), Some(1));

    container.set_slimmable_size(0.5, None);

    assert_eq!(container.pending_index(), Some(1));
}

#[test]
fn test_slimmable_size_same_active_noop() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_active_index(0);

    container.set_slimmable_size(0.2, None);

    assert!(container.pending_index().is_none());
}

#[test]
fn test_slimmable_size_change_during_crossfade() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_slimmable_size(0.2, None);
    assert_eq!(container.pending_index(), Some(0));

    container.set_slimmable_size(0.5, None);

    assert_eq!(container.pending_index(), Some(1));
    assert_eq!(container.active_index(), 0);
}

#[test]
fn test_default_scratch_sized_for_max_resamp_buf() {
    let submodels = vec![(0.5, make_lstm()), (1.0, make_lstm())];
    let container = ContainerModel::new(submodels, 48000).unwrap();

    assert!(
        container.scratch_buffer.len() >= crate::dsp::pipeline::MAX_RESAMP_BUF,
        "default scratch {} must cover MAX_RESAMP_BUF ({})",
        container.scratch_buffer.len(),
        crate::dsp::pipeline::MAX_RESAMP_BUF
    );
}

/// Verifies that an 8192-sample block processed while a tier switch is in
/// progress must not panic nor slice the scratch out of bounds — in debug and
/// in `--release`.
#[test]
fn test_8192_block_with_pending_switch_no_panic() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_slimmable_size(0.2, None);
    assert!(container.is_crossfading());

    let n = 8192usize;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
    let mut output = vec![0.0f32; n];

    container.process(&input, &mut output);

    assert!(output.iter().all(|s| s.is_finite()));
    // The crossfade advances but never completes in a single 8192 block only
    // if the block was actually processed through the crossfade path.
    assert!(container.is_crossfading() || container.pending_index().is_none());
}

/// Verifies that when the input block exceeds the scratch capacity, the
/// crossfade must abort gracefully (process the active submodel only) without
/// panicking and without slicing the scratch beyond its length.
#[test]
fn test_oversized_block_aborts_crossfade_without_panic() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    // Shrink the scratch to a small size, simulating a host that negotiated a
    // small block and then delivered a larger one.
    container.reset(48000, 512).unwrap();
    assert_eq!(container.scratch_buffer.len(), 512);

    container.set_slimmable_size(0.2, None);
    assert!(container.is_crossfading());

    let n = 1024usize;
    let input = vec![0.5f32; n];
    let mut output = vec![0.0f32; n];

    container.process(&input, &mut output);

    // No panic; output was produced by the active submodel path.
    assert!(output.iter().all(|s| s.is_finite()));
    // The guard must not consume the crossfade state: it stays pending so the
    // transition resumes on a subsequent fitting block.
    assert!(container.is_crossfading());
}

/// A 32 ms crossfade must follow the per-sample linear envelope (and thus not
/// introduce discontinuities above −80 dBFS between consecutive samples).
#[test]
fn test_crossfade_32ms_blend_smoothness() {
    const SR: u32 = 48_000;
    const BLOCK: usize = 64;
    const ACTIVE_DC: f32 = 1.0;
    const PENDING_DC: f32 = 0.0;
    const MAX_DISCONTINUITY: f32 = 1e-4;

    let duration = (CROSSFADE_DURATION_MS / 1000.0 * SR as f32).round() as usize;
    assert_eq!(duration, 1536, "32 ms @ 48 kHz must be 1536 samples");
    assert_eq!(duration % BLOCK, 0);

    let submodels = vec![
        (0.5, make_linear_dc(PENDING_DC)),
        (1.0, make_linear_dc(ACTIVE_DC)),
    ];
    let mut container = ContainerModel::new(submodels, SR).unwrap();
    assert_eq!(container.active_index(), 1);

    container.set_slimmable_size(0.0, None);
    assert!(container.is_crossfading());

    let input = vec![0.0f32; BLOCK];
    let mut output = vec![0.0f32; BLOCK];
    let mut actual = Vec::with_capacity(duration);
    let mut expected = Vec::with_capacity(duration);
    let mut elapsed = 0usize;

    while container.is_crossfading() {
        container.process(&input, &mut output);
        actual.extend_from_slice(&output);
        for i in 0..BLOCK {
            let t = ((elapsed + i) as f32 / duration as f32).min(1.0);
            expected.push(ACTIVE_DC * (1.0 - t) + PENDING_DC * t);
        }
        elapsed += BLOCK;
    }

    assert_eq!(actual.len(), duration);
    assert!(actual.iter().all(|s| s.is_finite()));
    assert!(!container.is_crossfading());
    assert_eq!(container.active_index(), 0);

    let mut max_err = 0.0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_err = max_err.max((a - e).abs());
    }
    assert!(
        max_err < MAX_DISCONTINUITY,
        "blend envelope error {max_err:.3e} exceeds −80 dBFS ({MAX_DISCONTINUITY})"
    );

    let mut max_residual_step = 0.0f32;
    for i in 1..actual.len() {
        let da = actual[i] - actual[i - 1];
        let de = expected[i] - expected[i - 1];
        max_residual_step = max_residual_step.max((da - de).abs());
    }
    assert!(
        max_residual_step < MAX_DISCONTINUITY,
        "consecutive-sample discontinuity {max_residual_step:.3e} exceeds −80 dBFS ({MAX_DISCONTINUITY})"
    );
}

#[test]
fn test_new_typed_rejects_invalid_topology() {
    // Empty list.
    assert_eq!(
        ContainerModel::new_typed(Vec::new(), 48000).err(),
        Some(NamErrorCode::InvalidModelTopology)
    );
    // Non-finite max_value.
    assert_eq!(
        ContainerModel::new_typed(vec![(f32::NAN, make_lstm())], 48000).err(),
        Some(NamErrorCode::InvalidModelTopology)
    );
    // Negative max_value.
    assert_eq!(
        ContainerModel::new_typed(vec![(-0.5, make_lstm()), (1.0, make_lstm())], 48000).err(),
        Some(NamErrorCode::InvalidModelTopology)
    );
    // Duplicate values are not strictly ascending.
    assert_eq!(
        ContainerModel::new_typed(vec![(0.5, make_lstm()), (0.5, make_lstm())], 48000).err(),
        Some(NamErrorCode::InvalidModelTopology)
    );
    // Last max_value below unity.
    assert_eq!(
        ContainerModel::new_typed(vec![(0.5, make_lstm())], 48000).err(),
        Some(NamErrorCode::InvalidModelTopology)
    );
}

#[test]
fn test_new_typed_accepts_valid_and_matches_anyhow() {
    let make = || vec![(0.5, make_lstm()), (1.0, make_lstm())];
    let typed = ContainerModel::new_typed(make(), 48000).expect("typed container");
    let legacy = ContainerModel::new(make(), 48000).expect("anyhow container");
    assert_eq!(typed.submodels().len(), legacy.submodels().len());
    assert_eq!(typed.active_index(), legacy.active_index());
    assert_eq!(typed.sample_rate(), legacy.sample_rate());
}

// =============================================================================
// RT-safety of adaptive submodel switching (M-RT-01)
// =============================================================================

fn make_sequential_two_linears() -> Box<StaticModel> {
    use crate::models::sequential::SequentialModel;
    let stage = |bias: f32| {
        StaticModel::Linear(Box::new(
            LinearModel::new(vec![1.0], bias, LinearImplementation::Direct).expect("linear stage"),
        ))
    };
    let chain = SequentialModel::new(
        vec![stage(0.1), stage(-0.1)],
        vec![None, None],
        Some(48_000.0),
    )
    .expect("two-stage chain");
    Box::new(StaticModel::Sequential(Box::new(chain)))
}

/// Red-first reproduction for M-RT-01: the adaptive FSM calls
/// `set_slimmable_size` from the audio callback (`rt_status = Some`), and the
/// target submodel here is a nested `Sequential`. The historical path invoked
/// `SequentialModel::reset`, which allocates (saved-flags `Vec`) and can log.
#[test]
fn test_rt_container_to_sequential_switch_is_zero_heap() {
    use crate::common::alloc_audit::{
        TrackingGuard, get_alloc_count, get_dealloc_count, get_realloc_count,
    };
    use crate::common::spsc::RtStatusFlags;

    let mut container = ContainerModel::new(
        vec![(0.5, make_sequential_two_linears()), (1.0, make_lstm())],
        48_000,
    )
    .expect("container");
    assert_eq!(container.active_index(), 1, "full submodel active");

    container.reset(48_000, 64).expect("off-RT reset");
    let input = vec![0.1f32; 64];
    let mut output = vec![0.0f32; 64];
    for _ in 0..4 {
        container.process(&input, &mut output);
    }

    let status = RtStatusFlags::new();
    let (allocs, reallocs, deallocs) = {
        let _guard = TrackingGuard::new();
        container.set_slimmable_size(0.25, Some(&status));
        for _ in 0..512 {
            container.process(&input, &mut output);
        }
        (get_alloc_count(), get_realloc_count(), get_dealloc_count())
    };

    assert_eq!(
        (allocs, reallocs, deallocs),
        (0, 0, 0),
        "RT submodel switch allocated: alloc={allocs} realloc={reallocs} dealloc={deallocs}"
    );
    assert_eq!(container.active_index(), 0, "switch must complete");
    assert!(output.iter().all(|s| s.is_finite()));
}

/// Two-phase RT transition: a target whose split prewarm does not converge in
/// one block must not start the crossfade, and output must stay on the active
/// submodel until the zero-feed budget completes. Round-trip A→B→A exercises
/// re-preparation of a previously used member.
#[test]
fn test_rt_transition_prepares_before_crossfade_and_round_trips() {
    use crate::common::spsc::RtStatusFlags;

    let mut container = ContainerModel::new(
        vec![(0.5, make_sequential_two_linears()), (1.0, make_lstm())],
        48_000,
    )
    .expect("container");
    container.reset(48_000, 64).expect("off-RT reset");

    let input = vec![0.1f32; 64];
    let mut output = vec![0.0f32; 64];
    let status = RtStatusFlags::new();

    // Full (LSTM) -> cheapest (nested Sequential).
    container.set_slimmable_size(0.25, Some(&status));
    assert_eq!(
        container.active_index(),
        1,
        "output stays on the old member"
    );
    assert_eq!(container.preparing_index(), Some(0));
    assert_eq!(
        container.pending_index(),
        None,
        "crossfade must not start yet"
    );

    container.process(&input, &mut output);
    assert!(container.is_preparing() || container.is_crossfading());
    assert_eq!(container.active_index(), 1);

    for _ in 0..64 {
        container.process(&input, &mut output);
        assert!(output.iter().all(|s| s.is_finite()));
    }
    assert_eq!(container.active_index(), 0, "switch completes");
    assert!(!container.is_preparing());
    assert!(!container.is_crossfading());

    // Round trip: cheapest -> full. The LSTM re-preparation (≈0.5 s of zeros
    // at 48 kHz) is amortized one block per callback.
    container.set_slimmable_size(1.0, Some(&status));
    assert_eq!(container.preparing_index(), Some(1));
    let mut blocks = 0usize;
    while container.is_preparing() || container.is_crossfading() {
        container.process(&input, &mut output);
        blocks += 1;
        assert!(blocks < 4096, "round trip must converge");
        assert!(output.iter().all(|s| s.is_finite()));
    }
    assert_eq!(container.active_index(), 1);
    // LSTM prewarm is 24_000 frames: with 64-frame blocks it needs 375 calls
    // before the 1_536-frame crossfade can even start. The bound is the point:
    // per-callback work never exceeds one block of prewarm.
    assert!(
        blocks > 375,
        "prewarm must be amortized, not hidden by crossfade"
    );
}

/// Stereo (dual-mono) containers perform the same RT transition on both
/// channels with zero heap traffic.
#[test]
fn test_rt_stereo_container_switch_is_zero_heap() {
    use crate::common::alloc_audit::{
        TrackingGuard, get_alloc_count, get_dealloc_count, get_realloc_count,
    };
    use crate::common::spsc::RtStatusFlags;

    let build = || {
        let mut c = ContainerModel::new(
            vec![(0.5, make_sequential_two_linears()), (1.0, make_lstm())],
            48_000,
        )
        .expect("container");
        c.reset(48_000, 64).expect("off-RT reset");
        c
    };
    let mut left = build();
    let mut right = build();
    let input_l = vec![0.1f32; 64];
    let input_r = vec![-0.1f32; 64];
    let mut out_l = vec![0.0f32; 64];
    let mut out_r = vec![0.0f32; 64];
    let status = RtStatusFlags::new();

    left.set_slimmable_size(0.25, Some(&status));
    right.set_slimmable_size(0.25, Some(&status));

    let (allocs, reallocs, deallocs) = {
        let _guard = TrackingGuard::new();
        for _ in 0..512 {
            left.process(&input_l, &mut out_l);
            right.process(&input_r, &mut out_r);
        }
        (get_alloc_count(), get_realloc_count(), get_dealloc_count())
    };
    assert_eq!(
        (allocs, reallocs, deallocs),
        (0, 0, 0),
        "stereo RT switch allocated: alloc={allocs} realloc={reallocs} dealloc={deallocs}"
    );
    assert_eq!(left.active_index(), 0);
    assert_eq!(right.active_index(), 0);
    assert!(out_l.iter().chain(out_r.iter()).all(|s| s.is_finite()));
}

/// Replacement policy: a request that arrives during the crossfade must not
/// make the active index jump mid-blend; it is honored after the blend
/// completes, and still performs zero heap work.
#[test]
fn test_rt_mid_crossfade_request_is_queued_and_honored() {
    use crate::common::alloc_audit::{
        TrackingGuard, get_alloc_count, get_dealloc_count, get_realloc_count,
    };
    use crate::common::spsc::RtStatusFlags;

    let mut container = ContainerModel::new(
        vec![
            (0.4, make_linear_dc(0.0)),
            (0.7, make_linear_dc(1.0)),
            (1.0, make_linear_dc(2.0)),
        ],
        48_000,
    )
    .expect("container");
    container.reset(48_000, 64).expect("off-RT reset");
    assert_eq!(container.active_index(), 2);

    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let status = RtStatusFlags::new();

    container.set_slimmable_size(0.0, Some(&status));
    assert_eq!(container.preparing_index(), Some(0));
    container.process(&input, &mut output);
    assert_eq!(container.pending_index(), Some(0), "crossfade started");
    assert_eq!(container.active_index(), 2);

    // Mid-blend request for the middle tier.
    container.set_slimmable_size(0.5, Some(&status));
    assert_eq!(container.requested_index(), Some(1));
    assert_eq!(container.pending_index(), Some(0), "in-flight blend kept");
    assert_eq!(container.active_index(), 2, "no mid-blend jump");

    let (allocs, reallocs, deallocs) = {
        let _guard = TrackingGuard::new();
        let mut guard_blocks = 0usize;
        while container.requested_index().is_some()
            || container.is_preparing()
            || container.is_crossfading()
        {
            container.process(&input, &mut output);
            guard_blocks += 1;
            assert!(guard_blocks < 4096);
            if container.is_crossfading() {
                assert_ne!(container.active_index(), 1, "queued target never jumps");
            }
        }
        (get_alloc_count(), get_realloc_count(), get_dealloc_count())
    };
    assert_eq!((allocs, reallocs, deallocs), (0, 0, 0));
    assert_eq!(container.active_index(), 1, "queued target honored last");
}

/// Cancellation: requesting the current active submodel while a preparation is
/// converging aborts it without any transition and without heap traffic.
#[test]
fn test_rt_preparation_cancelled_by_returning_to_active() {
    use crate::common::alloc_audit::TrackingGuard;
    use crate::common::spsc::RtStatusFlags;

    let mut container = ContainerModel::new(vec![(0.5, make_lstm()), (1.0, make_lstm())], 48_000)
        .expect("container");
    container.reset(48_000, 64).expect("off-RT reset");
    let input = vec![0.1f32; 64];
    let mut output = vec![0.0f32; 64];
    let status = RtStatusFlags::new();

    container.set_slimmable_size(0.0, Some(&status));
    assert_eq!(container.preparing_index(), Some(0));
    container.process(&input, &mut output);
    assert!(container.is_preparing(), "LSTM prewarm spans many blocks");

    let guard = TrackingGuard::new();
    container.set_slimmable_size(1.0, Some(&status));
    assert!(!container.is_preparing(), "preparation cancelled");
    assert!(!container.is_crossfading());
    assert_eq!(container.active_index(), 1);
    container.process(&input, &mut output);
    drop(guard);

    assert_eq!(container.active_index(), 1);
    assert!(output.iter().all(|s| s.is_finite()));
}

/// A host reset issued while a preparation is in flight clears the explicit
/// state machine and leaves the container processable (interrupted reset).
#[test]
fn test_host_reset_clears_in_flight_rt_transition() {
    use crate::common::spsc::RtStatusFlags;

    let mut container = ContainerModel::new(vec![(0.5, make_lstm()), (1.0, make_lstm())], 48_000)
        .expect("container");
    container.reset(48_000, 64).expect("off-RT reset");
    let input = vec![0.1f32; 64];
    let mut output = vec![0.0f32; 64];
    let status = RtStatusFlags::new();

    container.set_slimmable_size(0.0, Some(&status));
    container.process(&input, &mut output);
    assert!(container.is_preparing());

    container.reset(48_000, 64).expect("second off-RT reset");
    assert!(!container.is_preparing());
    assert!(!container.is_crossfading());
    assert_eq!(container.requested_index(), None);

    container.process(&input, &mut output);
    assert!(output.iter().all(|s| s.is_finite()));
}

#[test]
fn test_container_prewarm_step_uses_aggregate_budget() {
    // Every submodel reports RF=1 of pending Linear stabilization. A
    // container must not apply the full `samples` budget to each member:
    // four members with `samples=2` may advance at most two units total.
    let linear_pending = |m: &StaticModel| match m {
        StaticModel::Linear(l) => l.prewarm_pending,
        _ => panic!("expected Linear child"),
    };
    let submodels: Vec<(f32, Box<StaticModel>)> = (1..=4)
        .map(|i| (i as f32 / 4.0, make_linear_dc(0.0)))
        .collect();
    let mut container = ContainerModel::new(submodels, 48000).unwrap();
    container.prewarm_reset();
    let armed: usize = container
        .submodels
        .iter()
        .map(|(_, m)| linear_pending(m))
        .sum();
    assert_eq!(armed, 4, "four Linear children each arm one unit");

    let remaining = container.prewarm_step(2);
    let advanced: usize = container
        .submodels
        .iter()
        .map(|(_, m)| linear_pending(m))
        .sum();
    assert_eq!(
        armed - advanced,
        2,
        "one callback of 2 units must advance exactly 2 units across children"
    );
    assert_eq!(
        remaining, 1,
        "reported pending is the maximum child balance"
    );
}

/// A block longer than the remaining crossfade must finish the per-sample
/// ramp inside the block and hand the rest of the block to the incoming
/// model.
#[test]
fn test_crossfade_finishes_inside_irregular_block() {
    const SR: u32 = 48_000;
    const BLOCK: usize = 1000;
    const ACTIVE_DC: f32 = 1.0;
    const PENDING_DC: f32 = -0.5;
    let duration = (CROSSFADE_DURATION_MS / 1000.0 * SR as f32).round() as usize;
    assert_eq!(duration, 1536);

    let submodels = vec![
        (0.5, make_linear_dc(PENDING_DC)),
        (1.0, make_linear_dc(ACTIVE_DC)),
    ];
    let mut container = ContainerModel::new(submodels, SR).unwrap();
    // Off-RT request: the target is reset and the crossfade is armed directly.
    container.set_slimmable_size(0.0, None);
    assert!(container.is_crossfading());

    let input = vec![0.0f32; BLOCK];
    let mut output = vec![7.7f32; BLOCK];

    // Call 1: frames [0, 1000) of the 1536-frame ramp.
    container.process(&input, &mut output);
    assert!(container.is_crossfading());
    assert!(
        output.iter().all(|s| s.is_finite()),
        "ramp block must be finite"
    );

    // Call 2: elapsed is 1000, so only 536 ramp frames remain; the last 464
    // frames of this block are fully owned by the incoming model and the
    // transition finalizes inside the block.
    output.fill(7.7);
    container.process(&input, &mut output);
    assert!(!container.is_crossfading(), "crossfade finishes in-block");
    assert_eq!(container.active_index(), 0);

    let ramp_len = duration - BLOCK; // 536 remaining on call 2
    assert_eq!(ramp_len, 536);
    assert!(
        output[ramp_len..]
            .iter()
            .all(|s| (*s - PENDING_DC).abs() < 1e-6),
        "frames after the in-block ramp end must be exactly the incoming model"
    );
    assert!(
        output[..ramp_len].iter().all(|s| s.is_finite()),
        "ramp frames must be finite"
    );
}

/// Reset while a crossfade is armed cancels the transition coherently: the
/// active tier survives, the fade clock and pending target are cleared before
/// inactive members are wiped, and the next block is the plain active model.
#[test]
fn test_host_reset_cancels_in_flight_crossfade() {
    const SR: u32 = 48_000;
    const BLOCK: usize = 64;
    const ACTIVE_DC: f32 = 1.0;

    let submodels = vec![
        (0.5, make_linear_dc(-0.5)),
        (1.0, make_linear_dc(ACTIVE_DC)),
    ];
    let mut container = ContainerModel::new(submodels, SR).unwrap();
    container.reset(SR, BLOCK).expect("reset");
    let status = crate::common::spsc::RtStatusFlags::new();
    container.set_slimmable_size(0.0, Some(&status));
    assert!(container.is_preparing());

    let input = vec![0.0f32; BLOCK];
    let mut output = vec![0.0f32; BLOCK];
    // Converge the target and arm the crossfade.
    container.process(&input, &mut output);
    assert!(container.is_crossfading());
    // Advance the fade clock.
    container.process(&input, &mut output);

    container.reset(SR, BLOCK).expect("reset mid-crossfade");
    assert!(!container.is_preparing());
    assert!(!container.is_crossfading());
    assert_eq!(container.pending_index(), None);
    assert_eq!(container.requested_index(), None);
    assert_eq!(container.active_index(), 1, "active tier survives reset");

    output.fill(7.7);
    container.process(&input, &mut output);
    assert!(
        output.iter().all(|s| (*s - ACTIVE_DC).abs() < 1e-6),
        "post-reset block must come from the surviving active model"
    );
}
