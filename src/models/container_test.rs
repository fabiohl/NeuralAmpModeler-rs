// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
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

/// S3.T1 acceptance: an 8192-sample block processed while a tier switch is in
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

/// S3.T1 acceptance: when the input block exceeds the scratch capacity, the
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

/// T3.1: 32 ms crossfade must not introduce discontinuities above −80 dBFS
/// between consecutive samples in the blend region (beyond the block-constant
/// envelope of `crossfade_blend_mono_simd`).
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
        let t = (elapsed as f32 / duration as f32).min(1.0);
        let env = ACTIVE_DC * (1.0 - t) + PENDING_DC * t;
        actual.extend_from_slice(&output);
        expected.extend(std::iter::repeat_n(env, BLOCK));
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
