// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Property-Based Tests for the Noise Gate FSM.
//
//  Evaluates the behavior of the gate's Finite State Machine under adversarial
//  input sequences, verifying transitions, boundaries, and invariants.

use neural_amp_modeler_rs::dsp::gate::{DynamicHysteresis, GateParams, GateState};
use proptest::prelude::*;

prop_compose! {
    /// Strategy to generate gate configurations including edge cases.
    fn gate_params_strategy()(
        threshold_open_db in -100.0f32..=0.0f32,
        threshold_close_offset in 0.0f32..=20.0f32,
        hold_frames in 0..=5000usize,
        fade_frames in 0..=5000usize,
        mono_epsilon in 1e-5f32..=1e-3f32,
    ) -> (GateParams, f32, f32) {
        let threshold_close_db = threshold_open_db - threshold_close_offset;
        let params = GateParams::new(
            threshold_open_db,
            threshold_close_db,
            hold_frames,
            fade_frames,
            mono_epsilon,
        );

        // Convert dB to linear power/energy units (as used by the stages)
        let th_open = 10.0f32.powf(threshold_open_db / 10.0);
        let th_close = 10.0f32.powf(threshold_close_db / 10.0);

        (params, th_open, th_close)
    }
}

/// Strategy for block sizes (n_samples).
fn block_size_strategy() -> impl Strategy<Value = usize> {
    prop_oneof![
        1..=1usize,       // Unit blocks
        2..=64usize,      // Small blocks
        65..=1024usize,   // Standard blocks
        1025..=4096usize, // Large blocks
    ]
}

/// Strategy for input energies, generating random inputs with some bias
/// towards the threshold boundaries.
fn energy_sequence_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(0.0f32..=2.0f32, 1024)
}

/// Verification helper for FSM invariants.
fn verify_fsm_step(
    dh: &mut DynamicHysteresis,
    energy: f32,
    th_open: f32,
    th_close: f32,
    params: &GateParams,
    n_samples: usize,
) {
    let prev_state = dh.state();
    let prev_mult = dh.multiplier();

    dh.update(energy, th_open, th_close, params, n_samples);

    let state = dh.state();
    let mult = dh.multiplier();

    // 1. Multiplier must be in [0.0, 1.0] bounds
    assert!(
        (0.0..=1.0).contains(&mult),
        "Multiplier {} out of bounds [0.0, 1.0] at state {:?}",
        mult,
        state
    );

    // 2. Strict state multiplier conditions
    match state {
        GateState::Closed => {
            assert_eq!(mult, 0.0, "Closed state must have 0.0 multiplier");
        }
        GateState::Open => {
            assert_eq!(mult, 1.0, "Open state must have 1.0 multiplier");
        }
        _ => {}
    }

    // 3. Monotonicity checks under steady input zones
    if prev_state == GateState::FadingOut && state == GateState::FadingOut {
        assert!(
            mult <= prev_mult,
            "Multiplier increased during FadingOut: {} -> {}",
            prev_mult,
            mult
        );
    }
    if prev_state == GateState::FadingIn && state == GateState::FadingIn {
        assert!(
            mult >= prev_mult,
            "Multiplier decreased during FadingIn: {} -> {}",
            prev_mult,
            mult
        );
    }

    // 3b. Envelope continuity on reversal: the multiplier jump must be
    // bounded by the block step size (no large discontinuity).
    {
        let max_step = n_samples as f32 * params.inv_fade_frames;
        let is_reversal = (prev_state == GateState::FadingOut && state == GateState::FadingIn)
            || (prev_state == GateState::FadingIn && state == GateState::FadingOut);
        if is_reversal {
            assert!(
                (mult - prev_mult).abs() <= max_step + 1e-6,
                "Reversal discontinuity: {} -> {} (from {:?} to {:?}, max_step={})",
                prev_mult,
                mult,
                prev_state,
                state,
                max_step
            );
        }
    }

    // 4. Valid state transitions check
    match prev_state {
        GateState::Open => {
            assert!(
                state == GateState::Open
                    || state == GateState::FadingOut
                    || (params.fade_frames == 0 && state == GateState::Closed),
                "Invalid transition Open -> {:?}",
                state
            );
        }
        GateState::Closed => {
            assert!(
                state == GateState::Closed
                    || state == GateState::FadingIn
                    || state == GateState::Open,
                "Invalid transition Closed -> {:?}",
                state
            );
        }
        GateState::FadingOut => {
            assert!(
                state == GateState::FadingOut
                    || state == GateState::Closed
                    || state == GateState::FadingIn
                    || state == GateState::Open,
                "Invalid transition FadingOut -> {:?}",
                state
            );
        }
        GateState::FadingIn => {
            assert!(
                state == GateState::FadingIn
                    || state == GateState::Open
                    || state == GateState::FadingOut
                    || state == GateState::Closed,
                "Invalid transition FadingIn -> {:?}",
                state
            );
        }
    }

    // 5. Special check for fade_frames = 0
    if params.fade_frames == 0 {
        match state {
            GateState::FadingIn | GateState::FadingOut => {
                panic!("Should not remain in fading state when fade_frames is 0");
            }
            _ => {}
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(10_000)
    })]

    /// Runs adversarial proptest with randomized parameters, blocks, and input sequences.
    #[test]
    #[ignore]
    fn test_gate_fsm_adversarial_sequences(
        (params, th_open, th_close) in gate_params_strategy(),
        block_size in block_size_strategy(),
        energies in energy_sequence_strategy(),
    ) {
        let mut dh = DynamicHysteresis::new();
        for energy in energies {
            verify_fsm_step(&mut dh, energy, th_open, th_close, &params, block_size);
        }
    }

    /// Specifically sweeps threshold boundaries: energy values exactly equal to th_open or th_close.
    #[test]
    #[ignore]
    fn test_gate_fsm_boundary_values(
        (params, th_open, th_close) in gate_params_strategy(),
        block_size in block_size_strategy(),
    ) {
        let mut dh = DynamicHysteresis::new();
        // Sweep various exact boundary inputs
        let boundary_inputs = vec![
            th_open,
            th_close,
            th_open + 1e-7,
            th_open - 1e-7,
            th_close + 1e-7,
            th_close - 1e-7,
            0.0,
            th_open * 2.0,
        ];

        for _ in 0..100 {
            for &energy in &boundary_inputs {
                verify_fsm_step(&mut dh, energy, th_open, th_close, &params, block_size);
            }
        }
    }

    /// Simulates rapid jitter in/out (alternating above and below thresholds)
    /// to challenge the hold and fade counters.
    #[test]
    #[ignore]
    fn test_gate_fsm_jitter(
        (params, th_open, th_close) in gate_params_strategy(),
        block_size in block_size_strategy(),
        sequence_len in 100..200usize,
    ) {
        let mut dh = DynamicHysteresis::new();
        let high = th_open * 1.5;
        let low = th_close * 0.5;

        for i in 0..sequence_len {
            // Alternate high/low on every step or every few steps
            let energy = if (i % 2) == 0 { high } else { low };
            verify_fsm_step(&mut dh, energy, th_open, th_close, &params, block_size);
        }
    }
}
