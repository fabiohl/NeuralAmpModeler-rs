// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Property-Based Tests for the Adaptive Compute FSM.
//
//  Evaluates the behavior of the adaptive degradation hysteresis FSM under
//  adversarial input sequences, verifying transitions, invariants, and telemetry
//  consistency (RtStatusFlags).

use neural_amp_modeler_rs::common::spsc::{
    RT_STATUS_DEGRADE_MINIMAL, RT_STATUS_DEGRADE_REDUCED, RtStatusFlags,
};
use neural_amp_modeler_rs::dsp::adaptive::*;
use proptest::prelude::*;
use std::sync::atomic::Ordering;

prop_compose! {
    /// Strategy to generate adaptive FSM configurations.
    fn adaptive_config_strategy()(
        mode in prop_oneof![
            Just(AdaptiveComputeMode::Conservative),
            Just(AdaptiveComputeMode::Aggressive),
        ],
    ) -> AdaptiveComputeMode {
        mode
    }
}

/// Strategy for `budget_us` values — realistic DSP block budgets.
fn budget_strategy() -> impl Strategy<Value = u64> {
    prop_oneof![
        100u64..=1000u64,     // Tiny blocks
        1001u64..=10000u64,   // Standard blocks
        10001u64..=100000u64, // Large blocks
    ]
}

/// Strategy for sample rates.
fn sample_rate_strategy() -> impl Strategy<Value = u32> {
    prop_oneof![
        Just(44100u32),
        Just(48000u32),
        Just(88200u32),
        Just(96000u32),
    ]
}

/// Strategy for block sizes.
fn block_size_strategy() -> impl Strategy<Value = usize> {
    prop_oneof![
        1..=1usize,       // Unit blocks
        2..=128usize,     // Small blocks
        129..=1024usize,  // Standard blocks
        1025..=4096usize, // Large blocks
    ]
}

/// Verification helper for FSM invariants.
fn verify_adaptive_step(
    adaptive: &mut AdaptiveCompute,
    latency_us: u64,
    budget_us: u64,
    sample_rate: u32,
    rt_status: &RtStatusFlags,
    transitions_before: u32,
) -> u32 {
    adaptive.update(latency_us, budget_us, sample_rate, rt_status);

    let state = adaptive.state();

    // 1. State must be one of the three valid variants
    match state {
        AdaptiveState::Full | AdaptiveState::Reduced | AdaptiveState::Minimal => {}
    }

    // 2. slimmable_size() must be in [0.0, 1.0]
    let sz = adaptive.slimmable_size();
    assert!(
        (0.0..=1.0).contains(&sz),
        "slimmable_size() {} out of [0.0, 1.0] at state {:?}",
        sz,
        state
    );

    // 3. Crossfade sanity — if crossfading, multiplier must be in [0.0, 1.0]
    if adaptive.is_crossfading() {
        let mult = adaptive.crossfade_multiplier(sample_rate, 1);
        assert!(
            (0.0..=1.0).contains(&mult),
            "crossfade_multiplier {} out of [0.0, 1.0]",
            mult
        );
    }

    // 4. RT status flags must match the degradation state
    let is_reduced = rt_status.check_flag(RT_STATUS_DEGRADE_REDUCED);
    let is_minimal = rt_status.check_flag(RT_STATUS_DEGRADE_MINIMAL);

    match state {
        AdaptiveState::Full => {
            assert!(!is_reduced, "DEGRADE_REDUCED set in Full state");
            assert!(!is_minimal, "DEGRADE_MINIMAL set in Full state");
        }
        AdaptiveState::Reduced => {
            assert!(is_reduced, "DEGRADE_REDUCED not set in Reduced state");
            assert!(!is_minimal, "DEGRADE_MINIMAL set in Reduced state");
        }
        AdaptiveState::Minimal => {
            assert!(is_reduced, "DEGRADE_REDUCED not set in Minimal state");
            assert!(is_minimal, "DEGRADE_MINIMAL not set in Minimal state");
        }
    }

    // 5. degrade_transitions_total must be monotonically non-decreasing
    let transitions_after = rt_status.degrade_transitions_total.load(Ordering::Relaxed);
    assert!(
        transitions_after >= transitions_before,
        "degrade_transitions_total decreased: {} -> {}",
        transitions_before,
        transitions_after
    );

    // 6. wavenet_effective_layers must be in [1, total] for any total
    let el = adaptive.wavenet_effective_layers(8);
    assert!(
        (1..=8).contains(&el),
        "wavenet_effective_layers(8) = {} out of [1, 8]",
        el
    );
    let el_min = adaptive.wavenet_effective_layers(1);
    assert_eq!(el_min, 1, "wavenet_effective_layers(1) must be 1");

    // 7. If mode is Off (handled in a separate test), only check if mode = Off
    //    (the mode can change from Conservative/Aggressive to Off via set_mode)

    transitions_after
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(10_000)
    })]

    /// Runs adversarial proptest with randomized mode, budget, block size,
    /// sample rate, and latency sequences.
    #[test]
    #[ignore]
    fn test_adaptive_fsm_adversarial_sequences(
        mode in adaptive_config_strategy(),
        budget in budget_strategy(),
        sample_rate in sample_rate_strategy(),
        _block_size in block_size_strategy(),
        latency_ratios in proptest::collection::vec(0.01f64..2.0f64, 128),
    ) {
        let rt_status = RtStatusFlags::default();
        let mut adaptive = AdaptiveCompute::new(mode);
        let mut transitions = 0u32;

        for ratio in latency_ratios {
            let latency = ((budget as f64) * ratio).ceil() as u64;
            let latency_clamped = latency.max(1);
            transitions = verify_adaptive_step(
                &mut adaptive,
                latency_clamped,
                budget,
                sample_rate,
                &rt_status,
                transitions,
            );
        }
    }

    /// Sweeps exact threshold boundaries: latency values exactly at the
    /// degradation and recovery thresholds.
    #[test]
    #[ignore]
    fn test_adaptive_fsm_boundary_values(
        mode in adaptive_config_strategy(),
        budget in budget_strategy(),
        sample_rate in sample_rate_strategy(),
    ) {
        if mode == AdaptiveComputeMode::Off {
            return Ok(());
        }

        let rt_status = RtStatusFlags::default();
        let mut adaptive = AdaptiveCompute::new(mode);

        let (full_to_reduced, reduced_to_minimal) = match mode {
            AdaptiveComputeMode::Conservative => (0.70f32, 0.85f32),
            AdaptiveComputeMode::Aggressive => (0.55f32, 0.70f32),
            AdaptiveComputeMode::Off => unreachable!(),
        };

        let recovery_reduced = full_to_reduced * 0.5;
        let recovery_minimal = reduced_to_minimal * 0.5;

        let bf = budget as f32;

        let boundary_latencies = vec![
            (bf * full_to_reduced).ceil() as u64,      // Exact degrade threshold Full→Reduced
            (bf * full_to_reduced).floor() as u64,     // Just below
            (bf * full_to_reduced).ceil() as u64 + 1,  // Just above
            (bf * reduced_to_minimal).ceil() as u64,   // Exact degrade threshold Reduced→Minimal
            (bf * reduced_to_minimal).floor() as u64,  // Just below
            (bf * reduced_to_minimal).ceil() as u64 + 1, // Just above
            (bf * recovery_reduced).ceil() as u64,     // Exact recovery Reduced→Full
            (bf * recovery_reduced).floor() as u64,    // Just below
            (bf * recovery_minimal).ceil() as u64,     // Exact recovery Minimal→Reduced
            (bf * recovery_minimal).floor() as u64,    // Just below
            (bf * 0.50).ceil() as u64,                 // Mid-point
            budget,                                    // Exactly budget
        ];

        let mut transitions = 0u32;

        for _ in 0..100 {
            for &latency in &boundary_latencies {
                transitions = verify_adaptive_step(
                    &mut adaptive,
                    latency.max(1),
                    budget,
                    sample_rate,
                    &rt_status,
                    transitions,
                );
            }
        }
    }

    /// Simulates rapid jitter: alternates between overload and recovery
    /// to challenge consecutive-count logic and crossfade.
    #[test]
    #[ignore]
    fn test_adaptive_fsm_jitter(
        mode in adaptive_config_strategy(),
        budget in budget_strategy(),
        sample_rate in sample_rate_strategy(),
        sequence_len in 100..300usize,
    ) {
        if mode == AdaptiveComputeMode::Off {
            return Ok(());
        }

        let rt_status = RtStatusFlags::default();
        let mut adaptive = AdaptiveCompute::new(mode);

        let (full_to_reduced, _) = match mode {
            AdaptiveComputeMode::Conservative => (0.70f32, 0.85f32),
            AdaptiveComputeMode::Aggressive => (0.55f32, 0.70f32),
            AdaptiveComputeMode::Off => unreachable!(),
        };

        let recovery = full_to_reduced * 0.5;
        let bf = budget as f32;

        let overload_latency = (bf * full_to_reduced * 1.1).ceil() as u64;
        let recovery_latency = (bf * recovery * 0.9).floor() as u64;

        let mut transitions = 0u32;

        for i in 0..sequence_len {
            // Alternate patterns: 3 overloads + 5 recoveries, then random
            let latency = match i % 8 {
                0..=2 => overload_latency.max(1),
                3..=7 => recovery_latency.max(1),
                _ => budget / 2,
            };

            transitions = verify_adaptive_step(
                &mut adaptive,
                latency,
                budget,
                sample_rate,
                &rt_status,
                transitions,
            );
        }
    }

    /// Verifies that mode Off completely disables the FSM.
    #[test]
    fn test_adaptive_fsm_off_no_transitions(
        budget in budget_strategy(),
        sample_rate in sample_rate_strategy(),
        latencies in proptest::collection::vec(1u64..100000u64, 64),
    ) {
        let rt_status = RtStatusFlags::default();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

        for latency in latencies {
            adaptive.update(latency, budget, sample_rate, &rt_status);
            assert_eq!(adaptive.state(), AdaptiveState::Full,
                "Mode Off should never change state");
        }

        assert_eq!(
            rt_status.degrade_transitions_total.load(Ordering::Relaxed),
            0,
            "Mode Off should never increment degrade_transitions_total"
        );
    }

    /// Verifies that slimmable_size() returns correct values for each state.
    #[test]
    fn test_adaptive_slimmable_size(
        mode in adaptive_config_strategy(),
        budget in budget_strategy(),
        sample_rate in sample_rate_strategy(),
    ) {
        if mode == AdaptiveComputeMode::Off {
            return Ok(());
        }

        let rt_status = RtStatusFlags::default();
        let mut adaptive = AdaptiveCompute::new(mode);

        // Full state
        assert!((adaptive.slimmable_size() - 1.0).abs() < 1e-6,
            "Full state slimmable_size should be 1.0, got {}", adaptive.slimmable_size());

        let (full_to_reduced, reduced_to_minimal) = match mode {
            AdaptiveComputeMode::Conservative => (0.70f32, 0.85f32),
            AdaptiveComputeMode::Aggressive => (0.55f32, 0.70f32),
            AdaptiveComputeMode::Off => unreachable!(),
        };

        let bf = budget as f32;

        // Degrade to Reduced
        for _ in 0..3 {
            adaptive.update((bf * (full_to_reduced + 0.01)).ceil() as u64, budget, sample_rate, &rt_status);
        }
        assert_eq!(adaptive.state(), AdaptiveState::Reduced);
        assert!((adaptive.slimmable_size() - 0.25).abs() < 1e-6,
            "Reduced state slimmable_size should be 0.25, got {}", adaptive.slimmable_size());

        // Degrade to Minimal
        for _ in 0..3 {
            adaptive.update((bf * (reduced_to_minimal + 0.01)).ceil() as u64, budget, sample_rate, &rt_status);
        }
        assert_eq!(adaptive.state(), AdaptiveState::Minimal);
        assert!((adaptive.slimmable_size() - 0.0).abs() < 1e-6,
            "Minimal state slimmable_size should be 0.0, got {}", adaptive.slimmable_size());
    }
}
