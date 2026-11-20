// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_smoother_convergence() {
    let mut smoother = ParamSmoother::new(0.0, 48000.0, 20.0);
    smoother.set_target(1.0);

    // Should converge gradually
    let mut last_val = 0.0;
    for _ in 0..1000 {
        let current = smoother.tick();
        assert!(current >= last_val);
        last_val = current;
    }

    assert!(last_val > 0.5); // At 1000 samples @ 48k with 20Hz (~20ms), should already be pretty far along
}

#[test]
fn test_smoother_snap() {
    let mut smoother = ParamSmoother::new(0.0, 48000.0, 20.0);
    smoother.set_target(1.0);
    smoother.snap_to_target();
    assert_eq!(smoother.tick(), 1.0);
}

#[test]
fn test_smoother_convergence_high_gain() {
    // Verify that for target = 3.98 (≈ +12dB), the smoother converges within ≤ 2400 samples at 48kHz (50ms).
    // Note: The 45Hz cutoff perfectly illustrates the benefit of the relative threshold,
    // since with a fixed threshold (1e-6) convergence would take 2581 samples (exceeding 2400),
    // while the relative threshold allows convergence in 2347 samples.
    let mut smoother = ParamSmoother::new(0.0, 48000.0, 45.0);
    smoother.set_target(3.98);

    let mut samples = 0;
    for _ in 0..5000 {
        let current = smoother.tick();
        samples += 1;
        if current == 3.98 {
            break;
        }
    }
    assert!(
        samples <= 2400,
        "Convergence took {} samples (expected <= 2400)",
        samples
    );
}

#[test]
fn test_smoother_denormal_prevention() {
    // Verify that for target = 0.0 and initial = 1e-20, tick() returns exactly 0.0 after ≤ 10 iterations.
    let mut smoother = ParamSmoother::new(1e-20, 48000.0, 20.0);
    smoother.set_target(0.0);

    let mut converged = false;
    for _ in 0..10 {
        if smoother.tick() == 0.0 {
            converged = true;
            break;
        }
    }
    assert!(converged, "Did not converge to 0.0 in 10 iterations");
}

#[test]
fn test_smoother_relative_threshold() {
    // Verify that target = 0.001 still converges correctly (no premature snap).
    let mut smoother = ParamSmoother::new(0.0, 48000.0, 20.0);
    smoother.set_target(0.001);

    // The first tick should not hit 0.001 immediately (premature snap).
    let val1 = smoother.tick();
    assert!(val1 > 0.0);
    assert!(val1 < 0.001);

    // Should eventually converge
    let mut converged = false;
    for _ in 0..5000 {
        if smoother.tick() == 0.001 {
            converged = true;
            break;
        }
    }
    assert!(converged, "Should converge to 0.001");
}

#[test]
fn test_smoother_rejects_non_finite_and_recovers() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut s = ParamSmoother::new(bad, 48000.0, 20.0);
        assert_eq!(s.current_value(), 0.0);
        assert_eq!(s.target_value(), 0.0);
        assert!(s.tick().is_finite());

        let mut s_alpha = ParamSmoother::new(1.0, bad, 20.0);
        assert_eq!(s_alpha.alpha(), 1.0);
        s_alpha.set_target(0.5);
        assert_eq!(s_alpha.tick(), 0.5);

        let mut s_cutoff = ParamSmoother::new(1.0, 48000.0, bad);
        assert_eq!(s_cutoff.alpha(), 1.0);
        assert!(s_cutoff.tick().is_finite());

        s.set_target(0.75);
        s.set(0.25);
        s.set_target(bad);
        s.set(bad);
        assert_eq!(s.target_value(), 0.75);
        assert_eq!(s.current_value(), 0.25);
        let ticked = s.tick();
        assert!(ticked.is_finite());
        assert!(ticked > 0.25 && ticked <= 0.75);
    }

    // Even if current were corrupted directly (in-crate whitebox test), tick() recovers cleanly.
    let mut corrupted = ParamSmoother::new(1.0, 48000.0, 20.0);
    corrupted.current = f32::NAN;
    let recovered = corrupted.tick();
    assert_eq!(recovered, 1.0);
    assert!(corrupted.current_value().is_finite());
}
