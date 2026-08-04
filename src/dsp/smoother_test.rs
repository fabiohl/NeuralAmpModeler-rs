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
