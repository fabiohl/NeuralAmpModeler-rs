// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration tests for `WaveNetA2` via the `NamModel` trait.
//!
//! The inline unit tests in `model.rs` cover internal mechanics (weight loading,
//! error cases, weight count arithmetic). This file exercises the *public* interface
//! via the `NamModel` trait to catch regressions at the trait boundary.

#[cfg(test)]
mod tests_a2_real {
    use crate::models::NamModel;
    use crate::models::a2::WaveNetA2;
    use crate::models::a2::a2_weight_count;

    // ─── Weight helpers ─────────────────────────────────────────────────────

    /// Generates a deterministic weight stream of length `n`.
    fn make_weights(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s as f32 / u32::MAX as f32) * 0.2 - 0.1
            })
            .collect()
    }

    // ─── Legacy smoke tests (kept for historical coverage) ──────────────────

    /// Smoke: A2-Lite without weights outputs finite silence.
    #[test]
    fn test_wavenet_a2_lite_process() {
        let mut model = WaveNetA2::<3>::new().unwrap();
        model.prewarm();
        let input = vec![0.01f32; 64];
        let mut output = vec![0.0f32; 64];
        model.process(&input, &mut output);
        for &s in output.iter() {
            assert!(s.is_finite(), "A2-Lite output must be finite (no weights)");
        }
    }

    /// Smoke: A2-Full without weights outputs finite silence.
    #[test]
    fn test_wavenet_a2_full_process() {
        let mut model = WaveNetA2::<8>::new().unwrap();
        model.prewarm();
        let input = vec![0.01f32; 64];
        let mut output = vec![0.0f32; 64];
        model.process(&input, &mut output);
        for &s in output.iter() {
            assert!(s.is_finite(), "A2-Full output must be finite (no weights)");
        }
    }

    // ─── Trait-boundary tests ────────────────────────────────────────────────

    /// Validates `NamModel::process` for A2-Lite with loaded weights
    /// produces non-zero, finite output — exercises the full trait path.
    #[test]
    fn test_a2_lite_via_nam_model_process() {
        let mut model = WaveNetA2::<3>::new().unwrap();
        let weights = make_weights(a2_weight_count::<3>(), 42);
        model
            .set_weights(&weights)
            .expect("A2-Lite set_weights failed");
        model.prewarm();

        let input = vec![0.5f32; 64];
        let mut output = vec![0.0f32; 64];
        // Call through NamModel trait
        NamModel::process(&mut model, &input, &mut output);

        let any_nonzero = output.iter().any(|&v| v.abs() > 1e-30);
        assert!(
            any_nonzero,
            "A2-Lite NamModel::process produced all-zero output with loaded weights"
        );
        assert!(
            output.iter().all(|v| v.is_finite()),
            "A2-Lite NamModel::process produced non-finite output"
        );
    }

    /// Validates `NamModel::process` for A2-Full with loaded weights
    /// produces non-zero, finite output — exercises the full trait path.
    #[test]
    fn test_a2_full_via_nam_model_process() {
        let mut model = WaveNetA2::<8>::new().unwrap();
        let weights = make_weights(a2_weight_count::<8>(), 77);
        model
            .set_weights(&weights)
            .expect("A2-Full set_weights failed");
        model.prewarm();

        let input = vec![0.5f32; 64];
        let mut output = vec![0.0f32; 64];
        NamModel::process(&mut model, &input, &mut output);

        let any_nonzero = output.iter().any(|&v| v.abs() > 1e-30);
        assert!(
            any_nonzero,
            "A2-Full NamModel::process produced all-zero output with loaded weights"
        );
        assert!(
            output.iter().all(|v| v.is_finite()),
            "A2-Full NamModel::process produced non-finite output"
        );
    }

    /// Validates that `prewarm_samples()` returns exactly the receptive field size.
    #[test]
    fn test_a2_prewarm_samples_returns_rf() {
        let model_lite = WaveNetA2::<3>::new().unwrap();
        let model_full = WaveNetA2::<8>::new().unwrap();

        let rf_lite = model_lite.receptive_field();
        let rf_full = model_full.receptive_field();

        assert_eq!(
            model_lite.prewarm_samples(),
            rf_lite,
            "A2-Lite: prewarm_samples() must equal receptive_field_size"
        );
        assert_eq!(
            model_full.prewarm_samples(),
            rf_full,
            "A2-Full: prewarm_samples() must equal receptive_field_size"
        );
        // Sanity: full must be larger than lite (same layers, more channels → same RF,
        // but both are architecture-defined constants — assert they are non-zero)
        assert!(rf_lite > 0, "A2-Lite RF must be > 0");
        assert!(rf_full > 0, "A2-Full RF must be > 0");
        // Both share the same 23-layer dilation pattern; RF is channel-independent
        assert_eq!(
            rf_lite, rf_full,
            "A2 RF is topology-fixed, independent of CH"
        );
    }

    /// Validates that reset + prewarm gives bitwise-identical output for identical inputs.
    ///
    /// This is the determinism invariant: two runs with reset in between must
    /// produce exactly the same sequence of output samples.
    #[test]
    fn test_a2_reset_determinism() {
        let mut model = WaveNetA2::<3>::new().unwrap();
        let weights = make_weights(a2_weight_count::<3>(), 123);
        model.set_weights(&weights).expect("set_weights failed");
        model.prewarm();

        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).sin()).collect();

        // First run
        let mut out_a = vec![0.0f32; 64];
        model.process(&input, &mut out_a);

        // Reset + identical prewarm
        model.reset(48000, 64).unwrap();
        model.prewarm();

        // Second run with same input
        let mut out_b = vec![0.0f32; 64];
        model.process(&input, &mut out_b);

        // Must be bitwise identical
        for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "A2-Lite: frame {i} differs after reset — a={a}, b={b}"
            );
        }
    }
}
