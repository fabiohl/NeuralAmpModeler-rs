// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
/// Test module for the Noise Gate.
/// The Noise Gate silences audio when volume falls below a certain level,
/// eliminating unwanted noise (such as hum from an idle guitar) or
/// when we want to mute specific parts of the audio.
mod tests {
    use crate::dsp::gate::*;
    use proptest::prelude::*;

    /// Verifies that the noise gate default settings are correct.
    #[test]
    fn test_gate_params_default() {
        let params = GateParams::default();
        // The default is to start closing at -80dB and opening at -70dB.
        assert_eq!(params.threshold_open_db, -70.0);
        assert_eq!(params.threshold_close_db, -80.0);
        // Time it waits before starting to close (hold) and smoothing time (fade).
        assert_eq!(params.hold_frames, 2048);
        assert_eq!(params.fade_frames, 256);
    }

    /// Tests the basic lifecycle of the noise gate:
    /// Open -> Holding (Hold) -> Closing (FadeOut) -> Closed -> Opening (FadeIn) -> Open.
    #[test]
    fn test_hysteresis_basic_transitions() {
        let mut dh = DynamicHysteresis::new();
        let params = GateParams::new(-10.0, -20.0, 10, 10, 1e-4);
        // We use simple values for the test: 1.0 is "loud", 0.1 is "silence".
        let th_open = 1.0;
        let th_close = 0.5;

        // Starts fully open.
        assert_eq!(dh.state(), GateState::Open);
        assert_eq!(dh.multiplier(), 1.0);

        // 1. Volume drops below the closing threshold.
        // It should remain open for some time (hold period).
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(
            dh.state(),
            GateState::Open,
            "Should remain Open during hold"
        );

        // 2. Hold time expires. Now it starts closing smoothly.
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(
            dh.state(),
            GateState::FadingOut,
            "Should enter FadingOut after hold_frames"
        );

        // 3. Mid-fade out, volume should be at half.
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(dh.multiplier(), 0.5); // 5 of 10 steps completed.

        // 4. Completes the closing. Volume is now zero (fully muted).
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::Closed);
        assert_eq!(dh.multiplier(), 0.0);

        // 5. Sound becomes loud again (above the opening threshold).
        // It should start opening smoothly.
        dh.update(2.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(
            dh.multiplier(),
            0.1,
            "Starts opening immediately when sound returns"
        );

        // 6. Fade-in progress.
        dh.update(2.0, th_open, th_close, &params, 4);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(dh.multiplier(), 0.5);

        // 7. Fully open again.
        dh.update(2.0, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::Open);
        assert_eq!(dh.multiplier(), 1.0);
    }

    /// Tests what happens if sound returns while the gate was still closing.
    /// It should stop closing and start opening immediately from where it stopped.
    #[test]
    fn test_hysteresis_interrupted_fade() {
        let mut dh = DynamicHysteresis::new();
        let params = GateParams::new(-70.0, -80.0, 10, 10, 1e-4);
        let th_open = 1.0;
        let th_close = 0.5;

        // Forces the start of closing (fade out).
        dh.update(0.1, th_open, th_close, &params, 11);
        assert_eq!(dh.state(), GateState::FadingOut);

        // Advances the closing to halfway (multiplier = 0.5).
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.multiplier(), 0.5);

        // Sound comes back loud in the middle of closing!
        // The gate should decide to open from where it was.
        dh.update(2.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(dh.multiplier(), 0.6); // Was at 0.5 and rose to 0.6.

        // Advances the opening a bit more.
        dh.update(2.0, th_open, th_close, &params, 2);
        assert_eq!(dh.multiplier(), 0.8);

        // Sound drops out again. It starts closing immediately.
        dh.update(0.1, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(dh.multiplier(), 0.7);
    }

    /// Verifies that volume smoothing (gain ramp) is being correctly applied to the audio.
    #[test]
    fn test_hysteresis_apply_gain_ramp() {
        let params = GateParams::new(-70.0, -80.0, 2048, 100, 1e-4);
        let mut buffer = [1.0f32; 10];

        let mut dh = DynamicHysteresis::new();
        // Simulates silence almost completing the hold time.
        dh.update(0.0, 1.0, 0.5, &params, 2047);
        assert_eq!(dh.state(), GateState::Open);

        // Passes hold and starts the smooth closing process (fade out).
        dh.update(0.0, 1.0, 0.5, &params, 10);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(
            dh.multiplier(),
            1.0,
            "In the transition block the multiplier is still 1.0"
        );

        // First actual fade block.
        dh.update(0.0, 1.0, 0.5, &params, 10);
        assert_eq!(dh.multiplier(), 0.9); // 90 of 100 steps completed.

        buffer.fill(1.0);
        dh.apply_gain_rt(&mut buffer, 10);
        // Should be a smooth ramp: the first sample maintains volume and the last is already reduced.
        assert!((buffer[0] - 1.0).abs() < 1e-3);
        assert!((buffer[9] - 0.91).abs() < 1e-3);

        // Now tests the smooth opening (FadingIn).
        let mut dh = DynamicHysteresis::new();
        dh.update(0.0, 1.0, 0.5, &params, 2048); // Forces close.
        dh.update(0.0, 1.0, 0.5, &params, 101); // Ensures it fully closed.
        assert_eq!(dh.state(), GateState::Closed);

        // Sound returns, starts opening.
        dh.update(2.0, 1.0, 0.5, &params, 1);
        assert_eq!(dh.multiplier(), 0.01);

        dh.update(2.0, 1.0, 0.5, &params, 10);
        assert_eq!(dh.multiplier(), 0.11);

        buffer.fill(1.0);
        dh.apply_gain_rt(&mut buffer, 10);
        // Volume should rise gradually from nearly zero (0.01) to 0.11.
        assert!((buffer[0] - 0.01).abs() < 1e-3);
        assert!((buffer[9] - 0.10).abs() < 1e-3);
    }

    /// Tests how the system handles very large audio blocks all at once.
    /// Smoothing should only happen in the correct timeframe and the rest should be processed.
    #[test]
    fn test_sub_block_granularity() {
        let mut dh = DynamicHysteresis::new();
        let params = GateParams::new(-70.0, -80.0, 2048, 256, 1e-4);
        let th_open = 1.0;
        let th_close = 0.5;

        // Prepares to close (fade out).
        dh.update(0.0, th_open, th_close, &params, 2048);
        assert_eq!(dh.state(), GateState::FadingOut);

        // Processes a giant block of 4096 samples, but the smoothing time is only 256!
        // The system should close in the first 256 samples and silence the rest of the block.
        dh.update(0.0, th_open, th_close, &params, 4096);
        assert_eq!(dh.state(), GateState::Closed);
        assert_eq!(dh.multiplier(), 0.0);

        let mut buffer = vec![1.0f32; 4096];
        dh.apply_gain_rt(&mut buffer, 4096);

        // Verifies exact smoothing in the first 256 samples.
        assert!(
            (buffer[0] - 1.0).abs() < 1e-3,
            "Start of fade-out should be maximum volume"
        );
        assert!(
            (buffer[128] - 0.5).abs() < 1e-2,
            "Middle of fade-out should be half volume"
        );

        // From sample 256 onward, everything should be absolute silence.
        assert_eq!(buffer[256], 0.0, "End of ramp did not strictly silence");
        assert_eq!(
            buffer[4095], 0.0,
            "Remainder of buffer not filled with zeros"
        );

        // Tests the same for opening (FadingIn) with a giant block.
        dh.update(2.0, th_open, th_close, &params, 4096);
        assert_eq!(dh.state(), GateState::Open);
        assert_eq!(dh.multiplier(), 1.0);

        let mut buffer2 = vec![1.0f32; 4096];
        dh.apply_gain_rt(&mut buffer2, 4096);

        // Starts in silence.
        assert_eq!(buffer2[0], 0.0, "Start of fade-in should be silence");
        assert!(
            (buffer2[128] - 0.5).abs() < 1e-2,
            "Middle of fade-in should be half volume"
        );

        // From sample 256 onward, volume should be 100% open.
        assert_eq!(
            buffer2[256], 1.0,
            "End of ramp did not fully open the volume"
        );
        assert_eq!(
            buffer2[4095], 1.0,
            "Remainder of buffer not preserved at maximum volume"
        );
    }

    /// Tests processing with blocks of only 1 sample (n_samples = 1).
    /// This is critical for audio hosts that may arbitrarily subdivide blocks.
    #[test]
    fn test_unit_block_processing() {
        let mut dh = DynamicHysteresis::new();
        let params = GateParams::new(-70.0, -80.0, 2, 2, 1e-4);
        let th_open = 1.0;
        let th_close = 0.5;

        // 1. Transition Open -> Hold -> FadingOut
        assert_eq!(dh.state(), GateState::Open);
        dh.update(0.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::Open, "Hold 1/2");
        dh.update(0.0, th_open, th_close, &params, 1);
        assert_eq!(
            dh.state(),
            GateState::FadingOut,
            "Entered fade-out after hold=2"
        );

        // 2. Transition FadingOut -> Closed
        dh.update(0.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingOut, "FadeOut 1/2");
        assert_eq!(dh.multiplier(), 0.5);
        dh.update(0.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::Closed, "Full silence achieved");
        assert_eq!(dh.multiplier(), 0.0);

        // 3. Transition Closed -> FadingIn -> Open
        dh.update(2.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingIn, "Start of FadeIn");
        assert_eq!(dh.multiplier(), 0.5);
        dh.update(2.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::Open, "Fully open again");
        assert_eq!(dh.multiplier(), 1.0);

        // 4. Ramp test with n=1 (Should work without panic or division by zero)
        let mut buffer = [1.0f32; 1];
        dh.update(0.0, th_open, th_close, &params, 2); // Forces FadingOut
        dh.update(0.0, th_open, th_close, &params, 1); // FadeOut 1/2
        dh.apply_gain_rt(&mut buffer, 1);
        assert!(
            (buffer[0] - 1.0).abs() < 1e-6,
            "The ramp start multiplier is 1.0"
        );

        dh.update(0.0, th_open, th_close, &params, 1); // FadeOut 2/2 -> Closed
        dh.apply_gain_rt(&mut buffer, 1);
        assert!(
            (buffer[0] - 0.5).abs() < 1e-6,
            "The ramp start multiplier for this block was 0.5"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: Some(Box::new(
                proptest::test_runner::FileFailurePersistence::SourceParallel(
                    "tests/proptest-regressions"
                )
            )),
            .. ProptestConfig::with_cases(10_000)
        })]

        #[test]
        #[ignore = "proptest 10k casos; roda no tests-long (gate_envelope_continuity_proptest)"]
        fn gate_envelope_continuity_on_reversal(
            fade_frames in 4usize..256,
            n_samples in 1usize..256,
        ) {
            let params = GateParams::new(-70.0, -80.0, 1, fade_frames, 1e-4);
            let th_open = 1.0;
            let th_close = 0.5;
            let inv = params.inv_fade_frames;
            let max_step = n_samples as f32 * inv;

            // FadingOut→FadingIn reversal: advance fade_out to target, then reverse
            {
                let mut dh = DynamicHysteresis::new();
                // Enter FadingOut
                dh.update(0.0, th_open, th_close, &params, params.hold_frames.max(1));
                assert_eq!(dh.state(), GateState::FadingOut, "must enter FadingOut");
                // Advance to half-fade
                let target_fc = fade_frames / 2;
                let elapsed = fade_frames.saturating_sub(target_fc);
                dh.update(0.0, th_open, th_close, &params, elapsed);
                if dh.state() == GateState::Closed {
                    // Overshoot: not a meaningful mid-fade reversal test
                    return Ok(());
                }
                let prev_mult = dh.multiplier();
                dh.update(2.0, th_open, th_close, &params, n_samples);
                let new_mult = dh.multiplier();
                assert!(
                    (new_mult - prev_mult).abs() <= max_step + 1e-6,
                    "FadingOut→FadingIn discontinuity: prev={prev_mult}, new={new_mult}, max_step={max_step}"
                );
            }

            // FadingIn→FadingOut reversal: advance fade_in to target, then reverse
            {
                let mut dh = DynamicHysteresis::new();
                // Enter FadingOut then finish to Closed
                dh.update(0.0, th_open, th_close, &params, params.hold_frames.max(1));
                if dh.state() == GateState::FadingOut {
                    dh.update(0.0, th_open, th_close, &params, params.fade_frames);
                }
                // Enter FadingIn
                dh.update(2.0, th_open, th_close, &params, 1);
                if dh.state() != GateState::FadingIn {
                    return Ok(());
                }
                // Advance to ~3/4 of fade_in
                let target_fc = fade_frames * 3 / 4;
                let remaining = target_fc.saturating_sub(1);
                if remaining > 0 {
                    dh.update(2.0, th_open, th_close, &params, remaining);
                }
                if dh.state() != GateState::FadingIn {
                    return Ok(());
                }
                let prev_mult = dh.multiplier();
                dh.update(0.0, th_open, th_close, &params, n_samples);
                let new_mult = dh.multiplier();
                assert!(
                    (new_mult - prev_mult).abs() <= max_step + 1e-6,
                    "FadingIn→FadingOut discontinuity: prev={prev_mult}, new={new_mult}, max_step={max_step}"
                );
            }
        }
    }

    /// T4.3 / F-CLAP-010 — `reset()` must restore the FSM to its initial Open
    /// state (multiplier 1.0, no ramp), as if freshly constructed.
    #[test]
    fn test_reset_restores_initial_open_state() {
        let params = GateParams::new(-10.0, -20.0, 10, 10, 1e-4);
        let mut dh = DynamicHysteresis::new();

        // Drive the FSM to Closed (mid-ramp with a residual ramp).
        dh.update(0.1, 1.0, 0.5, &params, 10);
        dh.update(0.1, 1.0, 0.5, &params, 5);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert!(dh.multiplier() < 1.0);

        dh.reset();
        assert_eq!(dh.state(), GateState::Open, "reset must reopen the gate");
        assert_eq!(dh.multiplier(), 1.0, "reset must restore unity multiplier");
        assert!(dh.is_steady(), "reset must clear any in-flight ramp");

        // A fresh instance driven through the same sequence after reset must
        // behave identically.
        let mut fresh = DynamicHysteresis::new();
        dh.update(0.1, 1.0, 0.5, &params, 10);
        fresh.update(0.1, 1.0, 0.5, &params, 10);
        dh.update(0.1, 1.0, 0.5, &params, 5);
        fresh.update(0.1, 1.0, 0.5, &params, 5);
        assert_eq!(dh.state(), fresh.state());
        assert_eq!(dh.multiplier(), fresh.multiplier());
        let mut buf_a = vec![0.5f32; 16];
        let mut buf_b = buf_a.clone();
        dh.apply_gain_rt(&mut buf_a, 16);
        fresh.apply_gain_rt(&mut buf_b, 16);
        assert_eq!(buf_a, buf_b, "post-reset gate must match a fresh gate");
    }

    // ── T4.2 / F-05: defensive clamping in apply_gain_rt / apply_gain_rt_stereo ──

    /// Drives a fresh gate into the middle of a FadingOut ramp.
    /// Returns the gate with `ramp_samples > 0` and a multiplier < 1.0.
    fn mid_fade_out(params: &GateParams) -> DynamicHysteresis {
        let mut dh = DynamicHysteresis::new();
        dh.update(0.0, 1.0, 0.5, params, params.hold_frames.max(1));
        assert_eq!(dh.state(), GateState::FadingOut);
        dh.update(0.0, 1.0, 0.5, params, params.fade_frames / 2);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert!(dh.multiplier() < 1.0);
        dh
    }

    /// Ramp stress: `apply_gain_rt` must never panic for any buffer length —
    /// including `len == 0` — in steady, closed, and mid-ramp states, and
    /// `n_samples == 0` must be a strict no-op.
    #[test]
    fn test_apply_gain_rt_zero_length_and_zero_samples_no_panic() {
        let params = GateParams::new(-70.0, -80.0, 10, 100, 1e-4);

        // Steady Open (mult 1.0), steady Closed (mult 0.0), mid-ramp FadingOut.
        let mut closed = DynamicHysteresis::new();
        closed.update(0.0, 1.0, 0.5, &params, params.hold_frames.max(1));
        closed.update(0.0, 1.0, 0.5, &params, params.fade_frames + 1);
        assert_eq!(closed.state(), GateState::Closed);
        let cases = [DynamicHysteresis::new(), closed, mid_fade_out(&params)];

        for dh in &cases {
            // Empty buffer with both n_samples == 0 and n_samples > 0.
            let mut empty: Vec<f32> = Vec::new();
            dh.apply_gain_rt(&mut empty, 0);
            dh.apply_gain_rt(&mut empty, 16);

            // Non-empty buffer with n_samples == 0: strict no-op.
            let mut buf = vec![0.5f32; 8];
            dh.apply_gain_rt(&mut buf, 0);
            assert_eq!(buf, vec![0.5f32; 8], "n_samples == 0 must be a no-op");
        }
    }

    /// Ramp stress: buffer shorter than `ramp_samples` must not panic; the
    /// ramp is clamped to the buffer length and stays a smooth linear ramp.
    #[test]
    fn test_apply_gain_rt_ramp_longer_than_buffer() {
        let params = GateParams::new(-70.0, -80.0, 10, 100, 1e-4);
        let dh = mid_fade_out(&params);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(dh.multiplier(), 0.5);

        // Remaining ramp = 50 samples, but the host delivers a 4-sample block.
        let mut buffer = vec![1.0f32; 4];
        dh.apply_gain_rt(&mut buffer, 4);
        let step = (0.5 - 1.0) / 4.0;
        for (i, &s) in buffer.iter().enumerate() {
            let expected = 1.0 + i as f32 * step;
            assert!(
                (s - expected).abs() < 1e-4,
                "clamped ramp sample {i}: {s} != {expected}"
            );
        }

        // Same ramp, but the caller over-declares n_samples (100) — the clamp
        // must use the actual buffer length, not the declared block.
        let mut buffer2 = vec![1.0f32; 4];
        dh.apply_gain_rt(&mut buffer2, 100);
        for (i, &s) in buffer2.iter().enumerate() {
            let expected = 1.0 + i as f32 * step;
            assert!(
                (s - expected).abs() < 1e-4,
                "over-declared ramp sample {i}: {s} != {expected}"
            );
        }
    }

    /// Ramp stress: buffer much larger than `ramp_samples` (normal path) —
    /// the ramp occupies the first `ramp_samples`, the rest is constant.
    #[test]
    fn test_apply_gain_rt_buffer_much_larger_than_ramp() {
        let params = GateParams::new(-70.0, -80.0, 2048, 256, 1e-4);
        let mut dh = DynamicHysteresis::new();
        dh.update(0.0, 1.0, 0.5, &params, 2048);
        assert_eq!(dh.state(), GateState::FadingOut);
        dh.update(0.0, 1.0, 0.5, &params, 128);
        assert_eq!(dh.multiplier(), 0.5);

        let mut buffer = vec![1.0f32; 4096];
        dh.apply_gain_rt(&mut buffer, 4096);
        // ramp_samples = 128: smooth linear descent over the first 128 samples.
        let step = (0.5 - 1.0) / 128.0;
        for (i, &s) in buffer[..128].iter().enumerate() {
            let expected = 1.0 + i as f32 * step;
            assert!((s - expected).abs() < 1e-4, "ramp sample {i}: {s}");
        }
        // Constant tail at the stabilized multiplier.
        assert!(
            buffer[128..].iter().all(|&s| (s - 0.5).abs() < 1e-4),
            "tail must hold the end multiplier"
        );
    }

    /// Stereo stress: unequal L/R lengths must never panic — only the common
    /// prefix is processed; the longer channel's tail is left untouched.
    #[test]
    fn test_apply_gain_rt_stereo_unequal_lengths_no_panic() {
        use crate::math::common::Avx2Math;

        let params = GateParams::new(-70.0, -80.0, 10, 100, 1e-4);
        let dh = mid_fade_out(&params);
        assert_eq!(dh.multiplier(), 0.5);

        // Mid-ramp with ramp_samples=50 > n=8: Case B over the common prefix.
        let mut left = vec![1.0f32; 8];
        let mut right = vec![1.0f32; 16];
        dh.apply_gain_rt_stereo::<Avx2Math>(&mut left, &mut right, 8);
        let step = (0.5 - 1.0) / 8.0;
        for i in 0..8 {
            let expected = 1.0 + i as f32 * step;
            assert!((left[i] - expected).abs() < 1e-4, "left[{i}]: {}", left[i]);
            assert!(
                (right[i] - expected).abs() < 1e-4,
                "right[{i}]: {}",
                right[i]
            );
        }
        assert!(
            right[8..].iter().all(|&s| s == 1.0),
            "longer channel tail must be untouched"
        );

        // Swapped lengths.
        let mut left2 = vec![1.0f32; 16];
        let mut right2 = vec![1.0f32; 8];
        dh.apply_gain_rt_stereo::<Avx2Math>(&mut left2, &mut right2, 8);
        for i in 0..8 {
            let expected = 1.0 + i as f32 * step;
            assert!(
                (left2[i] - expected).abs() < 1e-4,
                "left2[{i}]: {}",
                left2[i]
            );
            assert!(
                (right2[i] - expected).abs() < 1e-4,
                "right2[{i}]: {}",
                right2[i]
            );
        }
        assert!(
            left2[8..].iter().all(|&s| s == 1.0),
            "longer channel tail must be untouched"
        );
    }

    /// Stereo stress: empty buffers (both or a single channel) must not panic
    /// and must be a strict no-op.
    #[test]
    fn test_apply_gain_rt_stereo_empty_buffers_no_panic() {
        use crate::math::common::Avx2Math;

        let params = GateParams::new(-70.0, -80.0, 10, 100, 1e-4);
        let dh = mid_fade_out(&params);

        // Both channels empty.
        let mut l: Vec<f32> = Vec::new();
        let mut r: Vec<f32> = Vec::new();
        dh.apply_gain_rt_stereo::<Avx2Math>(&mut l, &mut r, 0);
        dh.apply_gain_rt_stereo::<Avx2Math>(&mut l, &mut r, 16);

        // One empty channel → common prefix empty → no-op on the other.
        let mut l2: Vec<f32> = Vec::new();
        let mut r2 = vec![1.0f32; 8];
        dh.apply_gain_rt_stereo::<Avx2Math>(&mut l2, &mut r2, 8);
        assert_eq!(r2, vec![1.0f32; 8], "empty left must make the call a no-op");
    }

    /// Stereo stress: buffer much larger than `ramp_samples` (normal path) —
    /// the residual fade-out ramp renders over the first `ramp_samples`, the
    /// rest is absolute silence (mirrors the mono `test_sub_block_granularity`).
    #[test]
    fn test_apply_gain_rt_stereo_large_buffer_normal_path() {
        use crate::math::common::Avx2Math;

        let params = GateParams::new(-70.0, -80.0, 2048, 256, 1e-4);
        let mut dh = DynamicHysteresis::new();
        dh.update(0.0, 1.0, 0.5, &params, 2048);
        dh.update(0.0, 1.0, 0.5, &params, 4096);
        assert_eq!(dh.state(), GateState::Closed);
        assert_eq!(dh.multiplier(), 0.0);

        let mut left = vec![1.0f32; 4096];
        let mut right = vec![1.0f32; 4096];
        dh.apply_gain_rt_stereo::<Avx2Math>(&mut left, &mut right, 4096);
        // Start of the residual fade-out at full volume...
        assert!((left[0] - 1.0).abs() < 1e-3, "left start of fade-out");
        assert!((right[0] - 1.0).abs() < 1e-3, "right start of fade-out");
        // ...and absolute silence from sample 256 onward.
        assert!(
            left[256..].iter().all(|&s| s == 0.0),
            "closed gate must silence left tail"
        );
        assert!(
            right[256..].iter().all(|&s| s == 0.0),
            "closed gate must silence right tail"
        );
    }
}
