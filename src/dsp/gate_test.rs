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
}
