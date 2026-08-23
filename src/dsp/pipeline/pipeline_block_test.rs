// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod block_tests {
    use super::super::test_util::infra::{TrackingGuard, get_alloc_count};
    use super::super::*;
    use crate::common::params::AdaptiveComputeMode;
    use crate::common::spsc::RtStatusFlags;
    use crate::dsp::adaptive::AdaptiveCompute;
    use crate::dsp::gate::{DynamicHysteresis, GateParams};
    use crate::dsp::oversample::{OversampleEngine, OversampleFactor};
    use crate::dsp::resampler::NamResampler;
    use crate::loader::dispatcher::build_model;
    use crate::loader::nam_json::parse_nam_json;
    use crate::models::StaticModel;
    use proptest::prelude::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    /// Helper to resolve the path for test models.
    fn get_test_model_path(name: &str) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/fixtures/models");
        path.push(name);
        path
    }

    /// Helper to load a NAM model for testing.
    fn load_test_model(name: &str) -> Box<StaticModel> {
        let path = get_test_model_path(name);
        let json_data = fs::read_to_string(path).expect("Failed to read model file");
        let model_data = parse_nam_json(&json_data).expect("Failed to process model JSON");
        build_model(&model_data).expect("Failed to build model")
    }

    /// Main test executor for pipeline with variable block size.
    fn run_block_size_test(model_name: Option<&str>, block_size: usize) {
        run_block_size_test_with_iterations(model_name, block_size, 1);
    }

    /// Main test executor for pipeline with variable block size and multiple iterations.
    fn run_block_size_test_with_iterations(
        model_name: Option<&str>,
        block_size: usize,
        iterations: usize,
    ) {
        // If the block size is 0, there's no point testing.
        if block_size == 0 {
            return;
        }
        // If it exceeds the bridge limit, cap it so the test doesn't panic on fixed buffer overflow.
        let n = block_size.min(MAX_BRIDGE_BUF);

        // Loads the amplifier simulation model (LSTM or WaveNet) if a name was provided.
        let mut model = model_name.map(load_test_model);
        if let Some(ref mut m) = model {
            // "Prewarm" prepares the model's internal state to process audio immediately,
            // avoiding pops or silence in the first samples.
            m.prewarm(2048);
        }

        // Initializes the Resampler. Here we use 48kHz -> 48kHz (bypass)
        // only to test the resampler buffering infrastructure with odd block sizes.
        let mut resampler = NamResampler::new(48000, 48000, n).unwrap();

        // Real-time status flags (indicate whether clipping or other issues occurred).
        let rt_status = RtStatusFlags::default();

        // The DspBridge is our memory "bridge". It stores processed audio for
        // another thread (such as the GUI or recorder) to read.
        // We use Box to guarantee a fixed memory address (heap).
        let mut bridge = Box::new(DspBridge {
            // We create two buffers for the "Double Buffering" technique (prevents readers from disrupting writers).
            buffers: [
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
            ],
            // Atomic counters for safe synchronization between threads without locks.
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        // We allocate intermediate buffers needed for the processing stages.
        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        // Noise Gate configuration (noise suppressor).
        let gate_params = GateParams::default();
        // Hysteresis controls the smoothness of sound opening and closing to avoid "pops".
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        // We create test samples (a constant 0.1 signal) to process.
        let mut samples_l = vec![0.1; n];
        let mut samples_r = vec![0.1; n];

        let mut os_engine_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
        let mut os_engine_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();

        let _guard = TrackingGuard::new();

        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

        for _ in 0..iterations {
            // The Context (ctx) groups all the tools the pipeline needs to work.
            let ctx = DspPipelineContext {
                resampler: &mut resampler,
                os_l: &mut os_engine_l,
                os_r: &mut os_engine_r,
                active_model_l: &mut model,
                active_model_r: &mut None,
                input_gain_mult: 1.0,
                output_gain_mult: 1.0,
                gate_params: &gate_params,
                silence_hysteresis: &mut silence_hysteresis,
                mono_hysteresis: &mut mono_hysteresis,
                threshold_open_sq: 0.0,
                threshold_close_sq: 0.0,
                process_mono: &mut process_mono,
                rt_status: &rt_status,
                adaptive: &mut adaptive,
                // BridgeRef is a safe pointer to the audio bridge.
                bridge_writer: unsafe {
                    Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge))
                },
                conv: None,
            };

            let mut os_buf: [f32; MAX_RESAMP_BUF * 6] = [0.0f32; MAX_RESAMP_BUF * 6];
            let (os_in_l_slice, rest) = os_buf.split_at_mut(MAX_RESAMP_BUF);
            let (os_in_r_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
            let (os_model_l_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
            let (os_model_r_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
            let (crossfade_scratch_l, crossfade_scratch_r) = rest.split_at_mut(MAX_RESAMP_BUF);

            let bufs = DspBuffers {
                resamp_mid_l: &mut resamp_mid_l,
                resamp_mid_r: &mut resamp_mid_r,
                resamp_out_l: &mut resamp_out_l,
                resamp_out_r: &mut resamp_out_r,
                model_out_l: &mut model_out_l,
                model_out_r: &mut model_out_r,
                os_in_l: os_in_l_slice,
                os_in_r: os_in_r_slice,
                os_model_l: os_model_l_slice,
                os_model_r: os_model_r_slice,
                crossfade_scratch_l,
                crossfade_scratch_r,
            };

            // We run the main pipeline that orchestrates all NeuralAmpModeler-rs DSP.
            capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);
        }

        // We check how many allocations occurred during processing.
        let allocs = get_alloc_count();
        // We remove the watchdog.
        drop(_guard);

        assert_eq!(
            allocs, 0,
            "Allocation detected in {} iterations",
            iterations
        );

        let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
        let out_buf = &bridge.buffers[read_idx];
        assert_eq!(out_buf.n_samples as usize, n);

        // Mathematical sanity check: audio must not "blow up" (become NaN or Infinity).
        for i in 0..n {
            assert!(out_buf.buf_l[i].is_finite());
            assert!(out_buf.buf_r[i].is_finite());
        }
    }

    /// TEST: Unconventional block sizes for LSTM models.
    /// Hosts like Bitwig Studio can send blocks of any size (e.g. 7 or 17 samples).
    #[test]
    fn test_unconventional_block_sizes_lstm() {
        let sizes = [1, 3, 7, 8, 9, 17, 33, 53, 64, 128, 256, 512];
        for &size in &sizes {
            run_block_size_test(Some("BossLSTM-1x16.nam"), size);
        }
    }

    /// TEST: Unconventional block sizes for WaveNet models.
    #[test]
    fn test_unconventional_block_sizes_wavenet() {
        let sizes = [1, 3, 7, 8, 9, 17, 33, 53, 64, 128, 256, 512];
        for &size in &sizes {
            run_block_size_test(Some("BossWN-nano.nam"), size);
        }
    }

    /// TEST: Edge cases (extremes).
    #[test]
    fn test_zero_alloc_edge_cases() {
        // n_samples = 1 (minimum possible)
        run_block_size_test(Some("BossWN-nano.nam"), 1);
        // n_samples = MAX_BRIDGE_BUF (maximum supported by our internal buffer)
        run_block_size_test(Some("BossWN-nano.nam"), MAX_BRIDGE_BUF);
    }

    /// TEST: Zero-Allocation Stress for edge cases.
    /// Validates that 1-sample blocks and max-size blocks do not allocate on the hot-path under stress.
    #[test]
    fn test_zero_alloc_stress_edge_cases() {
        // 1. Scenario: n_frames = 1 (1000 consecutive invocations)
        run_block_size_test_with_iterations(Some("BossWN-nano.nam"), 1, 1000);
        // 2. Scenario: n_frames = MAX_BRIDGE_BUF (100 invocations)
        run_block_size_test_with_iterations(Some("BossWN-nano.nam"), MAX_BRIDGE_BUF, 100);
    }

    /// TEST: Ratio-aware resampling with maximum block size (8192 samples).
    /// Verifies that the chunking logic in the inference pipeline correctly processes
    /// all samples even when upsampling (e.g., 44100→48000) would exceed intermediate
    /// buffer capacity in a single pass.
    #[test]
    fn test_ratio_aware_resampling_max_block() {
        let model_name = "BossWN-nano.nam";
        let block_size = MAX_BRIDGE_BUF; // 8192 samples — max host quantum

        let rate_pairs = [
            (44100, 48000), // upsampling: input demands ≈8917 intermediate samples
            (48000, 44100), // downsampling: output shrinks, always fits
            (48000, 48000), // bypass: passthrough
            (96000, 48000), // downsampling 2:1
        ];

        for &(host_rate, nam_rate) in &rate_pairs {
            run_block_size_test_with_resampling(
                Some(model_name),
                block_size,
                1,
                host_rate,
                nam_rate,
            );
        }
    }

    /// TEST: Zero-allocation stress test for ratio-aware resampling.
    /// Processes 100 iterations of max-size blocks with resampling active.
    #[test]
    fn test_ratio_aware_zero_alloc_stress() {
        let model_name = "BossWN-nano.nam";
        let block_size = MAX_BRIDGE_BUF; // 8192 samples
        let iterations = 100;

        let rate_pairs = [(44100, 48000), (48000, 44100), (96000, 48000)];

        for &(host_rate, nam_rate) in &rate_pairs {
            run_block_size_test_with_resampling(
                Some(model_name),
                block_size,
                iterations,
                host_rate,
                nam_rate,
            );
        }
    }

    /// Test executor for pipeline with resampling at non-matching rates.
    fn run_block_size_test_with_resampling(
        model_name: Option<&str>,
        block_size: usize,
        iterations: usize,
        host_rate: u32,
        nam_rate: u32,
    ) {
        if block_size == 0 {
            return;
        }
        let n = block_size.min(MAX_BRIDGE_BUF);

        let mut model = model_name.map(load_test_model);
        if let Some(ref mut m) = model {
            m.prewarm(2048);
        }

        let mut resampler = NamResampler::new(host_rate, nam_rate, n).unwrap();

        let rt_status = RtStatusFlags::default();

        let mut bridge = Box::new(DspBridge {
            buffers: [
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
            ],
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        let gate_params = GateParams::default();
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut samples_l = vec![0.1; n];
        let mut samples_r = vec![0.1; n];

        let mut os_engine_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
        let mut os_engine_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();

        let _guard = TrackingGuard::new();

        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

        for _ in 0..iterations {
            let ctx = DspPipelineContext {
                resampler: &mut resampler,
                os_l: &mut os_engine_l,
                os_r: &mut os_engine_r,
                active_model_l: &mut model,
                active_model_r: &mut None,
                input_gain_mult: 1.0,
                output_gain_mult: 1.0,
                gate_params: &gate_params,
                silence_hysteresis: &mut silence_hysteresis,
                mono_hysteresis: &mut mono_hysteresis,
                threshold_open_sq: 0.0,
                threshold_close_sq: 0.0,
                process_mono: &mut process_mono,
                rt_status: &rt_status,
                adaptive: &mut adaptive,
                bridge_writer: unsafe {
                    Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge))
                },
                conv: None,
            };

            let mut os_buf: [f32; MAX_RESAMP_BUF * 6] = [0.0f32; MAX_RESAMP_BUF * 6];
            let (os_in_l_slice, rest) = os_buf.split_at_mut(MAX_RESAMP_BUF);
            let (os_in_r_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
            let (os_model_l_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
            let (os_model_r_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
            let (crossfade_scratch_l, crossfade_scratch_r) = rest.split_at_mut(MAX_RESAMP_BUF);

            let bufs = DspBuffers {
                resamp_mid_l: &mut resamp_mid_l,
                resamp_mid_r: &mut resamp_mid_r,
                resamp_out_l: &mut resamp_out_l,
                resamp_out_r: &mut resamp_out_r,
                model_out_l: &mut model_out_l,
                model_out_r: &mut model_out_r,
                os_in_l: os_in_l_slice,
                os_in_r: os_in_r_slice,
                os_model_l: os_model_l_slice,
                os_model_r: os_model_r_slice,
                crossfade_scratch_l,
                crossfade_scratch_r,
            };

            capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, host_rate);
        }

        let allocs = get_alloc_count();
        drop(_guard);

        assert_eq!(
            allocs, 0,
            "Allocation detected with host_rate={}, nam_rate={}, {} iterations",
            host_rate, nam_rate, iterations
        );

        let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
        let out_buf = &bridge.buffers[read_idx];
        assert_eq!(
            out_buf.n_samples as usize, n,
            "Incomplete output at host_rate={}, nam_rate={}: expected {} samples, got {}",
            host_rate, nam_rate, n, out_buf.n_samples
        );

        for i in 0..n {
            assert!(
                out_buf.buf_l[i].is_finite(),
                "Non-finite output at host_rate={}, nam_rate={}, sample={}",
                host_rate,
                nam_rate,
                i
            );
            assert!(
                out_buf.buf_r[i].is_finite(),
                "Non-finite output at host_rate={}, nam_rate={}, sample={}",
                host_rate,
                nam_rate,
                i
            );
        }
    }

    // Property-Based Testing (Proptest):
    // Instead of choosing the numbers ourselves, we let the computer generate 500 random
    // sizes between 1 and 8192 to try to break our code.
    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("tests/proptest-regressions"))),
            .. ProptestConfig::with_cases(2_000)
        })]
        #[test]
        #[ignore]
        fn test_random_block_sizes_proptest(size in 1..8192usize) {
            run_block_size_test(Some("BossWN-nano.nam"), size);
        }
    }
}
