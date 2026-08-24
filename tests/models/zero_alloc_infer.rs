// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Zero-allocation hot path verification tests using the CountingAllocator.
//
//  These tests prove that the inference hot path (`process()`) and the full DSP
//  pipeline (`capture_dsp_pipeline()`) perform zero heap allocations during
//  real-time audio processing. The `CountingAllocator` tracks `malloc` calls
//  per-thread via thread-local storage (TLS) and a `TrackingGuard` RAII gate.
//
//  When the `heap-audit` + `clap-plugin` features are active, the guard delegates
//  to `neural_amp_modeler_rs::common::alloc_audit::TrackingGuard`. Otherwise, it uses the local
//  `CountingAllocator` global allocator.

use neural_amp_modeler_rs::common::params::AdaptiveComputeMode;
use neural_amp_modeler_rs::dsp::adaptive::AdaptiveCompute;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{
    WavenetTopologyResult, get_wavenet_topology, parse_nam_json,
};
use neural_amp_modeler_rs::models::NamModel;
use std::fs;

use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};
use crate::common::*;

// =============================================================================
// Zero-Allocation Hot Path Tests
// =============================================================================

/// Zero-Allocation Verification Test for Static WaveNet
#[test]
fn test_zero_alloc_process_wavenet() {
    let path = model_path("BossWN-standard.nam");
    if !path.exists() {
        eprintln!("[STATUS] SKIP_OPTIONAL: model_not_found:BossWN-standard.nam — zero-alloc test");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in parser");
    let mut model = build_model(&model_data).expect("Failed to build model");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    {
        let _guard = TrackingGuard::new();
        model.process(&input, &mut output);
    }

    assert_eq!(
        get_alloc_count(),
        0,
        "Allocations detected in Static WaveNet hot path!"
    );
}

/// Zero-Allocation Verification Test for LSTM
#[test]
fn test_zero_alloc_process_lstm() {
    let path = model_path("BossLSTM-1x16.nam");
    if !path.exists() {
        eprintln!("[STATUS] SKIP_OPTIONAL: model_not_found:BossLSTM-1x16.nam — zero-alloc test");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in parser");
    let mut model = build_model(&model_data).expect("Failed to build model");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    {
        let _guard = TrackingGuard::new();
        model.process(&input, &mut output);
    }

    assert_eq!(
        get_alloc_count(),
        0,
        "Allocations detected in LSTM hot path!"
    );
}

/// Zero-Allocation Verification Test for Dynamic WaveNet
///
/// R-06 (S6.T3): this gate is strictly fail-closed — any allocation detected
/// in the dynamic WaveNet hot path fails the test unconditionally. A
/// regression here would otherwise pass CI with only a console warning.
#[test]
fn test_zero_alloc_process_wavenet_dynamic() {
    // We use Feather, which is allocated with a specific topology (or test with non-static topology)
    let path = model_path("BossWN-feather.nam");
    if !path.exists() {
        eprintln!("[STATUS] SKIP_OPTIONAL: model_not_found:BossWN-feather.nam — zero-alloc test");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in parser");

    // Builds a model that may be dynamic (fallback if it doesn't match const generics, or if we test build_wavenet_dynamic)
    // BossWN-feather.nam has 12 channels or another config.
    let mut model = build_model(&model_data).expect("Failed to build dynamic model");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    {
        let _guard = TrackingGuard::new();
        model.process(&input, &mut output);
    }

    let count = get_alloc_count();
    assert_eq!(
        count, 0,
        "Allocations detected in Dynamic WaveNet hot path! count={count} — \
         the zero-alloc gate is fail-closed (R-06): any heap allocation in \
         `process()` violates the RT contract."
    );
}

/// Zero-Allocation Verification Test for the Full DSP Pipeline
#[test]
fn test_zero_alloc_capture_pipeline() {
    use neural_amp_modeler_rs::common::spsc::RtStatusFlags;
    use neural_amp_modeler_rs::dsp::gate::{DynamicHysteresis, GateParams};
    use neural_amp_modeler_rs::dsp::oversample::{OversampleEngine, OversampleFactor};
    use neural_amp_modeler_rs::dsp::pipeline::{
        BridgeBuffer, DspBridge, DspBridgeWriter, DspPipelineContext, MAX_BRIDGE_BUF,
        MAX_RESAMP_BUF, capture_dsp_pipeline,
    };
    use neural_amp_modeler_rs::dsp::resampler::NamResampler;

    let path = model_path("BossWN-standard.nam");
    if !path.exists() {
        eprintln!("[STATUS] SKIP_OPTIONAL: model_not_found:BossWN-standard.nam");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in parser");
    let mut model_l = build_model(&model_data).expect("Failed to build model (L)");
    let mut model_r = build_model(&model_data).expect("Failed to build model (R)");

    model_l.prewarm(2048);
    model_r.prewarm(2048);

    let n = 64;
    let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
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
    let mut resamp_out_r = vec![0.0; MAX_RESAMP_BUF];
    let mut model_out_l = vec![0.0; MAX_RESAMP_BUF];
    let mut model_out_r = vec![0.0; MAX_RESAMP_BUF];

    let gate_params = GateParams::default();
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = false;

    let mut samples_l = generate_sine_440hz(n);
    let mut samples_r = generate_sine_440hz(n);

    let mut opt_model_l = Some(model_l);
    let mut opt_model_r = Some(model_r);

    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

    let mut os_engine_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut os_engine_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();

    let ctx = DspPipelineContext {
        resampler: &mut resampler,
        os_l: &mut os_engine_l,
        os_r: &mut os_engine_r,
        active_model_l: &mut opt_model_l,
        active_model_r: &mut opt_model_r,
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
        bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
        conv: None,
    };

    let mut os_buf: [f32; MAX_RESAMP_BUF * 6] = [0.0f32; MAX_RESAMP_BUF * 6];
    let (os_in_l_slice, rest) = os_buf.split_at_mut(MAX_RESAMP_BUF);
    let (os_in_r_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_l_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_r_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (crossfade_scratch_l, crossfade_scratch_r) = rest.split_at_mut(MAX_RESAMP_BUF);

    let bufs = neural_amp_modeler_rs::dsp::pipeline::DspBuffers {
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

    {
        let _guard = TrackingGuard::new();
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);
    }

    let count = get_alloc_count();
    assert_eq!(
        count, 0,
        "Allocation in capture_dsp_pipeline! The entire pipeline must be zero-alloc."
    );
}

/// Zero-Allocation Verification Test for `ContainerModel::set_slimmable_size()` alone.
///
/// Proves that `set_slimmable_size` (which calls `reset()` on the target submodel)
/// performs zero heap allocations — a prerequisite for RT-Safety.
#[test]
fn test_zero_alloc_set_slimmable_size() {
    use neural_amp_modeler_rs::models::StaticModel;
    use neural_amp_modeler_rs::models::container::ContainerModel;
    use neural_amp_modeler_rs::models::slimmable::SlimmableModel;

    let full_nam_path = model_path("wavenet_a2_full.nam");
    let lite_nam_path = model_path("wavenet_a2_lite.nam");
    if !full_nam_path.exists() || !lite_nam_path.exists() {
        eprintln!("[STATUS] SKIP_OPTIONAL: model_not_found:a2_full_or_lite");
        return;
    }

    let full_json = fs::read_to_string(&full_nam_path).expect("Failed to read A2-Full");
    let full_data = parse_nam_json(&full_json).expect("Failed to parse A2-Full");
    let full_model = build_model(&full_data).expect("Dispatcher failed for A2-Full");

    let lite_json = fs::read_to_string(&lite_nam_path).expect("Failed to read A2-Lite");
    let lite_data = parse_nam_json(&lite_json).expect("Failed to parse A2-Lite");
    let lite_model = build_model(&lite_data).expect("Dispatcher failed for A2-Lite");

    let sample_rate = full_data.sample_rate.map(|s| s as u32).unwrap_or(48000);

    let container = ContainerModel::new(vec![(0.5, lite_model), (1.0, full_model)], sample_rate)
        .expect("Failed to create ContainerModel");

    // Pre-run one block to ensure all internal buffers are allocated/sized.
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    let mut model = StaticModel::Container(Box::new(container));
    model.process(&input, &mut output);

    let count = {
        let _guard = TrackingGuard::new();
        if let StaticModel::Container(ref mut c) = model {
            c.set_slimmable_size(0.25, None);
        }
        get_alloc_count()
    };

    assert_eq!(
        count, 0,
        "Allocations detected during set_slimmable_size()! count={}",
        count
    );
}

/// Zero-Allocation Verification Test for ContainerModel Submodel Transition (RT-Safety).
#[test]
fn test_zero_alloc_container_transition() {
    use neural_amp_modeler_rs::models::StaticModel;
    use neural_amp_modeler_rs::models::slimmable::SlimmableModel;

    let full_nam_path = model_path("wavenet_a2_full.nam");
    let lite_nam_path = model_path("wavenet_a2_lite.nam");
    if !full_nam_path.exists() || !lite_nam_path.exists() {
        eprintln!(
            "[STATUS] SKIP_OPTIONAL: model_not_found:a2_full_or_lite — container transition test impossible"
        );
        return;
    }

    let full_json = fs::read_to_string(&full_nam_path).expect("Failed to read A2-Full");
    let full_data = parse_nam_json(&full_json).expect("Failed to parse A2-Full");
    let full_model = build_model(&full_data).expect("Dispatcher failed for A2-Full");

    let lite_json = fs::read_to_string(&lite_nam_path).expect("Failed to read A2-Lite");
    let lite_data = parse_nam_json(&lite_json).expect("Failed to parse A2-Lite");
    let lite_model = build_model(&lite_data).expect("Dispatcher failed for A2-Lite");

    let sample_rate = full_data.sample_rate.map(|s| s as u32).unwrap_or(48000);

    let container = neural_amp_modeler_rs::models::container::ContainerModel::new(
        vec![(0.5, lite_model), (1.0, full_model)],
        sample_rate,
    )
    .expect("Failed to create ContainerModel");

    let mut model = StaticModel::Container(Box::new(container));

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    // Pre-run one block
    model.process(&input, &mut output);

    let count = {
        let _guard = TrackingGuard::new();
        if let StaticModel::Container(ref mut c) = model {
            // Trigger transition: switch from Full to Lite
            c.set_slimmable_size(0.25, None);
        }
        // Run several blocks to process through the entire crossfade duration
        for _ in 0..30 {
            model.process(&input, &mut output);
        }
        get_alloc_count()
    };

    assert_eq!(
        count, 0,
        "Allocations detected during ContainerModel transition and crossfade! count={}",
        count
    );
}

/// Test parallel allocation tracking isolation.
///
/// Verifies that multiple threads running concurrently with their own `TrackingGuard`s
/// do not corrupt each other's allocation count and correctly isolate tracking state.
#[test]
fn test_parallel_allocation_tracking_isolation() {
    use std::thread;

    let num_threads = 8;
    let mut handles = Vec::new();

    for i in 0..num_threads {
        handles.push(thread::spawn(move || {
            // Each thread starts its own TrackingGuard
            let _guard = TrackingGuard::new();

            // Allocate some boxes to trigger allocations
            let mut v = Vec::new();
            for j in 0..(i + 1) * 10 {
                v.push(Box::new(j));
            }

            // Read local alloc count
            let count = get_alloc_count();
            // Ensure at least some allocations were captured
            assert!(
                count >= (i + 1) * 10,
                "Thread {} should have detected allocations, got {}",
                i,
                count
            );
            count
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.join().unwrap());
    }

    // Ensure different threads saw different allocation counts corresponding to their patterns,
    // and did not corrupt each other.
    for i in 1..num_threads {
        assert!(
            results[i] > results[i - 1],
            "Thread allocation counts should be isolated and distinct, got: {:?}",
            results
        );
    }
}

/// Zero-Allocation Verification Test for Nondist Models
#[test]
fn test_zero_alloc_nondist_models() {
    let mut nondist_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    nondist_path.push("tests/fixtures/models-nondist");

    if !nondist_path.exists() {
        println!("[STATUS] SKIP_OPTIONAL: models_nondist_absent");
        return;
    }

    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&nondist_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext == "nam" || ext == "json")
                && path.file_name().unwrap() != "manifest.json"
            {
                models.push(path);
            }
        }
    }

    if models.is_empty() {
        println!("[STATUS] SKIP_OPTIONAL: models_nondist_empty");
        return;
    }

    for model_path in models {
        let filename = model_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let json_data = fs::read_to_string(&model_path).expect("Failed to read model file");
        let model_data = parse_nam_json(&json_data).expect("Failed to parse model JSON");

        // Skip free-geometry WaveNet A1 models with >2 layer arrays — the dynamic
        // engine currently supports exactly 2 arrays. Models with N≠2 arrays
        // (edge geometry outside standard A1) are skipped gracefully.
        if model_data.architecture == "WaveNet"
            && !model_data.is_wavenet_a2()
            && matches!(
                get_wavenet_topology(&model_data),
                WavenetTopologyResult::Free(ref g) if g.num_arrays != 2
            )
        {
            println!("SKIP (free geometry, N≠2 arrays not yet supported): {filename}");
            continue;
        }

        let mut model = build_model(&model_data).expect("Failed to build model");

        model.prewarm(2048);

        let input = generate_sine_440hz(64);
        let mut output = vec![0.0f32; 64];

        let count = {
            let _guard = TrackingGuard::new();
            model.process(&input, &mut output);
            get_alloc_count()
        };

        println!(
            "Nondist model {} allocations in hot-path: {}",
            filename, count
        );

        // Free-geometry WaveNet models are skipped before build (see above), so every
        // model reaching this point is a supported fast-path (Known SKU / A2 / LSTM /
        // Linear) and must be strictly zero-alloc in the hot-path.
        assert_eq!(
            count, 0,
            "Allocations detected in hot-path for nondist model: {}!",
            filename
        );
    }
}
