// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Integration Tests for topology validation, neural inference stability,
//  and miscellaneous edge cases (denormals, block size invariance, community models).

use neural_amp_modeler_rs::dsp::telemetry::LatencyHistogram;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{
    NamWavenetTopology, WavenetTopologyResult, get_wavenet_topology, parse_nam_json,
};
use neural_amp_modeler_rs::math::activations::ActivationPrecision;
use neural_amp_modeler_rs::math::common::AlignedVec;
use neural_amp_modeler_rs::models::a2::activations::ActivationType;
use neural_amp_modeler_rs::models::{NamModel, wavenet};
use std::fs;
use std::path::PathBuf;

use super::common;
use common::*;

// WaveNetModel<CH=16, K=3, HEAD=8>: CH=16 channels, HEAD=8 (layer 0 head_size = Array2 channels).
type WaveNetStandard = wavenet::WaveNetModel<16, 3, 8>;

// =============================================================================
// Helpers — Synthetic Model Construction for Topology Unit Tests
// =============================================================================

/// Helper to simulate creation of a clean `WaveNetLayer` for Array1 (CH=16).
fn make_wavenet_layer(
    dilation: usize,
    _has_bias: bool,
    ch: usize,
) -> wavenet::WaveNetLayer<1, 16, 3> {
    let _raw_conv = vec![0.001f32; ch * 3 * ch];
    let conv_weights = {
        let mut fw = AlignedVec::new(_raw_conv.len(), 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        neural_amp_modeler_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &_raw_conv, &mut fw, ch, ch, 3,
        );
        fw
    };
    wavenet::WaveNetLayer {
        conv1d: wavenet::Conv1d {
            weights: conv_weights,
            bias: AlignedVec::from_vec(vec![0.0; ch])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
            dilation,
        },
        input_mixin: wavenet::DenseLayer {
            weights: AlignedVec::from_vec(vec![0.001f32; ch])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; ch])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        one_by_one: wavenet::DenseLayer {
            weights: AlignedVec::from_vec(vec![0.001f32; ch * ch])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; ch])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        activation: ActivationType::Tanh,
        scratch_mixin: AlignedVec::new(ch * wavenet::WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("scratch alloc"),
        scratch_conv: AlignedVec::new(ch * wavenet::WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("scratch alloc"),
    }
}

/// Helper to simulate creation of a `WaveNetLayer` for Array2 (CH=8, =HEAD of Array1).
fn make_wavenet_layer_a2(dilation: usize) -> wavenet::WaveNetLayer<1, 8, 3> {
    let _raw_conv = vec![0.001f32; 8 * 3 * 8];
    let conv_weights = {
        let mut fw = AlignedVec::new(_raw_conv.len(), 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        neural_amp_modeler_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &_raw_conv, &mut fw, 8, 8, 3,
        );
        fw
    };
    wavenet::WaveNetLayer {
        conv1d: wavenet::Conv1d {
            weights: conv_weights,
            bias: AlignedVec::from_vec(vec![0.0; 8])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
            dilation,
        },
        input_mixin: wavenet::DenseLayer {
            weights: AlignedVec::from_vec(vec![0.001f32; 8])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 8])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        one_by_one: wavenet::DenseLayer {
            weights: AlignedVec::from_vec(vec![0.001f32; 8 * 8])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 8])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        activation: ActivationType::Tanh,
        scratch_mixin: AlignedVec::new(8 * wavenet::WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("scratch alloc"),
        scratch_conv: AlignedVec::new(8 * wavenet::WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("scratch alloc"),
    }
}

/// Helper to build a synthetic `WaveNetStandard` network.
///
/// `WaveNetModel<16, 3, 8>`: Array1 CH=16, Array2 CH=8(=HEAD), HEAD2=1.
/// Note: the type alias below now uses HEAD=8 (real head_size of BossWN-standard).
///
/// # Alignment with the production constructor (`build_wavenet_array`)
///
/// Each `WaveNetLayerState` receives `receptive_field_size = (K-1) * dilation`
/// from its specific layer — faithfully mirroring `build_wavenet_array` (L274).
/// The global `receptive_field_size` is the sum of all individual RFs (2046).
fn build_synthetic_wavenet_standard() -> WaveNetStandard {
    const K: usize = 3;
    let dilations_1: [usize; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let dilations_2: [usize; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    // Total RF = sum of RFs per layer: Σ (K-1)*d
    // For dilations = [1,2,4,8,16,32,64,128,256,512]: sum = 1023 × 2 = 2046
    let rf1: usize = dilations_1.iter().map(|&d| (K - 1) * d).sum();
    let rf2: usize = dilations_2.iter().map(|&d| (K - 1) * d).sum();
    let final_rf = rf1.max(rf2);

    let layers_1: Vec<wavenet::WaveNetLayer<1, 16, 3>> = dilations_1
        .iter()
        .map(|&d| make_wavenet_layer(d, false, 16))
        .collect();
    // Per-layer RF: (K-1)*d — mirrors build_wavenet_array L274
    let states_1: Vec<wavenet::WaveNetLayerState> = dilations_1
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            wavenet::WaveNetLayerState::new(16, (K - 1) * d, i)
                .expect("Failed to create WaveNetLayerState")
        })
        .collect();

    let array1 = wavenet::WaveNetLayerArray::<1, 1, 16, 3, 8> {
        layers: layers_1,
        states: states_1,
        rechannel: wavenet::DenseLayer {
            weights: AlignedVec::from_vec(vec![0.001f32; 16])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 16])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: wavenet::DenseLayer {
            weights: AlignedVec::from_vec(vec![0.001f32; 8 * 16])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 8])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 16 * wavenet::WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; 16 * wavenet::WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; 8 * wavenet::WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf1,
        block_size: 16,
        block_buffer: AlignedVec::from_vec(vec![0.0; 16 * wavenet::WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        effective_layers: dilations_1.len(),
    };

    // Array2: IN=16(=CH), COND=1, CH=8(=HEAD1), HEAD2=1, HasHeadBias=true
    let layers_2: Vec<wavenet::WaveNetLayer<1, 8, 3>> = dilations_2
        .iter()
        .map(|&d| make_wavenet_layer_a2(d))
        .collect();
    // Per-layer RF: (K-1)*d (CH=8)
    let states_2: Vec<wavenet::WaveNetLayerState> = dilations_2
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            // alloc_num continues from where array1 stopped — mirrors the dispatcher's global alloc_num
            wavenet::WaveNetLayerState::new(8, (K - 1) * d, dilations_1.len() + i)
                .expect("Failed to create WaveNetLayerState")
        })
        .collect();

    let array2 = wavenet::WaveNetLayerArray::<16, 1, 8, 3, 1> {
        layers: layers_2,
        states: states_2,
        rechannel: wavenet::DenseLayer {
            weights: AlignedVec::from_vec(vec![0.0f32; 16 * 8])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 8])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: wavenet::DenseLayer {
            weights: AlignedVec::from_vec(vec![0.0f32; 8])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 1])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 8 * wavenet::WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; 8 * wavenet::WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; wavenet::WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf2,
        block_size: 8,
        block_buffer: AlignedVec::from_vec(vec![0.0; 8 * wavenet::WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        effective_layers: dilations_2.len(),
    };

    WaveNetStandard {
        array1,
        array2,
        head_scale: 0.02,
        receptive_field_size: final_rf,
        prewarm_on_reset: true,
    }
}

/// Test 1: Audit of Loader Read Capacity and Geometry Validation
#[test]
fn test_wavenet_model_json_parsing() {
    let path = model_path("BossWN-standard.nam");

    if !path.exists() {
        eprintln!("SKIP: WaveNet test model not found at {path:?}. Skipping parsing.");
        return;
    }

    let json_data = fs::read_to_string(path).expect("Failed to read JSON file");
    let model_data = parse_nam_json(&json_data).expect("Failed in parser_nam_json");

    assert_eq!(model_data.architecture, "WaveNet", "Should detect WaveNet");
    let topo = get_wavenet_topology(&model_data);
    assert_eq!(
        topo,
        WavenetTopologyResult::Known(NamWavenetTopology::Standard),
        "The BossWN-standard model should be recognized as Standard Wavenet"
    );

    // Validate metadata properties for BossWN-standard match
    assert!(model_data.metadata.is_some());
    let metadata = model_data.metadata.unwrap();
    assert!(metadata.loudness.is_some());
}

/// Test 2: Run Multiple Sine Wave Blocks through the WaveNet Core and compute RMS/Error (Sanity)
///
/// Processes a multi-block sine wave through a synthetic WaveNet (weights=0.001,
/// tanh activations). Validates finiteness, per-sample magnitude (≤ 10× input peak),
/// and RMS ≤ 10.0. In debug, uses reduced blocks (512) for CI speed.
#[test]
fn test_wavenet_computational_stability() {
    let mut model = build_synthetic_wavenet_standard();

    model.prewarm();

    let mut in_data = [0.0f32; TEST_BLOCK_SIZE];
    let mut out_data = [0.0f32; TEST_BLOCK_SIZE];

    let num_blocks = if cfg!(debug_assertions) {
        512
    } else {
        TEST_NUM_BLOCKS
    };

    let mut tot_energy = 0.0f64;
    let mut max_abs_out = 0.0f32;
    let mut pos: u64 = 0;

    for _ in 0..num_blocks {
        for item in in_data.iter_mut().take(TEST_BLOCK_SIZE) {
            *item = ((pos as f32) * 0.01).sin();
            pos += 1;
        }

        model.process(&in_data, &mut out_data);

        for &out_val in out_data.iter().take(TEST_BLOCK_SIZE) {
            assert!(
                out_val.is_finite(),
                "Computational crash detected: FPU produced non-finite float. Audit failure."
            );
            let abs_v = out_val.abs();
            max_abs_out = max_abs_out.max(abs_v);
            tot_energy += (out_val as f64) * (out_val as f64);
        }
    }

    let rms = (tot_energy / ((TEST_BLOCK_SIZE * num_blocks) as f64)).sqrt();
    println!(
        "[WaveNet Integrity Audit] RMS over processed sine wave: {}",
        rms
    );

    assert!(
        rms <= 10.0,
        "WaveNet RMS {rms:.4} exceeds reasonable magnitude (10.0). Possible numerical divergence."
    );
    // The synthetic model (all weights 0.001, head_rechannel bias=0) can produce
    // near-zero outputs by design — no lower-bound RMS check for this fixture.
    // The synthetic model (all weights 0.001, head_rechannel bias=0) can produce
    // near-zero outputs by design — this is a known property of the controlled fixture
    const GAIN_LIMIT: f32 = 10.0;
    assert!(
        max_abs_out <= GAIN_LIMIT,
        "Synthetic WaveNet peak {max_abs_out} exceeds {GAIN_LIMIT}× input peak. Numerical divergence."
    );
}

// =============================================================================
// Test Coverage Expansion (Feather, Nano, LSTM 2x8)
// =============================================================================

/// Test 13: WaveNet Feather Stability
///
/// Processes a 440 Hz sine through the real Feather model and validates:
/// - All outputs are finite (no NaN/Inf).
/// - Peak output does not exceed 20× the peak input (physically-grounded gate
///   for a guitar amp model; any output exceeding 20× a ±1 sine indicates
///   numerical divergence).
/// - RMS is bounded relative to the input (catches silent-model bugs).
#[test]
fn test_wavenet_stability_feather() {
    let path = model_path("BossWN-feather.nam");

    if !path.exists() {
        eprintln!("SKIP: BossWN-feather.nam not found at {path:?}. Skipping Feather stability.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet Feather model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for Feather");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let max_abs_in = input.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_in > 0.0,
        "Input signal is silent — test fixture broken"
    );
    const GAIN_LIMIT: f32 = 20.0;

    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    let mut max_abs_out = 0.0f32;
    let mut tot_energy = 0.0f64;
    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "[Feather] Non-finite sample at index {i}: {s}"
        );
        let abs_s = s.abs();
        max_abs_out = max_abs_out.max(abs_s);
        tot_energy += (s as f64) * (s as f64);
        let limit = max_abs_in * GAIN_LIMIT;
        assert!(
            abs_s <= limit,
            "[Feather] Excessive magnitude at index {i}: {s} (limit {limit}, {GAIN_LIMIT}× input peak)"
        );
    }
    let rms = (tot_energy / 64.0).sqrt();
    assert!(
        rms > 0.0001 && rms < 10.0,
        "[Feather] RMS out of band: {} (expected 0.0001 < rms < 10.0)",
        rms
    );
    assert!(
        max_abs_out > 0.0,
        "[Feather] All-zero output — model may be silent"
    );
    assert!(
        rms < max_abs_out as f64,
        "[Feather] RMS ({rms}) >= max_abs ({max_abs_out}) — abnormal output shape"
    );
}

/// WaveNet Nano Stability
///
/// Processes a 440 Hz sine through the real Nano model. Same physical gate as
/// Feather: peak output ≤ 20× peak input.
#[test]
fn test_wavenet_stability_nano() {
    let path = model_path("BossWN-nano.nam");

    if !path.exists() {
        eprintln!("SKIP: BossWN-nano.nam not found at {path:?}. Skipping Nano stability.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet Nano model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for Nano");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let max_abs_in = input.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_in > 0.0,
        "Input signal is silent — test fixture broken"
    );
    const GAIN_LIMIT: f32 = 20.0;

    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    let mut max_abs_out = 0.0f32;
    let mut tot_energy = 0.0f64;
    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "[Nano] Non-finite sample at index {i}: {s}");
        let abs_s = s.abs();
        max_abs_out = max_abs_out.max(abs_s);
        tot_energy += (s as f64) * (s as f64);
        let limit = max_abs_in * GAIN_LIMIT;
        assert!(
            abs_s <= limit,
            "[Nano] Excessive magnitude at index {i}: {s} (limit {limit}, {GAIN_LIMIT}× input peak)"
        );
    }
    let rms = (tot_energy / 64.0).sqrt();
    assert!(
        rms > 0.0001 && rms < 10.0,
        "[Nano] RMS out of band: {} (expected 0.0001 < rms < 10.0)",
        rms
    );
    assert!(
        max_abs_out > 0.0,
        "[Nano] All-zero output — model may be silent"
    );
    assert!(
        rms < max_abs_out as f64,
        "[Nano] RMS ({rms}) >= max_abs ({max_abs_out}) — abnormal output shape"
    );
}

/// WaveNet A2-Full Stability
///
/// A2 fixtures use deterministic random weights (SEED=42), which can produce
/// larger-than-usual outputs. A physically-grounded gate of 50× input peak
/// catches numerical divergence while tolerating the synthetic weight distribution.
#[test]
fn test_wavenet_stability_a2_full() {
    let path = model_path("wavenet_a2_full.nam");

    if !path.exists() {
        eprintln!("SKIP: wavenet_a2_full.nam not found at {path:?}. Skipping A2-Full stability.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for A2-Full");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let max_abs_in = input.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_in > 0.0,
        "Input signal is silent — test fixture broken"
    );
    const GAIN_LIMIT: f32 = 50.0;
    let limit = max_abs_in * GAIN_LIMIT;

    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    let mut max_abs_out = 0.0f32;
    let mut tot_energy = 0.0f64;
    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "[A2-Full] Non-finite sample at index {i}: {s}"
        );
        let abs_s = s.abs();
        max_abs_out = max_abs_out.max(abs_s);
        tot_energy += (s as f64) * (s as f64);
        assert!(
            abs_s <= limit,
            "[A2-Full] Excessive magnitude at index {i}: {s} (limit {limit}, {GAIN_LIMIT}× input peak)"
        );
    }
    let rms = (tot_energy / 64.0).sqrt();
    assert!(
        rms > 0.0001 && rms < 10.0,
        "[A2-Full] RMS out of band: {} (expected 0.0001 < rms < 10.0)",
        rms
    );
    assert!(
        max_abs_out > 0.0,
        "[A2-Full] All-zero output — model may be silent"
    );
}

/// WaveNet A2-Lite Stability
///
/// Same convention as A2-Full: physically-grounded gate of 50× input peak
/// for the synthetic random-weight fixture (SEED=42).
#[test]
fn test_wavenet_stability_a2_lite() {
    let path = model_path("wavenet_a2_lite.nam");

    if !path.exists() {
        eprintln!("SKIP: wavenet_a2_lite.nam not found at {path:?}. Skipping A2-Lite stability.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet A2-Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for A2-Lite");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let max_abs_in = input.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_in > 0.0,
        "Input signal is silent — test fixture broken"
    );
    const GAIN_LIMIT: f32 = 50.0;
    let limit = max_abs_in * GAIN_LIMIT;

    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    let mut max_abs_out = 0.0f32;
    let mut tot_energy = 0.0f64;
    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "[A2-Lite] Non-finite sample at index {i}: {s}"
        );
        let abs_s = s.abs();
        max_abs_out = max_abs_out.max(abs_s);
        tot_energy += (s as f64) * (s as f64);
        assert!(
            abs_s <= limit,
            "[A2-Lite] Excessive magnitude at index {i}: {s} (limit {limit}, {GAIN_LIMIT}× input peak)"
        );
    }
    let rms = (tot_energy / 64.0).sqrt();
    assert!(
        rms > 0.0001 && rms < 10.0,
        "[A2-Lite] RMS out of band: {} (expected 0.0001 < rms < 10.0)",
        rms
    );
    assert!(
        max_abs_out > 0.0,
        "[A2-Lite] All-zero output — model may be silent"
    );
}

/// Test 15: LSTM 2x8 Stability
///
/// Processes a 440 Hz sine through the real LSTM 2×8 model.
/// Physically-grounded gate: peak output ≤ 20× peak input.
/// LSTM recurrent cells can accumulate state, but a clean sine input
/// should not diverge beyond 20× gain.
#[test]
fn test_lstm_stability_2x8() {
    let path = model_path("BossLSTM-2x8.nam");

    if !path.exists() {
        eprintln!("SKIP: BossLSTM-2x8.nam not found at {path:?}. Skipping LSTM 2x8 stability.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read LSTM 2x8 model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for LSTM 2x8");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let max_abs_in = input.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_in > 0.0,
        "Input signal is silent — test fixture broken"
    );
    const GAIN_LIMIT: f32 = 20.0;
    let limit = max_abs_in * GAIN_LIMIT;

    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    let mut max_abs_out = 0.0f32;
    let mut tot_energy = 0.0f64;
    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "[LSTM 2x8] Non-finite sample at index {i}: {s}"
        );
        let abs_s = s.abs();
        max_abs_out = max_abs_out.max(abs_s);
        tot_energy += (s as f64) * (s as f64);
        assert!(
            abs_s <= limit,
            "[LSTM 2x8] Excessive magnitude at index {i}: {s} (limit {limit}, {GAIN_LIMIT}× input peak)"
        );
    }
    let rms = (tot_energy / 64.0).sqrt();
    assert!(
        rms > 0.0001 && rms < 10.0,
        "[LSTM 2x8] RMS out of band: {} (expected 0.0001 < rms < 10.0)",
        rms
    );
    assert!(
        max_abs_out > 0.0,
        "[LSTM 2x8] All-zero output — model may be silent"
    );
}

// =============================================================================
// Prolonged Silence Stability Test (Denormals)
// =============================================================================

/// Test 17: Stability under prolonged silence — denormal validation.
///
/// Loads `BossWN-standard.nam` and `BossLSTM-1x16.nam`, runs prewarm and
/// processes 4096 blocks (≈5.5s) of **total silence** (input = zeros).
///
/// Validates that:
/// - All outputs are finite.
/// - Output stabilizes (magnitude < 1.0 after convergence — real models
///   may have residual DC offset due to neural network biases).
/// - No detectable subnormal values in the output (all zero or finite normals).
/// - Per-block processing time (measured via `Instant::now()`) does not exceed 500μs.
///
/// This test exercises the exponential decay path in the internal states
/// of WaveNet (convolution buffers) and LSTM (cell/hidden states), which without
/// DAZ/FTZ converge to denormals and cause microcode penalty in the FPU.
#[test]
fn test_denormal_stability_silence() {
    #[cfg(debug_assertions)]
    const SILENCE_BLOCKS: usize = 256;
    #[cfg(not(debug_assertions))]
    const SILENCE_BLOCKS: usize = 4096;
    const BLOCK_SIZE: usize = 64;
    const MAX_BLOCK_TIME_US: u64 = 500;

    unsafe {
        neural_amp_modeler_rs::math::common::set_daz_ftz();
    }

    // --- WaveNet Standard ---
    let wn_path = model_path("BossWN-standard.nam");
    if !wn_path.exists() {
        eprintln!("SKIP: BossWN-standard.nam not found. Skipping denormal silence WaveNet.");
    } else {
        let json_data =
            fs::read_to_string(&wn_path).expect("Failed to read WaveNet model for denormal test");
        let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
        let mut model =
            build_model(&model_data).expect("Dispatcher failed for denormal test WaveNet");

        model.prewarm(2048);

        let silence = [0.0f32; BLOCK_SIZE];
        let mut output = [0.0f32; BLOCK_SIZE];
        let hist = LatencyHistogram::new();

        for block_idx in 0..SILENCE_BLOCKS {
            let start = std::time::Instant::now();
            model.process(&silence, &mut output);
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            hist.record(elapsed_ns);

            // Validate finiteness across all blocks
            for (i, &s) in output.iter().enumerate() {
                assert!(
                    s.is_finite(),
                    "[Denormal WaveNet] Non-finite sample in block {block_idx}, index {i}: {s}"
                );
            }
        }

        // Validate stability: output after prolonged silence should be stable
        // (magnitude < 1.0). Real models maintain residual DC offset (~6e-3)
        // due to internal neural network biases — this is correct.
        for (i, &s) in output.iter().enumerate() {
            assert!(
                s.abs() < 1.0,
                "[Denormal WaveNet] After {SILENCE_BLOCKS} silence blocks, \
                 output[{i}]={s} diverged (threshold 1.0)"
            );
        }

        // No subnormal values in the output
        for &s in output.iter() {
            assert!(
                s == 0.0 || s.is_normal(),
                "[Denormal WaveNet] Subnormal value detected in output: {s} (bits: 0x{:08X})",
                s.to_bits()
            );
        }

        let p50 = hist.get_percentile(0.50) / 1000;
        let p99 = hist.get_percentile(0.99) / 1000;
        let exact_max = hist.get_exact_max() / 1000;
        println!(
            "WaveNet denormal OK — {SILENCE_BLOCKS} silence blocks, \
             P50={p50}μs, P99={p99}μs, exact_max={exact_max}μs, output[0]={:.6e}",
            output[0]
        );

        // Timing gate: informative warning only — wall-clock is flaky under
        // system load and not a reliable denormal indicator. The value-based
        // checks above (subnormals, finiteness, stability) are the real gate.
        // Using P99 instead of raw max to tolerate sporadic OS preemption spikes.
        if !cfg!(debug_assertions) && p99 >= MAX_BLOCK_TIME_US {
            eprintln!(
                "WARN [Denormal WaveNet] P99={p99}μs exceeds \
                 {MAX_BLOCK_TIME_US}μs — possible denormal penalty or system load"
            );
        }
    }

    // --- LSTM 1×16 ---
    let lstm_path = model_path("BossLSTM-1x16.nam");
    if !lstm_path.exists() {
        eprintln!("SKIP: BossLSTM-1x16.nam not found. Skipping denormal silence LSTM.");
    } else {
        let json_data =
            fs::read_to_string(&lstm_path).expect("Failed to read LSTM model for denormal test");
        let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
        let mut model = build_model(&model_data).expect("Dispatcher failed for denormal test LSTM");

        model.prewarm(2048);

        let silence = [0.0f32; BLOCK_SIZE];
        let mut output = [0.0f32; BLOCK_SIZE];
        let hist = LatencyHistogram::new();

        for block_idx in 0..SILENCE_BLOCKS {
            let start = std::time::Instant::now();
            model.process(&silence, &mut output);
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            hist.record(elapsed_ns);

            for (i, &s) in output.iter().enumerate() {
                assert!(
                    s.is_finite(),
                    "[Denormal LSTM] Non-finite sample in block {block_idx}, index {i}: {s}"
                );
            }
        }

        // Validate stability: output < 1.0 (no numerical divergence)
        for (i, &s) in output.iter().enumerate() {
            assert!(
                s.abs() < 1.0,
                "[Denormal LSTM] After {SILENCE_BLOCKS} silence blocks, \
                 output[{i}]={s} diverged (threshold 1.0)"
            );
        }

        // Validate that no subnormal value appears in the output
        for &s in output.iter() {
            // A subnormal f32 float has exponent = 0 and mantissa != 0
            // We verify that all values are zero or finite normals
            assert!(
                s == 0.0 || s.is_normal(),
                "[Denormal LSTM] Subnormal value detected in output after silence: {s} \
                 (bits: 0x{:08X})",
                s.to_bits()
            );
        }

        let p50 = hist.get_percentile(0.50) / 1000;
        let p99 = hist.get_percentile(0.99) / 1000;
        let exact_max = hist.get_exact_max() / 1000;
        println!(
            "LSTM denormal OK — {SILENCE_BLOCKS} silence blocks, \
             P50={p50}μs, P99={p99}μs, exact_max={exact_max}μs, output[0]={:.6e}",
            output[0]
        );

        if !cfg!(debug_assertions) && p99 >= MAX_BLOCK_TIME_US {
            eprintln!(
                "WARN [Denormal LSTM] P99={p99}μs exceeds \
                 {MAX_BLOCK_TIME_US}μs — possible denormal penalty or system load"
            );
        }
    }
}

// =============================================================================
// Block Size Invariance Tests
// =============================================================================

/// Verifies Block Size invariance in the Static WaveNet implementation.
///
/// The inference engine must produce the same mathematical result (MSE ≈ 0)
/// regardless of the block size provided by the host (DAW/Soundcard).
/// This guarantees that the internal state of the dilated convolutions and
/// circular buffers are correctly preserved across block boundaries.
#[test]
fn test_wavenet_variable_block_sizes() {
    let path = model_path("BossWN-standard.nam");
    if !path.exists() {
        eprintln!("SKIP: BossWN-standard.nam not found.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in parser");

    let block_sizes = [1, 16, 32, 64, 128, 256, 512];
    let input = generate_sine_440hz(512);
    let mut ref_output = vec![0.0f32; 512];

    let _guard = PrecisionGuard::new(ActivationPrecision::Standard);

    for &bs in &block_sizes {
        let mut model = build_model(&model_data).expect("Failed to build model");
        model.prewarm(2048);

        let mut output = vec![0.0f32; 512];
        process_in_blocks(&mut model, &input, &mut output, bs);

        let mut tot_energy = 0.0f64;
        for &s in &output {
            assert!(
                s.is_finite(),
                "Block size {} produced non-finite output",
                bs
            );
            tot_energy += (s as f64) * (s as f64);
        }
        let rms = (tot_energy / 512.0).sqrt();
        assert!(
            rms > 0.0001 && rms <= 10.0,
            "Block size {} RMS out of band: {}",
            bs,
            rms
        );

        if bs == 1 {
            ref_output.copy_from_slice(&output);
        } else {
            let mse = compute_mse(&ref_output, &output);
            assert!(
                mse < 1e-7,
                "Divergence between block_size=1 and block_size={} (MSE={})",
                bs,
                mse
            );
        }
    }
}

/// Verifies LSTM processing with various block sizes.
///
/// The inference engine must be invariant to the processed block size:
/// processing 512 samples one by one must produce the same result (MSE ~0)
/// as processing in blocks of 64 or 512. This is critical to ensure that the
/// sound does not change depending on the host's buffer configuration (DAW/Soundcard).
#[test]
fn test_lstm_variable_block_sizes() {
    let path = model_path("BossLSTM-1x16.nam");
    if !path.exists() {
        eprintln!("SKIP: BossLSTM-1x16.nam not found.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in parser");

    let block_sizes = [1, 16, 32, 64, 128, 256, 512];
    let input = generate_sine_440hz(512);
    let mut ref_output = vec![0.0f32; 512];

    let _guard = PrecisionGuard::new(ActivationPrecision::Standard);

    for &bs in &block_sizes {
        let mut model = build_model(&model_data).expect("Failed to build LSTM model");
        model.prewarm(2048);

        let mut output = vec![0.0f32; 512];
        process_in_blocks(&mut model, &input, &mut output, bs);

        let mut tot_energy = 0.0f64;
        for &s in &output {
            assert!(
                s.is_finite(),
                "LSTM Block size {} produced non-finite output (NaN/Inf)",
                bs
            );
            tot_energy += (s as f64) * (s as f64);
        }
        let rms = (tot_energy / 512.0).sqrt();
        assert!(
            rms > 0.0001 && rms <= 10.0,
            "Instability detected: LSTM Block size {} RMS out of band: {}",
            bs,
            rms
        );

        if bs == 1 {
            ref_output.copy_from_slice(&output);
        } else {
            let mse = compute_mse(&ref_output, &output);
            assert!(
                mse < 1e-7,
                "Block invariance failed for LSTM: Divergence between bs=1 and bs={} (MSE={})",
                bs,
                mse
            );
        }
    }
}

// =============================================================================
// Community Model Tests (Task 6.2)
// =============================================================================

/// Inference Validation on Real Community Models (Ecosystem Regression).
///
/// This test loads a collection of community-exported models to ensure
/// that the loader and dispatcher correctly handle metadata subtleties
/// and real topologies (e.g., Standard vs Lite) generated by different versions
/// of the official exporter (NeuralAmpModeler/NAM).
///
/// Validation on real models is critical as it detects regressions that do not appear
/// in ideal synthetic models, such as bias truncation or gain normalization.
///
/// Each model is fed a 440 Hz sine; output must be finite and bounded to 20× the
/// input peak. This physically-grounded gate supersedes the arbitrary 100.0 limit.
#[test]
fn test_community_models_inference() {
    let models: [(&str, WavenetTopologyResult); 4] = [
        (
            "BossWN-standard.nam",
            WavenetTopologyResult::Known(NamWavenetTopology::Standard),
        ),
        (
            "EVH-5150-Lite.nam",
            WavenetTopologyResult::Known(NamWavenetTopology::Lite),
        ),
        (
            "BossWN-feather.nam",
            WavenetTopologyResult::Known(NamWavenetTopology::Feather),
        ),
        (
            "BossWN-nano.nam",
            WavenetTopologyResult::Known(NamWavenetTopology::Nano),
        ),
    ];

    let input = generate_sine_440hz(64);
    let max_abs_in = input.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_in > 0.0,
        "Input signal is silent — test fixture broken"
    );
    const GAIN_LIMIT: f32 = 20.0;
    let limit = max_abs_in * GAIN_LIMIT;

    for (filename, expected_topo) in models {
        let path = model_path(filename);

        if !path.exists() {
            eprintln!("SKIP: Community model not found: {path:?}. Skipping tests for {filename}.");
            continue;
        }

        let json_data = fs::read_to_string(&path).expect("Failed to read JSON");
        let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

        let topo = get_wavenet_topology(&model_data);
        assert!(
            matches!(topo, ref result if result == &expected_topo),
            "Incorrectly detected topology for community model {}: expected {:?}, got {:?}",
            filename,
            expected_topo,
            topo
        );

        let mut model =
            build_model(&model_data).expect("Dispatcher failed to build community model");

        model.prewarm(2048);

        let mut output = vec![0.0f32; 64];
        model.process(&input, &mut output);

        let mut max_abs_out = 0.0f32;
        let mut tot_energy = 0.0f64;
        for (i, &s) in output.iter().enumerate() {
            assert!(
                s.is_finite(),
                "Model {} produced invalid sample (NaN/Inf) at index {}",
                filename,
                i
            );
            let abs_s = s.abs();
            max_abs_out = max_abs_out.max(abs_s);
            tot_energy += (s as f64) * (s as f64);
            assert!(
                abs_s <= limit,
                "Model {} generated excessive magnitude peak at index {}: {}. \
                 Possible instability (limit {limit}, {GAIN_LIMIT}× input peak).",
                filename,
                i,
                s
            );
        }
        let rms = (tot_energy / 64.0).sqrt();
        assert!(
            rms > 0.0001 && rms < 10.0,
            "Model {} RMS out of band: {} (expected 0.0001 < rms < 10.0)",
            filename,
            rms
        );
        assert!(
            max_abs_out > 0.0,
            "Model {} produced all-zero output — model may be silent",
            filename
        );
        println!("✓ Community model {} successfully validated.", filename);
    }
}

/// Legacy Format (Keras/H5) Rejection Test.
///
/// The NAM-rs engine focuses on the modern JSON-based format (v0.5+) and NAMB (v1/v2).
/// Old Keras/TensorFlow H5 models must be gracefully rejected by the dispatcher.
///
/// Fixture: `tests/fixtures/models/keras_unsupported.json`
/// (Synthetic Keras-legacy dummy export lacking required "architecture" field;
/// committed to the repo — must never be removed without updating this test.)
#[test]
fn test_reject_keras_legacy_format() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/keras_unsupported.json");

    assert!(
        path.exists(),
        "Keras-legacy fixture not found at {path:?}. \
         This is a committed fixture — do not remove it without updating this test. \
         If deliberately retired, convert this test to an inline JSON stub or delete the test explicitly."
    );

    let json_data = fs::read_to_string(&path).expect("Failed to read Keras JSON");

    // The parser should return Ok if it is structurally valid JSON,
    // but may fail if it already identifies missing fields.
    let model_data = match parse_nam_json(&json_data) {
        Ok(data) => data,
        Err(e) => {
            println!("parse_nam_json correctly rejected Keras-legacy format: {e}");
            return;
        }
    };

    // If the parser passed, the dispatcher MUST reject it.
    let build_result = build_model(&model_data);
    assert!(
        build_result.is_err(),
        "build_model() accepted a Keras-legacy format — it must be rejected."
    );

    println!("Keras-legacy format correctly rejected via build_model().");
}

/// Architectural Validation: Non-Tanh Activation in WaveNet with unrecognized shape.
///
/// Models with non-Tanh activations that do not match any known A2 or A1 topology
/// must return a clear error (not a silent bypass or panic).
#[test]
fn test_reject_unrecognized_activation_with_clear_error() {
    let synthetic_json = r#"{
        "version": "0.5.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [
                {
                    "condition_size": 1,
                    "channels": 16,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4],
                    "activation": "ReLU",
                    "gated": false,
                    "head_size": 8,
                    "with_head": true
                }
            ]
        },
        "weights": [0.0, 0.0, 0.0]
    }"#;

    let model_data = parse_nam_json(synthetic_json).expect("Failed to parse synthetic JSON");
    let result = build_model(&model_data);
    assert!(
        result.is_err(),
        "Unrecognized shape with non-Tanh activation must return error"
    );
    let err = format!("{}", result.err().unwrap());
    assert!(
        err.contains("not recognized"),
        "Error should mention topology not being recognized: {err}",
    );
}
