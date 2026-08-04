// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Tests that the stack/heap allocated sizes of WaveNet structures
/// match the mathematical specifications for channels and frames.
///
/// Since WaveNet uses fixed vectors and arrays, it is crucial to ensure
/// that `head_outputs` has exact space (e.g., 2 channels * 64 frames),
/// preventing segfaults and "out of bounds" during real-time audio processing.
#[test]
fn test_wavenet_model_allocation() {
    let model = build_tiny_wavenet();
    assert_eq!(model.array1.layers.len(), 3);
    assert_eq!(model.array2.layers.len(), 3);
    assert_eq!(model.array1.head_outputs.len(), 2 * WAVENET_MAX_NUM_FRAMES); // HEAD1=2
    assert_eq!(model.array2.head_outputs.len(), WAVENET_MAX_NUM_FRAMES); // HEAD2=1 (always fixed)
    assert!((model.head_scale - 0.02).abs() < 1e-6);
}

/// Verifies that "hot" initialization (prewarm) of the model does not generate
/// numerical instability like NaN (Not a Number) or Inf (Infinity).
///
/// *Prewarm* serves to stabilize the network (especially delay buffers)
/// by iterating repeatedly with zeros before starting playback. An
/// uninitialized memory bug would rapidly appear here.
#[test]
fn test_wavenet_prewarm_no_nan() {
    let mut model = build_tiny_wavenet();
    model.prewarm();

    // Verify that internal buffers contain no NaN/Inf after prewarm
    for state in &model.array1.states {
        for &v in state.layer_buffer.iter() {
            assert!(v.is_finite(), "NaN/Inf detected in array1 after prewarm");
        }
    }
    for state in &model.array2.states {
        for &v in state.layer_buffer.iter() {
            assert!(v.is_finite(), "NaN/Inf detected in array2 after prewarm");
        }
    }
}

/// Processes an entire block of absolute silence (zeros) and ensures
/// that the model reacts in a linear and predictable fashion.
///
/// Silence at the input should, at most, extract the model's *bias*
/// in a stable manner. Any divergence to NaN or Inf means that
/// the model suffered division by zero or pointer corruption in the SIMD flow.
#[test]
fn test_wavenet_process_zeros() {
    // Instantiate the "tiny" model (mock) and stabilize the historical Ring Buffers.
    let mut model = build_tiny_wavenet();
    model.prewarm();

    // Prepare a block of 16 samples of absolute silence.
    let input = [0.0f32; 16];
    let mut output = [0.0f32; 16];

    // The forward pass should convert silence into stable values (usually small DC offsets).
    model.process(&input, &mut output);

    // Validate that the output signal is finite. If there's division by zero or overflow
    // in any SIMD layer, the result will be NaN or Inf, which is unacceptable in DSP.
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "Output sample [{}] is NaN/Inf: {}", i, v);
    }
}

/// Deterministic integrity test.
///
/// Runs two identical model instances processing the same stimulus.
/// The DSP engine (in this case, the AVX2 SIMD routines) must be mathematically
/// deterministic. Point differences (diverging floats) would indicate
/// state leakage from previous processing (*state bleeding*),
/// which would ruin the phase quality of the generated audio.
#[test]
fn test_wavenet_process_deterministic() {
    // Create two isolated and identical instances to ensure internal state
    // is not shared unsafely (memory leak or thread).
    let mut model_a = build_tiny_wavenet();
    let mut model_b = build_tiny_wavenet();

    model_a.prewarm();
    model_b.prewarm();

    // Apply a constant stimulus of 0.1 (DC impulse).
    let input = [0.1f32; 8];
    let mut out_a = [0.0f32; 8];
    let mut out_b = [0.0f32; 8];

    // Process the same signal in both models.
    model_a.process(&input, &mut out_a);
    model_b.process(&input, &mut out_b);

    // The NAM architecture must be deterministic: the same input must generate the same output
    // bit-for-bit or within hardware rounding tolerance (1e-6).
    for i in 0..8 {
        assert!(
            (out_a[i] - out_b[i]).abs() < 1e-6,
            "Non-deterministic result at sample [{}]: {} vs {}",
            i,
            out_a[i],
            out_b[i]
        );
    }
}
