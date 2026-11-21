// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::a2_weight_count;

/// Generates a deterministic weight stream of length `n`.
fn make_test_weights(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push(((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25);
    }
    v
}

#[test]
fn test_wavenet_a2_receptive_field_ch3() {
    let model = WaveNetA2::<3>::new().unwrap();
    // Reference: computed from A2_KERNEL_SIZES and A2_DILATIONS arrays.
    let expected = {
        let mut sum = 0usize;
        for i in 0..A2_NUM_LAYERS {
            sum += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
        }
        sum + (A2_HEAD_KERNEL_SIZE - 1)
    };
    assert_eq!(model.receptive_field_size, expected);
    assert_eq!(model.receptive_field(), expected);
    assert_eq!(model.channels(), 3);
}

#[test]
fn test_wavenet_a2_receptive_field_ch8() {
    let model = WaveNetA2::<8>::new().unwrap();
    let expected = {
        let mut sum = 0usize;
        for i in 0..A2_NUM_LAYERS {
            sum += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
        }
        sum + (A2_HEAD_KERNEL_SIZE - 1)
    };
    assert_eq!(model.receptive_field_size, expected);
    assert_eq!(model.channels(), 8);
}

#[test]
fn test_wavenet_a2_process_stub_output_silence() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let input = vec![0.5f32; 64];
    let mut output = vec![1.0f32; 64];
    model.process(&input, &mut output);
    for v in &output {
        assert!(v.abs() < 1e-9, "expected silence, got {}", v);
    }
}

#[test]
fn test_wavenet_a2_process_empty_input() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let input: [f32; 0] = [];
    let mut output: [f32; 0] = [];
    model.process(&input, &mut output);
    // Empty input should be a no-op.
}

#[test]
fn test_wavenet_a2_prewarm_fills_buffers() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    for buf in &mut model.layer_buffers {
        buf.fill(0.5);
    }
    model.head_accum.fill(0.5);
    model.layer_in.fill(0.5);
    model.prewarm();
    for buf in &model.layer_buffers {
        let len = buf.size();
        for &v in buf[..len].iter() {
            assert!(v.abs() < 1e-9, "layer_buffer not zeroed");
        }
    }
    for v in model.head_accum.iter() {
        assert!(v.abs() < 1e-9, "head_accum not zeroed");
    }
    for v in model.layer_in.iter() {
        assert!(v.abs() < 1e-9, "layer_in not zeroed");
    }
    assert_eq!(model.head_write_pos, model.receptive_field_size);
}

#[test]
fn test_wavenet_a2_reset_reallocates_and_prewarms() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let orig_rings: Vec<usize> = model.layer_ring_sizes.clone();
    model.reset(48000, 128).unwrap();
    assert_eq!(model.max_buffer_size, WAVENET_MAX_NUM_FRAMES);
    assert_eq!(model.layer_ring_sizes, orig_rings);
    for buf in &model.layer_buffers {
        let len = buf.size();
        for &v in buf[..len].iter() {
            assert!(v.abs() < 1e-9, "reset layer_buffer not zeroed");
        }
    }
}

#[test]
fn test_wavenet_a2_set_max_buffer_size_noop_on_smaller() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
    model.set_max_buffer_size(32).unwrap();
    assert_eq!(model.layer_ring_sizes, orig_sizes);
    assert_eq!(model.max_buffer_size, WAVENET_MAX_NUM_FRAMES);
}

#[test]
fn test_wavenet_a2_set_max_buffer_size_preserves_ring_capacity_and_clears() {
    let mut model = WaveNetA2::<8>::new().unwrap();
    let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
    model.head_accum.fill(0.5);
    model.head_write_pos = 123;
    model.set_max_buffer_size(8192).unwrap();
    // Buffers stay capped at WAVENET_MAX_NUM_FRAMES (64) without inflating rings.
    assert_eq!(model.max_buffer_size, WAVENET_MAX_NUM_FRAMES);
    assert_eq!(model.layer_ring_sizes, orig_sizes);
    assert_eq!(model.head_write_pos, model.receptive_field_size);
    assert!(model.head_accum.iter().all(|&v| v == 0.0));
}

#[test]
fn test_wavenet_a2_default_creates_valid_model() {
    let model = WaveNetA2::<3>::new().unwrap();
    assert_eq!(model.channels(), 3);
    assert!(model.receptive_field_size > 0);
    assert!(!model.head_accum.is_empty());
    assert!(!model.layer_buffers.is_empty());
    assert_eq!(model.rechannel_w_f32.len(), 3);
    assert_eq!(model.layer_buffers.len(), A2_NUM_LAYERS);
    assert_eq!(model.layer_ring_sizes.len(), A2_NUM_LAYERS);
    assert_eq!(model.layer_lookbacks.len(), A2_NUM_LAYERS);
    assert_eq!(model.layer_buffer_starts.len(), A2_NUM_LAYERS);
    assert_eq!(model.layer_in.len(), 3 * model.max_buffer_size);
}

#[test]
fn test_wavenet_a2_const_receptive_field_matches_runtime() {
    let rf_const = a2_receptive_field();
    let model3 = WaveNetA2::<3>::new().unwrap();
    let model8 = WaveNetA2::<8>::new().unwrap();
    assert_eq!(model3.receptive_field_size, rf_const);
    assert_eq!(model8.receptive_field_size, rf_const);
}

// ── set_weights tests ──────────────────────────────────────────────

#[test]
fn test_set_weights_exact_count_ch3() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let count = a2_weight_count::<3>();
    assert_eq!(count, 1871); // sanity-check known count
    let weights = make_test_weights(count, 42);
    assert!(model.set_weights(&weights).is_ok());
    assert!(model.has_weights());
    assert_eq!(model.layers.len(), A2_NUM_LAYERS);
    assert!(model.head_conv.is_some());
}

#[test]
fn test_set_weights_exact_count_ch8() {
    let mut model = WaveNetA2::<8>::new().unwrap();
    let count = a2_weight_count::<8>();
    assert_eq!(count, 12146); // sanity-check known count
    let weights = make_test_weights(count, 77);
    assert!(model.set_weights(&weights).is_ok());
    assert!(model.has_weights());
    assert_eq!(model.layers.len(), A2_NUM_LAYERS);
    assert!(model.head_conv.is_some());
}

#[test]
fn test_set_weights_wrong_count_ch3_too_few() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let count = a2_weight_count::<3>();
    let weights = make_test_weights(count - 10, 42);
    let err = model.set_weights(&weights);
    assert!(err.is_err(), "expected error with too few weights");
    let err_msg = err.unwrap_err();
    assert!(
        err_msg.contains("stream exhausted"),
        "error should mention exhaustion, got: {err_msg}"
    );
    assert!(!model.has_weights());
}

#[test]
fn test_set_weights_too_many_ch3() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let count = a2_weight_count::<3>();
    let weights = make_test_weights(count + 5, 42);
    let err = model.set_weights(&weights);
    assert!(err.is_err(), "expected error with too many weights");
    let err_msg = err.unwrap_err();
    assert!(
        err_msg.contains("unconsumed"),
        "error should mention unconsumed, got: {err_msg}"
    );
}

#[test]
fn test_set_weights_too_few_ch8() {
    let mut model = WaveNetA2::<8>::new().unwrap();
    let count = a2_weight_count::<8>();
    let weights = make_test_weights(count - 1, 99);
    let err = model.set_weights(&weights);
    assert!(err.is_err(), "expected error with too few weights");
    let err_msg = err.unwrap_err();
    assert!(
        err_msg.contains("stream exhausted"),
        "error should mention exhaustion"
    );
}

#[test]
fn test_set_weights_too_many_ch8() {
    let mut model = WaveNetA2::<8>::new().unwrap();
    let count = a2_weight_count::<8>();
    let weights = make_test_weights(count + 1, 88);
    let err = model.set_weights(&weights);
    assert!(err.is_err(), "expected error with too many weights");
    let err_msg = err.unwrap_err();
    assert!(
        err_msg.contains("unconsumed"),
        "error should mention unconsumed"
    );
}

#[test]
fn test_set_weights_has_weights_flag_ch3() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    assert!(!model.has_weights());
    let count = a2_weight_count::<3>();
    let weights = make_test_weights(count, 123);
    model.set_weights(&weights).unwrap();
    assert!(model.has_weights());
}

/// Smoke: load random weights, prewarm, process 1 frame — output should be non-zero
/// (random weights almost certainly produce non-zero output).
#[test]
fn test_set_weights_process_smoke_ch3() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let count = a2_weight_count::<3>();
    let weights = make_test_weights(count, 42);
    model.set_weights(&weights).unwrap();
    model.prewarm();

    let input = vec![0.5f32; 16];
    let mut output = vec![0.0f32; 16];
    model.process(&input, &mut output);

    // With random weights, output should be non-zero (statistical certainty).
    let any_nonzero = output.iter().any(|&v| v.abs() > 1e-30);
    assert!(
        any_nonzero,
        "process should produce non-zero output after weight loading"
    );
}

#[test]
fn test_set_weights_process_smoke_ch8() {
    let mut model = WaveNetA2::<8>::new().unwrap();
    let count = a2_weight_count::<8>();
    let weights = make_test_weights(count, 77);
    model.set_weights(&weights).unwrap();
    model.prewarm();

    let input = vec![0.5f32; 16];
    let mut output = vec![0.0f32; 16];
    model.process(&input, &mut output);

    let any_nonzero = output.iter().any(|&v| v.abs() > 1e-30);
    assert!(
        any_nonzero,
        "process should produce non-zero output after weight loading"
    );
}

// ── set_weights finitude & bounds tests ─────────────────────────────

/// NaN in weight stream: currently accepted silently (F1 documents the gap).
#[test]
fn test_set_weights_with_nan_ch3() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let count = a2_weight_count::<3>();
    let mut weights = make_test_weights(count, 42);
    weights[0] = f32::NAN;
    let result = model.set_weights(&weights);
    assert!(
        result.is_ok(),
        "NaN in weights is currently accepted silently: {result:?}"
    );
}

#[test]
fn test_set_weights_with_inf_ch3() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let count = a2_weight_count::<3>();
    let mut weights = make_test_weights(count, 42);
    weights[0] = f32::INFINITY;
    let result = model.set_weights(&weights);
    assert!(
        result.is_ok(),
        "Infinity in weights is currently accepted silently: {result:?}"
    );
}

#[test]
fn test_set_weights_with_neg_inf_ch3() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let count = a2_weight_count::<3>();
    let mut weights = make_test_weights(count, 42);
    weights[0] = f32::NEG_INFINITY;
    let result = model.set_weights(&weights);
    assert!(
        result.is_ok(),
        "negative Infinity is currently accepted silently: {result:?}"
    );
}

#[test]
fn test_set_weights_all_zeros_ch3() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let count = a2_weight_count::<3>();
    let weights = vec![0.0f32; count];
    assert!(model.set_weights(&weights).is_ok());
    assert!(model.has_weights());
}

#[test]
fn test_set_weights_all_zeros_ch8() {
    let mut model = WaveNetA2::<8>::new().unwrap();
    let count = a2_weight_count::<8>();
    let weights = vec![0.0f32; count];
    assert!(model.set_weights(&weights).is_ok());
    assert!(model.has_weights());
}

#[test]
fn test_set_weights_extreme_finite_values_ch3() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let count = a2_weight_count::<3>();
    let mut weights = make_test_weights(count, 99);
    weights[0] = 3.4e38;
    weights[1] = -3.4e38;
    let result = model.set_weights(&weights);
    assert!(
        result.is_ok(),
        "extreme finite values should be accepted: {result:?}"
    );
}

#[test]
fn test_set_weights_extreme_finite_values_ch8() {
    let mut model = WaveNetA2::<8>::new().unwrap();
    let count = a2_weight_count::<8>();
    let mut weights = make_test_weights(count, 88);
    weights[0] = 3.4e38;
    weights[1] = -3.4e38;
    let result = model.set_weights(&weights);
    assert!(
        result.is_ok(),
        "extreme finite values should be accepted: {result:?}"
    );
}

#[test]
fn test_set_weights_subnormal_values_ch3() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let count = a2_weight_count::<3>();
    let mut weights = make_test_weights(count, 55);
    weights[0] = 1.0e-45;
    weights[1] = -1.0e-45;
    let result = model.set_weights(&weights);
    assert!(
        result.is_ok(),
        "subnormal values should be accepted: {result:?}"
    );
}

#[test]
fn test_set_weights_subnormal_values_ch8() {
    let mut model = WaveNetA2::<8>::new().unwrap();
    let count = a2_weight_count::<8>();
    let mut weights = make_test_weights(count, 66);
    weights[0] = 1.0e-45;
    weights[1] = -1.0e-45;
    let result = model.set_weights(&weights);
    assert!(
        result.is_ok(),
        "subnormal values should be accepted: {result:?}"
    );
}

#[test]
fn test_set_weights_empty_slice_ch3() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    let weights: [f32; 0] = [];
    let err = model.set_weights(&weights);
    assert!(err.is_err(), "empty slice should error");
    let err_msg = err.unwrap_err();
    assert!(
        err_msg.contains("stream exhausted"),
        "error should mention exhaustion: {err_msg}"
    );
}

#[test]
fn test_set_weights_empty_slice_ch8() {
    let mut model = WaveNetA2::<8>::new().unwrap();
    let weights: [f32; 0] = [];
    let err = model.set_weights(&weights);
    assert!(err.is_err(), "empty slice should error");
    let err_msg = err.unwrap_err();
    assert!(
        err_msg.contains("stream exhausted"),
        "error should mention exhaustion: {err_msg}"
    );
}

// ── Block invariance & negative tests ─────────────────────────────────

/// Asserts that kernel frame capacity equals 64.
#[test]
fn test_wavenet_a2_max_kernel_frames_invariant() {
    assert_eq!(
        WAVENET_MAX_NUM_FRAMES, 64,
        "WAVENET_MAX_NUM_FRAMES must be 64; change this test if the constant changes"
    );
}

/// Block invariance: processing in 64-frame external chunks vs single-call
/// (internal chunking) must be bit-identical for CH=3 (Lite).
#[test]
fn test_wavenet_a2_block_invariance_ch3() {
    let count = a2_weight_count::<3>();
    let weights = make_test_weights(count, 42);
    let num_samples = 2048;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect();

    // Single-call model: process() rechunks internally to ≤64.
    let mut model_a = WaveNetA2::<3>::new().unwrap();
    model_a.set_weights(&weights).unwrap();
    model_a.set_max_buffer_size(num_samples).unwrap();
    model_a.prewarm();
    let mut out_a = vec![0.0f32; num_samples];
    model_a.process(&input, &mut out_a);

    // External 64-frame chunk model: same state, explicit sub-blocks.
    let mut model_b = WaveNetA2::<3>::new().unwrap();
    model_b.set_weights(&weights).unwrap();
    model_b.prewarm();
    let mut out_b = vec![0.0f32; num_samples];
    let mut pos = 0;
    while pos < num_samples {
        let end = (pos + 64).min(num_samples);
        model_b.process(&input[pos..end], &mut out_b[pos..end]);
        pos = end;
    }

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "A2 CH=3 block invariance: single-call vs 64-chunk diverges at sample {i}"
        );
    }
}

/// Block invariance: processing in 64-frame external chunks vs single-call
/// (internal chunking) must be bit-identical for CH=8 (Full).
#[test]
fn test_wavenet_a2_block_invariance_ch8() {
    let count = a2_weight_count::<8>();
    let weights = make_test_weights(count, 77);
    let num_samples = 2048;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect();

    let mut model_a = WaveNetA2::<8>::new().unwrap();
    model_a.set_weights(&weights).unwrap();
    model_a.set_max_buffer_size(num_samples).unwrap();
    model_a.prewarm();
    let mut out_a = vec![0.0f32; num_samples];
    model_a.process(&input, &mut out_a);

    let mut model_b = WaveNetA2::<8>::new().unwrap();
    model_b.set_weights(&weights).unwrap();
    model_b.prewarm();
    let mut out_b = vec![0.0f32; num_samples];
    let mut pos = 0;
    while pos < num_samples {
        let end = (pos + 64).min(num_samples);
        model_b.process(&input[pos..end], &mut out_b[pos..end]);
        pos = end;
    }

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "A2 CH=8 block invariance: single-call vs 64-chunk diverges at sample {i}"
        );
    }
}

/// Negative test: internal chunking of blocks >64 must never pass
/// num_frames > 64 to kernels. Verify by processing a non-multiple (65)
/// vs explicit 64+1 split — bit-identical output proves correct chunking. CH=3.
#[test]
fn test_wavenet_a2_neg_internal_chunking_ch3() {
    let count = a2_weight_count::<3>();
    let weights = make_test_weights(count, 42);
    let num_samples = 65;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect();

    let mut model_a = WaveNetA2::<3>::new().unwrap();
    model_a.set_weights(&weights).unwrap();
    model_a.set_max_buffer_size(num_samples).unwrap();
    model_a.prewarm();
    let mut out_a = vec![0.0f32; num_samples];
    model_a.process(&input, &mut out_a);

    let mut model_b = WaveNetA2::<3>::new().unwrap();
    model_b.set_weights(&weights).unwrap();
    model_b.prewarm();
    let mut out_b = vec![0.0f32; num_samples];
    let end0 = 64.min(num_samples);
    model_b.process(&input[..end0], &mut out_b[..end0]);
    if num_samples > 64 {
        model_b.process(&input[end0..], &mut out_b[end0..]);
    }

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "A2 CH=3 chunking negative test: sample {i} diverges"
        );
    }
}

/// Negative test: same as above for CH=8.
#[test]
fn test_wavenet_a2_neg_internal_chunking_ch8() {
    let count = a2_weight_count::<8>();
    let weights = make_test_weights(count, 77);
    let num_samples = 65;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect();

    let mut model_a = WaveNetA2::<8>::new().unwrap();
    model_a.set_weights(&weights).unwrap();
    model_a.set_max_buffer_size(num_samples).unwrap();
    model_a.prewarm();
    let mut out_a = vec![0.0f32; num_samples];
    model_a.process(&input, &mut out_a);

    let mut model_b = WaveNetA2::<8>::new().unwrap();
    model_b.set_weights(&weights).unwrap();
    model_b.prewarm();
    let mut out_b = vec![0.0f32; num_samples];
    let end0 = 64.min(num_samples);
    model_b.process(&input[..end0], &mut out_b[..end0]);
    if num_samples > 64 {
        model_b.process(&input[end0..], &mut out_b[end0..]);
    }

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "A2 CH=8 chunking negative test: sample {i} diverges"
        );
    }
}

#[test]
fn test_wavenet_a2_processes_512_and_16384_without_set_max_buffer_size() {
    for num_samples in [512usize, 16384usize] {
        let count = a2_weight_count::<8>();
        let weights = make_test_weights(count, 123);
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (i as f32 * 0.037).sin() * 0.5)
            .collect();

        // Direct build without set_max_buffer_size: must process 100% of samples.
        let mut model_full = WaveNetA2::<8>::new().unwrap();
        model_full.set_weights(&weights).unwrap();
        model_full.prewarm();
        let mut out_full = vec![f32::NAN; num_samples];
        model_full.process(&input, &mut out_full);

        let mut model_chunked = WaveNetA2::<8>::new().unwrap();
        model_chunked.set_weights(&weights).unwrap();
        model_chunked.prewarm();
        let mut out_chunked = vec![0.0f32; num_samples];
        for (in_c, out_c) in input.chunks(64).zip(out_chunked.chunks_mut(64)) {
            model_chunked.process(in_c, out_c);
        }

        for (i, (&a, &b)) in out_full.iter().zip(out_chunked.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "WaveNetA2<8> len={num_samples} diverged at sample {i}: full={a}, chunked={b}"
            );
        }
        assert!(out_full.iter().any(|&v| v != 0.0));
    }
}

#[test]
fn test_wavenet_a2_reset_determinism_with_and_without_prewarm() {
    let count = a2_weight_count::<3>();
    let weights = make_test_weights(count, 42);
    let probe: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin() * 0.4).collect();
    let dirty_in = vec![0.9f32; 64];
    let mut scratch = vec![0.0f32; 128];

    // 1. Prewarm enabled: dirty -> reset(48000, 8192) must match fresh prewarmed model bit-for-bit.
    let mut fresh_warm = WaveNetA2::<3>::new().unwrap();
    fresh_warm.set_weights(&weights).unwrap();
    fresh_warm.prewarm();

    let mut reset_warm = WaveNetA2::<3>::new().unwrap();
    reset_warm.set_weights(&weights).unwrap();
    reset_warm.prewarm();
    reset_warm.process(&dirty_in, &mut scratch[..64]);
    reset_warm.reset(48000, 8192).unwrap();

    let mut out_fresh = vec![0.0f32; probe.len()];
    let mut out_reset = vec![0.0f32; probe.len()];
    fresh_warm.process(&probe, &mut out_fresh);
    reset_warm.process(&probe, &mut out_reset);
    for (i, (&a, &b)) in out_fresh.iter().zip(out_reset.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "WaveNetA2<3> reset(prewarm=true) diverged at sample {i}: {a} vs {b}"
        );
    }

    // 2. Prewarm disabled: dirty -> reset(48000, 64) must clear all state to match un-prewarmed model.
    let mut fresh_cold = WaveNetA2::<3>::new().unwrap();
    fresh_cold.set_weights(&weights).unwrap();

    let mut reset_cold = WaveNetA2::<3>::new().unwrap();
    reset_cold.set_weights(&weights).unwrap();
    reset_cold.process(&dirty_in, &mut scratch[..64]);
    reset_cold.prewarm_on_reset = false;
    reset_cold.reset(48000, 64).unwrap();

    fresh_cold.process(&probe, &mut out_fresh);
    reset_cold.process(&probe, &mut out_reset);
    for (i, (&a, &b)) in out_fresh.iter().zip(out_reset.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "WaveNetA2<3> reset(prewarm=false) diverged at sample {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_inject_rt_status_rt_same_instance_is_noop() {
    use crate::common::spsc::RtStatusFlags;
    use std::sync::Arc;

    let flags = Arc::new(RtStatusFlags::new());
    let mut model = WaveNetA2::<3>::new().unwrap();
    // Off-RT loader pre-injects the canonical instance.
    model.inject_rt_status(Arc::clone(&flags));
    let strong_before = Arc::strong_count(&flags);

    // Repeated RT injection of the same instance must neither change the
    // reference count nor replace the stored pointer.
    for _ in 0..1024 {
        model.inject_rt_status_rt(&flags);
    }

    assert_eq!(
        Arc::strong_count(&flags),
        strong_before,
        "RT re-injection must not touch the strong count"
    );
    assert!(
        model
            .rt_status
            .as_ref()
            .is_some_and(|stored| Arc::ptr_eq(stored, &flags)),
        "stored pointer must remain the pre-injected instance"
    );
}

#[test]
fn test_inject_rt_status_rt_refuses_foreign_instance() {
    use crate::common::spsc::RtStatusFlags;
    use std::sync::Arc;

    let flags = Arc::new(RtStatusFlags::new());
    let foreign = Arc::new(RtStatusFlags::new());
    let mut model = WaveNetA2::<8>::new().unwrap();
    model.inject_rt_status(Arc::clone(&flags));
    let strong_before = Arc::strong_count(&flags);
    let foreign_before = Arc::strong_count(&foreign);

    // Replacing a live instance on RT would risk `Arc::drop_slow`; the call is
    // a documented no-op instead.
    model.inject_rt_status_rt(&foreign);

    assert_eq!(Arc::strong_count(&flags), strong_before);
    assert_eq!(Arc::strong_count(&foreign), foreign_before);
    assert!(
        model
            .rt_status
            .as_ref()
            .is_some_and(|stored| Arc::ptr_eq(stored, &flags))
    );
}

#[test]
fn test_inject_rt_status_rt_adopts_when_empty() {
    use crate::common::spsc::RtStatusFlags;
    use std::sync::Arc;

    let flags = Arc::new(RtStatusFlags::new());
    let strong_before = Arc::strong_count(&flags);
    let mut model = WaveNetA2::<3>::new().unwrap();

    model.inject_rt_status_rt(&flags);

    assert_eq!(
        Arc::strong_count(&flags),
        strong_before + 1,
        "adoption must add exactly one strong reference"
    );
    assert!(
        model
            .rt_status
            .as_ref()
            .is_some_and(|stored| Arc::ptr_eq(stored, &flags))
    );
}
