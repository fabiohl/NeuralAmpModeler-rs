// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use std::assert_matches;

// ── select_partition_size ──

#[test]
fn test_select_partition_size_power_of_two_n() {
    // N=256 is power of two, half=128, + threshold → 128
    assert_eq!(select_partition_size(256), 128);
    // N=512 → largest power ≤ 512 = 512 → ≥ N → half=256 < 512
    assert_eq!(select_partition_size(512), 256);
    // N=1024 → 512
    assert_eq!(select_partition_size(1024), 512);
}

#[test]
fn test_select_partition_size_non_power_of_two_n() {
    // N=300, max_p=150 → largest power of 2 ≤ 150 = 128
    assert_eq!(select_partition_size(300), 128);
    // N=999, max_p=499 → largest power of 2 ≤ 499 = 256
    assert_eq!(select_partition_size(999), 256);
    // N=2047, max_p=1023 → largest power of 2 ≤ 1023 = 512
    assert_eq!(select_partition_size(2047), 512);
}

// ── Mode resolution ──

#[test]
fn test_direct_explicit_always_direct() {
    let model = LinearModel::new(vec![1.0; 1024], 0.0, LinearImplementation::Direct).unwrap();
    assert_matches!(model.mode, LinearMode::Direct);
}

#[test]
fn test_auto_direct_below_threshold() {
    let model = LinearModel::new(vec![1.0; 128], 0.0, LinearImplementation::Auto).unwrap();
    assert_matches!(model.mode, LinearMode::Direct);
}

#[test]
fn test_auto_fft_above_threshold() {
    let model = LinearModel::new(vec![1.0; 512], 0.0, LinearImplementation::Auto).unwrap();
    assert_matches!(model.mode, LinearMode::Fft(_));
}

#[test]
fn test_fft_explicit_above_threshold() {
    let model = LinearModel::new(vec![1.0; 512], 0.0, LinearImplementation::Fft).unwrap();
    assert_matches!(model.mode, LinearMode::Fft(_));
}

#[test]
fn test_fft_explicit_below_threshold_fallback() {
    let model = LinearModel::new(vec![1.0; 128], 0.0, LinearImplementation::Fft).unwrap();
    assert_matches!(model.mode, LinearMode::Direct);
}

// ── FFT process_sample correctness ──

#[test]
fn test_fft_process_basic_no_tail() {
    // P=N: FFT state has zero tail partitions, equivalent to Direct
    let ir: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let mut model = LinearModel::new(ir.clone(), 0.1, LinearImplementation::Direct).unwrap();
    model.prewarm(0);

    let mut model_fft = LinearModel::new(ir, 0.1, LinearImplementation::Fft).unwrap();
    model_fft.prewarm(0);

    let inputs = [0.5, -0.3, 0.8, -0.1, 0.2];
    for &x in &inputs {
        let direct = unsafe { model.process_sample(x) };
        let fft = unsafe { model_fft.process_sample(x) };
        assert!(
            (direct - fft).abs() < 1e-5,
            "direct={direct} fft={fft} mismatch at input={x}"
        );
    }
}

#[test]
fn test_fft_process_with_tail() {
    // P=4, N=8: 2 partitions (head + 1 tail)
    let ir: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
    let mut direct = LinearModel::new(ir.clone(), 0.0, LinearImplementation::Direct).unwrap();
    direct.prewarm(0);

    let mut fft = LinearModel::new(ir, 0.0, LinearImplementation::Fft).unwrap();
    fft.prewarm(0);

    // Feed 100 samples; compare Direct vs FFT output sample by sample
    let mut max_diff = 0.0f32;
    for i in 0..100 {
        let x = (i as f32 * 0.7).sin();
        let d = unsafe { direct.process_sample(x) };
        let f = unsafe { fft.process_sample(x) };
        let diff = (d - f).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(
        max_diff < 1e-4,
        "max diff between Direct and FFT = {max_diff}"
    );
}

#[test]
fn test_fft_long_tail_many_partitions() {
    // P=256 but N=2048 (> 256 so actually P would be 1024 from select_partition_size)
    // Let's use explicit Fft with a small P to force many partitions
    // We'll bypass new() and construct manually for this edge case
    let ir = vec![1.0f32; 2048];
    let fft_state =
        LinearFftState::new(256, &ir).expect("construction should succeed for test-sized buffers");
    let mut aligned =
        AlignedVec::from_vec(ir.clone()).expect("allocation should succeed for test-sized buffers");
    aligned.reverse();
    let history = MirroredBuffer::<f32>::new(2048).unwrap();
    let limit = history.size();

    let mut model = LinearModel {
        weights: aligned,
        bias: 0.0,
        history,
        write_pos: limit,
        receptive_field: 2048,
        double_limit: limit.saturating_mul(2),
        prewarm_on_reset: true,
        implementation: LinearImplementation::Fft,
        mode: LinearMode::Fft(Box::new(fft_state)),
    };
    model.prewarm(0);

    // Verify processing does not panic over 2000 samples
    for i in 0..2000 {
        let x = (i as f32 * 0.3).sin();
        unsafe { model.process_sample(x) };
    }
}

#[test]
fn test_fft_reset_restores_behavior() {
    let ir = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    let mut model = LinearModel::new(ir, 0.2, LinearImplementation::Fft).unwrap();
    model.prewarm(0);

    let out1 = unsafe { model.process_sample(0.7) };
    let out2 = unsafe { model.process_sample(-0.3) };
    let out3 = unsafe { model.process_sample(0.4) };

    model.reset(0, 0);

    let out1b = unsafe { model.process_sample(0.7) };
    let out2b = unsafe { model.process_sample(-0.3) };
    let out3b = unsafe { model.process_sample(0.4) };

    assert!(
        (out1 - out1b).abs() < F32_EQUIVALENCE_TOLERANCE,
        "reset mismatch: {out1} vs {out1b}"
    );
    assert!(
        (out2 - out2b).abs() < F32_EQUIVALENCE_TOLERANCE,
        "reset mismatch: {out2} vs {out2b}"
    );
    assert!(
        (out3 - out3b).abs() < F32_EQUIVALENCE_TOLERANCE,
        "reset mismatch: {out3} vs {out3b}"
    );
}

#[test]
fn test_fft_process_block() {
    let ir = vec![1.0f32; 512];
    let mut model = LinearModel::new(ir, 0.0, LinearImplementation::Fft).unwrap();
    model.prewarm(0);

    let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.2).sin()).collect();
    let mut output = vec![0.0f32; 128];
    unsafe { model.process(&input, &mut output) };

    // Verify no NaN, no infinity
    for &v in &output {
        assert!(v.is_finite(), "output contains non-finite value: {v}");
    }
}

#[test]
fn test_direct_process_block_still_works() {
    let mut model = LinearModel::new(vec![0.5, 0.5], 0.0, LinearImplementation::Direct).unwrap();
    model.prewarm(0);

    let input = [0.1f32; 64];
    let mut output = [0.0f32; 64];
    unsafe { model.process(&input, &mut output) };

    for &v in &output {
        assert!(v.is_finite());
    }
}

// ── Existing tests kept ──

#[test]
fn test_linear_unit_weight() {
    let model = LinearModel::new(vec![1.0], 0.0, LinearImplementation::default()).unwrap();
    assert_eq!(model.receptive_field, 1);
    assert_eq!(model.weights.len(), 1);
}

#[test]
fn test_linear_known_output() {
    let weights = vec![0.2, 0.3, 0.5];
    let bias = 0.1;
    let mut model = LinearModel::new(weights, bias, LinearImplementation::default()).unwrap();

    model.prewarm(0);

    // After prewarm, history is all zeros. Stored weights are reversed: [0.5, 0.3, 0.2]
    // Feed [1.0]: window (oldest→newest) = [0, 0, 1.0]
    //   dot = 0.5*0 + 0.3*0 + 0.2*1.0 = 0.2 + bias=0.1 = 0.3
    let out0 = unsafe { model.process_sample(1.0) };
    let expected0 = 0.5 * 0.0 + 0.3 * 0.0 + 0.2 * 1.0 + 0.1;
    assert!(
        (out0 - expected0).abs() < F32_EQUIVALENCE_TOLERANCE,
        "out0={out0}, expected={expected0}"
    );

    // Feed [2.0]: window (oldest→newest) = [0, 1.0, 2.0]
    //   dot = 0.5*0 + 0.3*1.0 + 0.2*2.0 = 0.7 + bias=0.1 = 0.8
    let out1 = unsafe { model.process_sample(2.0) };
    let expected1 = 0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 2.0 + 0.1;
    assert!(
        (out1 - expected1).abs() < F32_EQUIVALENCE_TOLERANCE,
        "out1={out1}, expected={expected1}"
    );

    // Feed [3.0]: window (oldest→newest) = [1.0, 2.0, 3.0]
    //   dot = 0.5*1.0 + 0.3*2.0 + 0.2*3.0 = 0.5+0.6+0.6 = 1.7 + bias=0.1 = 1.8
    let out2 = unsafe { model.process_sample(3.0) };
    let expected2 = 0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0 + 0.1;
    assert!(
        (out2 - expected2).abs() < F32_EQUIVALENCE_TOLERANCE,
        "out2={out2}, expected={expected2}"
    );
}

#[test]
fn test_linear_zero_output() {
    let mut model =
        LinearModel::new(vec![0.0, 0.0, 0.0], 0.0, LinearImplementation::default()).unwrap();
    model.prewarm(0);
    let out = unsafe { model.process_sample(5.0) };
    assert!((out - 0.0).abs() < F32_EQUIVALENCE_TOLERANCE);
}

#[test]
fn test_linear_process_block() {
    let mut model = LinearModel::new(vec![1.0], 0.0, LinearImplementation::default()).unwrap();
    model.prewarm(0);

    let input = [0.1, 0.2, 0.3];
    let mut output = [0.0f32; 3];
    unsafe { model.process(&input, &mut output) };

    // With weight=1 (reversed), bias=0: output = input
    for i in 0..3 {
        assert!(
            (output[i] - input[i]).abs() < F32_EQUIVALENCE_TOLERANCE,
            "output[{i}]={}, expected={}",
            output[i],
            input[i]
        );
    }
}

#[test]
fn test_linear_reset() {
    let mut model = LinearModel::new(vec![0.5, 0.5], 0.0, LinearImplementation::default()).unwrap();
    model.prewarm(0);

    let out1 = unsafe { model.process_sample(1.0) };
    model.reset(0, 0);

    let out2 = unsafe { model.process_sample(1.0) };
    assert!(
        (out1 - out2).abs() < F32_EQUIVALENCE_TOLERANCE,
        "reset should reproduce the same output: {out1} != {out2}"
    );
}

#[test]
fn test_linear_prewarm_samples_zero() {
    let model = LinearModel::new(vec![1.0; 16], 0.0, LinearImplementation::default()).unwrap();
    assert_eq!(model.prewarm_samples(), 0);
}

// ── Numerical equivalence: Direct vs FFT (f32 precision tolerance) ──

/// f32 FFT precision tolerance for Direct vs FFT equivalence.
const F32_EQUIVALENCE_TOLERANCE: f32 = 1e-4;

/// Helper: returns the maximum absolute difference between Direct and FFT
/// outputs over `num_samples` of a sinusoidal input.
fn max_diff_direct_vs_fft(ir: &[f32], bias: f32, num_samples: usize, freq: f32) -> f32 {
    let mut direct = LinearModel::new(ir.to_vec(), bias, LinearImplementation::Direct).unwrap();
    let mut fft = LinearModel::new(ir.to_vec(), bias, LinearImplementation::Fft).unwrap();
    direct.prewarm(0);
    fft.prewarm(0);

    let mut max_diff = 0.0f32;
    for i in 0..num_samples {
        let x = (i as f32 * freq).sin();
        let d = unsafe { direct.process_sample(x) };
        let f = unsafe { fft.process_sample(x) };
        let diff = (d - f).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}

#[test]
fn test_equivalence_ir_256_sin() {
    let ir: Vec<f32> = (0..256).map(|i| (i as f32 * 0.03).sin()).collect();
    let diff = max_diff_direct_vs_fft(&ir, 0.2, 1024, 0.13);
    assert!(diff < F32_EQUIVALENCE_TOLERANCE, "max diff = {diff}");
}

#[test]
fn test_equivalence_ir_512_sin() {
    let ir: Vec<f32> = (0..512).map(|i| (i as f32 * 0.02).sin()).collect();
    let diff = max_diff_direct_vs_fft(&ir, -0.1, 2048, 0.07);
    assert!(diff < F32_EQUIVALENCE_TOLERANCE, "max diff = {diff}");
}

#[test]
fn test_equivalence_ir_1024_sin() {
    let ir: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.015).sin()).collect();
    let diff = max_diff_direct_vs_fft(&ir, 0.5, 4096, 0.05);
    assert!(diff < F32_EQUIVALENCE_TOLERANCE, "max diff = {diff}");
}

#[test]
fn test_equivalence_ir_2048_sin() {
    let ir: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();
    let diff = max_diff_direct_vs_fft(&ir, 0.0, 4096, 0.03);
    assert!(diff < F32_EQUIVALENCE_TOLERANCE, "max diff = {diff}");
}

#[test]
fn test_equivalence_ir_512_constant_input() {
    let ir: Vec<f32> = (0..512).map(|i| (i as f32) * 0.005).collect();
    let mut direct = LinearModel::new(ir.clone(), 0.3, LinearImplementation::Direct).unwrap();
    let mut fft = LinearModel::new(ir, 0.3, LinearImplementation::Fft).unwrap();
    direct.prewarm(0);
    fft.prewarm(0);

    for i in 0..1024 {
        let x = 0.75;
        let d = unsafe { direct.process_sample(x) };
        let f = unsafe { fft.process_sample(x) };
        let abs_diff = (d - f).abs();
        let scale = d.abs().max(f.abs()).max(1.0);
        assert!(
            abs_diff / scale < F32_EQUIVALENCE_TOLERANCE,
            "mismatch at sample {i}: d={d} f={f} diff={abs_diff}"
        );
    }
}

#[test]
fn test_equivalence_ir_256_impulse() {
    let mut ir = vec![0.0f32; 256];
    ir[0] = 1.0;
    ir[128] = 0.5;
    let mut direct = LinearModel::new(ir.clone(), 0.0, LinearImplementation::Direct).unwrap();
    let mut fft = LinearModel::new(ir, 0.0, LinearImplementation::Fft).unwrap();
    direct.prewarm(0);
    fft.prewarm(0);

    // Impulse at t=0
    let d0 = unsafe { direct.process_sample(1.0) };
    let f0 = unsafe { fft.process_sample(1.0) };
    assert!(
        (d0 - f0).abs() < F32_EQUIVALENCE_TOLERANCE,
        "impulse mismatch at t=0: d={d0} f={f0}"
    );

    // Silence for many samples — both should decay identically
    for i in 1..512 {
        let d = unsafe { direct.process_sample(0.0) };
        let f = unsafe { fft.process_sample(0.0) };
        assert!(
            (d - f).abs() < F32_EQUIVALENCE_TOLERANCE,
            "silence mismatch at sample {i}: d={d} f={f}"
        );
    }
}

#[test]
fn test_equivalence_block_boundary_crossing() {
    // IR 512 → P=256, one tail partition. Cross 3 block boundaries.
    let ir: Vec<f32> = (0..512).map(|i| (i as f32) * 0.01).collect();
    let mut direct = LinearModel::new(ir.clone(), 0.1, LinearImplementation::Direct).unwrap();
    let mut fft = LinearModel::new(ir, 0.1, LinearImplementation::Fft).unwrap();
    direct.prewarm(0);
    fft.prewarm(0);

    // Run 3 * P + 1 samples to ensure block boundaries are exercised
    let total = 3 * 256 + 1;
    for i in 0..total {
        let x = (i as f32 * 0.2).sin();
        let d = unsafe { direct.process_sample(x) };
        let f = unsafe { fft.process_sample(x) };
        assert!(
            (d - f).abs() < F32_EQUIVALENCE_TOLERANCE,
            "block boundary mismatch at sample {i}: d={d} f={f}"
        );
    }
}

#[test]
fn test_equivalence_ir_512_random() {
    let mut seed: u32 = 42;
    let ir: Vec<f32> = (0..512)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) as f32 / 32768.0) * 0.5
        })
        .collect();
    let mut direct = LinearModel::new(ir.clone(), -0.25, LinearImplementation::Direct).unwrap();
    let mut fft = LinearModel::new(ir, -0.25, LinearImplementation::Fft).unwrap();
    direct.prewarm(0);
    fft.prewarm(0);

    seed = 12345;
    for i in 0..1536 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let x = ((seed >> 16) as f32 / 32768.0) * 0.8;
        let d = unsafe { direct.process_sample(x) };
        let f = unsafe { fft.process_sample(x) };
        assert!(
            (d - f).abs() < F32_EQUIVALENCE_TOLERANCE,
            "random mismatch at sample {i}: d={d} f={f}"
        );
    }
}

#[test]
fn test_equivalence_multi_partition_manual() {
    // N=1024, manually set P=128 to get ceil((1024-128)/128) = 7 partitions
    let ir = vec![0.0f32; 1024];
    let mut direct_model = LinearModel::new(ir.clone(), 0.0, LinearImplementation::Direct).unwrap();
    direct_model.prewarm(0);

    let fft_state =
        LinearFftState::new(128, &ir).expect("construction should succeed for test-sized buffers");
    let mut aligned =
        AlignedVec::from_vec(ir.clone()).expect("allocation should succeed for test-sized buffers");
    aligned.reverse();
    let history = MirroredBuffer::<f32>::new(1024).unwrap();
    let limit = history.size();

    let mut fft_model = LinearModel {
        weights: aligned,
        bias: 0.0,
        history,
        write_pos: limit,
        receptive_field: 1024,
        double_limit: limit.saturating_mul(2),
        prewarm_on_reset: true,
        implementation: LinearImplementation::Fft,
        mode: LinearMode::Fft(Box::new(fft_state)),
    };
    fft_model.prewarm(0);

    // Fill IR with nonzero values for meaningful comparison
    let nonzero_ir: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut direct2 =
        LinearModel::new(nonzero_ir.clone(), 0.1, LinearImplementation::Direct).unwrap();
    direct2.prewarm(0);

    let fft_state2 = LinearFftState::new(128, &nonzero_ir)
        .expect("construction should succeed for test-sized buffers");
    let mut aligned2 = AlignedVec::from_vec(nonzero_ir.clone())
        .expect("allocation should succeed for test-sized buffers");
    aligned2.reverse();
    let history2 = MirroredBuffer::<f32>::new(1024).unwrap();
    let limit2 = history2.size();
    let mut fft_model2 = LinearModel {
        weights: aligned2,
        bias: 0.1,
        history: history2,
        write_pos: limit2,
        receptive_field: 1024,
        double_limit: limit2.saturating_mul(2),
        prewarm_on_reset: true,
        implementation: LinearImplementation::Fft,
        mode: LinearMode::Fft(Box::new(fft_state2)),
    };
    fft_model2.prewarm(0);

    // Run enough samples to exercise all 7 partitions (8 blocks × P)
    let total = 8 * 128;
    for i in 0..total {
        let x = (i as f32 * 0.15).sin();
        let d = unsafe { direct2.process_sample(x) };
        let f = unsafe { fft_model2.process_sample(x) };
        assert!(
            (d - f).abs() < F32_EQUIVALENCE_TOLERANCE,
            "multi-partition mismatch at sample {i}: d={d} f={f}"
        );
    }
}

#[test]
fn test_equivalence_after_reset() {
    let ir: Vec<f32> = (0..512).map(|i| (i as f32 * 0.02).sin()).collect();
    let mut direct = LinearModel::new(ir.clone(), 0.0, LinearImplementation::Direct).unwrap();
    let mut fft = LinearModel::new(ir, 0.0, LinearImplementation::Fft).unwrap();
    direct.prewarm(0);
    fft.prewarm(0);

    // Feed some signal
    for i in 0..256 {
        let x = (i as f32 * 0.1).sin();
        unsafe {
            direct.process_sample(x);
            fft.process_sample(x);
        }
    }

    // Reset both
    direct.reset(0, 0);
    fft.reset(0, 0);

    // After reset, outputs must match sample-by-sample
    for i in 0..512 {
        let x = (i as f32 * 0.2).sin();
        let d = unsafe { direct.process_sample(x) };
        let f = unsafe { fft.process_sample(x) };
        assert!(
            (d - f).abs() < F32_EQUIVALENCE_TOLERANCE,
            "post-reset mismatch at sample {i}: d={d} f={f}"
        );
    }
}

#[test]
fn test_equivalence_ir_8192_long_run() {
    // Large IR: 8192 taps → P=4096, one tail partition of 4096 samples
    let ir: Vec<f32> = (0..8192).map(|i| (i as f32 * 0.005).sin()).collect();
    let mut direct = LinearModel::new(ir.clone(), 0.0, LinearImplementation::Direct).unwrap();
    let mut fft = LinearModel::new(ir, 0.0, LinearImplementation::Fft).unwrap();
    direct.prewarm(0);
    fft.prewarm(0);

    // Enough samples to cross one FFT block boundary (P=4096)
    let total = 4096 + 256;
    let mut max_diff = 0.0f32;
    for i in 0..total {
        let x = (i as f32 * 0.05).sin();
        let d = unsafe { direct.process_sample(x) };
        let f = unsafe { fft.process_sample(x) };
        let diff = (d - f).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(
        max_diff < F32_EQUIVALENCE_TOLERANCE,
        "max diff = {max_diff}"
    );
}

#[test]
fn test_equivalence_ir_256_flipped_polarity() {
    // Test with negative weights to exercise sign handling
    let ir: Vec<f32> = (0..256).map(|i| -((i as f32) * 0.02).sin()).collect();
    let diff = max_diff_direct_vs_fft(&ir, -0.5, 1024, 0.17);
    assert!(diff < F32_EQUIVALENCE_TOLERANCE, "max diff = {diff}");
}

#[test]
fn test_equivalence_ir_512_extended_tail() {
    // Extended run over many blocks to verify no drift
    let ir: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).cos()).collect();
    let mut direct = LinearModel::new(ir.clone(), 0.0, LinearImplementation::Direct).unwrap();
    let mut fft = LinearModel::new(ir, 0.0, LinearImplementation::Fft).unwrap();
    direct.prewarm(0);
    fft.prewarm(0);

    // 16 blocks × P=256 = 4096 samples
    let total = 16 * 256;
    let mut max_diff = 0.0f32;
    for i in 0..total {
        let x = (i as f32 * 0.09).sin();
        let d = unsafe { direct.process_sample(x) };
        let f = unsafe { fft.process_sample(x) };
        let diff = (d - f).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(
        max_diff < F32_EQUIVALENCE_TOLERANCE,
        "max diff = {max_diff} across 16 blocks"
    );
}

/// S3.T2 acceptance: with `prewarm_on_reset = false`, `NamModel::reset` must
/// still zero the FIR history and FFT tail. After feeding energy, resetting and
/// processing silence must yield only the bias — no residual FIR energy.
#[test]
fn test_reset_cleans_state_with_prewarm_disabled() {
    for implementation in [
        LinearImplementation::Direct,
        LinearImplementation::Fft,
        LinearImplementation::Auto,
    ] {
        let ir: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin() + 0.1).collect();
        let bias = 0.05f32;
        let mut model = LinearModel::new(ir, bias, implementation).unwrap();
        model.prewarm(0);

        // Feed DC/energy so the FIR history and (for FFT) the tail are dirty.
        for i in 0..2048 {
            let x = 1.0f32 - (i as f32 * 0.01);
            unsafe {
                model.process_sample(x);
            }
        }

        model.set_prewarm_on_reset(false);
        NamModel::reset(&mut model, 48000, 512).unwrap();

        // Processing silence after reset must produce exactly the bias:
        // any residual FIR energy would show up as deviation.
        let zeros = [0.0f32; 256];
        let mut out = [0.0f32; 256];
        NamModel::process(&mut model, &zeros, &mut out);
        for (i, &s) in out.iter().enumerate() {
            assert!(
                (s - bias).abs() < 1e-4,
                "residual energy after reset with prewarm disabled: \
                 sample {i} = {s}, expected bias {bias} (mode {implementation:?})"
            );
        }
    }
}
