// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Verifies the basic "Identity" functionality of a Dense layer
/// (Fully Connected). This is used extensively in the *1x1* connections and
/// WaveNet output channel aggregations (*Skip Connections*).
#[test]
fn test_dense_layer_identity() {
    // 4x4 Identity weight matrix (f32).
    let mut f32_weights = AlignedVec::from_vec(vec![0.0f32; 16])
        .expect("allocation should succeed for test-sized buffers");
    for out_c in 0..4 {
        f32_weights[out_c * 4 + out_c] = 1.0;
    }

    let dense = DenseLayer::<4, 4> {
        weights: f32_weights,
        bias: AlignedVec::from_vec(vec![0.0; 4])
            .expect("allocation should succeed for test-sized buffers"),
        do_bias: false,
    };

    let input = vec![1.5, 2.5, 3.5, 4.5];
    let mut output = vec![0.0; 4];

    // 1x1 dense layers are fundamental for mixing channels without looking at time.
    // SAFETY: `input`/`output` were allocated in this test with lengths matching the
    // dense layer's `IN`/`OUT` per frame (4 each, one frame); `Avx2Math` is the
    // CPUID-selected backend whose `#[target_feature]` matches the host ISA.
    unsafe {
        dense.process_block::<crate::math::common::Avx2Math>(&input, &mut output, 1);
    }

    assert_eq!(output, vec![1.5, 2.5, 3.5, 4.5]);
}

/// Verifies the correct injection of *Bias* tensors in Dense Layers
/// via SIMD pointers, ensuring the final Output's linear alteration.
#[test]
fn test_dense_layer_with_bias() {
    // Identity f32 weights + Bias of 1.0.
    let mut f32_weights = AlignedVec::from_vec(vec![0.0f32; 16])
        .expect("allocation should succeed for test-sized buffers");
    for out_c in 0..4 {
        f32_weights[out_c * 4 + out_c] = 1.0;
    }

    let dense = DenseLayer::<4, 4> {
        weights: f32_weights,
        bias: AlignedVec::from_vec(vec![1.0; 4])
            .expect("allocation should succeed for test-sized buffers"),
        do_bias: true,
    };

    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];

    // The result should be translated by the bias across all 4 channels.
    // SAFETY: `input`/`output` were allocated in this test with lengths matching the
    // dense layer's `IN`/`OUT` per frame (4 each, one frame); `Avx2Math` is the
    // CPUID-selected backend whose `#[target_feature]` matches the host ISA.
    unsafe {
        dense.process_block::<crate::math::common::Avx2Math>(&input, &mut output, 1);
    }

    assert_eq!(output, vec![2.0, 3.0, 4.0, 5.0]);
}

/// Runs a Dense Layer with non-square dimensionality (IN=8, OUT=4).
///
/// Since WaveNet constantly changes matrices (from CH to HEAD and vice-versa),
/// SIMD engines must never assume perfectly symmetric matrices (NxN).
/// This test injects heterogeneous values to ensure that
/// nested FMA loop stops compute correctly up to the exact allocation limit.
#[test]
fn test_dense_layer_rectangular() {
    // Asymmetric Matrix: IN=8, OUT=4 (f32, row-major: in_c * out_ch + out_c).
    let mut f32_weights = AlignedVec::from_vec(vec![0.0f32; 32])
        .expect("allocation should succeed for test-sized buffers"); // 8 * 4
    f32_weights[0] = 1.0; // in_c=0, out_c=0 → 0*4+0
    f32_weights[4] = 2.0; // in_c=1, out_c=0 → 1*4+0
    f32_weights[9] = 3.0; // in_c=2, out_c=1 → 2*4+1
    f32_weights[13] = 4.0; // in_c=3, out_c=1 → 3*4+1
    f32_weights[18] = 0.5; // in_c=4, out_c=2 → 4*4+2
    f32_weights[31] = -1.0; // in_c=7, out_c=3 → 7*4+3

    let dense = DenseLayer::<8, 4> {
        weights: f32_weights,
        bias: AlignedVec::from_vec(vec![0.5, -0.5, 1.0, -1.0])
            .expect("allocation should succeed for test-sized buffers"),
        do_bias: true,
    };

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0; 4];

    // Validate that the SIMD loop correctly handles the matrix row end (stride).
    // SAFETY: `input` (8 elements = `IN` × 1 frame) and `output` (4 elements = `OUT` × 1
    // frame) were allocated in this test matching the dense layer's contract; `Avx2Math`
    // is the CPUID-selected backend whose `#[target_feature]` matches the host ISA.
    unsafe {
        dense.process_block::<crate::math::common::Avx2Math>(&input, &mut output, 1);
    }

    // Manual calculation trace:
    // out[0] = (1.0 * 1.0) + (2.0 * 2.0) + 0.5 = 1.0 + 4.0 + 0.5 = 5.5
    // out[1] = (3.0 * 3.0) + (4.0 * 4.0) - 0.5 = 9.0 + 16.0 - 0.5 = 24.5
    // out[2] = (5.0 * 0.5) + 1.0 = 2.5 + 1.0 = 3.5
    // out[3] = (8.0 * -1.0) - 1.0 = -8.0 - 1.0 = -9.0

    assert_eq!(output[0], 5.5);
    assert_eq!(output[1], 24.5);
    assert_eq!(output[2], 3.5);
    assert_eq!(output[3], -9.0);
}

/// Verifies that `DenseLayer::try_from_parts` fails closed with `Err` (without panicking)
/// on sub-dimensioned buffers and invalid dimensions (F-PERF-01).
#[test]
fn test_dense_layer_try_from_parts_validation() {
    // Under-dimensioned weights buffer (needs IN * OUT = 16, provide 15).
    let short_weights = AlignedVec::from_vec(vec![0.0f32; 15]).unwrap();
    let bias = AlignedVec::from_vec(vec![0.0f32; 4]).unwrap();
    let res = DenseLayer::<4, 4>::try_from_parts(short_weights, bias.clone(), false);
    assert!(
        res.is_err(),
        "DenseLayer should fail closed when weights are sub-dimensioned"
    );

    // Under-dimensioned bias buffer when do_bias is true (needs OUT = 4, provide 3).
    let weights = AlignedVec::from_vec(vec![0.0f32; 16]).unwrap();
    let short_bias = AlignedVec::from_vec(vec![0.0f32; 3]).unwrap();
    let res = DenseLayer::<4, 4>::try_from_parts(weights.clone(), short_bias, true);
    assert!(
        res.is_err(),
        "DenseLayer should fail closed when bias is sub-dimensioned and do_bias=true"
    );

    // Zero input dimension: IN = 0
    let empty_weights = AlignedVec::new(0, 0.0f32).unwrap();
    let res = DenseLayer::<0, 4>::try_from_parts(empty_weights, bias.clone(), false);
    assert!(res.is_err(), "DenseLayer should fail closed when IN == 0");

    // Valid construction succeeds
    let valid = DenseLayer::<4, 4>::try_from_parts(weights, bias, true);
    assert!(
        valid.is_ok(),
        "Valid parameters should construct successfully"
    );
}

/// Verifies that `DenseLayerDyn::try_from_parts` fails closed with `Err` (without panicking)
/// on sub-dimensioned buffers, invalid dimensions, and sub-4 channels (F-PERF-01).
#[test]
fn test_dense_layer_dyn_try_from_parts_validation() {
    use crate::models::wavenet::dense_dyn::DenseLayerDyn;

    // Sub-4 input channel (in_ch = 1 < 4, e.g. rechannel) with sub-dimensioned weights (needs 4, provide 3).
    let short_weights = AlignedVec::from_vec(vec![0.0f32; 3]).unwrap();
    let bias = AlignedVec::from_vec(vec![0.0f32; 4]).unwrap();
    let res = DenseLayerDyn::try_from_parts(short_weights, bias.clone(), false, 1, 4);
    assert!(
        res.is_err(),
        "DenseLayerDyn with in_ch=1 should fail closed when weights < in_ch*out_ch"
    );

    // in_ch = 2, out_ch = 8 with sub-dimensioned weights (needs 16, provide 15).
    let short_weights16 = AlignedVec::from_vec(vec![0.0f32; 15]).unwrap();
    let bias8 = AlignedVec::from_vec(vec![0.0f32; 8]).unwrap();
    let res = DenseLayerDyn::try_from_parts(short_weights16, bias8.clone(), false, 2, 8);
    assert!(
        res.is_err(),
        "DenseLayerDyn with in_ch=2 should fail closed when weights < 16"
    );

    // in_ch = 0 or out_ch = 0
    let empty_weights = AlignedVec::new(0, 0.0f32).unwrap();
    let res_in0 = DenseLayerDyn::try_from_parts(empty_weights.clone(), bias.clone(), false, 0, 4);
    assert!(
        res_in0.is_err(),
        "DenseLayerDyn with in_ch=0 must fail closed"
    );
    let res_out0 = DenseLayerDyn::try_from_parts(empty_weights, bias.clone(), false, 4, 0);
    assert!(
        res_out0.is_err(),
        "DenseLayerDyn with out_ch=0 must fail closed"
    );

    // Under-dimensioned bias buffer when do_bias is true.
    let weights = AlignedVec::from_vec(vec![0.0f32; 8]).unwrap();
    let short_bias = AlignedVec::from_vec(vec![0.0f32; 3]).unwrap();
    let res = DenseLayerDyn::try_from_parts(weights.clone(), short_bias, true, 2, 4);
    assert!(
        res.is_err(),
        "DenseLayerDyn should fail closed when bias is sub-dimensioned and do_bias=true"
    );

    // Valid construction succeeds
    let valid = DenseLayerDyn::try_from_parts(weights, bias, true, 2, 4);
    assert!(
        valid.is_ok(),
        "Valid parameters should construct successfully"
    );
}

/// Verifies that `Conv1d::try_from_parts` and `Conv1dDyn::try_from_parts` fail closed
/// with `Err` on sub-dimensioned buffers and invalid dimensions, including sub-4 input channels (F-PERF-01).
#[test]
fn test_conv1d_try_from_parts_validation() {
    // Conv1d::<1, 4, 3>: in_ch = 1 < 4, out_ch = 4, K = 3.
    // interleave_width = 4, num_blocks = 1, padded_total = 1 * 4 * 1 * 3 = 12.
    let short_weights = AlignedVec::from_vec(vec![0.0f32; 11]).unwrap();
    let bias = AlignedVec::from_vec(vec![0.0f32; 4]).unwrap();
    let res = Conv1d::<1, 4, 3>::try_from_parts(short_weights, bias.clone(), false, 1);
    assert!(
        res.is_err(),
        "Conv1d with in_ch=1 should fail closed when weights < padded_total"
    );

    // Sub-dimensioned bias when do_bias = true
    let weights = AlignedVec::from_vec(vec![0.0f32; 12]).unwrap();
    let short_bias = AlignedVec::from_vec(vec![0.0f32; 3]).unwrap();
    let res = Conv1d::<1, 4, 3>::try_from_parts(weights.clone(), short_bias, true, 1);
    assert!(
        res.is_err(),
        "Conv1d should fail closed when bias < OUT and do_bias=true"
    );

    // Valid Conv1d construction succeeds
    let valid = Conv1d::<1, 4, 3>::try_from_parts(weights, bias.clone(), true, 1);
    assert!(
        valid.is_ok(),
        "Valid Conv1d parameters should construct successfully"
    );

    // Conv1dDyn: in_ch = 1 < 4, out_ch = 4, kernel = 3, interleave_width = 4.
    // padded_total = 1 * 4 * 1 * 3 = 12.
    let short_weights_dyn = AlignedVec::from_vec(vec![0.0f32; 11]).unwrap();
    let res_dyn = Conv1dDyn::try_from_parts(short_weights_dyn, bias.clone(), false, 1, 1, 4, 3, 4);
    assert!(
        res_dyn.is_err(),
        "Conv1dDyn with in_ch=1 should fail closed when weights < padded_total"
    );

    // Invalid dimensions: in_ch = 0, out_ch = 0, kernel = 0, kernel > MAX_KERNEL
    let empty = AlignedVec::new(0, 0.0f32).unwrap();
    assert!(Conv1dDyn::try_from_parts(empty.clone(), bias.clone(), false, 1, 0, 4, 3, 4).is_err());
    assert!(Conv1dDyn::try_from_parts(empty.clone(), bias.clone(), false, 1, 1, 0, 3, 4).is_err());
    assert!(Conv1dDyn::try_from_parts(empty.clone(), bias.clone(), false, 1, 1, 4, 0, 4).is_err());
    assert!(Conv1dDyn::try_from_parts(empty.clone(), bias.clone(), false, 1, 1, 4, 65, 4).is_err());
    assert!(Conv1dDyn::try_from_parts(empty, bias.clone(), false, 1, 1, 4, 3, 5).is_err());

    // Valid Conv1dDyn construction succeeds
    let valid_weights = AlignedVec::from_vec(vec![0.0f32; 12]).unwrap();
    let valid_dyn = Conv1dDyn::try_from_parts(valid_weights, bias, true, 1, 1, 4, 3, 4);
    assert!(
        valid_dyn.is_ok(),
        "Valid Conv1dDyn parameters should construct successfully"
    );
}
