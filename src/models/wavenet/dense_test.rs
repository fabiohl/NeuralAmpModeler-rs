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
