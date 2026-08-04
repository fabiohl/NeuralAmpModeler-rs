// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::Avx2Math;

#[test]
fn test_conv1d_dyn_padding_non_multiple_of_4() {
    let in_ch = 2;
    let out_ch: usize = 6;
    let kernel = 3;
    let dilation = 1;

    let num_blocks = out_ch.div_ceil(4);
    let total_padded = num_blocks * 4 * in_ch * kernel;

    let mut raw_weights = vec![0.0f32; out_ch * kernel * in_ch];
    for out_c in 0..out_ch {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                let idx = (out_c * in_ch + in_c) * kernel + k;
                raw_weights[idx] = (out_c + 1) as f32;
            }
        }
    }

    let mut weights = AlignedVec::new(total_padded, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    for b in 0..num_blocks {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                for lane in 0..4 {
                    let out_c = b * 4 + lane;
                    let target_idx = b * (kernel * in_ch * 4) + k * (in_ch * 4) + in_c * 4 + lane;
                    if out_c < out_ch {
                        let raw_idx = (out_c * in_ch + in_c) * kernel + k;
                        weights[target_idx] = raw_weights[raw_idx];
                    } else {
                        weights[target_idx] = 0.0;
                    }
                }
            }
        }
    }

    let bias = AlignedVec::from_vec(vec![0.5f32; out_ch])
        .expect("allocation should succeed for test-sized buffers");

    let conv = Conv1dDyn {
        weights,
        bias,
        do_bias: true,
        dilation,
        in_ch,
        out_ch,
        num_blocks: out_ch.div_ceil(4),
        interleave_width: 4,
        kernel,
    };

    let layer_buffer = vec![1.0f32; 5 * in_ch];
    let mut block = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut block, 4, None);
    }

    let expected = vec![6.5, 12.5, 18.5, 24.5, 30.5, 36.5];
    assert_eq!(block, expected);
}

#[test]
fn test_conv1d_dyn_large_kernel_no_segfault() {
    let in_ch = 2;
    let out_ch: usize = 4;
    let kernel = 10;
    let dilation = 1;

    let num_blocks = out_ch.div_ceil(4);
    let total_padded = num_blocks * 4 * in_ch * kernel;

    let mut weights = AlignedVec::new(total_padded, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    for i in 0..total_padded {
        weights[i] = 1.0;
    }

    let bias = AlignedVec::from_vec(vec![0.5f32; out_ch])
        .expect("allocation should succeed for test-sized buffers");

    let conv = Conv1dDyn {
        weights,
        bias,
        do_bias: true,
        dilation,
        in_ch,
        out_ch,
        num_blocks: out_ch.div_ceil(4),
        interleave_width: 4,
        kernel,
    };

    let layer_buffer = vec![1.0f32; 24];
    let mut out_f0 = vec![0.0f32; out_ch];
    let mut out_f1 = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut out_f0, 9, None);
        conv.process_dual_frame::<Avx2Math>(
            &layer_buffer,
            &mut out_f0,
            &mut out_f1,
            9,
            10,
            None,
            None,
        );
    }

    // Single frame calculation: bias (0.5) + 10 (taps) * 2 (channels) * 1.0 (input) * 1.0 (weight) = 20.5
    for val in out_f0 {
        assert!((val - 20.5).abs() < 1e-4);
    }
    for val in out_f1 {
        assert!((val - 20.5).abs() < 1e-4);
    }
}

/// Verifies that `Conv1dDyn::from_parts` panics when the weights buffer
/// is smaller than the SIMD-padded total required by the interleaved layout.
/// This hardening protects against silent UB caused by out-of-bounds reads
/// in the SIMD convolution kernels for runtime-dimensional models.
#[test]
#[should_panic(expected = "Conv1d weights buffer is too small")]
fn test_conv1d_dyn_from_parts_subdimensioned_weights() {
    use crate::loader::dispatcher::wavenet::layout::select_interleave_width;
    use crate::loader::dispatcher::wavenet::traits::ConvWeightsOutput;
    use crate::math::common::AlignedVec;

    let in_ch = 2;
    let out_ch: usize = 6;
    let k_size = 3;
    let interleave_width = select_interleave_width(out_ch);
    let num_blocks = out_ch.div_ceil(interleave_width);
    let padded_total = num_blocks * interleave_width * in_ch * k_size;

    let undersized = padded_total / 2;
    let weights = AlignedVec::new(undersized, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    let bias =
        AlignedVec::new(out_ch, 0.0f32).expect("allocation should succeed for test-sized buffers");

    Conv1dDyn::from_parts(weights, bias, false, 1, in_ch, out_ch, k_size);
}
