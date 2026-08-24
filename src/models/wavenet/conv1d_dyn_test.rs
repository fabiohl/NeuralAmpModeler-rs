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

/// Verifies that `Conv1dDyn::from_parts` rejects a weights buffer smaller
/// than the SIMD-padded total required by the interleaved layout.
/// This hardening protects against silent UB caused by out-of-bounds reads
/// in the SIMD convolution kernels for runtime-dimensional models (F-01).
#[test]
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

    let err = match Conv1dDyn::from_parts(weights, bias, false, 1, in_ch, out_ch, k_size) {
        Ok(_) => panic!("sub-dimensioned weights must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("weights buffer is too small"),
        "unexpected error: {err}"
    );
}

/// Verifies that `Conv1dDyn::from_parts` rejects a zero kernel size
/// (would underflow the tap-offset arithmetic on the hot-path — F-01/F-03).
#[test]
fn test_conv1d_dyn_from_parts_rejects_zero_kernel() {
    use crate::loader::dispatcher::wavenet::traits::ConvWeightsOutput;
    use crate::math::common::AlignedVec;

    let weights =
        AlignedVec::new(64, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let bias =
        AlignedVec::new(4, 0.0f32).expect("allocation should succeed for test-sized buffers");

    let err = match Conv1dDyn::from_parts(weights, bias, false, 1, 2, 4, 0) {
        Ok(_) => panic!("kernel_size == 0 must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("kernel_size must be >= 1"),
        "unexpected error: {err}"
    );
}

/// Verifies that `Conv1dDyn::from_parts` rejects a kernel size above
/// `MAX_KERNEL` (the hot-path tap array is fixed at MAX_KERNEL entries — F-01).
#[test]
fn test_conv1d_dyn_from_parts_rejects_kernel_above_max() {
    use crate::loader::dispatcher::wavenet::traits::ConvWeightsOutput;
    use crate::math::common::AlignedVec;
    use crate::models::wavenet::MAX_KERNEL;

    let k_size = MAX_KERNEL + 1;
    let weights = AlignedVec::new(4 * 4 * 2 * k_size, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    let bias =
        AlignedVec::new(4, 0.0f32).expect("allocation should succeed for test-sized buffers");

    let err = match Conv1dDyn::from_parts(weights, bias, false, 1, 2, 4, k_size) {
        Ok(_) => panic!("kernel_size above MAX_KERNEL must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("exceeds maximum supported"),
        "unexpected error: {err}"
    );
}

/// Verifies that a frame index below the warm-up threshold
/// (`frame_idx < (kernel-1)*dilation`) is clamped instead of producing a
/// wrapped (out-of-bounds) tap pointer (F-01).
#[test]
fn test_conv1d_dyn_warmup_underflow_clamped() {
    use crate::math::common::Avx2Math;

    let in_ch = 2;
    let out_ch: usize = 4;
    let kernel = 4;
    let dilation = 8;

    let num_blocks = out_ch.div_ceil(4);
    let total_padded = num_blocks * 4 * in_ch * kernel;

    let weights = AlignedVec::new(total_padded, 1.0f32)
        .expect("allocation should succeed for test-sized buffers");
    let bias =
        AlignedVec::new(out_ch, 0.0f32).expect("allocation should succeed for test-sized buffers");

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

    // Warm-up threshold is (kernel-1)*dilation = 24; a frame_idx of 0 must be
    // clamped to the buffer start (no wrapping, no crash, no UB).
    let layer_buffer = vec![1.0f32; 48 * in_ch];
    let mut block = vec![0.0f32; out_ch];
    unsafe {
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut block, 0, None);
    }
    assert!(block.iter().all(|v| v.is_finite()));
}
