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

    // SAFETY: `layer_buffer` (5 frames × `in_ch`) and `block` (`out_ch`) were allocated in
    // this test so the K=3, dilation=1 taps at `frame_idx` 4 stay in bounds; `Avx2Math`
    // matches the CPU ISA required by `process_single_frame`.
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

    // SAFETY: `layer_buffer` (24 elements) covers the K=10, dilation=1 taps at `frame_idx`
    // 9/10, and `out_f0`/`out_f1` each hold `out_ch` elements; `Avx2Math` matches the CPU
    // ISA required by the single/dual-frame kernels.
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
    // SAFETY: `layer_buffer` (48 frames × `in_ch`) is sized beyond the warm-up threshold
    // (kernel-1)*dilation = 24, and `block` holds `out_ch` elements, so the kernel's
    // clamped tap reads stay in bounds; `Avx2Math` matches the CPU ISA.
    unsafe {
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut block, 0, None);
    }
    assert!(block.iter().all(|v| v.is_finite()));
}

/// A3 / R-2 regression: the release-stable model-side builder
/// (`Conv1dDyn::try_from_parts`) rejects non-SIMD-padded weight buffers for
/// every interleave width (16/8/4) and accepts the exact padded total. The
/// check is `anyhow::ensure!`, i.e. it stays compiled and enforced in release.
#[test]
fn test_conv1d_dyn_try_from_parts_rejects_unpadded_weights_all_widths() {
    use crate::loader::dispatcher::wavenet::layout::select_interleave_width;

    fn run(in_ch: usize, out_ch: usize, kernel: usize) {
        let width = select_interleave_width(out_ch);
        let padded_total = out_ch.div_ceil(width) * width * in_ch * kernel;
        // Raw row-major weight count with no SIMD tail padding.
        let natural_total = out_ch * in_ch * kernel;

        let bias = AlignedVec::new(out_ch, 0.0f32).expect("allocation should succeed in tests");

        // One element short of the padded total must be rejected.
        let short =
            AlignedVec::new(padded_total - 1, 0.0f32).expect("allocation should succeed in tests");
        let err = match Conv1dDyn::try_from_parts(
            short,
            bias.clone(),
            false,
            1,
            in_ch,
            out_ch,
            kernel,
            width,
        ) {
            Ok(_) => panic!("sub-padded weights must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("weights buffer is too small"),
            "unexpected error: {err}"
        );

        // Where SIMD padding is required (out_ch not an exact multiple of the
        // block width), the natural unpadded buffer must also be rejected.
        if natural_total < padded_total {
            let natural =
                AlignedVec::new(natural_total, 0.0f32).expect("allocation should succeed in tests");
            let err = match Conv1dDyn::try_from_parts(
                natural,
                bias.clone(),
                false,
                1,
                in_ch,
                out_ch,
                kernel,
                width,
            ) {
                Ok(_) => panic!("unpadded (natural-size) weights must be rejected"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains("weights buffer is too small"),
                "unexpected error: {err}"
            );
        }

        // The exact padded total must be accepted.
        let padded =
            AlignedVec::new(padded_total, 0.0f32).expect("allocation should succeed in tests");
        let conv = Conv1dDyn::try_from_parts(padded, bias, false, 1, in_ch, out_ch, kernel, width)
            .expect("exact padded buffer must be accepted");
        assert_eq!(conv.interleave_width, width);
        assert_eq!(conv.weights.len(), padded_total);
    }

    run(2, 12, 3); // out_ch = 12 → interleave width 16 (padded from 12 to 16)
    run(2, 8, 3); //  out_ch = 8  → interleave width 8  (exact multiple)
    run(2, 6, 3); //  out_ch = 6  → interleave width 4  (padded from 6 to 8)
}

/// A3 / R-2 regression: an `interleave_width` outside {4, 8, 16} cannot be
/// smuggled through the validated constructor (it would corrupt the hot-path
/// weight slicing).
#[test]
fn test_conv1d_dyn_try_from_parts_rejects_invalid_interleave_width() {
    let weights = AlignedVec::new(4 * 2 * 3 * 4, 0.0f32).expect("allocation should succeed");
    let bias = AlignedVec::new(4, 0.0f32).expect("allocation should succeed");

    let err = match Conv1dDyn::try_from_parts(weights, bias, false, 1, 2, 4, 3, 5) {
        Ok(_) => panic!("interleave_width 5 must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("interleave_width must be 4, 8 or 16"),
        "unexpected error: {err}"
    );
}

/// A6 / R-7 regression: `process_block` with an odd frame count reaches the
/// single-frame remainder path, whose mixin segment `[i*out_ch..(i+1)*out_ch]`
/// used to be sliced unguarded — a `mixin` shorter than `num_frames*out_ch`
/// (caller-contract violation) would panic on the audio thread. The segment is
/// now bounds-clamped like the dual-frame path, so a short mixin must behave
/// exactly as if the caller had zero-padded it up to the full block length.
#[test]
fn test_conv1d_dyn_process_block_short_mixin_remainder_clamped() {
    let in_ch = 2;
    let out_ch: usize = 4;
    let kernel = 3;
    let dilation = 1;
    let interleave_width = 4;

    let num_blocks = out_ch.div_ceil(interleave_width);
    let total_padded = num_blocks * interleave_width * in_ch * kernel;
    let weights = AlignedVec::new(total_padded, 1.0f32)
        .expect("allocation should succeed for test-sized buffers");
    let bias =
        AlignedVec::new(out_ch, 0.0f32).expect("allocation should succeed for test-sized buffers");

    let conv = Conv1dDyn::try_from_parts(
        weights,
        bias,
        false,
        dilation,
        in_ch,
        out_ch,
        kernel,
        interleave_width,
    )
    .expect("valid padded weights must be accepted");

    // 3 frames (odd -> one 2-frame chunk + one single-frame remainder), warm-up
    // threshold (kernel-1)*dilation = 2, so `buffer_start = 2` keeps every tap at
    // frame index >= 0 and <= buffer_start + num_frames - 1 = 4.
    let num_frames = 3;
    let buffer_start = 2;
    let layer_buffer = vec![1.0f32; (buffer_start + num_frames) * in_ch];
    let mut short_out = vec![0.0f32; num_frames * out_ch];

    // Covers the first frame segment fully, the second only partially, and
    // nothing beyond — the remainder frame's segment start (`2*out_ch`) is past
    // the end, which panicked before the R-7 clamp.
    let short_mixin = vec![2.0f32, 2.0, 2.0, 2.0, 3.0, 4.0];

    // SAFETY: `layer_buffer` (5 frames x `in_ch`) covers the K=3, dilation=1 taps of the
    // processed frames 2..=4, `block` holds `num_frames * out_ch` elements, and `short_mixin`
    // is a caller-contract violation that the R-7 clamp must absorb without panicking or
    // reading out of bounds; `Avx2Math` matches the CPU ISA required by the kernels.
    unsafe {
        conv.process_block::<Avx2Math>(
            &layer_buffer,
            &mut short_out,
            buffer_start,
            num_frames,
            Some(&short_mixin),
        );
    }

    // Reference: the same call with the short mixin zero-padded to the full block
    // length must produce bit-identical output (missing mixin lanes read as zero).
    let mut full_mixin = short_mixin.clone();
    full_mixin.resize(num_frames * out_ch, 0.0f32);
    let mut padded_out = vec![0.0f32; num_frames * out_ch];

    // SAFETY: same preconditions as the previous call, now with a full-length mixin.
    unsafe {
        conv.process_block::<Avx2Math>(
            &layer_buffer,
            &mut padded_out,
            buffer_start,
            num_frames,
            Some(&full_mixin),
        );
    }

    assert_eq!(short_out, padded_out);
    assert!(short_out.iter().all(|v| v.is_finite()));
}
