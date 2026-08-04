// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Batch GEMM kernel macros — parameterized SIMD loop skeletons.
//!
//! Provides the loop structure for AVX2 (4 frames × 8 channels, dual-column inner).
//! AVX-512 variants will be added when the AVX-512 kernels are macro-ized.

/// 4-frame batch outer loop — caller owns `$f` and `$num_frames`.
#[macro_export]
macro_rules! gemm_batch_frame_loop_avx2 {
    ($f:ident, $num_frames:expr, { $($batch_body:tt)* }, { $($tail_frame:tt)* }) => {
        while $f + 4 <= $num_frames {
            { $($batch_body)* }
            $f += 4;
        }
        while $f < $num_frames {
            { $($tail_frame)* }
            $f += 1;
        }
    };
}

/// 8-wide output channel loop + scalar tail — caller owns `$out_c`.
#[macro_export]
macro_rules! gemm_batch_outc_loop_avx2 {
    ($out_c:ident, $out_len:expr, { $($simd_body:tt)* }, { $($tail_body:tt)* }) => {
        while $out_c + 8 <= $out_len {
            { $($simd_body)* }
            $out_c += 8;
        }
        while $out_c < $out_len {
            { $($tail_body)* }
            $out_c += 1;
        }
    };
}

/// Dual-column inner loop + odd tail — caller owns `$in_c`.
///
/// Processes 2 input columns per iteration with dual accumulators (lo/hi)
/// for each frame, breaking FMA dependency chains.
#[macro_export]
macro_rules! gemm_batch_inner_dual_avx2 {
    ($in_c:ident, $in_len:expr, { $($simd_body:tt)* }, { $($tail_body:tt)* }) => {
        while $in_c + 2 <= $in_len {
            { $($simd_body)* }
            $in_c += 2;
        }
        if $in_c < $in_len {
            { $($tail_body)* }
        }
    };
}
