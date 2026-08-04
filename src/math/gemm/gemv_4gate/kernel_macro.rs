// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! GEMV 4-gate kernel macros — parameterized SIMD loop skeletons for LSTM gates.
//!
//! The 4-gate kernel computes the linear projection for the 4 LSTM gates
//! simultaneously. This module provides loop skeleton macros for AVX2 (8-wide).
//! AVX-512 variants will be added when the AVX-512 kernels are macro-ized.

/// 8-wide SIMD outer loop + scalar tail — caller owns `$out_c` and `$in_c`.
///
/// Use for the AVX2 variant which processes 2 input columns per iteration.
#[macro_export]
macro_rules! gemv_4gate_simd_outer_avx2 {
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

/// 8-wide SIMD dual-column inner loop + odd tail — caller owns `$in_c`.
///
/// Processes 2 input columns per iteration with double accumulators (lo/hi)
/// for each gate, breaking the FMA dependency chain.
#[macro_export]
macro_rules! gemv_4gate_inner_dual_avx2 {
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
