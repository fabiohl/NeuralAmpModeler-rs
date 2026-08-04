// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet accumulation kernel macros — parameterized SIMD loop skeletons.
//!
//! The macro provides the 8-wide AVX2 loop structure; the caller supplies
//! the SIMD body and tail as token trees. AVX-512 variants will be added
//! when the AVX-512 kernels are macro-ized.

/// 8-wide SIMD loop only (caller handles scalar tail via `#[cold]` fn).
#[macro_export]
macro_rules! wavenet_simd_avx2 {
    ($i:ident, $len:expr, { $($simd:tt)* }) => {
        while $i + 8 <= $len {
            { $($simd)* }
            $i += 8;
        }
    };
}
