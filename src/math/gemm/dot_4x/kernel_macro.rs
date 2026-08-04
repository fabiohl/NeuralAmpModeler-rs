// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dot Product 4x kernel macros — parameterized SIMD loop skeletons.
//!
//! AVX2 (256-bit) loop skeletons for interleaved and dual-frame dot product
//! kernels. AVX-512 variants will be added when the AVX-512 source files
//! are macro-ized.

// ── AVX2 loop skeletons ───────────────────────────────────────────────────────

/// 16-wide SIMD loop — caller owns `$i`.
#[macro_export]
macro_rules! dot4x_simd16_avx2 {
    ($i:ident, $len:expr, { $($simd:tt)* }) => {
        while $i + 16 <= $len {
            { $($simd)* }
            $i += 16;
        }
    };
}

/// 8-wide SIMD loop — caller owns `$i`.
#[macro_export]
macro_rules! dot4x_simd8_avx2 {
    ($i:ident, $len:expr, { $($simd:tt)* }) => {
        while $i + 8 <= $len {
            { $($simd)* }
            $i += 8;
        }
    };
}

/// 8-wide SIMD loop + 2-wide tail — caller owns `$i`.
#[macro_export]
macro_rules! dot4x_simd8_avx2_tail2 {
    ($i:ident, $len:expr, { $($simd:tt)* }, { $($tail2:tt)* }) => {
        while $i + 8 <= $len {
            { $($simd)* }
            $i += 8;
        }
        while $i + 2 <= $len {
            { $($tail2)* }
            $i += 2;
        }
    };
}

/// 4-wide SIMD loop — caller owns `$i` (used for \_\_m128 f32 variants; ISA-neutral).
#[macro_export]
macro_rules! dot4x_simd4 {
    ($i:ident, $len:expr, { $($simd:tt)* }) => {
        while $i + 4 <= $len {
            { $($simd)* }
            $i += 4;
        }
    };
}
