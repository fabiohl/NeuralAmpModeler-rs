// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// 8-wide SIMD loop + scalar tail (caller owns `i`).
///
/// Use when the scalar tail is self-contained and doesn't depend on state
/// computed after the SIMD loop.
#[macro_export]
macro_rules! gain_kernel_avx2 {
    ($i:ident, $len:expr, { $($simd:tt)* }, { $($tail:tt)* }) => {
        while $i + 8 <= $len {
            { $($simd)* }
            $i += 8;
        }
        while $i < $len {
            { $($tail)* }
            $i += 1;
        }
    };
}

/// 8-wide SIMD loop only (caller handles tail).
///
/// Use when the scalar tail needs variables computed between the SIMD loop
/// and the tail (e.g., ramp `g`, clip detection `clipped`, crossfade `one_minus_t`).
#[macro_export]
macro_rules! gain_simd_avx2 {
    ($i:ident, $len:expr, { $($simd:tt)* }) => {
        while $i + 8 <= $len {
            { $($simd)* }
            $i += 8;
        }
    };
}
