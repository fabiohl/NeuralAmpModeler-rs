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

/// 16-wide SIMD loop + masked tail (caller owns `i`).
///
/// For functions that use AVX-512 masked load/store for the tail.
#[macro_export]
macro_rules! gain_kernel_avx512_masked {
    ($i:ident, $len:expr, { $($simd:tt)* }, { $($masked:tt)* }) => {
        while $i + 16 <= $len {
            { $($simd)* }
            $i += 16;
        }
        if $i < $len {
            { $($masked)* }
        }
    };
}

/// 16-wide SIMD loop + scalar tail (caller owns `i`).
///
/// Use when the scalar tail is self-contained.
#[macro_export]
macro_rules! gain_kernel_avx512_scalar {
    ($i:ident, $len:expr, { $($simd:tt)* }, { $($tail:tt)* }) => {
        while $i + 16 <= $len {
            { $($simd)* }
            $i += 16;
        }
        while $i < $len {
            { $($tail)* }
            $i += 1;
        }
    };
}

/// 16-wide SIMD loop only (caller handles tail).
///
/// Use when the scalar tail needs variables computed between the SIMD loop
/// and the tail.
#[macro_export]
macro_rules! gain_simd_avx512 {
    ($i:ident, $len:expr, { $($simd:tt)* }) => {
        while $i + 16 <= $len {
            { $($simd)* }
            $i += 16;
        }
    };
}
