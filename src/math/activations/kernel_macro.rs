// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unification macros for activation-slice SIMD scanning loops.
//!
//! These macros extract the boilerplate slice-iteration logic shared across
//! all activation functions (`avx2` / `avx512`), so each activation kernel
//! only needs to supply its core arithmetic body.

/// AVX2 activation slice kernel: 16-wide (dual `__m256`) loop then
/// 8-wide (single `__m256`) remainder.
///
/// Caller owns `$i` and handles the scalar tail after this macro.
///
/// The caller must wrap the invocation in `unsafe { ... }`.
///
/// # Parameters
/// - `$i`: mutable index variable (e.g. `i`)
/// - `$len`: slice length expression (e.g. `len` or `data.len()`)
/// - `{ $($dual:tt)* }`: body for the 16-wide loop; receives
///   `$i` as current offset
/// - `{ $($single:tt)* }`: body for the 8-wide remainder loop
#[macro_export]
macro_rules! activation_simd_avx2 {
    ($i:ident, $len:expr, { $($dual:tt)* }, { $($single:tt)* }) => {
        while $i + 16 <= $len {
            {
                $($dual)*
            }
            $i += 16;
        }
        while $i + 8 <= $len {
            {
                $($single)*
            }
            $i += 8;
        }
    };
}

/// AVX-512 activation slice kernel: 16-wide (single `__m512`) loop.
///
/// Caller owns `$i` and handles the scalar tail after this macro.
///
/// The caller must wrap the invocation in `unsafe { ... }`.
///
/// # Parameters
/// - `$i`: mutable index variable
/// - `$len`: slice length expression
/// - `{ $($simd:tt)* }`: body for the 16-wide loop
#[macro_export]
macro_rules! activation_simd_avx512 {
    ($i:ident, $len:expr, { $($simd:tt)* }) => {
        while $i + 16 <= $len {
            {
                $($simd)*
            }
            $i += 16;
        }
    };
}
