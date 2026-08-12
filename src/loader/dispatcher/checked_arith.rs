// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Checked arithmetic helpers for size computations in the loader dispatcher.
//!
//! All multiplications and additions that feed into `AlignedVec` or slice
//! indexing must go through these helpers to prevent silent wrapping in
//! release builds (where `overflow-checks` is disabled by default).
//!
//! DoS/OOM prevention — unchecked arithmetic on adversarial dimensions.

/// Builds a `Layout::from_size_align` with checked element count.
///
/// Returns `Err` on overflow instead of panicking in debug or wrapping in
/// release. Use this before calling `AlignedVec::with_capacity` with
/// untrusted dimensions.
pub fn checked_mul(lhs: usize, rhs: usize) -> anyhow::Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        anyhow::anyhow!(
            "Arithmetic overflow: {lhs} * {rhs} would wrap — \
             DoS protection (F2)."
        )
    })
}

/// Checked multiplication chain: a * b * c.
pub fn checked_mul3(a: usize, b: usize, c: usize) -> anyhow::Result<usize> {
    checked_mul(checked_mul(a, b)?, c)
}

/// Checked multiplication chain: a * b * c * d.
pub fn checked_mul4(a: usize, b: usize, c: usize, d: usize) -> anyhow::Result<usize> {
    checked_mul(checked_mul3(a, b, c)?, d)
}

/// Checked addition.
pub fn checked_add(lhs: usize, rhs: usize) -> anyhow::Result<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        anyhow::anyhow!(
            "Arithmetic overflow: {lhs} + {rhs} would wrap — \
             DoS protection (F2)."
        )
    })
}

/// Computes `num_blocks * width * in_size * k_size` with overflow checking.
/// This is the canonical conv1d padded-total formula used by layout.rs and
/// convnet/mod.rs.
pub fn checked_conv_padded_total(
    num_blocks: usize,
    width: usize,
    in_size: usize,
    k_size: usize,
) -> anyhow::Result<usize> {
    checked_mul4(num_blocks, width, in_size, k_size)
}

/// Computes `out_size * in_size * k_size` with overflow checking.
pub fn checked_conv_total(out_size: usize, in_size: usize, k_size: usize) -> anyhow::Result<usize> {
    checked_mul3(out_size, in_size, k_size)
}

/// Computes `out_size * in_size` with overflow checking.
pub fn checked_dense_total(out_size: usize, in_size: usize) -> anyhow::Result<usize> {
    checked_mul(out_size, in_size)
}
