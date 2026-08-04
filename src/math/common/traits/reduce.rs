// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Interface definitions for vector reduction operations.

/// Trait for vector reduction mathematical operations (energy, peak absolute, sums).
pub trait VectorReduce {
    /// Computes the mean-square energy: `(1/N) * Σ x_i²`.
    ///
    /// # Safety
    /// `data` must be a valid slice.
    unsafe fn compute_energy(data: &[f32]) -> f32;

    /// Computes `max(|x[i]|)` for a single channel.
    ///
    /// # Safety
    /// `data` must be a valid slice.
    unsafe fn compute_peak_abs_mono(data: &[f32]) -> f32;

    /// Computes `max(|a[i] - b[i]|)`.
    ///
    /// # Safety
    /// `a.len() == b.len()`. Both slices must be valid.
    unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32;

    /// Horizontal sum of `N` consecutive f32 values starting at `ptr`.
    ///
    /// # Safety
    /// `ptr` must point to at least `N` valid, initialized `f32` elements.
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32;
}
