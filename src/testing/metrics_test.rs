// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for [`super`] numeric helpers.

use super::*;

#[test]
fn max_abs_diff_basic() {
    assert_eq!(max_abs_diff(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    assert!((max_abs_diff(&[1.0, 2.0, 3.0], &[1.0, 2.5, 3.0]) - 0.5).abs() < 1e-12);
    assert_eq!(max_abs_diff(&[], &[]), 0.0);
    // Only the common prefix is compared.
    assert_eq!(max_abs_diff(&[1.0, 2.0], &[1.0]), 0.0);
}

#[test]
fn percentile_exact_matches_rank_convention() {
    let mut samples: Vec<u64> = (1..=100).collect();
    // rank = (100 * 0.5) - 1 = 49 -> sorted value 50.
    assert_eq!(percentile_exact(&mut samples, 0.5), 50);
    assert_eq!(percentile_exact_copy(&samples, 0.99), 99);
    assert_eq!(percentile_exact_copy(&samples, 1.0), 100);
    assert_eq!(percentile_exact_copy(&samples, 0.0), 1);
}

#[test]
fn percentile_exact_empty_and_clamped() {
    let mut empty: Vec<u64> = Vec::new();
    assert_eq!(percentile_exact(&mut empty, 0.5), 0);
    let mut one = vec![7u64];
    assert_eq!(percentile_exact(&mut one, 1.5), 7);
    assert_eq!(percentile_exact_copy(&[1, 2, 3], -0.5), 1);
}
