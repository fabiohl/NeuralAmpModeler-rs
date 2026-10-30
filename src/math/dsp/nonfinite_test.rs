// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_clean_slice_is_finite() {
    let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01 - 0.3).collect();
    assert!(all_finite_f32(&v));
    assert!(all_finite_f32(&[]));
}

#[test]
fn test_nan_at_start_middle_end_detected() {
    for &pos in &[0usize, 17, 63] {
        let mut v = vec![0.25f32; 64];
        v[pos] = f32::NAN;
        assert!(!all_finite_f32(&v), "NaN at {pos} must be detected");
    }
}

#[test]
fn test_infinities_detected() {
    for &bad in &[f32::INFINITY, f32::NEG_INFINITY] {
        let mut v = vec![0.25f32; 65];
        v[33] = bad;
        assert!(!all_finite_f32(&v), "{bad} must be detected");
        // Scalar-tail lane (index 64, beyond the 8-wide loop).
        let mut tail = vec![0.25f32; 65];
        tail[64] = bad;
        assert!(!all_finite_f32(&tail), "{bad} in tail must be detected");
    }
}

#[test]
fn test_sanitize_zeroes_nonfinite_keeps_finite() {
    let mut v = vec![0.5f32; 40];
    v[0] = f32::NAN;
    v[13] = f32::INFINITY;
    v[27] = f32::NEG_INFINITY;
    v[39] = f32::NAN;
    sanitize_nonfinite_f32(&mut v);
    assert!(all_finite_f32(&v));
    for (i, &x) in v.iter().enumerate() {
        let expected = if [0, 13, 27, 39].contains(&i) {
            0.0
        } else {
            0.5
        };
        assert_eq!(x, expected, "mismatch at {i}");
    }
}

#[test]
fn test_sanitize_matches_scalar_reference() {
    let mut v: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
    v[3] = f32::NAN;
    v[50] = f32::INFINITY;
    v[99] = f32::NEG_INFINITY;
    let mut reference = v.clone();
    for x in &mut reference {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
    sanitize_nonfinite_f32(&mut v);
    assert_eq!(v, reference);
}

#[test]
fn test_push_pop_is_zero_alloc() {
    use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};

    let v = vec![0.25f32; 256];
    let mut m = v.clone();
    let _guard = TrackingGuard::new();
    assert!(all_finite_f32(&v));
    sanitize_nonfinite_f32(&mut m);
    assert_eq!(
        get_alloc_count(),
        0,
        "finite scan/sanitize must not allocate on the RT path"
    );
}

#[test]
fn test_sanitize_copy_peak_clean() {
    for len in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 128] {
        let src: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin() * 0.8).collect();
        let mut dst = vec![999.0f32; len];
        let peak = sanitize_copy_peak(&src, &mut dst);
        assert_eq!(src, dst, "clean data must copy identically for len={len}");
        let expected_peak = src.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        assert_eq!(
            peak.to_bits(),
            expected_peak.to_bits(),
            "peak mismatch for len={len}"
        );
    }
}

#[test]
fn test_sanitize_copy_peak_nonfinite() {
    let mut src = vec![0.4f32; 70];
    src[0] = f32::NAN;
    src[7] = f32::INFINITY;
    src[8] = f32::NEG_INFINITY;
    src[31] = f32::NAN;
    src[63] = f32::INFINITY;
    src[64] = f32::NEG_INFINITY; // tail
    src[69] = f32::NAN; // tail
    src[12] = -0.9f32; // max abs peak

    let mut dst = vec![1.0f32; 70];
    let peak = sanitize_copy_peak(&src, &mut dst);

    assert_eq!(peak, 0.9f32);
    assert!(all_finite_f32(&dst));
    for i in [0, 7, 8, 31, 63, 64, 69] {
        assert_eq!(dst[i], 0.0f32, "bad index {i} must be sanitized to 0.0");
    }
    assert_eq!(dst[12], -0.9f32);
    assert_eq!(dst[1], 0.4f32);
}

#[test]
fn test_sanitize_copy_peak_parity_with_separate_passes() {
    for len in [0, 1, 7, 8, 9, 32, 65, 128] {
        let mut src: Vec<f32> = (0..len).map(|i| (i as f32 - 16.0) * 0.05).collect();
        if len > 3 {
            src[1] = f32::NAN;
            src[len - 2] = f32::INFINITY;
        }
        let mut ref_dst = src.clone();
        sanitize_nonfinite_f32(&mut ref_dst);
        let ref_peak = ref_dst.iter().fold(0.0f32, |m, &x| m.max(x.abs()));

        let mut fused_dst = vec![0.0f32; len];
        let fused_peak = sanitize_copy_peak(&src, &mut fused_dst);

        assert_eq!(fused_dst, ref_dst, "dest buffer mismatch for len={len}");
        assert_eq!(
            fused_peak.to_bits(),
            ref_peak.to_bits(),
            "fused peak mismatch for len={len}"
        );
    }
}

#[test]
fn test_sanitize_copy_peak_zero_alloc() {
    use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};

    let src = vec![0.25f32; 256];
    let mut dst = vec![0.0f32; 256];
    let _guard = TrackingGuard::new();
    let _ = sanitize_copy_peak(&src, &mut dst);
    assert_eq!(
        get_alloc_count(),
        0,
        "sanitize_copy_peak must not allocate on the RT path"
    );
}
