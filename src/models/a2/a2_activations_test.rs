// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_activation_tanh() {
    let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
    ActivationType::Tanh.apply(&mut data);
    let expected = [
        -5.0f32.tanh(),
        -1.0f32.tanh(),
        0.0f32.tanh(),
        1.0f32.tanh(),
        5.0f32.tanh(),
    ];
    for (i, &v) in data.iter().enumerate() {
        assert!((v - expected[i]).abs() < 5e-3, "At index {}", i);
    }
}

#[test]
fn test_activation_hard_tanh() {
    let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
    ActivationType::HardTanh.apply(&mut data);
    let expected = [-1.0, -1.0, 0.0, 1.0, 1.0];
    assert_eq!(data, expected);
}

#[test]
fn test_activation_fast_tanh() {
    let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
    ActivationType::FastTanh.apply(&mut data);
    for &v in data.iter() {
        assert!((-1.0..=1.0).contains(&v));
    }
    assert!((data[2] - 0.0).abs() < 1e-6);
}

#[test]
fn test_activation_relu() {
    let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
    ActivationType::ReLU.apply(&mut data);
    let expected = [0.0, 0.0, 0.0, 1.0, 5.0];
    assert_eq!(data, expected);
}

#[test]
fn test_activation_leaky_relu() {
    let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
    ActivationType::LeakyReLU {
        negative_slope: 0.1,
    }
    .apply(&mut data);
    let expected = [-0.5, -0.1, 0.0, 1.0, 5.0];
    for (i, &v) in data.iter().enumerate() {
        assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
    }
}

#[test]
fn test_activation_prelu() {
    let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
    ActivationType::PReLU {
        negative_slopes: vec![0.1, 0.2],
    }
    .apply(&mut data);
    let expected = [-0.5, -0.2, 0.0, 1.0, 5.0];
    for (i, &v) in data.iter().enumerate() {
        assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
    }
}

#[test]
fn test_activation_sigmoid() {
    let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
    ActivationType::Sigmoid.apply(&mut data);
    fn sig(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    let expected = [sig(-5.0), sig(-1.0), sig(0.0), sig(1.0), sig(5.0)];
    for (i, &v) in data.iter().enumerate() {
        assert!((v - expected[i]).abs() < 5e-4, "At index {}", i);
    }
}

#[test]
fn test_activation_silu() {
    let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
    ActivationType::SiLU.apply(&mut data);
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
    let expected = [silu(-5.0), silu(-1.0), silu(0.0), silu(1.0), silu(5.0)];
    for (i, &v) in data.iter().enumerate() {
        assert!((v - expected[i]).abs() < 5e-3, "At index {}", i);
    }
}

#[test]
fn test_activation_hard_swish() {
    let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
    ActivationType::HardSwish.apply(&mut data);
    fn hswish(x: f32) -> f32 {
        x * (x + 3.0).clamp(0.0, 6.0) / 6.0
    }
    let expected = [
        hswish(-5.0),
        hswish(-1.0),
        hswish(0.0),
        hswish(1.0),
        hswish(5.0),
    ];
    for (i, &v) in data.iter().enumerate() {
        assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
    }
}

#[test]
fn test_activation_leaky_hardtanh() {
    let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
    ActivationType::LeakyHardTanh {
        min_val: -1.0,
        max_val: 1.0,
        min_slope: 0.1,
        max_slope: 0.2,
    }
    .apply(&mut data);
    let expected = [-1.4, -1.0, 0.0, 1.0, 1.8];
    for (i, &v) in data.iter().enumerate() {
        assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
    }
}

#[test]
fn test_activation_softsign() {
    let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
    ActivationType::Softsign.apply(&mut data);
    fn ss(x: f32) -> f32 {
        x / (1.0 + x.abs())
    }
    let expected = [ss(-5.0), ss(-1.0), ss(0.0), ss(1.0), ss(5.0)];
    for (i, &v) in data.iter().enumerate() {
        assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
    }
}

#[test]
fn test_prelu_empty_slopes() {
    let mut data = [-1.0, 1.0];
    ActivationType::PReLU {
        negative_slopes: vec![],
    }
    .apply(&mut data);
    assert_eq!(data, [-1.0, 1.0]);
}

#[test]
fn test_prelu_cycle() {
    let mut data = [-1.0, -1.0, -1.0, -1.0];
    ActivationType::PReLU {
        negative_slopes: vec![0.1, 0.5],
    }
    .apply(&mut data);
    assert_eq!(data, [-0.1, -0.5, -0.1, -0.5]);
}

#[test]
fn test_hard_tanh_avx2_parity() {
    #[target_feature(enable = "avx2")]
    unsafe fn scalar_ref(data: &mut [f32]) {
        for x in data.iter_mut() {
            *x = x.clamp(-1.0, 1.0);
        }
    }

    let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
    let mut ref_data = simd_data.clone();

    // SAFETY: `simd_data` is a valid `&mut [f32]` outliving the call; AVX2 guaranteed by `#[target_feature]`.
    unsafe { crate::math::activations::hard_tanh_slice_avx2(&mut simd_data) };
    // SAFETY: `ref_data` is a valid `&mut [f32]` outliving the call; AVX2 enabled by `#[target_feature]` on `scalar_ref`.
    unsafe { scalar_ref(&mut ref_data) };

    for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-6,
            "AVX2 parity mismatch at index {}: simd={}, ref={}",
            i,
            s,
            r
        );
    }
}

#[test]
fn test_hard_tanh_avx2_large_slice() {
    let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let mut expected = data.clone();
    for x in expected.iter_mut() {
        *x = x.clamp(-1.0, 1.0);
    }

    // SAFETY: `data` is a valid `&mut [f32]` outliving the call; AVX2 guaranteed by `#[target_feature]`.
    unsafe { crate::math::activations::hard_tanh_slice_avx2(&mut data) };

    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "AVX2 mismatch at index {}: got={}, expected={}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_hard_swish_avx2_parity() {
    #[target_feature(enable = "avx2")]
    unsafe fn scalar_ref(data: &mut [f32]) {
        for x in data.iter_mut() {
            let t = *x + 3.0;
            *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
        }
    }

    let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
    let mut ref_data = simd_data.clone();

    // SAFETY: `simd_data` is a valid `&mut [f32]` outliving the call; AVX2 guaranteed by `#[target_feature]`.
    unsafe { crate::math::activations::hard_swish_slice_avx2(&mut simd_data) };
    // SAFETY: `ref_data` is a valid `&mut [f32]` outliving the call; AVX2 enabled by `#[target_feature]` on `scalar_ref`.
    unsafe { scalar_ref(&mut ref_data) };

    for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-6,
            "AVX2 parity mismatch at index {}: simd={}, ref={}",
            i,
            s,
            r
        );
    }
}

#[test]
fn test_hard_swish_avx2_large_slice() {
    let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let mut expected = data.clone();
    for x in expected.iter_mut() {
        let t = *x + 3.0;
        *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
    }

    // SAFETY: `data` is a valid `&mut [f32]` outliving the call; AVX2 guaranteed by `#[target_feature]`.
    unsafe { crate::math::activations::hard_swish_slice_avx2(&mut data) };

    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "AVX2 mismatch at index {}: got={}, expected={}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_leaky_hard_tanh_avx2_parity() {
    let min_val = -1.0_f32;
    let max_val = 1.0_f32;
    let min_slope = 0.15_f32;
    let max_slope = 0.25_f32;

    #[target_feature(enable = "avx2")]
    unsafe fn scalar_ref(
        data: &mut [f32],
        min_val: f32,
        max_val: f32,
        min_slope: f32,
        max_slope: f32,
    ) {
        for x in data.iter_mut() {
            if *x < min_val {
                *x = (*x - min_val) * min_slope + min_val;
            } else if *x > max_val {
                *x = (*x - max_val) * max_slope + max_val;
            }
        }
    }

    let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
    let mut ref_data = simd_data.clone();

    // SAFETY: `simd_data` is a valid `&mut [f32]` outliving the call; AVX2+FMA guaranteed by `#[target_feature]`.
    unsafe {
        crate::math::activations::leaky_hard_tanh_slice_avx2(
            &mut simd_data,
            min_val,
            max_val,
            min_slope,
            max_slope,
        )
    };
    // SAFETY: `ref_data` and the f32 params are valid and outlive the call; AVX2 enabled by `#[target_feature]` on `scalar_ref`.
    unsafe { scalar_ref(&mut ref_data, min_val, max_val, min_slope, max_slope) };

    for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-6,
            "AVX2 parity mismatch at index {}: simd={}, ref={}",
            i,
            s,
            r
        );
    }
}

#[test]
fn test_leaky_hard_tanh_avx2_large_slice() {
    let min_val = -1.5_f32;
    let max_val = 1.5_f32;
    let min_slope = 0.1_f32;
    let max_slope = 0.3_f32;

    let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let mut expected = data.clone();
    for x in expected.iter_mut() {
        if *x < min_val {
            *x = (*x - min_val) * min_slope + min_val;
        } else if *x > max_val {
            *x = (*x - max_val) * max_slope + max_val;
        }
    }

    // SAFETY: `data` is a valid `&mut [f32]` outliving the call; AVX2+FMA guaranteed by `#[target_feature]`.
    unsafe {
        crate::math::activations::leaky_hard_tanh_slice_avx2(
            &mut data, min_val, max_val, min_slope, max_slope,
        )
    };

    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "AVX2 mismatch at index {}: got={}, expected={}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_fast_tanh_avx2_parity() {
    #[target_feature(enable = "avx2")]
    #[expect(
        clippy::excessive_precision,
        reason = "High-precision constants required for bit-exact numerical validation against reference"
    )]
    unsafe fn scalar_ref(data: &mut [f32]) {
        for x in data.iter_mut() {
            let xv = *x;
            let ax = xv.abs();
            let x2 = xv * xv;
            *x = (xv
                * (2.45550750702956
                    + 2.45550750702956 * ax
                    + (0.893229853513558 + 0.821226666969744 * ax) * x2))
                / (2.44506634652299
                    + (2.44506634652299 + x2) * (xv + 0.814642734961073 * xv * ax).abs());
        }
    }

    let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
    let mut ref_data = simd_data.clone();

    // SAFETY: `simd_data` is a valid `&mut [f32]` outliving the call; AVX2 guaranteed by `#[target_feature]`.
    unsafe { crate::math::activations::fast_tanh_slice_avx2(&mut simd_data) };
    // SAFETY: `ref_data` is a valid `&mut [f32]` outliving the call; AVX2 enabled by `#[target_feature]` on `scalar_ref`.
    unsafe { scalar_ref(&mut ref_data) };

    for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-5,
            "AVX2 parity mismatch at index {}: simd={}, ref={}",
            i,
            s,
            r
        );
    }
}

#[test]
fn test_fast_tanh_avx2_large_slice() {
    #[expect(
        clippy::excessive_precision,
        reason = "High-precision constants required for bit-exact numerical validation against reference"
    )]
    fn fast_tanh_scalar(x: f32) -> f32 {
        let ax = x.abs();
        let x2 = x * x;
        (x * (2.45550750702956
            + 2.45550750702956 * ax
            + (0.893229853513558 + 0.821226666969744 * ax) * x2))
            / (2.44506634652299 + (2.44506634652299 + x2) * (x + 0.814642734961073 * x * ax).abs())
    }

    let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let mut expected = data.clone();
    for x in expected.iter_mut() {
        *x = fast_tanh_scalar(*x);
    }

    // SAFETY: `data` is a valid `&mut [f32]` outliving the call; AVX2 guaranteed by `#[target_feature]`.
    unsafe { crate::math::activations::fast_tanh_slice_avx2(&mut data) };

    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "AVX2 mismatch at index {}: got={}, expected={}",
            i,
            a,
            b
        );
    }
}

// --- AVX-512 parity tests ---

#[cfg(feature = "avx512")]
#[test]
fn test_hard_tanh_avx512_parity() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
    let mut ref_data = simd_data.clone();

    // SAFETY: `simd_data` is a valid `&mut [f32]` outliving the call; AVX-512F is guaranteed by the
    // `is_x86_feature_detected!("avx512f")` guard above.
    unsafe { crate::math::activations::hard_tanh_slice_avx512(&mut simd_data) };
    for x in ref_data.iter_mut() {
        *x = x.clamp(-1.0, 1.0);
    }

    for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-6,
            "AVX-512 parity mismatch at index {}: simd={}, ref={}",
            i,
            s,
            r
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn test_hard_tanh_avx512_large_slice() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let mut expected = data.clone();
    for x in expected.iter_mut() {
        *x = x.clamp(-1.0, 1.0);
    }

    // SAFETY: `data` is a valid `&mut [f32]` outliving the call; AVX-512F is guaranteed by the
    // `is_x86_feature_detected!("avx512f")` guard above.
    unsafe { crate::math::activations::hard_tanh_slice_avx512(&mut data) };

    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "AVX-512 mismatch at index {}: got={}, expected={}",
            i,
            a,
            b
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn test_hard_swish_avx512_parity() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
    let mut ref_data = simd_data.clone();

    // SAFETY: `simd_data` is a valid `&mut [f32]` outliving the call; AVX-512F is guaranteed by the
    // `is_x86_feature_detected!("avx512f")` guard above.
    unsafe { crate::math::activations::hard_swish_slice_avx512(&mut simd_data) };
    for x in ref_data.iter_mut() {
        let t = *x + 3.0;
        *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
    }

    for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-6,
            "AVX-512 parity mismatch at index {}: simd={}, ref={}",
            i,
            s,
            r
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn test_hard_swish_avx512_large_slice() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let mut expected = data.clone();
    for x in expected.iter_mut() {
        let t = *x + 3.0;
        *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
    }

    // SAFETY: `data` is a valid `&mut [f32]` outliving the call; AVX-512F is guaranteed by the
    // `is_x86_feature_detected!("avx512f")` guard above.
    unsafe { crate::math::activations::hard_swish_slice_avx512(&mut data) };

    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "AVX-512 mismatch at index {}: got={}, expected={}",
            i,
            a,
            b
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn test_leaky_hard_tanh_avx512_parity() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    let min_val = -1.0_f32;
    let max_val = 1.0_f32;
    let min_slope = 0.15_f32;
    let max_slope = 0.25_f32;

    let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
    let mut ref_data = simd_data.clone();

    // SAFETY: `simd_data` is a valid `&mut [f32]` outliving the call; AVX-512F is guaranteed by the
    // `is_x86_feature_detected!("avx512f")` guard above.
    unsafe {
        crate::math::activations::leaky_hard_tanh_slice_avx512(
            &mut simd_data,
            min_val,
            max_val,
            min_slope,
            max_slope,
        )
    };
    for x in ref_data.iter_mut() {
        if *x < min_val {
            *x = (*x - min_val) * min_slope + min_val;
        } else if *x > max_val {
            *x = (*x - max_val) * max_slope + max_val;
        }
    }

    for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-6,
            "AVX-512 parity mismatch at index {}: simd={}, ref={}",
            i,
            s,
            r
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn test_leaky_hard_tanh_avx512_large_slice() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    let min_val = -1.5_f32;
    let max_val = 1.5_f32;
    let min_slope = 0.1_f32;
    let max_slope = 0.3_f32;

    let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let mut expected = data.clone();
    for x in expected.iter_mut() {
        if *x < min_val {
            *x = (*x - min_val) * min_slope + min_val;
        } else if *x > max_val {
            *x = (*x - max_val) * max_slope + max_val;
        }
    }

    // SAFETY: `data` is a valid `&mut [f32]` outliving the call; AVX-512F is guaranteed by the
    // `is_x86_feature_detected!("avx512f")` guard above.
    unsafe {
        crate::math::activations::leaky_hard_tanh_slice_avx512(
            &mut data, min_val, max_val, min_slope, max_slope,
        )
    };

    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "AVX-512 mismatch at index {}: got={}, expected={}",
            i,
            a,
            b
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn test_fast_tanh_avx512_parity() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    #[expect(
        clippy::excessive_precision,
        reason = "High-precision constants required for bit-exact numerical validation against reference"
    )]
    fn fast_tanh_scalar(x: f32) -> f32 {
        let ax = x.abs();
        let x2 = x * x;
        (x * (2.45550750702956
            + 2.45550750702956 * ax
            + (0.893229853513558 + 0.821226666969744 * ax) * x2))
            / (2.44506634652299 + (2.44506634652299 + x2) * (x + 0.814642734961073 * x * ax).abs())
    }

    let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
    let mut ref_data = simd_data.clone();

    // SAFETY: `simd_data` is a valid `&mut [f32]` outliving the call; AVX-512F is guaranteed by the
    // `is_x86_feature_detected!("avx512f")` guard above.
    unsafe { crate::math::activations::fast_tanh_slice_avx512(&mut simd_data) };
    for x in ref_data.iter_mut() {
        *x = fast_tanh_scalar(*x);
    }

    for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (s - r).abs() < 1e-5,
            "AVX-512 parity mismatch at index {}: simd={}, ref={}",
            i,
            s,
            r
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn test_fast_tanh_avx512_large_slice() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    #[expect(
        clippy::excessive_precision,
        reason = "High-precision constants required for bit-exact numerical validation against reference"
    )]
    fn fast_tanh_scalar(x: f32) -> f32 {
        let ax = x.abs();
        let x2 = x * x;
        (x * (2.45550750702956
            + 2.45550750702956 * ax
            + (0.893229853513558 + 0.821226666969744 * ax) * x2))
            / (2.44506634652299 + (2.44506634652299 + x2) * (x + 0.814642734961073 * x * ax).abs())
    }

    let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let mut expected = data.clone();
    for x in expected.iter_mut() {
        *x = fast_tanh_scalar(*x);
    }

    // SAFETY: `data` is a valid `&mut [f32]` outliving the call; AVX-512F is guaranteed by the
    // `is_x86_feature_detected!("avx512f")` guard above.
    unsafe { crate::math::activations::fast_tanh_slice_avx512(&mut data) };

    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "AVX-512 mismatch at index {}: got={}, expected={}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_a2_tanh_simd_precision_parity() {
    use crate::math::activations::{ActivationPrecision, set_thread_local_activation_precision};
    use crate::math::common::Avx2Math;

    // Test grid spanning negative, zero, positive, and saturation regions.
    let grid_size = 1024;
    let input_grid: Vec<f32> = (0..grid_size)
        .map(|i| -6.0 + (12.0 * i as f32 / (grid_size - 1) as f32))
        .collect();

    let oracle_f64: Vec<f32> = input_grid
        .iter()
        .map(|&x| (x as f64).tanh() as f32)
        .collect();

    // 1. High-Fidelity path (is_hf = true, Standard precision)
    let mut data_hf = input_grid.clone();
    // SAFETY: data_hf is a valid slice; Avx2Math is supported by baseline x86-64-v3.
    unsafe {
        ActivationType::Tanh.apply_simd_with_precision::<Avx2Math>(&mut data_hf, true);
    }
    let max_err_hf: f32 = data_hf
        .iter()
        .zip(oracle_f64.iter())
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);

    // 2. Fast path (is_hf = false, Fast precision / Padé)
    let mut data_fast = input_grid.clone();
    // SAFETY: data_fast is a valid slice; Avx2Math is supported by baseline x86-64-v3.
    unsafe {
        ActivationType::Tanh.apply_simd_with_precision::<Avx2Math>(&mut data_fast, false);
    }
    let max_err_fast: f32 = data_fast
        .iter()
        .zip(oracle_f64.iter())
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);

    // Measured: max_err_hf = 2.3841858e-7, max_err_fast = 2.3217201e-3
    assert!(
        max_err_hf < 5e-7,
        "Tanh Standard (HF) max error too high: {}",
        max_err_hf
    );
    assert!(
        max_err_fast < 3e-3,
        "Tanh Fast (Padé) max error too high: {}",
        max_err_fast
    );
    assert!(
        max_err_fast > 1e-3,
        "Tanh Fast should exhibit Padé approximation error (~2.32e-3), got {}",
        max_err_fast
    );

    // Verify divergence between HF and Fast paths
    let max_diff: f32 = data_hf
        .iter()
        .zip(data_fast.iter())
        .map(|(&h, &f)| (h - f).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "HF and Fast paths must diverge under Tanh, max_diff={}",
        max_diff
    );

    // 3. Test TLS integration via apply_simd (Standard vs Fast)
    {
        let _guard = set_thread_local_activation_precision(Some(ActivationPrecision::Standard));
        let mut data_tls_std = input_grid.clone();
        // SAFETY: data_tls_std is a valid slice; Avx2Math is supported by baseline x86-64-v3.
        unsafe {
            ActivationType::Tanh.apply_simd::<Avx2Math>(&mut data_tls_std);
        }
        assert_eq!(data_tls_std, data_hf);
    }
    {
        let _guard = set_thread_local_activation_precision(Some(ActivationPrecision::Fast));
        let mut data_tls_fast = input_grid.clone();
        // SAFETY: data_tls_fast is a valid slice; Avx2Math is supported by baseline x86-64-v3.
        unsafe {
            ActivationType::Tanh.apply_simd::<Avx2Math>(&mut data_tls_fast);
        }
        assert_eq!(data_tls_fast, data_fast);
    }
}

#[test]
fn test_a2_sigmoid_simd_precision_parity() {
    use crate::math::activations::{ActivationPrecision, set_thread_local_activation_precision};
    use crate::math::common::Avx2Math;

    // Test grid spanning negative, zero, positive, and saturation regions.
    let grid_size = 1024;
    let input_grid: Vec<f32> = (0..grid_size)
        .map(|i| -6.0 + (12.0 * i as f32 / (grid_size - 1) as f32))
        .collect();

    let oracle_f64: Vec<f32> = input_grid
        .iter()
        .map(|&x| (1.0 / (1.0 + (-x as f64).exp())) as f32)
        .collect();

    // 1. High-Fidelity path (is_hf = true, Standard precision)
    let mut data_hf = input_grid.clone();
    // SAFETY: data_hf is a valid slice; Avx2Math is supported by baseline x86-64-v3.
    unsafe {
        ActivationType::Sigmoid.apply_simd_with_precision::<Avx2Math>(&mut data_hf, true);
    }
    let max_err_hf: f32 = data_hf
        .iter()
        .zip(oracle_f64.iter())
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);

    // 2. Fast path (is_hf = false, Fast precision / Minimax)
    let mut data_fast = input_grid.clone();
    // SAFETY: data_fast is a valid slice; Avx2Math is supported by baseline x86-64-v3.
    unsafe {
        ActivationType::Sigmoid.apply_simd_with_precision::<Avx2Math>(&mut data_fast, false);
    }
    let max_err_fast: f32 = data_fast
        .iter()
        .zip(oracle_f64.iter())
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);

    // Measured: max_err_hf = 2.0861626e-7, max_err_fast = 4.0888786e-4
    assert!(
        max_err_hf < 5e-7,
        "Sigmoid Standard (HF) max error too high: {}",
        max_err_hf
    );
    assert!(
        max_err_fast < 1e-3,
        "Sigmoid Fast (Minimax) max error too high: {}",
        max_err_fast
    );
    assert!(
        max_err_fast > 1e-4,
        "Sigmoid Fast should exhibit minimax approximation error (~4.09e-4), got {}",
        max_err_fast
    );

    // Verify divergence between HF and Fast paths
    let max_diff: f32 = data_hf
        .iter()
        .zip(data_fast.iter())
        .map(|(&h, &f)| (h - f).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-4,
        "HF and Fast paths must diverge under Sigmoid, max_diff={}",
        max_diff
    );

    // 3. Test TLS integration via apply_simd (Standard vs Fast)
    {
        let _guard = set_thread_local_activation_precision(Some(ActivationPrecision::Standard));
        let mut data_tls_std = input_grid.clone();
        // SAFETY: data_tls_std is a valid slice; Avx2Math is supported by baseline x86-64-v3.
        unsafe {
            ActivationType::Sigmoid.apply_simd::<Avx2Math>(&mut data_tls_std);
        }
        assert_eq!(data_tls_std, data_hf);
    }
    {
        let _guard = set_thread_local_activation_precision(Some(ActivationPrecision::Fast));
        let mut data_tls_fast = input_grid.clone();
        // SAFETY: data_tls_fast is a valid slice; Avx2Math is supported by baseline x86-64-v3.
        unsafe {
            ActivationType::Sigmoid.apply_simd::<Avx2Math>(&mut data_tls_fast);
        }
        assert_eq!(data_tls_fast, data_fast);
    }
}
