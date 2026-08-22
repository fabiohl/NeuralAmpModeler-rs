// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

fn make_bn_simple() -> BatchNorm1D {
    // gamma = [1.0, 2.0], beta = [0.0, 0.0]
    // running_mean = [0.0, 0.0], running_var = [4.0, 4.0], eps = 0.0
    // scale = [1/2=0.5, 2/2=1.0], offset = [0, 0]
    BatchNorm1D::from_params(2, &[1.0, 2.0], &[0.0, 0.0], &[0.0, 0.0], &[4.0, 4.0], 0.0)
        .expect("construction should succeed for test-sized buffers")
}

#[test]
fn test_fused_params_simple() {
    let bn = make_bn_simple();
    assert!((bn.scale[0] - 0.5).abs() < 1e-7);
    assert!((bn.scale[1] - 1.0).abs() < 1e-7);
    assert!((bn.offset[0] - 0.0).abs() < 1e-7);
    assert!((bn.offset[1] - 0.0).abs() < 1e-7);
}

#[test]
fn test_fused_params_with_beta_mean() {
    // gamma=[1,1], beta=[2,3], mean=[1,2], var=[1,1], eps=0
    // scale = [1/1=1, 1/1=1]
    // offset[0] = 2 - 1*1 = 1
    // offset[1] = 3 - 2*1 = 1
    let bn = BatchNorm1D::from_params(2, &[1.0, 1.0], &[2.0, 3.0], &[1.0, 2.0], &[1.0, 1.0], 0.0)
        .expect("construction should succeed for test-sized buffers");
    assert!((bn.scale[0] - 1.0).abs() < 1e-7);
    assert!((bn.scale[1] - 1.0).abs() < 1e-7);
    assert!((bn.offset[0] - 1.0).abs() < 1e-7);
    assert!((bn.offset[1] - 1.0).abs() < 1e-7);
}

#[test]
fn test_epsilon_effect() {
    // gamma=[1,1], beta=[0,0], mean=[0,0], var=[0,0], eps=1e-2
    // inv_std = 1/sqrt(0.01) = 1/0.1 = 10
    // scale = 10, offset = 0
    let bn = BatchNorm1D::from_params(1, &[1.0], &[0.0], &[0.0], &[0.0], 1e-2)
        .expect("construction should succeed for test-sized buffers");
    assert!((bn.scale[0] - 10.0).abs() < 1e-7);
    assert!((bn.offset[0] - 0.0).abs() < 1e-7);
}

#[test]
fn test_process_scalar_identity_bn() {
    // scale=1, offset=0 -> output == input
    let bn = BatchNorm1D::from_fused(3, &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0])
        .expect("construction should succeed for test-sized buffers");

    let mut data = vec![
        1.0, 2.0, 3.0, // frame 0
        4.0, 5.0, 6.0, // frame 1
        7.0, 8.0, 9.0, // frame 2
    ];

    unsafe {
        bn.process_scalar(&mut data, 3);
    }

    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-7, "mismatch at index {i}: {a} != {b}");
    }
}

#[test]
fn test_process_scalar_simple() {
    let bn = make_bn_simple(); // scale=[0.5, 1.0], offset=[0,0]

    let mut data = vec![
        2.0, 3.0, // frame 0
        4.0, 5.0, // frame 1
        6.0, 7.0, // frame 2
    ];

    unsafe {
        bn.process_scalar(&mut data, 3);
    }

    // ch0 * 0.5: 2*0.5=1, 4*0.5=2, 6*0.5=3
    // ch1 * 1.0: 3*1.0=3, 5*1.0=5, 7*1.0=7
    let expected = [1.0, 3.0, 2.0, 5.0, 3.0, 7.0];
    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-7, "mismatch at index {i}: {a} != {b}");
    }
}

#[test]
fn test_process_scalar_with_offset() {
    // scale=[2, 0.5], offset=[1, -2]
    let bn = BatchNorm1D::from_fused(2, &[2.0, 0.5], &[1.0, -2.0])
        .expect("construction should succeed for test-sized buffers");

    let mut data = vec![
        0.0, 4.0, // frame 0
        2.0, 8.0, // frame 1
    ];

    unsafe {
        bn.process_scalar(&mut data, 2);
    }

    // ch0: 0*2+1=1, 2*2+1=5
    // ch1: 4*0.5-2=0, 8*0.5-2=2
    let expected = [1.0, 0.0, 5.0, 2.0];
    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-7, "mismatch at index {i}: {a} != {b}");
    }
}

#[test]
fn test_single_channel() {
    let bn = BatchNorm1D::from_fused(1, &[3.0], &[-1.0])
        .expect("construction should succeed for test-sized buffers");

    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    unsafe {
        bn.process_scalar(&mut data, 5);
    }

    // 1*3-1=2, 2*3-1=5, 3*3-1=8, 4*3-1=11, 5*3-1=14
    let expected = [2.0, 5.0, 8.0, 11.0, 14.0];
    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-7, "mismatch at index {i}: {a} != {b}");
    }
}

#[test]
fn test_single_frame_many_channels() {
    let bn = BatchNorm1D::from_fused(10, &[1.0; 10], &[0.0; 10])
        .expect("construction should succeed for test-sized buffers");

    let mut data: Vec<f32> = (0..10).map(|i| i as f32).collect();

    unsafe {
        bn.process_scalar(&mut data, 1);
    }

    for (i, &val) in data.iter().enumerate() {
        assert!((val - i as f32).abs() < 1e-7);
    }
}

#[test]
fn test_from_fused_roundtrip() {
    let bn1 = BatchNorm1D::from_params(
        3,
        &[2.0, 1.0, 0.5],
        &[0.0, 1.0, -1.0],
        &[0.1, 0.2, 0.3],
        &[1.0, 4.0, 0.25],
        1e-5,
    )
    .expect("construction should succeed for test-sized buffers");

    let bn2 = BatchNorm1D::from_fused(3, &bn1.scale, &bn1.offset)
        .expect("construction should succeed for test-sized buffers");

    for c in 0..3 {
        assert!((bn1.scale[c] - bn2.scale[c]).abs() < 1e-7);
        assert!((bn1.offset[c] - bn2.offset[c]).abs() < 1e-7);
    }
}

/// Validates that `from_fused` constructor preserves exact values.
#[test]
fn test_from_fused_exact() {
    let bn = BatchNorm1D::from_fused(2, &[1.0f32, 2.0], &[3.0f32, 4.0])
        .expect("construction should succeed for test-sized buffers");
    assert_eq!(bn.num_channels, 2);
    assert!((bn.scale[0] - 1.0).abs() < f32::EPSILON);
    assert!((bn.scale[1] - 2.0).abs() < f32::EPSILON);
    assert!((bn.offset[0] - 3.0).abs() < f32::EPSILON);
    assert!((bn.offset[1] - 4.0).abs() < f32::EPSILON);
}

/// Validates that `process_scalar` and `process` produce identical output.
#[test]
fn test_scalar_simd_parity() {
    let bn = make_bn_simple();

    let original = vec![
        1.0, -2.0, // frame 0
        3.0, 4.0, // frame 1
        -5.0, 6.0, // frame 2
        7.0, -8.0, // frame 3
        0.5, -0.3, // frame 4
        2.5, 1.5, // frame 5
        -1.0, 3.0, // frame 6
        4.5, -2.5, // frame 7
        9.0, 0.0, // frame 8
        -3.0, 5.0, // frame 9
    ];

    let mut data_simd = original.clone();
    unsafe {
        bn.process(&mut data_simd, 10);
    }

    let mut data_scalar = original;
    unsafe {
        bn.process_scalar(&mut data_scalar, 10);
    }

    for (i, (&a, &b)) in data_simd.iter().zip(data_scalar.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "SIMD vs scalar mismatch at index {i}: {a} != {b}"
        );
    }
}

/// Validates across a range of frame counts (including > 8 for AVX2 tail).
#[test]
fn test_scalar_simd_parity_various_frame_counts() {
    let bn = BatchNorm1D::from_params(
        4,
        &[0.8, 1.2, 0.5, 2.0],
        &[0.1, -0.1, 0.0, 0.3],
        &[0.0, 0.0, 0.0, 0.0],
        &[1.0, 1.0, 1.0, 1.0],
        1e-5,
    )
    .expect("construction should succeed for test-sized buffers");

    for n_frames in [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 64, 100] {
        let mut original = Vec::with_capacity(n_frames * 4);
        for f in 0..n_frames {
            original.push((f as f32) * 0.1);
            original.push(-(f as f32) * 0.1);
            original.push(1.0);
            original.push(-1.0);
        }

        let mut data_simd = original.clone();
        unsafe {
            bn.process(&mut data_simd, n_frames);
        }

        let mut data_scalar = original;
        unsafe {
            bn.process_scalar(&mut data_scalar, n_frames);
        }

        for (i, (&a, &b)) in data_simd.iter().zip(data_scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "SIMD vs scalar mismatch at n_frames={n_frames}, index {i}: {a} != {b}"
            );
        }
    }
}

/// Validates all public invariants when model parameters are zeroed.
#[test]
fn test_zero_gamma_noop() {
    // gamma=0 -> scale=0, offset=beta
    let bn = BatchNorm1D::from_params(2, &[0.0, 0.0], &[0.5, -0.5], &[1.0, 2.0], &[1.0, 1.0], 0.0)
        .expect("construction should succeed for test-sized buffers");
    assert!((bn.scale[0] - 0.0).abs() < 1e-7);
    assert!((bn.scale[1] - 0.0).abs() < 1e-7);
    assert!((bn.offset[0] - 0.5).abs() < 1e-7);
    assert!((bn.offset[1] - (-0.5)).abs() < 1e-7);

    let mut data = vec![100.0, 200.0, 300.0, 400.0];
    unsafe {
        bn.process_scalar(&mut data, 2);
    }
    // output should just be offset
    let expected = [0.5, -0.5, 0.5, -0.5];
    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "mismatch at index {i}");
    }
}

#[test]
#[should_panic(expected = "negative")]
fn test_negative_variance_panics() {
    BatchNorm1D::from_params(1, &[1.0], &[0.0], &[0.0], &[-1.0], 0.0)
        .expect("construction should succeed for test-sized buffers");
}

#[test]
fn test_clone_preserves_behavior() {
    let bn1 = BatchNorm1D::from_params(
        3,
        &[0.5, 1.0, 2.0],
        &[0.1, 0.2, 0.3],
        &[0.0, 1.0, 0.0],
        &[4.0, 1.0, 1.0],
        0.0,
    )
    .expect("construction should succeed for test-sized buffers");
    let bn2 = bn1.clone();

    let mut data1 = vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
    let mut data2 = data1.clone();

    unsafe {
        bn1.process_scalar(&mut data1, 2);
        bn2.process_scalar(&mut data2, 2);
    }

    assert_eq!(data1, data2);
}

/// Verifies that `BatchNorm1D` uses 64-byte alignment.
#[test]
fn test_struct_alignment() {
    assert_eq!(std::mem::align_of::<BatchNorm1D>(), 64);
}

/// Parity tests for the new frame-major SIMD batch_norm_process kernels
/// against the scalar fallback reference.
#[cfg(test)]
mod batch_norm_simd_parity {
    use crate::math::common::scalar_ref::utility::batch_norm_process_fallback;
    use crate::math::common::traits::SimdMath;

    #[test]
    fn avx2_vs_scalar_8_channels() {
        let n_ch = 8;
        let num_frames = 10;
        let scale: Vec<f32> = (0..n_ch).map(|i| 0.5 + (i as f32) * 0.1).collect();
        let offset: Vec<f32> = (0..n_ch).map(|i| -1.0 + (i as f32) * 0.2).collect();

        let original: Vec<f32> = (0..n_ch * num_frames)
            .map(|i| (i as f32) * 0.1 - 5.0)
            .collect();

        let mut simd = original.clone();
        unsafe {
            crate::math::common::Avx2Math::batch_norm_process(
                &mut simd, &scale, &offset, n_ch, num_frames,
            );
        }

        let mut scalar = original;
        unsafe {
            batch_norm_process_fallback(&mut scalar, &scale, &offset, n_ch, num_frames);
        }

        for (i, (&a, &b)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "n_ch={n_ch} num_frames={num_frames} idx={i}: simd={a} != scalar={b}"
            );
        }
    }

    #[test]
    fn avx2_vs_scalar_16_channels() {
        let n_ch = 16;
        let num_frames = 8;
        let scale: Vec<f32> = (0..n_ch).map(|i| 0.8 + (i as f32) * 0.05).collect();
        let offset: Vec<f32> = (0..n_ch).map(|i| (i as f32) * 0.15 - 0.5).collect();

        let original: Vec<f32> = (0..n_ch * num_frames)
            .map(|i| (i as f32) * 0.05 - 3.0)
            .collect();

        let mut simd = original.clone();
        unsafe {
            crate::math::common::Avx2Math::batch_norm_process(
                &mut simd, &scale, &offset, n_ch, num_frames,
            );
        }

        let mut scalar = original;
        unsafe {
            batch_norm_process_fallback(&mut scalar, &scale, &offset, n_ch, num_frames);
        }

        for (i, (&a, &b)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "n_ch={n_ch} num_frames={num_frames} idx={i}: simd={a} != scalar={b}"
            );
        }
    }

    #[test]
    fn avx2_vs_scalar_24_channels_unaligned() {
        let n_ch = 24;
        let num_frames = 12;
        let scale: Vec<f32> = (0..n_ch).map(|i| 1.0 + (i as f32) * 0.03).collect();
        let offset: Vec<f32> = (0..n_ch).map(|i| -0.2 * (i as f32)).collect();

        let original: Vec<f32> = (0..n_ch * num_frames)
            .map(|i| (i as f32) * 0.07 - 2.0)
            .collect();

        let mut simd = original.clone();
        unsafe {
            crate::math::common::Avx2Math::batch_norm_process(
                &mut simd, &scale, &offset, n_ch, num_frames,
            );
        }

        let mut scalar = original;
        unsafe {
            batch_norm_process_fallback(&mut scalar, &scale, &offset, n_ch, num_frames);
        }

        for (i, (&a, &b)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "n_ch={n_ch} num_frames={num_frames} idx={i}: simd={a} != scalar={b}"
            );
        }
    }

    #[test]
    fn avx2_vs_scalar_odd_channels() {
        let n_ch = 7;
        let num_frames = 20;
        let scale: Vec<f32> = (0..n_ch).map(|i| 0.7 + (i as f32) * 0.15).collect();
        let offset: Vec<f32> = (0..n_ch).map(|i| 0.3 - (i as f32) * 0.1).collect();

        let original: Vec<f32> = (0..n_ch * num_frames)
            .map(|i| (i as f32) * 0.12 - 1.0)
            .collect();

        let mut simd = original.clone();
        unsafe {
            crate::math::common::Avx2Math::batch_norm_process(
                &mut simd, &scale, &offset, n_ch, num_frames,
            );
        }

        let mut scalar = original;
        unsafe {
            batch_norm_process_fallback(&mut scalar, &scale, &offset, n_ch, num_frames);
        }

        for (i, (&a, &b)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "n_ch={n_ch} num_frames={num_frames} idx={i}: simd={a} != scalar={b}"
            );
        }
    }

    #[test]
    fn avx2_vs_scalar_single_channel() {
        let n_ch = 1;
        let num_frames = 50;
        let scale = vec![2.5];
        let offset = vec![-1.0];

        let original: Vec<f32> = (0..num_frames).map(|i| (i as f32) * 0.1).collect();

        let mut simd = original.clone();
        unsafe {
            crate::math::common::Avx2Math::batch_norm_process(
                &mut simd, &scale, &offset, n_ch, num_frames,
            );
        }

        let mut scalar = original;
        unsafe {
            batch_norm_process_fallback(&mut scalar, &scale, &offset, n_ch, num_frames);
        }

        for (i, (&a, &b)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "n_ch={n_ch} num_frames={num_frames} idx={i}: simd={a} != scalar={b}"
            );
        }
    }

    #[test]
    fn avx2_vs_scalar_many_frames() {
        let n_ch = 3;
        let num_frames = 200;
        let scale = vec![0.5, 1.2, 0.8];
        let offset = vec![0.1, -0.3, 0.0];

        let original: Vec<f32> = (0..n_ch * num_frames)
            .map(|i| (i as f32).sin() * 2.0)
            .collect();

        let mut simd = original.clone();
        unsafe {
            crate::math::common::Avx2Math::batch_norm_process(
                &mut simd, &scale, &offset, n_ch, num_frames,
            );
        }

        let mut scalar = original;
        unsafe {
            batch_norm_process_fallback(&mut scalar, &scale, &offset, n_ch, num_frames);
        }

        for (i, (&a, &b)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "n_ch={n_ch} num_frames={num_frames} idx={i}: simd={a} != scalar={b}"
            );
        }
    }

    #[test]
    fn avx2_vs_scalar_single_frame() {
        let n_ch = 9;
        let num_frames = 1;
        let scale: Vec<f32> = (0..n_ch).map(|i| 0.2 + (i as f32) * 0.1).collect();
        let offset: Vec<f32> = (0..n_ch).map(|_i| 0.0).collect();

        let original: Vec<f32> = (0..n_ch).map(|i| (i as f32) * 0.5).collect();

        let mut simd = original.clone();
        unsafe {
            crate::math::common::Avx2Math::batch_norm_process(
                &mut simd, &scale, &offset, n_ch, num_frames,
            );
        }

        let mut scalar = original;
        unsafe {
            batch_norm_process_fallback(&mut scalar, &scale, &offset, n_ch, num_frames);
        }

        for (i, (&a, &b)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "n_ch={n_ch} num_frames={num_frames} idx={i}: simd={a} != scalar={b}"
            );
        }
    }

    /// Tests AVX-512 parity when AVX-512F is supported at runtime.
    #[cfg(feature = "avx512")]
    #[test]
    fn avx512_vs_scalar() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }

        let n_ch = 16;
        let num_frames = 8;
        let scale: Vec<f32> = (0..n_ch).map(|i| 0.8 + (i as f32) * 0.05).collect();
        let offset: Vec<f32> = (0..n_ch).map(|i| (i as f32) * 0.15 - 0.5).collect();

        let original: Vec<f32> = (0..n_ch * num_frames)
            .map(|i| (i as f32) * 0.05 - 3.0)
            .collect();

        let mut simd = original.clone();
        unsafe {
            crate::math::common::Avx512Math::batch_norm_process(
                &mut simd, &scale, &offset, n_ch, num_frames,
            );
        }

        let mut scalar = original;
        unsafe {
            batch_norm_process_fallback(&mut scalar, &scale, &offset, n_ch, num_frames);
        }

        for (i, (&a, &b)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "AVX-512 n_ch={n_ch} num_frames={num_frames} idx={i}: simd={a} != scalar={b}"
            );
        }
    }

    /// Tests AVX-512 parity with 32 channels and unaligned channel count (35).
    #[cfg(feature = "avx512")]
    #[test]
    fn avx512_vs_scalar_32ch_and_35ch() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }

        for n_ch in [32, 35] {
            let num_frames = 6;
            let scale: Vec<f32> = (0..n_ch).map(|i| 0.5 + (i as f32) * 0.04).collect();
            let offset: Vec<f32> = (0..n_ch).map(|i| -0.1 * (i as f32)).collect();

            let original: Vec<f32> = (0..n_ch * num_frames)
                .map(|i| (i as f32) * 0.03 - 1.0)
                .collect();

            let mut simd = original.clone();
            unsafe {
                crate::math::common::Avx512Math::batch_norm_process(
                    &mut simd, &scale, &offset, n_ch, num_frames,
                );
            }

            let mut scalar = original;
            unsafe {
                batch_norm_process_fallback(&mut scalar, &scale, &offset, n_ch, num_frames);
            }

            for (i, (&a, &b)) in simd.iter().zip(scalar.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "AVX-512 n_ch={n_ch} num_frames={num_frames} idx={i}: simd={a} != scalar={b}"
                );
            }
        }
    }
}
