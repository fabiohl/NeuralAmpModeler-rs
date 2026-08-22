// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::ops::set_daz_ftz;
use super::traits::SimdMath;
use crate::math::common::Avx2Math;
#[cfg(feature = "avx512")]
use crate::math::common::Avx512Math;
use crate::math::dsp::stereo::{compute_energy_avx2, compute_max_diff_avx2};
use crate::math::gemm::dot_basic::dot_product_avx2;
#[cfg(feature = "avx512")]
use crate::math::gemm::dot_basic::dot_product_avx512;

#[test]
fn test_dot_product_avx2_fma() {
    // Tests the 'Dot Product' operation (multiply and sum everything).
    // This is the most frequent mathematical calculation inside neural networks.
    let vec_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let vec_b = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let result = unsafe { dot_product_avx2(&vec_a, &vec_b) };

    // Expected = (1*2 + 2*2 ... + 8*2) + 9*2
    // 72 * 2 + 18 = 144 + 18 = 90
    let expected: f32 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();

    // FMA math is accurate but compare with epsilon
    assert!(
        (result - expected).abs() < 1e-4,
        "Divergent result: expected {}, got {}",
        expected,
        result
    );
}

#[cfg(feature = "avx512")]
#[test]
fn test_dot_product_avx512() {
    // Dot product test version for ultra-modern processors (AVX-512).
    if crate::math::common::SimdMathConfig::get().instruction_set
        >= crate::math::common::InstructionSet::Avx512
    {
        let vec_a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ];
        let vec_b = vec![
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ];

        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let result = unsafe { dot_product_avx512(&vec_a, &vec_b) };
        let expected: f32 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (result - expected).abs() < 1e-4,
            "Divergent result: expected {}, got {}",
            expected,
            result
        );
    }
}

/// Verifies that `set_daz_ftz` correctly sets the DAZ (6) and FTZ (15) bits in MXCSR.
#[test]
fn test_set_daz_ftz() {
    // Checks if the processor is configured to ignore 'ghostly'
    // tiny numbers (denormals). This avoids extreme slowdown in audio processing.
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    unsafe {
        // Read current MXCSR and clear DAZ+FTZ to verify the function sets them
        let mut before: u32 = 0;
        core::arch::asm!("stmxcsr [{0}]", in(reg) &mut before);
        let cleared = before & !0x8040;
        core::arch::asm!("ldmxcsr [{0}]", in(reg) &cleared);

        set_daz_ftz();

        let mut after: u32 = 0;
        core::arch::asm!("stmxcsr [{0}]", in(reg) &mut after);
        assert!(
            (after & 0x8040) == 0x8040,
            "set_daz_ftz() did not set DAZ+FTZ: MXCSR=0x{:08X}",
            after
        );

        // Restore original MXCSR
        core::arch::asm!("ldmxcsr [{0}]", in(reg) &before);
    }
}

#[test]
fn test_compute_energy_avx2() {
    // Tests energy calculation (average volume) using AVX2 acceleration.
    let data = vec![1.0, 2.0, 3.0, 4.0];
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let energy = unsafe { compute_energy_avx2(&data) };
    // (1^2 + 2^2 + 3^2 + 4^2) / 4 = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    assert!((energy - 7.5).abs() < 1e-6);

    let data2 = vec![0.0; 16];
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let energy2 = unsafe { compute_energy_avx2(&data2) };
    assert_eq!(energy2, 0.0);
}

#[test]
fn test_compute_max_diff_avx2() {
    // Tests the calculation of the largest difference between two sounds using AVX2.
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.1, 1.9, 3.5, 3.8];
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let max_diff = unsafe { compute_max_diff_avx2(&a, &b) };
    // diffs: [0.1, 0.1, 0.5, 0.2] -> max = 0.5
    assert!((max_diff - 0.5).abs() < 1e-6);

    let a2 = vec![1.0; 8];
    let b2 = vec![1.0; 8];
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let max_diff2 = unsafe { compute_max_diff_avx2(&a2, &b2) };
    assert_eq!(max_diff2, 0.0);
}

#[test]
fn test_horizontal_sum() {
    // Tests the sum of all numbers inside a SIMD 'pack' (8 or 16 numbers).
    fn test_n<const N: usize>(data_ptr: *const f32, expected: f32) {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let res_avx2 = unsafe { Avx2Math::horizontal_sum::<N>(data_ptr) };
        assert!(
            (res_avx2 - expected).abs() < 1e-5,
            "AVX2 N={} failed: got {}, expected {}",
            N,
            res_avx2,
            expected
        );

        #[cfg(feature = "avx512")]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            let res_avx512 = unsafe { Avx512Math::horizontal_sum::<N>(data_ptr) };
            assert!(
                (res_avx512 - expected).abs() < 1e-5,
                "AVX512 N={} failed: got {}, expected {}",
                N,
                res_avx512,
                expected
            );
        }
    }

    let data: Vec<f32> = (0..64).map(|i| i as f32 + 1.0).collect();

    test_n::<1>(data.as_ptr(), data[..1].iter().sum());
    test_n::<4>(data.as_ptr(), data[..4].iter().sum());
    test_n::<6>(data.as_ptr(), data[..6].iter().sum());
    test_n::<8>(data.as_ptr(), data[..8].iter().sum());
    test_n::<12>(data.as_ptr(), data[..12].iter().sum());
    test_n::<16>(data.as_ptr(), data[..16].iter().sum());
    test_n::<32>(data.as_ptr(), data[..32].iter().sum());
}

#[test]
fn test_accumulate_head() {
    // Tests the accumulated sum of two sound blocks (used to combine layer results).
    fn test_backend<M: SimdMath>() {
        let mut dest = vec![1.0; 32];
        let src = vec![2.0; 32];
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe { M::accumulate_head(&mut dest, &src) };
        for val in dest {
            assert!((val - 3.0).abs() < 1e-6);
        }

        // Test with odd length
        let mut dest2 = vec![1.0; 7];
        let src2 = vec![3.0; 7];
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe { M::accumulate_head(&mut dest2, &src2) };
        for val in dest2 {
            assert!((val - 4.0).abs() < 1e-6);
        }
    }

    // test_backend::<ScalarMath>(); // ScalarMath removed (Project targets x86-64-v3+)
    test_backend::<Avx2Math>();
    #[cfg(feature = "avx512")]
    if std::is_x86_feature_detected!("avx512f") {
        test_backend::<Avx512Math>();
    }
}

#[test]
fn test_tanh_and_accumulate_with_seed() {
    fn test_backend<M: SimdMath>() {
        for len in [1, 4, 8, 15, 16, 17, 31, 32, 33, 64] {
            let seed: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
            let mut head_input = vec![0.0f32; len];
            let mut block = vec![0.5f32; len];
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { M::tanh_and_accumulate_with_seed(&mut head_input, &mut block, &seed) };
            let expected_tanh = 0.5f32.tanh();
            for i in 0..len {
                assert!(
                    (block[i] - expected_tanh).abs() < 1e-6,
                    "Length {} block[{}] failed: got {}, expected {}",
                    len,
                    i,
                    block[i],
                    expected_tanh
                );
                let expected_head = seed[i] + expected_tanh;
                assert!(
                    (head_input[i] - expected_head).abs() < 1e-6,
                    "Length {} head_input[{}] failed: got {}, expected {}",
                    len,
                    i,
                    head_input[i],
                    expected_head
                );
            }
        }
    }

    test_backend::<Avx2Math>();
    #[cfg(feature = "avx512")]
    if std::is_x86_feature_detected!("avx512f") {
        test_backend::<Avx512Math>();
    }
}

#[test]
fn test_store_bf16_avx2() {
    let vals = [
        1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32,
    ];
    let mut dest = [0u16; 8];
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    unsafe {
        let v = core::arch::x86_64::_mm256_loadu_ps(vals.as_ptr());
        Avx2Math::store_bf16(dest.as_mut_ptr(), v);
    }
    for i in 0..8 {
        assert_eq!(dest[i], (vals[i].to_bits() >> 16) as u16);
    }
}

/// Ensures that saving compact data (bfloat16) using the full AVX-512
/// width (512 bits) is done without data loss or corruption.
#[cfg(feature = "avx512")]
#[test]
fn test_store_bf16_avx512() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let vals: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
    let mut dest = [0u16; 16];
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    unsafe {
        let v = core::arch::x86_64::_mm512_loadu_ps(vals.as_ptr());
        Avx512Math::store_bf16(dest.as_mut_ptr(), v);
    }
    for i in 0..16 {
        assert_eq!(dest[i], (vals[i].to_bits() >> 16) as u16);
    }
}

#[test]
fn test_compute_energy_parity() {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let expected = unsafe { crate::math::common::compute_energy_fallback(&data) };

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let res_avx2 = unsafe { Avx2Math::compute_energy(&data) };
    assert!((res_avx2 - expected).abs() < 1e-6);

    #[cfg(feature = "avx512")]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let res_avx512 = unsafe { Avx512Math::compute_energy(&data) };
        assert!((res_avx512 - expected).abs() < 1e-6);
    }
}

#[test]
fn test_compute_energy_stereo_parity() {
    let l: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let r: Vec<f32> = (0..100).map(|i| (100 - i) as f32 * 0.01).collect();
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let expected = unsafe { crate::math::common::compute_energy_stereo_fallback(&l, &r) };

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let res_avx2 = unsafe { Avx2Math::compute_energy_stereo(&l, &r) };
    assert!((res_avx2 - expected).abs() < 1e-6);

    #[cfg(feature = "avx512")]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let res_avx512 = unsafe { Avx512Math::compute_energy_stereo(&l, &r) };
        assert!((res_avx512 - expected).abs() < 1e-6);
    }
}

#[test]
fn test_compute_max_diff_parity() {
    let a: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..100).map(|i| (i as f32 * 1.1) * 0.01).collect();
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let expected = unsafe { crate::math::common::compute_max_diff_fallback(&a, &b) };

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let res_avx2 = unsafe { Avx2Math::compute_max_diff(&a, &b) };
    assert!((res_avx2 - expected).abs() < 1e-6);

    #[cfg(feature = "avx512")]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let res_avx512 = unsafe { Avx512Math::compute_max_diff(&a, &b) };
        assert!((res_avx512 - expected).abs() < 1e-6);
    }
}

#[test]
fn test_compute_peak_abs_stereo_parity() {
    let l: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin() * 2.0).collect();
    let r: Vec<f32> = (0..100)
        .map(|i| ((100 - i) as f32 * 0.01).cos() * -1.5)
        .collect();
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let expected = unsafe { crate::math::common::compute_peak_abs_stereo_fallback(&l, &r) };

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let res_avx2 = unsafe { Avx2Math::compute_peak_abs_stereo(&l, &r) };
    assert!((res_avx2.0 - expected.0).abs() < 1e-6);
    assert!((res_avx2.1 - expected.1).abs() < 1e-6);

    #[cfg(feature = "avx512")]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let res_avx512 = unsafe { Avx512Math::compute_peak_abs_stereo(&l, &r) };
        assert!((res_avx512.0 - expected.0).abs() < 1e-6);
        assert!((res_avx512.1 - expected.1).abs() < 1e-6);
    }
}

#[test]
fn test_compute_peak_abs_mono_parity() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin() * 2.0).collect();
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let expected = unsafe { crate::math::common::compute_peak_abs_mono_fallback(&data) };

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let res_avx2 = unsafe { Avx2Math::compute_peak_abs_mono(&data) };
    assert!((res_avx2 - expected).abs() < 1e-6);

    #[cfg(feature = "avx512")]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let res_avx512 = unsafe { Avx512Math::compute_peak_abs_mono(&data) };
        assert!((res_avx512 - expected).abs() < 1e-6);
    }
}

#[test]
fn test_convolve_stereo_parity() {
    let coeffs =
        crate::math::common::AlignedVec::from_vec((0..32).map(|i| i as f32 * 0.01).collect())
            .expect("allocation should succeed for test-sized buffers");
    let input_l = crate::math::common::AlignedVec::from_vec(
        (0..32).map(|i| (32 - i) as f32 * 0.05).collect(),
    )
    .expect("allocation should succeed for test-sized buffers");
    let input_r =
        crate::math::common::AlignedVec::from_vec((0..32).map(|i| i as f32 * 0.03).collect())
            .expect("allocation should succeed for test-sized buffers");

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let expected = unsafe {
        crate::math::common::convolve_stereo_fallback(
            coeffs.as_ptr(),
            input_l.as_ptr(),
            input_r.as_ptr(),
            32,
        )
    };

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let res_avx2 = unsafe {
        Avx2Math::convolve_stereo(coeffs.as_ptr(), input_l.as_ptr(), input_r.as_ptr(), 32)
    };
    assert!((res_avx2.0 - expected.0).abs() < 1e-6);
    assert!((res_avx2.1 - expected.1).abs() < 1e-6);

    #[cfg(feature = "avx512")]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let res_avx512 = unsafe {
            Avx512Math::convolve_stereo(coeffs.as_ptr(), input_l.as_ptr(), input_r.as_ptr(), 32)
        };
        assert!((res_avx512.0 - expected.0).abs() < 1e-6);
        assert!((res_avx512.1 - expected.1).abs() < 1e-6);
    }
}

#[test]
fn test_convolve_mono_parity() {
    let coeffs =
        crate::math::common::AlignedVec::from_vec((0..32).map(|i| i as f32 * 0.01).collect())
            .expect("allocation should succeed for test-sized buffers");
    let input = crate::math::common::AlignedVec::from_vec(
        (0..32).map(|i| (32 - i) as f32 * 0.05).collect(),
    )
    .expect("allocation should succeed for test-sized buffers");

    let expected =
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe { crate::math::common::convolve_mono_fallback(coeffs.as_ptr(), input.as_ptr(), 32) };

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let res_avx2 = unsafe { Avx2Math::convolve_mono(coeffs.as_ptr(), input.as_ptr(), 32) };
    assert!((res_avx2 - expected).abs() < 1e-6);

    #[cfg(feature = "avx512")]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let res_avx512 = unsafe { Avx512Math::convolve_mono(coeffs.as_ptr(), input.as_ptr(), 32) };
        assert!((res_avx512 - expected).abs() < 1e-6);
    }
}

#[test]
fn test_convolve_stereo_dual_parity() {
    let coeffs0 =
        crate::math::common::AlignedVec::from_vec((0..32).map(|i| i as f32 * 0.01).collect())
            .expect("allocation should succeed for test-sized buffers");
    let coeffs1 = crate::math::common::AlignedVec::from_vec(
        (0..32).map(|i| (32 - i) as f32 * 0.007).collect(),
    )
    .expect("allocation should succeed for test-sized buffers");
    let input_l =
        crate::math::common::AlignedVec::from_vec((0..32).map(|i| i as f32 * 0.03).collect())
            .expect("allocation should succeed for test-sized buffers");
    let input_r = crate::math::common::AlignedVec::from_vec(
        (0..32).map(|i| (32 - i) as f32 * 0.04).collect(),
    )
    .expect("allocation should succeed for test-sized buffers");

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let expected = unsafe {
        crate::math::common::convolve_stereo_dual_fallback(
            coeffs0.as_ptr(),
            coeffs1.as_ptr(),
            input_l.as_ptr(),
            input_r.as_ptr(),
            32,
        )
    };

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let res_avx2 = unsafe {
        Avx2Math::convolve_stereo_dual(
            coeffs0.as_ptr(),
            coeffs1.as_ptr(),
            input_l.as_ptr(),
            input_r.as_ptr(),
            32,
        )
    };
    assert!((res_avx2.0.0 - expected.0.0).abs() < 1e-5);
    assert!((res_avx2.0.1 - expected.0.1).abs() < 1e-5);
    assert!((res_avx2.1.0 - expected.1.0).abs() < 1e-5);
    assert!((res_avx2.1.1 - expected.1.1).abs() < 1e-5);

    #[cfg(feature = "avx512")]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let res_avx512 = unsafe {
            Avx512Math::convolve_stereo_dual(
                coeffs0.as_ptr(),
                coeffs1.as_ptr(),
                input_l.as_ptr(),
                input_r.as_ptr(),
                32,
            )
        };
        assert!((res_avx512.0.0 - expected.0.0).abs() < 1e-5);
        assert!((res_avx512.0.1 - expected.0.1).abs() < 1e-5);
        assert!((res_avx512.1.0 - expected.1.0).abs() < 1e-5);
        assert!((res_avx512.1.1 - expected.1.1).abs() < 1e-5);
    }
}

#[test]
fn test_convolve_mono_dual_parity() {
    let coeffs0 =
        crate::math::common::AlignedVec::from_vec((0..32).map(|i| i as f32 * 0.01).collect())
            .expect("allocation should succeed for test-sized buffers");
    let coeffs1 = crate::math::common::AlignedVec::from_vec(
        (0..32).map(|i| (32 - i) as f32 * 0.007).collect(),
    )
    .expect("allocation should succeed for test-sized buffers");
    let input =
        crate::math::common::AlignedVec::from_vec((0..32).map(|i| i as f32 * 0.03).collect())
            .expect("allocation should succeed for test-sized buffers");

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let expected = unsafe {
        crate::math::common::convolve_mono_dual_fallback(
            coeffs0.as_ptr(),
            coeffs1.as_ptr(),
            input.as_ptr(),
            32,
        )
    };

    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let res_avx2 = unsafe {
        Avx2Math::convolve_mono_dual(coeffs0.as_ptr(), coeffs1.as_ptr(), input.as_ptr(), 32)
    };
    assert!((res_avx2.0 - expected.0).abs() < 1e-5);
    assert!((res_avx2.1 - expected.1).abs() < 1e-5);

    #[cfg(feature = "avx512")]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let res_avx512 = unsafe {
            Avx512Math::convolve_mono_dual(coeffs0.as_ptr(), coeffs1.as_ptr(), input.as_ptr(), 32)
        };
        assert!((res_avx512.0 - expected.0).abs() < 1e-5);
        assert!((res_avx512.1 - expected.1).abs() < 1e-5);
    }
}

// ── Regression Tests ────────────────────────────────────────────────────────

/// Converts an f32 slice to BF16 using the correct method (shift right by 16).
#[cfg(feature = "avx512")]
fn f32_to_bf16_ref(src: &[f32]) -> Vec<u16> {
    src.iter().map(|s| (s.to_bits() >> 16) as u16).collect()
}

/// Generates pseudo-random BF16 data from deterministic seeds.
#[cfg(feature = "avx512")]
fn gen_bf16_data(len: usize, seed: f32) -> Vec<u16> {
    (0..len)
        .map(|i| {
            let v = (i as f32 * 1.7 + seed).sin() * 0.8 + (i as f32 * 0.3).cos();
            (v.to_bits() >> 16) as u16
        })
        .collect()
}

/// Verifies that the F32→BF16 conversion via AVX-512 produces the same bits
/// as the scalar reference, including in the remainder (< 16 elements).
#[cfg(feature = "avx512")]
#[test]
fn test_f32_to_bf16_avx512_regression() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    for len in [
        0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 47, 48, 64, 100, 127, 128,
    ] {
        let src: Vec<f32> = (0..len).map(|i| (i as f32 * 0.77).sin() * 2.5).collect();
        let mut dest = vec![0u16; len];
        let expected = f32_to_bf16_ref(&src);

        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe {
            super::ops::f32_to_bf16_avx512(&src, &mut dest);
        }

        assert_eq!(
            dest,
            expected,
            "F32→BF16 diverged at len={}: simd={:?}, ref={:?}",
            len,
            &dest[..],
            &expected[..]
        );
    }
}

/// Validates that dot_product_bf16_avx512 matches the scalar fallback (already correct).
#[cfg(feature = "avx512")]
#[test]
fn test_dot_product_bf16_avx512_regression() {
    if !is_x86_feature_detected!("avx512bf16") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128, 255, 256, 511, 512,
    ];
    for &len in &sizes {
        let a = gen_bf16_data(len, 0.5);
        let b = gen_bf16_data(len, -1.3);
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let expected = unsafe { crate::math::common::dot_product_bf16_fallback(&a, &b) };
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        let result = unsafe { crate::math::gemm::dot_basic::dot_product_bf16_avx512(&a, &b) };

        let error = (result - expected).abs();
        assert!(
            error < 1e-5,
            "BF16 dot product diverged at len={}: simd={}, ref={}, err={}",
            len,
            result,
            expected,
            error
        );
    }
}

/// Validates that gemv_overwrite_bf16_avx512 matches the scalar fallback.
#[cfg(feature = "avx512")]
#[test]
fn test_gemv_overwrite_bf16_avx512_regression() {
    if !is_x86_feature_detected!("avx512bf16") {
        return;
    }

    let configs = [
        (1, 1),
        (2, 1),
        (4, 4),
        (8, 8),
        (16, 16),
        (17, 16),
        (32, 32),
        (33, 16),
        (47, 32),
        (64, 64),
        (128, 128),
    ];

    for &(in_len, out_len) in &configs {
        let in_frame = gen_bf16_data(in_len, 0.3);
        let weights: Vec<u16> = (0..(in_len * out_len))
            .map(|j| {
                let v = (j as f32 * 0.23).sin() * 0.6;
                (v.to_bits() >> 16) as u16
            })
            .collect();
        let bias: Vec<f32> = (0..out_len).map(|j| (j as f32 * 0.1).cos()).collect();

        let mut out_simd = vec![0.0f32; out_len];
        let mut out_ref = vec![0.0f32; out_len];

        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe {
            crate::math::gemm::gemv_bf16::gemv_overwrite_bf16_avx512(
                &in_frame,
                &weights,
                &bias,
                &mut out_simd,
                true,
            );
            crate::math::common::gemv_overwrite_bf16_fallback(
                &in_frame,
                &weights,
                &bias,
                &mut out_ref,
                true,
            );
        }

        for j in 0..out_len {
            let error = (out_simd[j] - out_ref[j]).abs();
            assert!(
                error < 1e-5,
                "BF16 GEMV diverged: in={} out={} ch={}: simd={}, ref={}, err={}",
                in_len,
                out_len,
                j,
                out_simd[j],
                out_ref[j],
                error
            );
        }
    }
}

#[test]
fn test_tanh_and_overwrite_block() {
    fn test_backend<M: SimdMath>() {
        let mut head_input = vec![-999.0; 64];
        let mut block = vec![0.5f32; 64];
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe { M::tanh_and_overwrite_block(&mut head_input, &mut block) };
        for i in 0..64 {
            let expected = 0.5f32.tanh();
            assert!((head_input[i] - expected).abs() < 1e-6);
            assert!((block[i] - expected).abs() < 1e-6);
        }
    }

    test_backend::<Avx2Math>();
    #[cfg(feature = "avx512")]
    if std::is_x86_feature_detected!("avx512f") {
        test_backend::<Avx512Math>();
    }
}

#[test]
fn test_tanh_and_accumulate_block() {
    fn test_backend<M: SimdMath>() {
        for len in [1, 4, 8, 15, 16, 17, 31, 32, 33, 64] {
            let mut head_input = vec![1.0f32; len];
            let mut block = vec![0.5f32; len];
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { M::tanh_and_accumulate_block(&mut head_input, &mut block) };
            let expected_tanh = 0.5f32.tanh();
            for i in 0..len {
                assert!(
                    (block[i] - expected_tanh).abs() < 1e-6,
                    "Length {} block[{}] failed: got {}, expected {}",
                    len,
                    i,
                    block[i],
                    expected_tanh
                );
                assert!(
                    (head_input[i] - (1.0 + expected_tanh)).abs() < 1e-6,
                    "Length {} head_input[{}] failed: got {}, expected {}",
                    len,
                    i,
                    head_input[i],
                    1.0 + expected_tanh
                );
            }
        }
    }

    test_backend::<Avx2Math>();
    #[cfg(feature = "avx512")]
    if std::is_x86_feature_detected!("avx512f") {
        test_backend::<Avx512Math>();
    }
}

#[test]
fn test_gated_activation_and_overwrite_block() {
    fn test_backend<M: SimdMath>() {
        let ch = 8;
        let num_frames = 4;
        let mut head_input = vec![-999.0; num_frames * ch];
        let mut block = vec![0.5f32; num_frames * 2 * ch];
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe { M::gated_activation_and_overwrite_block(&mut head_input, &mut block, ch) };
        for f in 0..num_frames {
            for c in 0..ch {
                let z1 = 0.5f32;
                let z2 = 0.5f32;
                let expected = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
                let head_idx = f * ch + c;
                assert!((head_input[head_idx] - expected).abs() < 1e-3);
            }
        }
    }

    test_backend::<Avx2Math>();
    #[cfg(feature = "avx512")]
    if std::is_x86_feature_detected!("avx512f") {
        test_backend::<Avx512Math>();
    }
}

#[test]
fn test_complex_mac_overwrite_parity() {
    let sizes = [0, 1, 3, 8, 15, 16, 17, 31, 32, 33, 64, 128, 255, 256];
    for &len in &sizes {
        let h_re: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).sin()).collect();
        let h_im: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).cos()).collect();
        let x_re: Vec<f32> = (0..len).map(|i| (i as f32 * 1.3).sin()).collect();
        let x_im: Vec<f32> = (0..len).map(|i| (i as f32 * 0.9).cos()).collect();
        let mut scalar_re = vec![0.0f32; len];
        let mut scalar_im = vec![0.0f32; len];
        // SAFETY: slices are valid, no aliasing.
        crate::math::common::complex_mac_overwrite_scalar(
            &h_re,
            &h_im,
            &x_re,
            &x_im,
            &mut scalar_re,
            &mut scalar_im,
        );

        #[expect(
            clippy::too_many_arguments,
            reason = "Math unit test helper with many configuration parameters for comprehensive SIMD kernel testing"
        )]
        fn test_simd<M: SimdMath>(
            h_re: &[f32],
            h_im: &[f32],
            x_re: &[f32],
            x_im: &[f32],
            expected_re: &[f32],
            expected_im: &[f32],
            len: usize,
            label: &str,
        ) {
            let mut simd_re = vec![9.9f32; len];
            let mut simd_im = vec![9.9f32; len];
            // SAFETY: slices are valid, and SimdMath preconditions hold.
            unsafe { M::complex_mac_overwrite(h_re, h_im, x_re, x_im, &mut simd_re, &mut simd_im) };
            for i in 0..len {
                assert!(
                    (simd_re[i] - expected_re[i]).abs() < 1e-6,
                    "{} overwrite re[{}] len={}: simd={}, ref={}",
                    label,
                    i,
                    len,
                    simd_re[i],
                    expected_re[i]
                );
                assert!(
                    (simd_im[i] - expected_im[i]).abs() < 1e-6,
                    "{} overwrite im[{}] len={}: simd={}, ref={}",
                    label,
                    i,
                    len,
                    simd_im[i],
                    expected_im[i]
                );
            }
        }

        test_simd::<Avx2Math>(
            &h_re, &h_im, &x_re, &x_im, &scalar_re, &scalar_im, len, "AVX2",
        );
        #[cfg(feature = "avx512")]
        if is_x86_feature_detected!("avx512f") {
            test_simd::<Avx512Math>(
                &h_re, &h_im, &x_re, &x_im, &scalar_re, &scalar_im, len, "AVX-512",
            );
        }
    }
}

#[test]
fn test_complex_mac_accumulate_parity() {
    let sizes = [0, 1, 3, 8, 15, 16, 17, 31, 32, 33, 64, 128, 255, 256];
    for &len in &sizes {
        let h_re: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).sin()).collect();
        let h_im: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).cos()).collect();
        let x_re: Vec<f32> = (0..len).map(|i| (i as f32 * 1.3).sin()).collect();
        let x_im: Vec<f32> = (0..len).map(|i| (i as f32 * 0.9).cos()).collect();
        let init_re: Vec<f32> = (0..len).map(|i| (i as f32 * 0.03).sin()).collect();
        let init_im: Vec<f32> = (0..len).map(|i| (i as f32 * 0.05).cos()).collect();
        let mut scalar_re = init_re.clone();
        let mut scalar_im = init_im.clone();
        // SAFETY: slices are valid — pure scalar fallback computation.
        crate::math::common::complex_mac_accumulate_scalar(
            &h_re,
            &h_im,
            &x_re,
            &x_im,
            &mut scalar_re,
            &mut scalar_im,
        );

        #[expect(
            clippy::too_many_arguments,
            reason = "Math unit test helper with many configuration parameters for comprehensive SIMD kernel testing"
        )]
        fn test_simd<M: SimdMath>(
            h_re: &[f32],
            h_im: &[f32],
            x_re: &[f32],
            x_im: &[f32],
            init_re: &[f32],
            init_im: &[f32],
            expected_re: &[f32],
            expected_im: &[f32],
            len: usize,
            label: &str,
        ) {
            let mut simd_re = init_re.to_vec();
            let mut simd_im = init_im.to_vec();
            // SAFETY: slices are valid; SimdMath preconditions hold.
            unsafe {
                M::complex_mac_accumulate(h_re, h_im, x_re, x_im, &mut simd_re, &mut simd_im)
            };
            for i in 0..len {
                assert!(
                    (simd_re[i] - expected_re[i]).abs() < 1e-6,
                    "{} accumulate re[{}] len={}: simd={}, ref={}",
                    label,
                    i,
                    len,
                    simd_re[i],
                    expected_re[i]
                );
                assert!(
                    (simd_im[i] - expected_im[i]).abs() < 1e-6,
                    "{} accumulate im[{}] len={}: simd={}, ref={}",
                    label,
                    i,
                    len,
                    simd_im[i],
                    expected_im[i]
                );
            }
        }

        test_simd::<Avx2Math>(
            &h_re, &h_im, &x_re, &x_im, &init_re, &init_im, &scalar_re, &scalar_im, len, "AVX2",
        );
        #[cfg(feature = "avx512")]
        if is_x86_feature_detected!("avx512f") {
            test_simd::<Avx512Math>(
                &h_re, &h_im, &x_re, &x_im, &init_re, &init_im, &scalar_re, &scalar_im, len,
                "AVX-512",
            );
        }
    }
}

mod huge_alloc_tests {
    use crate::math::common::huge_alloc::{
        HugePageStatus, HugePageVec, allocate_huge_pages, deallocate_huge,
    };

    #[test]
    fn test_allocate_small_uses_heap() {
        let result = allocate_huge_pages(1024);
        assert!(result.is_ok());
        let (ptr, info, status) = result.unwrap();
        assert_eq!(status, HugePageStatus::Heap);
        assert!(!ptr.is_null());
        // SAFETY: ptr is validly allocated via allocate_huge_pages above, and layout matches.
        unsafe { deallocate_huge(ptr, info, 1024) };
    }

    #[test]
    fn test_allocate_large_falls_back_gracefully() {
        let size = 2 * 1024 * 1024;
        let result = allocate_huge_pages(size);
        assert!(result.is_ok());
        let (ptr, info, _status) = result.unwrap();
        assert!(!ptr.is_null());
        // SAFETY: ptr is validly allocated via allocate_huge_pages above, and layout matches size.
        unsafe {
            std::ptr::write(ptr, 0x42u8);
            assert_eq!(std::ptr::read(ptr), 0x42u8);
            deallocate_huge(ptr, info, size);
        }
    }

    #[test]
    fn test_huge_page_vec_fallback() {
        let len = 3 * 1024 * 1024 / 4;
        let result = HugePageVec::<f32>::new(len, 0.0);
        assert!(result.is_ok());
        let (vec, status) = result.unwrap();
        assert_eq!(vec.len(), len);
        for &val in vec.iter() {
            assert_eq!(val, 0.0);
        }
        let _ = status;
        drop(vec);
    }

    #[test]
    fn test_huge_page_vec_with_capacity() {
        let result = HugePageVec::<f32>::with_capacity(128);
        assert!(result.is_ok());
        let (vec, _status) = result.unwrap();
        assert_eq!(vec.len(), 0);
    }
}
