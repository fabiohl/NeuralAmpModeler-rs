// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for WaveNet accumulation and activation kernels (scalar, AVX2, AVX-512, and AVX-512 VL256).

use super::avx2::*;
use super::avx512::*;
use super::avx512vl::*;
use super::scalar::*;

#[test]
fn test_accumulate_head_avx512vl_parity() {
    let sizes = [
        0, 1, 2, 3, 4, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 48, 64, 128, 255, 256,
    ];
    for &len in &sizes {
        let src: Vec<f32> = (0..len).map(|i| (i as f32 * 0.13).sin()).collect();
        let init_dest: Vec<f32> = (0..len).map(|i| (i as f32 * 0.29).cos()).collect();

        let mut dest_scalar = init_dest.clone();
        unsafe { accumulate_head_fallback(&mut dest_scalar, &src) };

        let mut dest_avx2 = init_dest.clone();
        unsafe { accumulate_head_avx2(&mut dest_avx2, &src) };

        for i in 0..len {
            assert!(
                (dest_avx2[i] - dest_scalar[i]).abs() < 1e-6,
                "AVX2 mismatch at len={}, i={}: got {}, expected {}",
                len,
                i,
                dest_avx2[i],
                dest_scalar[i]
            );
        }

        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            let mut dest_vl = init_dest.clone();
            unsafe { accumulate_head_avx512vl(&mut dest_vl, &src) };

            for i in 0..len {
                assert!(
                    (dest_vl[i] - dest_scalar[i]).abs() < 1e-6,
                    "AVX512VL mismatch at len={}, i={}: got {}, expected {}",
                    len,
                    i,
                    dest_vl[i],
                    dest_scalar[i]
                );
            }

            let mut dest_512 = init_dest.clone();
            unsafe { accumulate_head_avx512(&mut dest_512, &src) };

            for i in 0..len {
                assert!(
                    (dest_512[i] - dest_scalar[i]).abs() < 1e-6,
                    "AVX512 mismatch at len={}, i={}: got {}, expected {}",
                    len,
                    i,
                    dest_512[i],
                    dest_scalar[i]
                );
            }
        }
    }
}

#[test]
fn test_tanh_and_accumulate_block_avx512vl_parity() {
    let sizes = [
        0, 1, 2, 3, 4, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 48, 64, 128, 255, 256,
    ];
    for &len in &sizes {
        let init_block: Vec<f32> = (0..len).map(|i| (i as f32 * 0.17).sin() * 3.0).collect();
        let init_head: Vec<f32> = (0..len).map(|i| (i as f32 * 0.31).cos()).collect();

        let mut head_scalar = init_head.clone();
        let mut block_scalar = init_block.clone();
        unsafe { tanh_and_accumulate_block_fallback(&mut head_scalar, &mut block_scalar) };

        let mut head_avx2 = init_head.clone();
        let mut block_avx2 = init_block.clone();
        unsafe { tanh_and_accumulate_block_avx2(&mut head_avx2, &mut block_avx2) };

        for i in 0..len {
            assert!(
                (block_avx2[i] - block_scalar[i]).abs() < 1e-4,
                "AVX2 block mismatch at len={}, i={}",
                len,
                i
            );
            assert!(
                (head_avx2[i] - head_scalar[i]).abs() < 1e-4,
                "AVX2 head mismatch at len={}, i={}",
                len,
                i
            );
        }

        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            let mut head_vl = init_head.clone();
            let mut block_vl = init_block.clone();
            unsafe { tanh_and_accumulate_block_avx512vl(&mut head_vl, &mut block_vl) };

            for i in 0..len {
                assert!(
                    (block_vl[i] - block_avx2[i]).abs() < 1e-6,
                    "AVX512VL block mismatch vs AVX2 at len={}, i={}",
                    len,
                    i
                );
                assert!(
                    (head_vl[i] - head_avx2[i]).abs() < 1e-6,
                    "AVX512VL head mismatch vs AVX2 at len={}, i={}",
                    len,
                    i
                );
            }
        }
    }
}

#[test]
fn test_tanh_and_overwrite_block_avx512vl_parity() {
    let sizes = [
        0, 1, 2, 3, 4, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 48, 64, 128, 255, 256,
    ];
    for &len in &sizes {
        let init_block: Vec<f32> = (0..len).map(|i| (i as f32 * 0.17).sin() * 3.0).collect();
        let init_head: Vec<f32> = vec![-999.0; len];

        let mut head_scalar = init_head.clone();
        let mut block_scalar = init_block.clone();
        unsafe { tanh_and_overwrite_block_fallback(&mut head_scalar, &mut block_scalar) };

        let mut head_avx2 = init_head.clone();
        let mut block_avx2 = init_block.clone();
        unsafe { tanh_and_overwrite_block_avx2(&mut head_avx2, &mut block_avx2) };

        for i in 0..len {
            assert!((head_avx2[i] - head_scalar[i]).abs() < 1e-4);
            assert!((block_avx2[i] - block_scalar[i]).abs() < 1e-4);
        }

        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            let mut head_vl = init_head.clone();
            let mut block_vl = init_block.clone();
            unsafe { tanh_and_overwrite_block_avx512vl(&mut head_vl, &mut block_vl) };

            for i in 0..len {
                assert!((head_vl[i] - head_avx2[i]).abs() < 1e-6);
                assert!((block_vl[i] - block_avx2[i]).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn test_tanh_and_accumulate_with_seed_avx512vl_parity() {
    let sizes = [
        0, 1, 2, 3, 4, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 48, 64, 128, 255, 256,
    ];
    for &len in &sizes {
        let init_block: Vec<f32> = (0..len).map(|i| (i as f32 * 0.17).sin() * 3.0).collect();
        let seed: Vec<f32> = (0..len).map(|i| (i as f32 * 0.43).cos()).collect();
        let init_head: Vec<f32> = vec![-999.0; len];

        let mut head_scalar = init_head.clone();
        let mut block_scalar = init_block.clone();
        unsafe {
            tanh_and_accumulate_with_seed_fallback(&mut head_scalar, &mut block_scalar, &seed)
        };

        let mut head_avx2 = init_head.clone();
        let mut block_avx2 = init_block.clone();
        unsafe { tanh_and_accumulate_with_seed_avx2(&mut head_avx2, &mut block_avx2, &seed) };

        for i in 0..len {
            assert!((head_avx2[i] - head_scalar[i]).abs() < 1e-4);
            assert!((block_avx2[i] - block_scalar[i]).abs() < 1e-4);
        }

        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            let mut head_vl = init_head.clone();
            let mut block_vl = init_block.clone();
            unsafe { tanh_and_accumulate_with_seed_avx512vl(&mut head_vl, &mut block_vl, &seed) };

            for i in 0..len {
                assert!((head_vl[i] - head_avx2[i]).abs() < 1e-6);
                assert!((block_vl[i] - block_avx2[i]).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn test_gated_activation_and_accumulate_block_avx512vl_parity() {
    let channels = [1, 2, 3, 4, 7, 8, 12, 16, 24, 32];
    let num_frames_list = [1, 2, 4, 7, 8];

    for &ch in &channels {
        for &num_frames in &num_frames_list {
            let head_len = num_frames * ch;
            let block_len = num_frames * 2 * ch;

            let init_head: Vec<f32> = (0..head_len).map(|i| (i as f32 * 0.11).sin()).collect();
            let init_block: Vec<f32> = (0..block_len)
                .map(|i| (i as f32 * 0.23).cos() * 2.5)
                .collect();

            let mut head_scalar = init_head.clone();
            let mut block_scalar = init_block.clone();
            unsafe {
                gated_activation_and_accumulate_block_fallback(
                    &mut head_scalar,
                    &mut block_scalar,
                    ch,
                )
            };

            let mut head_avx2 = init_head.clone();
            let mut block_avx2 = init_block.clone();
            unsafe {
                gated_activation_and_accumulate_block_avx2(&mut head_avx2, &mut block_avx2, ch)
            };

            for i in 0..head_len {
                assert!(
                    (head_avx2[i] - head_scalar[i]).abs() < 1e-4,
                    "AVX2 head mismatch at ch={}, frames={}, i={}",
                    ch,
                    num_frames,
                    i
                );
            }

            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
                let mut head_vl = init_head.clone();
                let mut block_vl = init_block.clone();
                unsafe {
                    gated_activation_and_accumulate_block_avx512vl(&mut head_vl, &mut block_vl, ch)
                };

                for i in 0..head_len {
                    assert!(
                        (head_vl[i] - head_avx2[i]).abs() < 1e-6,
                        "AVX512VL head mismatch vs AVX2 at ch={}, frames={}, i={}",
                        ch,
                        num_frames,
                        i
                    );
                }
                for f in 0..num_frames {
                    for c in 0..ch {
                        let idx = f * 2 * ch + c;
                        assert!(
                            (block_vl[idx] - block_avx2[idx]).abs() < 1e-6,
                            "AVX512VL block mismatch vs AVX2 at ch={}, frames={}, idx={}",
                            ch,
                            num_frames,
                            idx
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_gated_activation_and_overwrite_block_avx512vl_parity() {
    let channels = [1, 2, 3, 4, 7, 8, 12, 16, 24, 32];
    let num_frames_list = [1, 2, 4, 7, 8];

    for &ch in &channels {
        for &num_frames in &num_frames_list {
            let head_len = num_frames * ch;
            let block_len = num_frames * 2 * ch;

            let init_head: Vec<f32> = vec![-999.0; head_len];
            let init_block: Vec<f32> = (0..block_len)
                .map(|i| (i as f32 * 0.23).cos() * 2.5)
                .collect();

            let mut head_scalar = init_head.clone();
            let mut block_scalar = init_block.clone();
            unsafe {
                gated_activation_and_overwrite_block_fallback(
                    &mut head_scalar,
                    &mut block_scalar,
                    ch,
                )
            };

            let mut head_avx2 = init_head.clone();
            let mut block_avx2 = init_block.clone();
            unsafe {
                gated_activation_and_overwrite_block_avx2(&mut head_avx2, &mut block_avx2, ch)
            };

            for i in 0..head_len {
                assert!(
                    (head_avx2[i] - head_scalar[i]).abs() < 1e-4,
                    "AVX2 head mismatch at ch={}, frames={}, i={}",
                    ch,
                    num_frames,
                    i
                );
            }

            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
                let mut head_vl = init_head.clone();
                let mut block_vl = init_block.clone();
                unsafe {
                    gated_activation_and_overwrite_block_avx512vl(&mut head_vl, &mut block_vl, ch)
                };

                for i in 0..head_len {
                    assert!(
                        (head_vl[i] - head_avx2[i]).abs() < 1e-6,
                        "AVX512VL head mismatch vs AVX2 at ch={}, frames={}, i={}",
                        ch,
                        num_frames,
                        i
                    );
                }
            }
        }
    }
}

#[test]
fn test_relu_and_accumulate_block_avx512vl_parity() {
    let sizes = [
        0, 1, 2, 3, 4, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 48, 64, 128, 255, 256,
    ];
    for &len in &sizes {
        let init_block: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.19).sin() * 5.0 - 2.0)
            .collect();
        let init_head: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).cos()).collect();

        let mut head_scalar = init_head.clone();
        let mut block_scalar = init_block.clone();
        unsafe { relu_and_accumulate_block_fallback(&mut head_scalar, &mut block_scalar) };

        let mut head_avx2 = init_head.clone();
        let mut block_avx2 = init_block.clone();
        unsafe { relu_and_accumulate_block_avx2(&mut head_avx2, &mut block_avx2) };

        for i in 0..len {
            assert!((head_avx2[i] - head_scalar[i]).abs() < 1e-6);
            assert!((block_avx2[i] - block_scalar[i]).abs() < 1e-6);
        }

        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            let mut head_vl = init_head.clone();
            let mut block_vl = init_block.clone();
            unsafe { relu_and_accumulate_block_avx512vl(&mut head_vl, &mut block_vl) };

            for i in 0..len {
                assert!((head_vl[i] - head_scalar[i]).abs() < 1e-6);
                assert!((block_vl[i] - block_scalar[i]).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn test_relu_and_overwrite_block_avx512vl_parity() {
    let sizes = [
        0, 1, 2, 3, 4, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 48, 64, 128, 255, 256,
    ];
    for &len in &sizes {
        let init_block: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.19).sin() * 5.0 - 2.0)
            .collect();
        let init_head: Vec<f32> = vec![-999.0; len];

        let mut head_scalar = init_head.clone();
        let mut block_scalar = init_block.clone();
        unsafe { relu_and_overwrite_block_fallback(&mut head_scalar, &mut block_scalar) };

        let mut head_avx2 = init_head.clone();
        let mut block_avx2 = init_block.clone();
        unsafe { relu_and_overwrite_block_avx2(&mut head_avx2, &mut block_avx2) };

        for i in 0..len {
            assert!((head_avx2[i] - head_scalar[i]).abs() < 1e-6);
            assert!((block_avx2[i] - block_scalar[i]).abs() < 1e-6);
        }

        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            let mut head_vl = init_head.clone();
            let mut block_vl = init_block.clone();
            unsafe { relu_and_overwrite_block_avx512vl(&mut head_vl, &mut block_vl) };

            for i in 0..len {
                assert!((head_vl[i] - head_scalar[i]).abs() < 1e-6);
                assert!((block_vl[i] - block_scalar[i]).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn test_relu_and_accumulate_with_seed_avx512vl_parity() {
    let sizes = [
        0, 1, 2, 3, 4, 7, 8, 9, 12, 15, 16, 17, 24, 31, 32, 48, 64, 128, 255, 256,
    ];
    for &len in &sizes {
        let init_block: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.19).sin() * 5.0 - 2.0)
            .collect();
        let seed: Vec<f32> = (0..len).map(|i| (i as f32 * 0.47).cos()).collect();
        let init_head: Vec<f32> = vec![-999.0; len];

        let mut head_scalar = init_head.clone();
        let mut block_scalar = init_block.clone();
        unsafe {
            relu_and_accumulate_with_seed_fallback(&mut head_scalar, &mut block_scalar, &seed)
        };

        let mut head_avx2 = init_head.clone();
        let mut block_avx2 = init_block.clone();
        unsafe { relu_and_accumulate_with_seed_avx2(&mut head_avx2, &mut block_avx2, &seed) };

        for i in 0..len {
            assert!((head_avx2[i] - head_scalar[i]).abs() < 1e-6);
            assert!((block_avx2[i] - block_scalar[i]).abs() < 1e-6);
        }

        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            let mut head_vl = init_head.clone();
            let mut block_vl = init_block.clone();
            unsafe { relu_and_accumulate_with_seed_avx512vl(&mut head_vl, &mut block_vl, &seed) };

            for i in 0..len {
                assert!((head_vl[i] - head_scalar[i]).abs() < 1e-6);
                assert!((block_vl[i] - block_scalar[i]).abs() < 1e-6);
            }
        }
    }
}
