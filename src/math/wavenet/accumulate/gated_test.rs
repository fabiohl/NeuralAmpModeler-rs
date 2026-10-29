// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Bit-exactness tests for the gated activation CH=4/CH=12 remainder pairing.
//!
//! The oracle below reproduces the previous per-frame arithmetic: full 8-wide
//! vectors, then an 8-wide kernel with the remainder lanes zero-filled (the
//! same lanes a `maskload` used to clear). The paired implementation must keep
//! every lane's bits unchanged.

use crate::arch::simd::*;

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn gated_reference(head_input: &mut [f32], block: &mut [f32], ch: usize, overwrite: bool) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        while c + 8 <= ch {
            // SAFETY: full vectors stay inside this frame's `2 * ch` window.
            unsafe {
                let z1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
                let z2 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));
                let (tanh_z1, sig_z2) =
                    crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
                let activated = _mm256_mul_ps(tanh_z1, sig_z2);
                _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);
                if overwrite {
                    _mm256_storeu_ps(head_input.as_mut_ptr().add(head_offset + c), activated);
                } else {
                    let vh = _mm256_loadu_ps(head_input.as_ptr().add(head_offset + c));
                    _mm256_storeu_ps(
                        head_input.as_mut_ptr().add(head_offset + c),
                        _mm256_add_ps(vh, activated),
                    );
                }
            }
            c += 8;
        }
        if c < ch {
            let rem = ch - c;
            let mut z1_buf = [0.0f32; 8];
            let mut z2_buf = [0.0f32; 8];
            z1_buf[..rem].copy_from_slice(&block[block_offset + c..block_offset + c + rem]);
            z2_buf[..rem]
                .copy_from_slice(&block[block_offset + ch + c..block_offset + ch + c + rem]);
            // SAFETY: both stack buffers hold exactly 8 f32; only `rem` lanes
            // carry data, as a maskload would.
            unsafe {
                let z1 = _mm256_loadu_ps(z1_buf.as_ptr());
                let z2 = _mm256_loadu_ps(z2_buf.as_ptr());
                let (tanh_z1, sig_z2) =
                    crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
                let activated = _mm256_mul_ps(tanh_z1, sig_z2);
                let mut out = [0.0f32; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), activated);
                block[block_offset + c..block_offset + c + rem].copy_from_slice(&out[..rem]);
                if overwrite {
                    head_input[head_offset + c..head_offset + c + rem].copy_from_slice(&out[..rem]);
                } else {
                    for lane in 0..rem {
                        head_input[head_offset + c + lane] += out[lane];
                    }
                }
            }
        }
    }
}

fn make_inputs(ch: usize, num_frames: usize) -> (Vec<f32>, Vec<f32>) {
    let head: Vec<f32> = (0..num_frames * ch)
        .map(|i| ((i as f32 * 0.113).sin() * 0.7) - 0.2)
        .collect();
    let block: Vec<f32> = (0..num_frames * 2 * ch)
        .map(|i| ((i as f32 * 0.231).cos() * 2.3) + 0.1)
        .collect();
    (head, block)
}

#[test]
fn test_gated_activation_pairing_bit_exact_accumulate() {
    for &ch in &[1usize, 2, 3, 4, 5, 6, 7, 12] {
        for &num_frames in &[1usize, 2, 3, 4, 5, 7, 63, 65] {
            let (head, block) = make_inputs(ch, num_frames);
            let mut head_ref = head.clone();
            let mut block_ref = block.clone();
            let mut head_new = head;
            let mut block_new = block;

            // SAFETY: buffers hold `num_frames * ch` and `num_frames * 2 * ch`
            // elements as required by both kernels.
            unsafe {
                gated_reference(&mut head_ref, &mut block_ref, ch, false);
                crate::math::wavenet::accumulate::gated_activation_and_accumulate_block_avx2(
                    &mut head_new,
                    &mut block_new,
                    ch,
                );
            }

            for i in 0..head_ref.len() {
                assert_eq!(
                    head_ref[i].to_bits(),
                    head_new[i].to_bits(),
                    "head ch={ch} frames={num_frames} i={i}"
                );
            }
            for f in 0..num_frames {
                for c in 0..ch {
                    let idx = f * 2 * ch + c;
                    assert_eq!(
                        block_ref[idx].to_bits(),
                        block_new[idx].to_bits(),
                        "block ch={ch} frames={num_frames} idx={idx}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_gated_activation_pairing_bit_exact_overwrite() {
    for &ch in &[1usize, 2, 3, 4, 5, 6, 7, 12] {
        for &num_frames in &[1usize, 2, 3, 4, 5, 7, 63, 65] {
            let (head, block) = make_inputs(ch, num_frames);
            let mut head_ref = head.clone();
            let mut block_ref = block.clone();
            let mut head_new = head;
            let mut block_new = block;

            // SAFETY: same buffer geometry as the accumulate test.
            unsafe {
                gated_reference(&mut head_ref, &mut block_ref, ch, true);
                crate::math::wavenet::accumulate::gated_activation_and_overwrite_block_avx2(
                    &mut head_new,
                    &mut block_new,
                    ch,
                );
            }

            for i in 0..head_ref.len() {
                assert_eq!(
                    head_ref[i].to_bits(),
                    head_new[i].to_bits(),
                    "head ch={ch} frames={num_frames} i={i}"
                );
            }
            for f in 0..num_frames {
                for c in 0..ch {
                    let idx = f * 2 * ch + c;
                    assert_eq!(
                        block_ref[idx].to_bits(),
                        block_new[idx].to_bits(),
                        "block ch={ch} frames={num_frames} idx={idx}"
                    );
                }
            }
        }
    }
}
