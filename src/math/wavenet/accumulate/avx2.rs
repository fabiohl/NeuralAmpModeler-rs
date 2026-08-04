// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX2 accumulation and activation kernels for WaveNet.

use crate::wavenet_simd_avx2;
use core::arch::x86_64::*;

#[cold]
#[inline(never)]
fn accumulate_head_avx2_tail(dest: &mut [f32], src: &[f32]) {
    for i in 0..dest.len() {
        let acc = dest[i] as f64 + src[i] as f64;
        dest[i] = acc as f32;
    }
}

/// Accumulates src into dest using AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn accumulate_head_avx2(dest: &mut [f32], src: &[f32]) {
    let len = dest.len();
    let mut i = 0;
    wavenet_simd_avx2!(i, len, {
        let vs = _mm256_loadu_ps(src.as_ptr().add(i));
        let vd = _mm256_loadu_ps(dest.as_ptr().add(i));
        _mm256_storeu_ps(dest.as_mut_ptr().add(i), _mm256_add_ps(vd, vs));
    });
    if i < len {
        accumulate_head_avx2_tail(&mut dest[i..], &src[i..]);
    }
}

#[cold]
#[inline(never)]
fn tanh_and_accumulate_block_avx2_tail(head_input: &mut [f32], block: &mut [f32]) {
    for i in 0..block.len() {
        let val = block[i].tanh();
        block[i] = val;
        let acc = head_input[i] as f64 + val as f64;
        head_input[i] = acc as f32;
    }
}

/// Applies tanh in-place on block and accumulates into head_input using AVX2.
/// Processes 2 ymm vectors per iteration to overlap `vdivps` latencies.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_and_accumulate_block_avx2(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    while i + 16 <= len {
        let vb0 = _mm256_loadu_ps(block.as_ptr().add(i));
        let vb1 = _mm256_loadu_ps(block.as_ptr().add(i + 8));
        let vt0 = crate::math::activations::simd_tanh_poly_avx2(vb0);
        let vt1 = crate::math::activations::simd_tanh_poly_avx2(vb1);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt0);
        _mm256_storeu_ps(block.as_mut_ptr().add(i + 8), vt1);

        let vh0 = _mm256_loadu_ps(head_input.as_ptr().add(i));
        let vh1 = _mm256_loadu_ps(head_input.as_ptr().add(i + 8));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vh0, vt0));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i + 8), _mm256_add_ps(vh1, vt1));
        i += 16;
    }
    wavenet_simd_avx2!(i, len, {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vh = _mm256_loadu_ps(head_input.as_ptr().add(i));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vh, vt));
    });
    if i < len {
        tanh_and_accumulate_block_avx2_tail(&mut head_input[i..], &mut block[i..]);
    }
}

#[cold]
#[inline(never)]
fn gated_activation_and_accumulate_block_avx2_tail(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
    f: usize,
    start_c: usize,
) {
    let block_offset = f * 2 * ch;
    let head_offset = f * ch;
    for c in start_c..ch {
        let z1 = block[block_offset + c];
        let z2 = block[block_offset + ch + c];
        let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
        block[block_offset + c] = activated;
        let acc = head_input[head_offset + c] as f64 + activated as f64;
        head_input[head_offset + c] = acc as f32;
    }
}

/// Applies gated activation (tanh * sigmoid) in-place on block and accumulates into head_input using AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gated_activation_and_accumulate_block_avx2(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        wavenet_simd_avx2!(c, ch, {
            let z1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
            let activated = _mm256_mul_ps(tanh_z1, sig_z2);

            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);

            let vh = _mm256_loadu_ps(head_input.as_ptr().add(head_offset + c));
            _mm256_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c),
                _mm256_add_ps(vh, activated),
            );
        });
        if c < ch {
            gated_activation_and_accumulate_block_avx2_tail(head_input, block, ch, f, c);
        }
    }
}

#[cold]
#[inline(never)]
fn tanh_and_overwrite_block_avx2_tail(head_input: &mut [f32], block: &mut [f32]) {
    for i in 0..block.len() {
        let val = block[i].tanh();
        block[i] = val;
        head_input[i] = val;
    }
}

/// Applies tanh in-place on block and overwrites head_input using AVX2.
/// Processes 2 ymm vectors per iteration to overlap `vdivps` latencies.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_and_overwrite_block_avx2(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    while i + 16 <= len {
        let vb0 = _mm256_loadu_ps(block.as_ptr().add(i));
        let vb1 = _mm256_loadu_ps(block.as_ptr().add(i + 8));
        let vt0 = crate::math::activations::simd_tanh_poly_avx2(vb0);
        let vt1 = crate::math::activations::simd_tanh_poly_avx2(vb1);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt0);
        _mm256_storeu_ps(block.as_mut_ptr().add(i + 8), vt1);
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), vt0);
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i + 8), vt1);
        i += 16;
    }
    wavenet_simd_avx2!(i, len, {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), vt);
    });
    if i < len {
        tanh_and_overwrite_block_avx2_tail(&mut head_input[i..], &mut block[i..]);
    }
}

#[cold]
#[inline(never)]
fn tanh_and_accumulate_with_seed_avx2_tail(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    for i in 0..block.len() {
        let val = block[i].tanh();
        block[i] = val;
        let acc = seed[i] as f64 + val as f64;
        head_input[i] = acc as f32;
    }
}

/// Fused Seed + Tanh + Head Accumulate using AVX2.
///
/// Computes `head_input[i] = seed[i] + tanh(block[i])`.
/// Eliminates the separate `copy_from_slice(seed)` before `tanh_and_accumulate_block`.
/// Processes 2 ymm vectors per iteration to overlap `vdivps` latencies.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_and_accumulate_with_seed_avx2(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    let len = block.len();
    let mut i = 0;
    while i + 16 <= len {
        let vb0 = _mm256_loadu_ps(block.as_ptr().add(i));
        let vb1 = _mm256_loadu_ps(block.as_ptr().add(i + 8));
        let vt0 = crate::math::activations::simd_tanh_poly_avx2(vb0);
        let vt1 = crate::math::activations::simd_tanh_poly_avx2(vb1);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt0);
        _mm256_storeu_ps(block.as_mut_ptr().add(i + 8), vt1);

        let vs0 = _mm256_loadu_ps(seed.as_ptr().add(i));
        let vs1 = _mm256_loadu_ps(seed.as_ptr().add(i + 8));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vs0, vt0));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i + 8), _mm256_add_ps(vs1, vt1));
        i += 16;
    }
    wavenet_simd_avx2!(i, len, {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vs = _mm256_loadu_ps(seed.as_ptr().add(i));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vs, vt));
    });
    if i < len {
        tanh_and_accumulate_with_seed_avx2_tail(&mut head_input[i..], &mut block[i..], &seed[i..]);
    }
}

#[cold]
#[inline(never)]
fn gated_activation_and_overwrite_block_avx2_tail(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
    f: usize,
    start_c: usize,
) {
    let block_offset = f * 2 * ch;
    let head_offset = f * ch;
    for c in start_c..ch {
        let z1 = block[block_offset + c];
        let z2 = block[block_offset + ch + c];
        let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
        block[block_offset + c] = activated;
        head_input[head_offset + c] = activated;
    }
}

/// Applies gated activation (tanh * sigmoid) in-place on block and overwrites head_input using AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gated_activation_and_overwrite_block_avx2(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        wavenet_simd_avx2!(c, ch, {
            let z1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
            let activated = _mm256_mul_ps(tanh_z1, sig_z2);

            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);
            _mm256_storeu_ps(head_input.as_mut_ptr().add(head_offset + c), activated);
        });
        if c < ch {
            gated_activation_and_overwrite_block_avx2_tail(head_input, block, ch, f, c);
        }
    }
}
