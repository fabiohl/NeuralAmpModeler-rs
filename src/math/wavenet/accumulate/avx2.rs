// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX2 accumulation and activation kernels for WaveNet.

use crate::wavenet_simd_avx2;
use core::arch::x86_64::*;

/// Builds an AVX2 tail mask with the low `rem` lanes enabled (`-1`) and the rest zero.
///
/// Lets remainder paths (`rem < 8`) run the full 256-bit vector kernels without
/// scalar fallback, heap allocation, or libm calls. Stack-resident only; the
/// mask lives in a register after load.
#[target_feature(enable = "avx2")]
unsafe fn avx2_tail_mask(rem: usize) -> __m256i {
    let mut vals = [0i32; 8];
    for item in vals.iter_mut().take(rem) {
        *item = -1;
    }
    // SAFETY: `vals` is an 8-element stack array, so the 256-bit unaligned load stays in bounds.
    unsafe { _mm256_loadu_si256(vals.as_ptr() as *const __m256i) }
}

/// Masked tail for head accumulation: `dest[i] += src[i]` in f32.
///
/// Matches the vector loop (`_mm256_add_ps`, f32) lane-for-lane. Fractional
/// blocks run the same vector add under an AVX2 mask, so they accumulate
/// identically to full vectors.
#[target_feature(enable = "avx2")]
unsafe fn accumulate_head_avx2_tail(dest: &mut [f32], src: &[f32]) {
    let rem = dest.len();
    if rem == 0 {
        return;
    }
    // SAFETY: caller guarantees `src.len() >= dest.len()` (public contract);
    // the mask enables only the low `rem` lanes, so masked loads/stores touch
    // exactly `dest[0..rem]` and `src[0..rem]` with fault suppression elsewhere.
    unsafe {
        let mask = avx2_tail_mask(rem);
        let vs = _mm256_maskload_ps(src.as_ptr(), mask);
        let vd = _mm256_maskload_ps(dest.as_ptr(), mask);
        _mm256_maskstore_ps(dest.as_mut_ptr(), mask, _mm256_add_ps(vd, vs));
    }
}

/// Accumulates src into dest using AVX2.
///
/// # Safety
///
/// - The CPU must support AVX2 (guaranteed by the ISA dispatch; never call
///   directly on an unchecked host).
/// - `src.len() >= dest.len()`: the vector loop performs unaligned 256-bit
///   raw-pointer loads from `src` up to `dest.len()`; a shorter `src` is read
///   out of bounds (UB).
#[target_feature(enable = "avx2")]
pub unsafe fn accumulate_head_avx2(dest: &mut [f32], src: &[f32]) {
    let len = dest.len();
    let mut i = 0;
    // SAFETY: caller guarantees `src.len() >= dest.len()`, loop guard `i + 8 <= len` keeps all 8-lane loads/stores in bounds.
    unsafe {
        wavenet_simd_avx2!(i, len, {
            let vs = _mm256_loadu_ps(src.as_ptr().add(i));
            let vd = _mm256_loadu_ps(dest.as_ptr().add(i));
            _mm256_storeu_ps(dest.as_mut_ptr().add(i), _mm256_add_ps(vd, vs));
        });
    }
    if i < len {
        // SAFETY: same slice contract as the vector loop above; masked tail touches only `..rem`.
        unsafe {
            accumulate_head_avx2_tail(&mut dest[i..], &src[i..]);
        }
    }
}

/// Masked tail for tanh + accumulate: polynomial tanh, f32 accumulation.
///
/// Runs the same `simd_tanh_poly_avx2` kernel and `_mm256_add_ps` f32 sum as
/// the vector loop under an AVX2 mask, so the remainder lanes activate and
/// accumulate identically to full vectors.
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_and_accumulate_block_avx2_tail(head_input: &mut [f32], block: &mut [f32]) {
    let rem = block.len();
    if rem == 0 {
        return;
    }
    // SAFETY: caller guarantees `head_input.len() >= block.len()` (public contract);
    // the mask enables only the low `rem` lanes, so masked loads/stores touch
    // exactly `block[0..rem]` and `head_input[0..rem]`.
    unsafe {
        let mask = avx2_tail_mask(rem);
        let vb = _mm256_maskload_ps(block.as_ptr(), mask);
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_maskstore_ps(block.as_mut_ptr(), mask, vt);
        let vh = _mm256_maskload_ps(head_input.as_ptr(), mask);
        _mm256_maskstore_ps(head_input.as_mut_ptr(), mask, _mm256_add_ps(vh, vt));
    }
}

/// Applies tanh in-place on block and accumulates into head_input using AVX2.
/// Processes 2 ymm vectors per iteration to overlap `vdivps` latencies.
///
/// # Safety
///
/// - The CPU must support AVX2 and FMA (guaranteed by the ISA dispatch).
/// - `head_input.len() >= block.len()`: the vector loop performs unaligned
///   256-bit raw-pointer loads/stores up to `block.len()` on both slices; a
///   shorter `head_input` is accessed out of bounds (UB).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_and_accumulate_block_avx2(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    // SAFETY: caller guarantees `head_input.len() >= block.len()`, loop guards keep all unaligned 256-bit loads/stores in bounds.
    unsafe {
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
    }
    if i < len {
        // SAFETY: same slice contract as the vector loop above; masked tail touches only `..rem`.
        unsafe {
            tanh_and_accumulate_block_avx2_tail(&mut head_input[i..], &mut block[i..]);
        }
    }
}

/// Masked tail for gated activation + accumulate: polynomial dual kernel, f32 sum.
///
/// Runs the same `simd_tanh_sigmoid_dual_poly_avx2` dual kernel and f32 add as
/// the vector loop under an AVX2 mask, so the remainder lanes activate and
/// accumulate identically to full vectors.
#[target_feature(enable = "avx2,fma")]
unsafe fn gated_activation_and_accumulate_block_avx2_tail(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
    f: usize,
    start_c: usize,
) {
    let block_offset = f * 2 * ch;
    let head_offset = f * ch;
    let rem = ch - start_c;
    if rem == 0 {
        return;
    }
    // SAFETY: caller guarantees `ch >= 1`, `block.len() >= 2 * ch * num_frames`,
    // and `start_c <= ch`; the mask enables only the low `rem` lanes, so masked
    // loads/stores touch exactly the remainder channels.
    unsafe {
        let mask = avx2_tail_mask(rem);
        let z1 = _mm256_maskload_ps(block.as_ptr().add(block_offset + start_c), mask);
        let z2 = _mm256_maskload_ps(block.as_ptr().add(block_offset + ch + start_c), mask);
        let (tanh_z1, sig_z2) = crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
        let activated = _mm256_mul_ps(tanh_z1, sig_z2);
        _mm256_maskstore_ps(
            block.as_mut_ptr().add(block_offset + start_c),
            mask,
            activated,
        );
        let vh = _mm256_maskload_ps(head_input.as_ptr().add(head_offset + start_c), mask);
        _mm256_maskstore_ps(
            head_input.as_mut_ptr().add(head_offset + start_c),
            mask,
            _mm256_add_ps(vh, activated),
        );
    }
}

/// Applies gated activation (tanh * sigmoid) in-place on block and accumulates into head_input using AVX2.
///
/// # Safety
///
/// - The CPU must support AVX2 and FMA (guaranteed by the ISA dispatch).
/// - `ch >= 1` and `block.len() >= 2 * ch * (head_input.len() / ch)`: the
///   vector loop performs unaligned 256-bit raw-pointer loads/stores at
///   `block[f * 2 * ch + {c, ch + c}]`; a shorter `block` is accessed out of
///   bounds (UB). `ch == 0` also divides by zero.
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
        // SAFETY: `ch >= 1` and `block.len() >= 2 * ch * num_frames`, loop guards keep loads/stores in bounds.
        unsafe {
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
        }
        if c < ch {
            // SAFETY: same channel geometry as the vector loop above; masked tail touches only the remainder channels.
            unsafe {
                gated_activation_and_accumulate_block_avx2_tail(head_input, block, ch, f, c);
            }
        }
    }
}

/// Masked tail for tanh + overwrite: polynomial tanh, in-place store.
///
/// Runs the same `simd_tanh_poly_avx2` kernel as the vector loop under an AVX2
/// mask, so the remainder lanes activate identically to full vectors.
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_and_overwrite_block_avx2_tail(head_input: &mut [f32], block: &mut [f32]) {
    let rem = block.len();
    if rem == 0 {
        return;
    }
    // SAFETY: caller guarantees `head_input.len() >= block.len()` (public contract);
    // the mask enables only the low `rem` lanes, so masked loads/stores touch
    // exactly `block[0..rem]` and `head_input[0..rem]`.
    unsafe {
        let mask = avx2_tail_mask(rem);
        let vb = _mm256_maskload_ps(block.as_ptr(), mask);
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_maskstore_ps(block.as_mut_ptr(), mask, vt);
        _mm256_maskstore_ps(head_input.as_mut_ptr(), mask, vt);
    }
}

/// Applies tanh in-place on block and overwrites head_input using AVX2.
/// Processes 2 ymm vectors per iteration to overlap `vdivps` latencies.
///
/// # Safety
///
/// - The CPU must support AVX2 and FMA (guaranteed by the ISA dispatch).
/// - `head_input.len() >= block.len()`: the vector loop performs unaligned
///   256-bit raw-pointer stores up to `block.len()` on both slices; a shorter
///   `head_input` is written out of bounds (UB).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_and_overwrite_block_avx2(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    // SAFETY: `head_input.len() >= block.len()`, loop guards keep loads/stores in bounds.
    unsafe {
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
    }
    if i < len {
        // SAFETY: same slice contract as the vector loop above; masked tail touches only `..rem`.
        unsafe {
            tanh_and_overwrite_block_avx2_tail(&mut head_input[i..], &mut block[i..]);
        }
    }
}

/// Masked tail for fused seed + tanh + accumulate: polynomial tanh, f32 sum.
///
/// Runs the same `simd_tanh_poly_avx2` kernel and f32 add as the vector loop
/// under an AVX2 mask, so the remainder lanes compute `seed + tanh(block)`
/// identically to full vectors.
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_and_accumulate_with_seed_avx2_tail(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    let rem = block.len();
    if rem == 0 {
        return;
    }
    // SAFETY: caller guarantees `head_input.len() >= block.len()` and
    // `seed.len() >= block.len()` (public contract); the mask enables only the
    // low `rem` lanes, so masked loads/stores touch exactly `..rem` on all slices.
    unsafe {
        let mask = avx2_tail_mask(rem);
        let vb = _mm256_maskload_ps(block.as_ptr(), mask);
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_maskstore_ps(block.as_mut_ptr(), mask, vt);
        let vs = _mm256_maskload_ps(seed.as_ptr(), mask);
        _mm256_maskstore_ps(head_input.as_mut_ptr(), mask, _mm256_add_ps(vs, vt));
    }
}

/// Fused Seed + Tanh + Head Accumulate using AVX2.
///
/// Computes `head_input[i] = seed[i] + tanh(block[i])`.
/// Eliminates the separate `copy_from_slice(seed)` before `tanh_and_accumulate_block`.
/// Processes 2 ymm vectors per iteration to overlap `vdivps` latencies.
///
/// # Safety
///
/// - The CPU must support AVX2 and FMA (guaranteed by the ISA dispatch).
/// - `head_input.len() >= block.len()` and `seed.len() >= block.len()`: the
///   vector loop performs unaligned 256-bit raw-pointer loads/stores up to
///   `block.len()` on all three slices; shorter slices are accessed out of
///   bounds (UB).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_and_accumulate_with_seed_avx2(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    let len = block.len();
    let mut i = 0;
    // SAFETY: `head_input.len() >= block.len()` and `seed.len() >= block.len()`, loop guards keep loads/stores in bounds.
    unsafe {
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
    }
    if i < len {
        // SAFETY: same slice contract as the vector loop above; masked tail touches only `..rem`.
        unsafe {
            tanh_and_accumulate_with_seed_avx2_tail(
                &mut head_input[i..],
                &mut block[i..],
                &seed[i..],
            );
        }
    }
}

/// Masked tail for ReLU + accumulate: vector max, f32 sum.
///
/// Runs the same `_mm256_max_ps` and f32 add as the vector loop under an AVX2
/// mask, so the remainder lanes activate and accumulate identically to full vectors.
#[target_feature(enable = "avx2")]
unsafe fn relu_and_accumulate_block_avx2_tail(head_input: &mut [f32], block: &mut [f32]) {
    let rem = block.len();
    if rem == 0 {
        return;
    }
    // SAFETY: caller guarantees `head_input.len() >= block.len()` (public contract);
    // the mask enables only the low `rem` lanes, so masked loads/stores touch
    // exactly `block[0..rem]` and `head_input[0..rem]`.
    unsafe {
        let mask = avx2_tail_mask(rem);
        let zero = _mm256_setzero_ps();
        let vb = _mm256_maskload_ps(block.as_ptr(), mask);
        let vt = _mm256_max_ps(vb, zero);
        _mm256_maskstore_ps(block.as_mut_ptr(), mask, vt);
        let vh = _mm256_maskload_ps(head_input.as_ptr(), mask);
        _mm256_maskstore_ps(head_input.as_mut_ptr(), mask, _mm256_add_ps(vh, vt));
    }
}

/// Applies ReLU in-place on block and accumulates into head_input using AVX2.
///
/// # Safety
///
/// - The CPU must support AVX2 and FMA (guaranteed by the ISA dispatch).
/// - `head_input.len() >= block.len()`: the vector loop performs unaligned
///   256-bit raw-pointer loads/stores up to `block.len()` on both slices; a
///   shorter `head_input` is accessed out of bounds (UB).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn relu_and_accumulate_block_avx2(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    // SAFETY: `head_input.len() >= block.len()`, loop guards keep loads/stores in bounds.
    unsafe {
        let zero = _mm256_setzero_ps();
        while i + 16 <= len {
            let vb0 = _mm256_loadu_ps(block.as_ptr().add(i));
            let vb1 = _mm256_loadu_ps(block.as_ptr().add(i + 8));
            let vt0 = _mm256_max_ps(vb0, zero);
            let vt1 = _mm256_max_ps(vb1, zero);
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
            let vt = _mm256_max_ps(vb, zero);
            _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);

            let vh = _mm256_loadu_ps(head_input.as_ptr().add(i));
            _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vh, vt));
        });
    }
    if i < len {
        // SAFETY: same slice contract as the vector loop above; masked tail touches only `..rem`.
        unsafe {
            relu_and_accumulate_block_avx2_tail(&mut head_input[i..], &mut block[i..]);
        }
    }
}

/// Masked tail for ReLU + overwrite: vector max, in-place store.
///
/// Runs the same `_mm256_max_ps` as the vector loop under an AVX2 mask, so the
/// remainder lanes activate identically to full vectors.
#[target_feature(enable = "avx2")]
unsafe fn relu_and_overwrite_block_avx2_tail(head_input: &mut [f32], block: &mut [f32]) {
    let rem = block.len();
    if rem == 0 {
        return;
    }
    // SAFETY: caller guarantees `head_input.len() >= block.len()` (public contract);
    // the mask enables only the low `rem` lanes, so masked loads/stores touch
    // exactly `block[0..rem]` and `head_input[0..rem]`.
    unsafe {
        let mask = avx2_tail_mask(rem);
        let zero = _mm256_setzero_ps();
        let vb = _mm256_maskload_ps(block.as_ptr(), mask);
        let vt = _mm256_max_ps(vb, zero);
        _mm256_maskstore_ps(block.as_mut_ptr(), mask, vt);
        _mm256_maskstore_ps(head_input.as_mut_ptr(), mask, vt);
    }
}

/// Applies ReLU in-place on block and overwrites head_input using AVX2.
///
/// # Safety
///
/// - The CPU must support AVX2 and FMA (guaranteed by the ISA dispatch).
/// - `head_input.len() >= block.len()`: the vector loop performs unaligned
///   256-bit raw-pointer stores up to `block.len()` on both slices; a shorter
///   `head_input` is written out of bounds (UB).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn relu_and_overwrite_block_avx2(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    // SAFETY: `head_input.len() >= block.len()`, loop guards keep loads/stores in bounds.
    unsafe {
        let zero = _mm256_setzero_ps();
        while i + 16 <= len {
            let vb0 = _mm256_loadu_ps(block.as_ptr().add(i));
            let vb1 = _mm256_loadu_ps(block.as_ptr().add(i + 8));
            let vt0 = _mm256_max_ps(vb0, zero);
            let vt1 = _mm256_max_ps(vb1, zero);
            _mm256_storeu_ps(block.as_mut_ptr().add(i), vt0);
            _mm256_storeu_ps(block.as_mut_ptr().add(i + 8), vt1);
            _mm256_storeu_ps(head_input.as_mut_ptr().add(i), vt0);
            _mm256_storeu_ps(head_input.as_mut_ptr().add(i + 8), vt1);
            i += 16;
        }
        wavenet_simd_avx2!(i, len, {
            let vb = _mm256_loadu_ps(block.as_ptr().add(i));
            let vt = _mm256_max_ps(vb, zero);
            _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);
            _mm256_storeu_ps(head_input.as_mut_ptr().add(i), vt);
        });
    }
    if i < len {
        // SAFETY: same slice contract as the vector loop above; masked tail touches only `..rem`.
        unsafe {
            relu_and_overwrite_block_avx2_tail(&mut head_input[i..], &mut block[i..]);
        }
    }
}

/// Masked tail for fused seed + ReLU + accumulate: vector max, f32 sum.
///
/// Runs the same `_mm256_max_ps` and f32 add as the vector loop under an AVX2
/// mask, so the remainder lanes compute `seed + max(0, block)` identically to
/// full vectors.
#[target_feature(enable = "avx2")]
unsafe fn relu_and_accumulate_with_seed_avx2_tail(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    let rem = block.len();
    if rem == 0 {
        return;
    }
    // SAFETY: caller guarantees `head_input.len() >= block.len()` and
    // `seed.len() >= block.len()` (public contract); the mask enables only the
    // low `rem` lanes, so masked loads/stores touch exactly `..rem` on all slices.
    unsafe {
        let mask = avx2_tail_mask(rem);
        let zero = _mm256_setzero_ps();
        let vb = _mm256_maskload_ps(block.as_ptr(), mask);
        let vt = _mm256_max_ps(vb, zero);
        _mm256_maskstore_ps(block.as_mut_ptr(), mask, vt);
        let vs = _mm256_maskload_ps(seed.as_ptr(), mask);
        _mm256_maskstore_ps(head_input.as_mut_ptr(), mask, _mm256_add_ps(vs, vt));
    }
}

/// Fused Seed + ReLU + Head Accumulate using AVX2.
///
/// Computes `head_input[i] = seed[i] + max(0.0, block[i])`.
///
/// # Safety
///
/// - The CPU must support AVX2 and FMA (guaranteed by the ISA dispatch).
/// - `head_input.len() >= block.len()` and `seed.len() >= block.len()`: the
///   vector loop performs unaligned 256-bit raw-pointer loads/stores up to
///   `block.len()` on all three slices; shorter slices are accessed out of
///   bounds (UB).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn relu_and_accumulate_with_seed_avx2(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    let len = block.len();
    let mut i = 0;
    // SAFETY: `head_input.len() >= block.len()` and `seed.len() >= block.len()`, loop guards keep loads/stores in bounds.
    unsafe {
        let zero = _mm256_setzero_ps();
        while i + 16 <= len {
            let vb0 = _mm256_loadu_ps(block.as_ptr().add(i));
            let vb1 = _mm256_loadu_ps(block.as_ptr().add(i + 8));
            let vt0 = _mm256_max_ps(vb0, zero);
            let vt1 = _mm256_max_ps(vb1, zero);
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
            let vt = _mm256_max_ps(vb, zero);
            _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);

            let vs = _mm256_loadu_ps(seed.as_ptr().add(i));
            _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vs, vt));
        });
    }
    if i < len {
        // SAFETY: same slice contract as the vector loop above; masked tail touches only `..rem`.
        unsafe {
            relu_and_accumulate_with_seed_avx2_tail(
                &mut head_input[i..],
                &mut block[i..],
                &seed[i..],
            );
        }
    }
}

/// Masked tail for gated activation + overwrite: polynomial dual kernel.
///
/// Runs the same `simd_tanh_sigmoid_dual_poly_avx2` dual kernel as the vector
/// loop under an AVX2 mask, so the remainder lanes activate identically to
/// full vectors.
#[target_feature(enable = "avx2,fma")]
unsafe fn gated_activation_and_overwrite_block_avx2_tail(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
    f: usize,
    start_c: usize,
) {
    let block_offset = f * 2 * ch;
    let head_offset = f * ch;
    let rem = ch - start_c;
    if rem == 0 {
        return;
    }
    // SAFETY: caller guarantees `ch >= 1`, `block.len() >= 2 * ch * num_frames`,
    // and `start_c <= ch`; the mask enables only the low `rem` lanes, so masked
    // loads/stores touch exactly the remainder channels.
    unsafe {
        let mask = avx2_tail_mask(rem);
        let z1 = _mm256_maskload_ps(block.as_ptr().add(block_offset + start_c), mask);
        let z2 = _mm256_maskload_ps(block.as_ptr().add(block_offset + ch + start_c), mask);
        let (tanh_z1, sig_z2) = crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
        let activated = _mm256_mul_ps(tanh_z1, sig_z2);
        _mm256_maskstore_ps(
            block.as_mut_ptr().add(block_offset + start_c),
            mask,
            activated,
        );
        _mm256_maskstore_ps(
            head_input.as_mut_ptr().add(head_offset + start_c),
            mask,
            activated,
        );
    }
}

/// Applies gated activation (tanh * sigmoid) in-place on block and overwrites head_input using AVX2.
///
/// # Safety
///
/// - The CPU must support AVX2 and FMA (guaranteed by the ISA dispatch).
/// - `ch >= 1` and `block.len() >= 2 * ch * (head_input.len() / ch)`: the
///   vector loop performs unaligned 256-bit raw-pointer loads/stores at
///   `block[f * 2 * ch + {c, ch + c}]`; a shorter `block` is accessed out of
///   bounds (UB). `ch == 0` also divides by zero.
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
        // SAFETY: `ch >= 1` and `block.len() >= 2 * ch * num_frames`, loop guards keep loads/stores in bounds.
        unsafe {
            wavenet_simd_avx2!(c, ch, {
                let z1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
                let z2 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));

                let (tanh_z1, sig_z2) =
                    crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
                let activated = _mm256_mul_ps(tanh_z1, sig_z2);

                _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);
                _mm256_storeu_ps(head_input.as_mut_ptr().add(head_offset + c), activated);
            });
        }
        if c < ch {
            // SAFETY: same channel geometry as the vector loop above; masked tail touches only the remainder channels.
            unsafe {
                gated_activation_and_overwrite_block_avx2_tail(head_input, block, ch, f, c);
            }
        }
    }
}
