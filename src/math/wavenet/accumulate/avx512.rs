// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 accumulation and activation kernels for WaveNet.

use core::arch::x86_64::*;

/// Applies the "gate" activation (tanh * sigmoid) to audio blocks.
/// Imagine each sound goes through two filters: one that shapes the timbre (tanh)
/// and another that controls the intensity (sigmoid). The result is added to "head_input".
///
/// # Safety
///
/// - The CPU must support AVX-512F and AVX-512VL (guaranteed by the ISA
///   dispatch, which additionally requires BW+DQ; never call directly on an
///   unchecked host).
/// - `ch >= 1` and `block.len() >= 2 * ch * (head_input.len() / ch)`: the
///   vector loop performs unaligned 512-bit raw-pointer loads/stores at
///   `block[f * 2 * ch + {c, ch + c}]`; a shorter `block` is accessed out of
///   bounds (UB). `ch == 0` also divides by zero.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gated_activation_and_accumulate_block_avx512(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        while c + 16 <= ch {
            let z1 = _mm512_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm512_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx512(z1, z2);
            let activated = _mm512_mul_ps(tanh_z1, sig_z2);

            _mm512_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);

            let vh = _mm512_loadu_ps(head_input.as_ptr().add(head_offset + c));
            _mm512_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c),
                _mm512_add_ps(vh, activated),
            );
            c += 16;
        }
        while c < ch {
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
            block[block_offset + c] = activated;
            let acc = head_input[head_offset + c] as f64 + activated as f64;
            head_input[head_offset + c] = acc as f32;
            c += 1;
        }
    }
}

/// Applies the "gate" activation (tanh * sigmoid) to audio blocks.
/// Overwrites to "head_input".
///
/// # Safety
///
/// - The CPU must support AVX-512F and AVX-512VL (guaranteed by the ISA
///   dispatch, which additionally requires BW+DQ; never call directly on an
///   unchecked host).
/// - `ch >= 1` and `block.len() >= 2 * ch * (head_input.len() / ch)`: the
///   vector loop performs unaligned 512-bit raw-pointer loads/stores at
///   `block[f * 2 * ch + {c, ch + c}]`; a shorter `block` is accessed out of
///   bounds (UB). `ch == 0` also divides by zero.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gated_activation_and_overwrite_block_avx512(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        while c + 16 <= ch {
            let z1 = _mm512_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm512_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx512(z1, z2);
            let activated = _mm512_mul_ps(tanh_z1, sig_z2);

            _mm512_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);
            _mm512_storeu_ps(head_input.as_mut_ptr().add(head_offset + c), activated);
            c += 16;
        }
        while c < ch {
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
            block[block_offset + c] = activated;
            head_input[head_offset + c] = activated;
            c += 1;
        }
    }
}

/// Accumulates src into dest using AVX-512.
///
/// # Safety
///
/// - The CPU must support AVX-512F and AVX-512VL (guaranteed by the ISA
///   dispatch, which additionally requires BW+DQ; never call directly on an
///   unchecked host).
/// - `src.len() >= dest.len()`: the unmasked vector loop performs unaligned
///   512-bit raw-pointer loads from `src` up to `dest.len()`; a shorter `src`
///   is read out of bounds (UB). The masked tail only covers
///   `dest.len() % 16` lanes with fault suppression.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn accumulate_head_avx512(dest: &mut [f32], src: &[f32]) {
    let len = dest.len();
    let mut i = 0;
    while i + 16 <= len {
        let vs = _mm512_loadu_ps(src.as_ptr().add(i));
        let vd = _mm512_loadu_ps(dest.as_ptr().add(i));
        _mm512_storeu_ps(dest.as_mut_ptr().add(i), _mm512_add_ps(vd, vs));
        i += 16;
    }
    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let vs = _mm512_maskz_loadu_ps(mask, src.as_ptr().add(i));
        let vd = _mm512_maskz_loadu_ps(mask, dest.as_ptr().add(i));
        _mm512_mask_storeu_ps(dest.as_mut_ptr().add(i), mask, _mm512_add_ps(vd, vs));
    }
}

/// Applies tanh in-place on block and accumulates into head_input using AVX-512.
///
/// # Safety
///
/// - The CPU must support AVX-512F and AVX-512VL (guaranteed by the ISA
///   dispatch, which additionally requires BW+DQ; never call directly on an
///   unchecked host).
/// - `head_input.len() >= block.len()`: the unmasked vector loop performs
///   unaligned 512-bit raw-pointer loads/stores up to `block.len()` on both
///   slices; a shorter `head_input` is accessed out of bounds (UB).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn tanh_and_accumulate_block_avx512(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    while i + 16 <= len {
        let vb = _mm512_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx512(vb);
        _mm512_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vh = _mm512_loadu_ps(head_input.as_ptr().add(i));
        _mm512_storeu_ps(head_input.as_mut_ptr().add(i), _mm512_add_ps(vh, vt));
        i += 16;
    }
    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let vb = _mm512_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx512(vb);
        _mm512_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);

        let vh = _mm512_maskz_loadu_ps(mask, head_input.as_ptr().add(i));
        _mm512_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, _mm512_add_ps(vh, vt));
    }
}

/// Applies tanh in-place on block and overwrites head_input using AVX-512.
///
/// # Safety
///
/// - The CPU must support AVX-512F and AVX-512VL (guaranteed by the ISA
///   dispatch, which additionally requires BW+DQ; never call directly on an
///   unchecked host).
/// - `head_input.len() >= block.len()`: the unmasked vector loop performs
///   unaligned 512-bit raw-pointer stores up to `block.len()` on both slices;
///   a shorter `head_input` is written out of bounds (UB).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn tanh_and_overwrite_block_avx512(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    while i + 16 <= len {
        let vb = _mm512_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx512(vb);
        _mm512_storeu_ps(block.as_mut_ptr().add(i), vt);
        _mm512_storeu_ps(head_input.as_mut_ptr().add(i), vt);
        i += 16;
    }
    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let vb = _mm512_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx512(vb);
        _mm512_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);
        _mm512_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, vt);
    }
}

/// Fused Seed + Tanh + Head Accumulate using AVX-512.
///
/// Computes `head_input[i] = seed[i] + tanh(block[i])`.
/// Eliminates the separate `copy_from_slice(seed)` before `tanh_and_accumulate_block`.
///
/// # Safety
///
/// - The CPU must support AVX-512F and AVX-512VL (guaranteed by the ISA
///   dispatch, which additionally requires BW+DQ; never call directly on an
///   unchecked host).
/// - `head_input.len() >= block.len()` and `seed.len() >= block.len()`: the
///   unmasked vector loop performs unaligned 512-bit raw-pointer loads/stores
///   up to `block.len()` on all three slices; shorter slices are accessed out
///   of bounds (UB).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn tanh_and_accumulate_with_seed_avx512(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    let len = block.len();
    let mut i = 0;
    while i + 16 <= len {
        let vb = _mm512_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx512(vb);
        _mm512_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vs = _mm512_loadu_ps(seed.as_ptr().add(i));
        _mm512_storeu_ps(head_input.as_mut_ptr().add(i), _mm512_add_ps(vs, vt));
        i += 16;
    }
    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let vb = _mm512_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx512(vb);
        _mm512_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);

        let vs = _mm512_maskz_loadu_ps(mask, seed.as_ptr().add(i));
        _mm512_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, _mm512_add_ps(vs, vt));
    }
}

/// Applies ReLU in-place on block and accumulates into head_input using AVX-512.
///
/// # Safety
///
/// - The CPU must support AVX-512F and AVX-512VL (guaranteed by the ISA
///   dispatch, which additionally requires BW+DQ; never call directly on an
///   unchecked host).
/// - `head_input.len() >= block.len()`: the unmasked vector loop performs
///   unaligned 512-bit raw-pointer loads/stores up to `block.len()` on both
///   slices; a shorter `head_input` is accessed out of bounds (UB).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn relu_and_accumulate_block_avx512(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    while i + 16 <= len {
        let vb = _mm512_loadu_ps(block.as_ptr().add(i));
        let vt = _mm512_max_ps(vb, _mm512_setzero_ps());
        _mm512_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vh = _mm512_loadu_ps(head_input.as_ptr().add(i));
        _mm512_storeu_ps(head_input.as_mut_ptr().add(i), _mm512_add_ps(vh, vt));
        i += 16;
    }
    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let vb = _mm512_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = _mm512_max_ps(vb, _mm512_setzero_ps());
        _mm512_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);

        let vh = _mm512_maskz_loadu_ps(mask, head_input.as_ptr().add(i));
        _mm512_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, _mm512_add_ps(vh, vt));
    }
}

/// Applies ReLU in-place on block and overwrites head_input using AVX-512.
///
/// # Safety
///
/// - The CPU must support AVX-512F and AVX-512VL (guaranteed by the ISA
///   dispatch, which additionally requires BW+DQ; never call directly on an
///   unchecked host).
/// - `head_input.len() >= block.len()`: the unmasked vector loop performs
///   unaligned 512-bit raw-pointer stores up to `block.len()` on both slices;
///   a shorter `head_input` is written out of bounds (UB).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn relu_and_overwrite_block_avx512(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    while i + 16 <= len {
        let vb = _mm512_loadu_ps(block.as_ptr().add(i));
        let vt = _mm512_max_ps(vb, _mm512_setzero_ps());
        _mm512_storeu_ps(block.as_mut_ptr().add(i), vt);
        _mm512_storeu_ps(head_input.as_mut_ptr().add(i), vt);
        i += 16;
    }
    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let vb = _mm512_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = _mm512_max_ps(vb, _mm512_setzero_ps());
        _mm512_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);
        _mm512_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, vt);
    }
}

/// Fused Seed + ReLU + Head Accumulate using AVX-512.
///
/// # Safety
///
/// - The CPU must support AVX-512F and AVX-512VL (guaranteed by the ISA
///   dispatch, which additionally requires BW+DQ; never call directly on an
///   unchecked host).
/// - `head_input.len() >= block.len()` and `seed.len() >= block.len()`: the
///   unmasked vector loop performs unaligned 512-bit raw-pointer loads/stores
///   up to `block.len()` on all three slices; shorter slices are accessed out
///   of bounds (UB).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn relu_and_accumulate_with_seed_avx512(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    let len = block.len();
    let mut i = 0;
    while i + 16 <= len {
        let vb = _mm512_loadu_ps(block.as_ptr().add(i));
        let vt = _mm512_max_ps(vb, _mm512_setzero_ps());
        _mm512_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vs = _mm512_loadu_ps(seed.as_ptr().add(i));
        _mm512_storeu_ps(head_input.as_mut_ptr().add(i), _mm512_add_ps(vs, vt));
        i += 16;
    }
    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let vb = _mm512_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = _mm512_max_ps(vb, _mm512_setzero_ps());
        _mm512_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);

        let vs = _mm512_maskz_loadu_ps(mask, seed.as_ptr().add(i));
        _mm512_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, _mm512_add_ps(vs, vt));
    }
}
