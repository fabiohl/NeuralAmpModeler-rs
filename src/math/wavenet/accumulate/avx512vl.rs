// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 VL256 accumulation and activation kernels for WaveNet (256-bit EVEX with opmasks).
//!
//! Provides fused activation and accumulation using 256-bit EVEX registers (`__m256`)
//! and hardware opmasks (`__mmask8`). Eliminates scalar tail branches for arbitrary
//! channel widths (`CH=3, 4, 8, 12, 16, 24, 32`) and avoids ZMM frequency/thermal penalties.

use core::arch::x86_64::*;

/// Accumulates src into dest using AVX-512 VL256 with EVEX masked tail.
///
/// Computes `dest[i] += src[i]`.
///
/// # Safety
///
/// - The CPU must support AVX-512F, AVX-512VL, AVX-512BW and AVX-512DQ
///   (guaranteed by the ISA dispatch; never call directly on an unchecked
///   host).
/// - Slice geometry is self-clamped (`len = min(dest.len(), src.len())`) and
///   the tail uses fault-suppressing opmasks, so no length contract applies.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub unsafe fn accumulate_head_avx512vl(dest: &mut [f32], src: &[f32]) {
    let len = dest.len().min(src.len());
    let mut i = 0;
    while i + 16 <= len {
        let vs0 = _mm256_loadu_ps(src.as_ptr().add(i));
        let vs1 = _mm256_loadu_ps(src.as_ptr().add(i + 8));
        let vd0 = _mm256_loadu_ps(dest.as_ptr().add(i));
        let vd1 = _mm256_loadu_ps(dest.as_ptr().add(i + 8));
        _mm256_storeu_ps(dest.as_mut_ptr().add(i), _mm256_add_ps(vd0, vs0));
        _mm256_storeu_ps(dest.as_mut_ptr().add(i + 8), _mm256_add_ps(vd1, vs1));
        i += 16;
    }
    if i + 8 <= len {
        let vs = _mm256_loadu_ps(src.as_ptr().add(i));
        let vd = _mm256_loadu_ps(dest.as_ptr().add(i));
        _mm256_storeu_ps(dest.as_mut_ptr().add(i), _mm256_add_ps(vd, vs));
        i += 8;
    }
    if i < len {
        let rem = len - i;
        let mask = _cvtu32_mask8((1u32 << rem) - 1);
        let vs = _mm256_maskz_loadu_ps(mask, src.as_ptr().add(i));
        let vd = _mm256_maskz_loadu_ps(mask, dest.as_ptr().add(i));
        _mm256_mask_storeu_ps(dest.as_mut_ptr().add(i), mask, _mm256_add_ps(vd, vs));
    }
}

/// Applies tanh in-place on block and accumulates into head_input using AVX-512 VL256.
///
/// Computes `block[i] = tanh(block[i])` and `head_input[i] += block[i]`.
/// Processes 2 ymm vectors per iteration with masked tail handling.
///
/// # Safety
///
/// - The CPU must support AVX-512F, AVX-512VL, AVX-512BW and AVX-512DQ
///   (guaranteed by the ISA dispatch; never call directly on an unchecked
///   host).
/// - Slice geometry is self-clamped (`len = min(block.len(), head_input.len())`)
///   and the tail uses fault-suppressing opmasks, so no length contract applies.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub unsafe fn tanh_and_accumulate_block_avx512vl(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len().min(head_input.len());
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
    if i + 8 <= len {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vh = _mm256_loadu_ps(head_input.as_ptr().add(i));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vh, vt));
        i += 8;
    }
    if i < len {
        let rem = len - i;
        let mask = _cvtu32_mask8((1u32 << rem) - 1);
        let vb = _mm256_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);

        let vh = _mm256_maskz_loadu_ps(mask, head_input.as_ptr().add(i));
        _mm256_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, _mm256_add_ps(vh, vt));
    }
}

/// Applies tanh in-place on block and overwrites head_input using AVX-512 VL256.
///
/// Computes `block[i] = tanh(block[i])` and `head_input[i] = block[i]`.
///
/// # Safety
///
/// - The CPU must support AVX-512F, AVX-512VL, AVX-512BW and AVX-512DQ
///   (guaranteed by the ISA dispatch; never call directly on an unchecked
///   host).
/// - Slice geometry is self-clamped (`len = min(block.len(), head_input.len())`)
///   and the tail uses fault-suppressing opmasks, so no length contract applies.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub unsafe fn tanh_and_overwrite_block_avx512vl(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len().min(head_input.len());
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
    if i + 8 <= len {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), vt);
        i += 8;
    }
    if i < len {
        let rem = len - i;
        let mask = _cvtu32_mask8((1u32 << rem) - 1);
        let vb = _mm256_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);
        _mm256_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, vt);
    }
}

/// Fused Seed + Tanh + Head Accumulate using AVX-512 VL256.
///
/// Computes `head_input[i] = seed[i] + tanh(block[i])` and updates `block[i] = tanh(block[i])`.
///
/// # Safety
///
/// - The CPU must support AVX-512F, AVX-512VL, AVX-512BW and AVX-512DQ
///   (guaranteed by the ISA dispatch; never call directly on an unchecked
///   host).
/// - Slice geometry is self-clamped (`len = min(block, head_input, seed)`)
///   and the tail uses fault-suppressing opmasks, so no length contract applies.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub unsafe fn tanh_and_accumulate_with_seed_avx512vl(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    let len = block.len().min(head_input.len()).min(seed.len());
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
    if i + 8 <= len {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vs = _mm256_loadu_ps(seed.as_ptr().add(i));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vs, vt));
        i += 8;
    }
    if i < len {
        let rem = len - i;
        let mask = _cvtu32_mask8((1u32 << rem) - 1);
        let vb = _mm256_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_poly_avx2(vb);
        _mm256_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);

        let vs = _mm256_maskz_loadu_ps(mask, seed.as_ptr().add(i));
        _mm256_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, _mm256_add_ps(vs, vt));
    }
}

/// Applies gated activation (tanh * sigmoid) in-place on block and accumulates into head_input using AVX-512 VL256.
///
/// Handles arbitrary channel widths `ch` without scalar branches via EVEX opmasks.
///
/// # Safety
///
/// - The CPU must support AVX-512F, AVX-512VL, AVX-512BW and AVX-512DQ
///   (guaranteed by the ISA dispatch; never call directly on an unchecked
///   host).
/// - `block.len() >= 2 * ch * (head_input.len() / ch)`: the vector loop
///   performs unaligned 256-bit raw-pointer loads/stores at
///   `block[f * 2 * ch + {c, ch + c}]`; a shorter `block` is accessed out of
///   bounds (UB). `ch == 0` is handled (early return).
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub unsafe fn gated_activation_and_accumulate_block_avx512vl(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    if ch == 0 {
        return;
    }
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        while c + 16 <= ch {
            let z1_0 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2_0 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));
            let z1_1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c + 8));
            let z2_1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c + 8));

            let (tanh_z1_0, sig_z2_0) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1_0, z2_0);
            let (tanh_z1_1, sig_z2_1) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1_1, z2_1);

            let act0 = _mm256_mul_ps(tanh_z1_0, sig_z2_0);
            let act1 = _mm256_mul_ps(tanh_z1_1, sig_z2_1);

            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), act0);
            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c + 8), act1);

            let vh0 = _mm256_loadu_ps(head_input.as_ptr().add(head_offset + c));
            let vh1 = _mm256_loadu_ps(head_input.as_ptr().add(head_offset + c + 8));
            _mm256_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c),
                _mm256_add_ps(vh0, act0),
            );
            _mm256_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c + 8),
                _mm256_add_ps(vh1, act1),
            );
            c += 16;
        }
        if c + 8 <= ch {
            let z1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
            let act = _mm256_mul_ps(tanh_z1, sig_z2);

            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), act);

            let vh = _mm256_loadu_ps(head_input.as_ptr().add(head_offset + c));
            _mm256_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c),
                _mm256_add_ps(vh, act),
            );
            c += 8;
        }
        if c < ch {
            let rem = ch - c;
            let mask = _cvtu32_mask8((1u32 << rem) - 1);
            let z1 = _mm256_maskz_loadu_ps(mask, block.as_ptr().add(block_offset + c));
            let z2 = _mm256_maskz_loadu_ps(mask, block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
            let act = _mm256_mul_ps(tanh_z1, sig_z2);

            _mm256_mask_storeu_ps(block.as_mut_ptr().add(block_offset + c), mask, act);

            let vh = _mm256_maskz_loadu_ps(mask, head_input.as_ptr().add(head_offset + c));
            _mm256_mask_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c),
                mask,
                _mm256_add_ps(vh, act),
            );
        }
    }
}

/// Applies gated activation (tanh * sigmoid) in-place on block and overwrites head_input using AVX-512 VL256.
///
/// # Safety
///
/// - The CPU must support AVX-512F, AVX-512VL, AVX-512BW and AVX-512DQ
///   (guaranteed by the ISA dispatch; never call directly on an unchecked
///   host).
/// - `block.len() >= 2 * ch * (head_input.len() / ch)`: the vector loop
///   performs unaligned 256-bit raw-pointer loads/stores at
///   `block[f * 2 * ch + {c, ch + c}]`; a shorter `block` is accessed out of
///   bounds (UB). `ch == 0` is handled (early return).
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub unsafe fn gated_activation_and_overwrite_block_avx512vl(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    if ch == 0 {
        return;
    }
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        while c + 16 <= ch {
            let z1_0 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2_0 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));
            let z1_1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c + 8));
            let z2_1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c + 8));

            let (tanh_z1_0, sig_z2_0) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1_0, z2_0);
            let (tanh_z1_1, sig_z2_1) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1_1, z2_1);

            let act0 = _mm256_mul_ps(tanh_z1_0, sig_z2_0);
            let act1 = _mm256_mul_ps(tanh_z1_1, sig_z2_1);

            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), act0);
            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c + 8), act1);

            _mm256_storeu_ps(head_input.as_mut_ptr().add(head_offset + c), act0);
            _mm256_storeu_ps(head_input.as_mut_ptr().add(head_offset + c + 8), act1);
            c += 16;
        }
        if c + 8 <= ch {
            let z1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
            let act = _mm256_mul_ps(tanh_z1, sig_z2);

            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), act);
            _mm256_storeu_ps(head_input.as_mut_ptr().add(head_offset + c), act);
            c += 8;
        }
        if c < ch {
            let rem = ch - c;
            let mask = _cvtu32_mask8((1u32 << rem) - 1);
            let z1 = _mm256_maskz_loadu_ps(mask, block.as_ptr().add(block_offset + c));
            let z2 = _mm256_maskz_loadu_ps(mask, block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
            let act = _mm256_mul_ps(tanh_z1, sig_z2);

            _mm256_mask_storeu_ps(block.as_mut_ptr().add(block_offset + c), mask, act);
            _mm256_mask_storeu_ps(head_input.as_mut_ptr().add(head_offset + c), mask, act);
        }
    }
}

/// Applies ReLU in-place on block and accumulates into head_input using AVX-512 VL256.
///
/// # Safety
///
/// - The CPU must support AVX-512F, AVX-512VL, AVX-512BW and AVX-512DQ
///   (guaranteed by the ISA dispatch; never call directly on an unchecked
///   host).
/// - Slice geometry is self-clamped (`len = min(block.len(), head_input.len())`)
///   and the tail uses fault-suppressing opmasks, so no length contract applies.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub unsafe fn relu_and_accumulate_block_avx512vl(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len().min(head_input.len());
    let mut i = 0;
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
    if i + 8 <= len {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = _mm256_max_ps(vb, zero);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vh = _mm256_loadu_ps(head_input.as_ptr().add(i));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vh, vt));
        i += 8;
    }
    if i < len {
        let rem = len - i;
        let mask = _cvtu32_mask8((1u32 << rem) - 1);
        let vb = _mm256_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = _mm256_max_ps(vb, zero);
        _mm256_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);

        let vh = _mm256_maskz_loadu_ps(mask, head_input.as_ptr().add(i));
        _mm256_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, _mm256_add_ps(vh, vt));
    }
}

/// Applies ReLU in-place on block and overwrites head_input using AVX-512 VL256.
///
/// # Safety
///
/// - The CPU must support AVX-512F, AVX-512VL, AVX-512BW and AVX-512DQ
///   (guaranteed by the ISA dispatch; never call directly on an unchecked
///   host).
/// - Slice geometry is self-clamped (`len = min(block.len(), head_input.len())`)
///   and the tail uses fault-suppressing opmasks, so no length contract applies.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub unsafe fn relu_and_overwrite_block_avx512vl(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len().min(head_input.len());
    let mut i = 0;
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
    if i + 8 <= len {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = _mm256_max_ps(vb, zero);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), vt);
        i += 8;
    }
    if i < len {
        let rem = len - i;
        let mask = _cvtu32_mask8((1u32 << rem) - 1);
        let vb = _mm256_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = _mm256_max_ps(vb, zero);
        _mm256_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);
        _mm256_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, vt);
    }
}

/// Fused Seed + ReLU + Head Accumulate using AVX-512 VL256.
///
/// Computes `head_input[i] = seed[i] + max(0.0, block[i])` and `block[i] = max(0.0, block[i])`.
///
/// # Safety
///
/// - The CPU must support AVX-512F, AVX-512VL, AVX-512BW and AVX-512DQ
///   (guaranteed by the ISA dispatch; never call directly on an unchecked
///   host).
/// - Slice geometry is self-clamped (`len = min(block, head_input, seed)`)
///   and the tail uses fault-suppressing opmasks, so no length contract applies.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub unsafe fn relu_and_accumulate_with_seed_avx512vl(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    let len = block.len().min(head_input.len()).min(seed.len());
    let mut i = 0;
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
    if i + 8 <= len {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = _mm256_max_ps(vb, zero);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vs = _mm256_loadu_ps(seed.as_ptr().add(i));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vs, vt));
        i += 8;
    }
    if i < len {
        let rem = len - i;
        let mask = _cvtu32_mask8((1u32 << rem) - 1);
        let vb = _mm256_maskz_loadu_ps(mask, block.as_ptr().add(i));
        let vt = _mm256_max_ps(vb, zero);
        _mm256_mask_storeu_ps(block.as_mut_ptr().add(i), mask, vt);

        let vs = _mm256_maskz_loadu_ps(mask, seed.as_ptr().add(i));
        _mm256_mask_storeu_ps(head_input.as_mut_ptr().add(i), mask, _mm256_add_ps(vs, vt));
    }
}
