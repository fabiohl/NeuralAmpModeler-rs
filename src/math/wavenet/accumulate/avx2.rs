// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX2 accumulation and activation kernels for WaveNet.

use crate::arch::simd::*;
use crate::wavenet_simd_avx2;

/// Builds an AVX2 tail mask with the low `rem` lanes enabled (`-1`) and the rest zero.
///
/// Lets remainder paths (`rem < 8`) run the full 256-bit vector kernels without
/// scalar fallback, heap allocation, or libm calls. Stack-resident only; the
/// mask lives in a register after load.
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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

/// Full 8-wide channel vectors of one frame: activation + accumulate/overwrite.
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn gated_full_vectors<const OVERWRITE: bool>(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
    f: usize,
    vec_ch: usize,
) {
    let block_offset = f * 2 * ch;
    let head_offset = f * ch;
    let mut c = 0;
    // SAFETY: `ch >= 1` and `block.len() >= 2 * ch * num_frames`; the loop
    // guard keeps every 8-lane load/store inside `..vec_ch <= ch`.
    unsafe {
        wavenet_simd_avx2!(c, vec_ch, {
            let z1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));
            let (tanh_z1, sig_z2) =
                crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
            let activated = _mm256_mul_ps(tanh_z1, sig_z2);
            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);
            if OVERWRITE {
                _mm256_storeu_ps(head_input.as_mut_ptr().add(head_offset + c), activated);
            } else {
                let vh = _mm256_loadu_ps(head_input.as_ptr().add(head_offset + c));
                _mm256_storeu_ps(
                    head_input.as_mut_ptr().add(head_offset + c),
                    _mm256_add_ps(vh, activated),
                );
            }
        });
    }
}

/// Remainder of 4 channels for one frame, zero-extended into a full YMM.
///
/// The disabled lanes are zero, exactly like the previous masked tail, so the
/// low four lanes keep the same arithmetic and bit patterns.
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn gated_rem4_single<const OVERWRITE: bool>(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
    f: usize,
) {
    let start_c = ch - 4;
    let block_offset = f * 2 * ch;
    let head_offset = f * ch;
    // SAFETY: caller guarantees `ch >= 4` and `block.len() >= 2 * ch * num_frames`;
    // all four-lane accesses stay inside the frame's channel window.
    unsafe {
        let z1 = _mm256_insertf128_ps(
            _mm256_setzero_ps(),
            _mm_loadu_ps(block.as_ptr().add(block_offset + start_c)),
            0,
        );
        let z2 = _mm256_insertf128_ps(
            _mm256_setzero_ps(),
            _mm_loadu_ps(block.as_ptr().add(block_offset + ch + start_c)),
            0,
        );
        let (tanh_z1, sig_z2) = crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
        let activated = _mm256_mul_ps(tanh_z1, sig_z2);
        let low = _mm256_castps256_ps128(activated);
        _mm_storeu_ps(block.as_mut_ptr().add(block_offset + start_c), low);
        if OVERWRITE {
            _mm_storeu_ps(head_input.as_mut_ptr().add(head_offset + start_c), low);
        } else {
            let vh = _mm_loadu_ps(head_input.as_ptr().add(head_offset + start_c));
            _mm_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + start_c),
                _mm_add_ps(vh, low),
            );
        }
    }
}

/// Remainder of 4 channels for two consecutive frames packed into one YMM.
///
/// Lanes 0..3 activate frame `f`, lanes 4..7 activate frame `f+1`; the
/// activation kernel is lane-local, so each lane computes the same value as
/// the former per-frame masked tail.
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn gated_rem4_pair<const OVERWRITE: bool>(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
    f: usize,
) {
    let start_c = ch - 4;
    let block0 = f * 2 * ch;
    let block1 = (f + 1) * 2 * ch;
    let head0 = f * ch + start_c;
    let head1 = (f + 1) * ch + start_c;
    // SAFETY: caller guarantees `f + 1 < num_frames`, `ch >= 4` and
    // `block.len() >= 2 * ch * num_frames`; each four-lane load stays inside
    // its frame and the head offsets are one frame apart.
    unsafe {
        let z1 = _mm256_insertf128_ps(
            _mm256_insertf128_ps(
                _mm256_setzero_ps(),
                _mm_loadu_ps(block.as_ptr().add(block0 + start_c)),
                0,
            ),
            _mm_loadu_ps(block.as_ptr().add(block1 + start_c)),
            1,
        );
        let z2 = _mm256_insertf128_ps(
            _mm256_insertf128_ps(
                _mm256_setzero_ps(),
                _mm_loadu_ps(block.as_ptr().add(block0 + ch + start_c)),
                0,
            ),
            _mm_loadu_ps(block.as_ptr().add(block1 + ch + start_c)),
            1,
        );
        let (tanh_z1, sig_z2) = crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
        let activated = _mm256_mul_ps(tanh_z1, sig_z2);
        let low = _mm256_castps256_ps128(activated);
        let high = _mm256_extractf128_ps(activated, 1);

        _mm_storeu_ps(block.as_mut_ptr().add(block0 + start_c), low);
        _mm_storeu_ps(block.as_mut_ptr().add(block1 + start_c), high);

        if OVERWRITE {
            _mm_storeu_ps(head_input.as_mut_ptr().add(head0), low);
            _mm_storeu_ps(head_input.as_mut_ptr().add(head1), high);
        } else {
            let h0 = _mm_loadu_ps(head_input.as_ptr().add(head0));
            let h1 = _mm_loadu_ps(head_input.as_ptr().add(head1));
            _mm_storeu_ps(head_input.as_mut_ptr().add(head0), _mm_add_ps(h0, low));
            _mm_storeu_ps(head_input.as_mut_ptr().add(head1), _mm_add_ps(h1, high));
        }
    }
}

/// Copies `rem` (1..=7) `f32` with a compile-time-sized move in each arm.
///
/// A runtime-length `copy_nonoverlapping` is lowered to an out-of-line
/// `memcpy`; the fixed arms keep these small tail moves in registers/stack.
#[inline]
unsafe fn copy_f32_rem(src: *const f32, dst: *mut f32, rem: usize) {
    // SAFETY: the caller guarantees `rem <= 7` and that both pointers are
    // valid for `rem` reads/writes.
    unsafe {
        match rem {
            1 => core::ptr::copy_nonoverlapping(src, dst, 1),
            2 => core::ptr::copy_nonoverlapping(src, dst, 2),
            3 => core::ptr::copy_nonoverlapping(src, dst, 3),
            4 => core::ptr::copy_nonoverlapping(src, dst, 4),
            5 => core::ptr::copy_nonoverlapping(src, dst, 5),
            6 => core::ptr::copy_nonoverlapping(src, dst, 6),
            7 => core::ptr::copy_nonoverlapping(src, dst, 7),
            _ => {}
        }
    }
}

/// Generic remainder (`rem != 4`): the unused lanes are zero-filled on the stack
/// and activated in one full YMM, then only `rem` lanes are written back.
///
/// Zero-filling is exactly what `maskload` did to the disabled lanes, so the
/// low-lane arithmetic is unchanged. It also removes `vmaskmovps` from the
/// gated path entirely (the emulated gather is expensive on aarch64).
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn gated_rem_generic<const OVERWRITE: bool>(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
    f: usize,
    start_c: usize,
) {
    let block_offset = f * 2 * ch;
    let head_offset = f * ch;
    let rem = ch - start_c;
    let mut z1_buf = [0.0f32; 8];
    let mut z2_buf = [0.0f32; 8];
    let mut out = [0.0f32; 8];
    // SAFETY: caller guarantees `ch >= 1`, `block.len() >= 2 * ch * num_frames`
    // and `start_c <= ch`, so `rem <= 7` and every copy stays inside its frame.
    unsafe {
        copy_f32_rem(
            block.as_ptr().add(block_offset + start_c),
            z1_buf.as_mut_ptr(),
            rem,
        );
        copy_f32_rem(
            block.as_ptr().add(block_offset + ch + start_c),
            z2_buf.as_mut_ptr(),
            rem,
        );
        let z1 = _mm256_loadu_ps(z1_buf.as_ptr());
        let z2 = _mm256_loadu_ps(z2_buf.as_ptr());
        let (tanh_z1, sig_z2) = crate::math::activations::simd_tanh_sigmoid_dual_poly_avx2(z1, z2);
        let activated = _mm256_mul_ps(tanh_z1, sig_z2);
        _mm256_storeu_ps(out.as_mut_ptr(), activated);
        copy_f32_rem(
            out.as_ptr(),
            block.as_mut_ptr().add(block_offset + start_c),
            rem,
        );
        if OVERWRITE {
            copy_f32_rem(
                out.as_ptr(),
                head_input.as_mut_ptr().add(head_offset + start_c),
                rem,
            );
        } else {
            for lane in 0..rem {
                let dst = head_input.get_unchecked_mut(head_offset + start_c + lane);
                *dst += *out.get_unchecked(lane);
            }
        }
    }
}

/// Shared gated-activation traversal for accumulate (`OVERWRITE = false`) and
/// overwrite (`OVERWRITE = true`).
///
/// `CH = 4` and the `CH = 12` remainder run two frames per YMM: one full
/// activation instead of two masked ones. Other remainders zero-fill the
/// unused lanes on the stack, which keeps the lane arithmetic and removes
/// `vmaskmovps` from this path.
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn gated_activation_block<const OVERWRITE: bool>(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    let vec_ch = ch & !7;
    let rem = ch - vec_ch;
    let mut f = 0;
    // SAFETY: every helper keeps its loads/stores inside the frame window
    // `2 * ch` of `block` and `ch` of `head_input`; `f < num_frames` throughout.
    unsafe {
        if rem == 4 {
            while f + 2 <= num_frames {
                gated_full_vectors::<OVERWRITE>(head_input, block, ch, f, vec_ch);
                gated_full_vectors::<OVERWRITE>(head_input, block, ch, f + 1, vec_ch);
                gated_rem4_pair::<OVERWRITE>(head_input, block, ch, f);
                f += 2;
            }
            if f < num_frames {
                gated_full_vectors::<OVERWRITE>(head_input, block, ch, f, vec_ch);
                gated_rem4_single::<OVERWRITE>(head_input, block, ch, f);
            }
        } else {
            while f < num_frames {
                gated_full_vectors::<OVERWRITE>(head_input, block, ch, f, vec_ch);
                if rem != 0 {
                    gated_rem_generic::<OVERWRITE>(head_input, block, ch, f, vec_ch);
                }
                f += 1;
            }
        }
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub unsafe fn gated_activation_and_accumulate_block_avx2(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    // SAFETY: the helper enforces the same channel geometry documented above.
    unsafe {
        gated_activation_block::<false>(head_input, block, ch);
    }
}

/// Masked tail for tanh + overwrite: polynomial tanh, in-place store.
///
/// Runs the same `simd_tanh_poly_avx2` kernel as the vector loop under an AVX2
/// mask, so the remainder lanes activate identically to full vectors.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
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

/// Applies gated activation (tanh * sigmoid) in-place on block and overwrites head_input using AVX2.
///
/// # Safety
///
/// - The CPU must support AVX2 and FMA (guaranteed by the ISA dispatch).
/// - `ch >= 1` and `block.len() >= 2 * ch * (head_input.len() / ch)`: the
///   vector loop performs unaligned 256-bit raw-pointer loads/stores at
///   `block[f * 2 * ch + {c, ch + c}]`; a shorter `block` is accessed out of
///   bounds (UB). `ch == 0` also divides by zero.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub unsafe fn gated_activation_and_overwrite_block_avx2(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    // SAFETY: the helper enforces the same channel geometry documented above.
    unsafe {
        gated_activation_block::<true>(head_input, block, ch);
    }
}
