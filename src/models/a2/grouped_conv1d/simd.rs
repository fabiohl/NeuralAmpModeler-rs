// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

// SIMD kernel implementation using specialized parameter lists.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! SIMD (AVX2/FMA, NEON) kernels for grouped dilated causal Conv1D.
//!
//! The depthwise kernel reads the tap-major SoA tail (`[kernel][channel]`)
//! appended to [`A2GroupedConv1d::weights`] at load time. It replaces the
//! per-tap 8-lane `_mm256_i32gather_ps` (scalar-emulated on aarch64 and slow on
//! Zen/Intel) with contiguous vector loads. The general grouped kernel keeps
//! the interleaved-4 layout, which already loads each 4-lane block directly.

use super::A2GroupedConv1d;
#[cfg(target_arch = "x86_64")]
use crate::math::common::{prefetch_strategy_2stage, prefetch_strategy_simple};
use crate::models::wavenet::common::MAX_KERNEL;

#[cfg(target_arch = "x86_64")]
use crate::arch::simd::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn load_mixin_4(
    mixin: Option<&[f32]>,
    out_c: usize,
    out_ch: usize,
) -> (f32, f32, f32, f32) {
    // SAFETY-CRITICAL: this precondition gates unchecked slice access
    // (`get_unchecked`) below. Must be `assert!`, not `debug_assert!` — the
    // latter is elided in release builds, turning an out-of-bounds `mixin`
    // into silent UB (out-of-bounds read) instead of a clean panic.
    assert!(
        mixin.is_none_or(|m| m.len() >= out_ch),
        "load_mixin_4: mixin len {:?} < out_ch {}",
        mixin.map(|m| m.len()),
        out_ch
    );
    if let Some(m) = mixin {
        if out_c + 3 < out_ch {
            (
                *m.get_unchecked(out_c),
                *m.get_unchecked(out_c + 1),
                *m.get_unchecked(out_c + 2),
                *m.get_unchecked(out_c + 3),
            )
        } else {
            let mut v = [0.0f32; 4];
            for (i, val) in v.iter_mut().enumerate() {
                if out_c + i < out_ch {
                    *val = *m.get_unchecked(out_c + i);
                }
            }
            (v[0], v[1], v[2], v[3])
        }
    } else {
        (0.0, 0.0, 0.0, 0.0)
    }
}

/// Builds the tap pointer table once per frame.
///
/// # Safety
/// Same contract as the kernels: `layer_buffer` must cover every dilated tap
/// of `frame_idx`, and `kernel <= MAX_KERNEL`.
#[inline(always)]
unsafe fn build_tap_ptrs(
    layer_buffer: &[f32],
    frame_idx: usize,
    in_ch: usize,
    kernel: usize,
    dilation: usize,
) -> [*const f32; MAX_KERNEL] {
    let mut tap_ptrs = [core::ptr::null::<f32>(); MAX_KERNEL];
    let k_limit = kernel.min(MAX_KERNEL);
    for (k, tap) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
        let offset = (dilation as isize) * ((k as isize) + 1 - (kernel as isize));
        let in_start = ((frame_idx as isize) + offset) as usize * in_ch;
        // SAFETY: the caller's assertions prove the tap base is in bounds.
        *tap = unsafe { layer_buffer.as_ptr().add(in_start) };
        // x86 prefetch policy was measured there; it is a scheduling barrier
        // on aarch64 (`prfm` inline asm) and is not emitted in these kernels.
        #[cfg(target_arch = "x86_64")]
        {
            let step = dilation * in_ch;
            if dilation >= 128 {
                prefetch_strategy_2stage(*tap, step, k, kernel, dilation);
            } else {
                prefetch_strategy_simple(*tap, step, k, kernel, dilation);
            }
        }
    }
    tap_ptrs
}

/// Depthwise convolution optimized path (groups == in_ch == out_ch).
///
/// When every channel is its own group, each output channel is a single-tap
/// convolution over the matching input channel. The tap-major depthwise
/// layout (`[kernel][channel]`) makes eight consecutive channel weights a
/// single contiguous load.
///
/// # Safety
/// Only call when `groups == in_ch == out_ch`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub(crate) unsafe fn process_single_frame_depthwise_avx2(
    conv: &A2GroupedConv1d,
    layer_buffer: &[f32],
    out_frame: &mut [f32],
    frame_idx: usize,
) {
    debug_assert_eq!(conv.groups, conv.in_ch);
    debug_assert_eq!(conv.groups, conv.out_ch);
    debug_assert_eq!(conv.in_per_group, 1);
    debug_assert_eq!(conv.out_per_group, 1);
    // SAFETY-CRITICAL: `frame_idx`/`layer_buffer` gate raw pointer arithmetic
    // (`layer_buffer.as_ptr().add(in_start)` below, with `in_start` derived
    // from `frame_idx` via signed-then-unsigned arithmetic that wraps to a
    // huge offset if `frame_idx` is too small). Must be `assert!`, not
    // `debug_assert!` — eliding this check in release does not fail cleanly,
    // it produces an out-of-bounds pointer dereference (confirmed SIGSEGV in
    // the sibling `grouped_conv1d_single_frame_simd`, same pattern.
    assert!(
        out_frame.len() >= conv.out_ch,
        "depthwise: out_frame len {} < out_ch {}",
        out_frame.len(),
        conv.out_ch
    );
    assert!(
        frame_idx >= conv.dilation * (conv.kernel.saturating_sub(1)),
        "depthwise: frame_idx {} < lookback {}",
        frame_idx,
        conv.dilation * (conv.kernel.saturating_sub(1))
    );
    assert!(
        layer_buffer.len() > frame_idx * conv.in_ch,
        "depthwise: layer_buffer len {} <= frame_idx {} * in_ch {}",
        layer_buffer.len(),
        frame_idx,
        conv.in_ch
    );

    // F-RES2-10: the 8-lane loads below read up to element
    // `(frame_idx + 1) * in_ch - 1` of `layer_buffer` (last tap = current frame
    // at offset `frame_idx * in_ch`, last lane of the `ch8 = in_ch & !7`
    // block), while the release `assert!` above only proves the tap *base* is
    // in bounds. Production callers guarantee the full-row extent by
    // construction (layer-state buffers are sized as `frames * in_ch` with
    // `frame_idx < frames`); this debug-only net checks exactly that limit and
    // is compiled out of release — zero hot-path cost.
    debug_assert!(
        layer_buffer.len() >= (frame_idx + 1) * conv.in_ch,
        "depthwise: layer_buffer len {} < (frame_idx {} + 1) * in_ch {} \
         (8-lane load upper bound)",
        layer_buffer.len(),
        frame_idx,
        conv.in_ch
    );

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the release assertions above verify `out_frame` length,
        // `frame_idx` lookback, and `layer_buffer` tap bases; the caller
        // passes a depthwise `conv` (`groups == in_ch == out_ch`), and NEON
        // is part of the `aarch64` baseline.
        unsafe {
            process_single_frame_depthwise_neon(conv, layer_buffer, out_frame, frame_idx);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        let ch = conv.in_ch;
        let kernel = conv.kernel;
        // SAFETY: the release assertions above guarantee every tap of
        // `frame_idx` is in bounds and `kernel <= MAX_KERNEL`.
        let tap_ptrs =
            unsafe { build_tap_ptrs(layer_buffer, frame_idx, ch, kernel, conv.dilation) };

        // Depthwise layout: `[kernel][channel]`, appended after the
        // interleaved general block.
        let soa_base = conv.total_blocks * 4 * conv.in_per_group * conv.kernel;
        let w_ptr = conv.weights.as_ptr().add(soa_base);
        let bias_ptr = conv.bias.as_ptr();
        let ch8 = ch & !7;

        let mut c = 0;
        while c < ch8 {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut k = 0;
            while k + 1 < kernel {
                let tap0 = *tap_ptrs.get_unchecked(k);
                let wv0 = _mm256_loadu_ps(w_ptr.add(k * ch + c));
                let sv0 = _mm256_loadu_ps(tap0.add(c));
                acc0 = _mm256_fmadd_ps(wv0, sv0, acc0);

                let tap1 = *tap_ptrs.get_unchecked(k + 1);
                let wv1 = _mm256_loadu_ps(w_ptr.add((k + 1) * ch + c));
                let sv1 = _mm256_loadu_ps(tap1.add(c));
                acc1 = _mm256_fmadd_ps(wv1, sv1, acc1);

                k += 2;
            }
            if k < kernel {
                let tap = *tap_ptrs.get_unchecked(k);
                let wv = _mm256_loadu_ps(w_ptr.add(k * ch + c));
                let sv = _mm256_loadu_ps(tap.add(c));
                acc0 = _mm256_fmadd_ps(wv, sv, acc0);
            }
            let mut acc = _mm256_add_ps(acc0, acc1);

            if conv.do_bias {
                let bv = _mm256_loadu_ps(bias_ptr.add(c));
                acc = _mm256_add_ps(acc, bv);
            }

            _mm256_storeu_ps(out_frame.as_mut_ptr().add(c), acc);
            c += 8;
        }

        for c in c..ch {
            let mut acc = 0.0f32;
            for k in 0..kernel {
                let tap = *tap_ptrs.get_unchecked(k);
                let w = *w_ptr.add(k * ch + c);
                let s = *tap.add(c);
                acc += w * s;
            }

            if conv.do_bias {
                acc += *conv.bias.get_unchecked(c);
            }

            *out_frame.get_unchecked_mut(c) = acc;
        }
    }
}

/// aarch64 depthwise kernel: four independent FMA chains (low/high half of
/// the 8-lane block, even/odd tap trees), identical reduction order to the
/// x86 kernel (`even + odd`, then bias).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn process_single_frame_depthwise_neon(
    conv: &A2GroupedConv1d,
    layer_buffer: &[f32],
    out_frame: &mut [f32],
    frame_idx: usize,
) {
    use core::arch::aarch64::*;

    let ch = conv.in_ch;
    let kernel = conv.kernel;
    // SAFETY: the caller's release assertions guarantee every tap of
    // `frame_idx` is in bounds and `kernel <= MAX_KERNEL`.
    let tap_ptrs = unsafe { build_tap_ptrs(layer_buffer, frame_idx, ch, kernel, conv.dilation) };

    // Depthwise layout: `[kernel][channel]`, appended after the interleaved
    // general block.
    let soa_base = conv.total_blocks * 4 * conv.in_per_group * conv.kernel;
    let w_ptr = conv.weights.as_ptr().add(soa_base);
    let bias_ptr = conv.bias.as_ptr();
    let out_ptr = out_frame.as_mut_ptr();
    let ch8 = ch & !7;

    let mut c = 0;
    while c < ch8 {
        let mut even_lo = vdupq_n_f32(0.0);
        let mut even_hi = vdupq_n_f32(0.0);
        let mut odd_lo = vdupq_n_f32(0.0);
        let mut odd_hi = vdupq_n_f32(0.0);
        let mut k = 0;
        while k + 1 < kernel {
            let tap0 = *tap_ptrs.get_unchecked(k);
            let w0_lo = vld1q_f32(w_ptr.add(k * ch + c));
            let w0_hi = vld1q_f32(w_ptr.add(k * ch + c + 4));
            let s0_lo = vld1q_f32(tap0.add(c));
            let s0_hi = vld1q_f32(tap0.add(c + 4));
            even_lo = vfmaq_f32(even_lo, w0_lo, s0_lo);
            even_hi = vfmaq_f32(even_hi, w0_hi, s0_hi);

            let tap1 = *tap_ptrs.get_unchecked(k + 1);
            let w1_lo = vld1q_f32(w_ptr.add((k + 1) * ch + c));
            let w1_hi = vld1q_f32(w_ptr.add((k + 1) * ch + c + 4));
            let s1_lo = vld1q_f32(tap1.add(c));
            let s1_hi = vld1q_f32(tap1.add(c + 4));
            odd_lo = vfmaq_f32(odd_lo, w1_lo, s1_lo);
            odd_hi = vfmaq_f32(odd_hi, w1_hi, s1_hi);

            k += 2;
        }
        if k < kernel {
            let tap = *tap_ptrs.get_unchecked(k);
            let w_lo = vld1q_f32(w_ptr.add(k * ch + c));
            let w_hi = vld1q_f32(w_ptr.add(k * ch + c + 4));
            let s_lo = vld1q_f32(tap.add(c));
            let s_hi = vld1q_f32(tap.add(c + 4));
            even_lo = vfmaq_f32(even_lo, w_lo, s_lo);
            even_hi = vfmaq_f32(even_hi, w_hi, s_hi);
        }

        let mut acc_lo = vaddq_f32(even_lo, odd_lo);
        let mut acc_hi = vaddq_f32(even_hi, odd_hi);
        if conv.do_bias {
            acc_lo = vaddq_f32(acc_lo, vld1q_f32(bias_ptr.add(c)));
            acc_hi = vaddq_f32(acc_hi, vld1q_f32(bias_ptr.add(c + 4)));
        }
        vst1q_f32(out_ptr.add(c), acc_lo);
        vst1q_f32(out_ptr.add(c + 4), acc_hi);
        c += 8;
    }

    for c in c..ch {
        let mut acc = 0.0f32;
        for k in 0..kernel {
            let tap = *tap_ptrs.get_unchecked(k);
            let w = *w_ptr.add(k * ch + c);
            let s = *tap.add(c);
            acc += w * s;
        }
        if conv.do_bias {
            acc += *conv.bias.get_unchecked(c);
        }
        *out_frame.get_unchecked_mut(c) = acc;
    }
}

/// Grouped dilated conv — one frame, `groups > 1` (and not depthwise).
///
/// # Safety
/// - `out_frame.len() >= conv.out_ch` (release-asserted inside).
/// - `frame_idx >= conv.dilation * (conv.kernel - 1)` (release-asserted;
///   the warm-up invariant, owned release-stable by the layer-state
///   construction).
/// - `layer_buffer` must cover the `in_ch`-wide lookback window for every tap
///   of `frame_idx` (release-asserted tap bases plus a debug-only
///   `(frame_idx + 1) * in_ch` upper-bound net, F-RES2-10) and outlive the call.
/// - `mixin` must be `None` or hold at least `conv.out_ch` elements
///   (release-asserted).
/// - The CPU must support AVX2+FMA / NEON (compile-time baseline).
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub unsafe fn grouped_conv1d_single_frame_simd(
    conv: &A2GroupedConv1d,
    layer_buffer: &[f32],
    out_frame: &mut [f32],
    frame_idx: usize,
    mixin: Option<&[f32]>,
) {
    // SAFETY-CRITICAL: `frame_idx`/`layer_buffer` gate raw pointer arithmetic
    // (`layer_buffer.as_ptr().add(in_start)` below, with `in_start` derived
    // from `frame_idx` via signed-then-unsigned arithmetic that wraps to a
    // huge offset if `frame_idx` is too small). Must be `assert!`, not
    // `debug_assert!` — confirmed to cause a SIGSEGV in release when this
    // check was elided.
    assert!(
        out_frame.len() >= conv.out_ch,
        "simd: out_frame len {} < out_ch {}",
        out_frame.len(),
        conv.out_ch
    );
    assert!(
        frame_idx >= conv.dilation * (conv.kernel.saturating_sub(1)),
        "simd: frame_idx {} < lookback {}",
        frame_idx,
        conv.dilation * (conv.kernel.saturating_sub(1))
    );
    assert!(
        layer_buffer.len() > frame_idx * conv.in_ch,
        "simd: layer_buffer len {} <= frame_idx {} * in_ch {}",
        layer_buffer.len(),
        frame_idx,
        conv.in_ch
    );
    assert!(
        mixin.is_none_or(|m| m.len() >= conv.out_ch),
        "simd: mixin len {:?} < out_ch {}",
        mixin.map(|m| m.len()),
        conv.out_ch
    );

    // F-RES2-10: the per-lane reads `*tap.add(group_in_start + ic)` reach up
    // to element `(frame_idx + 1) * in_ch - 1` of `layer_buffer` (last tap =
    // current frame, last group's last channel), while the release `assert!`
    // above only proves the tap *base* is in bounds. Production callers
    // guarantee the full-row extent by construction (layer-state buffers are
    // sized as `frames * in_ch` with `frame_idx < frames`); this debug-only
    // net checks exactly that limit and is compiled out of release.
    debug_assert!(
        layer_buffer.len() >= (frame_idx + 1) * conv.in_ch,
        "simd: layer_buffer len {} < (frame_idx {} + 1) * in_ch {} (channel row bound)",
        layer_buffer.len(),
        frame_idx,
        conv.in_ch
    );

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the release assertions above verify `out_frame` length,
        // `frame_idx` lookback, `layer_buffer` tap bases, and `mixin` length;
        // NEON is part of the `aarch64` baseline.
        unsafe {
            grouped_conv1d_single_frame_neon(conv, layer_buffer, out_frame, frame_idx, mixin);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        let in_per_group = conv.in_per_group;
        let kernel = conv.kernel;
        let dilation = conv.dilation;
        let num_blocks_per_group = conv.num_blocks_per_group;

        let mut tap_ptrs = [core::ptr::null::<f32>(); MAX_KERNEL];
        let k_limit = kernel.min(MAX_KERNEL);
        for (k, tap) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
            let offset = (dilation as isize) * ((k as isize) + 1 - (kernel as isize));
            let in_start = ((frame_idx as isize) + offset) as usize * conv.in_ch;
            // SAFETY: the release asserts above prove the tap base is in bounds.
            *tap = unsafe { layer_buffer.as_ptr().add(in_start) };
            let step = dilation * conv.in_ch;
            if dilation >= 128 {
                prefetch_strategy_2stage(*tap, step, k, kernel, dilation);
            } else {
                prefetch_strategy_simple(*tap, step, k, kernel, dilation);
            }
        }

        let w_ptr = conv.weights.as_ptr();

        for g in 0..conv.groups {
            let group_in_start = g * in_per_group;
            let group_out_start = g * conv.out_per_group;
            let wk_group_base = g * (num_blocks_per_group * kernel * in_per_group * 4);

            for b in 0..num_blocks_per_group {
                let out_c = group_out_start + b * 4;
                let (mu0, mu1, mu2, mu3) = load_mixin_4(mixin, out_c, conv.out_ch);

                let mut acc = if conv.do_bias {
                    _mm_setr_ps(
                        *conv.bias.get_unchecked(out_c) + mu0,
                        if out_c + 1 < conv.out_ch {
                            *conv.bias.get_unchecked(out_c + 1)
                        } else {
                            0.0
                        } + mu1,
                        if out_c + 2 < conv.out_ch {
                            *conv.bias.get_unchecked(out_c + 2)
                        } else {
                            0.0
                        } + mu2,
                        if out_c + 3 < conv.out_ch {
                            *conv.bias.get_unchecked(out_c + 3)
                        } else {
                            0.0
                        } + mu3,
                    )
                } else {
                    _mm_setr_ps(mu0, mu1, mu2, mu3)
                };

                for ik in 0..kernel {
                    let tap = *tap_ptrs.get_unchecked(ik);
                    for ic in 0..in_per_group {
                        let w_base = wk_group_base
                            + b * (kernel * in_per_group * 4)
                            + ik * (in_per_group * 4)
                            + ic * 4;
                        let wv = _mm_loadu_ps(w_ptr.add(w_base));
                        let sv = _mm_set1_ps(*tap.add(group_in_start + ic));
                        acc = _mm_fmadd_ps(wv, sv, acc);
                    }
                }

                if out_c + 3 < conv.out_ch {
                    _mm_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc);
                } else {
                    // SAFETY: `transmute` from __m128 to [f32; 4] is a
                    // no-op at the ABI level (both are 128-bit types with
                    // the same alignment). This is the standard idiom for
                    // extracting individual lanes from an XMM/NEON register
                    // for edge-case write-back when out_ch is not a
                    // multiple of 4.
                    let r = core::mem::transmute::<__m128, [f32; 4]>(acc);
                    for (lane, &val) in r.iter().enumerate() {
                        if out_c + lane < conv.out_ch {
                            *out_frame.get_unchecked_mut(out_c + lane) = val;
                        }
                    }
                }
            }
        }
    }
}

/// aarch64 grouped kernel: processes four independent 4-lane output blocks
/// per iteration, giving four `vfmaq_f32` chains. Accumulation order per
/// output lane is unchanged (`k` ascending, then `ic` ascending).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn grouped_conv1d_single_frame_neon(
    conv: &A2GroupedConv1d,
    layer_buffer: &[f32],
    out_frame: &mut [f32],
    frame_idx: usize,
    mixin: Option<&[f32]>,
) {
    use core::arch::aarch64::*;

    let in_per_group = conv.in_per_group;
    let kernel = conv.kernel;
    let num_blocks = conv.num_blocks_per_group;
    // SAFETY: the caller's release assertions guarantee every tap of
    // `frame_idx` is in bounds and `kernel <= MAX_KERNEL`.
    let tap_ptrs =
        unsafe { build_tap_ptrs(layer_buffer, frame_idx, conv.in_ch, kernel, conv.dilation) };
    let w_ptr = conv.weights.as_ptr();
    let out_ptr = out_frame.as_mut_ptr();

    let mut block = 0usize;
    while block + 4 <= conv.total_blocks {
        let b0 = block % num_blocks;
        let b1 = (block + 1) % num_blocks;
        let b2 = (block + 2) % num_blocks;
        let b3 = (block + 3) % num_blocks;
        let g0 = block / num_blocks;
        let g1 = (block + 1) / num_blocks;
        let g2 = (block + 2) / num_blocks;
        let g3 = (block + 3) / num_blocks;

        let out0 = g0 * conv.out_per_group + b0 * 4;
        let out1 = g1 * conv.out_per_group + b1 * 4;
        let out2 = g2 * conv.out_per_group + b2 * 4;
        let out3 = g3 * conv.out_per_group + b3 * 4;

        // Interleaved-4 base of (group, block) at `k = 0, ic = 0`.
        let w0 = g0 * (num_blocks * kernel * in_per_group * 4) + b0 * (kernel * in_per_group * 4);
        let w1 = g1 * (num_blocks * kernel * in_per_group * 4) + b1 * (kernel * in_per_group * 4);
        let w2 = g2 * (num_blocks * kernel * in_per_group * 4) + b2 * (kernel * in_per_group * 4);
        let w3 = g3 * (num_blocks * kernel * in_per_group * 4) + b3 * (kernel * in_per_group * 4);

        let in0 = g0 * in_per_group;
        let in1 = g1 * in_per_group;
        let in2 = g2 * in_per_group;
        let in3 = g3 * in_per_group;

        // SAFETY: `out0..out3` are the starting channel indices of four
        // blocks in `0..conv.total_blocks`. `mixin` (when `Some`) is
        // release-asserted at entry to have `>= conv.out_ch` elements, and
        // `conv.bias` is allocated with `conv.out_ch` elements when
        // constructed with a matching `raw_bias` slice (the loader and
        // `neon_init_acc` require `bias.len() >= conv.out_ch` whenever
        // `do_bias` is true).
        let (mut acc0, mut acc1, mut acc2, mut acc3) = unsafe {
            (
                neon_init_acc(conv, mixin, out0),
                neon_init_acc(conv, mixin, out1),
                neon_init_acc(conv, mixin, out2),
                neon_init_acc(conv, mixin, out3),
            )
        };

        for k in 0..kernel {
            let tap = *tap_ptrs.get_unchecked(k);
            let k_off = k * in_per_group * 4;
            for ic in 0..in_per_group {
                let off = k_off + ic * 4;
                let h0 = vdupq_n_f32(*tap.add(in0 + ic));
                let h1 = vdupq_n_f32(*tap.add(in1 + ic));
                let h2 = vdupq_n_f32(*tap.add(in2 + ic));
                let h3 = vdupq_n_f32(*tap.add(in3 + ic));
                acc0 = vfmaq_f32(acc0, vld1q_f32(w_ptr.add(w0 + off)), h0);
                acc1 = vfmaq_f32(acc1, vld1q_f32(w_ptr.add(w1 + off)), h1);
                acc2 = vfmaq_f32(acc2, vld1q_f32(w_ptr.add(w2 + off)), h2);
                acc3 = vfmaq_f32(acc3, vld1q_f32(w_ptr.add(w3 + off)), h3);
            }
        }

        // SAFETY: `out_ptr` derives from `out_frame`, which is
        // release-asserted to have `>= conv.out_ch` elements, and
        // `neon_store_acc` bounds-checks partial final blocks against
        // `conv.out_ch`.
        unsafe {
            neon_store_acc(conv, out_ptr, out0, acc0);
            neon_store_acc(conv, out_ptr, out1, acc1);
            neon_store_acc(conv, out_ptr, out2, acc2);
            neon_store_acc(conv, out_ptr, out3, acc3);
        }
        block += 4;
    }

    while block < conv.total_blocks {
        let g = block / num_blocks;
        let b = block % num_blocks;
        let out_c = g * conv.out_per_group + b * 4;
        let w_base = g * (num_blocks * kernel * in_per_group * 4) + b * (kernel * in_per_group * 4);
        let in_base = g * in_per_group;

        // SAFETY: `out_c` is the starting channel of `block < conv.total_blocks`;
        // `mixin` (if `Some`) is release-asserted to have `>= conv.out_ch`
        // elements, and `conv.bias` holds `>= conv.out_ch` elements when
        // `do_bias` is enabled.
        let mut acc = unsafe { neon_init_acc(conv, mixin, out_c) };
        for k in 0..kernel {
            let tap = *tap_ptrs.get_unchecked(k);
            let k_off = k * in_per_group * 4;
            for ic in 0..in_per_group {
                let off = k_off + ic * 4;
                let h = vdupq_n_f32(*tap.add(in_base + ic));
                acc = vfmaq_f32(acc, vld1q_f32(w_ptr.add(w_base + off)), h);
            }
        }
        // SAFETY: `out_ptr` points to `out_frame` (`len >= conv.out_ch`,
        // release-asserted), and `neon_store_acc` writes at most up to
        // `conv.out_ch`.
        unsafe { neon_store_acc(conv, out_ptr, out_c, acc) };
        block += 1;
    }
}

/// Initializes a 4-lane accumulator exactly like the x86 kernel:
/// `bias + mixin` when the block is fully in range, lane-wise otherwise.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_init_acc(
    conv: &A2GroupedConv1d,
    mixin: Option<&[f32]>,
    out_c: usize,
) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;

    let out_ch = conv.out_ch;
    let m = mixin.map(|m| m.as_ptr());
    if out_c + 3 < out_ch {
        let mut acc = if conv.do_bias {
            vld1q_f32(conv.bias.as_ptr().add(out_c))
        } else {
            vdupq_n_f32(0.0)
        };
        if let Some(mp) = m {
            acc = vaddq_f32(acc, vld1q_f32(mp.add(out_c)));
        }
        acc
    } else {
        let mut lanes = [0.0f32; 4];
        for (lane, slot) in lanes.iter_mut().enumerate() {
            let idx = out_c + lane;
            if idx < out_ch {
                let mut v = if conv.do_bias {
                    *conv.bias.get_unchecked(idx)
                } else {
                    0.0
                };
                if let Some(mp) = m {
                    v += *mp.add(idx);
                }
                *slot = v;
            }
        }
        vld1q_f32(lanes.as_ptr())
    }
}

/// Stores a 4-lane accumulator, writing only the valid lanes of a partial
/// final block.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_store_acc(
    conv: &A2GroupedConv1d,
    out_ptr: *mut f32,
    out_c: usize,
    acc: core::arch::aarch64::float32x4_t,
) {
    use core::arch::aarch64::*;

    if out_c + 3 < conv.out_ch {
        vst1q_f32(out_ptr.add(out_c), acc);
    } else {
        let mut lanes = [0.0f32; 4];
        vst1q_f32(lanes.as_mut_ptr(), acc);
        for (lane, &val) in lanes.iter().enumerate() {
            if out_c + lane < conv.out_ch {
                *out_ptr.add(out_c + lane) = val;
            }
        }
    }
}
