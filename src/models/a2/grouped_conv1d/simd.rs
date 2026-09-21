// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

// SIMD kernel implementation using specialized parameter lists.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! SIMD (AVX2/FMA) kernels for grouped dilated causal Conv1D.

use super::A2GroupedConv1d;
use crate::math::common::{prefetch_strategy_2stage, prefetch_strategy_simple};
use crate::models::wavenet::common::MAX_KERNEL;

use core::arch::x86_64::*;

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

/// Depthwise convolution optimized path (groups == in_ch == out_ch).
///
/// When every channel is its own group, the inner loop simplifies to a single
/// tap convolution per channel — one weight per input per tap.
///
/// # Safety
/// Only call when `groups == in_ch == out_ch`.
#[target_feature(enable = "avx2,fma")]
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

    // F-RES2-10: the 8-lane `_mm256_loadu_ps(tap.add(c))` loads below read up
    // to element `(frame_idx + 1) * in_ch - 1` of `layer_buffer` (last tap =
    // current frame at offset `frame_idx * in_ch`, last lane of the
    // `ch8 = in_ch & !7` block), while the release `assert!` above only proves
    // the tap *base* is in bounds. Production callers guarantee the full-row
    // extent by construction (layer-state buffers are sized as
    // `frames * in_ch` with `frame_idx < frames`); this debug-only net checks
    // exactly that limit and is compiled out of release — zero hot-path cost.
    debug_assert!(
        layer_buffer.len() >= (frame_idx + 1) * conv.in_ch,
        "depthwise: layer_buffer len {} < (frame_idx {} + 1) * in_ch {} \
         (8-lane load upper bound)",
        layer_buffer.len(),
        frame_idx,
        conv.in_ch
    );

    let ch = conv.in_ch;
    let kernel = conv.kernel;
    let dilation = conv.dilation;

    let mut tap_ptrs = [core::ptr::null::<f32>(); MAX_KERNEL];
    let k_limit = kernel.min(MAX_KERNEL);

    for (k, tap) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
        let offset = (dilation as isize) * ((k as isize) + 1 - (kernel as isize));
        let in_start = ((frame_idx as isize) + offset) as usize * ch;
        // SAFETY: `layer_buffer` has length > `frame_idx * in_ch` (release
        // `assert!` above), so the tap base `in_start` is in bounds for all
        // k in [0, kernel) — `dilation * (kernel - 1)` ≤ `frame_idx` keeps the
        // most negative offset from underflowing into a negative pointer.
        // The 8-lane `_mm256_loadu_ps` loads additionally read
        // `tap_base .. tap_base + ch8 - 1`, reaching at most
        // `(frame_idx + 1) * in_ch - 1`; that full-row extent is guaranteed by
        // the callers' frame-sized buffer layout and checked in debug by the
        // `debug_assert!` above (F-RES2-10).
        unsafe {
            *tap = layer_buffer.as_ptr().add(in_start);
            if dilation >= 128 {
                prefetch_strategy_2stage(*tap, dilation * ch, k, kernel, dilation);
            } else {
                prefetch_strategy_simple(*tap, dilation * ch, k, kernel, dilation);
            }
        }
    }

    let w_ptr = conv.weights.as_ptr();
    let bias_ptr = conv.bias.as_ptr();
    let group_stride = kernel * 4;

    let ch8 = ch & !7;

    let gather_idx = _mm256_setr_epi32(
        0,
        group_stride as i32,
        (group_stride * 2) as i32,
        (group_stride * 3) as i32,
        (group_stride * 4) as i32,
        (group_stride * 5) as i32,
        (group_stride * 6) as i32,
        (group_stride * 7) as i32,
    );

    let mut c = 0;
    while c < ch8 {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut k = 0;
        while k + 1 < kernel {
            let tap0 = *tap_ptrs.get_unchecked(k);
            let w_base0 = w_ptr.add(c * group_stride + k * 4);
            let wv0 = _mm256_i32gather_ps(w_base0, gather_idx, 4);
            let sv0 = _mm256_loadu_ps(tap0.add(c));
            acc0 = _mm256_fmadd_ps(wv0, sv0, acc0);

            let tap1 = *tap_ptrs.get_unchecked(k + 1);
            let w_base1 = w_ptr.add(c * group_stride + (k + 1) * 4);
            let wv1 = _mm256_i32gather_ps(w_base1, gather_idx, 4);
            let sv1 = _mm256_loadu_ps(tap1.add(c));
            acc1 = _mm256_fmadd_ps(wv1, sv1, acc1);

            k += 2;
        }
        if k < kernel {
            let tap = *tap_ptrs.get_unchecked(k);
            let w_base = w_ptr.add(c * group_stride + k * 4);
            let wv = _mm256_i32gather_ps(w_base, gather_idx, 4);
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
            let w = *w_ptr.add(c * group_stride + k * 4);
            let s = *tap.add(c);
            acc += w * s;
        }

        if conv.do_bias {
            acc += *conv.bias.get_unchecked(c);
        }

        *out_frame.get_unchecked_mut(c) = acc;
    }
}

/// Grouped dilated conv — alternative inner loop using `__m128` accumulator registers
/// directly for the 4-lane block, avoiding the scalar extract/reinsert pattern.
///
/// Same semantics as `process_single_frame_avx2` but keeps accumulators in XMM
/// registers across the entire inner loop, yielding better code generation.
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
/// - The CPU must support AVX2+FMA (runtime dispatch by the caller).
#[target_feature(enable = "avx2,fma")]
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

    let in_ch = conv.in_ch;
    let in_per_group = conv.in_per_group;
    let out_per_group = conv.out_per_group;
    let kernel = conv.kernel;
    let dilation = conv.dilation;
    let num_blocks_per_group = conv.num_blocks_per_group;
    let groups = conv.groups;

    let mut tap_ptrs = [core::ptr::null::<f32>(); MAX_KERNEL];
    let k_limit = kernel.min(MAX_KERNEL);

    for (k, tap) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
        let offset = (dilation as isize) * ((k as isize) + 1 - (kernel as isize));
        let in_start = ((frame_idx as isize) + offset) as usize * in_ch;
        // SAFETY: `layer_buffer` has length > `frame_idx * in_ch` (release
        // `assert!` above), so the tap base `in_start` is in bounds; same
        // pointer-arithmetic invariants as `process_single_frame_depthwise_avx2`.
        // The lane reads `tap.add(group_in_start + ic)` reach at most
        // `(frame_idx + 1) * in_ch - 1`; that full-row extent is guaranteed by
        // the callers' frame-sized buffer layout and checked in debug by the
        // `debug_assert!` above (F-RES2-10).
        unsafe {
            *tap = layer_buffer.as_ptr().add(in_start);
            if dilation >= 128 {
                prefetch_strategy_2stage(*tap, dilation * in_ch, k, kernel, dilation);
            } else {
                prefetch_strategy_simple(*tap, dilation * in_ch, k, kernel, dilation);
            }
        }
    }

    for g in 0..groups {
        let group_in_start = g * in_per_group;
        let group_out_start = g * out_per_group;

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
                let wk_group_base = g * (num_blocks_per_group * kernel * in_per_group * 4);

                for ic in 0..in_per_group {
                    let w_base = wk_group_base
                        + b * (kernel * in_per_group * 4)
                        + ik * (in_per_group * 4)
                        + ic * 4;

                    let wv = _mm_loadu_ps(conv.weights.as_ptr().add(w_base));
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
                // extracting individual lanes from an XMM register for
                // edge-case write-back when out_ch is not a multiple
                // of 4.
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
