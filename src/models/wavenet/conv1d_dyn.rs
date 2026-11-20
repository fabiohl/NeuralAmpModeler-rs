// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Runtime-dimensional convolution components for WaveNet architectures.
//!
//! Contains the fundamental convolution structures that operate with
//! runtime-defined dimensions, serving as a foundation for A2 architecture
//! stages and static WaveNet test/stress kernels.

use crate::math::common::{
    AlignedVec, SimdMath, prefetch_strategy_2stage, prefetch_strategy_simple,
};

use super::common::MAX_KERNEL;

/// Structure for causal 1D convolution with dynamic dimensions.
///
/// **Layout invariant (F-16 / R-2):** the hot path reinterprets `weights` as
/// SIMD-interleaved `[f32; W]` blocks (`W == self.interleave_width`, 4/8/16),
/// reading up to `out_ch.div_ceil(W) * W * in_ch * kernel` f32s. `weights`
/// must therefore hold **at least** that SIMD-padded total. Prefer
/// [`Conv1dDyn::try_from_parts`], which validates the padded length in a
/// release-stable check off the real-time path; direct struct-literal
/// construction must uphold the invariant manually.
#[derive(Clone)]
#[repr(align(64))]
pub struct Conv1dDyn {
    /// Full-precision f32 convolution weights `[num_blocks][KERNEL][IN][W]`
    /// (lane-interleaved), zero-padded so every `[f32; W]` block is fully
    /// covered (length >= `out_ch.div_ceil(W) * W * in_ch * kernel`).
    pub weights: AlignedVec<f32>,
    /// Bias vector `[OUT]`.
    pub bias: AlignedVec<f32>,
    /// Flag indicating whether bias should be applied.
    pub do_bias: bool,
    /// Temporal dilation factor.
    pub dilation: usize,
    /// Number of input channels.
    pub in_ch: usize,
    /// Number of output channels.
    pub out_ch: usize,
    /// Number of 4-channel blocks (used by dual-frame path).
    pub num_blocks: usize,
    /// Interleave width (4, 8, or 16) used for single-frame processing.
    pub interleave_width: usize,
    /// Physical kernel size.
    pub kernel: usize,
}

impl Conv1dDyn {
    /// Validated constructor (release-stable, off-RT) for a runtime-dimensional
    /// `Conv1dDyn`.
    ///
    /// This is the release-stable owner of the interleaved-weights padding
    /// invariant (F-16 / R-2). The hot-path kernels derive their `[f32; W]`
    /// slices from `self.weights` for every output block
    /// `b < out_ch.div_ceil(W)` and every tap `k < kernel`, whose last element
    /// ends at `out_ch.div_ceil(W) * W * in_ch * kernel` (computed with checked
    /// arithmetic, F-RES2-04); rejecting smaller buffers here keeps every
    /// `from_raw_parts` in the process methods
    /// structurally in bounds in release builds — no hot-path check is needed
    /// or performed.
    ///
    /// `interleave_width` must be the width used to store `weights` (4, 8 or
    /// 16; callers such as the loader or the A2/ConvNet builders select it and
    /// interleave the buffer accordingly).
    ///
    /// # Errors
    /// Returns an error if:
    /// - `kernel` is zero or exceeds `MAX_KERNEL` — the hot path reads taps
    ///   through a fixed `MAX_KERNEL`-entry array;
    /// - `interleave_width` is not one of 4/8/16;
    /// - the padded size product `out_ch.div_ceil(W) * W * in_ch * kernel`
    ///   overflows `usize` (checked, F-RES2-04 — never wraps in release);
    /// - `weights` holds fewer than the SIMD-padded total
    ///   `out_ch.div_ceil(W) * W * in_ch * kernel` f32s, which would make the
    ///   interleaved SIMD kernels read out of bounds on the hot path.
    #[inline]
    #[expect(
        clippy::too_many_arguments,
        reason = "validated Conv1dDyn constructor receiving the full runtime-dimensional layout descriptor (weights, bias, do_bias, dilation, in_ch, out_ch, kernel, interleave_width)"
    )]
    pub fn try_from_parts(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        dilation: usize,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        interleave_width: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(in_ch > 0, "Conv1dDyn in_ch must be >= 1, got {in_ch}");
        anyhow::ensure!(out_ch > 0, "Conv1dDyn out_ch must be >= 1, got {out_ch}");
        anyhow::ensure!(
            kernel > 0,
            "Conv1dDyn kernel_size must be >= 1, got {kernel}"
        );
        anyhow::ensure!(
            kernel <= MAX_KERNEL,
            "Conv1dDyn kernel_size ({kernel}) exceeds maximum supported ({MAX_KERNEL}) — \
             the dynamic convolution hot-path reads taps through a fixed \
             {MAX_KERNEL}-entry pointer array"
        );
        anyhow::ensure!(
            interleave_width == 4 || interleave_width == 8 || interleave_width == 16,
            "Conv1dDyn interleave_width must be 4, 8 or 16, got {interleave_width}"
        );
        let num_blocks = out_ch.div_ceil(interleave_width);
        // Defense-in-depth (F-RES2-04): checked size product so hostile
        // `in_ch`/`out_ch`/`kernel` can never wrap into a smaller contract in
        // release builds (overflow-checks off). Mirrors the
        // `checked_arith::checked_conv_padded_total` pattern used by the loader.
        let padded_total = num_blocks
            .checked_mul(interleave_width)
            .and_then(|v| v.checked_mul(in_ch))
            .and_then(|v| v.checked_mul(kernel))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Conv1dDyn weights size overflows usize: num_blocks ({num_blocks}) * interleave_width ({interleave_width}) * in_ch ({in_ch}) * kernel ({kernel}) — DoS protection (F-RES2-04)"
                )
            })?;
        anyhow::ensure!(
            weights.len() >= padded_total,
            "Conv1d weights buffer is too small: expected >= {padded_total} \
             (SIMD-padded, interleave width {interleave_width}), got {}",
            weights.len()
        );
        if do_bias {
            anyhow::ensure!(
                bias.len() >= out_ch,
                "Conv1dDyn bias buffer is too small: expected >= {out_ch}, got {}",
                bias.len()
            );
        }
        Ok(Conv1dDyn {
            weights,
            bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            num_blocks: out_ch.div_ceil(4),
            interleave_width,
            kernel,
        })
    }

    /// Returns whether this convolution still satisfies every invariant the
    /// unchecked SIMD kernels rely on.
    ///
    /// The fields are public, so consumers may replace `weights`, `bias`,
    /// `kernel`, `dilation` or the channel counts after
    /// [`Self::try_from_parts`]. The safe model entry points consult this
    /// predicate before dispatching a kernel that reads interleaved weights
    /// through raw pointers.
    #[inline]
    pub(crate) fn is_sound(&self) -> bool {
        let width = self.interleave_width;
        if width != 4 && width != 8 && width != 16 {
            return false;
        }
        if self.in_ch == 0 || self.out_ch == 0 {
            return false;
        }
        if self.kernel == 0 || self.kernel > MAX_KERNEL {
            return false;
        }
        if self.dilation > isize::MAX as usize {
            return false;
        }
        // `num_blocks` is a descriptive field (the kernels derive their own
        // block count from `out_ch` and `interleave_width`), so it is not
        // part of the memory-safety contract and is not validated here.
        let Some(padded_total) = self
            .out_ch
            .div_ceil(width)
            .checked_mul(width)
            .and_then(|v| v.checked_mul(self.in_ch))
            .and_then(|v| v.checked_mul(self.kernel))
        else {
            return false;
        };
        if self.weights.len() < padded_total {
            return false;
        }
        if self.do_bias && self.bias.len() < self.out_ch {
            return false;
        }
        true
    }

    /// Largest causal lookback in frames required by the taps.
    #[inline]
    pub(crate) fn backreach(&self) -> Option<usize> {
        self.kernel.checked_sub(1)?.checked_mul(self.dilation)
    }

    /// F32-native single-frame convolution (full-precision f32 weights).
    ///
    /// # Safety
    /// The caller must guarantee that `layer_buffer` and `out_frame` have sizes
    /// compatible with the layer dimensions, and that `frame_idx` satisfies the
    /// warm-up invariant `frame_idx >= (kernel - 1) * dilation` — otherwise the
    /// tap offsets below would be negative. That invariant is owned release-stable
    /// by the layer-state construction (`WaveNetLayerState::new` enforces
    /// `buffer_start >= receptive_field_size >= (kernel-1) * dilation`; see
    /// `common.rs`). A defensive `.max(0)` clamp keeps any residual violation
    /// in-bounds (no `usize` wrapping into a wild pointer), but only a caller
    /// honoring the invariant produces correct audio (F-01).
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        let in_ch = self.in_ch;
        let kernel = self.kernel;
        let mut tap_ptrs = [core::mem::MaybeUninit::<*const f32>::uninit(); MAX_KERNEL];
        let k_limit = kernel.min(MAX_KERNEL);

        for (k, tap) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (kernel as isize));
            // F-01: `frame_idx + offset` is negative when frame_idx is below the
            // warm-up threshold; the `.max(0)` clamp prevents `as usize` from
            // wrapping into an out-of-bounds pointer.
            let in_start = ((frame_idx as isize) + offset).max(0) as usize * in_ch;
            // SAFETY: `in_start` is clamped non-negative by the `.max(0)` above (F-01), and the
            // warm-up invariant is owned release-stable by the layer-state construction
            // (`WaveNetLayerState::new` enforces
            // `buffer_start >= receptive_field_size >= dilation*(kernel-1)`; see `common.rs`),
            // so on the model path the clamp is inactive and the tap lands on the causal
            // receptive field; the caller's documented contract guarantees `layer_buffer` is
            // sized so the `in_ch`-wide tap at that offset is in bounds; the prefetch calls
            // only touch the address, not the memory contents.
            unsafe {
                let ptr = layer_buffer.as_ptr().add(in_start);
                tap.write(ptr);
                if self.dilation >= 128 {
                    prefetch_strategy_2stage(ptr, self.dilation * in_ch, k, kernel, self.dilation);
                } else {
                    prefetch_strategy_simple(ptr, self.dilation * in_ch, k, kernel, self.dilation);
                }
            }
        }

        // SAFETY: `tap_ptrs[0..k_limit]` were all initialized in the loop above.
        let tap_ptrs =
            unsafe { core::slice::from_raw_parts(tap_ptrs.as_ptr() as *const *const f32, k_limit) };

        // SAFETY: the `process_blocks_*_nocopy` helpers are `unsafe fn`s reached only under the
        // preconditions of `process_single_frame` (valid `out_frame` of `out_ch` lanes, tap
        // pointers set above from the in-bounds `layer_buffer`, `k_limit <= MAX_KERNEL`), and the
        // x86-64-v3 / aarch64 NEON baseline guarantees AVX2+FMA / NEON availability.
        unsafe {
            match self.interleave_width {
                16 => self.process_blocks_16_nocopy(out_frame, k_limit, tap_ptrs, in_ch, mixin),
                8 => self.process_blocks_8_nocopy(out_frame, k_limit, tap_ptrs, in_ch, mixin),
                _ => self.process_blocks_4_nocopy(out_frame, k_limit, tap_ptrs, in_ch, mixin),
            }
        }
    }

    #[inline]
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
    #[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
    unsafe fn process_blocks_4_nocopy(
        &self,
        out_frame: &mut [f32],
        kernel: usize,
        tap_ptrs: &[*const f32],
        in_ch: usize,
        mixin: Option<&[f32]>,
    ) {
        use crate::arch::simd::*;

        let num_blocks = self.out_ch.div_ceil(4);
        let w_stride = kernel * in_ch * 4;
        let k_stride = in_ch * 4;
        for b in 0..num_blocks {
            let out_c = b * 4;
            let w = 4.min(self.out_ch - out_c);
            let b_base = b * w_stride;
            // SAFETY: `bias.len() == out_ch`, `weights.len() >= padded_total`, `tap_ptrs[k]`
            // covers `in_ch` valid f32s, `out_frame` has `out_ch` lanes, and the x86-64-v3 /
            // aarch64 NEON baseline guarantees SSE/AVX2/FMA/NEON intrinsics.
            unsafe {
                let (mu0, mu1, mu2, mu3) = Self::load_mixin_4(mixin, out_c);

                let mut acc0 = if self.do_bias {
                    let b0 = *self.bias.get_unchecked(out_c) + mu0;
                    let b1 = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mu1;
                    let b2 = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mu2;
                    let b3 = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mu3;
                    _mm_setr_ps(b0, b1, b2, b3)
                } else {
                    _mm_setr_ps(mu0, mu1, mu2, mu3)
                };
                let mut acc1 = _mm_setzero_ps();
                let mut acc2 = _mm_setzero_ps();
                let mut acc3 = _mm_setzero_ps();

                for k in 0..kernel {
                    let w_start = b_base + k * k_stride;
                    debug_assert!(
                        w_start + 4 * in_ch <= self.weights.len(),
                        "conv1d_dyn: interleave-4 weight slice exceeds padded weights buffer"
                    );
                    let w_ptr = self.weights.as_ptr().add(w_start);
                    let tap_ptr = *tap_ptrs.get_unchecked(k);
                    let mut ic = 0usize;
                    while ic + 4 <= in_ch {
                        let w0 = _mm_loadu_ps(w_ptr.add(ic * 4));
                        let s0 = _mm_set1_ps(*tap_ptr.add(ic));
                        acc0 = _mm_fmadd_ps(w0, s0, acc0);

                        let w1 = _mm_loadu_ps(w_ptr.add((ic + 1) * 4));
                        let s1 = _mm_set1_ps(*tap_ptr.add(ic + 1));
                        acc1 = _mm_fmadd_ps(w1, s1, acc1);

                        let w2 = _mm_loadu_ps(w_ptr.add((ic + 2) * 4));
                        let s2 = _mm_set1_ps(*tap_ptr.add(ic + 2));
                        acc2 = _mm_fmadd_ps(w2, s2, acc2);

                        let w3 = _mm_loadu_ps(w_ptr.add((ic + 3) * 4));
                        let s3 = _mm_set1_ps(*tap_ptr.add(ic + 3));
                        acc3 = _mm_fmadd_ps(w3, s3, acc3);
                        ic += 4;
                    }
                    while ic < in_ch {
                        let w0 = _mm_loadu_ps(w_ptr.add(ic * 4));
                        let s0 = _mm_set1_ps(*tap_ptr.add(ic));
                        acc0 = _mm_fmadd_ps(w0, s0, acc0);
                        ic += 1;
                    }
                }

                acc0 = _mm_add_ps(acc0, acc1);
                acc2 = _mm_add_ps(acc2, acc3);
                acc0 = _mm_add_ps(acc0, acc2);

                if out_c + 3 < self.out_ch {
                    _mm_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
                } else {
                    let mut tmp = [0.0f32; 4];
                    _mm_storeu_ps(tmp.as_mut_ptr(), acc0);
                    for (lane, &val) in tmp.iter().enumerate().take(w) {
                        *out_frame.get_unchecked_mut(out_c + lane) = val;
                    }
                }
            }
        }
    }

    #[inline]
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
    #[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
    unsafe fn process_blocks_8_nocopy(
        &self,
        out_frame: &mut [f32],
        kernel: usize,
        tap_ptrs: &[*const f32],
        in_ch: usize,
        mixin: Option<&[f32]>,
    ) {
        use crate::arch::simd::*;

        let num_blocks = self.out_ch.div_ceil(8);
        let w_stride = kernel * in_ch * 8;
        let k_stride = in_ch * 8;
        for b in 0..num_blocks {
            let out_c = b * 8;
            let w = 8.min(self.out_ch - out_c);
            let b_base = b * w_stride;
            // SAFETY: `bias.len() == out_ch`, `weights.len() >= padded_total`, `tap_ptrs[k]`
            // covers `in_ch` valid f32s, `out_frame` has `out_ch` lanes, and the x86-64-v3 /
            // aarch64 NEON baseline guarantees AVX2/FMA/NEON intrinsics.
            unsafe {
                let mut acc0 = if w == 8 && self.do_bias && mixin.is_none() {
                    _mm256_loadu_ps(self.bias.as_ptr().add(out_c))
                } else {
                    let mut init = [0.0f32; 8];
                    for (j, item) in init.iter_mut().enumerate().take(w) {
                        let v_bias = if self.do_bias {
                            *self.bias.get_unchecked(out_c + j)
                        } else {
                            0.0
                        };
                        let v_mixin = if let Some(m) = mixin {
                            if out_c + j < m.len() {
                                *m.get_unchecked(out_c + j)
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        *item = v_bias + v_mixin;
                    }
                    _mm256_loadu_ps(init.as_ptr())
                };
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();

                for k in 0..kernel {
                    let w_start = b_base + k * k_stride;
                    debug_assert!(
                        w_start + 8 * in_ch <= self.weights.len(),
                        "conv1d_dyn: interleave-8 weight slice exceeds padded weights buffer"
                    );
                    let w_ptr = self.weights.as_ptr().add(w_start);
                    let tap_ptr = *tap_ptrs.get_unchecked(k);
                    let mut ic = 0usize;
                    while ic + 4 <= in_ch {
                        let w0 = _mm256_loadu_ps(w_ptr.add(ic * 8));
                        let s0 = _mm256_set1_ps(*tap_ptr.add(ic));
                        acc0 = _mm256_fmadd_ps(w0, s0, acc0);

                        let w1 = _mm256_loadu_ps(w_ptr.add((ic + 1) * 8));
                        let s1 = _mm256_set1_ps(*tap_ptr.add(ic + 1));
                        acc1 = _mm256_fmadd_ps(w1, s1, acc1);

                        let w2 = _mm256_loadu_ps(w_ptr.add((ic + 2) * 8));
                        let s2 = _mm256_set1_ps(*tap_ptr.add(ic + 2));
                        acc2 = _mm256_fmadd_ps(w2, s2, acc2);

                        let w3 = _mm256_loadu_ps(w_ptr.add((ic + 3) * 8));
                        let s3 = _mm256_set1_ps(*tap_ptr.add(ic + 3));
                        acc3 = _mm256_fmadd_ps(w3, s3, acc3);
                        ic += 4;
                    }
                    while ic < in_ch {
                        let w0 = _mm256_loadu_ps(w_ptr.add(ic * 8));
                        let s0 = _mm256_set1_ps(*tap_ptr.add(ic));
                        acc0 = _mm256_fmadd_ps(w0, s0, acc0);
                        ic += 1;
                    }
                }

                acc0 = _mm256_add_ps(acc0, acc1);
                acc2 = _mm256_add_ps(acc2, acc3);
                acc0 = _mm256_add_ps(acc0, acc2);

                if w == 8 {
                    _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
                } else {
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), acc0);
                    for (j, &item) in tmp.iter().enumerate().take(w) {
                        *out_frame.get_unchecked_mut(out_c + j) = item;
                    }
                }
            }
        }
    }

    #[inline]
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
    #[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
    unsafe fn process_blocks_16_nocopy(
        &self,
        out_frame: &mut [f32],
        kernel: usize,
        tap_ptrs: &[*const f32],
        in_ch: usize,
        mixin: Option<&[f32]>,
    ) {
        use crate::arch::simd::*;

        let num_blocks = self.out_ch.div_ceil(16);
        let w_stride = kernel * in_ch * 16;
        let k_stride = in_ch * 16;
        for b in 0..num_blocks {
            let out_c = b * 16;
            let w = 16.min(self.out_ch - out_c);
            let b_base = b * w_stride;
            // SAFETY: `bias.len() == out_ch`, `weights.len() >= padded_total`, `tap_ptrs[k]`
            // covers `in_ch` valid f32s, `out_frame` has `out_ch` lanes, and the x86-64-v3 /
            // aarch64 NEON baseline guarantees AVX2/FMA/NEON intrinsics.
            unsafe {
                let (mut acc_lo0, mut acc_hi0) = if w == 16 && self.do_bias && mixin.is_none() {
                    (
                        _mm256_loadu_ps(self.bias.as_ptr().add(out_c)),
                        _mm256_loadu_ps(self.bias.as_ptr().add(out_c + 8)),
                    )
                } else {
                    let mut init = [0.0f32; 16];
                    for (j, item) in init.iter_mut().enumerate().take(w) {
                        let v_bias = if self.do_bias {
                            *self.bias.get_unchecked(out_c + j)
                        } else {
                            0.0
                        };
                        let v_mixin = if let Some(m) = mixin {
                            if out_c + j < m.len() {
                                *m.get_unchecked(out_c + j)
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        *item = v_bias + v_mixin;
                    }
                    (
                        _mm256_loadu_ps(init.as_ptr()),
                        _mm256_loadu_ps(init.as_ptr().add(8)),
                    )
                };
                let mut acc_lo1 = _mm256_setzero_ps();
                let mut acc_lo2 = _mm256_setzero_ps();
                let mut acc_lo3 = _mm256_setzero_ps();
                let mut acc_hi1 = _mm256_setzero_ps();
                let mut acc_hi2 = _mm256_setzero_ps();
                let mut acc_hi3 = _mm256_setzero_ps();

                for k in 0..kernel {
                    let w_start = b_base + k * k_stride;
                    debug_assert!(
                        w_start + 16 * in_ch <= self.weights.len(),
                        "conv1d_dyn: interleave-16 weight slice exceeds padded weights buffer"
                    );
                    let w_ptr = self.weights.as_ptr().add(w_start);
                    let tap_ptr = *tap_ptrs.get_unchecked(k);
                    let mut ic = 0usize;
                    while ic + 4 <= in_ch {
                        let s0 = _mm256_set1_ps(*tap_ptr.add(ic));
                        let w0_lo = _mm256_loadu_ps(w_ptr.add(ic * 16));
                        let w0_hi = _mm256_loadu_ps(w_ptr.add(ic * 16 + 8));
                        acc_lo0 = _mm256_fmadd_ps(w0_lo, s0, acc_lo0);
                        acc_hi0 = _mm256_fmadd_ps(w0_hi, s0, acc_hi0);

                        let s1 = _mm256_set1_ps(*tap_ptr.add(ic + 1));
                        let w1_lo = _mm256_loadu_ps(w_ptr.add((ic + 1) * 16));
                        let w1_hi = _mm256_loadu_ps(w_ptr.add((ic + 1) * 16 + 8));
                        acc_lo1 = _mm256_fmadd_ps(w1_lo, s1, acc_lo1);
                        acc_hi1 = _mm256_fmadd_ps(w1_hi, s1, acc_hi1);

                        let s2 = _mm256_set1_ps(*tap_ptr.add(ic + 2));
                        let w2_lo = _mm256_loadu_ps(w_ptr.add((ic + 2) * 16));
                        let w2_hi = _mm256_loadu_ps(w_ptr.add((ic + 2) * 16 + 8));
                        acc_lo2 = _mm256_fmadd_ps(w2_lo, s2, acc_lo2);
                        acc_hi2 = _mm256_fmadd_ps(w2_hi, s2, acc_hi2);

                        let s3 = _mm256_set1_ps(*tap_ptr.add(ic + 3));
                        let w3_lo = _mm256_loadu_ps(w_ptr.add((ic + 3) * 16));
                        let w3_hi = _mm256_loadu_ps(w_ptr.add((ic + 3) * 16 + 8));
                        acc_lo3 = _mm256_fmadd_ps(w3_lo, s3, acc_lo3);
                        acc_hi3 = _mm256_fmadd_ps(w3_hi, s3, acc_hi3);
                        ic += 4;
                    }
                    while ic < in_ch {
                        let s0 = _mm256_set1_ps(*tap_ptr.add(ic));
                        let w0_lo = _mm256_loadu_ps(w_ptr.add(ic * 16));
                        let w0_hi = _mm256_loadu_ps(w_ptr.add(ic * 16 + 8));
                        acc_lo0 = _mm256_fmadd_ps(w0_lo, s0, acc_lo0);
                        acc_hi0 = _mm256_fmadd_ps(w0_hi, s0, acc_hi0);
                        ic += 1;
                    }
                }

                acc_lo0 = _mm256_add_ps(acc_lo0, acc_lo1);
                acc_lo2 = _mm256_add_ps(acc_lo2, acc_lo3);
                acc_lo0 = _mm256_add_ps(acc_lo0, acc_lo2);

                acc_hi0 = _mm256_add_ps(acc_hi0, acc_hi1);
                acc_hi2 = _mm256_add_ps(acc_hi2, acc_hi3);
                acc_hi0 = _mm256_add_ps(acc_hi0, acc_hi2);

                if w == 16 {
                    _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc_lo0);
                    _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c + 8), acc_hi0);
                } else {
                    let mut tmp = [0.0f32; 16];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), acc_lo0);
                    _mm256_storeu_ps(tmp.as_mut_ptr().add(8), acc_hi0);
                    for (j, &item) in tmp.iter().enumerate().take(w) {
                        *out_frame.get_unchecked_mut(out_c + j) = item;
                    }
                }
            }
        }
    }

    /// T=8 frame-tiled 4-wide interleaved Conv1D forward pass (with bias, no mixin).
    ///
    /// Processes `num_frames` in tiles of 8 frames, keeping 8 XMM accumulators in
    /// registers across all `kernel * in_ch` FMAs per 4-channel block to amortize
    /// weight loads and saturate FMA ports.
    ///
    /// # Safety
    /// - `self.interleave_width == 4` and `self.do_bias` must hold.
    /// - `frame_start >= (self.kernel - 1) * self.dilation` must hold so every tap
    ///   address stays non-negative without clamping.
    /// - `layer_buffer` must cover `(frame_start + num_frames) * self.in_ch` elements.
    /// - `z_out` must have length `>= num_frames * self.out_ch`.
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
    #[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
    pub(crate) unsafe fn process_block_no_mixin_4wide(
        &self,
        layer_buffer: &[f32],
        z_out: &mut [f32],
        frame_start: usize,
        num_frames: usize,
    ) {
        use crate::arch::simd::*;

        debug_assert_eq!(self.interleave_width, 4);
        debug_assert!(self.do_bias);
        debug_assert!(self.kernel >= 1);
        debug_assert!(frame_start >= (self.kernel - 1) * self.dilation);
        debug_assert!(z_out.len() >= num_frames * self.out_ch);

        let in_ch = self.in_ch;
        let out_ch = self.out_ch;
        let kernel = self.kernel;
        let d = self.dilation as isize;
        let k_i = kernel as isize;
        let num_blocks = out_ch.div_ceil(4);
        let w_stride = kernel * in_ch * 4;
        let k_stride = in_ch * 4;
        let buf = layer_buffer.as_ptr();
        let w_base = self.weights.as_ptr();
        let z_ptr = z_out.as_mut_ptr();

        const T: usize = 8;
        let n_tiled = (num_frames / T) * T;

        for b in 0..num_blocks {
            let out_c = b * 4;
            let w = 4.min(out_ch - out_c);
            // SAFETY: caller preconditions guarantee `bias`, `weights`, `layer_buffer`, and
            // `z_out` cover all indexed lanes; `target_feature` enables AVX2+FMA / NEON.
            unsafe {
                let b0 = *self.bias.get_unchecked(out_c);
                let b1 = if out_c + 1 < out_ch {
                    *self.bias.get_unchecked(out_c + 1)
                } else {
                    0.0
                };
                let b2 = if out_c + 2 < out_ch {
                    *self.bias.get_unchecked(out_c + 2)
                } else {
                    0.0
                };
                let b3 = if out_c + 3 < out_ch {
                    *self.bias.get_unchecked(out_c + 3)
                } else {
                    0.0
                };
                let bias_v = _mm_setr_ps(b0, b1, b2, b3);
                let bw_ptr = w_base.add(b * w_stride);

                for f in (0..n_tiled).step_by(T) {
                    let frame0 = (frame_start + f) as isize;
                    let mut a0 = bias_v;
                    let mut a1 = bias_v;
                    let mut a2 = bias_v;
                    let mut a3 = bias_v;
                    let mut a4 = bias_v;
                    let mut a5 = bias_v;
                    let mut a6 = bias_v;
                    let mut a7 = bias_v;

                    for k in 0..kernel {
                        let taps_back = k_i - 1 - k as isize;
                        let tap0 = (frame0 - d * taps_back) as usize;
                        let hb = buf.add(tap0 * in_ch);
                        let kw_ptr = bw_ptr.add(k * k_stride);
                        for ic in 0..in_ch {
                            let wv = _mm_loadu_ps(kw_ptr.add(ic * 4));
                            a0 = _mm_fmadd_ps(wv, _mm_set1_ps(*hb.add(ic)), a0);
                            a1 = _mm_fmadd_ps(wv, _mm_set1_ps(*hb.add(in_ch + ic)), a1);
                            a2 = _mm_fmadd_ps(wv, _mm_set1_ps(*hb.add(2 * in_ch + ic)), a2);
                            a3 = _mm_fmadd_ps(wv, _mm_set1_ps(*hb.add(3 * in_ch + ic)), a3);
                            a4 = _mm_fmadd_ps(wv, _mm_set1_ps(*hb.add(4 * in_ch + ic)), a4);
                            a5 = _mm_fmadd_ps(wv, _mm_set1_ps(*hb.add(5 * in_ch + ic)), a5);
                            a6 = _mm_fmadd_ps(wv, _mm_set1_ps(*hb.add(6 * in_ch + ic)), a6);
                            a7 = _mm_fmadd_ps(wv, _mm_set1_ps(*hb.add(7 * in_ch + ic)), a7);
                        }
                    }

                    if w == 4 {
                        _mm_storeu_ps(z_ptr.add(f * out_ch + out_c), a0);
                        _mm_storeu_ps(z_ptr.add((f + 1) * out_ch + out_c), a1);
                        _mm_storeu_ps(z_ptr.add((f + 2) * out_ch + out_c), a2);
                        _mm_storeu_ps(z_ptr.add((f + 3) * out_ch + out_c), a3);
                        _mm_storeu_ps(z_ptr.add((f + 4) * out_ch + out_c), a4);
                        _mm_storeu_ps(z_ptr.add((f + 5) * out_ch + out_c), a5);
                        _mm_storeu_ps(z_ptr.add((f + 6) * out_ch + out_c), a6);
                        _mm_storeu_ps(z_ptr.add((f + 7) * out_ch + out_c), a7);
                    } else {
                        let accs = [a0, a1, a2, a3, a4, a5, a6, a7];
                        for (t, &acc) in accs.iter().enumerate() {
                            let mut tmp = [0.0f32; 4];
                            _mm_storeu_ps(tmp.as_mut_ptr(), acc);
                            for (lane, &val) in tmp.iter().enumerate().take(w) {
                                *z_ptr.add((f + t) * out_ch + out_c + lane) = val;
                            }
                        }
                    }
                }

                for f in n_tiled..num_frames {
                    let frame_idx = (frame_start + f) as isize;
                    let mut acc = bias_v;
                    for k in 0..kernel {
                        let taps_back = k_i - 1 - k as isize;
                        let tap = (frame_idx - d * taps_back) as usize;
                        let hb = buf.add(tap * in_ch);
                        let kw_ptr = bw_ptr.add(k * k_stride);
                        for ic in 0..in_ch {
                            let wv = _mm_loadu_ps(kw_ptr.add(ic * 4));
                            acc = _mm_fmadd_ps(wv, _mm_set1_ps(*hb.add(ic)), acc);
                        }
                    }
                    if w == 4 {
                        _mm_storeu_ps(z_ptr.add(f * out_ch + out_c), acc);
                    } else {
                        let mut tmp = [0.0f32; 4];
                        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
                        for (lane, &val) in tmp.iter().enumerate().take(w) {
                            *z_ptr.add(f * out_ch + out_c + lane) = val;
                        }
                    }
                }
            }
        }
    }

    /// F32-native block processing (full-precision f32 weights).
    ///
    /// Same dual-frame tiling as the generic block processing, but uses full-precision
    /// f32 weights and scalar dot products.
    #[inline(always)]
    pub(crate) unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        debug_assert_eq!(num_frames * self.out_ch, block.len());
        let mut i = 0;

        let mut chunks = block.chunks_exact_mut(2 * self.out_ch);
        for chunk in chunks.by_ref() {
            let (out_f0, out_f1) = chunk.split_at_mut(self.out_ch);

            let (m_f0, m_f1) = if let Some(m) = mixin {
                let start0 = i * self.out_ch;
                let end0 = (start0 + self.out_ch).min(m.len());
                let start1 = (i + 1) * self.out_ch;
                let end1 = (start1 + self.out_ch).min(m.len());
                (
                    if start0 < m.len() {
                        Some(&m[start0..end0])
                    } else {
                        None
                    },
                    if start1 < m.len() {
                        Some(&m[start1..end1])
                    } else {
                        None
                    },
                )
            } else {
                (None, None)
            };

            // SAFETY: `process_dual_frame` is an `unsafe fn` reached here with `out_f0`/`out_f1`
            // each exactly `out_ch` lanes (from `chunks_exact_mut(2 * self.out_ch)` +
            // `split_at_mut`), `layer_buffer` sized per the caller contract, and
            // `buffer_start + i` honoring the caller's warm-up invariant; the mixin slices were
            // bounds-clamped above.
            unsafe {
                self.process_dual_frame::<M>(
                    layer_buffer,
                    out_f0,
                    out_f1,
                    buffer_start + i,
                    buffer_start + i + 1,
                    m_f0,
                    m_f1,
                );
            }
            i += 2;
        }

        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            // R-7: the remainder is the `i`-th frame of this block, so its mixin segment is
            // `[i*out_ch..(i+1)*out_ch]`; it is bounds-clamped exactly like the dual-frame
            // slices above — a short (caller-contract-violating) `mixin` yields `None` or a
            // partial tail instead of panicking on the audio thread, and the guarded mixin
            // loaders read missing lanes as zero.
            let m = if let Some(m) = mixin {
                let start = i * self.out_ch;
                let end = (start + self.out_ch).min(m.len());
                if start < m.len() {
                    Some(&m[start..end])
                } else {
                    None
                }
            } else {
                None
            };
            // SAFETY: `process_single_frame` is an `unsafe fn` reached here with `rem` exactly
            // `out_ch` lanes (the remainder of `chunks_exact_mut(2 * self.out_ch)` over a
            // `num_frames * out_ch` block), `layer_buffer` per the caller contract, and the mixin
            // slice `m`, when present, is a bounds-clamped (possibly partial) tail of the caller's
            // mixin whose missing lanes are read as zero by the guarded mixin loaders.
            unsafe {
                self.process_single_frame::<M>(layer_buffer, rem, buffer_start + i, m);
            }
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn load_mixin_4(mixin: Option<&[f32]>, out_c: usize) -> (f32, f32, f32, f32) {
        if let Some(m) = mixin {
            if out_c + 3 < m.len() {
                // SAFETY: this branch is guarded by `out_c + 3 < m.len()`, so all four
                // `get_unchecked` lanes `out_c..out_c + 3` are in bounds of `m`.
                unsafe {
                    (
                        *m.get_unchecked(out_c),
                        *m.get_unchecked(out_c + 1),
                        *m.get_unchecked(out_c + 2),
                        *m.get_unchecked(out_c + 3),
                    )
                }
            } else {
                let mut v = [0.0f32; 4];
                for (i, val) in v.iter_mut().enumerate() {
                    if out_c + i < m.len() {
                        // SAFETY: the enclosing `if out_c + i < m.len()` guard guarantees
                        // `out_c + i` is in bounds of `m`.
                        unsafe {
                            *val = *m.get_unchecked(out_c + i);
                        }
                    }
                }
                (v[0], v[1], v[2], v[3])
            }
        } else {
            (0.0, 0.0, 0.0, 0.0)
        }
    }
}
