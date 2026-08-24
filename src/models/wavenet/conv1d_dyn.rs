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
#[derive(Clone)]
#[repr(align(64))]
pub struct Conv1dDyn {
    /// Full-precision f32 convolution weights `[OUT][KERNEL][IN]` (interleaved).
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
    /// F32-native single-frame convolution (full-precision f32 weights).
    ///
    /// # Safety
    /// The caller must guarantee that `layer_buffer` and `out_frame` have sizes
    /// compatible with the layer dimensions, and that `frame_idx` satisfies the
    /// warm-up invariant `frame_idx >= (kernel - 1) * dilation` — otherwise the
    /// tap offsets below would be negative. A defensive clamp to the buffer
    /// start keeps any violation in-bounds (no `usize` wrapping into a wild
    /// pointer), but only a caller honoring the invariant produces correct
    /// audio (F-01).
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
        let mut tap_ptrs = [core::ptr::null::<f32>(); MAX_KERNEL];
        let k_limit = kernel.min(MAX_KERNEL);

        for (k, tap) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (kernel as isize));
            // F-01: `frame_idx + offset` is negative when frame_idx is below the
            // warm-up threshold; the `.max(0)` clamp prevents `as usize` from
            // wrapping into an out-of-bounds pointer.
            let in_start = ((frame_idx as isize) + offset).max(0) as usize * in_ch;
            // SAFETY: `in_start` is clamped non-negative by the `.max(0)` above (F-01), and the
            // caller's documented contract guarantees `layer_buffer` is sized so the `in_ch`-wide
            // tap at that offset is in bounds; the prefetch calls only touch the address, not the
            // memory contents.
            unsafe {
                *tap = layer_buffer.as_ptr().add(in_start);
                if self.dilation >= 128 {
                    prefetch_strategy_2stage(*tap, self.dilation * in_ch, k, kernel, self.dilation);
                } else {
                    prefetch_strategy_simple(*tap, self.dilation * in_ch, k, kernel, self.dilation);
                }
            }
        }

        // SAFETY: the `process_blocks_*_nocopy` helpers are `unsafe fn`s reached only under the
        // preconditions of `process_single_frame` (valid `out_frame` of `out_ch` lanes, tap
        // pointers set above from the in-bounds `layer_buffer`, `kernel <= MAX_KERNEL`), and `M`
        // is selected by the runtime CPUID dispatch matching its `#[target_feature]` backend.
        unsafe {
            match self.interleave_width {
                16 => {
                    self.process_blocks_16_nocopy::<M>(out_frame, kernel, &tap_ptrs, in_ch, mixin)
                }
                8 => self.process_blocks_8_nocopy::<M>(out_frame, kernel, &tap_ptrs, in_ch, mixin),
                _ => self.process_blocks_4_nocopy::<M>(out_frame, kernel, &tap_ptrs, in_ch, mixin),
            }
        }
    }

    #[inline(always)]
    unsafe fn process_blocks_4_nocopy<M: SimdMath>(
        &self,
        out_frame: &mut [f32],
        kernel: usize,
        tap_ptrs: &[*const f32],
        in_ch: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch.div_ceil(4);
        for b in 0..num_blocks {
            let out_c = b * 4;
            let w = 4.min(self.out_ch - out_c);
            let (mu0, mu1, mu2, mu3) =
                // SAFETY: `load_mixin_4` internally guards every lane access with
                // `out_c + i < m.len()` before dereferencing, so `get_unchecked` stays in bounds
                // for any `mixin` slice (or returns zeros when `mixin` is `None`).
                unsafe { Self::load_mixin_4(mixin, out_c) };

            let mut acc = [0.0f32; 4];
            // SAFETY: `bias.len() == out_ch` and `out_c = b * 4 < out_ch` for
            // `b < out_ch.div_ceil(4)`, so the `get_unchecked(out_c)` access is in bounds; lanes
            // `out_c + 1..out_c + 3` are each guarded by `out_c + i < self.out_ch` before
            // dereferencing.
            unsafe {
                if self.do_bias {
                    acc[0] = *self.bias.get_unchecked(out_c) + mu0;
                    acc[1] = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mu1;
                    acc[2] = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mu2;
                    acc[3] = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mu3;
                } else {
                    acc[0] = mu0;
                    acc[1] = mu1;
                    acc[2] = mu2;
                    acc[3] = mu3;
                }
            }

            for k in 0..kernel {
                let w_start = b * kernel * in_ch * 4 + k * in_ch * 4;
                // F-16: the interleaved-4 slice covers `in_ch` taps of `[f32; 4]`;
                // it must lie within the zero-padded weights buffer.
                debug_assert!(
                    w_start + 4 * in_ch <= self.weights.len(),
                    "conv1d_dyn: interleave-4 weight slice exceeds padded weights buffer"
                );
                // SAFETY: the `debug_assert!` above proves `w_start + 4 * in_ch` lies within the
                // zero-padded weights buffer, so the slice of `in_ch` `[f32; 4]` blocks is in
                // bounds; `self.weights` is an `AlignedVec<f32>` aligned to 64 bytes.
                let w_slice: &[[f32; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                    core::slice::from_raw_parts(ptr, in_ch)
                };
                // SAFETY: `k < kernel <= MAX_KERNEL` (loader-enforced, see `MAX_KERNEL_SIZE`), so
                // `tap_ptrs.get_unchecked(k)` is in bounds, and the tap pointer was set above from
                // `layer_buffer` covering `in_ch` initialized f32s (caller contract).
                let tap_slice =
                    unsafe { core::slice::from_raw_parts(*tap_ptrs.get_unchecked(k), in_ch) };
                // SAFETY: `w_slice` (`in_ch` `[f32; 4]` blocks), `tap_slice` (`in_ch` f32s) and
                // `acc` (`[f32; 4]`) match the 4-wide accumulate kernel's lane counts, and `M` is
                // selected by the runtime CPUID dispatch matching its `#[target_feature]` backend.
                acc = unsafe { M::dot_product_4x_f32_accumulate(w_slice, tap_slice, &acc) };
            }

            // SAFETY: the fast path is guarded by `out_c + 3 < self.out_ch`, and the tail path
            // writes at most `w = 4.min(self.out_ch - out_c)` lanes so `out_c + lane < out_ch`;
            // `out_frame` has `out_ch` lanes (caller contract).
            unsafe {
                if out_c + 3 < self.out_ch {
                    *out_frame.get_unchecked_mut(out_c) = acc[0];
                    *out_frame.get_unchecked_mut(out_c + 1) = acc[1];
                    *out_frame.get_unchecked_mut(out_c + 2) = acc[2];
                    *out_frame.get_unchecked_mut(out_c + 3) = acc[3];
                } else {
                    for (lane, &val) in acc.iter().enumerate().take(w) {
                        *out_frame.get_unchecked_mut(out_c + lane) = val;
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn process_blocks_8_nocopy<M: SimdMath>(
        &self,
        out_frame: &mut [f32],
        kernel: usize,
        tap_ptrs: &[*const f32],
        in_ch: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch.div_ceil(8);
        for b in 0..num_blocks {
            let out_c = b * 8;
            let w = 8.min(self.out_ch - out_c);
            let mut acc = [0.0f32; 8];
            // SAFETY: `j < w = 8.min(self.out_ch - out_c)` so `out_c + j < out_ch == bias.len()`
            // (`bias` is an `AlignedVec` of `out_ch` elements); the mixin lane is additionally
            // guarded by `out_c + j < m.len()` before dereferencing.
            unsafe {
                for (j, item) in acc.iter_mut().enumerate().take(w) {
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
            }

            for k in 0..kernel {
                let w_start = b * kernel * in_ch * 8 + k * in_ch * 8;
                // F-16: the interleaved-8 slice covers `in_ch` taps of `[f32; 8]`;
                // it must lie within the zero-padded weights buffer.
                debug_assert!(
                    w_start + 8 * in_ch <= self.weights.len(),
                    "conv1d_dyn: interleave-8 weight slice exceeds padded weights buffer"
                );
                // SAFETY: the `debug_assert!` above proves `w_start + 8 * in_ch` lies within the
                // zero-padded weights buffer, so the slice of `in_ch` `[f32; 8]` blocks is in
                // bounds; `self.weights` is an `AlignedVec<f32>` aligned to 64 bytes.
                let w_slice: &[[f32; 8]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 8];
                    core::slice::from_raw_parts(ptr, in_ch)
                };
                // SAFETY: `k < kernel <= MAX_KERNEL` (loader-enforced, see `MAX_KERNEL_SIZE`), so
                // `tap_ptrs.get_unchecked(k)` is in bounds, and the tap pointer was set above from
                // `layer_buffer` covering `in_ch` initialized f32s (caller contract).
                let tap_slice =
                    unsafe { core::slice::from_raw_parts(*tap_ptrs.get_unchecked(k), in_ch) };
                // SAFETY: `w_slice` (`in_ch` `[f32; 8]` blocks), `tap_slice` (`in_ch` f32s) and
                // `acc` (`[f32; 8]`) match the 8-wide accumulate kernel's lane counts, and `M` is
                // selected by the runtime CPUID dispatch matching its `#[target_feature]` backend.
                acc = unsafe { M::dot_product_8x_f32_accumulate(w_slice, tap_slice, &acc) };
            }

            // SAFETY: `j < w = 8.min(self.out_ch - out_c)` so `out_c + j < out_ch`, and
            // `out_frame` has `out_ch` lanes (caller contract).
            unsafe {
                for (j, &item) in acc.iter().enumerate().take(w) {
                    *out_frame.get_unchecked_mut(out_c + j) = item;
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn process_blocks_16_nocopy<M: SimdMath>(
        &self,
        out_frame: &mut [f32],
        kernel: usize,
        tap_ptrs: &[*const f32],
        in_ch: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch.div_ceil(16);
        for b in 0..num_blocks {
            let out_c = b * 16;
            let w = 16.min(self.out_ch - out_c);
            let mut acc = [0.0f32; 16];
            // SAFETY: `j < w = 16.min(self.out_ch - out_c)` so `out_c + j < out_ch == bias.len()`
            // (`bias` is an `AlignedVec` of `out_ch` elements); the mixin lane is additionally
            // guarded by `out_c + j < m.len()` before dereferencing.
            unsafe {
                for (j, item) in acc.iter_mut().enumerate().take(w) {
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
            }

            for k in 0..kernel {
                let w_start = b * kernel * in_ch * 16 + k * in_ch * 16;
                // F-16: the interleaved-16 slice covers `in_ch` taps of `[f32; 16]`;
                // it must lie within the zero-padded weights buffer.
                debug_assert!(
                    w_start + 16 * in_ch <= self.weights.len(),
                    "conv1d_dyn: interleave-16 weight slice exceeds padded weights buffer"
                );
                // SAFETY: the `debug_assert!` above proves `w_start + 16 * in_ch` lies within the
                // zero-padded weights buffer, so the slice of `in_ch` `[f32; 16]` blocks is in
                // bounds; `self.weights` is an `AlignedVec<f32>` aligned to 64 bytes.
                let w_slice: &[[f32; 16]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 16];
                    core::slice::from_raw_parts(ptr, in_ch)
                };
                // SAFETY: `k < kernel <= MAX_KERNEL` (loader-enforced, see `MAX_KERNEL_SIZE`), so
                // `tap_ptrs.get_unchecked(k)` is in bounds, and the tap pointer was set above from
                // `layer_buffer` covering `in_ch` initialized f32s (caller contract).
                let tap_slice =
                    unsafe { core::slice::from_raw_parts(*tap_ptrs.get_unchecked(k), in_ch) };
                // SAFETY: `w_slice` (`in_ch` `[f32; 16]` blocks), `tap_slice` (`in_ch` f32s) and
                // `acc` (`[f32; 16]`) match the 16-wide accumulate kernel's lane counts, and `M`
                // is selected by the runtime CPUID dispatch matching its `#[target_feature]`
                // backend.
                acc = unsafe { M::dot_product_16x_f32_accumulate(w_slice, tap_slice, &acc) };
            }

            // SAFETY: `j < w = 16.min(self.out_ch - out_c)` so `out_c + j < out_ch`, and
            // `out_frame` has `out_ch` lanes (caller contract).
            unsafe {
                for (j, &item) in acc.iter().enumerate().take(w) {
                    *out_frame.get_unchecked_mut(out_c + j) = item;
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
            let m = mixin.map(|m| &m[i * self.out_ch..(i + 1) * self.out_ch]);
            // SAFETY: `process_single_frame` is an `unsafe fn` reached here with `rem` exactly
            // `out_ch` lanes (the remainder of `chunks_exact_mut(2 * self.out_ch)` over a
            // `num_frames * out_ch` block), `layer_buffer` per the caller contract, and the mixin
            // slice `m` sized `out_ch` when present.
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
