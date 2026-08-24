// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Static Causal CNN Mesh for WaveNet Inference (Data-Oriented Design, SoA).
//!
//! **Cohesion Justification:** Single static 1D convolution unit: `Conv1d` struct +
//! single-frame kernel + mixin wrappers form a cohesive algorithmic unit.
//! The f32-native dot-product helpers live in `conv_input.rs`.
//! Further splitting the single-frame kernel would break the locality
//! of `unsafe` aliasing contracts and plain accumulators.

use super::conv_input::{store_4_accums, store_8_accums, store_16_accums};
use crate::loader::dispatcher::wavenet::layout::select_interleave_width;
use crate::math::common::{
    AlignedVec, SimdMath, prefetch_strategy_2stage, prefetch_strategy_simple,
};

/// Dilated Causal Convolution (WaveNet Conv1D).
#[derive(Clone)]
#[repr(align(64))]
pub struct Conv1d<const IN: usize, const OUT: usize, const K: usize> {
    /// Flattened weight matrix of size OUT * K * IN in full-precision f32.
    pub weights: AlignedVec<f32>,
    /// Causal bias, applied if do_bias is true. Total: OUT.
    pub bias: AlignedVec<f32>,
    /// Determines if the bias array should be added.
    pub do_bias: bool,
    /// Dilation factor on the causal temporal axis (e.g.: 1, 2, 4.. 512).
    pub dilation: usize,
}

impl<const IN: usize, const OUT: usize, const K: usize> Conv1d<IN, OUT, K> {
    /// Processes a single frame applying convolution to the ring buffer,
    /// fusing a Mixin vector (conditioning) directly into the accumulator.
    ///
    /// Uses full-precision f32 weights via `M::dot_product_4x_f32` (AVX2/FMA or AVX-512 kernel).
    ///
    /// # Safety
    /// The caller must guarantee that `frame_idx`, `mixin`, `layer_buffer`,
    /// and `out_frame` have sizes compatible with the layer dimensions.
    #[inline(always)]
    pub unsafe fn process_single_frame_with_mixin<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: &[f32],
    ) {
        let interleave_width = select_interleave_width(OUT);
        let num_blocks = OUT.div_ceil(interleave_width);

        let mut in_taps = [[0.0f32; IN]; K];
        for (k, in_tap) in in_taps.iter_mut().enumerate() {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            // SAFETY: The causal receptive-field invariant guarantees
            // frame_idx >= dilation * (K-1), so (frame_idx as isize) + offset >= 0.
            debug_assert!(
                frame_idx >= self.dilation * (K - 1),
                "frame_idx {} must be >= dilation*K_minus_1 = {}",
                frame_idx,
                self.dilation * (K - 1)
            );
            let in_slice_start = ((frame_idx as isize) + offset) as usize * IN;
            // SAFETY: the `debug_assert!` above (`frame_idx >= dilation * (K - 1)`) keeps
            // `(frame_idx as isize) + offset` non-negative, and the caller contract guarantees
            // `layer_buffer` is sized for the causal receptive field, so the copy of `IN` f32s at
            // `in_slice_start` stays in bounds; `in_tap` is a `[f32; IN]` stack array and the two
            // buffers are distinct (no overlap).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer_buffer.as_ptr().add(in_slice_start),
                    in_tap.as_mut_ptr(),
                    IN,
                );
            }
            // SAFETY: `in_slice_start` is in bounds of `layer_buffer` (same invariant as the tap
            // copy above: `frame_idx >= dilation * (K - 1)` and the caller's size contract), so
            // `.add(in_slice_start)` is valid; `_mm_prefetch` only touches the address, not the
            // memory contents.
            unsafe {
                if self.dilation >= 128 {
                    prefetch_strategy_2stage(
                        layer_buffer.as_ptr().add(in_slice_start),
                        self.dilation * IN,
                        k,
                        K,
                        self.dilation,
                    );
                } else {
                    prefetch_strategy_simple(
                        layer_buffer.as_ptr().add(in_slice_start),
                        self.dilation * IN,
                        k,
                        K,
                        self.dilation,
                    );
                }
            }
        }

        let flat_taps: &[f32] =
            // SAFETY: `in_taps` is a `[[f32; IN]; K]` stack array with all `K * IN` f32
            // elements initialized by the tap copies above, so reinterpreting its storage as
            // `&[f32]` of length `K * IN` is valid; the pointer is non-null and `f32`-aligned.
            unsafe { core::slice::from_raw_parts(in_taps.as_ptr() as *const f32, K * IN) };

        for b in 0..num_blocks {
            let out_c = b * interleave_width;
            let w = interleave_width.min(OUT - out_c);
            let w_start = b * K * IN * interleave_width;

            match interleave_width {
                16 => {
                    let mut init = [0.0f32; 16];
                    for (j, item) in init.iter_mut().enumerate().take(w) {
                        if self.do_bias {
                            *item = self.bias[out_c + j] + mixin[out_c + j];
                        } else {
                            *item = mixin[out_c + j];
                        }
                    }
                    // F-16: the interleaved-16 block must be fully covered by the
                    // zero-padded weights buffer (padded to
                    // `num_blocks * 16 * K * IN` f32s by the loader).
                    debug_assert!(
                        w_start + 16 * K * IN <= self.weights.len(),
                        "conv1d: interleave-16 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the `debug_assert!` above proves `w_start + 16 * K * IN` lies within
                    // the zero-padded weights buffer, so the slice of `K * IN` `[f32; 16]` blocks
                    // is in bounds; `self.weights` is an `AlignedVec<f32>` aligned to 64 bytes.
                    let w_slice: &[[f32; 16]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 16];
                        core::slice::from_raw_parts(ptr, K * IN)
                    };
                    // SAFETY: `w_slice` (K*IN `[f32; 16]` blocks), `flat_taps` (K*IN f32s) and
                    // `init` (`[f32; 16]`) match the 16-wide accumulate kernel's required lane
                    // counts, and `M` is selected by the runtime CPUID dispatch matching its
                    // `#[target_feature]` backend.
                    let r = unsafe { M::dot_product_16x_f32_accumulate(w_slice, flat_taps, &init) };
                    // SAFETY: `out_c` is a block start with `out_c < OUT` and `out_frame` has
                    // `OUT` channels (caller contract), so the store stays in bounds;
                    // `store_16_accums` only takes the full-SIMD path when all 16 lanes are valid
                    // and falls back to guarded scalar writes otherwise.
                    unsafe { store_16_accums(out_frame, out_c, r, OUT) };
                }
                8 => {
                    let mut init = [0.0f32; 8];
                    for (j, item) in init.iter_mut().enumerate().take(w) {
                        if self.do_bias {
                            *item = self.bias[out_c + j] + mixin[out_c + j];
                        } else {
                            *item = mixin[out_c + j];
                        }
                    }
                    // F-16: see interleave-16 note; same boundary proof for width 8.
                    debug_assert!(
                        w_start + 8 * K * IN <= self.weights.len(),
                        "conv1d: interleave-8 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the `debug_assert!` above proves `w_start + 8 * K * IN` lies within
                    // the zero-padded weights buffer, so the slice of `K * IN` `[f32; 8]` blocks
                    // is in bounds; `self.weights` is an `AlignedVec<f32>` aligned to 64 bytes.
                    let w_slice: &[[f32; 8]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 8];
                        core::slice::from_raw_parts(ptr, K * IN)
                    };
                    // SAFETY: `w_slice` (K*IN `[f32; 8]` blocks), `flat_taps` (K*IN f32s) and
                    // `init` (`[f32; 8]`) match the 8-wide accumulate kernel's required lane
                    // counts, and `M` is selected by the runtime CPUID dispatch matching its
                    // `#[target_feature]` backend.
                    let r = unsafe { M::dot_product_8x_f32_accumulate(w_slice, flat_taps, &init) };
                    // SAFETY: `out_c` is a block start with `out_c < OUT` and `out_frame` has
                    // `OUT` channels (caller contract), so the store stays in bounds;
                    // `store_8_accums` only takes the full-SIMD path when all 8 lanes are valid
                    // and falls back to guarded scalar writes otherwise.
                    unsafe { store_8_accums(out_frame, out_c, r, OUT) };
                }
                _ => {
                    let mut init = [0.0f32; 4];
                    for (j, item) in init.iter_mut().enumerate().take(w) {
                        if self.do_bias {
                            *item = self.bias[out_c + j] + mixin[out_c + j];
                        } else {
                            *item = mixin[out_c + j];
                        }
                    }
                    // F-16: see interleave-16 note; same boundary proof for width 4.
                    debug_assert!(
                        w_start + 4 * K * IN <= self.weights.len(),
                        "conv1d: interleave-4 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the `debug_assert!` above proves `w_start + 4 * K * IN` lies within
                    // the zero-padded weights buffer, so the slice of `K * IN` `[f32; 4]` blocks
                    // is in bounds; `self.weights` is an `AlignedVec<f32>` aligned to 64 bytes.
                    let w_slice: &[[f32; 4]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                        core::slice::from_raw_parts(ptr, K * IN)
                    };
                    // SAFETY: `w_slice` (K*IN `[f32; 4]` blocks), `flat_taps` (K*IN f32s) and
                    // `init` (`[f32; 4]`) match the 4-wide accumulate kernel's required lane
                    // counts, and `M` is selected by the runtime CPUID dispatch matching its
                    // `#[target_feature]` backend.
                    let r = unsafe { M::dot_product_4x_f32_accumulate(w_slice, flat_taps, &init) };
                    // SAFETY: `out_c` is a block start with `out_c < OUT` and `out_frame` has
                    // `OUT` channels (caller contract), so the store stays in bounds;
                    // `store_4_accums` only takes the full-SIMD path when all 4 lanes are valid
                    // and falls back to guarded scalar writes otherwise.
                    unsafe { store_4_accums(out_frame, out_c, r, OUT) };
                }
            }
        }
    }

    /// Executes causal convolution over a flat bidirectional array (`layer_buffer`).
    ///
    /// # Safety
    /// Dynamically depends on the `SimdMath` trait provided.
    #[cfg(test)]
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
    ) {
        let interleave_width = select_interleave_width(OUT);
        let num_blocks = OUT.div_ceil(interleave_width);

        let mut in_taps = [[0.0f32; IN]; K];
        for (k, in_tap) in in_taps.iter_mut().enumerate() {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            // SAFETY: Receptive-field invariant: frame_idx >= dilation*(K-1).
            debug_assert!(
                frame_idx >= self.dilation * (K - 1),
                "frame_idx {} must be >= dilation*K_minus_1 = {}",
                frame_idx,
                self.dilation * (K - 1)
            );
            let in_slice_start = ((frame_idx as isize) + offset) as usize * IN;
            // SAFETY: the `debug_assert!` above (`frame_idx >= dilation * (K - 1)`) keeps
            // `(frame_idx as isize) + offset` non-negative, and the caller contract guarantees
            // `layer_buffer` is sized for the causal receptive field, so the copy of `IN` f32s at
            // `in_slice_start` stays in bounds; `in_tap` is a `[f32; IN]` stack array and the two
            // buffers are distinct (no overlap).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer_buffer.as_ptr().add(in_slice_start),
                    in_tap.as_mut_ptr(),
                    IN,
                );
            }
            // SAFETY: `in_slice_start` is in bounds of `layer_buffer` (same invariant as the tap
            // copy above: `frame_idx >= dilation * (K - 1)` and the caller's size contract), so
            // `.add(in_slice_start)` is valid; `_mm_prefetch` only touches the address, not the
            // memory contents.
            unsafe {
                if self.dilation >= 128 {
                    prefetch_strategy_2stage(
                        layer_buffer.as_ptr().add(in_slice_start),
                        self.dilation * IN,
                        k,
                        K,
                        self.dilation,
                    );
                } else {
                    prefetch_strategy_simple(
                        layer_buffer.as_ptr().add(in_slice_start),
                        self.dilation * IN,
                        k,
                        K,
                        self.dilation,
                    );
                }
            }
        }

        let flat_taps: &[f32] =
            // SAFETY: `in_taps` is a `[[f32; IN]; K]` stack array with all `K * IN` f32
            // elements initialized by the tap copies above, so reinterpreting its storage as
            // `&[f32]` of length `K * IN` is valid; the pointer is non-null and `f32`-aligned.
            unsafe { core::slice::from_raw_parts(in_taps.as_ptr() as *const f32, K * IN) };

        for b in 0..num_blocks {
            let out_c = b * interleave_width;
            let w = interleave_width.min(OUT - out_c);
            let w_start = b * K * IN * interleave_width;

            match interleave_width {
                16 => {
                    let mut init = [0.0f32; 16];
                    for (j, item) in init.iter_mut().enumerate().take(w) {
                        if self.do_bias {
                            *item = self.bias[out_c + j];
                        }
                    }
                    // F-16: see the mixin kernel; same boundary proof for width 16.
                    debug_assert!(
                        w_start + 16 * K * IN <= self.weights.len(),
                        "conv1d: interleave-16 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the `debug_assert!` above proves `w_start + 16 * K * IN` lies within
                    // the zero-padded weights buffer, so the slice of `K * IN` `[f32; 16]` blocks
                    // is in bounds; `self.weights` is an `AlignedVec<f32>` aligned to 64 bytes.
                    let w_slice: &[[f32; 16]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 16];
                        core::slice::from_raw_parts(ptr, K * IN)
                    };
                    // SAFETY: `w_slice` (K*IN `[f32; 16]` blocks), `flat_taps` (K*IN f32s) and
                    // `init` (`[f32; 16]`) match the 16-wide accumulate kernel's required lane
                    // counts, and `M` is selected by the runtime CPUID dispatch matching its
                    // `#[target_feature]` backend.
                    let r = unsafe { M::dot_product_16x_f32_accumulate(w_slice, flat_taps, &init) };
                    // SAFETY: `out_c` is a block start with `out_c < OUT` and `out_frame` has
                    // `OUT` channels (caller contract), so the store stays in bounds;
                    // `store_16_accums` only takes the full-SIMD path when all 16 lanes are valid
                    // and falls back to guarded scalar writes otherwise.
                    unsafe { store_16_accums(out_frame, out_c, r, OUT) };
                }
                8 => {
                    let mut init = [0.0f32; 8];
                    for (j, item) in init.iter_mut().enumerate().take(w) {
                        if self.do_bias {
                            *item = self.bias[out_c + j];
                        }
                    }
                    // F-16: see the mixin kernel; same boundary proof for width 8.
                    debug_assert!(
                        w_start + 8 * K * IN <= self.weights.len(),
                        "conv1d: interleave-8 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the `debug_assert!` above proves `w_start + 8 * K * IN` lies within
                    // the zero-padded weights buffer, so the slice of `K * IN` `[f32; 8]` blocks
                    // is in bounds; `self.weights` is an `AlignedVec<f32>` aligned to 64 bytes.
                    let w_slice: &[[f32; 8]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 8];
                        core::slice::from_raw_parts(ptr, K * IN)
                    };
                    // SAFETY: `w_slice` (K*IN `[f32; 8]` blocks), `flat_taps` (K*IN f32s) and
                    // `init` (`[f32; 8]`) match the 8-wide accumulate kernel's required lane
                    // counts, and `M` is selected by the runtime CPUID dispatch matching its
                    // `#[target_feature]` backend.
                    let r = unsafe { M::dot_product_8x_f32_accumulate(w_slice, flat_taps, &init) };
                    // SAFETY: `out_c` is a block start with `out_c < OUT` and `out_frame` has
                    // `OUT` channels (caller contract), so the store stays in bounds;
                    // `store_8_accums` only takes the full-SIMD path when all 8 lanes are valid
                    // and falls back to guarded scalar writes otherwise.
                    unsafe { store_8_accums(out_frame, out_c, r, OUT) };
                }
                _ => {
                    let mut init = [0.0f32; 4];
                    for (j, item) in init.iter_mut().enumerate().take(w) {
                        if self.do_bias {
                            *item = self.bias[out_c + j];
                        }
                    }
                    // F-16: see the mixin kernel; same boundary proof for width 4.
                    debug_assert!(
                        w_start + 4 * K * IN <= self.weights.len(),
                        "conv1d: interleave-4 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the `debug_assert!` above proves `w_start + 4 * K * IN` lies within
                    // the zero-padded weights buffer, so the slice of `K * IN` `[f32; 4]` blocks
                    // is in bounds; `self.weights` is an `AlignedVec<f32>` aligned to 64 bytes.
                    let w_slice: &[[f32; 4]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                        core::slice::from_raw_parts(ptr, K * IN)
                    };
                    // SAFETY: `w_slice` (K*IN `[f32; 4]` blocks), `flat_taps` (K*IN f32s) and
                    // `init` (`[f32; 4]`) match the 4-wide accumulate kernel's required lane
                    // counts, and `M` is selected by the runtime CPUID dispatch matching its
                    // `#[target_feature]` backend.
                    let r = unsafe { M::dot_product_4x_f32_accumulate(w_slice, flat_taps, &init) };
                    // SAFETY: `out_c` is a block start with `out_c < OUT` and `out_frame` has
                    // `OUT` channels (caller contract), so the store stays in bounds;
                    // `store_4_accums` only takes the full-SIMD path when all 4 lanes are valid
                    // and falls back to guarded scalar writes otherwise.
                    unsafe { store_4_accums(out_frame, out_c, r, OUT) };
                }
            }
        }
    }

    /// Processes a sequential iterative block.
    /// For cache efficiency, instead of processing the entire layer by multiple blocks,
    /// we limit calls to consecutive frame-by-frame calls (`process_single_frame`).
    ///
    /// # Safety
    /// Pointer must be valid and num_frames must fit within the layer_buffer bounds.
    #[cfg(test)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            // SAFETY: `i < num_frames` and `block` has at least `num_frames * OUT`
            // elements (caller contract), so `i * OUT..i * OUT + OUT` is in bounds.
            let out_frame = unsafe { block.get_unchecked_mut(i * OUT..i * OUT + OUT) };
            // SAFETY: `out_frame` is a valid `OUT`-element slice of `block`, and
            // `layer_buffer`/`buffer_start + i` satisfy `process_single_frame`'s causal
            // receptive-field contract (warm-up invariant asserted inside the kernel).
            unsafe {
                self.process_single_frame::<M>(layer_buffer, out_frame, buffer_start + i);
            }
        }
    }
}
