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
///
/// **Layout invariant (F-16 / R-2):** the hot path reinterprets `weights` as
/// SIMD-interleaved `[f32; W]` blocks (W = 4, 8 or 16, selected from `OUT`),
/// reading up to `num_blocks * W * K * IN` f32s where
/// `num_blocks = OUT.div_ceil(W)`. `weights` must therefore hold **at least**
/// that SIMD-padded total. Prefer [`Conv1d::try_from_parts`], which validates
/// the padded length in a release-stable check off the real-time path; direct
/// struct-literal construction must uphold the invariant manually.
#[derive(Clone)]
#[repr(align(64))]
pub struct Conv1d<const IN: usize, const OUT: usize, const K: usize> {
    /// Interleaved weight matrix `[num_blocks][K][IN][W]` in full-precision f32,
    /// zero-padded so every `[f32; W]` block is fully covered
    /// (length >= `OUT.div_ceil(W) * W * K * IN`).
    pub weights: AlignedVec<f32>,
    /// Causal bias, applied if do_bias is true. Total: OUT.
    pub bias: AlignedVec<f32>,
    /// Determines if the bias array should be added.
    pub do_bias: bool,
    /// Dilation factor on the causal temporal axis (e.g.: 1, 2, 4.. 512).
    pub dilation: usize,
}

impl<const IN: usize, const OUT: usize, const K: usize> Conv1d<IN, OUT, K> {
    /// Validated constructor (release-stable, off-RT) for a static `Conv1d`.
    ///
    /// This is the release-stable owner of the interleaved-weights padding
    /// invariant (F-16 / R-2). The hot-path kernels derive their
    /// `[f32; W]`-block slices from `self.weights` for every output block
    /// `b < OUT.div_ceil(W)`, whose last element ends at
    /// `OUT.div_ceil(W) * W * K * IN`; rejecting smaller buffers here keeps
    /// every `from_raw_parts` in the process methods structurally in bounds in
    /// release builds — no hot-path check is needed or performed.
    ///
    /// # Errors
    /// Returns an error if `weights` holds fewer than the SIMD-padded total
    /// `OUT.div_ceil(W) * W * K * IN` f32s (with `W` the interleave width for
    /// `OUT`), which would make the interleaved SIMD kernels read out of
    /// bounds on the hot path.
    #[inline]
    pub fn try_from_parts(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        dilation: usize,
    ) -> anyhow::Result<Self> {
        let interleave_width = select_interleave_width(OUT);
        let num_blocks = OUT.div_ceil(interleave_width);
        let padded_total = num_blocks * interleave_width * IN * K;
        anyhow::ensure!(
            weights.len() >= padded_total,
            "Conv1d weights buffer is too small: expected >= {padded_total} \
             (SIMD-padded, interleave width {interleave_width}), got {}",
            weights.len()
        );
        Ok(Self {
            weights,
            bias,
            do_bias,
            dilation,
        })
    }

    /// Processes a single frame applying convolution to the ring buffer,
    /// fusing a Mixin vector (conditioning) directly into the accumulator.
    ///
    /// Uses full-precision f32 weights via `M::dot_product_4x_f32` (AVX2/FMA or AVX-512 kernel).
    ///
    /// # Safety
    /// The caller must guarantee that `frame_idx`, `mixin`, `layer_buffer`,
    /// and `out_frame` have sizes compatible with the layer dimensions.
    ///
    /// The causal tap read requires `frame_idx >= dilation * (K - 1)` for correct
    /// audio; that invariant is owned release-stable by the layer-state construction
    /// (`WaveNetLayerState::new`, see `common.rs`), and the kernel clamps the tap
    /// offset to the buffer start (F-01) so a violating caller still cannot produce
    /// an out-of-bounds read in release builds.
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
            // R-2 / A4: the signed tap offset is clamped to 0 **before** the `as usize`
            // conversion, so a frame index below the warm-up threshold can never wrap into
            // a huge offset that would make `.add(in_slice_start)` read out of bounds.
            // Release-stable (F-01 pattern, same as `Conv1dDyn::process_single_frame`):
            // a contract violation becomes a defined in-bounds read of the buffer start,
            // never UB. On the model path the clamp is statically inactive — the invariant
            // `frame_idx >= dilation * (K-1)` is owned release-stable by the layer-state
            // construction (`WaveNetLayerState::new` enforces
            // `buffer_start >= receptive_field_size >= dilation*(K-1)`; see `common.rs`).
            let in_slice_start = (((frame_idx as isize) + offset).max(0)) as usize * IN;
            // SAFETY: `in_slice_start` is non-negative by the `.max(0)` clamp above (F-01) and
            // `(frame_idx as isize) + offset <= frame_idx`, so it stays below
            // `frame_idx * IN < layer_buffer.len()` (the mirrored layer buffer spans
            // `2 * buffer_frames * IN` elements and `frame_idx <= 2 * buffer_frames - 1` via the
            // `buffer_start + WAVENET_MAX_NUM_FRAMES <= 2 * buffer_frames` wrap margin); the
            // caller contract guarantees `layer_buffer` is sized for the causal receptive field,
            // so the copy of `IN` f32s at `in_slice_start` stays in bounds; `in_tap` is a
            // `[f32; IN]` stack array and the two buffers are distinct (no overlap).
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
                    // PROOF (release-stable, R-2): `weights.len() >= num_blocks * W * K * IN`
                    // is enforced by the validated constructors (`Conv1d::try_from_parts` /
                    // `ConvWeightsOutput::from_parts`), so `b < num_blocks` implies
                    // `w_start + 16 * K * IN <= padded_total <= weights.len()`. The
                    // `debug_assert!` below is a redundant debug-only net.
                    debug_assert!(
                        w_start + 16 * K * IN <= self.weights.len(),
                        "conv1d: interleave-16 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the construction-time padding invariant above proves
                    // `w_start + 16 * K * IN` lies within the zero-padded weights buffer, so the
                    // slice of `K * IN` `[f32; 16]` blocks is in bounds; `self.weights` is an
                    // `AlignedVec<f32>` aligned to 64 bytes.
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
                    // PROOF (release-stable, R-2): `weights.len() >= num_blocks * W * K * IN`
                    // is enforced by the validated constructors (`Conv1d::try_from_parts` /
                    // `ConvWeightsOutput::from_parts`), so `b < num_blocks` implies
                    // `w_start + 8 * K * IN <= padded_total <= weights.len()`. The
                    // `debug_assert!` below is a redundant debug-only net.
                    debug_assert!(
                        w_start + 8 * K * IN <= self.weights.len(),
                        "conv1d: interleave-8 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the construction-time padding invariant above proves
                    // `w_start + 8 * K * IN` lies within the zero-padded weights buffer, so the
                    // slice of `K * IN` `[f32; 8]` blocks is in bounds; `self.weights` is an
                    // `AlignedVec<f32>` aligned to 64 bytes.
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
                    // PROOF (release-stable, R-2): `weights.len() >= num_blocks * W * K * IN`
                    // is enforced by the validated constructors (`Conv1d::try_from_parts` /
                    // `ConvWeightsOutput::from_parts`), so `b < num_blocks` implies
                    // `w_start + 4 * K * IN <= padded_total <= weights.len()`. The
                    // `debug_assert!` below is a redundant debug-only net.
                    debug_assert!(
                        w_start + 4 * K * IN <= self.weights.len(),
                        "conv1d: interleave-4 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the construction-time padding invariant above proves
                    // `w_start + 4 * K * IN` lies within the zero-padded weights buffer, so the
                    // slice of `K * IN` `[f32; 4]` blocks is in bounds; `self.weights` is an
                    // `AlignedVec<f32>` aligned to 64 bytes.
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
            // R-2 / A4: same release-stable `.max(0)` clamp as the mixin kernel above —
            // a sub-threshold `frame_idx` reads the buffer start instead of wrapping
            // through `as usize` into an out-of-bounds pointer (F-01).
            let in_slice_start = (((frame_idx as isize) + offset).max(0)) as usize * IN;
            // SAFETY: `in_slice_start` is non-negative by the `.max(0)` clamp above (F-01) and
            // bounded above by `frame_idx * IN` (offsets are <= 0), which lies within the
            // mirrored layer buffer (`frame_idx <= 2 * buffer_frames - 1`); the caller contract
            // guarantees `layer_buffer` is sized for the causal receptive field, so the copy of
            // `IN` f32s at `in_slice_start` stays in bounds; `in_tap` is a `[f32; IN]` stack
            // array and the two buffers are distinct (no overlap).
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
                    // PROOF (release-stable, R-2): `weights.len() >= num_blocks * W * K * IN`
                    // is enforced by the validated constructors (`Conv1d::try_from_parts` /
                    // `ConvWeightsOutput::from_parts`), so `b < num_blocks` implies
                    // `w_start + 16 * K * IN <= padded_total <= weights.len()`. The
                    // `debug_assert!` below is a redundant debug-only net.
                    debug_assert!(
                        w_start + 16 * K * IN <= self.weights.len(),
                        "conv1d: interleave-16 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the construction-time padding invariant above proves
                    // `w_start + 16 * K * IN` lies within the zero-padded weights buffer, so the
                    // slice of `K * IN` `[f32; 16]` blocks is in bounds; `self.weights` is an
                    // `AlignedVec<f32>` aligned to 64 bytes.
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
                    // PROOF (release-stable, R-2): `weights.len() >= num_blocks * W * K * IN`
                    // is enforced by the validated constructors (`Conv1d::try_from_parts` /
                    // `ConvWeightsOutput::from_parts`), so `b < num_blocks` implies
                    // `w_start + 8 * K * IN <= padded_total <= weights.len()`. The
                    // `debug_assert!` below is a redundant debug-only net.
                    debug_assert!(
                        w_start + 8 * K * IN <= self.weights.len(),
                        "conv1d: interleave-8 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the construction-time padding invariant above proves
                    // `w_start + 8 * K * IN` lies within the zero-padded weights buffer, so the
                    // slice of `K * IN` `[f32; 8]` blocks is in bounds; `self.weights` is an
                    // `AlignedVec<f32>` aligned to 64 bytes.
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
                    // PROOF (release-stable, R-2): `weights.len() >= num_blocks * W * K * IN`
                    // is enforced by the validated constructors (`Conv1d::try_from_parts` /
                    // `ConvWeightsOutput::from_parts`), so `b < num_blocks` implies
                    // `w_start + 4 * K * IN <= padded_total <= weights.len()`. The
                    // `debug_assert!` below is a redundant debug-only net.
                    debug_assert!(
                        w_start + 4 * K * IN <= self.weights.len(),
                        "conv1d: interleave-4 weight block exceeds padded weights buffer"
                    );
                    // SAFETY: the construction-time padding invariant above proves
                    // `w_start + 4 * K * IN` lies within the zero-padded weights buffer, so the
                    // slice of `K * IN` `[f32; 4]` blocks is in bounds; `self.weights` is an
                    // `AlignedVec<f32>` aligned to 64 bytes.
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
            // receptive-field contract (owned release-stable by `WaveNetLayerState`'s
            // construction; the kernel additionally clamps under-threshold taps, F-01).
            unsafe {
                self.process_single_frame::<M>(layer_buffer, out_frame, buffer_start + i);
            }
        }
    }
}
