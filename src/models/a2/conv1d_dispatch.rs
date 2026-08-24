// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Inference dispatch for `A2Conv1d` — delegates to `Conv1dDyn` (groups=1) or
//! `A2GroupedConv1d` (groups>1) with AVX2 depthwise fast‑path auto‑selection.
//!
//! All hot-path methods are monomorphized over `<M: SimdMath>`, receiving the ISA
//! type from the top-level `dispatch_simd!` in the model's `process` method.

use crate::math::common::SimdMath;

use super::A2Conv1d;

impl A2Conv1d {
    /// Processes a single frame (f32) through the dilated convolution.
    ///
    /// Uses the SIMD-accelerated path: `Conv1dDyn` for groups=1,
    /// AVX2 grouped kernel (with depthwise dispatch) for groups>1.
    ///
    /// `M` is the ISA monomorphization type propagated from the top-level
    /// `dispatch_simd!` — eliminates per-frame `is_x86_feature_detected` branches.
    ///
    /// # Safety
    /// `layer_buffer` must contain valid elements for the dilated tap indices.
    /// `out_frame` must have length at least `self.out_ch`.
    /// `frame_idx` must allow `kernel` lookback taps within `layer_buffer`.
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        // SAFETY: this `unsafe fn` is only reachable under its documented preconditions:
        // `layer_buffer` holds valid elements for the dilated tap indices, `out_frame`
        // has length at least `self.out_ch`, and `frame_idx` allows kernel lookback taps
        // within `layer_buffer`.
        unsafe {
            match self {
                Self::Standard(c) => {
                    c.process_single_frame::<M>(layer_buffer, out_frame, frame_idx, mixin);
                }
                Self::Grouped(g) => {
                    g.process_single_frame(layer_buffer, out_frame, frame_idx, mixin);
                }
            }
        }
    }

    /// Processes a block of `num_frames` consecutive frames (f32).
    ///
    /// `M` is the ISA monomorphization type propagated from the top-level
    /// `dispatch_simd!`.
    ///
    /// # Safety
    /// `layer_buffer` must be large enough for `buffer_start..buffer_start + num_frames`
    /// plus kernel*dilation lookback. `block` must have size at least `num_frames * out_ch`.
    #[cfg(test)]
    #[inline(always)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        // SAFETY: this `unsafe fn` is only reachable under its documented preconditions:
        // `layer_buffer` covers `buffer_start..buffer_start + num_frames` plus
        // kernel*dilation lookback, and `block` has at least `num_frames * out_ch` elements.
        unsafe {
            match self {
                Self::Standard(c) => {
                    c.process_block::<M>(layer_buffer, block, buffer_start, num_frames, mixin);
                }
                Self::Grouped(g) => {
                    g.process_block(layer_buffer, block, buffer_start, num_frames, mixin);
                }
            }
        }
    }

    /// Returns a reference to the inner `Conv1dDyn` for the `Standard` variant.
    ///
    /// # Panics
    /// Panics if called on a `Grouped` variant — only valid in tests of the
    /// standard const-generic fast-path.
    #[cfg(test)]
    pub fn standard_inner(&self) -> &crate::models::wavenet::conv1d_dyn::Conv1dDyn {
        match self {
            Self::Standard(c) => c,
            Self::Grouped(_) => panic!("standard_inner called on Grouped variant"),
        }
    }
}
