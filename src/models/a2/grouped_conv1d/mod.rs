// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Grouped dilated causal Conv1D (`groups > 1`) for the A2 architecture.
//!
//! FiLM modulation requires `_cond_to_scale_shift` — a `Conv1x1` that maps
//! condition channels to scale (and optionally shift) channels using `groups > 1`
//! for independent per-group projections (C++ `NAM/film.h`).
//!
//! ## Layout
//!
//! NAM JSON weight order: `raw[out_ch][in_ch][kernel]` (row-major).
//! Internal grouped-interleaved-4-wide: `[group][num_blocks][kernel][in_per_group][4]`
//!
//! For each `(group, block, kernel-tap, input-channel)` the 4 output-channel weights
//! are contiguous — one `_mm_loadu_ps` load, one `_mm_fmadd_ps` broadcast-FMA.
//!
//! ## Fast-path
//!
//! When `groups == 1` the computation is identical to a standard dilated Conv1D —
//! this module delegates directly to `Conv1dDyn` to preserve the extreme fast-path.
//!
//! ## Source of truth
//! - `NAM/conv1d.h:11-136` (`Conv1D` with `_num_groups`, `_is_depthwise`)
//! - `NAM/conv1d.cpp:55-252` (grouped weight loading + depthwise execution)
//! - `NAM/film.h:28-85` (`_cond_to_scale_shift` pattern)

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;

pub mod reference;
pub mod simd;

pub use reference::{grouped_conv1d_block_ref, grouped_conv1d_single_frame_ref};

use simd::{grouped_conv1d_single_frame_simd, process_single_frame_depthwise_avx2};

/// Grouped dilated causal Conv1D with interleaved-4-wide f32 weights.
///
/// Divides input and output channels into `groups` independent sub-convolutions.
/// Each group operates on `in_per_group = in_ch / groups` input channels and
/// produces `out_per_group = out_ch / groups` output channels — no cross-group
/// connections.
///
/// When `groups == in_ch && groups == out_ch` this is a **depthwise** convolution:
/// each channel is filtered independently.
#[derive(Clone)]
#[repr(align(64))]
pub struct A2GroupedConv1d {
    /// Grouped-interleaved-4-wide f32 weights.
    /// Layout: `[group][num_blocks][kernel][in_per_group][4]`
    pub weights: AlignedVec<f32>,
    /// Bias vector `[out_ch]`.
    pub bias: AlignedVec<f32>,
    /// Whether to add bias to the accumulator.
    pub do_bias: bool,
    /// Temporal dilation factor.
    pub dilation: usize,
    /// Number of input channels (must be divisible by `groups`).
    pub in_ch: usize,
    /// Number of output channels (must be divisible by `groups`).
    pub out_ch: usize,
    /// Number of groups for the grouped convolution.
    pub groups: usize,
    /// Input channels per group (`in_ch / groups`).
    pub in_per_group: usize,
    /// Output channels per group (`out_ch / groups`).
    pub out_per_group: usize,
    /// Number of 4-channel blocks per group (`ceil(out_per_group / 4)`).
    pub num_blocks_per_group: usize,
    /// Total 4-channel blocks across all groups (`groups * num_blocks_per_group`).
    pub total_blocks: usize,
    /// Kernel size.
    pub kernel: usize,
}

impl A2GroupedConv1d {
    /// Builds a grouped conv1d from raw NAM JSON row-major weights.
    ///
    /// `raw_weights` is in `[out_ch][in_ch][kernel]` order.
    /// The constructor permutes to grouped-interleaved-4-wide:
    /// `[group][num_blocks][kernel][in_per_group][4]`.
    ///
    /// # Panics
    /// Panics if `in_ch % groups != 0` or `out_ch % groups != 0`.
    pub fn new(
        raw_weights: &[f32],
        raw_bias: &[f32],
        do_bias: bool,
        dilation: usize,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        groups: usize,
    ) -> Result<Self, NamErrorCode> {
        assert!(groups > 0, "groups must be > 0");
        assert_eq!(
            in_ch % groups,
            0,
            "in_ch={} must be divisible by groups={}",
            in_ch,
            groups
        );
        assert_eq!(
            out_ch % groups,
            0,
            "out_ch={} must be divisible by groups={}",
            out_ch,
            groups
        );
        assert_eq!(
            raw_weights.len(),
            out_ch * in_ch * kernel,
            "raw_weights len {} != expected {} (out_ch={} * in_ch={} * kernel={})",
            raw_weights.len(),
            out_ch * in_ch * kernel,
            out_ch,
            in_ch,
            kernel
        );

        let in_per_group = in_ch / groups;
        let out_per_group = out_ch / groups;
        let num_blocks_per_group = out_per_group.div_ceil(4);
        let total_blocks = groups * num_blocks_per_group;
        let total_padded = total_blocks * 4 * in_per_group * kernel;

        let mut weights = AlignedVec::new(total_padded, 0.0f32)?;
        let bias = AlignedVec::from_vec(raw_bias.to_vec())?;

        for g in 0..groups {
            let group_in_start = g * in_per_group;
            let group_out_start = g * out_per_group;
            for b in 0..num_blocks_per_group {
                for k in 0..kernel {
                    for ic in 0..in_per_group {
                        for lane in 0..4 {
                            let out_idx = b * 4 + lane;
                            if out_idx < out_per_group {
                                let src = (group_out_start + out_idx) * in_ch * kernel
                                    + (group_in_start + ic) * kernel
                                    + k;
                                let dst = g * (num_blocks_per_group * kernel * in_per_group * 4)
                                    + b * (kernel * in_per_group * 4)
                                    + k * (in_per_group * 4)
                                    + ic * 4
                                    + lane;
                                weights[dst] = raw_weights[src];
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            weights,
            bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            groups,
            in_per_group,
            out_per_group,
            num_blocks_per_group,
            total_blocks,
            kernel,
        })
    }

    /// Processes a single frame through the grouped dilated convolution using SIMD AVX2.
    /// Dispatches to the depthwise fast-path when `groups == in_ch == out_ch`,
    /// otherwise uses the general grouped AVX2 kernel.
    ///
    /// # Safety
    /// `layer_buffer` must contain valid elements for the dilated tap indices.
    /// `out_frame` must have length at least `self.out_ch`.
    #[inline(always)]
    pub unsafe fn process_single_frame(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        // SAFETY-CRITICAL: these preconditions gate the raw pointer
        // arithmetic performed by `process_single_frame_depthwise_avx2`/
        // `grouped_conv1d_single_frame_simd` below (`frame_idx` in particular
        // drives an offset into `layer_buffer` computed via
        // signed-then-unsigned arithmetic that wraps to a huge value if
        // `frame_idx` is too small). Must be `assert!`, not `debug_assert!`
        // — confirmed to cause a SIGSEGV in release when elided.
        assert!(
            out_frame.len() >= self.out_ch,
            "out_frame len {} < out_ch {}",
            out_frame.len(),
            self.out_ch
        );
        let lookback = self.dilation * (self.kernel.saturating_sub(1));
        assert!(
            frame_idx >= lookback,
            "frame_idx {} < lookback {} (dilation={} * kernel-1={})",
            frame_idx,
            lookback,
            self.dilation,
            self.kernel.saturating_sub(1)
        );
        assert!(
            layer_buffer.len() > frame_idx * self.in_ch,
            "layer_buffer len {} <= frame_idx {} * in_ch {}",
            layer_buffer.len(),
            frame_idx,
            self.in_ch
        );
        assert!(
            mixin.is_none_or(|m| m.len() >= self.out_ch),
            "mixin len {:?} < out_ch {}",
            mixin.map(|m| m.len()),
            self.out_ch
        );

        // SAFETY: the `assert!`s above guarantee the documented preconditions of this
        // `unsafe fn`: `out_frame.len() >= self.out_ch`, `frame_idx >= lookback`,
        // `layer_buffer.len() > frame_idx * self.in_ch`, and the `mixin` length check,
        // keeping the raw pointer arithmetic in the kernels in-bounds.
        unsafe {
            if self.groups == self.in_ch && self.groups == self.out_ch {
                process_single_frame_depthwise_avx2(self, layer_buffer, out_frame, frame_idx);
                if let Some(m) = mixin {
                    for c in 0..self.out_ch {
                        *out_frame.get_unchecked_mut(c) += *m.get_unchecked(c);
                    }
                }
            } else {
                grouped_conv1d_single_frame_simd(self, layer_buffer, out_frame, frame_idx, mixin);
            }
        }
    }

    /// Processes a block of `num_frames` consecutive frames.
    ///
    /// # Safety
    /// `layer_buffer` must be large enough for the dilated lookback.
    /// `block` must have size at least `num_frames * out_ch`.
    #[cfg(test)]
    #[inline(always)]
    pub unsafe fn process_block(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        // SAFETY-CRITICAL: gates the per-frame slicing below, which feeds
        // directly into the unsafe SIMD kernel's raw pointer arithmetic.
        // Must be `assert!`, not `debug_assert!` — see note in
        // `process_single_frame` above.
        assert!(
            block.len() >= num_frames * self.out_ch,
            "process_block: block len {} < num_frames {} * out_ch {}",
            block.len(),
            num_frames,
            self.out_ch
        );
        assert!(
            mixin.is_none_or(|m| m.len() >= num_frames * self.out_ch),
            "process_block: mixin len {:?} < num_frames {} * out_ch {}",
            mixin.map(|m| m.len()),
            num_frames,
            self.out_ch
        );
        for f in 0..num_frames {
            let out_slice = &mut block[f * self.out_ch..(f + 1) * self.out_ch];
            let m = mixin.map(|full| &full[f * self.out_ch..(f + 1) * self.out_ch]);
            // SAFETY: the asserts above guarantee `out_slice` has exactly `self.out_ch`
            // elements and `m` (if present) has at least `self.out_ch`; the kernel
            // re-asserts the `layer_buffer`/`frame_idx` lookback bounds before any raw
            // pointer arithmetic.
            unsafe {
                grouped_conv1d_single_frame_simd(
                    self,
                    layer_buffer,
                    out_slice,
                    buffer_start + f,
                    m,
                );
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn make_test_weights_grouped(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    _groups: usize,
    seed: u32,
) -> (Vec<f32>, Vec<f32>) {
    let total_w = out_ch * in_ch * kernel;
    let mut raw_weights = Vec::with_capacity(total_w);
    let mut state = seed;
    for _ in 0..total_w {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        raw_weights.push(((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25);
    }
    let mut bias = Vec::with_capacity(out_ch);
    for _ in 0..out_ch {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        bias.push(((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1);
    }
    (raw_weights, bias)
}

#[cfg(test)]
pub(crate) fn make_layer_buffer(buf_frames: usize, in_ch: usize, seed: u32) -> Vec<f32> {
    let mut buf = vec![0.0f32; buf_frames * in_ch];
    let mut state = seed;
    for val in &mut buf {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
    }
    buf
}

#[cfg(test)]
#[path = "../grouped_conv1d_test.rs"]
mod tests;
