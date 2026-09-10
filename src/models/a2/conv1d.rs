// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dilated causal Conv1D for the A2 architecture (kernel sizes 6 and 15).
//!
//! Supports two variants:
//! - `Standard` — groups=1, delegates to the battle-tested `Conv1dDyn` engine.
//! - `Grouped` — groups>1, uses the AVX2 `A2GroupedConv1d` with depthwise fast-path.
//!
//! Operates over `MirrorBuffer`-backed slices — the dilation tap pointers
//! access a contiguous virtual window where physical wrap is handled by the
//! mirrored mapping, eliminating branch logic in the inner loop.

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;

use super::grouped_conv1d::A2GroupedConv1d;

/// A2-specific dilated causal conv1d — polymorphic over grouping.
///
/// When `groups == 1` the standard interleaved-4-wide `Conv1dDyn` is used.
/// When `groups > 1` the grouped-interleaved-4-wide `A2GroupedConv1d` is used,
/// with automatic depthwise dispatch when `groups == in_ch == out_ch`.
#[derive(Clone)]
pub enum A2Conv1d {
    /// Standard conv (groups=1). Uses interleaved-4-wide `Conv1dDyn`.
    Standard(Conv1dDyn),
    /// Grouped conv (groups>1). Uses grouped-interleaved-4-wide AVX2 kernel.
    Grouped(A2GroupedConv1d),
}

impl A2Conv1d {
    /// Builds an A2 conv1d with pre-validated A2 parameters (groups=1).
    ///
    /// The A2 loader always stores the fallback `Conv1dDyn` weights in
    /// interleaved-4-wide layout (`transpose_conv1d_interleaved_4wide`), so the
    /// validated constructor is invoked with `interleave_width = 4`.
    ///
    /// Attempts to build an A2 conv1d with pre-validated A2 parameters (groups=1).
    ///
    /// The A2 loader always stores the fallback `Conv1dDyn` weights in
    /// interleaved-4-wide layout (`transpose_conv1d_interleaved_4wide`), so the
    /// validated constructor is invoked with `interleave_width = 4`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - `kernel_size` is 0;
    /// - `in_ch` or `out_ch` is 0;
    /// - `bias.len() < out_ch`;
    /// - weights buffer size is insufficient for interleaved-4 SIMD padding.
    pub fn try_new(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        dilation: usize,
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
    ) -> Result<Self, String> {
        if kernel_size < 1 {
            return Err("A2 kernel size must be >= 1".to_string());
        }
        if in_ch == 0 || out_ch == 0 {
            return Err("channels must be > 0".to_string());
        }
        if bias.len() < out_ch {
            return Err("bias buffer length is smaller than out_ch".to_string());
        }

        let conv = Conv1dDyn::try_from_parts(
            weights,
            bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel_size,
            4,
        )
        .map_err(|e| {
            let mut msg = String::from("A2 fallback Conv1dDyn validation failed: ");
            msg.push_str(&e.to_string());
            msg
        })?;

        Ok(Self::Standard(conv))
    }

    /// Builds an A2 conv1d with pre-validated A2 parameters (groups=1).
    ///
    /// # Panics
    /// Panics if the weights buffer is sub-dimensioned for the interleaved-4
    /// padded layout or the kernel/channel parameters are invalid.
    /// Prefer [`try_new`](Self::try_new) in fallible/hostile model loading paths.
    pub fn new(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        dilation: usize,
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
    ) -> Self {
        Self::try_new(weights, bias, do_bias, dilation, in_ch, out_ch, kernel_size)
            .expect("A2Conv1d::new: invalid conv parameters or undersized buffer")
    }

    /// Builds a grouped A2 conv1d from raw NAM JSON row-major weights.
    ///
    /// Wraps `A2GroupedConv1d::new()`. The raw_weights are in
    /// `[out_ch][in_ch][kernel]` order and are permuted to
    /// grouped-interleaved-4-wide internally.
    ///
    /// # Panics
    /// Panics in debug if `in_ch % groups != 0` or `out_ch % groups != 0`.
    #[expect(
        clippy::too_many_arguments,
        reason = "A2 grouped convolution kernel requiring many shape/stride/group parameters for efficient neural network inference"
    )]
    pub fn new_grouped(
        raw_weights: &[f32],
        raw_bias: &[f32],
        do_bias: bool,
        dilation: usize,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        groups: usize,
    ) -> Result<Self, NamErrorCode> {
        Ok(Self::Grouped(A2GroupedConv1d::new(
            raw_weights,
            raw_bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
        )?))
    }

    /// Returns the number of groups (1 for `Standard`, >1 for `Grouped`).
    #[inline(always)]
    pub fn groups(&self) -> usize {
        match self {
            Self::Standard(_) => 1,
            Self::Grouped(g) => g.groups,
        }
    }

    /// Returns true if this is a depthwise convolution (groups == in_ch == out_ch).
    #[inline(always)]
    pub fn is_depthwise(&self) -> bool {
        match self {
            Self::Standard(_) => false,
            Self::Grouped(g) => g.groups == g.in_ch && g.groups == g.out_ch,
        }
    }

    /// Kernel size of this convolution.
    #[inline(always)]
    pub fn kernel_size(&self) -> usize {
        match self {
            Self::Standard(c) => c.kernel,
            Self::Grouped(g) => g.kernel,
        }
    }

    /// Dilation factor.
    #[inline(always)]
    pub fn dilation(&self) -> usize {
        match self {
            Self::Standard(c) => c.dilation,
            Self::Grouped(g) => g.dilation,
        }
    }

    /// Number of input channels.
    #[inline(always)]
    pub fn in_ch(&self) -> usize {
        match self {
            Self::Standard(c) => c.in_ch,
            Self::Grouped(g) => g.in_ch,
        }
    }

    /// Number of output channels.
    #[inline(always)]
    pub fn out_ch(&self) -> usize {
        match self {
            Self::Standard(c) => c.out_ch,
            Self::Grouped(g) => g.out_ch,
        }
    }
}

#[path = "conv1d_dispatch.rs"]
mod dispatch;

#[cfg(test)]
#[path = "conv1d_test.rs"]
mod tests;
