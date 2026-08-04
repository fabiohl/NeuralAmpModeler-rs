// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Layer — per-layer forward pass for the A2 WaveNet fast-path.
//!
//! Implements the sequence from `a2_fast.cpp:514`:
//! 1. Dilated causal Conv1D → 2. Input mixin (condition × mixin_w, no bias) →
//! 3. LeakyReLU(0.01) → 4. Head accumulator (assign layer 0, accumulate layers 1-22) →
//! 5. Layer1x1 residual `layer_in += l1x1_b + l1x1_w * z` (skipped on last layer).
//!
//! ## Source of truth
//! - `a2_fast.cpp:417-690` (`_layer_forward_k`)
//! - `NAM/wavenet/detail.h` (`Layer`)

use super::conv1d::A2Conv1d;
use super::conv1d_ch::A2Conv1dCh;
use super::film::FiLMLayer;
use super::params::A2_LEAKY_SLOPE;
use crate::math::common::{AlignedVec, SimdMath};

/// CH-optimized conv wrapper (enum dispatch).
///
/// Both variants hold the same underlying `A2Conv1dCh<CH>` type; the enum
/// preserves the monomorphized type so the hot-path can call the correct
/// SIMD kernel without indirect dispatch.
pub enum A2ConvCh {
    /// CH=3 variant (SSE 128-bit, stride-16).
    Ch3(A2Conv1dCh<3>),
    /// CH=8 variant (AVX2 256-bit, stride-64).
    Ch8(A2Conv1dCh<8>),
}

/// Single A2 WaveNet layer.
///
/// Holds the weights for: dilated conv (via `A2Conv1d`), input mixin (`Conv1x1 condition→CH`, no bias),
/// and layer1x1 (`Conv1x1 CH→CH`, with bias, col-major).
///
/// When `conv_ch` is `Some`, it holds f32 col-major-per-tap weights for the CH=3 or CH=8
/// optimized path. When `None`, the standard `A2Conv1d` (u16 interleaved) is used (fallback).
///
/// FiLM layers (8 insertion points) are `Some` only when the JSON config marks them `active: true`.
pub struct A2Layer {
    /// Dilated causal Conv1D (kernel ∈ {6, 15}). Standard (groups=1) or Grouped (groups>1).
    pub conv: A2Conv1d,
    /// CH-optimized conv (col-major-per-tap f32). `Some` when CH ∈ {3, 8}.
    pub conv_ch: Option<A2ConvCh>,
    /// Input mixin weights (`CH` elements, f32).
    pub mixin_w: AlignedVec<f32>,
    /// Layer1x1 weights (`CH × CH`, col-major: `[bottleneck][out]`).
    pub l1x1_w: AlignedVec<f32>,
    /// Layer1x1 bias (`CH` elements, f32).
    pub l1x1_b: AlignedVec<f32>,
    /// FiLM before dilated convolution — modulates `layer_in` new frames in history buffer.
    pub conv_pre_film: Option<FiLMLayer>,
    /// FiLM after dilated convolution — modulates `z_buf` before mixin.
    pub conv_post_film: Option<FiLMLayer>,
    /// FiLM before input mixin (same insertion point as `conv_post_film`).
    pub input_mixin_pre_film: Option<FiLMLayer>,
    /// FiLM after input mixin — modulates `z_buf` after mixin, before activation.
    pub input_mixin_post_film: Option<FiLMLayer>,
    /// FiLM before activation (same insertion point as `input_mixin_post_film`).
    pub activation_pre_film: Option<FiLMLayer>,
    /// FiLM after activation — modulates `z_buf` after LeakyReLU.
    pub activation_post_film: Option<FiLMLayer>,
    /// FiLM after layer 1x1 residual — modulates `layer_in` after l1x1 accumulation.
    pub layer1x1_post_film: Option<FiLMLayer>,
    /// FiLM after head 1x1 (reserved for future general A2 engine).
    pub head1x1_post_film: Option<FiLMLayer>,
}

impl A2Layer {
    /// The mixin has no bias in the A2 fast-path (cond_size=1, `Conv1x1 condition→CH`).
    pub fn new(
        conv: A2Conv1d,
        mixin_w: AlignedVec<f32>,
        l1x1_w: AlignedVec<f32>,
        l1x1_b: AlignedVec<f32>,
    ) -> Self {
        let ch = conv.out_ch();
        debug_assert_eq!(mixin_w.len(), ch);
        debug_assert_eq!(l1x1_w.len(), ch * ch);
        debug_assert_eq!(l1x1_b.len(), ch);
        Self {
            conv,
            conv_ch: None,
            mixin_w,
            l1x1_w,
            l1x1_b,
            conv_pre_film: None,
            conv_post_film: None,
            input_mixin_pre_film: None,
            input_mixin_post_film: None,
            activation_pre_film: None,
            activation_post_film: None,
            layer1x1_post_film: None,
            head1x1_post_film: None,
        }
    }

    /// Creates a CH=3 layer with f32-native col-major-per-tap weights.
    pub fn new_with_ch3(
        conv: A2Conv1d,
        ch3_conv: A2Conv1dCh<3>,
        mixin_w: AlignedVec<f32>,
        l1x1_w: AlignedVec<f32>,
        l1x1_b: AlignedVec<f32>,
    ) -> Self {
        debug_assert_eq!(conv.out_ch(), 3);
        debug_assert_eq!(mixin_w.len(), 3);
        debug_assert_eq!(l1x1_w.len(), 9);
        debug_assert_eq!(l1x1_b.len(), 3);
        Self {
            conv,
            conv_ch: Some(A2ConvCh::Ch3(ch3_conv)),
            mixin_w,
            l1x1_w,
            l1x1_b,
            conv_pre_film: None,
            conv_post_film: None,
            input_mixin_pre_film: None,
            input_mixin_post_film: None,
            activation_pre_film: None,
            activation_post_film: None,
            layer1x1_post_film: None,
            head1x1_post_film: None,
        }
    }

    /// Creates a layer with CH=8 optimized weights.
    pub fn new_with_ch8(
        conv: A2Conv1d,
        ch8_conv: A2Conv1dCh<8>,
        mixin_w: AlignedVec<f32>,
        l1x1_w: AlignedVec<f32>,
        l1x1_b: AlignedVec<f32>,
    ) -> Self {
        debug_assert_eq!(conv.out_ch(), 8);
        debug_assert_eq!(mixin_w.len(), 8);
        debug_assert_eq!(l1x1_w.len(), 64);
        debug_assert_eq!(l1x1_b.len(), 8);
        Self {
            conv,
            conv_ch: Some(A2ConvCh::Ch8(ch8_conv)),
            mixin_w,
            l1x1_w,
            l1x1_b,
            conv_pre_film: None,
            conv_post_film: None,
            input_mixin_pre_film: None,
            input_mixin_post_film: None,
            activation_pre_film: None,
            activation_post_film: None,
            layer1x1_post_film: None,
            head1x1_post_film: None,
        }
    }

    /// Creates a layer with arbitrary l1x1 dimensions (bottleneck → channels).
    ///
    /// Used by the dynamic A2 engine where `l1x1_out_ch` (channels) may differ
    /// from `conv.out_ch()` (bottleneck). When gating/blending is active,
    /// `conv.out_ch()` is `2*bottleneck` but l1x1 operates on the post-gating
    /// bottleneck-wide output.
    ///
    /// `condition_size` controls the mixin weight layout:
    /// `mixin_w` has `conv.out_ch() * condition_size` elements, laid out as
    /// row-major `[output_channel][condition_index]`.
    pub fn new_dyn(
        conv: A2Conv1d,
        mixin_w: AlignedVec<f32>,
        l1x1_w: AlignedVec<f32>,
        l1x1_b: AlignedVec<f32>,
        l1x1_out_ch: usize,
        bottleneck: usize,
        condition_size: usize,
    ) -> Self {
        let ch = conv.out_ch();
        debug_assert_eq!(mixin_w.len(), ch * condition_size);
        debug_assert_eq!(l1x1_w.len(), bottleneck * l1x1_out_ch);
        debug_assert_eq!(l1x1_b.len(), l1x1_out_ch);
        Self {
            conv,
            conv_ch: None,
            mixin_w,
            l1x1_w,
            l1x1_b,
            conv_pre_film: None,
            conv_post_film: None,
            input_mixin_pre_film: None,
            input_mixin_post_film: None,
            activation_pre_film: None,
            activation_post_film: None,
            layer1x1_post_film: None,
            head1x1_post_film: None,
        }
    }

    /// Channel count (bottleneck == channels in A2 fast-path).
    #[inline(always)]
    pub fn channels(&self) -> usize {
        self.conv.out_ch()
    }

    /// Number of groups for the dilated conv (1 = standard, >1 = grouped/depthwise).
    #[inline(always)]
    pub fn groups(&self) -> usize {
        self.conv.groups()
    }

    /// Returns true if the dilated conv is depthwise (groups == channels).
    #[inline(always)]
    pub fn is_depthwise(&self) -> bool {
        self.conv.is_depthwise()
    }

    /// Kernel size of this layer's dilated conv.
    #[inline(always)]
    pub fn kernel_size(&self) -> usize {
        self.conv.kernel_size()
    }

    /// Dilation factor of this layer.
    #[inline(always)]
    pub fn dilation(&self) -> usize {
        self.conv.dilation()
    }

    /// Processes a single frame through this layer.
    ///
    /// ## Data flow (matches `_layer_forward_k` in `a2_fast.cpp:514`)
    ///
    /// 1. Dilated conv over `layer_history` → `z_buf[..CH]`.
    /// 2. Mixin: `z_buf[c] += mixin_w[c] * input_cond`.
    /// 3. LeakyReLU(0.01) in-place.
    /// 4. Head accumulator: assign (layer 0) or add (layer > 0) into `head_accum`.
    /// 5. L1x1 residual: `layer_in[c] += l1x1_b[c] + sum_u(l1x1_w[u*CH+c] * z_buf[u])`.
    ///    Skipped on the last layer (output of last layer is dead — only head matters).
    ///
    /// # Parameters
    /// * `layer_history` — per-layer ring buffer (column-major: CH rows × N cols).
    /// * `frame_idx` — absolute column index in `layer_history` for the dilated conv (already ring-masked or linear).
    /// * `input_cond` — scalar input condition for the mixin (original input signal at this frame).
    /// * `head_accum` — head accumulator ring buffer (column-major).
    /// * `head_col` — column index in `head_accum` for this frame's output.
    /// * `z_buf` — scratch buffer for conv output (length ≥ `channels()`).
    /// * `layer_in_out` — mutable reference to this frame's layer_in (will be updated).
    /// * `is_first` — layer 0 writes to head, layers 1-22 accumulate.
    /// * `is_last` — layer 22 skips l1x1 residual.
    #[expect(
        clippy::too_many_arguments,
        reason = "A2 neural network layer requiring many shape/stride/buffer parameters for dynamic topology construction"
    )]
    #[inline(always)]
    pub fn process_single_frame<M: SimdMath>(
        &self,
        layer_history: &[f32],
        frame_idx: usize,
        input_cond: f32,
        head_accum: &mut [f32],
        head_col: usize,
        z_buf: &mut [f32],
        layer_in_out: &mut [f32],
        is_first: bool,
        is_last: bool,
    ) {
        let ch = self.channels();
        debug_assert!(z_buf.len() >= ch);
        debug_assert!(layer_in_out.len() >= ch);

        // 1. Dilated conv (no mixin — A2 adds mixin after conv).
        unsafe {
            self.conv
                .process_single_frame::<M>(layer_history, z_buf, frame_idx, None);
        }

        // 2. Input mixin: z_buf[c] += mixin_w[c] * input_cond.
        let mixin = &self.mixin_w;
        for c in 0..ch {
            z_buf[c] += mixin[c] * input_cond;
        }

        // 3. LeakyReLU(0.01) in-place.
        for z in z_buf.iter_mut().take(ch) {
            if *z < 0.0 {
                *z *= A2_LEAKY_SLOPE;
            }
        }

        // 4. Head accumulator.
        let head_off = head_col * ch;
        if is_first {
            head_accum[head_off..head_off + ch].copy_from_slice(&z_buf[..ch]);
        } else {
            for (c, z_val) in z_buf.iter().enumerate().take(ch) {
                head_accum[head_off + c] += *z_val;
            }
        }

        // 5. L1x1 residual (skipped on last layer).
        if !is_last {
            let l1x1 = &self.l1x1_w;
            let bias = &self.l1x1_b;
            for c in 0..ch {
                let mut sum = bias[c];
                for u in 0..ch {
                    // Col-major: l1x1_w[u * ch + c] = weight from bottleneck u to output c.
                    sum += l1x1[u * ch + c] * z_buf[u];
                }
                layer_in_out[c] += sum;
            }
        }
    }
}

// =============================================================================
// Scalar reference (oracle) for parity testing
// =============================================================================

/// Scalar reference for the A2Layer forward pass on a single frame.
///
/// Replicates the exact computation using the scalar conv fallback, then
/// applies mixin, LeakyReLU, head accumulation, and l1x1 residual.
/// Used as oracle in parity tests.
#[expect(
    clippy::too_many_arguments,
    reason = "A2 neural network layer requiring many shape/stride/buffer parameters for dynamic topology construction"
)]
pub fn a2_layer_single_frame_scalar_ref(
    conv_weights: &[f32],
    conv_bias: &[f32],
    conv_do_bias: bool,
    dilation: usize,
    kernel_size: usize,
    layer_history: &[f32],
    frame_idx: usize,
    mixin_w: &[f32],
    input_cond: f32,
    l1x1_w: &[f32],
    l1x1_b: &[f32],
    head_accum: &mut [f32],
    head_col: usize,
    layer_in_out: &mut [f32],
    is_first: bool,
    is_last: bool,
) {
    let ch = mixin_w.len();

    // 1. Dilated conv (scalar fallback).
    let mut z_buf = vec![0.0f32; ch];
    super::conv1d_fallback::a2_conv1d_single_frame_fallback(
        conv_weights,
        conv_bias,
        conv_do_bias,
        dilation,
        ch, // in_ch == out_ch for A2 fast-path
        ch,
        kernel_size,
        layer_history,
        frame_idx,
        None,
        &mut z_buf,
    );

    // 2. Mixin.
    for c in 0..ch {
        z_buf[c] += mixin_w[c] * input_cond;
    }

    // 3. LeakyReLU(0.01).
    for z in z_buf.iter_mut().take(ch) {
        if *z < 0.0 {
            *z *= A2_LEAKY_SLOPE;
        }
    }

    // 4. Head accumulator.
    let head_off = head_col * ch;
    if is_first {
        head_accum[head_off..head_off + ch].copy_from_slice(&z_buf[..ch]);
    } else {
        for c in 0..ch {
            head_accum[head_off + c] += z_buf[c];
        }
    }

    // 5. L1x1 residual.
    if !is_last {
        for c in 0..ch {
            let mut sum = l1x1_b[c];
            for u in 0..ch {
                sum += l1x1_w[u * ch + c] * z_buf[u];
            }
            layer_in_out[c] += sum;
        }
    }
}

#[cfg(test)]
#[path = "layer_test.rs"]
mod tests;
