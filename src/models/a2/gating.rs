// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Gating and blending module for the NAM A2 architecture.
//!
//! Implements the real gating and blending operations with parity to
//! `NAM/gating_activations.h`. Buffers are pre-allocated at initialization
//! (zero-alloc on the hot-path, RT-Safe).

use super::activations::{ActivationFn, ActivationType};
use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;
use crate::math::common::SimdMath;

/// Gating modes for WaveNet layers.
///
/// Determines how the layer processes duplicated bottleneck channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GatingMode {
    /// No gating or blending — standard activation (fast-path A2).
    #[default]
    None,
    /// Traditional gating: element-wise multiply of activated input × activated gate.
    Gated,
    /// Blending: weighted average between activated and pre-activated input, with
    /// alpha computed from gate channels.
    Blended,
}

/// Configuration for Gating-type activation.
///
/// Splits a 2×channels buffer into input (first half) and gating (second half).
/// Applies `input_activation` to the input half and `gating_activation` to the
/// gating half, then multiplies them element-wise.
///
/// Gating is fully in-place — no scratch buffer needed. Reads and writes
/// operate on disjoint halves of the buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct GatingActivationConfig {
    /// Activation function for input channels.
    pub input_activation: ActivationType,
    /// Activation function for gating channels.
    pub gating_activation: ActivationType,
}

impl GatingActivationConfig {
    /// Creates a new gating config.
    pub fn new(input_activation: ActivationType, gating_activation: ActivationType) -> Self {
        Self {
            input_activation,
            gating_activation,
        }
    }

    /// Applies gating in-place on a buffer of `2*ch` elements.
    ///
    /// - `buf[0..ch]` — input channels: activated, then multiplied by gate.
    /// - `buf[ch..2*ch]` — gating channels: activated, then used as multiplier.
    ///
    /// After this call, `buf[0..ch]` contains the gated output (the gate half
    /// is consumed). Matches `GatingActivation::process()` in the C++ reference.
    #[inline]
    pub fn apply_gating(&self, buf: &mut [f32]) {
        let ch = buf.len() / 2;
        debug_assert!(
            ch * 2 == buf.len(),
            "gating: buf length must be even (2×ch)"
        );

        self.input_activation.apply(&mut buf[..ch]);
        self.gating_activation.apply(&mut buf[ch..]);

        for i in 0..ch {
            buf[i] *= buf[ch + i];
        }
    }

    /// Applies gating in-place using the monomorphized SIMD backend `M`.
    ///
    /// Same semantics as [`apply_gating`](Self::apply_gating) but calls
    /// `apply_simd::<M>` on both activations, eliminating runtime redispatch.
    ///
    /// # Safety
    /// `buf` must be valid and `M` must match the CPU ISA capabilities.
    #[inline(always)]
    pub unsafe fn apply_gating_simd<M: SimdMath>(&self, buf: &mut [f32]) {
        let ch = buf.len() / 2;
        debug_assert!(
            ch * 2 == buf.len(),
            "gating: buf length must be even (2×ch)"
        );

        unsafe {
            self.input_activation.apply_simd::<M>(&mut buf[..ch]);
            self.gating_activation.apply_simd::<M>(&mut buf[ch..]);
        }

        for i in 0..ch {
            buf[i] *= buf[ch + i];
        }
    }
}

/// Configuration for Blending-type activation.
///
/// Splits a 2×channels buffer into input (first half) and blending (second half).
/// Applies `input_activation` to the input half and `blending_activation` to
/// produce alpha ∈ \[0,1\] from the second half. Output is the weighted average:
///
/// ```text
/// output[i] = activated_input[i] * alpha[i] + original_input[i] * (1 - alpha[i])
/// ```
///
/// A scratch buffer is pre-allocated at construction to save original input
/// values, ensuring zero allocations on the hot-path (RT-Safe).
#[derive(Debug, Clone)]
pub struct BlendingActivationConfig {
    /// Activation function for input channels.
    pub input_activation: ActivationType,
    /// Activation function for blending channels (determines alpha ∈ \[0,1\]).
    pub blending_activation: ActivationType,
    /// Scratch buffer for original input values during blending.
    scratch: AlignedVec<f32>,
}

impl PartialEq for BlendingActivationConfig {
    fn eq(&self, other: &Self) -> bool {
        self.input_activation == other.input_activation
            && self.blending_activation == other.blending_activation
            && self.scratch.len() == other.scratch.len()
    }
}

impl BlendingActivationConfig {
    /// Creates a new blending config with a pre-allocated scratch buffer.
    ///
    /// `ch` is the number of channels per half (total buffer = 2×ch).
    pub fn new(
        input_activation: ActivationType,
        blending_activation: ActivationType,
        ch: usize,
    ) -> Result<Self, NamErrorCode> {
        Ok(Self {
            input_activation,
            blending_activation,
            scratch: AlignedVec::new(ch, 0.0f32)?,
        })
    }

    /// Applies blending in-place on a buffer of `2*ch` elements.
    ///
    /// - `buf[0..ch]` — input channels: activated, then blended with original.
    /// - `buf[ch..2*ch]` — blending channels: activated to produce alpha.
    ///
    /// After this call, `buf[0..ch]` contains the blended output.
    /// Matches `BlendingActivation::process()` in the C++ reference.
    ///
    /// # Panics
    ///
    /// Panics in debug if `buf.len()` exceeds the pre-allocated scratch capacity.
    #[inline]
    pub fn apply_blending(&mut self, buf: &mut [f32]) {
        let ch = buf.len() / 2;
        debug_assert!(
            ch * 2 == buf.len(),
            "blending: buf length must be even (2×ch)"
        );
        debug_assert!(
            self.scratch.len() >= ch,
            "blending: scratch too small ({}) for ch={}",
            self.scratch.len(),
            ch
        );

        self.scratch[..ch].copy_from_slice(&buf[..ch]);

        self.input_activation.apply(&mut buf[..ch]);
        self.blending_activation.apply(&mut buf[ch..]);

        for i in 0..ch {
            let alpha = buf[ch + i];
            buf[i] = (buf[i] - self.scratch[i]).mul_add(alpha, self.scratch[i]);
        }
    }

    /// Returns the pre-allocated channel count.
    #[inline]
    pub fn channels(&self) -> usize {
        self.scratch.len()
    }

    /// Applies blending in-place using the monomorphized SIMD backend `M`.
    ///
    /// Same semantics as [`apply_blending`](Self::apply_blending) but calls
    /// `apply_simd::<M>` on both activations, eliminating runtime redispatch.
    ///
    /// # Safety
    /// `buf` must be valid and `M` must match the CPU ISA capabilities.
    /// Panics in debug if `buf.len()` exceeds scratch capacity.
    #[inline(always)]
    pub unsafe fn apply_blending_simd<M: SimdMath>(&mut self, buf: &mut [f32]) {
        let ch = buf.len() / 2;
        debug_assert!(
            ch * 2 == buf.len(),
            "blending: buf length must be even (2×ch)"
        );
        debug_assert!(
            self.scratch.len() >= ch,
            "blending: scratch too small ({}) for ch={}",
            self.scratch.len(),
            ch
        );

        self.scratch[..ch].copy_from_slice(&buf[..ch]);

        unsafe {
            self.input_activation.apply_simd::<M>(&mut buf[..ch]);
            self.blending_activation.apply_simd::<M>(&mut buf[ch..]);
        }

        for i in 0..ch {
            let alpha = buf[ch + i];
            buf[i] = (buf[i] - self.scratch[i]).mul_add(alpha, self.scratch[i]);
        }
    }
}

#[cfg(test)]
#[path = "gating_test.rs"]
mod tests;
