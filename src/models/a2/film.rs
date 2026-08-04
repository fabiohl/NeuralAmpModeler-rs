// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! Feature-wise Linear Modulation (FiLM) module for the NAM A2 architecture.
//!
//! FiLM enables the model to adapt its behavior based on external
//! conditioning signals, applying per-channel scale and shift:
//!
//! `output[c] = input[c] * scale[c] + shift[c]`
//!
//! The `_cond_to_scale_shift` operation (C++ `NAM/film.h:28-85`) maps
//! the condition vector to per-channel scale+shift via a Conv1x1
//! implemented as a dense GEMV (matrix-vector multiply with AVX2 FMA).
//!
//! # RT-Safety
//! - The `scale_shift_buf` is pre-allocated in `load()` — zero allocation
//!   on the hot-path.
//! - The `apply_modulation` inner loop uses `chunks_exact_mut` with no
//!   internal branches.
//! - All SIMD operates on 64-byte aligned `AlignedVec` buffers.
//!
//! # Source of truth
//! - `NAM/film.h:28-85` (`_cond_to_scale_shift` via Conv1x1 / Dense)
//! - `NeuralAmpModelerCore/NAM/film.cpp` (per-channel modulation)

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;

use core::arch::x86_64::*;

/// Configuration for a FiLM layer or operation.
///
/// Corresponds to the `_FiLMParams` struct in C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiLMConfig {
    /// Whether FiLM is active at this location.
    pub active: bool,
    /// Whether to apply both scale and shift (true) or only scale (false).
    pub shift: bool,
    /// Number of groups for grouped convolution in the conditioning submodule.
    pub groups: u32,
}

impl Default for FiLMConfig {
    fn default() -> Self {
        Self {
            active: false,
            shift: true,
            groups: 1,
        }
    }
}

/// Concrete FiLM layer — `_cond_to_scale_shift` + per-channel modulation.
///
/// # Weight layout
/// Row-major `[out_rows][cond_size]` where:
/// - When `groups == 1`: `out_rows = channels * (shift ? 2 : 1)`.
///   First `channels` rows produce `scale`; next `channels` rows produce `shift`.
/// - When `groups > 1`: weights are partitioned into `groups` independent blocks,
///   each of size `out_per_group * (cond_size / groups)`.
///
/// The `scale_shift_buf` always has `channels * 2` elements — when `shift == false`
/// the second half remains zero so the modulation step stays branch-free.
#[derive(Clone)]
#[repr(align(64))]
pub struct FiLMLayer {
    /// FiLM configuration (active, shift, groups).
    pub config: FiLMConfig,
    /// Dimension of the conditioning vector.
    pub cond_size: usize,
    /// Number of feature channels to modulate.
    pub channels: usize,
    /// Row-major weight matrix for `_cond_to_scale_shift`.
    pub weights: AlignedVec<f32>,
    /// Bias vector, indexed by global output channel.
    pub bias: AlignedVec<f32>,
    /// Pre-allocated scratch buffer: `[scale[..channels], shift[..channels]]`.
    scale_shift_buf: AlignedVec<f32>,
}

impl FiLMLayer {
    /// Allocates a FiLM layer with the given topology and copies in the weights.
    ///
    /// # Buffers
    /// `scale_shift_buf` is pre-allocated here — no heap traffic on the DSP
    /// hot-path.
    pub fn load(
        config: FiLMConfig,
        cond_size: usize,
        channels: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, NamErrorCode> {
        let expected_bias = if config.shift { channels * 2 } else { channels };
        let mut bias_padded = bias;
        if bias_padded.len() < expected_bias {
            bias_padded.resize(expected_bias, 0.0);
        }
        Ok(Self {
            config,
            cond_size,
            channels,
            weights: AlignedVec::from_vec(weights)?,
            bias: AlignedVec::from_vec(bias_padded)?,
            scale_shift_buf: AlignedVec::new(channels * 2, 0.0f32)?,
        })
    }

    /// Processes FiLM modulation over the input buffer.
    ///
    /// 1. `_cond_to_scale_shift`: maps `condition` → `scale[..channels]` +
    ///    `shift[..channels]` via a dense GEMV with AVX2 FMA.
    /// 2. Per-channel modulation: `input[c] = input[c] * scale[c] + shift[c]`.
    ///
    /// # Safety
    /// `condition.len()` must equal `self.cond_size`.
    /// `input.len()` may be ≤ `self.channels` (e.g. activation_post_film on
    /// bottleneck-sized slice when bottleneck < channels).
    #[inline(always)]
    pub unsafe fn process(&mut self, input: &mut [f32], condition: &[f32]) {
        debug_assert!(
            condition.len() >= self.cond_size,
            "FiLM process: condition slice length ({}) must be >= cond_size ({})",
            condition.len(),
            self.cond_size
        );
        self.cond_to_scale_shift(condition);
        self.apply_modulation(input);
    }

    // ── _cond_to_scale_shift ──────────────────────────────────────────

    #[inline(always)]
    unsafe fn cond_to_scale_shift(&mut self, condition: &[f32]) {
        debug_assert!(
            condition.len() >= self.cond_size,
            "FiLM cond_to_scale_shift: condition slice length ({}) must be >= cond_size ({})",
            condition.len(),
            self.cond_size
        );
        let g = self.config.groups as usize;
        let ch_per_group = self.channels / g;
        let cond_per_group = self.cond_size / g;
        let out_per_group = if self.config.shift {
            ch_per_group * 2
        } else {
            ch_per_group
        };

        for grp in 0..g {
            let cond_slice =
                condition.get_unchecked(grp * cond_per_group..(grp + 1) * cond_per_group);
            let w_offset = grp * out_per_group * cond_per_group;

            for row in 0..out_per_group {
                let global_out = if row < ch_per_group {
                    grp * ch_per_group + row
                } else {
                    self.channels + grp * ch_per_group + (row - ch_per_group)
                };

                let mut sum = *self.bias.get_unchecked(global_out);
                let w_start = w_offset + row * cond_per_group;
                let w_row = self
                    .weights
                    .get_unchecked(w_start..w_start + cond_per_group);
                sum += dot_product_avx2(w_row, cond_slice);
                *self.scale_shift_buf.get_unchecked_mut(global_out) = sum;
            }
        }

        if !self.config.shift {
            for c in self.channels..self.channels * 2 {
                *self.scale_shift_buf.get_unchecked_mut(c) = 0.0;
            }
        }
    }

    // ── Per-channel modulation ────────────────────────────────────────

    /// `input[c] *= scale[c]; input[c] += shift[c]` — AVX2, 8-wide.
    ///
    /// Uses [`as_chunks_mut`](core::slice::slice::as_chunks_mut) for the SIMD
    /// block, yielding `&mut [[f32; 8]]` with compile-time guaranteed width.
    /// The remainder tail uses safe indexing — bounds are guaranteed by the
    /// chunk decomposition.
    #[inline(always)]
    unsafe fn apply_modulation(&mut self, input: &mut [f32]) {
        let scale = &self.scale_shift_buf[..self.channels];
        let shift = &self.scale_shift_buf[self.channels..self.channels * 2];
        let limit = input.len().min(self.channels);

        let (in_chunks, in_rem) = input[..limit].as_chunks_mut::<8>();
        let (scale_chunks, _) = scale[..limit].as_chunks::<8>();
        let (shift_chunks, _) = shift[..limit].as_chunks::<8>();

        for i in 0..in_chunks.len() {
            let v_in = _mm256_loadu_ps(in_chunks[i].as_ptr());
            let v_scale = _mm256_loadu_ps(scale_chunks[i].as_ptr());
            let v_shift = _mm256_loadu_ps(shift_chunks[i].as_ptr());
            _mm256_storeu_ps(
                in_chunks[i].as_mut_ptr(),
                _mm256_fmadd_ps(v_in, v_scale, v_shift),
            );
        }

        let off = in_chunks.len() * 8;
        for c in 0..in_rem.len() {
            input[off + c] = input[off + c] * scale[off + c] + shift[off + c];
        }
    }
}

// ── FilmBlock: mutable FiLM references for block-level dispatch ──────────

/// Bundle of mutable references to the 8 FiLM insertion points in an A2 layer.
///
/// Used by `layer_forward_ch{3,8}_block` to conditionally apply FiLM
/// at the correct positions in the signal chain without per-point parameter
/// explosion.
pub struct FilmBlock<'a> {
    /// FiLM before dilated convolution.
    pub conv_pre_film: Option<&'a mut FiLMLayer>,
    /// FiLM after dilated convolution.
    pub conv_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM before input mixin (same insertion point as conv_post_film).
    pub input_mixin_pre_film: Option<&'a mut FiLMLayer>,
    /// FiLM after input mixin, before activation.
    pub input_mixin_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM before activation (same insertion point as input_mixin_post_film).
    pub activation_pre_film: Option<&'a mut FiLMLayer>,
    /// FiLM after activation.
    pub activation_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM after layer 1x1 residual.
    pub layer1x1_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM after head 1x1 (reserved for future general A2 engine).
    pub head1x1_post_film: Option<&'a mut FiLMLayer>,
}

impl<'a> FilmBlock<'a> {
    /// Creates an empty `FilmBlock` with all fields set to `None`.
    /// Useful for tests and for the fast-path when no FiLM is active.
    pub fn empty() -> Self {
        Self {
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
}

impl super::layer::A2Layer {
    /// Returns a [`FilmBlock`] with mutable references to all 8 FiLM insertion points.
    #[inline]
    pub fn film_block(&mut self) -> FilmBlock<'_> {
        FilmBlock {
            conv_pre_film: self.conv_pre_film.as_mut(),
            conv_post_film: self.conv_post_film.as_mut(),
            input_mixin_pre_film: self.input_mixin_pre_film.as_mut(),
            input_mixin_post_film: self.input_mixin_post_film.as_mut(),
            activation_pre_film: self.activation_pre_film.as_mut(),
            activation_post_film: self.activation_post_film.as_mut(),
            layer1x1_post_film: self.layer1x1_post_film.as_mut(),
            head1x1_post_film: self.head1x1_post_film.as_mut(),
        }
    }
}

// ── AVX2 dot product ───────────────────────────────────────────────────

/// Dot product `sum(a[i] * b[i])` — AVX2+FMA, 8-wide.
///
/// Uses dual accumulator and horizontal reduction matching the
/// `convolve_mono_avx2` pattern.
#[inline(always)]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= len {
        let ha0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let hb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(ha0, hb0, sum0);

        let ha1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let hb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        sum1 = _mm256_fmadd_ps(ha1, hb1, sum1);

        i += 16;
    }

    while i + 8 <= len {
        let ha = _mm256_loadu_ps(a.as_ptr().add(i));
        let hb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(ha, hb, sum0);
        i += 8;
    }

    let sum = _mm256_add_ps(sum0, sum1);
    let hi128 = _mm256_extractf128_ps(sum, 1);
    let lo128 = _mm256_castps256_ps128(sum);
    let s128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let r = _mm_add_ss(sums, shuf2);
    let mut out = _mm_cvtss_f32(r);

    while i < len {
        out += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }

    out
}

#[cfg(test)]
#[path = "film_test.rs"]
mod tests;
