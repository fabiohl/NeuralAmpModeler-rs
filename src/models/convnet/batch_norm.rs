// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 1-Dimensional Batch Normalization layer for Inference-Only DSP.
//!
//! During inference, batch normalization is an affine transformation
//! `y = x * scale + offset` where the fused parameters are:
//!
//! ```text
//! scale  = gamma / sqrt(running_var + eps)
//! offset = beta - gamma * running_mean / sqrt(running_var + eps)
//! ```
//!
//! These are precomputed at construction time so the hot-path is a single
//! FMA per element (lowered to `vfmadd231ps` on x86-64-v3).

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::{AlignedVec, SimdMath};

/// 1-Dimensional Batch Normalization layer (inference-only).
///
/// Stores per-channel pre-fused affine parameters for the hot-path:
/// `output[ch + f * n_ch] = input[ch + f * n_ch] * scale[ch] + offset[ch]`.
///
/// Input/output layout is frame-interleaved:
/// `[f0_c0, f0_c1, ..., f0_c{n-1}, f1_c0, ...]`.
#[derive(Clone)]
#[repr(align(64))]
pub struct BatchNorm1D {
    /// Number of channels.
    pub(crate) num_channels: usize,
    /// Fused multiplicative factor per channel.
    pub(crate) scale: AlignedVec<f32>,
    /// Fused additive factor per channel.
    pub(crate) offset: AlignedVec<f32>,
}

/// Validates raw training parameters for `BatchNorm1D` before affine fusion.
///
/// Rejects slice length mismatches (`WeightCountMismatch`), non-finite inputs
/// (`NamJsonWeightNotFinite`), and negative `eps`, negative `running_var`, or
/// non-positive `running_var + eps` (`ModelBuildFailed`).
pub(crate) fn validate_raw_batchnorm_params(
    num_channels: usize,
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    eps: f32,
) -> Result<(), NamErrorCode> {
    if gamma.len() != num_channels
        || beta.len() != num_channels
        || running_mean.len() != num_channels
        || running_var.len() != num_channels
    {
        return Err(NamErrorCode::WeightCountMismatch);
    }
    if !eps.is_finite() {
        return Err(NamErrorCode::NamJsonWeightNotFinite);
    }
    if eps < 0.0 {
        return Err(NamErrorCode::ModelBuildFailed);
    }
    for c in 0..num_channels {
        let g = gamma[c];
        let b = beta[c];
        let mean = running_mean[c];
        let var = running_var[c];
        if !g.is_finite() || !b.is_finite() || !mean.is_finite() || !var.is_finite() {
            return Err(NamErrorCode::NamJsonWeightNotFinite);
        }
        let denom = var + eps;
        if var < 0.0 || !denom.is_finite() || denom <= 0.0 {
            return Err(NamErrorCode::ModelBuildFailed);
        }
    }
    Ok(())
}

impl BatchNorm1D {
    /// Constructs a `BatchNorm1D` from the raw training parameters,
    /// fusing them into a single affine transform for inference.
    ///
    /// # Parameters
    /// - `gamma`: learnable scale (length `num_channels`).
    /// - `beta`: learnable bias (length `num_channels`).
    /// - `running_mean`: EMA of activation means (length `num_channels`).
    /// - `running_var`: EMA of activation variances (length `num_channels`).
    /// - `eps`: small constant for numerical stability (e.g. 1e-5).
    ///
    /// # Errors
    /// Returns [`NamErrorCode::WeightCountMismatch`] on slice length mismatch,
    /// [`NamErrorCode::NamJsonWeightNotFinite`] on non-finite parameters,
    /// [`NamErrorCode::ModelBuildFailed`] if `eps < 0.0`, `running_var[c] < 0.0`,
    /// `(running_var[c] + eps) <= 0.0`, or any fused coefficient is non-finite,
    /// and [`NamErrorCode::OutOfMemory`] if allocation fails.
    pub fn from_params(
        num_channels: usize,
        gamma: &[f32],
        beta: &[f32],
        running_mean: &[f32],
        running_var: &[f32],
        eps: f32,
    ) -> Result<Self, NamErrorCode> {
        validate_raw_batchnorm_params(num_channels, gamma, beta, running_mean, running_var, eps)?;

        let mut scale = AlignedVec::new(num_channels, 0.0f32)?;
        let mut offset = AlignedVec::new(num_channels, 0.0f32)?;

        for c in 0..num_channels {
            let inv_std = 1.0 / (running_var[c] + eps).sqrt();
            scale[c] = gamma[c] * inv_std;
            offset[c] = beta[c] - running_mean[c] * scale[c];
            if !scale[c].is_finite() || !offset[c].is_finite() {
                return Err(NamErrorCode::ModelBuildFailed);
            }
        }

        Ok(Self {
            num_channels,
            scale,
            offset,
        })
    }

    /// Constructs a `BatchNorm1D` from already-fused parameters.
    ///
    /// Useful when the fused scale/offset are provided directly by the
    /// model loader (e.g. from a `.namb` blob prepared by the training script).
    ///
    /// # Errors
    /// Returns [`NamErrorCode::WeightCountMismatch`] on slice length mismatch,
    /// [`NamErrorCode::NamJsonWeightNotFinite`] on non-finite values, and
    /// [`NamErrorCode::OutOfMemory`] if allocation fails.
    pub fn from_fused(
        num_channels: usize,
        scale: &[f32],
        offset: &[f32],
    ) -> Result<Self, NamErrorCode> {
        if scale.len() != num_channels || offset.len() != num_channels {
            return Err(NamErrorCode::WeightCountMismatch);
        }
        if scale.iter().chain(offset.iter()).any(|v| !v.is_finite()) {
            return Err(NamErrorCode::NamJsonWeightNotFinite);
        }

        let mut s = AlignedVec::new(num_channels, 0.0f32)?;
        let mut o = AlignedVec::new(num_channels, 0.0f32)?;
        s.copy_from_slice(scale);
        o.copy_from_slice(offset);

        Ok(Self {
            num_channels,
            scale: s,
            offset: o,
        })
    }

    /// Returns the number of channels.
    #[inline(always)]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Returns the per-channel fused multiplicative scale vector.
    #[inline(always)]
    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// Returns the per-channel fused additive offset vector.
    #[inline(always)]
    pub fn offset(&self) -> &[f32] {
        &self.offset
    }

    /// Applies the batch normalization affine transform in-place, via SIMD dispatch.
    ///
    /// `data` layout: `[f0_c0, f0_c1, ..., f0_c{n_ch-1}, f1_c0, ...]`.
    /// `data.len()` must be at least `num_frames * num_channels`.
    ///
    /// # Safety
    /// The caller must ensure `data.len() >= num_frames * num_channels` and that
    /// the slice holds valid initialized frame data.
    #[inline(always)]
    pub unsafe fn process(&self, data: &mut [f32], num_frames: usize) {
        let needed = num_frames
            .checked_mul(self.num_channels)
            .expect("BatchNorm1D::process frame*channel count overflow");
        assert!(
            data.len() >= needed,
            "BatchNorm1D::process: data slice too short ({} < {})",
            data.len(),
            needed
        );
        // SAFETY: caller guarantees `data.len()` matches the frame×channel
        // layout. The dispatch macro selects an ISA-specific `process_simd`
        // implementation whose safety contract is the same.
        unsafe { crate::math::common::dispatch_simd!(self, process_simd, data, num_frames) };
    }

    /// Monomorphized batch normalization affine transform.
    ///
    /// Delegates directly to `M::batch_norm_process` for compile-time
    /// dispatch with no dynamic ISA lookup.
    ///
    /// # Safety
    /// `data.len() >= num_frames * self.num_channels`.
    #[inline(always)]
    pub unsafe fn process_simd<M: SimdMath>(&self, data: &mut [f32], num_frames: usize) {
        let needed = num_frames
            .checked_mul(self.num_channels)
            .expect("BatchNorm1D::process_simd frame*channel count overflow");
        assert!(
            data.len() >= needed,
            "BatchNorm1D::process_simd: data slice too short ({} < {})",
            data.len(),
            needed
        );
        // SAFETY: caller guarantees the frame×channel layout invariant
        // (data.len >= num_frames * num_channels). scale and offset each
        // have `num_channels` elements, validated at construction time.
        // `M::batch_norm_process` uses these pre-validated buffers without
        // additional bounds checks.
        unsafe {
            M::batch_norm_process(
                data,
                &self.scale,
                &self.offset,
                self.num_channels,
                num_frames,
            );
        }
    }

    /// Scalar reference implementation — always available, used for testing parity.
    ///
    /// # Safety
    /// `data.len()` must be at least `num_frames * self.num_channels`.
    #[inline(always)]
    pub unsafe fn process_scalar(&self, data: &mut [f32], num_frames: usize) {
        let needed = num_frames
            .checked_mul(self.num_channels)
            .expect("BatchNorm1D::process_scalar frame*channel count overflow");
        assert!(
            data.len() >= needed,
            "BatchNorm1D::process_scalar: data slice too short ({} < {})",
            data.len(),
            needed
        );
        // SAFETY: caller guarantees the frame×channel layout invariant.
        // scale and offset each have `num_channels` elements (validated at
        // construction). `process_scalar_ref` iterates within these bounds.
        unsafe {
            process_scalar_ref(
                data,
                &self.scale,
                &self.offset,
                self.num_channels,
                num_frames,
            );
        }
    }
}

/// Scalar reference: plain `mul_add` loop, channel-major across frames.
///
/// # Safety
/// `data.len()` must equal `num_frames * n_ch`.
#[inline(always)]
unsafe fn process_scalar_ref(
    data: &mut [f32],
    scale: &[f32],
    offset: &[f32],
    n_ch: usize,
    num_frames: usize,
) {
    for ch in 0..n_ch {
        // SAFETY: ch ∈ [0, n_ch) by the loop bound. scale and offset
        // each have `n_ch` elements (guaranteed at construction time
        // by from_params/from_fused).
        let s = unsafe { *scale.get_unchecked(ch) };
        // SAFETY: `ch < n_ch` by the loop bound and `offset` has `n_ch`
        // elements (validated at construction time by from_params/from_fused).
        let o = unsafe { *offset.get_unchecked(ch) };
        for f in 0..num_frames {
            let idx = ch + f * n_ch;
            // SAFETY: ch < n_ch and f < num_frames, so idx < n_ch * num_frames.
            // The caller guarantees data.len() == num_frames * n_ch.
            unsafe {
                *data.get_unchecked_mut(idx) = (*data.get_unchecked(idx)).mul_add(s, o);
            }
        }
    }
}

#[cfg(test)]
#[path = "batch_norm_test.rs"]
mod tests;
