// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::{AlignedVec, SimdMath};

/// 1x1 Dense Layer (The Channel Mixer) with runtime dimensions.
///
/// Think of this layer as a 'digital mixing console'. It blends the various
/// audio channels coming from the previous stage to create the final timbre combination.
#[derive(Clone)]
pub struct DenseLayerDyn {
    /// Number of input channels.
    pub in_ch: usize,
    /// Number of output channels.
    pub out_ch: usize,
    /// Weight matrix: Defines 'how much' of each channel goes into the mix.
    pub weights: AlignedVec<f32>,
    /// Bias: A basic 'volume' adjustment for each output channel.
    pub bias: AlignedVec<f32>,
    /// Flag indicating whether bias should be applied.
    pub do_bias: bool,
}

impl DenseLayerDyn {
    /// Validated constructor (release-stable, off-RT) for a dynamic `DenseLayerDyn`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - `in_ch == 0` or `out_ch == 0`;
    /// - the size product `in_ch * out_ch` overflows `usize` (checked,
    ///   F-RES2-04 — never wraps in release);
    /// - `weights.len() < in_ch * out_ch`;
    /// - `do_bias` is true and `bias.len() < out_ch`.
    #[inline]
    pub fn try_from_parts(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        in_ch: usize,
        out_ch: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(in_ch > 0, "DenseLayerDyn in_ch must be >= 1, got {in_ch}");
        anyhow::ensure!(
            out_ch > 0,
            "DenseLayerDyn out_ch must be >= 1, got {out_ch}"
        );
        // Defense-in-depth (F-RES2-04): checked size product so hostile
        // `in_ch`/`out_ch` can never wrap into a smaller contract in release
        // builds (overflow-checks off). Mirrors
        // `checked_arith::checked_dense_total` used by the loader.
        let min_weights = in_ch.checked_mul(out_ch).ok_or_else(|| {
            anyhow::anyhow!(
                "DenseLayerDyn weights size overflows usize: in_ch ({in_ch}) * out_ch ({out_ch}) — DoS protection (F-RES2-04)"
            )
        })?;
        anyhow::ensure!(
            weights.len() >= min_weights,
            "DenseLayerDyn weights buffer too small: expected >= {min_weights}, got {}",
            weights.len()
        );
        if do_bias {
            anyhow::ensure!(
                bias.len() >= out_ch,
                "DenseLayerDyn bias buffer too small: expected >= {out_ch}, got {}",
                bias.len()
            );
        }
        Ok(Self {
            in_ch,
            out_ch,
            weights,
            bias,
            do_bias,
        })
    }

    /// Residual Sum (The Final 'Shortcut'):
    /// This function mixes channels AND adds the original sound
    /// (residual) to the result, all without needing to copy extra data in memory.
    ///
    /// # Safety
    /// The caller must guarantee compatible sizes and buffer validity.
    #[inline(always)]
    pub unsafe fn process_residual_batch<M: SimdMath>(
        &self,
        input: &[f32],
        residual: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        // SAFETY: `M::fused_gemm_residual_batch_f32` is called with `input`/`output` sized
        // `num_frames` frames of `in_ch`/`out_ch` (caller contract), and `M` is selected by the
        // runtime CPUID dispatch matching its `#[target_feature]` backend.
        unsafe {
            M::fused_gemm_residual_batch_f32(
                input,
                &self.weights,
                &self.bias,
                residual,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    /// Full-precision f32 head projection.
    ///
    /// Dispatches to the appropriate SIMD kernel via the `SimdMath` trait,
    /// replacing the previous scalar triple-nested loop with shape-dependent
    /// vectorization (frame-batching for OUT≤4, channel-batching for OUT≥8).
    ///
    /// # Safety
    /// The caller must ensure that `input` and `output` have sizes
    /// compatible with `in_ch`, `out_ch`, and `num_frames`, and that the SIMD
    /// instructions for `M` are available on the host CPU.
    #[inline(always)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        // SAFETY: the GEMV kernels require lane counts satisfied by `input`/`output` (sized per
        // `in_ch`/`out_ch` and `num_frames` by the caller contract), and `M`'s target features are
        // validated by the runtime CPUID dispatch that selected it.
        unsafe {
            if self.do_bias {
                M::gemv_with_bias_f32(input, &self.weights, &self.bias, output, num_frames);
            } else {
                M::gemv_no_bias_f32(input, &self.weights, output, num_frames);
            }
        }
    }
}
