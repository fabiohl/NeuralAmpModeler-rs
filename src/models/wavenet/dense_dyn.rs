// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic 1x1 dense channel-mixing layer (`DenseLayerDyn`) with runtime dimensions.

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

    /// Returns whether this layer still satisfies the size contract of the
    /// unchecked GEMV kernels.
    #[inline]
    pub(crate) fn is_sound(&self) -> bool {
        if self.in_ch == 0 || self.out_ch == 0 {
            return false;
        }
        let Some(min_weights) = self.in_ch.checked_mul(self.out_ch) else {
            return false;
        };
        if self.weights.len() < min_weights {
            return false;
        }
        if self.do_bias && self.bias.len() < self.out_ch {
            return false;
        }
        true
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
        // SAFETY: the const-generic GEMM kernels are only reached for their matching
        // `(in_ch, out_ch)` shapes with buffers validated by this wrapper's caller contract,
        // and `M::fused_gemm_residual_batch_f32` requires lane counts satisfied by
        // `input`/`residual`/`output` sized `num_frames` frames of `in_ch`/`out_ch`
        // (caller contract); `M` is selected by the runtime CPUID dispatch matching its
        // `#[target_feature]` backend.
        unsafe {
            match (self.in_ch, self.out_ch) {
                (4, 4) => {
                    crate::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_const::<4, 4>(
                        input,
                        &self.weights,
                        &self.bias,
                        residual,
                        output,
                        num_frames,
                        self.do_bias,
                    );
                }
                (8, 8) => {
                    crate::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_const::<8, 8>(
                        input,
                        &self.weights,
                        &self.bias,
                        residual,
                        output,
                        num_frames,
                        self.do_bias,
                    );
                }
                (12, 12) => {
                    crate::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_12x12(
                        input,
                        &self.weights,
                        &self.bias,
                        residual,
                        output,
                        num_frames,
                        self.do_bias,
                    );
                }
                (16, 16) => {
                    crate::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_const::<16, 16>(
                        input,
                        &self.weights,
                        &self.bias,
                        residual,
                        output,
                        num_frames,
                        self.do_bias,
                    );
                }
                _ => {
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
        // SAFETY: the broadcast GEMV kernels are reached with the target-feature guarantee
        // of the caller's dispatch path (AVX2+FMA on x86_64, NEON on aarch64), and the
        // buffer slices are trimmed to `num_frames` / `num_frames * out_ch` matching the
        // kernel lane counts per this wrapper's documented contract; the generic `M::gemv_*`
        // paths require the same sizes plus `M`'s backend features validated by runtime CPUID.
        unsafe {
            if self.in_ch == 1 {
                let in_slice = &input[..num_frames];
                let out_slice = &mut output[..num_frames * self.out_ch];
                macro_rules! dispatch_broadcast {
                    ($out_c:literal) => {{
                        if self.do_bias {
                            crate::math::gemm::gemv::broadcast_scale_with_bias_f32_avx2::<$out_c>(
                                in_slice,
                                &self.weights,
                                &self.bias,
                                out_slice,
                            );
                        } else {
                            crate::math::gemm::gemv::broadcast_scale_f32_avx2::<$out_c>(
                                in_slice,
                                &self.weights,
                                out_slice,
                            );
                        }
                        return;
                    }};
                }
                match self.out_ch {
                    4 => dispatch_broadcast!(4),
                    8 => dispatch_broadcast!(8),
                    12 => dispatch_broadcast!(12),
                    16 => dispatch_broadcast!(16),
                    _ => {}
                }
            }
            if self.do_bias {
                M::gemv_with_bias_f32(input, &self.weights, &self.bias, output, num_frames);
            } else {
                M::gemv_no_bias_f32(input, &self.weights, output, num_frames);
            }
        }
    }
}
