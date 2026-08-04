// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Modularized Fused Residual Batch GEMM kernels across ISAs.

mod avx2;
mod avx512;
mod scalar;

pub use avx2::*;
pub use avx512::*;
pub use scalar::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_residual_batch_parity_avx2_vs_scalar() {
        let num_frames = 4;
        let in_len = 8;
        let out_len = 8;

        let in_frames: Vec<f32> = (0..num_frames * in_len).map(|i| i as f32 * 0.1).collect();
        let weights: Vec<f32> = (0..in_len * out_len)
            .map(|i| (i as f32 * 0.05) - 0.2)
            .collect();
        let bias: Vec<f32> = (0..out_len).map(|i| i as f32 * 0.01).collect();
        let residual: Vec<f32> = (0..num_frames * out_len)
            .map(|i| 1.0 + i as f32 * 0.02)
            .collect();

        let mut out_scalar = vec![0.0f32; num_frames * out_len];
        let mut out_avx2 = vec![0.0f32; num_frames * out_len];

        unsafe {
            fused_gemm_residual_batch_scalar(
                &in_frames,
                &weights,
                &bias,
                &residual,
                &mut out_scalar,
                num_frames,
                true,
            );

            fused_gemm_residual_batch_avx2(
                &in_frames,
                &weights,
                &bias,
                &residual,
                &mut out_avx2,
                num_frames,
                true,
            );
        }

        for (s, a) in out_scalar.iter().zip(out_avx2.iter()) {
            assert!(
                (s - a).abs() < 1e-5,
                "Mismatch between scalar ({s}) and AVX2 ({a})"
            );
        }
    }

    #[test]
    fn test_fused_residual_batch_f32_12x12_parity() {
        let num_frames = 8;
        let in_frames: Vec<f32> = (0..num_frames * 12).map(|i| i as f32 * 0.05).collect();
        let weights: Vec<f32> = (0..144).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let bias: Vec<f32> = (0..12).map(|i| i as f32 * 0.02).collect();
        let residual: Vec<f32> = (0..num_frames * 12)
            .map(|i| 0.5 + i as f32 * 0.01)
            .collect();

        let mut out_scalar = vec![0.0f32; num_frames * 12];
        let mut out_12x12 = vec![0.0f32; num_frames * 12];

        unsafe {
            fused_gemm_residual_batch_f32_scalar(
                &in_frames,
                &weights,
                &bias,
                &residual,
                &mut out_scalar,
                num_frames,
                true,
            );

            fused_gemm_residual_batch_f32_12x12(
                &in_frames,
                &weights,
                &bias,
                &residual,
                &mut out_12x12,
                num_frames,
                true,
            );
        }

        for (s, a) in out_scalar.iter().zip(out_12x12.iter()) {
            assert!(
                (s - a).abs() < 1e-5,
                "Mismatch between scalar ({s}) and 12x12 ({a})"
            );
        }
    }
}
