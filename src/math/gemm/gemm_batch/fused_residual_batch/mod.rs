// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Modularized Fused Residual Batch GEMM kernels across ISAs.

mod avx2;
#[cfg(nam_avx512)]
mod avx512;
mod scalar;

pub use avx2::*;
#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
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

        // SAFETY: all buffers are sized to the declared dims (num_frames=4, in_len=8,
        // out_len=8): `in_frames`/`residual` hold `num_frames*in_len`/`num_frames*out_len`
        // elements, `weights` holds `in_len*out_len`, `bias` holds `out_len`, and the
        // output buffers hold `num_frames*out_len`; the kernels re-assert these bounds.
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

        // SAFETY: all buffers are sized to the declared dims (num_frames=8, 12x12):
        // `in_frames`/`residual` and the output buffers hold `num_frames*12` elements,
        // `weights` holds `12*12` elements, `bias` holds 12; the kernels re-assert these.
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

    #[test]
    fn test_fused_residual_batch_f32_12x12_per_frame_bit_exact() {
        // The two-frame remainder must not change any frame's arithmetic:
        // frame `n` of a multi-frame call has to match a 1-frame call bitwise.
        for &num_frames in &[1usize, 2, 3, 4, 5, 6, 7] {
            for &with_bias in &[false, true] {
                let in_frames: Vec<f32> = (0..num_frames * 12)
                    .map(|i| ((i as f32 * 0.173).sin() * 1.3) - 0.11)
                    .collect();
                let weights: Vec<f32> = (0..144)
                    .map(|i| ((i as f32 * 0.317).cos() * 0.8) + 0.02)
                    .collect();
                let bias: Vec<f32> = (0..12).map(|i| (i as f32 * 0.21) - 0.3).collect();
                let residual: Vec<f32> = (0..num_frames * 12)
                    .map(|i| ((i as f32 * 0.071).sin() * 0.6) + 0.05)
                    .collect();

                let mut out_batch = vec![0.0f32; num_frames * 12];
                // SAFETY: buffers hold `num_frames * 12` elements and the
                // weights/bias hold 144/12, matching the 12x12 contract.
                unsafe {
                    fused_gemm_residual_batch_f32_12x12(
                        &in_frames,
                        &weights,
                        &bias,
                        &residual,
                        &mut out_batch,
                        num_frames,
                        with_bias,
                    );
                }

                for frame in 0..num_frames {
                    let mut out_one = vec![0.0f32; 12];
                    // SAFETY: single frame slices hold exactly 12 elements and
                    // reuse the same weight/bias buffers as the batch call.
                    unsafe {
                        fused_gemm_residual_batch_f32_12x12(
                            &in_frames[frame * 12..(frame + 1) * 12],
                            &weights,
                            &bias,
                            &residual[frame * 12..(frame + 1) * 12],
                            &mut out_one,
                            1,
                            with_bias,
                        );
                    }
                    for (i, (batch, one)) in out_batch[frame * 12..(frame + 1) * 12]
                        .iter()
                        .zip(out_one.iter())
                        .enumerate()
                    {
                        assert_eq!(
                            batch.to_bits(),
                            one.to_bits(),
                            "frames={num_frames} bias={with_bias} frame={frame} i={i}: \
                             {batch} vs {one}"
                        );
                    }
                }
            }
        }
    }

    fn check_const_pair_per_frame_bit_exact<const IN: usize, const OUT: usize>() {
        // The pair tail must not change any frame's arithmetic: frames 1..3
        // have to match 1-frame calls bitwise. Frame 4+ use the pre-existing
        // dual-accumulator batch block, which is out of this contract.
        for &num_frames in &[1usize, 2, 3] {
            for &with_bias in &[false, true] {
                let in_frames: Vec<f32> = (0..num_frames * IN)
                    .map(|i| ((i as f32 * 0.173).sin() * 1.3) - 0.11)
                    .collect();
                let weights: Vec<f32> = (0..IN * OUT)
                    .map(|i| ((i as f32 * 0.317).cos() * 0.8) + 0.02)
                    .collect();
                let bias: Vec<f32> = (0..OUT).map(|i| (i as f32 * 0.21) - 0.3).collect();
                let residual: Vec<f32> = (0..num_frames * OUT)
                    .map(|i| ((i as f32 * 0.071).sin() * 0.6) + 0.05)
                    .collect();

                let mut out_batch = vec![0.0f32; num_frames * OUT];
                // SAFETY: buffers hold `num_frames * IN/OUT` elements and the
                // weights/bias hold IN*OUT/OUT, matching the const contract.
                unsafe {
                    fused_gemm_residual_batch_f32_const::<IN, OUT>(
                        &in_frames,
                        &weights,
                        &bias,
                        &residual,
                        &mut out_batch,
                        num_frames,
                        with_bias,
                    );
                }

                for frame in 0..num_frames {
                    let mut out_one = vec![0.0f32; OUT];
                    // SAFETY: single-frame slices hold exactly IN/OUT elements
                    // and reuse the same weight/bias buffers as the batch call.
                    unsafe {
                        fused_gemm_residual_batch_f32_const::<IN, OUT>(
                            &in_frames[frame * IN..(frame + 1) * IN],
                            &weights,
                            &bias,
                            &residual[frame * OUT..(frame + 1) * OUT],
                            &mut out_one,
                            1,
                            with_bias,
                        );
                    }
                    for (i, (batch, one)) in out_batch[frame * OUT..(frame + 1) * OUT]
                        .iter()
                        .zip(out_one.iter())
                        .enumerate()
                    {
                        assert_eq!(
                            batch.to_bits(),
                            one.to_bits(),
                            "IN={IN} OUT={OUT} frames={num_frames} bias={with_bias} \
                             frame={frame} i={i}: {batch} vs {one}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_fused_residual_batch_f32_8x8_pair_per_frame_bit_exact() {
        check_const_pair_per_frame_bit_exact::<8, 8>();
    }

    #[test]
    fn test_fused_residual_batch_f32_4x4_bit_exact_vs_scalar() {
        // The 4-wide SSE residual must reproduce the previous scalar
        // arithmetic bit-for-bit: `residual + bias` first, then one
        // `mul` + `add` per input channel, in ascending `in_c` order.
        for &num_frames in &[1usize, 2, 3, 4, 5, 7, 63, 65] {
            for &with_bias in &[false, true] {
                let in_frames: Vec<f32> = (0..num_frames * 4)
                    .map(|i| ((i as f32 * 0.173).sin() * 1.7) - 0.21)
                    .collect();
                let weights: Vec<f32> = (0..16)
                    .map(|i| ((i as f32 * 0.317).cos() * 0.9) + 0.05)
                    .collect();
                let bias: Vec<f32> = (0..4).map(|i| (i as f32 * 0.21) - 0.3).collect();
                let residual: Vec<f32> = (0..num_frames * 4)
                    .map(|i| ((i as f32 * 0.071).sin() * 0.6) + 0.11)
                    .collect();

                let mut out_scalar = vec![0.0f32; num_frames * 4];
                let mut out_sse = vec![0.0f32; num_frames * 4];

                // SAFETY: all buffers hold exactly `num_frames * 4` (or 16)
                // elements, matching the documented 4x4 kernel contract.
                unsafe {
                    fused_gemm_residual_batch_f32_scalar(
                        &in_frames,
                        &weights,
                        &bias,
                        &residual,
                        &mut out_scalar,
                        num_frames,
                        with_bias,
                    );
                    fused_gemm_residual_batch_f32_const::<4, 4>(
                        &in_frames,
                        &weights,
                        &bias,
                        &residual,
                        &mut out_sse,
                        num_frames,
                        with_bias,
                    );
                }

                for (i, (s, a)) in out_scalar.iter().zip(out_sse.iter()).enumerate() {
                    assert_eq!(
                        s.to_bits(),
                        a.to_bits(),
                        "frames={num_frames} bias={with_bias} i={i}: {s} vs {a}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_fused_residual_batch_f32_4x4_signed_zero_bit_exact() {
        // `-0.0 + +0.0` is `+0.0`, so the seed must match each replaced
        // scalar branch: the batch path seeds from the residual, the
        // single-frame tail seeds from bias/+0.0 and then adds the residual.
        for &num_frames in &[1usize, 4, 5] {
            for &with_bias in &[false, true] {
                let in_frames = vec![0.0f32; num_frames * 4];
                let weights = vec![-0.0f32; 16];
                let bias = vec![0.0f32; 4];
                let residual = vec![-0.0f32; num_frames * 4];

                let mut out_scalar = vec![0.0f32; num_frames * 4];
                let mut out_sse = vec![0.0f32; num_frames * 4];

                // SAFETY: buffers match the documented 4x4 batch contract.
                unsafe {
                    fused_gemm_residual_batch_f32_scalar(
                        &in_frames,
                        &weights,
                        &bias,
                        &residual,
                        &mut out_scalar,
                        num_frames,
                        with_bias,
                    );
                    fused_gemm_residual_batch_f32_const::<4, 4>(
                        &in_frames,
                        &weights,
                        &bias,
                        &residual,
                        &mut out_sse,
                        num_frames,
                        with_bias,
                    );
                }

                for (i, (s, a)) in out_scalar.iter().zip(out_sse.iter()).enumerate() {
                    assert_eq!(
                        s.to_bits(),
                        a.to_bits(),
                        "signed zero frames={num_frames} bias={with_bias} i={i}: {s} vs {a}"
                    );
                }
            }
        }
    }
}
