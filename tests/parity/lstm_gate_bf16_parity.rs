// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![cfg(feature = "avx512")]

use neural_amp_modeler_rs::math::common::gemv_4gate_bf16_fallback;
use neural_amp_modeler_rs::math::gemm::gemv_4gate_bf16_avx512;
use proptest::prelude::*;

fn f32_to_bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

prop_compose! {
    fn gemv_4gate_strategy()(
        out_len in 1usize..=128usize,
        in_len in 1usize..=64usize,
        do_bias in proptest::bool::ANY,
    )(
        in_frame in prop::collection::vec(-1.0f32..1.0f32, in_len),
        w0 in prop::collection::vec(-1.0f32..1.0f32, in_len * out_len),
        w1 in prop::collection::vec(-1.0f32..1.0f32, in_len * out_len),
        w2 in prop::collection::vec(-1.0f32..1.0f32, in_len * out_len),
        w3 in prop::collection::vec(-1.0f32..1.0f32, in_len * out_len),
        bias in prop::collection::vec(-1.0f32..1.0f32, 4 * out_len),
        out_len in Just(out_len),
        do_bias in Just(do_bias),
    ) -> (Vec<u16>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize, bool) {
        let bf16_in: Vec<u16> = in_frame.iter().map(|&v| f32_to_bf16_bits(v)).collect();
        (bf16_in, w0, w1, w2, w3, bias, out_len, do_bias)
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(10_000)
    })]

    #[test]
    #[ignore]
    fn test_gemv_4gate_bf16_simd_vs_scalar(
        (in_frame, w0, w1, w2, w3, bias, out_len, do_bias)
        in gemv_4gate_strategy()
    ) {
        if !std::is_x86_feature_detected!("avx512bf16") {
            return Ok(());
        }

        let mut out_simd = vec![0.0f32; 4 * out_len];
        let mut out_scalar = vec![0.0f32; 4 * out_len];

        unsafe {
            gemv_4gate_bf16_avx512(
                &in_frame,
                &w0.iter().map(|&v| f32_to_bf16_bits(v)).collect::<Vec<u16>>(),
                &w1.iter().map(|&v| f32_to_bf16_bits(v)).collect::<Vec<u16>>(),
                &w2.iter().map(|&v| f32_to_bf16_bits(v)).collect::<Vec<u16>>(),
                &w3.iter().map(|&v| f32_to_bf16_bits(v)).collect::<Vec<u16>>(),
                &bias,
                &mut out_simd,
                do_bias,
            );
            gemv_4gate_bf16_fallback(
                &in_frame, &w0, &w1, &w2, &w3, &bias, &mut out_scalar, do_bias,
            );
        }

        for i in 0..out_simd.len() {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            let max_val = out_simd[i].abs().max(out_scalar[i].abs()).max(1.0);
            let rel_diff = diff / max_val;
            assert!(
                rel_diff < 5e-3
                    || (out_simd[i].is_nan() && out_scalar[i].is_nan())
                    || (out_simd[i].is_infinite() && out_scalar[i].is_infinite()),
                "GEMV 4-gate BF16 parity failed at index {}: SIMD={}, Scalar={}, Diff={}, RelDiff={}",
                i, out_simd[i], out_scalar[i], diff, rel_diff
            );
        }
    }
}
