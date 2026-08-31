// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use neural_amp_modeler_rs::models::lstm::LstmModel1;
use proptest::prelude::*;

prop_compose! {
    fn lstm_parity_strategy()(
        inputs in prop::collection::vec(-1.0f32..1.0f32, 1),
        weights in prop::collection::vec(-1.0f32..1.0f32, 288),
        bias in prop::collection::vec(-1.0f32..1.0f32, 32),
        head_weights in prop::collection::vec(-1.0f32..1.0f32, 8),
        head_bias in -1.0f32..1.0f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        (inputs, weights, bias, head_weights, head_bias)
    }
}

prop_compose! {
    fn lstm_1x40_parity_strategy()(
        inputs in prop::collection::vec(-1.0f32..1.0f32, 256),
        weights in prop::collection::vec(-1.5f32..1.5f32, 6560),
        bias in prop::collection::vec(-1.5f32..1.5f32, 160),
        head_weights in prop::collection::vec(-1.5f32..1.5f32, 40),
        head_bias in -1.5f32..1.5f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        (inputs, weights, bias, head_weights, head_bias)
    }
}

prop_compose! {
    fn lstm_2x24_parity_strategy()(
        inputs in prop::collection::vec(-1.0f32..1.0f32, 256),
        w1 in prop::collection::vec(-1.5f32..1.5f32, 2400),
        w2 in prop::collection::vec(-1.5f32..1.5f32, 4608),
        b1 in prop::collection::vec(-1.5f32..1.5f32, 96),
        b2 in prop::collection::vec(-1.5f32..1.5f32, 96),
        head_weights in prop::collection::vec(-1.5f32..1.5f32, 24),
        head_bias in -1.5f32..1.5f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        (inputs, w1, w2, b1, b2, head_weights, head_bias)
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(1_000)
    })]

    #[test]
    #[ignore]
    fn test_lstm_scalar_vs_simd_parity((inputs, weights, bias, head_weights, head_bias) in lstm_parity_strategy()) {
        let scale = f32::MAX / 4.0;

        // Force all weights and biases to extreme values (+scale / -scale)
        // to guarantee full saturation according to acceptance criteria.
        let scaled_weights: Vec<f32> = weights.iter().map(|&w| if w >= 0.0 { scale } else { -scale }).collect();
        let scaled_bias: Vec<f32> = bias.iter().map(|&b| if b >= 0.0 { scale } else { -scale }).collect();
        let scaled_head_weights: Vec<f32> = head_weights.iter().map(|&w| if w >= 0.0 { scale } else { -scale }).collect();
        let scaled_head_bias = if head_bias >= 0.0 { scale } else { -scale };

        let mut model_simd = LstmModel1::<8, 9, 32>::new();
        let mut model_scalar = LstmModel1::<8, 9, 32>::new();

        let mut w_idx = 0;
        for k in 0..4 {
            for j in 0..9 {
                for i in 0..8 {
                    let w = scaled_weights[w_idx];
                    model_simd.layer.input_hidden_weights[k][j][i] = w;
                    model_scalar.layer.input_hidden_weights[k][j][i] = w;
                    w_idx += 1;
                }
            }
        }

        model_simd.layer.bias.copy_from_slice(&scaled_bias[..32]);
        model_scalar.layer.bias.copy_from_slice(&scaled_bias[..32]);

        for (i, &w) in scaled_head_weights.iter().enumerate().take(8) {
            model_simd.head_weights_f32[i] = w;
            model_scalar.head_weights_f32[i] = w;
        }

        model_simd.head_bias = scaled_head_bias;
        model_scalar.head_bias = scaled_head_bias;

        model_simd.reset_states();
        model_scalar.reset_states();

        let mut out_simd = vec![0.0f32; inputs.len()];
        let mut out_scalar = vec![0.0f32; inputs.len()];

        model_simd.process(&inputs, &mut out_simd);
        model_scalar.process_scalar(&inputs, &mut out_scalar);

        for i in 0..inputs.len() {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            let max_val = out_simd[i].abs().max(out_scalar[i].abs()).max(1.0);
            let rel_diff = diff / max_val;
            assert!(
                rel_diff < 5.0e-3 || (out_simd[i].is_nan() && out_scalar[i].is_nan()) || (out_simd[i].is_infinite() && out_scalar[i].is_infinite()),
                "LSTM Parity failed at index {}: SIMD={}, Scalar={}, Delta={}, Relative Delta={}",
                i,
                out_simd[i],
                out_scalar[i],
                diff,
                rel_diff
            );
        }
    }

    #[test]
    #[ignore]
    fn test_lstm_1x40_scalar_simd_parity((inputs, weights, bias, head_weights, head_bias) in lstm_1x40_parity_strategy()) {
        let mut model_simd = LstmModel1::<40, 41, 160>::new();
        let mut model_scalar = LstmModel1::<40, 41, 160>::new();

        let mut w_idx = 0;
        for k in 0..4 {
            for j in 0..41 {
                for i in 0..40 {
                    // Scale weights for recurrent stability
                    let w = weights[w_idx] / 8.0;
                    model_simd.layer.input_hidden_weights[k][j][i] = w;
                    model_scalar.layer.input_hidden_weights[k][j][i] = w;
                    w_idx += 1;
                }
            }
        }

        model_simd.layer.bias.copy_from_slice(&bias[..160]);
        model_scalar.layer.bias.copy_from_slice(&bias[..160]);
        // Scale bias down for stability
        for i in 0..160 {
            model_simd.layer.bias[i] /= 8.0;
            model_scalar.layer.bias[i] /= 8.0;
        }

        for (i, &w) in head_weights.iter().enumerate().take(40) {
            model_simd.head_weights_f32[i] = w;
            model_scalar.head_weights_f32[i] = w;
        }

        model_simd.head_bias = head_bias;
        model_scalar.head_bias = head_bias;

        model_simd.reset_states();
        model_scalar.reset_states();

        let mut out_simd = vec![0.0f32; inputs.len()];
        let mut out_scalar = vec![0.0f32; inputs.len()];

        model_simd.process(&inputs, &mut out_simd);
        model_scalar.process_scalar(&inputs, &mut out_scalar);

        for i in 0..inputs.len() {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            let max_val = out_simd[i].abs().max(out_scalar[i].abs()).max(1.0);
            let rel_diff = diff / max_val;
            assert!(
                rel_diff < 5.0e-3 || (out_simd[i].is_nan() && out_scalar[i].is_nan()) || (out_simd[i].is_infinite() && out_scalar[i].is_infinite()),
                "LSTM 1x40 Parity failed at index {}: SIMD={}, Scalar={}, Delta={}, Relative Delta={}",
                i,
                out_simd[i],
                out_scalar[i],
                diff,
                rel_diff
            );
        }
    }

    #[test]
    #[ignore]
    fn test_lstm_2x24_scalar_simd_parity((inputs, w1, w2, b1, b2, head_weights, head_bias) in lstm_2x24_parity_strategy()) {
        use neural_amp_modeler_rs::models::lstm::LstmModel2;

        let mut model_simd = LstmModel2::<24, 25, 48, 96>::new();
        let mut model_scalar = LstmModel2::<24, 25, 48, 96>::new();

        let mut w1_idx = 0;
        for k in 0..4 {
            for j in 0..25 {
                for i in 0..24 {
                    // Scale weights for stability
                    let w = w1[w1_idx] / 8.0;
                    model_simd.layer1.input_hidden_weights[k][j][i] = w;
                    model_scalar.layer1.input_hidden_weights[k][j][i] = w;
                    w1_idx += 1;
                }
            }
        }

        let mut w2_idx = 0;
        for k in 0..4 {
            for j in 0..48 {
                for i in 0..24 {
                    // Scale weights for stability
                    let w = w2[w2_idx] / 8.0;
                    model_simd.layer2.input_hidden_weights[k][j][i] = w;
                    model_scalar.layer2.input_hidden_weights[k][j][i] = w;
                    w2_idx += 1;
                }
            }
        }

        model_simd.layer1.bias.copy_from_slice(&b1[..96]);
        model_scalar.layer1.bias.copy_from_slice(&b1[..96]);

        model_simd.layer2.bias.copy_from_slice(&b2[..96]);
        model_scalar.layer2.bias.copy_from_slice(&b2[..96]);

        // Scale bias down for stability
        for i in 0..96 {
            model_simd.layer1.bias[i] /= 8.0;
            model_scalar.layer1.bias[i] /= 8.0;
            model_simd.layer2.bias[i] /= 8.0;
            model_scalar.layer2.bias[i] /= 8.0;
        }

        for (i, &w) in head_weights.iter().enumerate().take(24) {
            model_simd.head_weights_f32[i] = w;
            model_scalar.head_weights_f32[i] = w;
        }

        model_simd.head_bias = head_bias;
        model_scalar.head_bias = head_bias;

        model_simd.reset_states();
        model_scalar.reset_states();

        let mut out_simd = vec![0.0f32; inputs.len()];
        let mut out_scalar = vec![0.0f32; inputs.len()];

        model_simd.process(&inputs, &mut out_simd);
        model_scalar.process_scalar(&inputs, &mut out_scalar);

        for i in 0..inputs.len() {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            let max_val = out_simd[i].abs().max(out_scalar[i].abs()).max(1.0);
            let rel_diff = diff / max_val;
            assert!(
                rel_diff < 5.0e-3 || (out_simd[i].is_nan() && out_scalar[i].is_nan()) || (out_simd[i].is_infinite() && out_scalar[i].is_infinite()),
                "LSTM 2x24 Parity failed at index {}: SIMD={}, Scalar={}, Delta={}, Relative Delta={}",
                i,
                out_simd[i],
                out_scalar[i],
                diff,
                rel_diff
            );
        }
    }
}
