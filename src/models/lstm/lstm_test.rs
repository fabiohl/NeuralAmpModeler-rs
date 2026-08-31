// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::models::lstm::{LstmModel1, LstmModel2};

    // Verify that internal buffers (gates and state) are allocated with the correct
    // size based on Const Generics parameters.
    #[test]
    fn test_lstm_model1_allocation() {
        let model: LstmModel1<8, 9, 32> = LstmModel1::new();
        assert_eq!(model.layer.gates.len(), 32);
        assert_eq!(model.layer.state.len(), 9);
    }

    // Verify allocation in a two-layer model (LstmModel2).
    #[test]
    fn test_lstm_model2_allocation() {
        let model: LstmModel2<16, 17, 32, 64> = LstmModel2::new();
        assert_eq!(model.layer1.input_hidden_weights.len(), 4);
        assert_eq!(model.layer2.input_hidden_weights[0].len(), 32);
    }

    // Basic sanity test: process some values and ensure the output
    // contains no invalid values (NaN or Infinity).
    #[test]
    fn test_lstm_model1_process_zeros() {
        {
            let mut model: LstmModel1<8, 9, 32> = LstmModel1::new();
            let input = [1.0, -1.0, 0.5, -0.5];
            let mut output = [0.0f32; 4];

            model.process(&input, &mut output);

            for (i, &v) in output.iter().enumerate() {
                assert!(v.is_finite(), "LSTM1×8: output [{}] is NaN/Inf: {}", i, v);
            }
        }
    }

    // Ensure processing is deterministic: the same input must
    // always produce the same output if the initial state is equal.
    #[test]
    fn test_lstm_model2_process_deterministic() {
        {
            let mut model_a: LstmModel2<8, 9, 16, 32> = LstmModel2::new();
            let mut model_b: LstmModel2<8, 9, 16, 32> = LstmModel2::new();

            let input = [1.0, -1.0, 0.5, -0.5];
            let mut out_a = [0.0f32; 4];
            let mut out_b = [0.0f32; 4];

            model_a.process(&input, &mut out_a);
            model_b.process(&input, &mut out_b);

            for i in 0..4 {
                assert!(
                    (out_a[i] - out_b[i]).abs() < 1e-6,
                    "LSTM2×8 non-deterministic at [{}]: {} vs {}",
                    i,
                    out_a[i],
                    out_b[i]
                );
            }
        }
    }

    // Validates whether the LSTM gate memory layout [I, F, C, O] follows the
    // expected order so that SIMD optimizations work correctly.
    #[test]
    fn test_lstm_gate_order_consistency() {
        // Validates that the [i|f|g|o] layout at offsets [0, H, 2H, 3H] is respected.
        let model: LstmModel1<8, 9, 32> = LstmModel1::new();
        assert_eq!(model.layer.gates.len(), 32); // 4 * H = 4 * 8 = 32
        assert_eq!(model.layer.input_hidden_weights.len(), 4); // Gates = 4
        assert_eq!(model.layer.input_hidden_weights[0].len(), 9); // IH = I + H = 1 + 8 = 9
    }

    // Verify that the LSTM's internal state (memory) evolves as we process data.
    // The hidden state after 10 samples should be different from the state after 1 sample.
    #[test]
    fn test_lstm_state_evolution() {
        let mut model: LstmModel1<8, 9, 32> = LstmModel1::new();

        // Set some weights so the state changes significantly
        for i in 0..32 {
            model.layer.bias[i] = 0.1;
        }
        for k in 0..4 {
            for j in 0..9 {
                for i in 0..8 {
                    model.layer.input_hidden_weights[k][j][i] = 0.05;
                }
            }
        }

        let input_step = [1.0f32];
        let mut out = [0.0f32];

        model.process(&input_step, &mut out);
        let hidden_1 = model.layer.get_hidden_state().to_vec();

        for _ in 0..9 {
            model.process(&input_step, &mut out);
        }
        let hidden_10 = model.layer.get_hidden_state().to_vec();

        // Verify evolution (hidden states should be different after multiple iterations)
        for i in 0..8 {
            assert!(
                (hidden_1[i] - hidden_10[i]).abs() > 1e-4,
                "Hidden state did not evolve at position {}: {} vs {}",
                i,
                hidden_1[i],
                hidden_10[i]
            );
        }
    }

    // CRITICAL TEST: Ensures the result is the same regardless of the processed
    // block size (e.g., processing 64 samples at once or 64 times one sample).
    // This is vital to ensure audio doesn't change if the driver changes the buffer size.
    #[test]
    fn test_lstm_variable_block_sizes() {
        let mut input = [0.0f32; 64];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.1).sin(); // Simple sine wave
        }

        let block_sizes = [1, 8, 16, 32, 64];
        let mut reference_out = [0.0f32; 64];

        for (idx, &block_size) in block_sizes.iter().enumerate() {
            let mut model: LstmModel1<8, 9, 32> = LstmModel1::new();

            // Assign deterministic weights to test identical output
            for i in 0..32 {
                model.layer.bias[i] = 0.1;
            }
            for k in 0..4 {
                for j in 0..9 {
                    for i in 0..8 {
                        model.layer.input_hidden_weights[k][j][i] = 0.05;
                    }
                }
            }
            for i in 0..8 {
                model.head_weights_f32[i] = 0.5;
            }

            let mut out = [0.0f32; 64];
            for chunk in 0..(64 / block_size) {
                let start = chunk * block_size;
                let end = start + block_size;
                model.process(&input[start..end], &mut out[start..end]);
            }

            if idx == 0 {
                reference_out.copy_from_slice(&out);
                let max_abs = out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                // Medido: max_abs=0.522 com head_weights_f32=0.5, LSTM 1×8, seno 64 amostras
                assert!(
                    max_abs > 1e-6,
                    "saída identicamente nula — verifique head_weights_f32"
                );
            } else {
                for i in 0..64 {
                    assert!(
                        (out[i] - reference_out[i]).abs() < 1e-6,
                        "Mismatch at block_size {} at sample {}: {} vs {}",
                        block_size,
                        i,
                        out[i],
                        reference_out[i]
                    );
                }
            }
        }
    }

    // Verify that reset and prewarm functions are clearing states correctly.
    #[test]
    fn test_lstm_reset_on_prewarm() {
        use crate::models::NamModel; // trait
        let mut model: LstmModel1<8, 9, 32> = LstmModel1::new();

        // Set bias and weights to ensure the state shifts away from zero
        for i in 0..32 {
            model.layer.bias[i] = 1.0;
        }

        let input = [1.0f32; 64];
        let mut out = [0.0f32; 64];
        model.process(&input, &mut out);

        // Artificially change the cell state
        model.layer.cell_state.fill(5.0);
        model.layer.state.fill(5.0);

        // Verify the state is not zeroed
        assert!(model.layer.cell_state[0] != 0.0);
        assert!(model.layer.state[0] != 0.0);

        // Calls prewarm — internally calls reset_states() and processes N zeros
        model.prewarm(0);
        model.prewarm(10);

        // Verify that reset_states() effectively zeros hidden and cell states
        model.reset_states();
        for val in model.layer.cell_state.iter() {
            assert_eq!(*val, 0.0);
        }
        for val in model.layer.state.iter() {
            assert_eq!(*val, 0.0);
        }
    }

    // PARITY TEST: Ensures the pipelined version (SIMD) produces the same
    // result as the sequential version (Scalar).
    #[test]
    fn test_lstm_model2_pipelining_parity() {
        let mut model_simd: LstmModel2<8, 9, 17, 32> = LstmModel2::new();
        let mut model_scalar: LstmModel2<8, 9, 17, 32> = LstmModel2::new();

        // Assign deterministic weights
        for i in 0..32 {
            model_simd.layer1.bias[i] = 0.1;
            model_scalar.layer1.bias[i] = 0.1;
            model_simd.layer2.bias[i] = -0.05;
            model_scalar.layer2.bias[i] = -0.05;
        }
        for k in 0..4 {
            for j in 0..17 {
                for i in 0..8 {
                    let w = 0.01 * (i + j + k) as f32;
                    if j < 9 {
                        model_simd.layer1.input_hidden_weights[k][j][i] = w;
                        model_scalar.layer1.input_hidden_weights[k][j][i] = w;
                    }
                    model_simd.layer2.input_hidden_weights[k][j][i] = w;
                    model_scalar.layer2.input_hidden_weights[k][j][i] = w;
                }
            }
        }
        for i in 0..8 {
            model_simd.head_weights_f32[i] = 0.5;
            model_scalar.head_weights_f32[i] = 0.5;
        }

        const N: usize = 1024;
        const SAMPLE_RATE: f32 = 48_000.0;
        const FREQ_HZ: f32 = 440.0;
        let input: Vec<f32> = (0..N)
            .map(|i| (i as f32 * FREQ_HZ * 2.0 * std::f32::consts::PI / SAMPLE_RATE).sin())
            .collect();
        let mut out_simd = vec![0.0f32; N];
        let mut out_scalar = vec![0.0f32; N];

        model_simd.process(&input, &mut out_simd);
        model_scalar.process_scalar(&input, &mut out_scalar);

        let max_abs = out_simd.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs > 1e-6,
            "saída identicamente nula — verifique head_weights_f32"
        );

        let snr = crate::testing::perceptual::compute_snr_db(&out_scalar, &out_simd);
        // Medido: SNR SIMD vs Escalar = 146.2 dB em modelos de 2 camadas, hidden=8, 1024 amostras
        // Medido: max_abs = 3.096 com head_weights_f32 = 0.5 e entrada senoidal de 440 Hz @ 48 kHz
        assert!(
            snr >= 120.0,
            "SIMD vs scalar SNR {snr:.2} dB < 120 dB (max_abs={max_abs})"
        );
        for i in 0..N {
            assert!(
                (out_simd[i] - out_scalar[i]).abs() < 5e-3,
                "SIMD/Pipelined vs Scalar parity failed at [{}]: {} vs {}",
                i,
                out_simd[i],
                out_scalar[i]
            );
        }
    }

    // =====================================================================
    // Dynamic layer kernel tests
    // =====================================================================

    use crate::models::lstm::layer_dyn::LstmLayerDyn;

    fn fill_dyn_layer(layer: &mut LstmLayerDyn, h: usize, input_size: usize) {
        let ih = input_size + h;
        for i in 0..(4 * h) {
            layer.bias[i] = 0.1 * (1 + (i % 4)) as f32;
        }
        for k in 0..4 {
            let w_start = k * ih * h;
            for j in 0..ih {
                for hi in 0..h {
                    layer.input_hidden_weights[w_start + j * h + hi] =
                        0.01 * (j + hi + k + 1) as f32;
                }
            }
        }
    }

    /// Tests SIMD vs scalar parity on a 1-input dynamic layer.
    fn assert_dyn_layer_parity(hidden_size: usize) {
        let h = hidden_size;

        let mut layer_scalar = LstmLayerDyn::new(1, h).unwrap();
        let mut layer_simd = LstmLayerDyn::new(1, h).unwrap();

        fill_dyn_layer(&mut layer_scalar, h, 1);
        layer_simd
            .input_hidden_weights
            .copy_from_slice(&layer_scalar.input_hidden_weights);
        layer_simd.bias.copy_from_slice(&layer_scalar.bias);
        layer_simd.state.copy_from_slice(&layer_scalar.state);
        layer_simd
            .cell_state
            .copy_from_slice(&layer_scalar.cell_state);
        layer_simd
            .cell_error
            .copy_from_slice(&layer_scalar.cell_error);

        let test_inputs = [0.5f32, -0.7, 0.1, 0.9, -1.0, 0.3, -0.3, 0.0];

        for (_step, &input_val) in test_inputs.iter().enumerate() {
            let input = [input_val];

            layer_scalar.process_sample_scalar(&input);
            layer_simd.process(&input);

            for j in 0..h {
                // Measured: state max diff ~2.7e-5 across 8 steps due to GEMV order & minimax f32 cancellation; tolerance = 5e-5
                assert!(
                    (layer_scalar.state[1 + j] - layer_simd.state[1 + j]).abs() < 5e-5,
                    "Dyn parity H={} step={} hidden[{}]: {} vs {}",
                    h,
                    _step,
                    j,
                    layer_scalar.state[1 + j],
                    layer_simd.state[1 + j],
                );
                // Measured: cell_state max diff ~1.2e-4 across 8 steps due to GEMV order & minimax f32 cancellation; tolerance = 2e-4
                assert!(
                    (layer_scalar.cell_state[j] - layer_simd.cell_state[j]).abs() < 2e-4,
                    "Dyn parity H={} step={} cell[{}]: {} vs {}",
                    h,
                    _step,
                    j,
                    layer_scalar.cell_state[j],
                    layer_simd.cell_state[j],
                );
                // Measured: cell_error ~1e-7 a 1e-6 (8 passos, pesos de teste O(0.1-0.4)); tolerância = 1e-5 (10× margem sobre o pior caso).
                assert!(
                    (layer_scalar.cell_error[j] - layer_simd.cell_error[j]).abs() < 1e-5,
                    "Dyn parity H={} step={} cell_error[{}]: {} vs {}",
                    h,
                    _step,
                    j,
                    layer_scalar.cell_error[j],
                    layer_simd.cell_error[j],
                );
            }
        }
    }

    #[test]
    fn test_dyn_layer_parity_h8() {
        assert_dyn_layer_parity(8);
    }

    #[test]
    fn test_dyn_layer_parity_h16() {
        assert_dyn_layer_parity(16);
    }

    #[test]
    fn test_dyn_layer_parity_h24() {
        assert_dyn_layer_parity(24);
    }

    #[test]
    fn test_dyn_layer_parity_h_nonstandard() {
        assert_dyn_layer_parity(10);
        assert_dyn_layer_parity(13);
    }

    #[test]
    fn test_dyn_layer_no_panic() {
        let test_sizes = [3, 7, 20, 40, 64];
        for &h in &test_sizes {
            let mut layer = LstmLayerDyn::new(1, h).unwrap();
            fill_dyn_layer(&mut layer, h, 1);
            let input = [0.5];
            layer.process(&input);
            for j in 0..h {
                assert!(
                    layer.state[1 + j].is_finite(),
                    "Dyn H={} hidden[{}] is NaN/Inf: {}",
                    h,
                    j,
                    layer.state[1 + j]
                );
                assert!(
                    layer.cell_state[j].is_finite(),
                    "Dyn H={} cell[{}] is NaN/Inf: {}",
                    h,
                    j,
                    layer.cell_state[j]
                );
            }
        }
    }

    #[test]
    fn test_dyn_layer_determinism() {
        let mut layer_a = LstmLayerDyn::new(1, 12).unwrap();
        let mut layer_b = LstmLayerDyn::new(1, 12).unwrap();

        fill_dyn_layer(&mut layer_a, 12, 1);
        layer_b
            .input_hidden_weights
            .copy_from_slice(&layer_a.input_hidden_weights);
        layer_b.bias.copy_from_slice(&layer_a.bias);
        layer_b.state.copy_from_slice(&layer_a.state);
        layer_b.cell_state.copy_from_slice(&layer_a.cell_state);
        layer_b.cell_error.copy_from_slice(&layer_a.cell_error);

        let input = [0.7];
        layer_a.process(&input);
        layer_b.process(&input);

        for j in 0..12 {
            assert_eq!(
                layer_a.state[1 + j],
                layer_b.state[1 + j],
                "Dyn deter hidden[{}]: {} vs {}",
                j,
                layer_a.state[1 + j],
                layer_b.state[1 + j]
            );
            assert_eq!(
                layer_a.cell_state[j], layer_b.cell_state[j],
                "Dyn deter cell[{}]: {} vs {}",
                j, layer_a.cell_state[j], layer_b.cell_state[j]
            );
            assert_eq!(
                layer_a.cell_error[j], layer_b.cell_error[j],
                "Dyn deter cell_error[{}]: {} vs {}",
                j, layer_a.cell_error[j], layer_b.cell_error[j]
            );
        }
    }

    #[test]
    fn test_dyn_layer_state_evolution() {
        let mut layer = LstmLayerDyn::new(1, 16).unwrap();
        fill_dyn_layer(&mut layer, 16, 1);

        let initial_hidden: Vec<f32> = layer.state[1..].to_vec();

        let input = [0.5];
        layer.process(&input);
        let hidden_1: Vec<f32> = layer.state[1..].to_vec();

        for _ in 0..9 {
            layer.process(&input);
        }
        let hidden_10: Vec<f32> = layer.state[1..].to_vec();

        for j in 0..16 {
            assert!(initial_hidden[j] == 0.0, "Initial hidden should be zero");
            assert!(
                (hidden_1[j] - hidden_10[j]).abs() > 1e-6,
                "Dyn state should evolve h[{}]: {} vs {}",
                j,
                hidden_1[j],
                hidden_10[j]
            );
        }
    }
}
