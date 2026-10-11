// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Validation suite for LSTM Multi-I/O (T4.2).
//!
//! Covers:
//! 1. LSTM 2->2 H=1 mathematical parity against an f64 oracle.
//! 2. Equivalence between planar `process_raw` (`run_staged_io`) and interleaved `process`.
//! 3. Diagnostic rejection of `input_size != in_channels` with E1316 (`InvalidFeatureTopology`).
//! 4. Rejection of invalid weight count (`WeightCountMismatch`).
//! 5. Multichannel geometry declarations (`num_input_channels` / `num_output_channels`).

use neural_amp_modeler_rs::common::diagnostics::NamErrorCode;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{get_lstm_topology, parse_nam_json};
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::lstm::{LstmModelDyn, lstm_weight_count_io};

/// Mathematical sigmoid in f64.
fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Per-layer serialized tensors for the multi-layer LSTM decoder oracle.
struct SerializedLayer {
    weights: Vec<f32>,
    bias: Vec<f32>,
    hidden_init: Vec<f32>,
    cell_init: Vec<f32>,
}

#[test]
fn test_lstm_2to2_h1_f64_oracle_parity() {
    let num_layers = 1;
    let hidden_size = 1;
    let in_channels = 2;
    let out_channels = 2;

    let mut model = LstmModelDyn::new_io(num_layers, hidden_size, in_channels, out_channels)
        .expect("build lstm dyn");

    // Weights setup:
    // ih = in_channels + hidden_size = 3
    // 4 gates * 3 * 1 = 12 weights
    let w_ih = [
        // gate 0 (i): in0, in1, h0
        0.2f32, -0.1, 0.3, // gate 1 (f): in0, in1, h0
        0.5, 0.4, -0.2, // gate 2 (g): in0, in1, h0
        -0.3, 0.6, 0.1, // gate 3 (o): in0, in1, h0
        0.1, -0.5, 0.4,
    ];
    let bias_vals = [0.05f32, -0.02, 0.1, -0.05];
    model.layers[0].input_hidden_weights.copy_from_slice(&w_ih);
    model.layers[0].bias.copy_from_slice(&bias_vals);

    // Head: out_channels * hidden_size = 2 * 1 = 2
    model.head_weights_f32.copy_from_slice(&[0.7, -0.8]);
    model.head_bias.copy_from_slice(&[0.1, -0.15]);

    // Test signal: 32 stereo frames (64 samples interleaved)
    let n_frames = 32;
    let mut input_interleaved = vec![0.0f32; n_frames * in_channels];
    for f in 0..n_frames {
        input_interleaved[f * 2] = (f as f32 * 0.1).sin();
        input_interleaved[f * 2 + 1] = (f as f32 * 0.15).cos();
    }

    let mut output_interleaved = vec![0.0f32; n_frames * out_channels];
    model.process(&input_interleaved, &mut output_interleaved);

    // f64 Reference Simulation:
    let mut c_state = 0.0f64;
    let mut h_state = 0.0f64;
    let mut expected_output = vec![0.0f64; n_frames * out_channels];

    for f in 0..n_frames {
        let in0 = input_interleaved[f * 2] as f64;
        let in1 = input_interleaved[f * 2 + 1] as f64;

        // gate 0 (i)
        let gi_raw = (w_ih[0] as f64) * in0
            + (w_ih[1] as f64) * in1
            + (w_ih[2] as f64) * h_state
            + (bias_vals[0] as f64);
        let gi = sigmoid_f64(gi_raw);

        // gate 1 (f)
        let gf_raw = (w_ih[3] as f64) * in0
            + (w_ih[4] as f64) * in1
            + (w_ih[5] as f64) * h_state
            + (bias_vals[1] as f64);
        let gf = sigmoid_f64(gf_raw);

        // gate 2 (g)
        let gg_raw = (w_ih[6] as f64) * in0
            + (w_ih[7] as f64) * in1
            + (w_ih[8] as f64) * h_state
            + (bias_vals[2] as f64);
        let gg = (gg_raw).tanh();

        // gate 3 (o)
        let go_raw = (w_ih[9] as f64) * in0
            + (w_ih[10] as f64) * in1
            + (w_ih[11] as f64) * h_state
            + (bias_vals[3] as f64);
        let go = sigmoid_f64(go_raw);

        c_state = gf * c_state + gi * gg;
        h_state = go * c_state.tanh();

        expected_output[f * 2] =
            (model.head_weights_f32[0] as f64) * h_state + (model.head_bias[0] as f64);
        expected_output[f * 2 + 1] =
            (model.head_weights_f32[1] as f64) * h_state + (model.head_bias[1] as f64);
    }

    // Compare with tolerance for f32 SIMD/Pade approximations vs f64 libm
    for i in 0..expected_output.len() {
        let diff = (output_interleaved[i] as f64 - expected_output[i]).abs();
        assert!(
            diff < 1e-3,
            // Measured: max diff between SIMD Pade/Minimax and f64 oracle is < 1e-3
            "Sample {i} diverged from f64 oracle: actual={}, expected={}, diff={diff}",
            output_interleaved[i],
            expected_output[i]
        );
    }

    // Test Planar equivalence via NamModel::process_raw
    let mut model_planar =
        LstmModelDyn::new_io(num_layers, hidden_size, in_channels, out_channels).unwrap();
    model_planar.layers[0]
        .input_hidden_weights
        .copy_from_slice(&w_ih);
    model_planar.layers[0].bias.copy_from_slice(&bias_vals);
    model_planar
        .head_weights_f32
        .copy_from_slice(&model.head_weights_f32);
    model_planar.head_bias.copy_from_slice(&model.head_bias);

    let in_ch0: Vec<f32> = (0..n_frames).map(|f| input_interleaved[f * 2]).collect();
    let in_ch1: Vec<f32> = (0..n_frames)
        .map(|f| input_interleaved[f * 2 + 1])
        .collect();
    let input_ptrs = [in_ch0.as_ptr(), in_ch1.as_ptr()];

    let mut out_ch0 = vec![0.0f32; n_frames];
    let mut out_ch1 = vec![0.0f32; n_frames];
    let mut output_ptrs = [out_ch0.as_mut_ptr(), out_ch1.as_mut_ptr()];

    // SAFETY: buffers allocated for n_frames, matching pointers and channels
    unsafe {
        model_planar.process_raw(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr(), n_frames);
    }

    for f in 0..n_frames {
        assert_eq!(
            out_ch0[f],
            output_interleaved[f * 2],
            "Planar channel 0 frame {f} must match interleaved output exactly"
        );
        assert_eq!(
            out_ch1[f],
            output_interleaved[f * 2 + 1],
            "Planar channel 1 frame {f} must match interleaved output exactly"
        );
    }
}

#[test]
fn test_lstm_mismatch_input_size_in_channels_rejected() {
    let json = r#"{
        "version": "0.5.4",
        "architecture": "LSTM",
        "config": {
            "num_layers": 1,
            "hidden_size": 8,
            "in_channels": 2,
            "input_size": 1,
            "out_channels": 2,
            "layers": []
        },
        "weights": [0.0]
    }"#;
    let parsed = parse_nam_json(json).expect("parse valid JSON");
    let err = get_lstm_topology(&parsed).unwrap_err();
    assert_eq!(err.code, NamErrorCode::InvalidFeatureTopology);
    assert_eq!(err.code.code(), "E1316");
}

#[test]
fn test_lstm_weight_count_mismatch_rejected() {
    let expected = lstm_weight_count_io(1, 8, 2, 2);
    // 4*8*(2+8) + 4*8 + 2*8 + 2 + 2*8 (preload) = 386
    assert_eq!(expected, 386);

    let wrong_weights = vec![0.1f32; expected - 1];
    let json = format!(
        r#"{{
            "version": "0.5.4",
            "architecture": "LSTM",
            "config": {{
                "num_layers": 1,
                "hidden_size": 8,
                "in_channels": 2,
                "out_channels": 2,
                "layers": []
            }},
            "weights": {:?}
        }}"#,
        wrong_weights
    );
    let parsed = parse_nam_json(&json).expect("parse valid JSON");
    let err = match build_model(&parsed) {
        Err(e) => e,
        Ok(_) => panic!("expected build_model to fail on wrong weight count"),
    };
    let err_str = err.to_string();
    assert!(
        err_str.contains("WEIGHT_COUNT_MISMATCH")
            || err_str.contains("Weight count mismatch")
            || err_str.contains("E1302")
            || err_str.contains("DimensionMismatch"),
        "expected weight count mismatch, got {err_str}"
    );
}

#[test]
fn test_lstm_dyn_multichannel_geometry() {
    let m_2x3 = LstmModelDyn::new_io(1, 4, 2, 3).expect("2->3");
    assert_eq!(m_2x3.num_input_channels(), 2);
    assert_eq!(m_2x3.num_output_channels(), 3);

    let m_1x1 = LstmModelDyn::new_io(1, 4, 1, 1).expect("1->1");
    assert_eq!(m_1x1.num_input_channels(), 1);
    assert_eq!(m_1x1.num_output_channels(), 1);
}

#[test]
fn test_lstm_multichannel_zero_alloc_process() {
    let mut model = LstmModelDyn::new_io(1, 4, 2, 2).expect("model 2x2");
    let input = vec![0.0f32; 128];
    let mut output = vec![0.0f32; 128];

    // Warm-up to prime any internal structures
    model.process(&input, &mut output);

    let count = {
        let _guard = crate::common::alloc_audit::TrackingGuard::new();
        model.process(&input, &mut output);
        crate::common::alloc_audit::get_alloc_count()
    };
    assert_eq!(
        count, 0,
        "LstmModelDyn::process must never allocate in hot-path"
    );

    let in_ch0 = vec![0.0f32; 64];
    let in_ch1 = vec![0.0f32; 64];
    let input_ptrs = [in_ch0.as_ptr(), in_ch1.as_ptr()];
    let mut out_ch0 = vec![0.0f32; 64];
    let mut out_ch1 = vec![0.0f32; 64];
    let mut output_ptrs = [out_ch0.as_mut_ptr(), out_ch1.as_mut_ptr()];

    let count_raw = {
        let _guard = crate::common::alloc_audit::TrackingGuard::new();
        unsafe {
            model.process_raw(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr(), 64);
        }
        crate::common::alloc_audit::get_alloc_count()
    };
    assert_eq!(
        count_raw, 0,
        "NamModel::process_raw must never allocate in hot-path"
    );
}

/// Serialized `.nam` decode path (`parse_nam_json` → `build_model` → `LstmModelDyn`)
/// for **2 layers, hidden_size = 2, 2→2** must reproduce an f64 oracle.
///
/// This exercises the per-layer weight/bias/initial-state reading order
/// (`read_lstm_layer_dyn`), the Original→GateMajor transposition, and the
/// multi-channel head projection (`read_lstm_head_multi`) for `hidden_size > 1`
/// and `num_layers > 1` — none of which the H=1 / single-layer oracle covers.
#[test]
fn test_lstm_multichannel_serialized_decoder_2layer_h2_f64_oracle() {
    let num_layers = 2usize;
    let hidden = 2usize;
    let in_ch = 2usize;
    let out_ch = 2usize;

    // Deterministic logical weights W_l[k][i][j], interleaved per layer in the
    // serialized stream exactly as C++ `LSTMLayerT::SetNAMWeights` expects:
    // weights (4*H*IH) + bias (4*H) + hidden_init (H) + cell_init (H).
    let mut layers: Vec<SerializedLayer> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    for li in 0..num_layers {
        let layer_in = if li == 0 { in_ch } else { hidden };
        let ih = layer_in + hidden;
        let mut wl = vec![0.0f32; 4 * hidden * ih];
        for k in 0..4 {
            for i in 0..hidden {
                for j in 0..ih {
                    wl[k * hidden * ih + i * ih + j] = 0.05 * (1 + k) as f32 + 0.01 * i as f32
                        - 0.007 * j as f32
                        + 0.001 * li as f32;
                }
            }
        }
        let mut bl = vec![0.0f32; 4 * hidden];
        for k in 0..4 {
            for i in 0..hidden {
                bl[k * hidden + i] = 0.02 * (1 + k) as f32 - 0.01 * i as f32;
            }
        }
        let hl: Vec<f32> = (0..hidden)
            .map(|i| 0.1 * (li + 1) as f32 + 0.05 * i as f32)
            .collect();
        let cl: Vec<f32> = (0..hidden)
            .map(|i| -0.05 * (li + 1) as f32 + 0.02 * i as f32)
            .collect();

        weights.extend_from_slice(&wl);
        weights.extend_from_slice(&bl);
        weights.extend_from_slice(&hl);
        weights.extend_from_slice(&cl);
        layers.push(SerializedLayer {
            weights: wl,
            bias: bl,
            hidden_init: hl,
            cell_init: cl,
        });
    }
    let mut head_w = vec![0.0f32; out_ch * hidden];
    for o in 0..out_ch {
        for i in 0..hidden {
            head_w[o * hidden + i] = 0.2 * (o + 1) as f32 - 0.1 * i as f32;
        }
    }
    let head_b = [0.05f32, -0.07];
    weights.extend_from_slice(&head_w);
    weights.extend_from_slice(&head_b);

    assert_eq!(
        weights.len(),
        lstm_weight_count_io(num_layers, hidden, in_ch, out_ch)
    );

    let json = format!(
        r#"{{
            "version": "0.5.4",
            "architecture": "LSTM",
            "config": {{
                "num_layers": {num_layers},
                "hidden_size": {hidden},
                "in_channels": {in_ch},
                "input_size": {in_ch},
                "out_channels": {out_ch},
                "layers": []
            }},
            "weights": {weights:?}
        }}"#
    );
    let parsed = parse_nam_json(&json).expect("parse serialized LSTM JSON");
    let topo = get_lstm_topology(&parsed)
        .expect("topology query")
        .expect("LSTM topology present");
    assert_eq!(topo.num_layers, num_layers);
    assert_eq!(topo.hidden_size, hidden);
    assert_eq!(topo.in_channels, in_ch);
    assert_eq!(topo.out_channels, out_ch);

    let mut model = build_model(&parsed).expect("build serialized LSTM 2x2 H=2");
    assert_eq!(model.num_input_channels(), in_ch);
    assert_eq!(model.num_output_channels(), out_ch);

    let n_frames = 24usize;
    let mut input = vec![0.0f32; n_frames * in_ch];
    for f in 0..n_frames {
        input[f * in_ch] = (f as f32 * 0.1).sin();
        input[f * in_ch + 1] = (f as f32 * 0.15).cos();
    }
    let mut output = vec![0.0f32; n_frames * out_ch];
    model.process(&input, &mut output);

    // f64 oracle: run the two layers, then project the final hidden state.
    let mut h_state = layers
        .iter()
        .map(|l| l.hidden_init.iter().map(|&v| v as f64).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut c_state = layers
        .iter()
        .map(|l| l.cell_init.iter().map(|&v| v as f64).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    for f in 0..n_frames {
        let x: Vec<f64> = (0..in_ch).map(|c| input[f * in_ch + c] as f64).collect();
        let mut layer_input = x;
        for li in 0..num_layers {
            let wl = &layers[li].weights;
            let bl = &layers[li].bias;
            let layer_in = if li == 0 { in_ch } else { hidden };
            let ih = layer_in + hidden;
            let mut u = layer_input.clone();
            u.extend_from_slice(&h_state[li]);
            let mut new_h = vec![0.0f64; hidden];
            for i in 0..hidden {
                let gate = |k: usize| -> f64 {
                    let mut acc = bl[k * hidden + i] as f64;
                    for j in 0..ih {
                        acc += u[j] * wl[k * hidden * ih + i * ih + j] as f64;
                    }
                    acc
                };
                let gi = sigmoid_f64(gate(0));
                let gf = sigmoid_f64(gate(1));
                let gg = gate(2).tanh();
                let go = sigmoid_f64(gate(3));
                c_state[li][i] = gf * c_state[li][i] + gi * gg;
                new_h[i] = go * c_state[li][i].tanh();
            }
            h_state[li] = new_h.clone();
            layer_input = new_h;
        }
        for o in 0..out_ch {
            let mut acc = head_b[o] as f64;
            for i in 0..hidden {
                acc += head_w[o * hidden + i] as f64 * h_state[num_layers - 1][i];
            }
            let got = output[f * out_ch + o] as f64;
            assert!(
                (got - acc).abs() < 1e-3,
                "frame {f} out {o}: got {got}, f64 oracle {acc}"
            );
        }
    }
}
