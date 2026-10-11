// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration tests for multi-channel ConvNet and grouped convolution (T4.3).
//!
//! Validates:
//! 1. Rejection of invalid feature topologies (`in_channels % groups != 0` or
//!    `channels % groups != 0`) with `NamErrorCode::InvalidFeatureTopology` (E1316).
//! 2. Depthwise weight placement verification (`groups == channels`).
//! 3. Parity between multi-channel ConvNet (2->2, channels=4, groups=2) and an f64 causal reference oracle.
//! 4. Channel geometry contract (`num_input_channels`, `num_output_channels`) and safe host guard.

use neural_amp_modeler_rs::common::diagnostics::NamErrorCode;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{
    TopologyError, get_convnet_topology, parse_nam_json,
};
use neural_amp_modeler_rs::models::NamModel;

#[test]
fn test_convnet_groups_topology_rejection_e1316() {
    // 1. in_channels % groups != 0 (in_channels=3, channels=4, groups=2)
    let json_in_mismatch = r#"{
        "version": "0.5.0",
        "architecture": "ConvNet",
        "config": {
            "in_channels": 3,
            "out_channels": 1,
            "channels": 4,
            "groups": 2,
            "kernel_size": 2,
            "dilations": [1],
            "batchnorm": false,
            "activation": "ReLU"
        },
        "weights": [],
        "sample_rate": 48000.0
    }"#;

    let parsed = parse_nam_json(json_in_mismatch).expect("JSON syntax is valid");
    let topo_res = get_convnet_topology(&parsed);
    assert!(
        matches!(
            topo_res,
            Err(TopologyError {
                code: NamErrorCode::InvalidFeatureTopology,
                ..
            })
        ),
        "in_channels % groups != 0 must be rejected with E1316, got: {topo_res:?}"
    );
    let build_res = build_model(&parsed);
    assert!(
        build_res.is_err(),
        "build_model must fail when topology validation fails"
    );

    // 2. channels % groups != 0 (in_channels=4, channels=5, groups=2)
    let json_ch_mismatch = r#"{
        "version": "0.5.0",
        "architecture": "ConvNet",
        "config": {
            "in_channels": 4,
            "out_channels": 1,
            "channels": 5,
            "groups": 2,
            "kernel_size": 2,
            "dilations": [1],
            "batchnorm": false,
            "activation": "ReLU"
        },
        "weights": [],
        "sample_rate": 48000.0
    }"#;

    let parsed_ch = parse_nam_json(json_ch_mismatch).expect("JSON syntax is valid");
    let topo_res2 = get_convnet_topology(&parsed_ch);
    assert!(
        matches!(
            topo_res2,
            Err(TopologyError {
                code: NamErrorCode::InvalidFeatureTopology,
                ..
            })
        ),
        "channels % groups != 0 must be rejected with E1316, got: {topo_res2:?}"
    );
    let build_res2 = build_model(&parsed_ch);
    assert!(
        build_res2.is_err(),
        "build_model must fail when topology validation fails"
    );
}

#[test]
fn test_convnet_depthwise_weight_placement() {
    // 4->4 depthwise conv: in_channels=4, channels=4, groups=4, kernel=2.
    // Each group g has in_per_group=1, out_per_group=1.
    // Group g connects channel g to channel g.
    // Conv weights: 4 groups * 1 opg * 1 ipg * 2 kernel = 8 floats.
    // Conv bias: 4 floats (all zero).
    // Linear head: in_channels=4, out_channels=4 -> 16 floats (diagonal identity).
    // Head bias: 4 floats (all zero).
    let mut weights = Vec::new();
    // Block 0 conv weights:
    // for g in 0..4: for k in 0..2:
    for g in 0..4 {
        for k in 0..2 {
            weights.push((g + 1) as f32 * 10.0 + k as f32);
        }
    }
    // Block 0 conv bias: 4 floats
    weights.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);

    // Head weights: 4x4 identity matrix (row-major: out_c * 4 + in_c)
    for out_c in 0..4 {
        for in_c in 0..4 {
            weights.push(if out_c == in_c { 1.0 } else { 0.0 });
        }
    }
    // Head bias: 4 floats
    weights.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);

    let weights_json = serde_json::to_string(&weights).unwrap();
    let json = format!(
        r#"{{
        "version": "0.5.0",
        "architecture": "ConvNet",
        "config": {{
            "in_channels": 4,
            "out_channels": 4,
            "channels": 4,
            "groups": 4,
            "kernel_size": 2,
            "dilations": [1],
            "batchnorm": false,
            "activation": "ReLU"
        }},
        "weights": {weights_json},
        "sample_rate": 48000.0
    }}"#
    );

    let parsed = parse_nam_json(&json).expect("valid JSON");
    let mut model = build_model(&parsed).expect("valid depthwise model builds successfully");

    // Feed an impulse on channel 1 only (in_channels = [0.0, 1.0, 0.0, 0.0] at t=0, then zero)
    // In causal Conv1dDyn with kernel=2:
    // tap 0 (k=0) is lookback history (t-1). Weight for g=1 is (1+1)*10.0 + 0 = 20.0.
    // tap 1 (k=1) is current sample (t). Weight for g=1 is (1+1)*10.0 + 1 = 21.0.
    // All other channels (0, 2, 3) must be 0.0!
    let num_frames = 4;
    let mut in_planes = vec![vec![0.0f32; num_frames]; 4];
    in_planes[1][0] = 1.0; // impulse at t=0 on ch 1
    let mut out_planes = vec![vec![0.0f32; num_frames]; 4];

    let in_slices: Vec<&[f32]> = in_planes.iter().map(|p| p.as_slice()).collect();
    let mut out_slices: Vec<&mut [f32]> = out_planes.iter_mut().map(|p| p.as_mut_slice()).collect();

    let processed = model
        .process_multichannel(&in_slices, &mut out_slices)
        .expect("multichannel process succeeds");
    assert_eq!(processed, num_frames);

    // Verify channel 1 received the signal and other channels remained strictly zero
    assert_eq!(
        out_planes[0][0], 0.0,
        "channel 0 should receive no crosstalk from channel 1"
    );
    assert!(
        (out_planes[1][0] - 21.0).abs() < 1e-6,
        "channel 1 at t=0 should match current tap weight (21.0), got {}",
        out_planes[1][0]
    );
    assert_eq!(
        out_planes[2][0], 0.0,
        "channel 2 should receive no crosstalk from channel 1"
    );
    assert_eq!(
        out_planes[3][0], 0.0,
        "channel 3 should receive no crosstalk from channel 1"
    );

    // At t=1 (lookback tap 0 is active on previous impulse):
    assert_eq!(out_planes[0][1], 0.0);
    assert!(
        (out_planes[1][1] - 20.0).abs() < 1e-6,
        "channel 1 at t=1 should match lookback tap weight (20.0), got {}",
        out_planes[1][1]
    );
    assert_eq!(out_planes[2][1], 0.0);
    assert_eq!(out_planes[3][1], 0.0);
}

#[test]
fn test_convnet_depthwise_all_channels_stimulus() {
    // Same 4->4 depthwise topology as `test_convnet_depthwise_weight_placement`,
    // but exciting **every** channel with distinct signals. Each output channel g
    // must reproduce the causal FIR of its own input channel only (diagonal),
    // proving the depthwise grouping has no cross-channel leakage.
    let mut weights = Vec::new();
    for g in 0..4 {
        for k in 0..2 {
            weights.push((g + 1) as f32 * 10.0 + k as f32);
        }
    }
    weights.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]); // conv bias
    for out_c in 0..4 {
        for in_c in 0..4 {
            weights.push(if out_c == in_c { 1.0 } else { 0.0 });
        }
    }
    weights.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]); // head bias

    let weights_json = serde_json::to_string(&weights).unwrap();
    let json = format!(
        r#"{{
        "version": "0.5.0",
        "architecture": "ConvNet",
        "config": {{
            "in_channels": 4,
            "out_channels": 4,
            "channels": 4,
            "groups": 4,
            "kernel_size": 2,
            "dilations": [1],
            "batchnorm": false,
            "activation": "ReLU"
        }},
        "weights": {weights_json},
        "sample_rate": 48000.0
    }}"#
    );

    let parsed = parse_nam_json(&json).expect("valid JSON");
    let mut model = build_model(&parsed).expect("depthwise model builds");

    let num_frames = 12;
    // Distinct per-channel excitations (some negative, some large) so a leak
    // into any other channel would change its output.
    let mut in_planes = vec![vec![0.0f32; num_frames]; 4];
    for (g, plane) in in_planes.iter_mut().enumerate() {
        for (t, v) in plane.iter_mut().enumerate() {
            *v = (g as f32 + 1.0) * 0.5 + 0.1 * t as f32 + if g % 2 == 1 { -0.3 } else { 0.0 };
        }
    }
    let mut out_planes = vec![vec![0.0f32; num_frames]; 4];
    let in_slices: Vec<&[f32]> = in_planes.iter().map(|p| p.as_slice()).collect();
    let mut out_slices: Vec<&mut [f32]> = out_planes.iter_mut().map(|p| p.as_mut_slice()).collect();
    let processed = model
        .process_multichannel(&in_slices, &mut out_slices)
        .expect("multichannel depthwise process succeeds");
    assert_eq!(processed, num_frames);

    for g in 0..4 {
        let w0 = (g + 1) as f64 * 10.0; // lookback tap (t-1)
        let w1 = (g + 1) as f64 * 10.0 + 1.0; // current tap (t)
        for t in 0..num_frames {
            let lookback = if t == 0 {
                0.0
            } else {
                in_planes[g][t - 1] as f64
            };
            let current = in_planes[g][t] as f64;
            let expected = (w0 * lookback + w1 * current).max(0.0);
            let got = out_planes[g][t] as f64;
            assert!(
                (got - expected).abs() < 1e-4,
                "channel {g} frame {t}: got {got}, depthwise oracle {expected}"
            );
        }
    }
}

#[test]
fn test_convnet_multichannel_vs_f64_oracle() {
    // ConvNet 2 -> 2:
    // in_channels = 2, out_channels = 2, channels = 4, groups = 2, kernel_size = 2.
    // dilations = [1, 1] (2 blocks)
    // activation = "ReLU", batchnorm = false.
    //
    // Block 0: in_ch = 2, out_ch = 4, groups = 2 (ipg = 1, opg = 2).
    //   Weights count: 4 * 1 * 2 = 8.
    //   Group 0: in_ch 0 -> out_ch 0, 1.
    //   Group 1: in_ch 1 -> out_ch 2, 3.
    //
    // Block 1: in_ch = 4, out_ch = 4, groups = 2 (ipg = 2, opg = 2).
    //   Weights count: 4 * 2 * 2 = 16.
    //   Group 0: in_ch 0, 1 -> out_ch 0, 1.
    //   Group 1: in_ch 2, 3 -> out_ch 2, 3.
    //
    // Head (Linear): in_ch = 4, out_ch = 2.
    //   Weights count: 2 * 4 = 8.
    //   Bias count: 2.
    // Total weights: 8 + 4 (b0 bias) + 16 + 4 (b1 bias) + 8 + 2 = 42 floats.

    let b0_weights: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let b0_bias: Vec<f32> = vec![0.02, -0.01, 0.03, -0.02];
    let b1_weights: Vec<f32> = (0..16).map(|i| ((i as f32 + 1.0) * 0.05) - 0.2).collect();
    let b1_bias: Vec<f32> = vec![-0.01, 0.02, -0.02, 0.01];
    let head_weights: Vec<f32> = vec![
        0.5, -0.3, 0.2, 0.1, // out_c 0
        -0.2, 0.4, 0.1, -0.5, // out_c 1
    ];
    let head_bias: Vec<f32> = vec![0.05, -0.05];

    let mut all_weights = Vec::new();
    all_weights.extend_from_slice(&b0_weights);
    all_weights.extend_from_slice(&b0_bias);
    all_weights.extend_from_slice(&b1_weights);
    all_weights.extend_from_slice(&b1_bias);
    all_weights.extend_from_slice(&head_weights);
    all_weights.extend_from_slice(&head_bias);

    let weights_json = serde_json::to_string(&all_weights).unwrap();
    let json = format!(
        r#"{{
        "version": "0.5.0",
        "architecture": "ConvNet",
        "config": {{
            "in_channels": 2,
            "out_channels": 2,
            "channels": 4,
            "groups": 2,
            "kernel_size": 2,
            "dilations": [1, 1],
            "batchnorm": false,
            "activation": "ReLU"
        }},
        "weights": {weights_json},
        "sample_rate": 48000.0
    }}"#
    );

    let parsed = parse_nam_json(&json).expect("valid JSON");
    let mut model = build_model(&parsed).expect("ConvNet 2->2 model builds");

    // Test channel geometry
    assert_eq!(model.num_input_channels(), 2);
    assert_eq!(model.num_output_channels(), 2);

    let num_frames = 64;
    // Generate deterministic test signals for input channels 0 and 1
    let mut input_c0 = vec![0.0f32; num_frames];
    let mut input_c1 = vec![0.0f32; num_frames];
    for t in 0..num_frames {
        input_c0[t] = ((t as f32) * 0.1).sin();
        input_c1[t] = ((t as f32) * 0.15).cos();
    }

    // ── f64 Reference Oracle ──
    let mut oracle_out_c0 = vec![0.0f64; num_frames];
    let mut oracle_out_c1 = vec![0.0f64; num_frames];

    // State delay lines for block 0 (receptive field 1) and block 1 (receptive field 1)
    let mut b0_state = [0.0f64; 2]; // 2 input channels
    let mut b1_state = [0.0f64; 4]; // 4 channels

    for t in 0..num_frames {
        let x0 = input_c0[t] as f64;
        let x1 = input_c1[t] as f64;

        // Block 0:
        // Group 0: in_c 0 -> out_c 0, 1.
        //   k=0 is lookback (b0_state[0]), k=1 is current (x0)
        //   out_c 0: tap0 = b0_weights[0], tap1 = b0_weights[1]
        //   out_c 1: tap0 = b0_weights[2], tap1 = b0_weights[3]
        // Group 1: in_c 1 -> out_c 2, 3.
        //   k=0 is lookback (b0_state[1]), k=1 is current (x1)
        //   out_c 2: tap0 = b0_weights[4], tap1 = b0_weights[5]
        //   out_c 3: tap0 = b0_weights[6], tap1 = b0_weights[7]
        let mut b0_out = [0.0f64; 4];
        b0_out[0] = (b0_bias[0] as f64)
            + (b0_weights[0] as f64) * b0_state[0]
            + (b0_weights[1] as f64) * x0;
        b0_out[1] = (b0_bias[1] as f64)
            + (b0_weights[2] as f64) * b0_state[0]
            + (b0_weights[3] as f64) * x0;
        b0_out[2] = (b0_bias[2] as f64)
            + (b0_weights[4] as f64) * b0_state[1]
            + (b0_weights[5] as f64) * x1;
        b0_out[3] = (b0_bias[3] as f64)
            + (b0_weights[6] as f64) * b0_state[1]
            + (b0_weights[7] as f64) * x1;

        // Update block 0 state
        b0_state[0] = x0;
        b0_state[1] = x1;

        // Activation: ReLU
        for v in &mut b0_out {
            *v = v.max(0.0);
        }

        // Block 1:
        // Group 0: in_c 0, 1 -> out_c 0, 1. (ipg=2, opg=2, kernel=2)
        // C++ order: for oi in 0..2: for ii in 0..2: for k in 0..2:
        //   oi=0, ii=0: k=0 (idx 0), k=1 (idx 1)
        //   oi=0, ii=1: k=0 (idx 2), k=1 (idx 3)
        //   oi=1, ii=0: k=0 (idx 4), k=1 (idx 5)
        //   oi=1, ii=1: k=0 (idx 6), k=1 (idx 7)
        // Group 1: in_c 2, 3 -> out_c 2, 3. (offset +8)
        let mut b1_out = [0.0f64; 4];
        // oi=0 (out_c 0):
        b1_out[0] = (b1_bias[0] as f64)
            + (b1_weights[0] as f64) * b1_state[0]
            + (b1_weights[1] as f64) * b0_out[0]
            + (b1_weights[2] as f64) * b1_state[1]
            + (b1_weights[3] as f64) * b0_out[1];
        // oi=1 (out_c 1):
        b1_out[1] = (b1_bias[1] as f64)
            + (b1_weights[4] as f64) * b1_state[0]
            + (b1_weights[5] as f64) * b0_out[0]
            + (b1_weights[6] as f64) * b1_state[1]
            + (b1_weights[7] as f64) * b0_out[1];
        // Group 1, oi=0 (out_c 2):
        b1_out[2] = (b1_bias[2] as f64)
            + (b1_weights[8] as f64) * b1_state[2]
            + (b1_weights[9] as f64) * b0_out[2]
            + (b1_weights[10] as f64) * b1_state[3]
            + (b1_weights[11] as f64) * b0_out[3];
        // Group 1, oi=1 (out_c 3):
        b1_out[3] = (b1_bias[3] as f64)
            + (b1_weights[12] as f64) * b1_state[2]
            + (b1_weights[13] as f64) * b0_out[2]
            + (b1_weights[14] as f64) * b1_state[3]
            + (b1_weights[15] as f64) * b0_out[3];

        // Update block 1 state
        b1_state.copy_from_slice(&b0_out);

        // Activation: ReLU
        for v in &mut b1_out {
            *v = v.max(0.0);
        }

        // Linear Head:
        // out_c 0:
        oracle_out_c0[t] = (head_bias[0] as f64)
            + (head_weights[0] as f64) * b1_out[0]
            + (head_weights[1] as f64) * b1_out[1]
            + (head_weights[2] as f64) * b1_out[2]
            + (head_weights[3] as f64) * b1_out[3];
        // out_c 1:
        oracle_out_c1[t] = (head_bias[1] as f64)
            + (head_weights[4] as f64) * b1_out[0]
            + (head_weights[5] as f64) * b1_out[1]
            + (head_weights[6] as f64) * b1_out[2]
            + (head_weights[7] as f64) * b1_out[3];
    }

    // Process via model.process_multichannel:
    let mut model_out_c0 = vec![0.0f32; num_frames];
    let mut model_out_c1 = vec![0.0f32; num_frames];
    {
        let in_slices = [&input_c0[..], &input_c1[..]];
        let mut out_slices = [&mut model_out_c0[..], &mut model_out_c1[..]];
        model
            .process_multichannel(&in_slices, &mut out_slices)
            .expect("process_multichannel succeeds");
    }

    // Compare with f64 oracle across both planes:
    let mut max_err_c0 = 0.0f64;
    let mut max_err_c1 = 0.0f64;
    for t in 0..num_frames {
        let err0 = ((model_out_c0[t] as f64) - oracle_out_c0[t]).abs();
        let err1 = ((model_out_c1[t] as f64) - oracle_out_c1[t]).abs();
        if err0 > max_err_c0 {
            max_err_c0 = err0;
        }
        if err1 > max_err_c1 {
            max_err_c1 = err1;
        }
    }

    // Measured: max_err_c0 < 5e-7, max_err_c1 < 5e-7
    assert!(
        max_err_c0 < 1e-5,
        "channel 0 max error exceeds threshold: {max_err_c0}"
    );
    assert!(
        max_err_c1 < 1e-5,
        "channel 1 max error exceeds threshold: {max_err_c1}"
    );

    // Also verify process_raw gives identical results to process_multichannel
    model.clear_state();
    let mut raw_out_c0 = vec![0.0f32; num_frames];
    let mut raw_out_c1 = vec![0.0f32; num_frames];
    let raw_inputs = [input_c0.as_ptr(), input_c1.as_ptr()];
    let mut raw_outputs = [raw_out_c0.as_mut_ptr(), raw_out_c1.as_mut_ptr()];
    unsafe {
        model.process_raw(raw_inputs.as_ptr(), raw_outputs.as_mut_ptr(), num_frames);
    }
    for t in 0..num_frames {
        assert_eq!(
            raw_out_c0[t], model_out_c0[t],
            "raw output ch0 matches staged multichannel"
        );
        assert_eq!(
            raw_out_c1[t], model_out_c1[t],
            "raw output ch1 matches staged multichannel"
        );
    }

    // Verify under-sized plane sets reject with HostFormatContractViolation (E2304)
    let in_insufficient = [&input_c0[..]]; // only 1 plane provided, requires 2
    let mut out_slices = [&mut model_out_c0[..], &mut model_out_c1[..]];
    let err = model.process_multichannel(&in_insufficient, &mut out_slices);
    assert_eq!(
        err,
        Err(NamErrorCode::HostFormatContractViolation),
        "insufficient input planes must fail with E2304"
    );
}
