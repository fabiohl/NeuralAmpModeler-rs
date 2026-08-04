// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! f64 reference oracle for LSTM neural models.
//!
//! Provides a pure-f64, non-SIMD implementation of the LSTM forward pass
//! used as the ground-truth reference for regression testing. The oracle
//! supports both the original weight layout (`[gate][H][IH]`) and the
//! `GateMajorLstm` transposed layout (`[gate][IH][H]`).
//!
//! The forward pass follows the standard LSTM recurrence:
//! 1. Build input-hidden state vector `[input; hidden]` per layer.
//! 2. Four GEMV operations compute the gate pre-activations.
//! 3. Fused gate application: forget, input, cell, and output gates.
//! 4. Hidden state is projected through a linear head to produce
//!    the final output sample.

#![allow(missing_docs)]

use crate::loader::nam_json::WeightsLayout;
use crate::loader::nam_json::model::NamModelData;

use super::*;

/// Runs the LSTM model forward pass in f64 precision, processing the
/// entire `input` buffer frame-by-frame. Returns the output audio vector.
pub(crate) fn oracle_lstm_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    let h = model_data.config.hidden_size.unwrap_or(16);
    let nlayers = model_data.config.num_layers.unwrap_or(1);
    let mut cursor = Cursor::new(&model_data.weights, config.weight_precision);

    struct LstmLW {
        ih_w: Vec<Vec<Vec<f64>>>, // [gate=4][row=ih][col=h]
        bias: Vec<f64>,
        hidden: Vec<f64>,
        cell: Vec<f64>,
        in_size: usize,
    }

    // Flattened weights: depends on weights_layout
    // Original: [gate][H][IH] = row-major within each gate
    // GateMajor: [gate][IH][H] = row-major within each gate
    let is_gate_major = model_data.weights_layout == WeightsLayout::GateMajorLstm;
    let mut ll: Vec<LstmLW> = Vec::new();
    for l in 0..nlayers {
        let ins = if l == 0 { 1 } else { h };
        let ih = ins + h;
        let raw = cursor.read_f64(4 * ih * h);
        let bias = cursor.read_f64(4 * h);
        let hidden = cursor.read_f64(h);
        let cell = cursor.read_f64(h);

        // Build weight matrix [gate][row][col]
        let mut wh = vec![vec![vec![0.0f64; h]; ih]; 4];
        for g in 0..4 {
            for r in 0..ih {
                for c in 0..h {
                    wh[g][r][c] = if is_gate_major {
                        // GateMajor: raw[gate][row=IH][col=H]
                        raw[g * ih * h + r * h + c]
                    } else {
                        // Original: raw[gate][col=H][row=IH]
                        raw[g * h * ih + c * ih + r]
                    };
                }
            }
        }

        ll.push(LstmLW {
            ih_w: wh,
            bias,
            hidden,
            cell,
            in_size: ins,
        });
    }

    let head_w = cursor.read_f64(h);
    let head_b = cursor.read_one_f64();

    let num_frames = input.len();
    let mut output = vec![0.0f64; num_frames];
    let acc_mode = config.accumulation;

    // Clone cell states for independent runs (decomposition needs fresh state)
    let _orig_cell: Vec<Vec<f64>> = ll.iter().map(|l| l.cell.clone()).collect();
    let _orig_hidden: Vec<Vec<f64>> = ll.iter().map(|l| l.hidden.clone()).collect();

    #[expect(
        clippy::needless_range_loop,
        reason = "Range loop required for explicit SIMD lane indexing not expressible via iterator"
    )]
    for f in 0..num_frames {
        let x = input[f];

        // Process layers sequentially: each layer's output hidden state
        // becomes the next layer's input. LSTM is recurrent — state
        // carries forward across frames, not reset.
        for l in 0..nlayers {
            let ins = ll[l].in_size;
            let ih = ins + h;

            // Build state: [input_part; hidden]
            let mut state = vec![0.0f64; ih];
            if l == 0 {
                state[0] = x;
            } else {
                state[..ins].copy_from_slice(&ll[l - 1].hidden[..ins]);
            }
            state[ins..ins + h].copy_from_slice(&ll[l].hidden[..h]);

            // GEMV: compute gate pre-activations for all 4 gates.
            // gates[g*h + i] = bias[g*h + i] + Σⱼ state[j] · wh[g][j][i]
            let mut gates = vec![0.0f64; 4 * h];
            for g in 0..4 {
                for i in 0..h {
                    let mut sum = ll[l].bias[g * h + i];
                    for j in 0..ih {
                        sum = mul_add_f64(state[j], ll[l].ih_w[g][j][i], sum, acc_mode);
                    }
                    gates[g * h + i] = sum;
                }
            }

            // Fused gates
            for i in 0..h {
                let gi = gates[i];
                let gf = gates[h + i];
                let gg = gates[2 * h + i];
                let go = gates[3 * h + i];

                let fg = oracle_sigmoid(gf, config.activation);
                let ig = oracle_sigmoid(gi, config.activation);
                let gv = oracle_tanh(gg, config.activation);
                let og = oracle_sigmoid(go, config.activation);

                let nc = fg * ll[l].cell[i] + ig * gv;
                let hv = og * oracle_tanh(nc, config.activation);

                ll[l].cell[i] = nc;
                ll[l].hidden[i] = hv;
            }
        }

        // Linear head projection: out = head_bias + Σᵢ last_hidden[i] · head_w[i]
        let last_h = &ll.last().unwrap().hidden;
        let mut y = head_b;
        for i in 0..h {
            y = mul_add_f64(last_h[i], head_w[i], y, acc_mode);
        }
        output[f] = y;
    }

    output
}

// =============================================================================
// ConvNet Oracle
// =============================================================================
