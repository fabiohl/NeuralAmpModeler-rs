// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use crate::loader::dispatcher::checked_arith;
use crate::models::lstm::{LstmLayer, LstmLayerDyn};

/// Fills `output` (buffer of `f32` with size `4 * hidden * (input + hidden)`)
/// with the LSTM weights read from `raw` (`f32` slice already read from the cursor).
///
/// Applies layout transposition when needed (Original → GateMajor).
pub(crate) fn read_lstm_weights_into(
    raw: &[f32],
    output: &mut [f32],
    is_gate_major: bool,
    hidden: usize,
    input: usize,
) {
    let ih = input + hidden;
    debug_assert!(
        raw.len() >= 4 * hidden * ih,
        "raw slice too short for LSTM weights"
    );
    if is_gate_major {
        output.copy_from_slice(&raw[..output.len()]);
    } else {
        for k in 0..4 {
            let gate_offset = k * hidden * ih;
            for i in 0..hidden {
                for j in 0..ih {
                    let v = raw[k * ih * hidden + i * ih + j];
                    output[gate_offset + j * hidden + i] = v;
                }
            }
        }
    }
}

/// Reads the weights of an `LstmLayer<I, H, IH, H4>`.
///
/// Layout NAM JSON (C++ `LSTMLayerT::SetNAMWeights`):
/// ```text
/// input_hidden_weights: [H4 rows × IH cols] — row-major, direct mapping
/// bias:                 [H4]
/// initial_hidden:       [H]  → state[I..I+H]
/// initial_cell_state:   [H]  → cell_state[0..H]
/// ```
pub(crate) fn read_lstm_layer<const I: usize, const H: usize, const IH: usize, const H4: usize>(
    cursor: &mut WeightCursor<'_>,
) -> anyhow::Result<LstmLayer<I, H, IH, H4>> {
    let mut layer = LstmLayer::<I, H, IH, H4>::new();

    assert_eq!(H4, 4 * H, "H4 must equal 4*H for LSTM layer");
    assert_eq!(IH, I + H, "IH must equal I+H for LSTM layer");

    let raw_weights = cursor.read_slice(H4 * IH)?;
    let is_gate_major = cursor.is_gate_major_lstm();

    let flat = layer.input_hidden_weights.as_flattened_mut();
    let ih = I + H;
    if is_gate_major {
        for (row, chunk) in flat.iter_mut().zip(raw_weights.chunks_exact(H)) {
            row.copy_from_slice(chunk);
        }
    } else {
        for k in 0..4 {
            for i in 0..H {
                for j in 0..ih {
                    flat[k * ih + j][i] = raw_weights[k * ih * H + i * ih + j];
                }
            }
        }
    }

    let bias_data = cursor.read_slice(H4)?;
    layer.bias.copy_from_slice(bias_data);

    let hidden_init = cursor.read_slice(H)?;
    layer.state[I..I + H].copy_from_slice(hidden_init);

    let cell_init = cursor.read_slice(H)?;
    layer.cell_state.copy_from_slice(cell_init);

    Ok(layer)
}

/// Reads the weights of an `LstmLayerDyn` (runtime-sized LSTM layer).
///
/// This is the dynamic counterpart to [`read_lstm_layer`], allocating
/// [`AlignedVec`] buffers instead of const-generic stack arrays.
///
/// Layout NAM JSON (C++ `LSTMLayerT::SetNAMWeights`):
/// ```text
/// input_hidden_weights: [H4 × IH] — row-major or Gate-Major
/// bias:                 [H4]
/// initial_hidden:       [H]  → state[input_size..]
/// initial_cell_state:   [H]  → cell_state[0..H]
/// ```
pub(crate) fn read_lstm_layer_dyn(
    cursor: &mut WeightCursor<'_>,
    input_size: usize,
    hidden_size: usize,
) -> anyhow::Result<LstmLayerDyn> {
    let ih = checked_arith::checked_add(input_size, hidden_size)?;
    let h4 = checked_arith::checked_mul(4, hidden_size)?;
    let weights_len = checked_arith::checked_mul(h4, ih)?;

    let mut layer = LstmLayerDyn::new(input_size, hidden_size)?;

    let raw_weights = cursor.read_slice(weights_len)?;
    let is_gate_major = cursor.is_gate_major_lstm();

    read_lstm_weights_into(
        raw_weights,
        &mut layer.input_hidden_weights,
        is_gate_major,
        hidden_size,
        input_size,
    );

    let bias_data = cursor.read_slice(h4)?;
    layer.bias.copy_from_slice(bias_data);

    let hidden_init = cursor.read_slice(hidden_size)?;
    layer.state[input_size..].copy_from_slice(hidden_init);

    let cell_init = cursor.read_slice(hidden_size)?;
    layer.cell_state.copy_from_slice(cell_init);

    Ok(layer)
}
