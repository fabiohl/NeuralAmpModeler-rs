// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::weights::read_lstm_layer;
use crate::loader::loaded_model_pair::DEFAULT_SAMPLE_RATE;
use crate::loader::nam_json::NamModelData;
use crate::models::lstm::{LstmModel1, LstmModel2};
use log::info;

/// Builds an `LstmModel1<H, H1_IH, H_H4>` with sequentially read weights.
///
/// LSTM NAM Layout (C++ `LSTMLayerT::SetNAMWeights`):
/// ```text
/// layer.input_hidden_weights[H4 * IH]  (row-major)
/// layer.bias[H4]
/// layer.initial_hidden_state[H]
/// layer.initial_cell_state[H]
/// head_weights[H]
/// head_bias
/// ```
pub(crate) fn build_lstm_1layer<const H: usize, const H1_IH: usize, const H_H4: usize>(
    data: &NamModelData,
    hidden_size: usize,
) -> anyhow::Result<LstmModel1<H, H1_IH, H_H4>> {
    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);
    let sample_rate = data.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE) as f64;

    // Layer 1: input_size=1
    let layer = read_lstm_layer::<1, H, H1_IH, H_H4>(&mut cursor)?;

    // Head: output linear projection weights
    let head_weights_data = cursor.read_slice(H)?;
    let mut head_weights_f32 = [0.0f32; H];
    head_weights_f32.copy_from_slice(head_weights_data);
    let head_bias = cursor.read_f32_finite()?;

    cursor.verify_exhausted()?;

    let model = LstmModel1::<H, H1_IH, H_H4> {
        layer,
        head_weights_f32,
        head_bias,
        prewarm_on_reset: true,
        expected_sample_rate: sample_rate,
    };

    info!(
        "[Dispatcher] LSTM 1×{} built — weights={}",
        hidden_size,
        data.weights.len()
    );

    Ok(model)
}

/// Builds an `LstmModel2<H, H1_IH, H2_IH, H_H4>` with sequentially read weights.
pub(crate) fn build_lstm_2layer<
    const H: usize,
    const H1_IH: usize,
    const H2_IH: usize,
    const H_H4: usize,
>(
    data: &NamModelData,
    num_layers: usize,
    hidden_size: usize,
) -> anyhow::Result<LstmModel2<H, H1_IH, H2_IH, H_H4>> {
    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);
    let sample_rate = data.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE) as f64;

    // Layer 1: input_size=1
    let layer1 = read_lstm_layer::<1, H, H1_IH, H_H4>(&mut cursor)?;

    // Layer 2: input_size=H (hidden state from previous layer)
    let layer2 = read_lstm_layer::<H, H, H2_IH, H_H4>(&mut cursor)?;

    // Head: final projection weights
    let head_weights_data = cursor.read_slice(H)?;
    let mut head_weights_f32 = [0.0f32; H];
    head_weights_f32.copy_from_slice(head_weights_data);
    let head_bias = cursor.read_f32_finite()?;

    cursor.verify_exhausted()?;

    let model = LstmModel2::<H, H1_IH, H2_IH, H_H4> {
        layer1,
        layer2,
        head_weights_f32,
        head_bias,
        prewarm_on_reset: true,
        expected_sample_rate: sample_rate,
    };

    info!(
        "[Dispatcher] LSTM {}×{} built — weights={}",
        num_layers,
        hidden_size,
        data.weights.len()
    );

    Ok(model)
}
