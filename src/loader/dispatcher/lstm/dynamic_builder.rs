// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::weights::read_lstm_layer_dyn;
use crate::loader::loaded_model_pair::DEFAULT_SAMPLE_RATE;
use crate::loader::nam_json::NamModelData;
use crate::models::lstm::LstmModelDyn;
use log::info;

/// Builds an `LstmModelDyn` with sequentially read weights.
///
/// This is the dynamic fallback builder for LSTM topologies that do not
/// match the 10 static const-generic profiles.
///
/// LSTM NAM Layout (C++ `LSTMLayerT::SetNAMWeights`), repeated per layer:
/// ```text
/// layer.input_hidden_weights[H4 * IH]  (row-major or Gate-Major)
/// layer.bias[H4]
/// layer.initial_hidden_state[H]
/// layer.initial_cell_state[H]
/// ...
/// head_weights[H]
/// head_bias
/// ```
pub(crate) fn build_lstm_dynamic(
    data: &NamModelData,
    num_layers: usize,
    hidden_size: usize,
) -> anyhow::Result<LstmModelDyn> {
    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);
    let sample_rate = data.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE) as f64;

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let input_size = if i == 0 { 1 } else { hidden_size };
        let layer = read_lstm_layer_dyn(&mut cursor, input_size, hidden_size)?;
        layers.push(layer);
    }

    let h = hidden_size;
    let head_weights_data = cursor.read_slice(h)?;
    let mut head_weights_f32 = crate::math::common::AlignedVec::new(h, 0.0f32)?;
    head_weights_f32.copy_from_slice(head_weights_data);
    let head_bias = cursor.read_f32_finite()?;

    cursor.verify_exhausted()?;

    let model = LstmModelDyn {
        layers,
        head_weights_f32,
        head_bias,
        prewarm_on_reset: true,
        expected_sample_rate: sample_rate,
    };

    info!(
        "[Dispatcher] LSTM {num_layers}×{hidden_size} (dynamic) built — weights={}",
        data.weights.len()
    );

    Ok(model)
}
