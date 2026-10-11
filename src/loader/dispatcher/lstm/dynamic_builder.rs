// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::weights::{read_lstm_head_multi, read_lstm_layer_dyn};
use crate::common::diagnostics::NamErrorCode;
use crate::loader::loaded_model_pair::DEFAULT_SAMPLE_RATE;
use crate::loader::nam_json::{LstmTopology, NamModelData};
use crate::models::lstm::{LstmModelDyn, checked_lstm_weight_count_io};
use log::info;

/// Builds an `LstmModelDyn` with sequentially read weights.
///
/// This is the dynamic fallback builder for LSTM topologies that do not
/// match the 10 static const-generic profiles or utilize multi-channel I/O.
///
/// LSTM NAM Layout (C++ `LSTMLayerT::SetNAMWeights`), repeated per layer:
/// ```text
/// layer.input_hidden_weights[H4 * IH]  (row-major or Gate-Major)
/// layer.bias[H4]
/// layer.initial_hidden_state[H]
/// layer.initial_cell_state[H]
/// ...
/// head_weights[out_channels * H]
/// head_bias[out_channels]
/// ```
pub(crate) fn build_lstm_dynamic(
    data: &NamModelData,
    topo: LstmTopology,
) -> anyhow::Result<LstmModelDyn> {
    let expected_weights = checked_lstm_weight_count_io(
        topo.num_layers,
        topo.hidden_size,
        topo.input_size,
        topo.out_channels,
    )
    .ok_or(NamErrorCode::WeightCountMismatch)?;
    if data.weights.len() != expected_weights {
        anyhow::bail!(NamErrorCode::WeightCountMismatch);
    }

    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);
    let sample_rate = data.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE) as f64;

    let mut layers = Vec::with_capacity(topo.num_layers);
    for i in 0..topo.num_layers {
        let input_size = if i == 0 {
            topo.input_size
        } else {
            topo.hidden_size
        };
        let layer = read_lstm_layer_dyn(&mut cursor, input_size, topo.hidden_size)?;
        layers.push(layer);
    }

    let (head_weights_data, head_bias_data) =
        read_lstm_head_multi(&mut cursor, topo.hidden_size, topo.out_channels)?;
    let head_len = topo
        .out_channels
        .checked_mul(topo.hidden_size)
        .ok_or(NamErrorCode::WeightCountMismatch)?;
    let mut head_weights_f32 = crate::math::common::AlignedVec::new(head_len, 0.0f32)?;
    head_weights_f32.copy_from_slice(head_weights_data);

    let mut head_bias = crate::math::common::AlignedVec::new(topo.out_channels, 0.0f32)?;
    head_bias.copy_from_slice(head_bias_data);

    cursor.verify_exhausted()?;

    let model = LstmModelDyn {
        in_channels: topo.in_channels,
        out_channels: topo.out_channels,
        layers,
        head_weights_f32,
        head_bias,
        prewarm_on_reset: true,
        expected_sample_rate: sample_rate,
        prewarm_pending: 0,
    };

    info!(
        "[Dispatcher] LSTM {}×{} (in={}, out={}, dynamic) built — weights={}",
        topo.num_layers,
        topo.hidden_size,
        topo.in_channels,
        topo.out_channels,
        data.weights.len()
    );

    Ok(model)
}
