// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of LSTM topologies from model data.

use super::super::data::{JsonError, NamModelData};

/// Checks and returns the LSTM geometry (num_layers, hidden_size).
///
/// Returns `Ok(Some(...))` for valid LSTM topologies that can be dispatched.
/// Returns `Ok(None)` when the model is not LSTM or when required config fields
/// (`num_layers`, `hidden_size`) are missing.
/// Returns `Err` for invalid structural parameters:
/// - `num_layers == 0`
/// - `num_layers > MAX_LSTM_LAYERS` (16)
/// - `hidden_size > MAX_LSTM_HIDDEN_SIZE` (1024)
/// - Multi-channel I/O (`in_channels != 1` or `out_channels != 1`)
pub fn get_lstm_topology(data: &NamModelData) -> Result<Option<(usize, usize)>, JsonError> {
    use super::super::validation::{MAX_LSTM_HIDDEN_SIZE, MAX_LSTM_LAYERS};

    if data.architecture != "LSTM" {
        return Ok(None);
    }

    let num_layers = data.config.num_layers;
    let hidden_size = data.config.hidden_size;

    if let Some(c) = data.config.in_channels
        && c != 1
    {
        return Err(JsonError::UnsupportedMultiChannel {
            architecture: data.architecture.clone(),
            field: "in_channels",
            value: c,
        });
    }
    if let Some(c) = data.config.out_channels
        && c != 1
    {
        return Err(JsonError::UnsupportedMultiChannel {
            architecture: data.architecture.clone(),
            field: "out_channels",
            value: c,
        });
    }

    let num_layers = match num_layers {
        Some(n) => n,
        None => return Ok(None),
    };
    let hidden_size = match hidden_size {
        Some(h) => h,
        None => return Ok(None),
    };

    if num_layers == 0 {
        return Err(JsonError::UnsupportedTopology {
            architecture: data.architecture.clone(),
            issue: "num_layers=0 (no valid model can have zero layers)".into(),
            limit: 0,
        });
    }
    if num_layers > MAX_LSTM_LAYERS {
        return Err(JsonError::UnsupportedTopology {
            architecture: data.architecture.clone(),
            issue: format!("num_layers={num_layers} exceeds maximum {MAX_LSTM_LAYERS}"),
            limit: MAX_LSTM_LAYERS,
        });
    }
    if hidden_size > MAX_LSTM_HIDDEN_SIZE {
        return Err(JsonError::UnsupportedTopology {
            architecture: data.architecture.clone(),
            issue: format!("hidden_size={hidden_size} exceeds maximum {MAX_LSTM_HIDDEN_SIZE}"),
            limit: MAX_LSTM_HIDDEN_SIZE,
        });
    }
    Ok(Some((num_layers, hidden_size)))
}
