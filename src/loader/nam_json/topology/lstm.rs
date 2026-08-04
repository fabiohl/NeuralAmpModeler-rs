// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of LSTM topologies from model data.

use super::super::data::{JsonError, NamModelData};

/// Checks and returns the LSTM geometry (num_layers, hidden_size).
///
/// Returns `Ok(Some(...))` for valid LSTM topologies that can be dispatched.
/// Returns `Ok(None)` when the model is not LSTM or has unsupported structural
/// parameters (num_layers=0, bounds exceeded — these don't carry a dedicated
/// error code because they are caught upstream or are DoS-guard rejections).
/// Returns `Err` for multi-channel LSTM — an explicit diagnostic error.
///
/// Rejects topologies that exceed safe bounds to prevent DoS/OOM:
/// - `num_layers > MAX_LSTM_LAYERS` (16)
/// - `hidden_size > MAX_LSTM_HIDDEN_SIZE` (1024)
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
        log::error!("LSTM num_layers=0 — rejected (no valid model can have zero layers)");
        return Ok(None);
    }
    if num_layers > MAX_LSTM_LAYERS {
        log::error!("LSTM num_layers={num_layers} exceeds maximum {MAX_LSTM_LAYERS} — rejected");
        return Ok(None);
    }
    if hidden_size > MAX_LSTM_HIDDEN_SIZE {
        log::error!(
            "LSTM hidden_size={hidden_size} exceeds maximum {MAX_LSTM_HIDDEN_SIZE} — rejected"
        );
        return Ok(None);
    }
    Ok(Some((num_layers, hidden_size)))
}
