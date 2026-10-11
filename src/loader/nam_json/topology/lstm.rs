// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of LSTM topologies from model data.

use super::super::data::NamModelData;
use super::wavenet::TopologyError;

/// Detected LSTM topology.
///
/// Contains the structural parameters of an LSTM model including layer count,
/// hidden size, input feature dimension, and multi-channel audio configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LstmTopology {
    /// Number of recurrent layers.
    pub num_layers: usize,
    /// Number of hidden units per cell.
    pub hidden_size: usize,
    /// Input feature dimension for layer 0 (must match `in_channels`).
    pub input_size: usize,
    /// Number of input audio channels.
    pub in_channels: usize,
    /// Number of output audio channels.
    pub out_channels: usize,
}

impl LstmTopology {
    /// Creates a mono 1->1 topology with default channels.
    pub const fn new(num_layers: usize, hidden_size: usize) -> Self {
        Self {
            num_layers,
            hidden_size,
            input_size: 1,
            in_channels: 1,
            out_channels: 1,
        }
    }
}

/// Checks and returns the LSTM geometry (`LstmTopology`).
///
/// Returns `Ok(Some(...))` for valid LSTM topologies that can be dispatched.
/// Returns `Ok(None)` when the model is not LSTM or when required config fields
/// (`num_layers`, `hidden_size`) are missing.
/// Returns `Err` with a typed [`TopologyError`] for invalid structural parameters:
/// - `input_size != in_channels` (E1316 `NamErrorCode::InvalidFeatureTopology`)
/// - `num_layers == 0`, `hidden_size == 0`, `in_channels == 0`, `out_channels == 0`, `input_size == 0`
/// - `num_layers > MAX_LSTM_LAYERS` (16)
/// - `hidden_size > MAX_LSTM_HIDDEN_SIZE` (1024)
pub fn get_lstm_topology(data: &NamModelData) -> Result<Option<LstmTopology>, TopologyError> {
    use super::super::validation::{MAX_LSTM_HIDDEN_SIZE, MAX_LSTM_IO_CHANNELS, MAX_LSTM_LAYERS};
    use crate::common::diagnostics::NamErrorCode;

    if data.architecture != "LSTM" {
        return Ok(None);
    }

    let Some(num_layers) = data.config.num_layers else {
        return Ok(None);
    };
    let Some(hidden_size) = data.config.hidden_size else {
        return Ok(None);
    };

    let in_channels = data.config.in_channels.unwrap_or(1);
    let out_channels = data.config.out_channels.unwrap_or(1);

    // DEC-LSTM-01: C++ NAMCore requires `input_size` in config (`config["input_size"]`),
    // while Rust infers `input_size = in_channels.unwrap_or(1)` when omitted for resilience
    // with synthetic models and backwards compatibility. When explicitly present,
    // `input_size != in_channels` is strictly rejected with `E1316` (`InvalidFeatureTopology`).
    let input_size = data.config.input_size.unwrap_or(in_channels);

    if input_size != in_channels {
        return Err(TopologyError {
            code: NamErrorCode::InvalidFeatureTopology,
            reason: format!(
                "LSTM input_size ({input_size}) does not match in_channels ({in_channels})"
            ),
        });
    }

    if in_channels == 0 {
        return Err(TopologyError {
            code: NamErrorCode::InvalidFeatureTopology,
            reason: "in_channels=0 (must have at least 1 input channel)".into(),
        });
    }
    if out_channels == 0 {
        return Err(TopologyError {
            code: NamErrorCode::InvalidFeatureTopology,
            reason: "out_channels=0 (must have at least 1 output channel)".into(),
        });
    }
    if input_size == 0 {
        return Err(TopologyError {
            code: NamErrorCode::InvalidFeatureTopology,
            reason: "input_size=0 (must have at least 1 input feature)".into(),
        });
    }
    if in_channels > MAX_LSTM_IO_CHANNELS {
        return Err(TopologyError {
            code: NamErrorCode::InvalidFeatureTopology,
            reason: format!("in_channels={in_channels} exceeds maximum {MAX_LSTM_IO_CHANNELS}"),
        });
    }
    if out_channels > MAX_LSTM_IO_CHANNELS {
        return Err(TopologyError {
            code: NamErrorCode::InvalidFeatureTopology,
            reason: format!("out_channels={out_channels} exceeds maximum {MAX_LSTM_IO_CHANNELS}"),
        });
    }
    if input_size > MAX_LSTM_IO_CHANNELS {
        return Err(TopologyError {
            code: NamErrorCode::InvalidFeatureTopology,
            reason: format!("input_size={input_size} exceeds maximum {MAX_LSTM_IO_CHANNELS}"),
        });
    }
    if num_layers == 0 {
        return Err(TopologyError {
            code: NamErrorCode::InvalidModelTopology,
            reason: "num_layers=0 (no valid model can have zero layers)".into(),
        });
    }
    if hidden_size == 0 {
        return Err(TopologyError {
            code: NamErrorCode::InvalidModelTopology,
            reason: "hidden_size=0 (no valid model can have zero hidden units)".into(),
        });
    }
    if num_layers > MAX_LSTM_LAYERS {
        return Err(TopologyError {
            code: NamErrorCode::InvalidModelTopology,
            reason: format!("num_layers={num_layers} exceeds maximum {MAX_LSTM_LAYERS}"),
        });
    }
    if hidden_size > MAX_LSTM_HIDDEN_SIZE {
        return Err(TopologyError {
            code: NamErrorCode::InvalidModelTopology,
            reason: format!("hidden_size={hidden_size} exceeds maximum {MAX_LSTM_HIDDEN_SIZE}"),
        });
    }

    Ok(Some(LstmTopology {
        num_layers,
        hidden_size,
        input_size,
        in_channels,
        out_channels,
    }))
}
