// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! LSTM Gate-Major weight transposition.
//!
//! **Cross-module dependency:** The output layout must remain in lock-step with
//! the decoder in `crate::loader::dispatcher::lstm` (weight reading order in `weights.rs`,
//! bias/state layout in `static_builder.rs` and `dynamic_builder.rs`).

use crate::loader::nam_json::NamModelData;
use anyhow::{Context, Result};

/// Transposes LSTM weight tensors into Gate-Major memory layout.
///
/// Re-indexes raw weights from `[4, H, IH]` row-major format into `[4, IH, H]` column-major
/// layout to enable contiguous SIMD vectorization during real-time fused matrix-vector operations.
pub(crate) fn transpose_lstm_gate_major(data: &NamModelData) -> Result<Vec<f32>> {
    if data.architecture != "LSTM" {
        anyhow::bail!("Layout GateMajorLstm requires LSTM architecture");
    }

    let hidden_size = data
        .config
        .hidden_size
        .context("LSTM without hidden_size")?;
    let num_layers = data.config.num_layers.unwrap_or(1);
    let in_channels = data.config.in_channels.unwrap_or(1);
    let out_channels = data.config.out_channels.unwrap_or(1);
    let initial_input_size = data.config.input_size.unwrap_or(in_channels);

    let mut cursor = 0;
    let mut out_weights = Vec::with_capacity(data.weights.len());

    let mut current_input_size = initial_input_size;
    for l in 0..num_layers {
        let ih = current_input_size
            .checked_add(hidden_size)
            .context("LSTM ih overflow")?;
        let h = hidden_size;
        let layer_size = (4usize)
            .checked_mul(h)
            .and_then(|w| w.checked_mul(ih))
            .context("LSTM layer size overflow")?;

        if cursor + layer_size > data.weights.len() {
            anyhow::bail!("Insufficient weights for LSTM at layer {l}");
        }

        let raw = &data.weights[cursor..cursor + layer_size];
        let mut transposed = vec![0.0f32; layer_size];

        // 3D tensor transposition: re-indexes weights from [4, H, IH] row-major to [4, IH, H]
        // column-major layout to optimize SIMD fused matrix-vector products during inference.
        for k in 0..4 {
            for i in 0..h {
                for j in 0..ih {
                    let val = raw[k * h * ih + i * ih + j];
                    transposed[k * h * ih + j * h + i] = val;
                }
            }
        }
        out_weights.extend(transposed);
        cursor += layer_size;

        let bias_size = (4usize).checked_mul(h).context("LSTM bias overflow")?;
        if cursor + bias_size > data.weights.len() {
            anyhow::bail!("Insufficient bias for LSTM at layer {l}");
        }
        out_weights.extend_from_slice(&data.weights[cursor..cursor + bias_size]);
        cursor += bias_size;

        // Layer initial state vectors (hidden_init and cell_init: 2 * H floats total).
        // Preserves exact sequential layout expected by `dispatcher/lstm.rs` state decoder.
        let state_size = (2usize)
            .checked_mul(h)
            .context("LSTM state size overflow")?; // hidden_init (h) + cell_init (h)
        if cursor + state_size > data.weights.len() {
            anyhow::bail!("Insufficient initial states for LSTM at layer {l}");
        }
        out_weights.extend_from_slice(&data.weights[cursor..cursor + state_size]);
        cursor += state_size;

        current_input_size = hidden_size;
    }

    // Trailing weights (head_weights and head_bias for final output projection).
    // Appended verbatim to maintain decoder parity.
    let head_w = out_channels
        .checked_mul(hidden_size)
        .context("LSTM head_weights overflow")?;
    let head_size = head_w
        .checked_add(out_channels)
        .context("LSTM head size overflow")?;
    if cursor + head_size > data.weights.len() {
        anyhow::bail!("Insufficient head weights for LSTM");
    }
    out_weights.extend_from_slice(&data.weights[cursor..cursor + head_size]);
    cursor += head_size;

    if cursor != data.weights.len() {
        anyhow::bail!(
            "Extra trailing weights in LSTM model: cursor={}, total={}",
            cursor,
            data.weights.len()
        );
    }

    Ok(out_weights)
}
