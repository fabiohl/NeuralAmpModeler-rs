// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet Interleaved-4 weight transposition.
//!
//! **Cross-module dependency:** The output layout (interleave order, padding,
//! layer sub-component sequence) must remain in lock-step with the decoder in
//! `dispatcher/wavenet/`. See `dispatcher/wavenet/mod.rs` comments on the
//! interleaved-4 block read pattern.

use super::super::namb_encoder::ensure_capacity;
use crate::loader::nam_json::NamModelData;
use anyhow::{Context, Result};

/// Transposes WaveNet weight tensors into Interleaved-4 memory layout.
///
/// Reorders Conv1D and mixing weights into 4-wide SIMD vector blocks to allow parallel
/// multi-channel execution during WaveNet dilated convolution inference.
pub(crate) fn transpose_wavenet_interleaved4(data: &NamModelData) -> Result<Vec<f32>> {
    if data.architecture != "WaveNet" {
        anyhow::bail!("Layout Interleaved4WaveNet requires WaveNet architecture");
    }

    let mut cursor = 0;
    let mut out_weights = Vec::with_capacity(data.weights.len());

    for (li, layer_cfg) in data.config.layers.iter().enumerate() {
        let in_ch = layer_cfg.input_size.unwrap_or(1);
        let ch = layer_cfg.channels.unwrap_or(16);
        let cond_ch = layer_cfg.condition_size.unwrap_or(1);
        let k = layer_cfg.kernel_size.unwrap_or(3);
        let head_ch = layer_cfg.head_size.unwrap_or(8);
        let dilations = layer_cfg
            .dilations
            .as_ref()
            .context("WaveNet without dilations")?;
        let gated = layer_cfg.gated.unwrap_or(false);
        let conv_out_ch = if gated { 2 * ch } else { ch };

        // 1. Rechannel projection: Transposes [Output Channels][Input Channels] -> [Input][Output].
        let size = ch * in_ch;
        ensure_capacity(
            &data.weights,
            cursor,
            size,
            format!("Array {} Rechannel Weights", li),
        )?;
        let raw = &data.weights[cursor..cursor + size];
        for in_c in 0..in_ch {
            for out_c in 0..ch {
                out_weights.push(raw[out_c * in_ch + in_c]);
            }
        }
        cursor += size;

        // 2. Dilated Conv1D Layers & Interleaved-4 Reordering
        for (di, _) in dilations.iter().enumerate() {
            // Re-indexes Conv1D weights into Interleaved-4 channel blocks (4-wide SIMD vectors).
            let size = conv_out_ch * ch * k;
            ensure_capacity(
                &data.weights,
                cursor,
                size,
                format!("Array {} Layer {} Conv1D Weights", li, di),
            )?;
            let raw = &data.weights[cursor..cursor + size];
            let num_blocks = conv_out_ch.div_ceil(4);
            for b in 0..num_blocks {
                for ki in 0..k {
                    for in_c in 0..ch {
                        for lane in 0..4 {
                            let out_c = b * 4 + lane;
                            if out_c < conv_out_ch {
                                out_weights.push(raw[(out_c * ch + in_c) * k + ki]);
                            } else {
                                out_weights.push(0.0);
                            }
                        }
                    }
                }
            }
            cursor += size;

            // Conv1D filter additive bias vector.
            ensure_capacity(
                &data.weights,
                cursor,
                conv_out_ch,
                format!("Array {} Layer {} Conv1D Bias", li, di),
            )?;
            out_weights.extend_from_slice(&data.weights[cursor..cursor + conv_out_ch]);
            cursor += conv_out_ch;

            // Input Mixin projection: Linear mapping from conditioning state to channel dimension.
            let size = ch * cond_ch;
            ensure_capacity(
                &data.weights,
                cursor,
                size,
                format!("Array {} Layer {} Input Mixin Weights", li, di),
            )?;
            let raw = &data.weights[cursor..cursor + size];
            for in_c in 0..cond_ch {
                for out_c in 0..ch {
                    out_weights.push(raw[out_c * cond_ch + in_c]);
                }
            }
            cursor += size;

            // 1x1 Pointwise Convolution: Linear channel-mixing matrix across residual features.
            let size = ch * ch;
            ensure_capacity(
                &data.weights,
                cursor,
                size,
                format!("Array {} Layer {} 1x1 Weights", li, di),
            )?;
            let raw = &data.weights[cursor..cursor + size];
            for in_c in 0..ch {
                for out_c in 0..ch {
                    out_weights.push(raw[out_c * ch + in_c]);
                }
            }
            cursor += size;

            // 1x1 Pointwise Convolution additive bias vector.
            ensure_capacity(
                &data.weights,
                cursor,
                ch,
                format!("Array {} Layer {} 1x1 Bias", li, di),
            )?;
            out_weights.extend_from_slice(&data.weights[cursor..cursor + ch]);
            cursor += ch;
        }

        // 3. Head Rechannel projection: Linear mapping from residual channels to head dimension.
        let size = head_ch * ch;
        ensure_capacity(
            &data.weights,
            cursor,
            size,
            format!("Array {} Head Rechannel Weights", li),
        )?;
        let raw = &data.weights[cursor..cursor + size];
        for in_c in 0..ch {
            for out_c in 0..head_ch {
                out_weights.push(raw[out_c * ch + in_c]);
            }
        }
        cursor += size;

        // Head Rechannel additive bias vector (when enabled in layer topology).
        if layer_cfg.head_bias.unwrap_or(false) {
            ensure_capacity(
                &data.weights,
                cursor,
                head_ch,
                format!("Array {} Head Rechannel Bias", li),
            )?;
            out_weights.extend_from_slice(&data.weights[cursor..cursor + head_ch]);
            cursor += head_ch;
        }
    }

    // Trailing weights (e.g., final output scalar post-gain or head projection parameters).
    if cursor < data.weights.len() {
        out_weights.extend_from_slice(&data.weights[cursor..]);
    }

    Ok(out_weights)
}
