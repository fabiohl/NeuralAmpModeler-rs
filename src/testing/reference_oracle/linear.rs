// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! f64 reference oracle for the Linear architecture (mono and multichannel).
//!
//! Provides pure-f64 FIR convolution with strict adherence to C++ NAMCore accumulation
//! order and bias placement rules:
//! - $1 \to N$: 1 input channel, $N$ independent kernels, per-channel bias
//! - $N \to 1$: $N$ input channels, $N$ independent kernels, strict ascending summation order:
//!   `acc = bias[0] + ch_0 + ch_1 + ... + ch_{N-1}`
//! - $N \to N$: $N$ input channels, $N$ output channels, 1 single shared kernel, replicated bias

use super::PrecisionConfig;
use crate::loader::nam_json::model::NamModelData;

/// Computes the forward pass of a mono Linear model in double precision (f64).
pub fn oracle_linear_forward(
    model_data: &NamModelData,
    input: &[f64],
    _config: &PrecisionConfig,
) -> Vec<f64> {
    let in_channels = model_data.config.in_channels.unwrap_or(1);
    let out_channels = model_data.config.out_channels.unwrap_or(1);
    if in_channels == 1 && out_channels == 1 {
        let mc_in = vec![input.to_vec()];
        let mc_out = oracle_linear_multichannel(model_data, &mc_in);
        mc_out.into_iter().next().unwrap_or_default()
    } else {
        vec![0.0; input.len()]
    }
}

/// Computes the multichannel forward pass of a Linear model in double precision (f64).
///
/// Handles:
/// - $1 \to N$ (1 input channel, $N$ output channels, $N$ independent kernels)
/// - $N \to 1$ ($N$ input channels, 1 output channel, $N$ independent kernels, strict summation order:
///   `bias[0] + ch_0 + ch_1 + ... + ch_{N-1}`)
/// - $N \to N$ ($N$ input channels, $N$ output channels, 1 single shared kernel, bias replication)
pub fn oracle_linear_multichannel(model_data: &NamModelData, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let in_channels = model_data.config.in_channels.unwrap_or(1);
    let out_channels = model_data.config.out_channels.unwrap_or(1);
    let receptive_field = model_data.config.receptive_field.unwrap_or(1);
    let has_bias = model_data.config.bias.unwrap_or(false);

    let num_frames = input.iter().map(|ch| ch.len()).min().unwrap_or(0);
    if num_frames == 0 {
        return vec![Vec::new(); out_channels];
    }

    let kernels = if in_channels == out_channels {
        1
    } else {
        in_channels.max(out_channels)
    };

    let weights_f64: Vec<f64> = model_data.weights.iter().map(|&w| w as f64).collect();

    // Extract biases matching upstream C++ NAMCore:
    // const int biases = in_channels == out_channels ? 1 : out_channels;
    let bias_offset = kernels * receptive_field;
    let mut channel_biases = vec![0.0f64; out_channels];
    if has_bias {
        if in_channels == out_channels {
            let b = if weights_f64.len() > bias_offset {
                weights_f64[bias_offset]
            } else {
                0.0
            };
            channel_biases.fill(b);
        } else {
            for ch in 0..out_channels {
                if weights_f64.len() > bias_offset + ch {
                    channel_biases[ch] = weights_f64[bias_offset + ch];
                }
            }
        }
    }

    let mut output = vec![vec![0.0f64; num_frames]; out_channels];

    if in_channels == 1 && out_channels >= 1 {
        // 1 -> N
        let in_buf = &input[0];
        for ch in 0..out_channels {
            let kernel_start = ch * receptive_field;
            let b = channel_biases[ch];
            for i in 0..num_frames {
                let mut acc = b;
                for k in 0..receptive_field {
                    if i >= k {
                        acc += weights_f64[kernel_start + k] * in_buf[i - k];
                    }
                }
                output[ch][i] = acc;
            }
        }
    } else if in_channels > 1 && out_channels == 1 {
        // N -> 1: strict ascending summation order:
        // acc = bias[0] + ch_0 + ch_1 + ... + ch_{N-1}
        let b = channel_biases[0];
        for i in 0..num_frames {
            let mut acc = b;
            for (ch, in_buf) in input.iter().enumerate().take(in_channels) {
                let kernel_start = ch * receptive_field;
                let mut ch_conv = 0.0f64;
                for k in 0..receptive_field {
                    if i >= k {
                        ch_conv += weights_f64[kernel_start + k] * in_buf[i - k];
                    }
                }
                acc += ch_conv;
            }
            output[0][i] = acc;
        }
    } else if in_channels == out_channels {
        // N -> N (single shared kernel 0 for all channels)
        for (ch, in_buf) in input.iter().enumerate().take(in_channels) {
            let b = channel_biases[ch];
            for i in 0..num_frames {
                let mut acc = b;
                for k in 0..receptive_field {
                    if i >= k {
                        acc += weights_f64[k] * in_buf[i - k];
                    }
                }
                output[ch][i] = acc;
            }
        }
    }

    output
}
