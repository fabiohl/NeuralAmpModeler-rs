// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! f64 reference oracle for ConvNet neural models.
//!
//! Supports both FlatCpp (old NAM) and standard layered formats with
//! optional batch-norm folding. The forward pass stacks per-layer
//! 1D convolutions, batchnorm/bias, and LeakyReLU activations,
//! followed by a linear head projection to mono output.

#![allow(missing_docs)]

use crate::loader::nam_json::model::NamModelData;

use super::*;

pub(crate) fn oracle_convnet_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    let layers = &model_data.config.layers;
    let mut head_scale = model_data.config.head_scale.unwrap_or(1.0) as f64;
    let mut cursor = Cursor::new(&model_data.weights, config.weight_precision);
    let num_frames = input.len();
    let acc_mode = config.accumulation;

    // ── FlatCpp format detection ──
    let is_flat_cpp = layers.is_empty()
        && model_data.config.conv_channels.is_some()
        && model_data.config.conv_dilations.is_some()
        && model_data.config.conv_batchnorm.is_some();

    if !is_flat_cpp && layers.is_empty() {
        return vec![0.0; num_frames];
    }

    struct BlockW {
        conv_w: Vec<f64>,
        conv_b: Vec<f64>,
        bn_scale: Vec<f64>,
        bn_offset: Vec<f64>,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        activation: String,
    }

    let mut blocks: Vec<BlockW> = Vec::new();

    if is_flat_cpp {
        let ch = model_data.config.conv_channels.unwrap();
        let dilations = model_data.config.conv_dilations.as_ref().unwrap();
        let batchnorm = model_data.config.conv_batchnorm.unwrap();
        let kernel = 2usize;
        head_scale = 1.0;

        for (i, &dilation) in dilations.iter().enumerate() {
            let in_ch = if i == 0 { 1 } else { ch };
            let out_ch = ch;

            let conv_w = cursor.read_f64(in_ch * out_ch * kernel);

            let (conv_b, bn_scale, bn_offset) = if batchnorm {
                let running_mean = cursor.read_f64(out_ch);
                let running_var = cursor.read_f64(out_ch);
                let gamma = cursor.read_f64(out_ch);
                let beta = cursor.read_f64(out_ch);
                let eps = cursor.read_f64(1)[0];

                let mut scale = vec![0.0f64; out_ch];
                let mut offset = vec![0.0f64; out_ch];
                let bias = vec![0.0f64; out_ch];
                for c in 0..out_ch {
                    scale[c] = gamma[c] / (eps + running_var[c]).sqrt();
                    offset[c] = beta[c] - scale[c] * running_mean[c];
                }
                (bias, scale, offset)
            } else {
                let bias = cursor.read_f64(out_ch);
                let scale = vec![1.0f64; out_ch];
                let offset = vec![0.0f64; out_ch];
                (bias, scale, offset)
            };

            blocks.push(BlockW {
                conv_w,
                conv_b,
                bn_scale,
                bn_offset,
                in_ch,
                out_ch,
                kernel,
                dilation,
                activation: "Tanh".to_string(),
            });
        }
    } else {
        for (i, layer) in layers.iter().enumerate() {
            let out_ch = layer.channels.unwrap_or(8);
            let in_ch = if i == 0 {
                1
            } else {
                layers[i - 1].channels.unwrap_or(out_ch)
            };
            let kernel = layer.kernel_size.unwrap_or(3);
            let dilation = layer
                .dilations
                .as_ref()
                .and_then(|d| d.first().copied())
                .unwrap_or(1);
            let activation = layer
                .activation
                .clone()
                .unwrap_or_else(|| "Tanh".to_string());

            let conv_w = cursor.read_f64(in_ch * out_ch * kernel);
            let conv_b = cursor.read_f64(out_ch);
            let bn_scale = cursor.read_f64(out_ch);
            let bn_offset = cursor.read_f64(out_ch);

            blocks.push(BlockW {
                conv_w,
                conv_b,
                bn_scale,
                bn_offset,
                in_ch,
                out_ch,
                kernel,
                dilation,
                activation,
            });
        }
    }

    let head = if is_flat_cpp {
        let last_out_ch = blocks.last().map(|b| b.out_ch).unwrap_or(1);
        let h_w = cursor.read_f64(last_out_ch);
        let h_b = cursor.read_f64(1);
        Some((h_w, h_b, last_out_ch, 1usize, 1usize, "Linear".to_string()))
    } else {
        let head_config = model_data.config.parse_head();
        head_config.map(|hc| {
            let last_out_ch = blocks.last().map(|b| b.out_ch).unwrap_or(1);
            let h_in_ch = hc.channels.unwrap_or(last_out_ch);
            let h_out_ch = hc.out_channels.unwrap_or(1);
            let h_kernel = hc.kernel_size.unwrap_or(1);
            let h_has_bias = hc.bias.unwrap_or(true);
            let h_activation = hc.activation.unwrap_or_else(|| "Tanh".to_string());

            let h_w = cursor.read_f64(h_in_ch * h_out_ch * h_kernel);
            let h_b = if h_has_bias {
                cursor.read_f64(h_out_ch)
            } else {
                vec![0.0; h_out_ch]
            };

            (h_w, h_b, h_in_ch, h_out_ch, h_kernel, h_activation)
        })
    };

    let max_rf: usize = blocks
        .iter()
        .map(|b| (b.kernel - 1) * b.dilation)
        .max()
        .unwrap_or(0)
        + 64;
    let hist_size = max_rf + num_frames + 64;

    let mut block_hists: Vec<Vec<f64>> = blocks
        .iter()
        .map(|b| vec![0.0f64; hist_size * b.in_ch])
        .collect();

    let mut output = vec![0.0f64; num_frames];

    for f in 0..num_frames {
        let hist_i = max_rf + f;

        block_hists[0][hist_i * blocks[0].in_ch] = input[f];

        let mut last_out: Option<Vec<f64>> = None;

        for (bi, block) in blocks.iter().enumerate() {
            let in_ch = block.in_ch;
            let out_ch = block.out_ch;
            let kernel = block.kernel;
            let dilation = block.dilation;

            let mut conv_out = vec![0.0f64; out_ch];
            let hist = &block_hists[bi];

            for (oc, cv) in conv_out.iter_mut().enumerate() {
                let mut sum = block.conv_b[oc];
                let wb = oc * in_ch * kernel;
                for kt in 0..kernel {
                    let off = (dilation as isize) * ((kt as isize) + 1 - (kernel as isize));
                    let ins = ((hist_i as isize) + off) as usize;
                    if ins < hist_size {
                        for ic in 0..in_ch {
                            sum = mul_add_f64(
                                hist[ins * in_ch + ic],
                                block.conv_w[wb + ic * kernel + kt],
                                sum,
                                acc_mode,
                            );
                        }
                    }
                }
                *cv = sum;
            }

            for (c, cv) in conv_out.iter_mut().enumerate() {
                *cv = cv.mul_add(block.bn_scale[c], block.bn_offset[c]);
            }

            oracle_apply_activation(&mut conv_out, &block.activation, config);

            if bi + 1 < blocks.len() {
                let next_in_ch = blocks[bi + 1].in_ch;
                for c in 0..out_ch.min(next_in_ch) {
                    block_hists[bi + 1][hist_i * next_in_ch + c] = conv_out[c];
                }
            }

            if bi == blocks.len() - 1 {
                last_out = Some(conv_out);
            }
        }

        let block_out = last_out.unwrap();

        let y = if let Some((ref hw, ref hb, h_in_ch, h_out_ch, h_kernel, ref h_act)) = head {
            let mut h_out = vec![0.0f64; h_out_ch];
            for oc in 0..h_out_ch {
                let mut sum = hb[oc];
                let wb = oc * h_in_ch * h_kernel;
                for kt in 0..h_kernel {
                    for ic in 0..h_in_ch {
                        sum =
                            mul_add_f64(block_out[ic], hw[wb + ic * h_kernel + kt], sum, acc_mode);
                    }
                }
                h_out[oc] = sum;
            }
            oracle_apply_activation(&mut h_out, h_act, config);
            h_out[0]
        } else {
            block_out[0]
        };

        output[f] = y * head_scale;
    }

    output
}
