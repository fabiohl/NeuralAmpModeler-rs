// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! f64 reference oracle for WaveNet neural models.
//!
//! Implements the standard WaveNet forward pass (non-A2, non-FiLM) in
//! pure f64 precision. Supports the canonical 4-variant family:
//! Standard (12 channels, 3 kernel, 6 head-scale), Lite (6,3,3),
//! Feather (8,3,4), and Nano (4,3,2).
//!
//! The forward pass applies stacked dilated 1D convolutions with
//! residual connections and a gated activation (tanh ⊙ sigmoid),
//! followed by a 1×1 head-mix projection to mono output.

#![allow(missing_docs)]

use crate::loader::nam_json::model::NamModelData;

use super::*;

pub(crate) fn oracle_wavenet_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    oracle_wavenet_forward_inner(model_data, input, config, false)
}

pub(crate) fn oracle_wavenet_all_channels(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    oracle_wavenet_forward_inner(model_data, input, config, true)
}

fn oracle_wavenet_forward_inner(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
    _all_channels: bool,
) -> Vec<f64> {
    let layers = &model_data.config.layers;
    let mut cursor = Cursor::new(&model_data.weights, config.weight_precision);
    let num_frames = input.len();

    if layers.len() < 2 {
        return vec![0.0; num_frames];
    }

    let l0 = &layers[0];
    let l1 = &layers[1];
    let a0_ch = l0.channels.unwrap_or(16);
    let a0_head = l0.head_size.unwrap_or(8);
    let a0_k = l0.kernel_size.unwrap_or(3);
    let a0_dilations = l0.dilations.clone().unwrap_or_else(|| vec![1, 2, 4, 8]);
    let a0_cond = l0.condition_size.unwrap_or(1);

    // T1.2: Process condition_dsp sub-model to obtain per-frame condition
    // vectors. Use oracle_condition_dsp_channels for ALL output channels
    // (matching C++ _condition_dsp_output_buffers), falling back to broadcast
    // only when the sub-model outputs a single channel (e.g. LSTM).
    let cond_output: Option<Vec<f64>> = if _all_channels {
        // Inner call: computing a condition_dsp sub-model itself — no nested
        // condition_dsp to process (would be infinite recursion).
        None
    } else {
        model_data.config.condition_dsp.as_ref().map(|json| {
            let cond_model: NamModelData =
                serde_json::from_value(json.clone()).expect("Failed to parse condition_dsp JSON");
            let raw = oracle_condition_dsp_channels(&cond_model, input, config);
            let cond_size = a0_cond.max(1);
            if cond_size > 1 && raw.len() == num_frames {
                let mut broadcasted = vec![0.0f64; num_frames * cond_size];
                for f in 0..num_frames {
                    let val = raw[f];
                    for c in 0..cond_size {
                        broadcasted[f * cond_size + c] = val;
                    }
                }
                broadcasted
            } else {
                raw
            }
        })
    };

    let a1_ch = a0_head;
    let a1_head = l1.head_size.unwrap_or(1);
    let a1_k = l1.kernel_size.unwrap_or(3);
    let a1_dilations = l1.dilations.clone().unwrap_or_else(|| vec![1, 2, 4, 8]);
    let a1_cond = l1.condition_size.unwrap_or(1);

    // Receptive field per array
    let a0_rf: usize = a0_dilations.iter().map(|&d| (a0_k - 1) * d).sum();
    let a1_rf: usize = a1_dilations.iter().map(|&d| (a1_k - 1) * d).sum();
    let max_rf = a0_rf.max(a1_rf) + 64;
    let acc_mode = config.accumulation;

    let mut output = vec![0.0f64; num_frames];

    struct LayerW {
        conv_w: Vec<f64>,
        conv_b: Vec<f64>,
        mixin_w: Vec<f64>,
        l1x1_w: Vec<f64>,
        l1x1_b: Vec<f64>,
        dilation: usize,
    }

    // ── Read Array0 weights ────────────────────────────────────────────────
    let a0_rechannel_w = cursor.read_f64(a0_ch);
    let a0_num_layers = a0_dilations.len();
    let mut a0_lws: Vec<LayerW> = Vec::new();
    for &dil in &a0_dilations {
        let conv_w = cursor.read_f64(a0_ch * a0_ch * a0_k);
        let conv_b = cursor.read_f64(a0_ch);
        let mixin_w = cursor.read_f64(a0_cond * a0_ch);
        let l1x1_w = cursor.read_f64(a0_ch * a0_ch);
        let l1x1_b = cursor.read_f64(a0_ch);
        a0_lws.push(LayerW {
            conv_w,
            conv_b,
            mixin_w,
            l1x1_w,
            l1x1_b,
            dilation: dil,
        });
    }
    let a0_head_w = cursor.read_f64(a0_ch * a0_head);

    // ── Read Array1 weights ────────────────────────────────────────────────
    let a1_rechannel_w = cursor.read_f64(a0_ch * a1_ch);
    let a1_num_layers = a1_dilations.len();
    let mut a1_lws: Vec<LayerW> = Vec::new();
    for &dil in &a1_dilations {
        let conv_w = cursor.read_f64(a1_ch * a1_ch * a1_k);
        let conv_b = cursor.read_f64(a1_ch);
        let mixin_w = cursor.read_f64(a1_cond * a1_ch);
        let l1x1_w = cursor.read_f64(a1_ch * a1_ch);
        let l1x1_b = cursor.read_f64(a1_ch);
        a1_lws.push(LayerW {
            conv_w,
            conv_b,
            mixin_w,
            l1x1_w,
            l1x1_b,
            dilation: dil,
        });
    }
    let a1_head_w = cursor.read_f64(a1_ch * a1_head);
    let a1_head_b = cursor.read_f64(a1_head);

    // head_scale is the LAST weight in the WaveNet weight stream
    // (C++ model.cpp constructs read it as the final weight after all arrays).
    // The JSON config's head_scale field is metadata and may not match the
    // weight-stream value (e.g. models generated by test scripts may write
    // random weights that overwrite the config default). Production engines
    // always use the weight-stream value.
    let head_scale = if cursor.pos < cursor.data.len() {
        cursor.read_one_f64()
    } else {
        model_data.config.head_scale.unwrap_or(1.0) as f64
    };

    // ── Array0 per-layer buffers ───────────────────────────────────────────
    // a0_bufs[0..N] where N = num_layers + 1 (buf[0] for rechannel, buf[i+1] for layer i output)
    let buf_size = max_rf + num_frames + 64;
    let bs = max_rf;
    let a0_buf_count = a0_num_layers + 1;
    let mut a0_bufs: Vec<Vec<f64>> = (0..a0_buf_count)
        .map(|_| vec![0.0f64; buf_size * a0_ch])
        .collect();
    let mut a0_ch_out = vec![0.0f64; num_frames * a0_ch];
    let mut a0_out = vec![0.0f64; num_frames * a0_head];
    let mut a0_head_accum = vec![0.0f64; num_frames * a0_ch];

    // Rechannel → a0_bufs[0]
    for (f, &inp) in input.iter().enumerate() {
        let idx = bs + f;
        for (c, rec_w) in a0_rechannel_w.iter().enumerate() {
            a0_bufs[0][idx * a0_ch + c] = inp * *rec_w;
        }
    }

    // Array0 layer cascade
    for (li, lw) in a0_lws.iter().enumerate() {
        let is_first = li == 0;
        for (f, &inp) in input.iter().enumerate() {
            let idx = bs + f;

            // Conv1d + mixin (reads from a0_bufs[li])
            let conv_out = {
                let hist = &a0_bufs[li];
                let mut conv_out = vec![0.0f64; a0_ch];
                for (oc, cv) in conv_out.iter_mut().enumerate() {
                    let mut sum = lw.conv_b[oc];
                    let wb = oc * a0_ch * a0_k;
                    for kt in 0..a0_k {
                        let off = (lw.dilation as isize) * ((kt as isize) + 1 - (a0_k as isize));
                        let ins = ((idx as isize) + off) as usize * a0_ch;
                        for ic in 0..a0_ch {
                            if ins + ic < hist.len() {
                                sum = mul_add_f64(
                                    hist[ins + ic],
                                    lw.conv_w[wb + ic * a0_k + kt],
                                    sum,
                                    acc_mode,
                                );
                            }
                        }
                    }
                    *cv = sum;
                }
                for (c, co) in conv_out.iter_mut().enumerate() {
                    let mix = if a0_cond == 1 {
                        inp * lw.mixin_w[c]
                    } else if let Some(ref co_vec) = cond_output {
                        let mut s = 0.0f64;
                        for j in 0..a0_cond {
                            s = mul_add_f64(
                                co_vec[f * a0_cond + j],
                                lw.mixin_w[c * a0_cond + j],
                                s,
                                acc_mode,
                            );
                        }
                        s
                    } else {
                        inp * lw.mixin_w[c]
                    };
                    *co = accum_f64(*co, mix, acc_mode);
                }
                for cv in conv_out.iter_mut() {
                    *cv = oracle_tanh(*cv, config.activation);
                }
                conv_out
            };

            if is_first {
                for c in 0..a0_ch {
                    a0_head_accum[f * a0_ch + c] = conv_out[c];
                }
            } else {
                for c in 0..a0_ch {
                    a0_head_accum[f * a0_ch + c] =
                        accum_f64(a0_head_accum[f * a0_ch + c], conv_out[c], acc_mode);
                }
            }

            // L1x1 residual → next layer's buffer (reads a0_bufs[li] and writes a0_bufs[li+1])
            for oc in 0..a0_ch {
                let mut sum = lw.l1x1_b[oc];
                for (ic, co) in conv_out.iter().enumerate() {
                    sum = mul_add_f64(*co, lw.l1x1_w[oc * a0_ch + ic], sum, acc_mode);
                }
                a0_bufs[li + 1][idx * a0_ch + oc] =
                    accum_f64(a0_bufs[li][idx * a0_ch + oc], sum, acc_mode);
            }
        }
    }

    // Array0 head rechannel
    for f in 0..num_frames {
        for hc in 0..a0_head {
            let mut sum = 0.0f64;
            for c in 0..a0_ch {
                sum = mul_add_f64(
                    a0_head_accum[f * a0_ch + c],
                    a0_head_w[hc * a0_ch + c],
                    sum,
                    acc_mode,
                );
            }
            a0_out[f * a0_head + hc] = sum;
        }
    }

    // Save Array0 channel output for Array1's rechannel
    for f in 0..num_frames {
        let idx = bs + f;
        a0_ch_out[f * a0_ch..f * a0_ch + a0_ch]
            .copy_from_slice(&a0_bufs[a0_num_layers][idx * a0_ch..idx * a0_ch + a0_ch]);
    }

    // ── Array1 per-layer buffers ───────────────────────────────────────────
    let a1_buf_count = a1_num_layers + 1;
    let mut a1_bufs: Vec<Vec<f64>> = (0..a1_buf_count)
        .map(|_| vec![0.0f64; buf_size * a1_ch])
        .collect();
    let mut a1_head_accum = vec![0.0f64; num_frames * a1_ch];

    // Array1 rechannel from a0_ch_out → a1_bufs[0]
    for f in 0..num_frames {
        let idx = bs + f;
        for c in 0..a1_ch {
            let mut sum = 0.0f64;
            for ic in 0..a0_ch {
                sum = mul_add_f64(
                    a0_ch_out[f * a0_ch + ic],
                    a1_rechannel_w[c * a0_ch + ic],
                    sum,
                    acc_mode,
                );
            }
            a1_bufs[0][idx * a1_ch + c] = sum;
        }
    }

    // Array1 layer cascade
    for (li, lw) in a1_lws.iter().enumerate() {
        let is_first = li == 0;
        for (f, &inp) in input.iter().enumerate() {
            let idx = bs + f;

            let conv_out = {
                let hist = &a1_bufs[li];
                let mut conv_out = vec![0.0f64; a1_ch];
                for (oc, cv) in conv_out.iter_mut().enumerate() {
                    let mut sum = lw.conv_b[oc];
                    let wb = oc * a1_ch * a1_k;
                    for kt in 0..a1_k {
                        let off = (lw.dilation as isize) * ((kt as isize) + 1 - (a1_k as isize));
                        let ins = ((idx as isize) + off) as usize * a1_ch;
                        for ic in 0..a1_ch {
                            if ins + ic < hist.len() {
                                sum = mul_add_f64(
                                    hist[ins + ic],
                                    lw.conv_w[wb + ic * a1_k + kt],
                                    sum,
                                    acc_mode,
                                );
                            }
                        }
                    }
                    *cv = sum;
                }
                for (c, co) in conv_out.iter_mut().enumerate() {
                    let mix = if a1_cond == 1 {
                        inp * lw.mixin_w[c]
                    } else if let Some(ref co_vec) = cond_output {
                        let mut s = 0.0f64;
                        for j in 0..a1_cond {
                            s = mul_add_f64(
                                co_vec[f * a1_cond + j],
                                lw.mixin_w[c * a1_cond + j],
                                s,
                                acc_mode,
                            );
                        }
                        s
                    } else {
                        inp * lw.mixin_w[c]
                    };
                    *co = accum_f64(*co, mix, acc_mode);
                }
                for cv in conv_out.iter_mut() {
                    *cv = oracle_tanh(*cv, config.activation);
                }
                conv_out
            };

            if is_first {
                for c in 0..a1_ch {
                    a1_head_accum[f * a1_ch + c] =
                        accum_f64(a0_out[f * a1_ch + c], conv_out[c], acc_mode);
                }
            } else {
                for c in 0..a1_ch {
                    a1_head_accum[f * a1_ch + c] =
                        accum_f64(a1_head_accum[f * a1_ch + c], conv_out[c], acc_mode);
                }
            }

            // L1x1 residual → next layer's buffer
            for oc in 0..a1_ch {
                let mut sum = lw.l1x1_b[oc];
                for (ic, co) in conv_out.iter().enumerate() {
                    sum = mul_add_f64(*co, lw.l1x1_w[oc * a1_ch + ic], sum, acc_mode);
                }
                a1_bufs[li + 1][idx * a1_ch + oc] =
                    accum_f64(a1_bufs[li][idx * a1_ch + oc], sum, acc_mode);
            }
        }
    }

    // Array1 head rechannel → output × head_scale
    // T1.2: Use oracle_wavenet_head_final for both mono and all-channels output.
    oracle_wavenet_head_final(
        &mut output,
        &a1_head_accum,
        &a1_head_w,
        &a1_head_b,
        a1_ch,
        a1_head,
        head_scale,
        num_frames,
        acc_mode,
        _all_channels,
    );

    output
}

/// Compute WaveNet head finalization.
///
/// When `all_channels` is `false`, writes `num_frames` mono samples (channel 0 only).
/// When `all_channels` is `true`, writes `num_frames * head_size` interleaved samples
/// for all head output channels — used for `condition_dsp` sub-model output.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "WaveNet reference oracle requiring many parameters to reproduce original NAM implementation bit-exact behavior"
)]
fn oracle_wavenet_head_final(
    output: &mut Vec<f64>,
    head_accum: &[f64],
    head_w: &[f64],
    head_b: &[f64],
    accum_ch: usize,
    head_size: usize,
    head_scale: f64,
    num_frames: usize,
    acc_mode: AccumulationMode,
    all_channels: bool,
) {
    if all_channels {
        *output = vec![0.0f64; num_frames * head_size];
        for f in 0..num_frames {
            for hc in 0..head_size {
                let mut y = if hc < head_b.len() { head_b[hc] } else { 0.0 };
                for c in 0..accum_ch {
                    y = mul_add_f64(
                        head_accum[f * accum_ch + c],
                        head_w[hc * accum_ch + c],
                        y,
                        acc_mode,
                    );
                }
                output[f * head_size + hc] = y * head_scale;
            }
        }
    } else {
        *output = vec![0.0f64; num_frames];
        for f in 0..num_frames {
            let mut y = head_b[0];
            for c in 0..accum_ch {
                y = mul_add_f64(
                    head_accum[f * accum_ch + c],
                    head_w[c * head_size],
                    y,
                    acc_mode,
                );
            }
            output[f] = y * head_scale;
        }
    }
}
