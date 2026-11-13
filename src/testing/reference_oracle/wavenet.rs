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

// Reference oracles mirror production formulas without individual item documentation.
#![allow(missing_docs)]

use crate::loader::nam_json::model::NamModelData;

use super::*;

/// Maximum kernel accepted by the independent post-stack head oracle.
const ORACLE_MAX_HEAD_KERNEL: usize = 16;

/// Activation subset implemented independently for the post-stack head.
#[derive(Clone, Copy, Debug)]
enum OracleHeadActivation {
    Tanh,
    ReLU,
    Sigmoid,
    SiLU,
    HardTanh,
    Softsign,
    PReLU(f64),
}

impl OracleHeadActivation {
    #[inline]
    fn apply(self, x: f64) -> f64 {
        match self {
            Self::Tanh => x.tanh(),
            Self::ReLU => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::SiLU => x / (1.0 + (-x).exp()),
            Self::HardTanh => x.clamp(-1.0, 1.0),
            Self::Softsign => x / (1.0 + x.abs()),
            Self::PReLU(slope) => {
                if x >= 0.0 {
                    x
                } else {
                    x * slope
                }
            }
        }
    }
}

/// One stage of the independent f64 post-stack head.
struct OracleHeadStage {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    do_bias: bool,
    weights: Vec<f64>,
    bias: Vec<f64>,
}

/// Independent f64 post-stack head description.
struct OracleHeadSpec {
    canonical: bool,
    activation: OracleHeadActivation,
    stages: Vec<OracleHeadStage>,
}

/// Parses and validates the post-stack `head` object without reusing the
/// production `HeadConfig` interpretation.
fn parse_oracle_post_head(head: &serde_json::Value) -> Result<OracleHeadSpec, OracleError> {
    let obj = head.as_object().ok_or(OracleError::UnsupportedTopology {
        architecture: "WaveNet",
        reason: "post-stack head must be a JSON object",
    })?;

    let kernel_sizes = obj.get("kernel_sizes");
    let kernel_size = obj.get("kernel_size");
    if kernel_sizes.is_some() && kernel_size.is_some() {
        return Err(OracleError::UnsupportedTopology {
            architecture: "WaveNet",
            reason: "post-stack head declares both kernel_size and kernel_sizes",
        });
    }

    let canonical = kernel_sizes.is_some();
    let kernels: Vec<usize> = if let Some(ks) = kernel_sizes {
        let arr = ks.as_array().ok_or(OracleError::UnsupportedTopology {
            architecture: "WaveNet",
            reason: "head.kernel_sizes must be an array",
        })?;
        if arr.is_empty() {
            return Err(OracleError::UnsupportedTopology {
                architecture: "WaveNet",
                reason: "head.kernel_sizes must be non-empty",
            });
        }
        let mut out = Vec::with_capacity(arr.len());
        for v in arr {
            let k = v.as_u64().and_then(|v| usize::try_from(v).ok()).ok_or(
                OracleError::UnsupportedTopology {
                    architecture: "WaveNet",
                    reason: "head.kernel_sizes entries must be positive integers",
                },
            )?;
            if k == 0 || k > ORACLE_MAX_HEAD_KERNEL {
                return Err(OracleError::UnsupportedTopology {
                    architecture: "WaveNet",
                    reason: "head.kernel_sizes entry outside 1..=16",
                });
            }
            out.push(k);
        }
        out
    } else {
        let k = match kernel_size {
            Some(v) => v.as_u64().and_then(|v| usize::try_from(v).ok()).ok_or(
                OracleError::UnsupportedTopology {
                    architecture: "WaveNet",
                    reason: "head.kernel_size must be a positive integer",
                },
            )?,
            None => 3,
        };
        if k == 0 || k > ORACLE_MAX_HEAD_KERNEL {
            return Err(OracleError::UnsupportedTopology {
                architecture: "WaveNet",
                reason: "head.kernel_size outside 1..=16",
            });
        }
        vec![k]
    };

    let channels = if canonical {
        obj.get("channels")
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
            .ok_or(OracleError::UnsupportedTopology {
                architecture: "WaveNet",
                reason: "canonical head.channels is required",
            })?
    } else {
        match obj.get("channels") {
            Some(v) => v.as_u64().and_then(|v| usize::try_from(v).ok()).ok_or(
                OracleError::UnsupportedTopology {
                    architecture: "WaveNet",
                    reason: "head.channels must be a positive integer",
                },
            )?,
            None => 0,
        }
    };
    let out_channels = if canonical {
        obj.get("out_channels")
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
            .ok_or(OracleError::UnsupportedTopology {
                architecture: "WaveNet",
                reason: "canonical head.out_channels is required",
            })?
    } else {
        match obj.get("out_channels") {
            Some(v) => v.as_u64().and_then(|v| usize::try_from(v).ok()).ok_or(
                OracleError::UnsupportedTopology {
                    architecture: "WaveNet",
                    reason: "head.out_channels must be a positive integer",
                },
            )?,
            None => 1,
        }
    };
    if out_channels == 0 || (canonical && channels == 0) {
        return Err(OracleError::UnsupportedTopology {
            architecture: "WaveNet",
            reason: "head channel counts must be positive",
        });
    }

    let do_bias = if canonical {
        true
    } else {
        obj.get("bias").and_then(|v| v.as_bool()).unwrap_or(false)
    };

    let activation = parse_oracle_head_activation(obj.get("activation"))?;

    // Stage widths: canonical uses `channels` for every non-final stage and
    // `out_channels` for the last. The legacy single stage historically used
    // `channels` as its input width (filled in by the caller once the layer
    // array is known).
    let mut stages = Vec::with_capacity(kernels.len());
    for (i, &kernel) in kernels.iter().enumerate() {
        let stage_out = if i + 1 == kernels.len() {
            out_channels
        } else {
            channels
        };
        stages.push(OracleHeadStage {
            in_ch: if canonical { 0 } else { channels },
            out_ch: stage_out,
            kernel,
            do_bias,
            weights: Vec::new(),
            bias: Vec::new(),
        });
    }

    Ok(OracleHeadSpec {
        canonical,
        activation,
        stages,
    })
}

fn parse_oracle_head_activation(
    value: Option<&serde_json::Value>,
) -> Result<OracleHeadActivation, OracleError> {
    let Some(value) = value else {
        return Ok(OracleHeadActivation::Tanh);
    };
    if let Some(name) = value.as_str() {
        return match name {
            "Tanh" | "FastTanh" => Ok(OracleHeadActivation::Tanh),
            "ReLU" => Ok(OracleHeadActivation::ReLU),
            "Sigmoid" => Ok(OracleHeadActivation::Sigmoid),
            "SiLU" => Ok(OracleHeadActivation::SiLU),
            "HardTanh" | "Hardtanh" => Ok(OracleHeadActivation::HardTanh),
            "Softsign" => Ok(OracleHeadActivation::Softsign),
            _ => Err(OracleError::UnsupportedTopology {
                architecture: "WaveNet",
                reason: "unsupported post-stack head activation",
            }),
        };
    }
    let obj = value.as_object().ok_or(OracleError::UnsupportedTopology {
        architecture: "WaveNet",
        reason: "unsupported post-stack head activation object",
    })?;
    if obj.get("type").and_then(|v| v.as_str()) == Some("PReLU") {
        let slope = obj
            .get("negative_slopes")
            .and_then(|v| v.as_array())
            .filter(|v| v.len() == 1)
            .and_then(|v| v[0].as_f64())
            .or_else(|| obj.get("negative_slope").and_then(|v| v.as_f64()))
            .ok_or(OracleError::UnsupportedTopology {
                architecture: "WaveNet",
                reason: "oracle PReLU head activation supports a scalar slope only",
            })?;
        return Ok(OracleHeadActivation::PReLU(slope));
    }
    Err(OracleError::UnsupportedTopology {
        architecture: "WaveNet",
        reason: "unsupported post-stack head activation object",
    })
}

/// Validates a non-null `config.head` object for the oracle domain.
pub(crate) fn validate_oracle_post_head(head: &serde_json::Value) -> Result<(), OracleError> {
    parse_oracle_post_head(head).map(|_| ())
}

impl OracleHeadSpec {
    /// Reads stage weights in NAMCore order: per stage `[out][in][kernel]`
    /// followed by `out` bias values.
    fn read_weights(&mut self, cursor: &mut Cursor<'_>) {
        for stage in &mut self.stages {
            let conv_len = stage.in_ch * stage.out_ch * stage.kernel;
            stage.weights = cursor.read_f64(conv_len);
            if stage.do_bias {
                stage.bias = cursor.read_f64(stage.out_ch);
            }
        }
    }

    /// Runs the head over interleaved `[frame][in_ch]` input.
    ///
    /// `input_scale` is applied before the first stage for canonical heads
    /// (NAMCore scales the layer-array head outputs before the head), or after
    /// the single legacy stage.
    fn forward(
        &self,
        input: &[f64],
        num_frames: usize,
        input_scale: f64,
        acc_mode: AccumulationMode,
    ) -> Vec<f64> {
        let mut current = input.to_vec();
        if self.canonical {
            for value in &mut current {
                *value = self.activation.apply(*value * input_scale);
            }
        }

        for (idx, stage) in self.stages.iter().enumerate() {
            let mut out = vec![0.0f64; num_frames * stage.out_ch];
            for f in 0..num_frames {
                for oc in 0..stage.out_ch {
                    let mut sum = if stage.do_bias { stage.bias[oc] } else { 0.0 };
                    let weight_base = oc * stage.in_ch * stage.kernel;
                    for kt in 0..stage.kernel {
                        let src = f as isize - (stage.kernel - 1 - kt) as isize;
                        if src < 0 {
                            continue;
                        }
                        let src = src as usize;
                        for ic in 0..stage.in_ch {
                            sum = mul_add_f64(
                                current[src * stage.in_ch + ic],
                                stage.weights[weight_base + ic * stage.kernel + kt],
                                sum,
                                acc_mode,
                            );
                        }
                    }
                    out[f * stage.out_ch + oc] = sum;
                }
            }

            if !self.canonical {
                for value in &mut out {
                    *value = self.activation.apply(*value);
                }
            } else if idx + 1 < self.stages.len() {
                // Activation before the next stage's convolution.
                for value in &mut out {
                    *value = self.activation.apply(*value);
                }
            } else {
                // Final canonical stage: no activation after the last conv.
            }
            current = out;
        }

        if !self.canonical {
            for value in &mut current {
                *value *= input_scale;
            }
        }
        current
    }
}

pub(crate) fn oracle_wavenet_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Result<Vec<f64>, OracleError> {
    oracle_wavenet_forward_inner(model_data, input, config, false)
}

pub(crate) fn oracle_wavenet_all_channels(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Result<Vec<f64>, OracleError> {
    oracle_wavenet_forward_inner(model_data, input, config, true)
}

fn oracle_wavenet_forward_inner(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
    _all_channels: bool,
) -> Result<Vec<f64>, OracleError> {
    let layers = &model_data.config.layers;
    let mut cursor = Cursor::new(&model_data.weights, config.weight_precision);
    let num_frames = input.len();

    if layers.len() < 2 {
        return Err(OracleError::UnsupportedTopology {
            architecture: "WaveNet",
            reason: "A1 oracle requires two layer arrays",
        });
    }

    let l0 = &layers[0];
    let l1 = &layers[1];
    let a0_ch = l0.channels.unwrap_or(16);
    let a0_head = l0.head_size.unwrap_or(8);
    let a0_k = l0.kernel_size.unwrap_or(3);
    let a0_dilations = l0.dilations.clone().unwrap_or_else(|| vec![1, 2, 4, 8]);
    let a0_cond = l0.condition_size.unwrap_or(1);

    // Process condition_dsp sub-model to obtain per-frame condition
    // vectors. Use oracle_condition_dsp_channels for ALL output channels
    // (matching C++ _condition_dsp_output_buffers), falling back to broadcast
    // only when the sub-model outputs a single channel (e.g. LSTM).
    let cond_output: Option<Vec<f64>> = if _all_channels {
        // Inner call: computing a condition_dsp sub-model itself — no nested
        // condition_dsp to process (would be infinite recursion).
        None
    } else {
        match model_data.config.condition_dsp.as_ref() {
            Some(json) => {
                let cond_model = parse_condition_dsp(json)?;
                let raw = oracle_condition_dsp_channels(&cond_model, input, config)?;
                let cond_size = a0_cond.max(1);
                if cond_size > 1 && raw.len() == num_frames {
                    let mut broadcasted = vec![0.0f64; num_frames * cond_size];
                    for f in 0..num_frames {
                        let val = raw[f];
                        for c in 0..cond_size {
                            broadcasted[f * cond_size + c] = val;
                        }
                    }
                    Some(broadcasted)
                } else {
                    Some(raw)
                }
            }
            None => None,
        }
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

    // Independent post-stack head oracle. Canonical NAMCore heads consume
    // their stage weights before `head_scale`; the legacy Rust-only head keeps
    // the historical `head_scale`-then-head order.
    let mut post_head = match model_data.config.head.as_ref() {
        Some(head) if !head.is_null() => Some(parse_oracle_post_head(head)?),
        _ => None,
    };
    if let Some(spec) = post_head.as_mut() {
        spec.stages[0].in_ch = if spec.canonical || spec.stages[0].in_ch == 0 {
            a1_head
        } else {
            spec.stages[0].in_ch
        };
        for i in 1..spec.stages.len() {
            spec.stages[i].in_ch = spec.stages[i - 1].out_ch;
        }
    }

    let head_scale = if let Some(spec) = post_head.as_mut() {
        if spec.canonical {
            spec.read_weights(&mut cursor);
            if cursor.pos < cursor.data.len() {
                cursor.read_one_f64()
            } else {
                model_data.config.head_scale.unwrap_or(1.0) as f64
            }
        } else {
            let scale = if cursor.pos < cursor.data.len() {
                cursor.read_one_f64()
            } else {
                model_data.config.head_scale.unwrap_or(1.0) as f64
            };
            spec.read_weights(&mut cursor);
            scale
        }
    } else if cursor.pos < cursor.data.len() {
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

    // Array1 head rechannel → post-stack head (or final output) × head_scale.
    if let Some(spec) = post_head {
        // The head consumes every channel of the last array head projection;
        // head_scale is applied before (canonical) or after (legacy) the head.
        let mut projection = Vec::new();
        oracle_wavenet_head_final(
            &mut projection,
            &a1_head_accum,
            &a1_head_w,
            &a1_head_b,
            a1_ch,
            a1_head,
            1.0,
            num_frames,
            acc_mode,
            true,
        );
        let head_out = spec.forward(&projection, num_frames, head_scale, acc_mode);
        let out_ch = spec.stages.last().map_or(1, |stage| stage.out_ch);
        if _all_channels {
            output = head_out;
        } else {
            output = (0..num_frames).map(|f| head_out[f * out_ch]).collect();
        }
    } else {
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
    }

    Ok(output)
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
