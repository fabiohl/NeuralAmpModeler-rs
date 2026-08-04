// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(missing_docs)]

use crate::loader::nam_json::model::{NamLayerConfig, NamModelData};
use crate::models::a2::weights_layout::{
    FILM_KEYS, film_bias_count, film_bias_count_generic, film_weight_count,
    film_weight_count_generic,
};

use super::super::*;

pub(crate) const A2_HEAD_KERNEL: usize = 16;

#[derive(Clone)]
pub(crate) struct FiLMOracleSlot {
    pub shift: bool,
    pub groups: u32,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
    pub buf: Vec<f64>,
}

impl FiLMOracleSlot {
    pub(crate) fn new(
        shift: bool,
        groups: u32,
        weights: Vec<f64>,
        bias: Vec<f64>,
        channels: usize,
    ) -> Self {
        let expected_bias = if shift { channels * 2 } else { channels };
        let mut padded_bias = bias;
        padded_bias.resize(expected_bias, 0.0);
        Self {
            shift,
            groups,
            weights,
            bias: padded_bias,
            buf: vec![0.0f64; channels * 2],
        }
    }

    pub(crate) fn apply(&mut self, input: &mut [f64], condition: &[f64]) {
        let constructed_ch = self.buf.len() / 2;
        let g = self.groups as usize;
        let ch_per_group = constructed_ch / g;
        let cond_per_group = condition.len().checked_div(g).unwrap_or(0);
        let out_per_group = if self.shift {
            ch_per_group * 2
        } else {
            ch_per_group
        };

        self.buf.fill(0.0);
        let buf = &mut self.buf;

        for grp in 0..g {
            let cond_off = grp * cond_per_group;
            let row_off = grp * out_per_group;
            let w_off = row_off * cond_per_group;
            for row in 0..out_per_group {
                let global_out = if row < ch_per_group {
                    grp * ch_per_group + row
                } else {
                    constructed_ch + grp * ch_per_group + (row - ch_per_group)
                };
                let mut sum = self.bias[global_out];
                for k in 0..cond_per_group {
                    sum += self.weights[w_off + row * cond_per_group + k] * condition[cond_off + k];
                }
                buf[global_out] = sum;
            }
        }

        let apply_len = input.len().min(constructed_ch);
        for c in 0..apply_len {
            let scale = buf[c];
            let shift = if self.shift {
                buf[c + constructed_ch]
            } else {
                0.0
            };
            input[c] = input[c] * scale + shift;
        }
    }
}

// ── Architecture parameter extraction ─────────────────────────────────────

pub(crate) fn a2_read_topology(
    layer_cfg: &NamLayerConfig,
) -> Option<(Vec<usize>, Vec<usize>, usize, usize)> {
    let dil = layer_cfg.dilations.clone()?;
    let nlayers = dil.len();
    if nlayers == 0 {
        return None;
    }
    let ks = if let Some(ks_vec) = layer_cfg.kernel_sizes.clone() {
        if ks_vec.len() != nlayers {
            return None;
        }
        ks_vec
    } else {
        let ks_scalar = layer_cfg.kernel_size?;
        vec![ks_scalar; nlayers]
    };
    let bn = layer_cfg
        .bottleneck
        .unwrap_or(layer_cfg.channels.unwrap_or(8));
    Some((ks, dil, nlayers, bn))
}

pub(crate) fn a2_read_activation(
    raw: &serde_json::Value,
    li: usize,
    _num_layers: usize,
) -> ActivationConfig {
    let arr = raw.get("activation").and_then(|v| v.as_array());
    if let Some(arr) = arr
        && li < arr.len()
    {
        return ActivationConfig::from_json(&arr[li]);
    }
    if let Some(obj) = raw.get("activation").and_then(|v| v.as_object()) {
        return ActivationConfig::from_json_obj(obj);
    }
    ActivationConfig::LeakyReLU {
        negative_slope: 0.01,
    }
}

pub(crate) fn a2_read_secondary_activation(raw: &serde_json::Value, li: usize) -> ActivationConfig {
    let arr = raw.get("secondary_activation").and_then(|v| v.as_array());
    if let Some(arr) = arr
        && li < arr.len()
    {
        if arr[li].is_null() {
            return ActivationConfig::Sigmoid;
        }
        return ActivationConfig::from_json(&arr[li]);
    }
    if let Some(obj) = raw.get("secondary_activation").and_then(|v| v.as_object()) {
        return ActivationConfig::from_json_obj(obj);
    }
    ActivationConfig::Sigmoid
}

pub(crate) fn a2_read_gating_mode(raw: &serde_json::Value, li: usize) -> GatingModeOracle {
    let arr = raw.get("gating_mode").and_then(|v| v.as_array());
    if let Some(arr) = arr
        && li < arr.len()
        && let Some(s) = arr[li].as_str()
    {
        return match s {
            "gated" => GatingModeOracle::Gated,
            "blended" => GatingModeOracle::Blended,
            _ => GatingModeOracle::None,
        };
    }
    GatingModeOracle::None
}

pub(crate) fn a2_read_head1x1_active(raw: &serde_json::Value) -> bool {
    raw.get("head1x1")
        .and_then(|v| v.get("active"))
        .and_then(|a| a.as_bool())
        .unwrap_or(false)
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum GatingModeOracle {
    None,
    Gated,
    Blended,
}

#[derive(Clone)]
pub(crate) enum ActivationConfig {
    Tanh,
    HardTanh,
    FastTanh,
    ReLU,
    LeakyReLU { negative_slope: f64 },
    Sigmoid,
    SiLU,
    HardSwish,
    Softsign,
}

impl ActivationConfig {
    pub(crate) fn from_json(v: &serde_json::Value) -> Self {
        let obj = v.as_object();
        if let Some(obj) = obj {
            return Self::from_json_obj(obj);
        }
        if let Some(s) = v.as_str() {
            return match s {
                "Tanh" => Self::Tanh,
                "HardTanh" => Self::HardTanh,
                "FastTanh" => Self::FastTanh,
                "ReLU" => Self::ReLU,
                "Sigmoid" => Self::Sigmoid,
                "SiLU" => Self::SiLU,
                "HardSwish" => Self::HardSwish,
                "Softsign" => Self::Softsign,
                _ => Self::Tanh,
            };
        }
        Self::Tanh
    }

    pub(crate) fn from_json_obj(obj: &serde_json::Map<String, serde_json::Value>) -> Self {
        let t = obj.get("type").and_then(|v| v.as_str()).unwrap_or("Tanh");
        match t {
            "Tanh" => Self::Tanh,
            "HardTanh" => Self::HardTanh,
            "FastTanh" => Self::FastTanh,
            "ReLU" => Self::ReLU,
            "LeakyReLU" => {
                let slope = obj
                    .get("negative_slope")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.01);
                Self::LeakyReLU {
                    negative_slope: slope,
                }
            }
            "Sigmoid" => Self::Sigmoid,
            "SiLU" => Self::SiLU,
            "HardSwish" => Self::HardSwish,
            "Softsign" => Self::Softsign,
            _ => Self::Tanh,
        }
    }

    pub(crate) fn apply(&self, z: &mut [f64], activation_mode: ActivationMode) {
        match self {
            Self::Tanh => {
                for v in z.iter_mut() {
                    *v = oracle_tanh(*v, activation_mode);
                }
            }
            Self::HardTanh => {
                for v in z.iter_mut() {
                    *v = v.clamp(-1.0, 1.0);
                }
            }
            Self::FastTanh => {
                for v in z.iter_mut() {
                    *v = oracle_tanh(*v, activation_mode);
                }
            }
            Self::ReLU => {
                for v in z.iter_mut() {
                    *v = v.max(0.0);
                }
            }
            Self::LeakyReLU { negative_slope } => {
                let s = *negative_slope;
                for v in z.iter_mut() {
                    if *v < 0.0 {
                        *v *= s;
                    }
                }
            }
            Self::Sigmoid => {
                for v in z.iter_mut() {
                    *v = oracle_sigmoid(*v, activation_mode);
                }
            }
            Self::SiLU => {
                for v in z.iter_mut() {
                    let s = oracle_sigmoid(*v, activation_mode);
                    *v *= s;
                }
            }
            Self::HardSwish => {
                for v in z.iter_mut() {
                    let relu6 = (*v + 3.0).clamp(0.0, 6.0);
                    *v = *v * relu6 / 6.0;
                }
            }
            Self::Softsign => {
                for v in z.iter_mut() {
                    *v /= 1.0 + v.abs();
                }
            }
        }
    }
}

pub(crate) struct A2OracleLayerWeights {
    pub conv_w: Vec<f64>,
    pub conv_b: Vec<f64>,
    pub mixin_w: Vec<f64>,
    pub l1x1_w: Vec<f64>,
    pub l1x1_b: Vec<f64>,
    pub ks: usize,
    pub dil: usize,
    pub film: Vec<Option<FiLMOracleSlot>>,
    pub gating_mode: GatingModeOracle,
    pub activation: ActivationConfig,
    pub secondary_activation: ActivationConfig,
    pub conv_out: usize,
}

pub(crate) struct ArrayState {
    pub ch: usize,
    pub head_accum_size: usize,
    pub bottleneck: usize,
    pub cond_size: usize,
    pub head_size: usize,
    pub head_is_rechannel: bool,
    pub rechannel_w: Vec<f64>,
    pub lws: Vec<A2OracleLayerWeights>,
    pub head1x1_active: bool,
    pub h1_groups: usize,
    pub h1_in_size: usize,
    pub head1x1_w: Vec<f64>,
    pub head1x1_b: Vec<f64>,
    pub head_w: Vec<f64>,
    pub head_b: Vec<f64>,
    pub fwd_bufs: Vec<Vec<f64>>,
}

/// Builds reference f64 `ArrayState` structures from parsed A2 model configuration and raw weight data.
///
/// Iterates over each A2 array layer, extracting topology parameters (kernel sizes, dilations,
/// bottleneck channels), gating/activation configurations, FiLM modulation slots, rechannel matrices,
/// and head output weights.
pub(crate) fn build_a2_arrays(
    model_data: &NamModelData,
    cursor: &mut Cursor,
) -> Option<Vec<ArrayState>> {
    let layers = &model_data.config.layers;
    let num_arrays = layers.len();
    let mut arrays: Vec<ArrayState> = Vec::with_capacity(num_arrays);

    // Process each array in the A2 architecture cascade
    for (ai, layer_cfg) in layers.iter().enumerate() {
        let ch = layer_cfg.channels.unwrap_or(8);
        let layer_raw = layer_cfg.layer_raw.clone();
        let cond_size = layer_cfg.condition_size.unwrap_or(1);

        let (kernel_sizes, dilations, num_layers, bottleneck) = a2_read_topology(layer_cfg)?;

        if num_layers == 0 {
            return None;
        }

        let head1x1_active = layer_raw
            .as_ref()
            .map(a2_read_head1x1_active)
            .unwrap_or(false);

        let pre_gating_modes: Vec<GatingModeOracle> = if let Some(ref raw) = layer_raw {
            (0..num_layers)
                .map(|li| a2_read_gating_mode(raw, li))
                .collect()
        } else {
            vec![GatingModeOracle::None; num_layers]
        };

        let pre_activations: Vec<ActivationConfig> = if let Some(ref raw) = layer_raw {
            (0..num_layers)
                .map(|li| a2_read_activation(raw, li, num_layers))
                .collect()
        } else {
            vec![
                ActivationConfig::LeakyReLU {
                    negative_slope: 0.01,
                };
                num_layers
            ]
        };

        let pre_secondary_activations: Vec<ActivationConfig> = if let Some(ref raw) = layer_raw {
            (0..num_layers)
                .map(|li| a2_read_secondary_activation(raw, li))
                .collect()
        } else {
            vec![ActivationConfig::Sigmoid; num_layers]
        };

        let film_configs: [bool; 8] = if let Some(ref raw) = layer_raw {
            let mut active = [false; 8];
            for &(key, idx) in FILM_KEYS {
                let cfg = raw.get(key).and_then(|v| v.as_object());
                if let Some(obj) = cfg {
                    let a = obj.get("active").and_then(|a| a.as_bool()).unwrap_or(false);
                    if a {
                        active[idx] = true;
                    }
                }
            }
            active
        } else {
            [false; 8]
        };

        let in_ch: usize = if ai == 0 {
            1
        } else {
            layers[ai - 1].channels.unwrap_or(8)
        };
        let rechannel_w = cursor.read_f64(in_ch * ch);

        let mut lws: Vec<A2OracleLayerWeights> = Vec::new();
        for li in 0..num_layers {
            let ks = kernel_sizes[li];
            let dil = dilations[li];
            let gmode = pre_gating_modes[li];
            let use_gating = gmode == GatingModeOracle::Gated || gmode == GatingModeOracle::Blended;
            let conv_out = if use_gating {
                bottleneck * 2
            } else {
                bottleneck
            };

            let conv_w = cursor.read_f64(ch * conv_out * ks);
            let conv_b = cursor.read_f64(conv_out);
            let mixin_w = cursor.read_f64(conv_out * cond_size);
            let l1x1_w = cursor.read_f64(bottleneck * ch);
            let l1x1_b = cursor.read_f64(ch);

            let mut film_slots: Vec<Option<FiLMOracleSlot>> = vec![None; 8];
            for slot_idx in 0..8 {
                if !film_configs[slot_idx] {
                    continue;
                }
                let g = layer_raw
                    .as_ref()
                    .and_then(|raw| {
                        let key = FILM_KEYS.iter().find(|(_, idx)| *idx == slot_idx)?.0;
                        raw.get(key)
                    })
                    .and_then(|v| v.get("groups"))
                    .and_then(|g| g.as_u64())
                    .unwrap_or(1) as u32;
                let shift = layer_raw
                    .as_ref()
                    .and_then(|raw| {
                        let key = FILM_KEYS.iter().find(|(_, idx)| *idx == slot_idx)?.0;
                        raw.get(key)
                    })
                    .and_then(|v| v.get("shift"))
                    .and_then(|s| s.as_bool())
                    .unwrap_or(true);

                let film_ch = match slot_idx {
                    2 => cond_size,
                    7 => 1.min(cond_size),
                    _ => ch,
                };

                let (w_count, b_count) = if cond_size > 1 {
                    (
                        film_weight_count_generic(g, cond_size, film_ch, shift),
                        film_bias_count_generic(film_ch),
                    )
                } else {
                    (
                        film_weight_count(g, cond_size, film_ch, shift),
                        film_bias_count(film_ch, shift),
                    )
                };
                let weights = cursor.read_f64(w_count);
                let bias = cursor.read_f64(b_count);
                film_slots[slot_idx] = Some(FiLMOracleSlot::new(shift, g, weights, bias, film_ch));
            }

            lws.push(A2OracleLayerWeights {
                conv_w,
                conv_b,
                mixin_w,
                l1x1_w,
                l1x1_b,
                ks,
                dil,
                film: film_slots,
                gating_mode: gmode,
                activation: pre_activations[li].clone(),
                secondary_activation: pre_secondary_activations[li].clone(),
                conv_out,
            });
        }

        let h1_groups = layer_raw
            .as_ref()
            .and_then(|raw| raw.get("head1x1"))
            .and_then(|h| h.get("groups"))
            .and_then(|g| g.as_u64())
            .unwrap_or(1) as usize;
        let h1_in_size = if head1x1_active {
            bottleneck / h1_groups
        } else {
            0
        };
        let head_accum_size = if head1x1_active {
            layer_raw
                .as_ref()
                .and_then(|raw| raw.get("head1x1"))
                .and_then(|h| h.get("out_channels"))
                .and_then(|a| a.as_u64())
                .unwrap_or(bottleneck as u64) as usize
        } else {
            bottleneck
        };
        let head1x1_w: Vec<f64> = if head1x1_active {
            cursor.read_f64(head_accum_size * h1_in_size)
        } else {
            vec![]
        };
        let head1x1_b: Vec<f64> = if head1x1_active {
            cursor.read_f64(head_accum_size)
        } else {
            vec![]
        };

        let head_size_raw = layer_cfg.head_size;
        let head_is_rechannel = head_size_raw.is_some();
        let head_size = head_size_raw.unwrap_or(1);
        let (head_w, head_b) = if head_is_rechannel {
            let hw_count = head_accum_size * head_size;
            let head_w = cursor.read_f64(hw_count);
            let head_bias = layer_cfg.head_bias.unwrap_or(false);
            let head_b = if head_bias {
                cursor.read_f64(head_size)
            } else {
                vec![0.0f64; head_size]
            };
            (head_w, head_b)
        } else {
            let head_w_raw = cursor.read_f64(A2_HEAD_KERNEL * head_accum_size);
            let mut head_w = vec![0.0f64; A2_HEAD_KERNEL * head_accum_size];
            for tap in 0..A2_HEAD_KERNEL {
                for c in 0..head_accum_size {
                    head_w[tap * head_accum_size + c] = head_w_raw[c * A2_HEAD_KERNEL + tap];
                }
            }
            let head_b = vec![cursor.read_one_f64()];
            let _head_scale_val = cursor.read_one_f64();
            (head_w, head_b)
        };

        arrays.push(ArrayState {
            ch,
            head_accum_size,
            bottleneck,
            cond_size,
            head_size,
            head_is_rechannel,
            rechannel_w,
            lws,
            head1x1_active,
            h1_groups,
            h1_in_size,
            head1x1_w,
            head1x1_b,
            head_w,
            head_b,
            fwd_bufs: vec![],
        });
    }

    Some(arrays)
}
