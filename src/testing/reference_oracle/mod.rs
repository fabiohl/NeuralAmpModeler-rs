// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! f64 reference oracles for regression testing of all NAM architectures.
//!
//! Each sub-module provides a pure-f64, non-SIMD implementation of a neural
//! model's forward pass. These oracles serve as the ground-truth reference
//! against which the optimized f32 SIMD kernels are validated.
//!
//! # Architecture & Parity Maps
//! Detailed mathematical formulations and C++ NAMCore parity maps are documented in
//! `docs/cpp_parity_map.md` (see §4.4 for oracle accumulation modes and gate thresholds).
//!
//! # Sub-modules
//! - [`wavenet`] — WaveNet (Standard, Lite, Feather, Nano, A2, Dynamic).
//! - [`lstm`] — LSTM (1-layer and 2-layer).
//! - [`convnet`] — ConvNet (FlatCpp and layered formats).
//! - [`a2`] — WaveNet A2 with FiLM conditioning and gating.
//!
//! # Precision configuration
//! [`PrecisionConfig`] controls the weight precision (f32/f64/bf16),
//! activation approximation mode, and accumulation strategy used by
//! the oracles, enabling targeted decomposition comparisons.

#![allow(missing_docs)]

use crate::loader::nam_json::model::NamModelData;
use crate::models::a2::weights_layout::FILM_KEYS;

pub(crate) use a2::oracle_a2_forward;
pub(crate) use convnet::oracle_convnet_forward;
pub(crate) use lstm::oracle_lstm_forward;
pub(crate) use wavenet::{oracle_wavenet_all_channels, oracle_wavenet_forward};
pub mod a2;
pub mod convnet;
pub mod lstm;
pub mod wavenet;

// =============================================================================
// Condition DSP multi-channel forward
// =============================================================================

/// Compute the full multi-channel output of a `condition_dsp` sub-model.
///
/// Unlike `oracle_forward` (which always returns mono audio — 1 sample/frame),
/// this function returns all head output channels in **interleaved** order
/// (`[ch0_f0, ch1_f0, ..., chN_f0, ch0_f1, ...]`), matching the layout used
/// by the Rust production engine (`condition_dsp_output`) and C++ NAMcore
/// (`_condition_dsp_output_buffers`).
///
/// For LSTM sub-models the output is always mono (LSTM head has 1 output),
/// so it returns `num_frames` samples — the caller's broadcast logic handles
/// the dimensional mismatch identically to production code.
pub(crate) fn oracle_condition_dsp_channels(
    sub_model: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    match sub_model.architecture.as_str() {
        "WaveNet" => {
            if is_a2_model(sub_model) {
                // A2 condition_dsp: fall back to single-channel oracle_forward
                // (full A2 multi-channel support tracked in §4.4 of cpp_parity_map.md).
                oracle_a2_forward(sub_model, input, config)
            } else {
                oracle_wavenet_all_channels(sub_model, input, config)
            }
        }
        "LSTM" => oracle_lstm_forward(sub_model, input, config),
        "ConvNet" => oracle_convnet_forward(sub_model, input, config),
        _ => vec![0.0; input.len()],
    }
}

// =============================================================================
// Precision Configuration
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightPrecision {
    /// Full f64 precision (weights converted from f32).
    F64Exact,
    /// Weights quantized to f16c (binary16) then converted to f64.
    F16C,
    /// Weights quantized to bf16 then converted to f64.
    BF16,
    /// Weights kept in f32 precision (cast to f64 at compute time).
    F32,
}

/// Activation function precision mode for source decomposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationMode {
    /// Exact: `f64::tanh`, `f64` exp-based sigmoid.
    Exact,
    /// Approximate: Padé \[5,4\] tanh, minimax degree-17 sigmoid.
    PadeMinimax,
}

/// Accumulation mode for dot products and residual sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccumulationMode {
    /// Plain f64 accumulation (no compensation).
    F64Plain,
    /// Kahan compensated summation.
    Kahan,
    /// Neumaier compensated summation.
    Neumaier,
    /// Accumulation in f32 to simulate f32 error.
    F32Plain,
}

/// Precision configuration for one oracle run.
#[derive(Clone, Copy, Debug)]
pub struct PrecisionConfig {
    /// Weight representation precision.
    pub weight_precision: WeightPrecision,
    /// Activation function approximation mode.
    pub activation: ActivationMode,
    /// Accumulation algorithm.
    pub accumulation: AccumulationMode,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            weight_precision: WeightPrecision::F64Exact,
            activation: ActivationMode::Exact,
            accumulation: AccumulationMode::Neumaier,
        }
    }
}

// =============================================================================
// f64 Activation Functions
// =============================================================================

/// Padé \[5,4\] rational approximant for tanh in f64.
#[inline]
pub fn pade_tanh_f64(x: f64) -> f64 {
    let x = x.clamp(-4.0, 4.0);
    let x2 = x * x;
    let num = x * (x2 + 105.0).mul_add(x2, 945.0);
    let den = (15.0f64.mul_add(x2, 420.0)).mul_add(x2, 945.0);
    (num / den).clamp(-1.0, 1.0)
}

#[inline]
pub fn exact_tanh_f64(x: f64) -> f64 {
    x.tanh()
}

#[inline]
pub fn oracle_tanh(x: f64, mode: ActivationMode) -> f64 {
    match mode {
        ActivationMode::Exact => exact_tanh_f64(x),
        ActivationMode::PadeMinimax => pade_tanh_f64(x),
    }
}

#[inline]
pub fn exact_sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn minimax_sigmoid_f64(x: f64) -> f64 {
    let c0 = 2.4885319190e-01_f64;
    let c1 = -1.9318685012e-02_f64;
    let c2 = 1.4623214305e-03_f64;
    let c3 = -7.9953400187e-05_f64;
    let c4 = 2.9140652422e-06_f64;
    let c5 = -6.8000246432e-08_f64;
    let c6 = 9.6897239158e-10_f64;
    let c7 = -7.6498626314e-12_f64;
    let c8 = 2.5585471676e-14_f64;
    let xc = x.clamp(-8.0, 8.0);
    let x2 = xc * xc;
    let p = c8.mul_add(x2, c7);
    let p = p.mul_add(x2, c6);
    let p = p.mul_add(x2, c5);
    let p = p.mul_add(x2, c4);
    let p = p.mul_add(x2, c3);
    let p = p.mul_add(x2, c2);
    let p = p.mul_add(x2, c1);
    let p = p.mul_add(x2, c0);
    (xc.mul_add(p, 0.5)).clamp(0.0, 1.0)
}

#[inline]
pub fn oracle_sigmoid(x: f64, mode: ActivationMode) -> f64 {
    match mode {
        ActivationMode::Exact => exact_sigmoid_f64(x),
        ActivationMode::PadeMinimax => minimax_sigmoid_f64(x),
    }
}

// =============================================================================
// Weight Conversion
// =============================================================================

#[inline]
fn weight_f32_to_f64(w: f32, mode: WeightPrecision) -> f64 {
    match mode {
        WeightPrecision::F64Exact | WeightPrecision::F32 => w as f64,
        WeightPrecision::F16C => {
            let bits = crate::math::common::half::f32_to_f16_bits(w);
            crate::math::common::half::f16_bits_to_f32(bits) as f64
        }
        WeightPrecision::BF16 => {
            let bits = w.to_bits();
            let bf16_bits = (bits >> 16) as u16;
            f32::from_bits((bf16_bits as u32) << 16) as f64
        }
    }
}

/// Simple weight stream cursor with optional weight-precision quantization.
pub(crate) struct Cursor<'a> {
    data: &'a [f32],
    pos: usize,
    weight_mode: WeightPrecision,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [f32], weight_mode: WeightPrecision) -> Self {
        Self {
            data,
            pos: 0,
            weight_mode,
        }
    }

    fn read_f64(&mut self, count: usize) -> Vec<f64> {
        let end = self.pos + count;
        let out: Vec<f64> = self.data[self.pos..end]
            .iter()
            .map(|&x| weight_f32_to_f64(x, self.weight_mode))
            .collect();
        self.pos = end;
        out
    }

    fn read_one_f64(&mut self) -> f64 {
        let v = weight_f32_to_f64(self.data[self.pos], self.weight_mode);
        self.pos += 1;
        v
    }
}

/// Helper for accumulation mode: optionally casts through f32 to simulate f32 error.
#[inline]
fn accum_f64(a: f64, b: f64, mode: AccumulationMode) -> f64 {
    match mode {
        AccumulationMode::F64Plain | AccumulationMode::Kahan | AccumulationMode::Neumaier => a + b,
        AccumulationMode::F32Plain => {
            let s = (a as f32) + (b as f32);
            s as f64
        }
    }
}

/// Multiply-add with optional f32 accumulation.
#[inline]
fn mul_add_f64(a: f64, b: f64, c: f64, mode: AccumulationMode) -> f64 {
    match mode {
        AccumulationMode::F64Plain | AccumulationMode::Kahan | AccumulationMode::Neumaier => {
            a.mul_add(b, c)
        }
        AccumulationMode::F32Plain => {
            let p = (a as f32) * (b as f32);
            let s = p + (c as f32);
            s as f64
        }
    }
}

// =============================================================================
// ESR
// =============================================================================

pub fn compute_esr_f64(reference: &[f64], test: &[f64]) -> f64 {
    let len = reference.len().min(test.len());
    if len == 0 {
        return 0.0;
    }
    let mut noise = 0.0f64;
    let mut signal = 0.0f64;
    for i in 0..len {
        let diff = reference[i] - test[i];
        noise += diff * diff;
        signal += reference[i] * reference[i];
    }
    if signal == 0.0 {
        if noise == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        noise / signal
    }
}

pub fn esr_to_db_f64(esr: f64) -> f64 {
    if esr <= 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * esr.log10()
    }
}

// =============================================================================
// Top-level Oracle Dispatcher
// =============================================================================

pub fn oracle_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    match model_data.architecture.as_str() {
        "WaveNet" => {
            if is_a2_model(model_data) {
                oracle_a2_forward(model_data, input, config)
            } else {
                oracle_wavenet_forward(model_data, input, config)
            }
        }
        "LSTM" => oracle_lstm_forward(model_data, input, config),
        "ConvNet" => oracle_convnet_forward(model_data, input, config),
        _ => vec![0.0; input.len()],
    }
}

fn is_a2_model(model_data: &NamModelData) -> bool {
    let layers = &model_data.config.layers;
    if layers.is_empty() {
        return false;
    }
    // A2 detection requires:
    // 1. At least one layer array with dilations+channels.
    // 2. head_scale is present (standard WaveNets don't have it).
    // 3. No post-stack head.
    // 4. For multi-array (> 1) models: at least one A2-specific feature.
    let l0 = &layers[0];
    if l0.dilations.is_none() || l0.channels.is_none() {
        return false;
    }
    if model_data.config.head_scale.is_none() {
        return false;
    }
    if let Some(ref head) = model_data.config.head
        && !head.is_null()
    {
        return false;
    }
    // S16.4 (T5.1): condition_dsp models with Tanh activation route through
    // the A1 WaveNet oracle path, matching the production builder's is_a2_shape
    // routing. The A2 path would try to read A2-specific weights (FiLM, head1x1)
    // that don't exist in A1-style models. The A1 oracle supports condition_dsp
    // and multi-channel conditioning via oracle_wavenet_forward.
    if model_data.config.condition_dsp.is_some() {
        for l in layers.iter() {
            if l.activation.as_deref() == Some("Tanh") {
                return false;
            }
        }
        return true;
    }
    // Multi-array: require A2-specific features to avoid misclassifying
    // standard WaveNet models (which also have head_scale + dilations).
    if layers.len() > 1 {
        for l in layers.iter() {
            if let Some(ref raw) = l.layer_raw {
                let has_head1x1 = raw
                    .get("head1x1")
                    .and_then(|h| h.get("active"))
                    .and_then(|a| a.as_bool())
                    .unwrap_or(false);
                if has_head1x1 {
                    return true;
                }
                for key in FILM_KEYS.iter().map(|(k, _)| *k) {
                    if raw
                        .get(key)
                        .and_then(|v| v.get("active"))
                        .and_then(|a| a.as_bool())
                        .unwrap_or(false)
                    {
                        return true;
                    }
                }
                if let Some(act) = raw.get("activation")
                    && (act.is_array() || act.is_object())
                {
                    return true;
                }
            }
        }
        return false;
    }
    true
}

// =============================================================================
// WaveNet Oracle
// =============================================================================

fn oracle_apply_activation(data: &mut [f64], activation: &str, config: &PrecisionConfig) {
    match activation {
        "Tanh" => {
            for v in data.iter_mut() {
                *v = oracle_tanh(*v, config.activation);
            }
        }
        "HardTanh" => {
            for v in data.iter_mut() {
                *v = v.clamp(-1.0, 1.0);
            }
        }
        "FastTanh" => {
            for v in data.iter_mut() {
                *v = oracle_tanh(*v, config.activation);
            }
        }
        "ReLU" => {
            for v in data.iter_mut() {
                *v = v.max(0.0);
            }
        }
        "Sigmoid" => {
            for v in data.iter_mut() {
                *v = oracle_sigmoid(*v, config.activation);
            }
        }
        "SiLU" => {
            for v in data.iter_mut() {
                let s = oracle_sigmoid(*v, config.activation);
                *v *= s;
            }
        }
        "HardSwish" => {
            for v in data.iter_mut() {
                let relu6 = (*v + 3.0).clamp(0.0, 6.0);
                *v = *v * relu6 / 6.0;
            }
        }
        "Softsign" => {
            for v in data.iter_mut() {
                *v /= 1.0 + v.abs();
            }
        }
        "Linear" | "Identity" => {}
        _ => {
            for v in data.iter_mut() {
                *v = oracle_tanh(*v, config.activation);
            }
        }
    }
}

// =============================================================================
// Decomposition
// =============================================================================

#[derive(Debug)]
pub struct DecompositionResult {
    pub label: String,
    pub architecture: String,
    pub esr_f32_vs_f64: f64,
    pub esr_quant_f16c: Option<f64>,
    pub esr_quant_bf16: Option<f64>,
    pub esr_activation: Option<f64>,
    pub esr_accumulation: Option<f64>,
    pub esr_combined: Option<f64>,
}

impl DecompositionResult {
    pub fn esr_quant_f16c_display(&self) -> f64 {
        self.esr_quant_f16c.unwrap_or(0.0)
    }
    pub fn esr_quant_bf16_display(&self) -> f64 {
        self.esr_quant_bf16.unwrap_or(0.0)
    }
    pub fn esr_activation_display(&self) -> f64 {
        self.esr_activation.unwrap_or(0.0)
    }
    pub fn esr_accumulation_display(&self) -> f64 {
        self.esr_accumulation.unwrap_or(0.0)
    }
    pub fn esr_combined_display(&self) -> f64 {
        self.esr_combined.unwrap_or(0.0)
    }
}

pub fn run_decomposition(
    label: &str,
    architecture: &str,
    model_data: &NamModelData,
    production_output: &[f64],
    input_signal: &[f64],
) -> DecompositionResult {
    let oracle_cfg = PrecisionConfig::default();

    let oracle_out = oracle_forward(model_data, input_signal, &oracle_cfg);
    let esr_f32_vs_f64 = compute_esr_f64(&oracle_out, production_output);

    let mut cfg_f16c = oracle_cfg;
    cfg_f16c.weight_precision = WeightPrecision::F16C;
    let out_f16c = oracle_forward(model_data, input_signal, &cfg_f16c);
    let esr_f16c = compute_esr_f64(&oracle_out, &out_f16c);

    let mut cfg_bf16 = oracle_cfg;
    cfg_bf16.weight_precision = WeightPrecision::BF16;
    let out_bf16 = oracle_forward(model_data, input_signal, &cfg_bf16);
    let esr_bf16 = compute_esr_f64(&oracle_out, &out_bf16);

    let mut cfg_act = oracle_cfg;
    cfg_act.activation = ActivationMode::PadeMinimax;
    let out_act = oracle_forward(model_data, input_signal, &cfg_act);
    let esr_act = compute_esr_f64(&oracle_out, &out_act);

    let mut cfg_acc = oracle_cfg;
    cfg_acc.accumulation = AccumulationMode::F32Plain;
    let out_acc = oracle_forward(model_data, input_signal, &cfg_acc);
    let esr_acc = compute_esr_f64(&oracle_out, &out_acc);

    let combined_cfg = PrecisionConfig {
        weight_precision: WeightPrecision::F16C,
        activation: ActivationMode::PadeMinimax,
        accumulation: AccumulationMode::F32Plain,
    };
    let out_combined = oracle_forward(model_data, input_signal, &combined_cfg);
    let esr_combined = compute_esr_f64(&oracle_out, &out_combined);

    DecompositionResult {
        label: label.to_string(),
        architecture: architecture.to_string(),
        esr_f32_vs_f64,
        esr_quant_f16c: Some(esr_f16c),
        esr_quant_bf16: Some(esr_bf16),
        esr_activation: Some(esr_act),
        esr_accumulation: Some(esr_acc),
        esr_combined: Some(esr_combined),
    }
}
