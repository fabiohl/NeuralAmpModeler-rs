// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Weight loading for WaveNet A2 models.
//!
//! Parses a flat f32 weight stream in NAM JSON order and populates the
//! model layers with quantized weights and col-major-per-tap layouts.
//!
//! When the layer JSON contains active FiLM entries, FiLM weights are read
//! from the stream after each layer's standard weights.

use super::WaveNetA2;
use crate::math::common::AlignedVec;
use crate::models::a2::conv1d_ch3::A2Conv1dCh3;
use crate::models::a2::conv1d_ch8::A2Conv1dCh8;
use crate::models::a2::film::{FiLMConfig, FiLMLayer};
use crate::models::a2::head::A2HeadConv;
use crate::models::a2::layer::A2Layer;
use crate::models::a2::params::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_NUM_LAYERS,
};
use crate::models::a2::weights_layout::{
    FILM_KEYS, film_bias_count, film_bias_count_generic, film_weight_count,
    film_weight_count_generic, transpose_conv1d_interleaved_4wide, transpose_dense_f32,
    transpose_head_w,
};

impl<const CH: usize> WaveNetA2<CH> {
    /// Loads weights from a flat f32 slice in the exact A2 stream order.
    ///
    /// ## Weight order (mirrors `a2_fast.cpp:196-282`)
    ///
    /// 1. `_rechannel`: weights `CH` f32 (no bias — matches C++ A2FastModel)
    /// 2. Per layer 0..22:
    ///    - `_conv`: weights `CH*CH*K` f32 + bias `CH` f32
    ///    - `_input_mixin`: weights `CH` f32 (no bias)
    ///    - `_layer1x1`: weights `CH*CH` f32 (col-major) + bias `CH` f32
    /// 3. `_head_rechannel`: conv k=16 weights `16*CH` f32 + head_bias `1` f32
    /// 4. `head_scale`: last f32 in the stream
    ///
    /// ## Acceptance criteria
    /// - Calls `verify_exhaustion()` — consumed count must equal `weights.len()`.
    /// - Returns a clear error if the weight stream is shorter or longer than expected.
    #[expect(
        clippy::too_many_lines,
        reason = "Function body contains exhaustive A2 kernel-size dispatch table — splitting into sub-functions would scatter related logic"
    )]
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<(), String> {
        let total = weights.len();
        let mut pos: usize = 0;

        // ── 1. Rechannel: Conv1x1(1 → CH) (no bias) ─────────────────────
        let rw_f32 = read_slice(weights, &mut pos, CH, total, "rechannel_w")?;
        let rechannel_w = AlignedVec::from_vec(rw_f32.to_vec())
            .map_err(|e| format!("A2 weight buffer allocation failed: {e}"))?;

        // ── 2. Per-layer weights ──────────────────────────────────────────
        let mut layers = Vec::with_capacity(A2_NUM_LAYERS);

        for i in 0..A2_NUM_LAYERS {
            let ksize = A2_KERNEL_SIZES[i];
            let dilation = A2_DILATIONS[i];
            let conv_w_count = CH * CH * ksize;
            let num_blocks = CH.div_ceil(4);
            let conv_w_padded = num_blocks * 4 * CH * ksize;

            // 2a. Dilated conv weights: read CH×CH×K, store padded interleaved 4-wide.
            // For CH=8 we also keep a f32 copy for the col-major-per-tap path.
            let conv_w_f32 = read_slice(
                weights,
                &mut pos,
                conv_w_count,
                total,
                &format!("layer[{i}].conv_w"),
            )?;
            // Owned copy: needed for CH=8 col-major-per-tap (re-indexed, not interleaved-4-wide).
            let conv_w_f32_owned: Vec<f32> = conv_w_f32.to_vec();
            // Interleave-4-wide f32 for the fallback conv path.
            let mut conv_w = AlignedVec::new(conv_w_padded, 0.0f32)
                .map_err(|e| format!("A2 weight buffer allocation failed: {e}"))?;
            transpose_conv1d_interleaved_4wide(conv_w_f32, &mut conv_w, CH, CH, ksize);

            // 2b. Conv bias (f32, one per output channel).
            let conv_b_f32 =
                read_slice(weights, &mut pos, CH, total, &format!("layer[{i}].conv_b"))?;
            let conv_b = AlignedVec::from_vec(conv_b_f32.to_vec())
                .map_err(|e| format!("A2 weight buffer allocation failed: {e}"))?;

            // Build the scalar/SIMD fallback conv (interleaved-4-wide f32 weights).
            let conv = crate::models::a2::conv1d::A2Conv1d::new(
                conv_w,
                conv_b.clone(),
                true,
                dilation,
                CH,
                CH,
                ksize,
            );

            // Optional col-major-per-tap f32 conv for CH=3 and CH=8.
            // Uses the original (non-interleaved) f32 weights for SIMD-friendly access.
            let conv_ch = match CH {
                3 => {
                    let ch3 = A2Conv1dCh3::new(conv_w_f32, CH, CH, ksize, dilation, conv_b_f32)
                        .map_err(|e| format!("{e}"))?;
                    Some(crate::models::a2::layer::A2ConvCh::Ch3(ch3))
                }
                8 => {
                    let ch8 = A2Conv1dCh8::new(&conv_w_f32_owned, CH, CH, ksize, dilation, &conv_b)
                        .map_err(|e| format!("{e}"))?;
                    Some(crate::models::a2::layer::A2ConvCh::Ch8(ch8))
                }
                _ => None,
            };

            // 2c. Input mixin: per-channel scalar weights (f32, no bias).
            // Applied as `z[c] += mixin[c] * input` after the dilated conv.
            let mixin_w_f32 =
                read_slice(weights, &mut pos, CH, total, &format!("layer[{i}].mixin_w"))?;
            let mixin_w = AlignedVec::from_vec(mixin_w_f32.to_vec())
                .map_err(|e| format!("A2 weight buffer allocation failed: {e}"))?;

            // 2d. Layer 1×1 projection: CH×CH dense matrix (f32, col-major).
            // NAM JSON stores row-major; we transpose to col-major for SIMD dot products.
            let l1x1_w_f32 = read_slice(
                weights,
                &mut pos,
                CH * CH,
                total,
                &format!("layer[{i}].l1x1_w"),
            )?;
            let mut l1x1_w = AlignedVec::new(CH * CH, 0.0f32)
                .map_err(|e| format!("A2 weight buffer allocation failed: {e}"))?;
            transpose_dense_f32(l1x1_w_f32, &mut l1x1_w, CH, CH);

            // 2e. Layer 1×1 bias: one f32 per output channel.
            let l1x1_b_f32 =
                read_slice(weights, &mut pos, CH, total, &format!("layer[{i}].l1x1_b"))?;
            let l1x1_b = AlignedVec::from_vec(l1x1_b_f32.to_vec())
                .map_err(|e| format!("A2 weight buffer allocation failed: {e}"))?;

            // Assemble the layer: priority is ch3_conv > ch8_conv > scalar fallback.
            let mut layer = A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b);
            if let Some(conv_ch) = conv_ch {
                layer.conv_ch = Some(conv_ch);
            }

            // 2f. FiLM layers (if active in layer_raw JSON) — read weights after l1x1 bias.
            if let Some(ref raw) = self.layer_raw {
                let configs = parse_film_configs(raw);
                load_film_for_layer(&mut layer, &configs, CH, 1, 1, weights, &mut pos, total, i)?;
            }

            layers.push(layer);
        }

        // ── 3. Head rechannel: Conv1D(CH → 1, K=16, bias) ─────────────────
        let head_w_f32 = read_slice(weights, &mut pos, A2_HEAD_KERNEL_SIZE * CH, total, "head_w")?;
        let mut head_w = AlignedVec::new(A2_HEAD_KERNEL_SIZE * CH, 0.0f32)
            .map_err(|e| format!("A2 weight buffer allocation failed: {e}"))?;
        transpose_head_w(head_w_f32, &mut head_w, CH, A2_HEAD_KERNEL_SIZE);

        let head_b = {
            let s = read_slice(weights, &mut pos, 1, total, "head_b")?;
            if !s[0].is_finite() {
                return Err(format!(
                    "set_weights: head_b is not finite (value: {:e})",
                    s[0]
                ));
            }
            s[0]
        };

        // ── 4. Head scale (last float) ─────────────────────────────────────
        let head_scale = {
            let s = read_slice(weights, &mut pos, 1, total, "head_scale")?;
            if !s[0].is_finite() {
                return Err(format!(
                    "set_weights: head_scale is not finite (value: {:e})",
                    s[0]
                ));
            }
            s[0]
        };

        // ── 5. Exhaustion check ────────────────────────────────────────────
        if pos != total {
            return Err(format!(
                "set_weights: stream has {} unconsumed f32 after loading all weights (consumed {}, total {})",
                total - pos,
                pos,
                total
            ));
        }

        // ── 6. Commit to self (all-or-nothing) ──────────────────────────────
        self.rechannel_w_f32 = rechannel_w;
        self.layers = layers;
        self.head_conv = Some(A2HeadConv::new(head_w, head_b, head_scale, CH));

        Ok(())
    }
}

// =============================================================================
// Private helpers for set_weights
// =============================================================================

/// Reads a contiguous slice of `n` f32 values from `weights[pos..]`,
/// advancing `pos`. Returns an error with the label if out of bounds.
#[inline]
pub(crate) fn read_slice<'a>(
    weights: &'a [f32],
    pos: &mut usize,
    n: usize,
    total: usize,
    label: &str,
) -> Result<&'a [f32], String> {
    if *pos + n > total {
        return Err(format!(
            "set_weights: stream exhausted at position {} (need {} for \"{}\", total {})",
            *pos, n, label, total
        ));
    }
    let slice = &weights[*pos..*pos + n];
    *pos += n;
    Ok(slice)
}

// =============================================================================
// FiLM loading helpers
// =============================================================================

pub(crate) fn parse_single_film_config(raw: &serde_json::Value, key: &str) -> FiLMConfig {
    let obj = match raw.get(key).and_then(|v| v.as_object()) {
        Some(o) => o,
        None => return FiLMConfig::default(),
    };
    FiLMConfig {
        active: obj.get("active").and_then(|a| a.as_bool()).unwrap_or(false),
        shift: obj.get("shift").and_then(|s| s.as_bool()).unwrap_or(true),
        groups: obj
            .get("groups")
            .and_then(|g| g.as_u64())
            .map(|g| g as u32)
            .unwrap_or(1),
    }
}

pub(crate) fn parse_film_configs(raw: &serde_json::Value) -> [FiLMConfig; 8] {
    let mut configs = [FiLMConfig::default(); 8];
    for &(key, idx) in FILM_KEYS {
        configs[idx] = parse_single_film_config(raw, key);
    }
    configs
}

pub(crate) fn set_layer_film(
    layer: &mut A2Layer,
    _config: &FiLMConfig,
    idx: usize,
    film: FiLMLayer,
) -> Result<(), String> {
    match idx {
        0 => layer.conv_pre_film = Some(film),
        1 => layer.conv_post_film = Some(film),
        2 => layer.input_mixin_pre_film = Some(film),
        3 => layer.input_mixin_post_film = Some(film),
        4 => layer.activation_pre_film = Some(film),
        5 => layer.activation_post_film = Some(film),
        6 => layer.layer1x1_post_film = Some(film),
        7 => layer.head1x1_post_film = Some(film),
        _ => return Err(format!("FiLM slot index {} out of range (0-7)", idx)),
    }
    Ok(())
}

/// Convenience wrapper — see `weights_layout::film_weight_count`.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retained for future integration when gated feature is enabled"
    )
)]
pub(crate) fn film_weight_count_cfg(
    config: &FiLMConfig,
    cond_size: usize,
    channels: usize,
) -> usize {
    film_weight_count(config.groups, cond_size, channels, config.shift)
}

/// Convenience wrapper — see `weights_layout::film_bias_count`.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retained for future integration when gated feature is enabled"
    )
)]
pub(crate) fn film_bias_count_cfg(config: &FiLMConfig, channels: usize) -> usize {
    film_bias_count(channels, config.shift)
}

#[expect(
    clippy::too_many_arguments,
    reason = "A2 model weight-setter requiring many dimension parameters to safely map weight slices to layer buffers"
)]
pub(crate) fn load_film_for_layer(
    layer: &mut A2Layer,
    configs: &[FiLMConfig; 8],
    channels: usize,
    cond_size: usize,
    head_channels: usize,
    weights: &[f32],
    pos: &mut usize,
    total: usize,
    layer_idx: usize,
) -> Result<(), String> {
    for (idx, config) in configs.iter().enumerate() {
        if !config.active {
            continue;
        }
        let film_channels = match idx {
            2 => cond_size,
            7 => head_channels,
            _ => channels,
        };
        let (w_count, b_count) = if cond_size > 1 {
            // A2 generic: integer-safe formula with channels-sized bias
            (
                film_weight_count_generic(config.groups, cond_size, film_channels, config.shift),
                film_bias_count_generic(film_channels, config.shift),
            )
        } else {
            (
                film_weight_count(config.groups, cond_size, film_channels, config.shift),
                film_bias_count(film_channels, config.shift),
            )
        };
        let key = FILM_KEYS[idx].0;

        let film_w = read_slice(
            weights,
            pos,
            w_count,
            total,
            &format!("layer[{layer_idx}].{key}.w"),
        )?;
        let film_b = read_slice(
            weights,
            pos,
            b_count,
            total,
            &format!("layer[{layer_idx}].{key}.b"),
        )?;

        let film_layer = FiLMLayer::load(
            *config,
            cond_size,
            film_channels,
            film_w.to_vec(),
            film_b.to_vec(),
        )
        .map_err(|e| format!("{e}"))?;
        set_layer_film(layer, config, idx, film_layer)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "set_weights_test.rs"]
mod set_weights_test;
