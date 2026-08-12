// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Weight loading for the dynamic A2 model.
//!
//! Parses a flat f32 weight stream in NAM JSON order and populates
//! the model layers with runtime-dimensioned weights. Supports
//! gating/blending (2× bottleneck), head1x1, and per-layer FiLM.

use crate::math::common::AlignedVec;
use crate::models::a2::gating::GatingMode;
use crate::models::a2::head::A2HeadConv;
use crate::models::a2::layer::A2Layer;
use crate::models::a2::weights_layout::{
    transpose_conv1d_interleaved_4wide, transpose_dense_f32, transpose_head_w,
};

use super::WaveNetA2Dyn;

impl WaveNetA2Dyn {
    /// Loads weights from a flat f32 slice in A2 stream order.
    ///
    /// Convenience wrapper: starts at position 0 and checks exhaustion.
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<(), String> {
        let total = weights.len();
        let mut pos: usize = 0;
        self.load_weights_inner(weights, &mut pos, total)?;
        if pos != total {
            return Err(format!(
                "set_weights: stream has {} unconsumed f32 (consumed {}, total {})",
                total - pos,
                pos,
                total
            ));
        }
        Ok(())
    }

    /// Loads weights starting at `*pos` in the stream.
    ///
    /// Advances `*pos` past the consumed weights. Does NOT check for
    /// exhaustion — the caller is responsible for managing the total
    /// weight stream (used for multi-array cascade loading).
    pub(crate) fn load_weights_inner(
        &mut self,
        weights: &[f32],
        pos: &mut usize,
        total: usize,
    ) -> Result<(), String> {
        self.load_rechannel_weights(weights, pos, total)?;

        let mut layers = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let layer = self.load_per_layer_weights(weights, pos, total, i)?;
            layers.push(layer);
        }

        self.load_head_conv_and_scale(weights, pos, total)?;

        self.layers = layers;

        Ok(())
    }

    /// Loads rechannel weights from the stream: `Conv1x1(input_channels → channels)` (no bias).
    fn load_rechannel_weights(
        &mut self,
        weights: &[f32],
        pos: &mut usize,
        total: usize,
    ) -> Result<(), String> {
        let channels = self.channels;
        let in_ch = self.input_channels;
        let rw_count = in_ch * channels;
        let rw_f32 =
            super::super::set_weights::read_slice(weights, pos, rw_count, total, "rechannel_w")?;
        self.rechannel_w_f32 = AlignedVec::new(rw_count, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        self.rechannel_w_f32.copy_from_slice(rw_f32);
        Ok(())
    }

    /// Loads a single layer's weights (conv, mixin, l1x1, optional FiLM).
    fn load_per_layer_weights(
        &mut self,
        weights: &[f32],
        pos: &mut usize,
        total: usize,
        i: usize,
    ) -> Result<A2Layer, String> {
        let channels = self.channels;
        let bottleneck = self.bottleneck;
        let ksize = self.kernel_sizes[i];
        let dilation = self.dilations[i];
        let use_gating = self.gating_modes[i] == GatingMode::Gated
            || self.gating_modes[i] == GatingMode::Blended;
        let conv_out = if use_gating {
            bottleneck * 2
        } else {
            bottleneck
        };

        // 2a. Dilated conv weights — interleave-4-wide.
        let conv_w_count = channels * conv_out * ksize;
        let conv_w_padded = conv_out.div_ceil(4) * 4 * channels * ksize;
        let conv_w_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            conv_w_count,
            total,
            &format!("layer[{i}].conv_w"),
        )?;
        let mut conv_w = AlignedVec::new(conv_w_padded, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        transpose_conv1d_interleaved_4wide(conv_w_f32, &mut conv_w, channels, conv_out, ksize);

        // 2b. Conv bias.
        let conv_b_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            conv_out,
            total,
            &format!("layer[{i}].conv_b"),
        )?;
        let conv_b = AlignedVec::from_vec(conv_b_f32.to_vec())
            .expect("allocation should succeed for test-sized buffers");

        let conv = crate::models::a2::conv1d::A2Conv1d::new(
            conv_w, conv_b, true, dilation, channels, conv_out, ksize,
        );

        // 2c. Mixin (group-aware).
        // C++ Conv1x1 with groups: the weight stream stores only block-diagonal
        // entries — G × out_per_group × in_per_group. Compact storage:
        // [out_ch × in_per_group] row-major; group is determined by output index.
        //
        // After reorder: transpose from row-major [out_ch][in_pg] to
        // col-major [in_pg][out_ch] so the hot path can use contiguous
        // 8-wide SIMD loads across output channels with broadcast condition
        // (T4.3 vectorization).
        let mg: u32 = self.mixin_groups.max(1);
        let mixin_in_pg = self.condition_size / mg as usize;
        let mixin_out_per_g = conv_out / mg as usize;
        let mixin_count = conv_out * mixin_in_pg;
        let mixin_w_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            mixin_count,
            total,
            &format!("layer[{i}].mixin_w"),
        )?;
        let mut mixin_w_row = AlignedVec::new(mixin_count, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        // Reorder from group-major to per-output-channel-major:
        // Stream: for each group g, then oc, then ic.
        // Storage: mixin_w_row[oc * in_per_group + ic_local].
        {
            let mut src_idx = 0usize;
            for g in 0..mg as usize {
                let out_start = g * mixin_out_per_g;
                for oc in out_start..out_start + mixin_out_per_g {
                    let dst_base = oc * mixin_in_pg;
                    for ic in 0..mixin_in_pg {
                        mixin_w_row[dst_base + ic] = mixin_w_f32[src_idx];
                        src_idx += 1;
                    }
                }
            }
        }
        // Transpose from row-major [out_per_g][in_pg] to col-major
        // [in_pg][out_per_g] within each group block, so the hot path can
        // use contiguous 8-wide SIMD loads across output channels with
        // broadcast condition (T4.3).
        let mut mixin_w = AlignedVec::new(mixin_count, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        for g in 0..mg as usize {
            let group_base = g * mixin_out_per_g * mixin_in_pg;
            let out_start = g * mixin_out_per_g;
            for ic in 0..mixin_in_pg {
                for oc in 0..mixin_out_per_g {
                    mixin_w[group_base + ic * mixin_out_per_g + oc] =
                        mixin_w_row[(out_start + oc) * mixin_in_pg + ic];
                }
            }
        }

        // 2d. L1x1 (group-aware).
        // Groups=1: dense col-major `[bottleneck][channels]` (backward compat).
        // Groups>1: compact `[channels × in_per_group]` row-major per output channel.
        let lg: u32 = self.l1x1_groups.max(1);
        let l1x1_in_pg = bottleneck / lg as usize;
        let l1x1_out_per_g = channels / lg as usize;
        let l1x1_w_count = if lg > 1 {
            channels * l1x1_in_pg
        } else {
            bottleneck * channels
        };
        let l1x1_w_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            l1x1_w_count,
            total,
            &format!("layer[{i}].l1x1_w"),
        )?;
        let l1x1_w = if lg > 1 {
            let mut w = AlignedVec::new(l1x1_w_count, 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
            let mut src_idx = 0usize;
            for g in 0..lg as usize {
                let out_start = g * l1x1_out_per_g;
                for oc in out_start..out_start + l1x1_out_per_g {
                    let dst_base = oc * l1x1_in_pg;
                    for ic in 0..l1x1_in_pg {
                        w[dst_base + ic] = l1x1_w_f32[src_idx];
                        src_idx += 1;
                    }
                }
            }
            w
        } else {
            let mut w = AlignedVec::new(l1x1_w_count, 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
            transpose_dense_f32(l1x1_w_f32, &mut w, bottleneck, channels);
            w
        };

        let l1x1_b_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            channels,
            total,
            &format!("layer[{i}].l1x1_b"),
        )?;
        let l1x1_b = AlignedVec::from_vec(l1x1_b_f32.to_vec())
            .expect("allocation should succeed for test-sized buffers");

        let mut layer = A2Layer::new_dyn(
            conv,
            mixin_w,
            l1x1_w,
            l1x1_b,
            channels,
            bottleneck,
            self.condition_size,
        );
        layer.mixin_groups = mg;
        layer.l1x1_groups = lg;
        debug_assert_eq!(
            layer.l1x1_w.len(),
            if lg > 1 {
                channels * (bottleneck / lg as usize)
            } else {
                bottleneck * channels
            },
            "l1x1_w dimension mismatch: len={}, channels={}, bottleneck={}, groups={}",
            layer.l1x1_w.len(),
            channels,
            bottleneck,
            lg
        );
        debug_assert_eq!(
            layer.mixin_w.len(),
            conv_out * (self.condition_size / mg.max(1) as usize),
            "mixin_w dimension mismatch: len={}, expected conv_out={} * (condition_size={} / groups={}) = {}",
            layer.mixin_w.len(),
            conv_out,
            self.condition_size,
            mg.max(1),
            conv_out * (self.condition_size / mg.max(1) as usize)
        );

        // Load per-layer head1x1 projection weights (C++ `Layer::set_weights_`
        // loads `_head1x1` immediately after `_layer1x1` before FiLM).
        self.load_head1x1_for_layer(&mut layer, weights, pos, total, i, bottleneck)?;

        // FiLM layers (if active in layer_raw JSON) — read weights after l1x1 bias.
        if let Some(ref raw) = self.layer_raw {
            let configs = super::super::set_weights::parse_film_configs(raw);
            super::super::set_weights::load_film_for_layer(
                &mut layer,
                &configs,
                channels,
                self.condition_size,
                self.head_accum_size.max(1),
                weights,
                pos,
                total,
                i,
            )?;
        }

        Ok(layer)
    }

    /// Loads per-layer head1x1 projection weights and bias into the layer.
    ///
    /// C++ `Layer::set_weights_` loads `_head1x1` immediately after `_layer1x1`
    /// before FiLM. Supports both dense (groups=1) and grouped (>1) layouts.
    fn load_head1x1_for_layer(
        &self,
        layer: &mut A2Layer,
        weights: &[f32],
        pos: &mut usize,
        total: usize,
        i: usize,
        bottleneck: usize,
    ) -> Result<(), String> {
        if !self.head1x1_active {
            return Ok(());
        }
        let h1_in = self.head1x1_h1_in;
        let h1_out = self.head_accum_size;
        let h1_groups = bottleneck.checked_div(h1_in).unwrap_or(1);
        let h1_is_grouped = h1_groups > 1;
        let h1_w_count = h1_out * h1_in;
        let h1_w_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            h1_w_count,
            total,
            &format!("layer[{i}].head1x1_w"),
        )?;
        let h1_w = if h1_is_grouped {
            let mut w = AlignedVec::new(h1_w_count, 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
            let out_per_g = h1_out / h1_groups;
            let in_per_g = h1_in;
            let mut src_idx = 0usize;
            for g in 0..h1_groups {
                let out_start = g * out_per_g;
                for oc in out_start..out_start + out_per_g {
                    let dst_base = oc * in_per_g;
                    for ic in 0..in_per_g {
                        w[dst_base + ic] = h1_w_f32[src_idx];
                        src_idx += 1;
                    }
                }
            }
            w
        } else {
            let mut w = AlignedVec::new(h1_w_count, 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
            transpose_dense_f32(h1_w_f32, &mut w, h1_in, h1_out);
            w
        };
        let h1_b_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            h1_out,
            total,
            &format!("layer[{i}].head1x1_b"),
        )?;
        let mut h1_b = AlignedVec::new(h1_out, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        h1_b.copy_from_slice(h1_b_f32);
        debug_assert_eq!(
            h1_w.len(),
            h1_out * h1_in,
            "head1x1_w dimension mismatch: len={}, head_accum_size={}, h1_in={}, groups={}",
            h1_w.len(),
            h1_out,
            h1_in,
            h1_groups
        );
        layer.head1x1_active = true;
        layer.head1x1_w = h1_w;
        layer.head1x1_b = h1_b;
        Ok(())
    }

    /// Loads head conv weights (K=16), bias, and head scale from the stream.
    ///
    /// For head_size == 1, builds a mono `A2HeadConv`.
    /// For head_size > 1 (multi-array cascade arrays with
    /// multi-channel output), loads a full Conv1D per output channel:
    /// `head_size × K × head_accum_size` weights + `head_size` bias + `head_size` scale.
    fn load_head_conv_and_scale(
        &mut self,
        weights: &[f32],
        pos: &mut usize,
        total: usize,
    ) -> Result<(), String> {
        let channels = self.head_accum_size;
        let head_k = self.head_kernel_size;
        let head_size = self.head_size;

        if head_size == 1 {
            let head_w_f32 = super::super::set_weights::read_slice(
                weights,
                pos,
                head_k * channels,
                total,
                "head_w",
            )?;
            let mut head_w = AlignedVec::new(head_k * channels, 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
            transpose_head_w(head_w_f32, &mut head_w, channels, head_k);

            let head_b = {
                let s = super::super::set_weights::read_slice(weights, pos, 1, total, "head_b")?;
                if !s[0].is_finite() {
                    return Err(format!(
                        "set_weights: head_b is not finite (value: {:e})",
                        s[0]
                    ));
                }
                s[0]
            };

            let head_scale = {
                let s =
                    super::super::set_weights::read_slice(weights, pos, 1, total, "head_scale")?;
                if !s[0].is_finite() {
                    return Err(format!(
                        "set_weights: head_scale is not finite (value: {:e})",
                        s[0]
                    ));
                }
                s[0]
            };

            self.head_conv = Some(A2HeadConv::new_with_kernel(
                head_w, head_b, head_scale, channels, head_k,
            ));
        } else {
            // Multi-channel head: full Conv1D per output channel.
            let per_oc_w_count = head_k * channels;
            let total_w_count = head_size * per_oc_w_count;
            let head_w_f32 = super::super::set_weights::read_slice(
                weights,
                pos,
                total_w_count,
                total,
                "head_rechannel_w",
            )?;
            let mut head_w = AlignedVec::new(total_w_count, 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
            for oc in 0..head_size {
                let src = &head_w_f32[oc * per_oc_w_count..(oc + 1) * per_oc_w_count];
                let dst = &mut head_w[oc * per_oc_w_count..(oc + 1) * per_oc_w_count];
                transpose_head_w(src, dst, channels, head_k);
            }

            let head_b_f32 = super::super::set_weights::read_slice(
                weights,
                pos,
                head_size,
                total,
                "head_rechannel_b",
            )?;
            for &b in head_b_f32 {
                if !b.is_finite() {
                    return Err(format!(
                        "set_weights: head_rechannel_b contains non-finite value (value: {:e})",
                        b
                    ));
                }
            }
            let mut head_b = AlignedVec::new(head_size, 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
            head_b.copy_from_slice(head_b_f32);

            let head_scale_f32 = super::super::set_weights::read_slice(
                weights,
                pos,
                head_size,
                total,
                "head_rechannel_scale",
            )?;
            for &s in head_scale_f32 {
                if !s.is_finite() {
                    return Err(format!(
                        "set_weights: head_rechannel_scale contains non-finite value (value: {:e})",
                        s
                    ));
                }
            }
            let mut head_scale = AlignedVec::new(head_size, 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
            head_scale.copy_from_slice(head_scale_f32);

            self.head_rechannel_w = head_w;
            self.head_rechannel_b = head_b;
            self.head_rechannel_scale = head_scale;
        }

        Ok(())
    }
}
