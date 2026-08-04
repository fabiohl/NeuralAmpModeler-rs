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

        self.load_head1x1_weights(weights, pos, total)?;

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

        // 2c. Mixin (conv_out * condition_size elements, applied after conv).
        // Standard A2: condition_size == 1, so conv_out elements.
        // A2 generic: condition_size can be > 1, giving conv_out * condition_size.
        let mixin_count = conv_out * self.condition_size;
        let mixin_w_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            mixin_count,
            total,
            &format!("layer[{i}].mixin_w"),
        )?;
        let mixin_w = AlignedVec::from_vec(mixin_w_f32.to_vec())
            .expect("allocation should succeed for test-sized buffers");

        // 2d. L1x1: bottleneck×channels + channels bias.
        let l1x1_w_count = bottleneck * channels;
        let l1x1_w_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            l1x1_w_count,
            total,
            &format!("layer[{i}].l1x1_w"),
        )?;
        let mut l1x1_w = AlignedVec::new(l1x1_w_count, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        transpose_dense_f32(l1x1_w_f32, &mut l1x1_w, bottleneck, channels);

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

    /// Loads head1x1 projection weights from the stream (if active).
    fn load_head1x1_weights(
        &mut self,
        weights: &[f32],
        pos: &mut usize,
        total: usize,
    ) -> Result<(), String> {
        if !self.head1x1_active {
            return Ok(());
        }
        let channels = self.head_accum_size;

        // S13.2: A2 generic models with condition_size > 1 may have
        // grouped head1x1 (head1x1.groups > 1). Use reduced input dimension.
        let h1_in = self.h1_in_size;
        let h1_w_count = channels * h1_in;
        let h1_w_f32 =
            super::super::set_weights::read_slice(weights, pos, h1_w_count, total, "head1x1_w")?;
        let mut h1_w = AlignedVec::new(h1_w_count, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        transpose_dense_f32(h1_w_f32, &mut h1_w, h1_in, channels);

        let h1_b_f32 =
            super::super::set_weights::read_slice(weights, pos, channels, total, "head1x1_b")?;
        let mut h1_b = AlignedVec::new(channels, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        h1_b.copy_from_slice(h1_b_f32);

        self.head1x1_w = h1_w;
        self.head1x1_b = h1_b;
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
        let head_k = crate::models::a2::params::A2_HEAD_KERNEL_SIZE;
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

            self.head_conv = Some(A2HeadConv::new(head_w, head_b, head_scale, channels));
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
