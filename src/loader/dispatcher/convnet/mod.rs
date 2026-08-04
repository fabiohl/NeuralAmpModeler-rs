// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! ConvNet model builder — reads weights from `NamModelData` and constructs a `ConvNetModel`.

use super::WeightCursor;
use crate::loader::nam_json::{ConvNetFormat, NamModelData, get_convnet_topology};
use crate::math::common::AlignedVec;
use crate::models::StaticModel;
use crate::models::convnet::{ConvNetBlock, ConvNetModel, LinearHead};
use crate::models::wavenet::PostStackHead;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;
use crate::models::wavenet::post_stack_head::parse_activation;
use anyhow::Context;
use log::{info, warn};

use crate::loader::dispatcher::checked_arith;
use crate::loader::dispatcher::wavenet::layout;

/// Builds a `Box<StaticModel::ConvNet>` from the parsed model data.
///
/// Supports two formats:
/// - **Layers** (nam-rs native): per-block config via `layers` array,
///   pre-fused BatchNorm scale/offset, optional PostStackHead + head_scale.
/// - **FlatCpp**: C++ NAMCore flat config (`channels` scalar, `dilations` array,
///   `batchnorm` bool), raw BatchNorm params, separate linear head.
pub(crate) fn build_convnet(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    let topo = get_convnet_topology(data).context("ConvNet topology not detectable")?;

    match topo.format {
        ConvNetFormat::FlatCpp => build_convnet_flat_cpp(data, &topo),
        ConvNetFormat::Layers => build_convnet_layers(data, &topo),
    }
}

/// Builds ConvNet from the C++ flat format (scalar channels, global dilations, batchnorm).
fn build_convnet_flat_cpp(
    data: &NamModelData,
    topo: &crate::loader::nam_json::ConvNetTopology,
) -> anyhow::Result<Box<StaticModel>> {
    let batchnorm = topo.batchnorm.unwrap_or(false);
    let activation_str = topo.activation.as_deref().unwrap_or("Tanh");
    let activation = parse_activation(activation_str);

    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);
    let mut blocks = Vec::with_capacity(topo.num_blocks);
    let mut total_rf = 0usize;

    for (alloc_num, i) in (0..topo.num_blocks).enumerate() {
        let in_ch = if i == 0 { 1 } else { topo.channels[i - 1] };
        let out_ch = topo.channels[i];
        let kernel = topo.kernel_sizes[i];
        let dilation = *topo.dilations[i]
            .first()
            .context(format!("ConvNet block {i}: empty dilations list"))?;

        let do_bias = !batchnorm;

        let mut block = ConvNetBlock::new(
            in_ch,
            out_ch,
            kernel,
            dilation,
            do_bias,
            activation.clone(),
            alloc_num,
        )
        .map_err(|e| anyhow::anyhow!("ConvNet block {i}: failed to create: {e}"))?;

        let total_conv = out_ch * in_ch * kernel;
        let raw_conv = cursor.read_slice(total_conv)?;

        let padded_total = (out_ch.div_ceil(4)) * kernel * in_ch * 4;
        let mut interleaved = AlignedVec::new(padded_total, 0.0f32)
            .map_err(|e| anyhow::anyhow!("ConvNet block {i}: {e}"))?;

        transpose_cpp_conv_to_interleaved(raw_conv, &mut interleaved, out_ch, in_ch, kernel);
        block.set_conv_weights(&interleaved);

        if batchnorm {
            let running_mean = cursor.read_slice(out_ch)?;
            let running_var = cursor.read_slice(out_ch)?;
            let gamma = cursor.read_slice(out_ch)?;
            let beta = cursor.read_slice(out_ch)?;
            let eps = cursor.read_f32_finite()?;

            let mut scale = vec![0.0f32; out_ch];
            let mut offset = vec![0.0f32; out_ch];
            for c in 0..out_ch {
                scale[c] = gamma[c] / (eps + running_var[c]).sqrt();
                offset[c] = beta[c] - scale[c] * running_mean[c];
            }
            block.set_bn_params(&scale, &offset)?;
        } else {
            let bias_raw = cursor.read_slice(out_ch)?;
            block.set_conv_bias(bias_raw);
            let scale: Vec<f32> = vec![1.0; out_ch];
            let offset: Vec<f32> = vec![0.0; out_ch];
            block.set_bn_params(&scale, &offset)?;
        }

        total_rf += (kernel - 1) * dilation;
        blocks.push(block);
    }

    let last_out_ch = *topo.channels.last().context("ConvNet: no blocks")?;
    let out_ch = 1usize;

    let head_weight_raw = cursor.read_slice(out_ch * last_out_ch)?;
    let head_bias_raw = cursor.read_slice(out_ch)?;

    let mut head_weight = AlignedVec::new(out_ch * last_out_ch, 0.0f32)
        .map_err(|e| anyhow::anyhow!("head_weight alloc: {e}"))?;
    let mut head_bias =
        AlignedVec::new(out_ch, 0.0f32).map_err(|e| anyhow::anyhow!("head_bias alloc: {e}"))?;
    head_bias.copy_from_slice(head_bias_raw);

    transpose_cpp_head_to_row_major(head_weight_raw, &mut head_weight, last_out_ch, out_ch);

    let linear_head = Some(LinearHead {
        weight: head_weight,
        bias: head_bias,
        in_ch: last_out_ch,
        out_ch,
    });

    cursor.verify_exhausted()?;

    // head_scale = 1.0: C++ flat format has no head_scale; gain is identity.
    let head_scale: f32 = 1.0;

    if let Some(cfg_head_scale) = data.config.head_scale
        && (cfg_head_scale - 1.0).abs() > f32::EPSILON
    {
        warn!(
            "[Dispatcher] ConvNet FlatCpp: config declares head_scale={}, \
             but NAMcore does not support head_scale for ConvNet. \
             Forcing head_scale=1.0 for strict parity with C++.",
            cfg_head_scale
        );
    }

    let head_output_scratch = AlignedVec::new(out_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?;

    let max_scratch_ch = blocks.iter().map(|b| b.conv.out_ch).max().unwrap_or(1);
    let scratch_a = AlignedVec::new(max_scratch_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?;
    let scratch_b = AlignedVec::new(max_scratch_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?;

    let model = ConvNetModel {
        blocks,
        head_scale,
        receptive_field_size: total_rf,
        post_stack_head: None,
        head_output_scratch,
        scratch_a,
        scratch_b,
        prewarm_on_reset: true,
        linear_head,
    };

    let bn_label = if batchnorm { "with BN" } else { "no BN" };
    info!(
        "[Dispatcher] ConvNet FlatCpp built — num_blocks={}, {bn_label}, weights={}",
        topo.num_blocks,
        data.weights.len()
    );

    Ok(Box::new(StaticModel::ConvNet(Box::new(model))))
}

/// Transposes C++ column-major, kernel-interleaved conv weights to Rust
/// interleaved 4-wide format.
///
/// C++ layout (groups=1, kernel=2):
///   for out_i in 0..out_ch:
///     for in_j in 0..in_ch:
///       src[(out_i * in_ch + in_j) * 2 + 0] = tap0::weight(out_i, in_j)
///       src[(out_i * in_ch + in_j) * 2 + 1] = tap1::weight(out_i, in_j)
///
/// Rust interleaved 4-wide:
///   for block in 0..num_blocks(max=4):
///     for tap in 0..kernel:
///       for in_j in 0..in_ch:
///         for lane in 0..4:
///           dst[block * k * in_ch * 4 + tap * in_ch * 4 + in_j * 4 + lane]
///           = src[tap position at (out=block*4+lane, in=in_j)]
fn transpose_cpp_conv_to_interleaved(
    src: &[f32],
    dst: &mut [f32],
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
) {
    let num_blocks = out_ch.div_ceil(4);
    for block in 0..num_blocks {
        for tap in 0..kernel {
            for in_j in 0..in_ch {
                for lane in 0..4 {
                    let out_i = block * 4 + lane;
                    let dst_idx = block * kernel * in_ch * 4 + tap * in_ch * 4 + in_j * 4 + lane;
                    if out_i < out_ch {
                        let src_idx = (out_i * in_ch + in_j) * kernel + tap;
                        dst[dst_idx] = src[src_idx];
                    } else {
                        dst[dst_idx] = 0.0;
                    }
                }
            }
        }
    }
}

/// Transposes C++ head weights to Rust row-major layout.
///
/// C++ `_Head` writes serialized weights in row-major order:
///   for i in 0..out_ch:
///     for j in 0..in_ch:
///       weight(i, j) = *(weights++)
///
/// Stored to Eigen column-major [out_ch × in_ch] but the serialized
/// order is row-major: src[i * in_ch + j] = weight(i, j).
///
/// Rust LinearHead also uses row-major: dst[i * in_ch + j] = weight(i, j).
/// This is an identity copy.
fn transpose_cpp_head_to_row_major(src: &[f32], dst: &mut [f32], _in_ch: usize, _out_ch: usize) {
    dst[..src.len()].copy_from_slice(src);
}

/// Builds ConvNet from the nam-rs `layers` format (existing behavior).
fn build_convnet_layers(
    data: &NamModelData,
    topo: &crate::loader::nam_json::ConvNetTopology,
) -> anyhow::Result<Box<StaticModel>> {
    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);
    let mut blocks = Vec::with_capacity(topo.num_blocks);

    let mut total_rf = 0usize;

    for (alloc_num, i) in (0..topo.num_blocks).enumerate() {
        let in_ch = if i == 0 { 1 } else { topo.channels[i - 1] };
        let out_ch = topo.channels[i];
        let kernel = topo.kernel_sizes[i];
        let dilations = &topo.dilations[i];

        let dilation = *dilations
            .first()
            .context(format!("ConvNet block {i}: empty dilations list"))?;

        let do_bias = true;

        let activation_str = data
            .config
            .layers
            .get(i)
            .and_then(|l| l.activation.as_deref())
            .unwrap_or("Tanh");
        let activation = parse_activation(activation_str);

        let mut block = ConvNetBlock::new(
            in_ch, out_ch, kernel, dilation, do_bias, activation, alloc_num,
        )
        .map_err(|e| anyhow::anyhow!("ConvNet block {i}: failed to create: {e}"))?;

        let interleaved = cursor.is_interleaved4();
        let num_blocks = out_ch.div_ceil(4);
        let padded_total = checked_arith::checked_conv_padded_total(num_blocks, 4, in_ch, kernel)?;

        if interleaved {
            let raw = cursor.read_slice(padded_total)?;
            block.set_conv_weights(raw);
        } else {
            let total = checked_arith::checked_conv_total(out_ch, in_ch, kernel)?;
            let raw = cursor.read_slice(total)?;
            let mut interleaved_weights = AlignedVec::new(padded_total, 0.0f32)?;
            layout::transpose_conv1d_interleaved_4wide(
                raw,
                &mut interleaved_weights,
                in_ch,
                out_ch,
                kernel,
            );
            block.set_conv_weights(&interleaved_weights);
        }

        if do_bias {
            let bias_raw = cursor.read_slice(out_ch)?;
            block.set_conv_bias(bias_raw);
        }

        let bn_scale = cursor.read_slice(out_ch)?;
        let bn_offset = cursor.read_slice(out_ch)?;
        block.set_bn_params(bn_scale, bn_offset)?;

        for &d in dilations {
            total_rf += (kernel - 1) * d;
        }

        blocks.push(block);
    }

    let post_stack_head = if let Some(ref head_config) = topo.head {
        let in_ch = *topo
            .channels
            .last()
            .context("ConvNet: no blocks, cannot build head")?;

        let mut head = PostStackHead::from_config(head_config, in_ch)
            .map_err(|e| anyhow::anyhow!("Failed to build post-stack head: {}", e))?;

        let ch = head.in_channels();
        let out_ch = head.out_channels();
        let kernel = head.conv.kernel;
        let num_blocks = out_ch.div_ceil(4);
        let padded_total = checked_arith::checked_conv_padded_total(num_blocks, 4, ch, kernel)?;

        if cursor.is_interleaved4() {
            let raw = cursor.read_slice(padded_total)?;
            head.set_weights(raw);
        } else {
            let total = checked_arith::checked_conv_total(out_ch, ch, kernel)?;
            let raw = cursor.read_slice(total)?;
            let mut interleaved = AlignedVec::new(padded_total, 0.0f32)?;
            layout::transpose_conv1d_interleaved_4wide(raw, &mut interleaved, ch, out_ch, kernel);
            head.set_weights(&interleaved);
        }

        if head.conv.do_bias {
            let bias_raw = cursor.read_slice(out_ch)?;
            head.set_bias(bias_raw);
        }

        total_rf += head.receptive_field() - 1;

        Some(head)
    } else {
        None
    };

    let head_scale = cursor.read_f32_finite()?;

    cursor.verify_exhausted()?;

    let head_out_ch = post_stack_head
        .as_ref()
        .map(|h| h.out_channels())
        .unwrap_or(1);
    let head_output_scratch = AlignedVec::new(head_out_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?;

    let max_scratch_ch = blocks.iter().map(|b| b.conv.out_ch).max().unwrap_or(1);
    let scratch_a = AlignedVec::new(max_scratch_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?;
    let scratch_b = AlignedVec::new(max_scratch_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?;

    let model = ConvNetModel {
        blocks,
        head_scale,
        receptive_field_size: total_rf,
        post_stack_head,
        head_output_scratch,
        scratch_a,
        scratch_b,
        prewarm_on_reset: true,
        linear_head: None,
    };

    info!(
        "[Dispatcher] ConvNet built — num_blocks={}, head_scale={:.6}, weights={}",
        topo.num_blocks,
        head_scale,
        data.weights.len()
    );

    Ok(Box::new(StaticModel::ConvNet(Box::new(model))))
}
