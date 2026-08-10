// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::layout;
use super::static_factory::reject_condition_dsp_lstm;
use crate::loader::dispatcher::checked_arith;
use crate::loader::nam_json::{
    FreeWavenetGeometry, NamModelData, WavenetTopologyResult, get_wavenet_topology,
};
use crate::math::common::AlignedVec;
use crate::models::a2::activations::ActivationType;
use crate::models::wavenet::post_stack_head::parse_activation;
use crate::models::wavenet::{
    DenseLayerDyn, PostStackHead, WAVENET_MAX_NUM_FRAMES, WaveNetLayerArrayDyn, WaveNetLayerDyn,
    WaveNetLayerState, WaveNetModelDyn,
};
use log::info;

// =============================================================================
// Dynamic WaveNet array constructor
// =============================================================================

/// Builds a `WaveNetLayerArrayDyn` with runtime dimensions.
///
/// Layout por array (C++ `WaveNetLayerArrayT::SetWeights`):
/// ```text
/// rechannel.weights[IN*CH]
/// for layer in layers:
///     conv1d.weights[CH*K*CH] + conv1d.bias[CH]         (DoBias=true)
///     input_mixin.weights[COND*CH]                       (DoBias=false)
///     one_by_one.weights[CH*CH] + one_by_one.bias[CH]    (DoBias=true)
/// head_rechannel.weights[CH*HEAD] + head_rechannel.bias[HEAD]? (HasHeadBias)
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "WaveNet dynamic model builder requiring many topology parameters for runtime layer construction"
)]
fn build_wavenet_array_dyn(
    cursor: &mut WeightCursor<'_>,
    in_ch: usize,
    cond: usize,
    ch: usize,
    k: usize,
    head: usize,
    dilations: &[usize],
    has_head_bias: bool,
    alloc_num: &mut usize,
    activation: &ActivationType,
) -> anyhow::Result<WaveNetLayerArrayDyn> {
    let rechannel = layout::read_dense_weights_typed::<DenseLayerDyn>(cursor, in_ch, ch, false)?;

    let mut layers = Vec::with_capacity(dilations.len());
    let mut states = Vec::with_capacity(dilations.len());

    for &dilation in dilations {
        let conv1d = layout::read_conv1d_weights_typed::<crate::models::wavenet::Conv1dDyn>(
            cursor, ch, ch, k, dilation, true,
        )?;

        let input_mixin =
            layout::read_dense_weights_typed::<DenseLayerDyn>(cursor, cond, ch, false)?;

        let one_by_one = layout::read_dense_weights_typed::<DenseLayerDyn>(cursor, ch, ch, true)?;

        layers.push(WaveNetLayerDyn::new(ch, conv1d, input_mixin, one_by_one)?);
        layers.last_mut().unwrap().activation = activation.clone();

        let rf = (k - 1) * dilation;
        states.push(WaveNetLayerState::new(ch, rf, *alloc_num)?);
        *alloc_num += 1;
    }

    let head_rechannel =
        layout::read_dense_head_weights_typed::<DenseLayerDyn>(cursor, ch, head, has_head_bias)?;

    let receptive_field_size: usize = dilations.iter().map(|&d| (k - 1) * d).sum();

    let block_size = ch;
    let block_buffer = AlignedVec::new(block_size * WAVENET_MAX_NUM_FRAMES, 0.0)?;
    let num_layers = layers.len();

    Ok(WaveNetLayerArrayDyn {
        in_ch,
        cond,
        ch,
        k,
        head,
        layers,
        states,
        rechannel,
        head_rechannel,
        array_outputs: AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        head_accum: AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        head_outputs: AlignedVec::new(head * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        receptive_field_size,
        block_size,
        block_buffer,
        effective_layers: num_layers,
    })
}

/// Maximum combined nesting depth across Container and condition_dsp recursion.
/// Prevents stack overflow from alternating Container/WaveNet(condition_dsp) chains.
const MAX_UNIFIED_DEPTH: usize = 8;

// =============================================================================
// Sub-model builder for condition_dsp recursion control
// =============================================================================

/// Builds a nested DSP sub-model with combined depth tracking.
///
/// Only WaveNet Free-geometry sub-models contribute to the condition_dsp depth
/// counter (const-generic SKUs and non-WaveNet architectures cannot nest further).
/// Container sub-models forward the combined depth to their own builder.
fn build_sub_model(
    data: &NamModelData,
    depth: usize,
    container_depth: usize,
) -> anyhow::Result<Box<crate::models::StaticModel>> {
    if data.architecture == "SlimmableContainer" {
        let combined = depth + container_depth;
        if combined > MAX_UNIFIED_DEPTH {
            anyhow::bail!(crate::loader::nam_json::JsonError::SubmodelsTooDeep {
                depth: combined,
                max_depth: MAX_UNIFIED_DEPTH,
            });
        }
        return super::super::container::build_container_inner(data, combined);
    }
    if data.architecture == "WaveNet"
        && let WavenetTopologyResult::Free(ref geom) = get_wavenet_topology(data)
    {
        let model = build_wavenet_dynamic_inner(data, geom, depth, container_depth)?;
        return Ok(Box::new(crate::models::StaticModel::WavenetDyn(Box::new(
            model,
        ))));
    }
    crate::loader::dispatcher::build_model(data)
}

// =============================================================================
// Dynamic WaveNet model entry point
// =============================================================================

/// Builds a `WaveNetModelDyn` from parsed model data using runtime dimensions.
///
/// Consumes the geometry description from [`FreeWavenetGeometry`] and reads
/// weights sequentially in the same layout as the const-generic path.
///
/// Weight layout (C++ WaveNet.h `SetWeights`):
/// ```text
/// [array1.rechannel] [array1.layers...] [array1.head_rechannel]
/// [array2.rechannel] [array2.layers...] [array2.head_rechannel]
/// [head_scale]
/// ```
pub(crate) fn build_wavenet_dynamic(
    data: &NamModelData,
    geom: &FreeWavenetGeometry,
) -> anyhow::Result<WaveNetModelDyn> {
    build_wavenet_dynamic_inner(data, geom, 0, 0)
}

/// Internal recursive builder for dynamic WaveNet layer arrays and condition DSP sub-models.
fn build_wavenet_dynamic_inner(
    data: &NamModelData,
    geom: &FreeWavenetGeometry,
    depth: usize,
    container_depth: usize,
) -> anyhow::Result<WaveNetModelDyn> {
    super::static_factory::validate_layer_activations(data)?;

    if geom.num_arrays == 0 {
        anyhow::bail!("WaveNet dynamic engine requires at least 1 layer array");
    }

    let ch = geom.channels[0];
    let head = geom.head_sizes[0];
    let cond = geom.condition_size;

    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);

    debug_assert_eq!(geom.channels.len(), geom.num_arrays);
    debug_assert_eq!(geom.head_sizes.len(), geom.num_arrays);
    debug_assert_eq!(geom.head_biases.len(), geom.num_arrays);
    debug_assert_eq!(geom.dilations.len(), geom.num_arrays);
    debug_assert_eq!(geom.kernel_sizes.len(), geom.num_arrays);

    let mut alloc_num = 0usize;

    let mut arrays = Vec::with_capacity(geom.num_arrays);
    for i in 0..geom.num_arrays {
        let in_ch = if i == 0 { 1 } else { geom.channels[i - 1] };
        let array_ch = geom.channels[i];
        let array_head = geom.head_sizes[i];
        let dilations = &geom.dilations[i];
        let has_head_bias = *geom.head_biases.get(i).unwrap_or(&false);
        let array_k = if geom.kernel_sizes[i] > 0 {
            geom.kernel_sizes[i]
        } else {
            geom.kernel_size
        };

        let act_str = if let Some(lcfg) = data.config.layers.get(i)
            && let Some(ref act) = lcfg.activation
        {
            act.as_str()
        } else {
            "Tanh"
        };
        let activation = parse_activation(act_str);

        let array = build_wavenet_array_dyn(
            &mut cursor,
            in_ch,
            cond,
            array_ch,
            array_k,
            array_head,
            dilations,
            has_head_bias,
            &mut alloc_num,
            &activation,
        )?;
        arrays.push(array);
    }

    let head_scale = cursor.read_f32_finite()?;

    let post_stack_head = if let Some(ref head_config) = geom.post_stack_head {
        let in_ch = *geom.head_sizes.last().unwrap_or(&1);
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

        Some(head)
    } else {
        None
    };

    cursor.verify_exhausted()?;

    // Build condition_dsp sub-model if present in JSON config.
    // The sub-model is a self-contained `.nam` model that pre-processes the
    // audio input before it reaches the main layer arrays. Its weights are
    // consumed independently during sub-model construction (C++ get_dsp
    // inside parse_config_json, model.cpp:834-838).
    let condition_dsp = if let Some(ref cond_dsp_json) = data.config.condition_dsp {
        let combined = depth + container_depth;
        if combined >= MAX_UNIFIED_DEPTH {
            anyhow::bail!(
                "condition_dsp nesting depth ({}) exceeds maximum ({})",
                combined,
                MAX_UNIFIED_DEPTH
            );
        }

        let cond_dsp_data: NamModelData = serde_json::from_value(cond_dsp_json.clone())?;

        reject_condition_dsp_lstm(&cond_dsp_data)?;

        if let (Some(main_sr), Some(cond_sr)) = (data.sample_rate, cond_dsp_data.sample_rate)
            && (main_sr - cond_sr).abs() > 1.0
        {
            anyhow::bail!(
                "condition_dsp sample rate mismatch: main={} Hz, condition_dsp={} Hz",
                main_sr,
                cond_sr
            );
        }

        let cond_model = build_sub_model(&cond_dsp_data, depth + 1, container_depth)?;

        let cond_out = cond_model.num_output_channels();
        if cond_out > cond {
            anyhow::bail!(
                "condition_dsp output channels ({}) exceed WaveNet condition_size ({})",
                cond_out,
                cond
            );
        }

        info!(
            "[Dispatcher] condition_dsp built — architecture={}, output_channels={}",
            cond_dsp_data.architecture, cond_out
        );

        Some(cond_model)
    } else {
        None
    };

    let cond_dsp_output_size = cond * WAVENET_MAX_NUM_FRAMES;

    let head_out_ch = post_stack_head
        .as_ref()
        .map(|h| h.out_channels())
        .unwrap_or(1);
    let head_output_scratch = AlignedVec::new(head_out_ch * WAVENET_MAX_NUM_FRAMES, 0.0)?;

    let mut rf = arrays
        .iter()
        .map(|a| a.receptive_field_size)
        .max()
        .unwrap_or(0);
    if let Some(ref head_proc) = post_stack_head {
        rf += head_proc.receptive_field() - 1;
    }

    let model = WaveNetModelDyn {
        ch,
        k: geom.kernel_size,
        head,
        arrays,
        head_scale,
        receptive_field_size: rf,
        condition_dsp,
        condition_dsp_output: AlignedVec::new(cond_dsp_output_size, 0.0)?,
        post_stack_head,
        head_output_scratch,
        prewarm_on_reset: true,
        slimmable_capable: data.is_slimmable_capable() || geom.allowed_channels.is_some(),
        allowed_channels: geom.allowed_channels.clone(),
        pending_slim_channel: None,
    };

    info!(
        "[Dispatcher] WaveNet Dynamic built — CH={}, K={}, HEAD={}, arrays={}, \
         head_scale={:.6}, weights={}",
        ch,
        geom.kernel_size,
        head,
        geom.num_arrays,
        head_scale,
        data.weights.len()
    );

    Ok(model)
}
