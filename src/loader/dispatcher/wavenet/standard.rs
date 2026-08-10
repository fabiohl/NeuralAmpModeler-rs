// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::layout;
use crate::loader::nam_json::{NamModelData, NamWavenetTopology};
use crate::math::common::AlignedVec;
use crate::models::a2::activations::ActivationType;
use crate::models::wavenet::post_stack_head::parse_activation;
use crate::models::wavenet::{DenseLayer, WaveNetLayer, WaveNetLayerArray, WaveNetModel};
use crate::models::wavenet::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use anyhow::Context;
use log::info;

// =============================================================================
// Generic static constructor
// =============================================================================

/// Builds a `WaveNetModel<CH, K, HEAD>` with sequentially read weights.
///
/// Layout dos pesos (C++ WaveNet.h `SetWeights`):
/// ```text
/// [array1.rechannel] [array1.layers...] [array1.head_rechannel]
/// [array2.rechannel] [array2.layers...] [array2.head_rechannel]
/// [head_scale]
/// ```
pub(crate) fn build_wavenet_typed<const CH: usize, const K: usize, const HEAD: usize>(
    data: &NamModelData,
    topo: NamWavenetTopology,
) -> anyhow::Result<WaveNetModel<CH, K, HEAD>> {
    super::static_factory::validate_layer_activations(data)?;

    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);

    // Invariant: WaveNet topology guarantees at least 2 layers (enforced by topology.rs:85).
    debug_assert!(data.config.layers.len() >= 2);
    let l0 = &data.config.layers[0];
    let l1 = &data.config.layers[1];
    let act0 = parse_activation(l0.activation.as_deref().unwrap_or("Tanh"));
    let act1 = parse_activation(l1.activation.as_deref().unwrap_or("Tanh"));
    let dils_0 = l0
        .dilations
        .as_deref()
        .context("Missing dilations in layer 0")?;
    let dils_1 = l1
        .dilations
        .as_deref()
        .context("Missing dilations in layer 1")?;

    let mut alloc_num = 0usize;

    let array1 = build_wavenet_array::<1, 1, CH, K, HEAD>(WaveNetArrayConfig {
        cursor: &mut cursor,
        dilations: dils_0,
        has_head_bias: false,
        alloc_num: &mut alloc_num,
        activation: act0,
    })?;

    let array2 = build_wavenet_array::<CH, 1, HEAD, K, 1>(WaveNetArrayConfig {
        cursor: &mut cursor,
        dilations: dils_1,
        has_head_bias: true,
        alloc_num: &mut alloc_num,
        activation: act1,
    })?;

    let head_scale = cursor.read_f32_finite()?;

    cursor.verify_exhausted()?;

    let rf = array1.receptive_field_size.max(array2.receptive_field_size);

    let model = WaveNetModel::<CH, K, HEAD> {
        array1,
        array2,
        head_scale,
        receptive_field_size: rf,
        prewarm_on_reset: true,
    };

    info!(
        "[Dispatcher] WaveNet {:?} built — CH={}, K={}, HEAD={}, head_scale={:.6}, weights={}",
        topo,
        CH,
        K,
        HEAD,
        head_scale,
        data.weights.len()
    );

    Ok(model)
}

// =============================================================================
// Static array constructor
// =============================================================================

/// Configuration for building a static WaveNetLayerArray.
pub(crate) struct WaveNetArrayConfig<'a, 'b, 'c> {
    pub cursor: &'a mut WeightCursor<'b>,
    pub dilations: &'c [usize],
    pub has_head_bias: bool,
    pub alloc_num: &'a mut usize,
    pub activation: ActivationType,
}

/// Builds a `WaveNetLayerArray` by reading weights cursor-forward.
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
pub(crate) fn build_wavenet_array<
    const IN: usize,
    const COND: usize,
    const CH: usize,
    const K: usize,
    const HEAD: usize,
>(
    config: WaveNetArrayConfig<'_, '_, '_>,
) -> anyhow::Result<WaveNetLayerArray<IN, COND, CH, K, HEAD>> {
    let WaveNetArrayConfig {
        cursor,
        dilations,
        has_head_bias,
        alloc_num,
        activation,
    } = config;

    let rechannel = layout::read_dense_weights_typed::<DenseLayer<IN, CH>>(cursor, IN, CH, false)?;

    let mut layers = Vec::with_capacity(dilations.len());
    let mut states = Vec::with_capacity(dilations.len());

    for &dilation in dilations {
        let conv1d = layout::read_conv1d_weights_typed::<crate::models::wavenet::Conv1d<CH, CH, K>>(
            cursor, CH, CH, K, dilation, true,
        )?;

        let input_mixin =
            layout::read_dense_weights_typed::<DenseLayer<COND, CH>>(cursor, COND, CH, false)?;

        let one_by_one =
            layout::read_dense_weights_typed::<DenseLayer<CH, CH>>(cursor, CH, CH, true)?;

        layers.push(WaveNetLayer {
            conv1d,
            input_mixin,
            one_by_one,
            activation: activation.clone(),
            scratch_mixin: AlignedVec::new(CH * WAVENET_MAX_NUM_FRAMES, 0.0f32)?,
            scratch_conv: AlignedVec::new(CH * WAVENET_MAX_NUM_FRAMES, 0.0f32)?,
        });

        let rf = (K - 1) * dilation;
        states.push(WaveNetLayerState::new(CH, rf, *alloc_num)?);
        *alloc_num += 1;
    }

    let head_rechannel = layout::read_dense_head_weights_typed::<DenseLayer<CH, HEAD>>(
        cursor,
        CH,
        HEAD,
        has_head_bias,
    )?;

    let receptive_field_size: usize = dilations.iter().map(|&d| (K - 1) * d).sum();

    let block_size = CH;
    let block_buffer = AlignedVec::new(block_size * WAVENET_MAX_NUM_FRAMES, 0.0)?;
    let num_layers = layers.len();

    Ok(WaveNetLayerArray {
        layers,
        states,
        rechannel,
        head_rechannel,
        array_outputs: AlignedVec::new(CH * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        head_accum: AlignedVec::new(CH * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        head_outputs: AlignedVec::new(HEAD * WAVENET_MAX_NUM_FRAMES, 0.0)?,
        receptive_field_size,
        block_size,
        block_buffer,
        effective_layers: num_layers,
    })
}

// =============================================================================
// Topology instantiation (Standard)
// =============================================================================

pub(crate) fn build_wavenet_standard(
    data: &NamModelData,
) -> anyhow::Result<WaveNetModel<16, 3, 8>> {
    build_wavenet_typed::<16, 3, 8>(data, NamWavenetTopology::Standard)
}
