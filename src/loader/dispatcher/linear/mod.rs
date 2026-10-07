// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Linear model builder — reads weights from `NamModelData` and constructs a `LinearModel`.

use super::WeightCursor;
use super::checked_arith::checked_mul;
use crate::loader::nam_json::NamModelData;
use crate::models::StaticModel;
use crate::models::linear::LinearModel;
use anyhow::Context;
use log::info;

pub(crate) fn build_linear(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    let topo = crate::loader::nam_json::get_linear_topology(data)
        .context("Linear topology not detectable (check receptive_field and bias)")?;

    topo.validate_channels().map_err(anyhow::Error::from)?;
    topo.validate_weights_count(data.weights.len())
        .map_err(anyhow::Error::from)?;

    let num_kernels = topo.num_kernels();
    let num_biases = topo.num_biases();

    let total_kernel_weights = checked_mul(topo.receptive_field, num_kernels)?;

    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);

    let weight_data = cursor.read_slice(total_kernel_weights)?;
    let weights: Vec<f32> = weight_data.to_vec();

    let biases = if topo.has_bias {
        if topo.in_channels == topo.out_channels {
            // Equal channel counts share a single bias scalar replicated across all out_channels
            let b = cursor.read_f32_finite()?;
            vec![b; topo.out_channels]
        } else {
            // 1 -> N or N -> 1: out_channels separate bias scalars
            let mut b_vec = Vec::with_capacity(num_biases);
            for _ in 0..num_biases {
                b_vec.push(cursor.read_f32_finite()?);
            }
            b_vec
        }
    } else {
        vec![0.0; topo.out_channels]
    };

    cursor.verify_exhausted()?;

    let model = LinearModel::new_with_topology(topo, weights, biases)
        .context("Failed to create LinearModel")?;

    info!(
        "[Dispatcher] Linear built — in_channels={}, out_channels={}, receptive_field={}, has_bias={}, implementation={:?}, weights_count={}",
        topo.in_channels,
        topo.out_channels,
        topo.receptive_field,
        topo.has_bias,
        topo.implementation,
        data.weights.len()
    );

    Ok(Box::new(StaticModel::Linear(Box::new(model))))
}
