// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Linear model builder — reads weights from `NamModelData` and constructs a `LinearModel`.

use super::WeightCursor;
use crate::loader::nam_json::NamModelData;
use crate::models::StaticModel;
use crate::models::linear::LinearModel;
use anyhow::Context;
use log::info;

pub(crate) fn build_linear(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    let (receptive_field, has_bias, implementation) =
        crate::loader::nam_json::get_linear_topology(data)
            .context("Linear topology not detectable (check receptive_field and bias)")?;

    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);

    let weight_data = cursor.read_slice(receptive_field)?;
    let weights: Vec<f32> = weight_data.to_vec();

    let bias = if has_bias {
        cursor.read_f32_finite()?
    } else {
        0.0
    };

    cursor.verify_exhausted()?;

    let model =
        LinearModel::new(weights, bias, implementation).context("Failed to create LinearModel")?;

    info!(
        "[Dispatcher] Linear built — receptive_field={}, has_bias={}, implementation={:?}, weights_count={}",
        receptive_field,
        has_bias,
        implementation,
        data.weights.len()
    );

    Ok(Box::new(StaticModel::Linear(Box::new(model))))
}
