// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! LSTM model builder dispatcher.
//!
//! Maps the geometry `(num_layers, hidden_size)` detected from the NAM model
//! onto one of ten compile-time-optimized static profiles (`Lstm1x3` …
//! `Lstm2x24`, `Lstm1x40`), or falls back to a dynamically-shaped
//! [`StaticModel::LstmDyn`] via [`super::dynamic_builder::build_lstm_dynamic`]
//! when the geometry does not match any known profile.
//!
//! Each static profile is constructed with [`super::static_builder::build_lstm_1layer`]
//! or [`super::static_builder::build_lstm_2layer`], passing interleave and
//! alignment constants (`B`, `BI`, `B2`) optimized for that profile's hidden size.

use super::dynamic_builder::build_lstm_dynamic;
use super::static_builder::{build_lstm_1layer, build_lstm_2layer};
use crate::loader::nam_json::{NamModelData, get_lstm_topology};
use crate::models::StaticModel;
use anyhow::Context;

/// Builds a boxed [`StaticModel`] by matching LSTM layer count and hidden
/// size against the ten known static profiles, falling back to
/// [`StaticModel::LstmDyn`] for unrecognized geometries.
///
/// # Static profile table
/// | Layers | Hidden | Variant            | 1D-B    | 1D-BI   | 2D-B   | 2D-BI    |
/// |:------:|:------:|:------------------|:-------:|:-------:|:------:|:--------:|
/// | 1      | 3      | `Lstm1x3`         | 3       | 4       | —      | —        |
/// | 1      | 8      | `Lstm1x8`         | 8       | 9       | —      | —        |
/// | 1      | 12     | `Lstm1x12`        | 12      | 13      | —      | —        |
/// | 1      | 16     | `Lstm1x16`        | 16      | 17      | —      | —        |
/// | 1      | 24     | `Lstm1x24`        | 24      | 25      | —      | —        |
/// | 1      | 40     | `Lstm1x40`        | 40      | 41      | —      | —        |
/// | 2      | 8      | `Lstm2x8`         | 8       | 9       | 16     | 32       |
/// | 2      | 12     | `Lstm2x12`        | 12      | 13      | 24     | 48       |
/// | 2      | 16     | `Lstm2x16`        | 16      | 17      | 32     | 64       |
/// | 2      | 24     | `Lstm2x24`        | 24      | 25      | 48     | 96       |
/// | _      | _      | `LstmDyn`         | dynamic | dynamic | dynamic | dynamic |
pub(crate) fn build_lstm(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    let result = get_lstm_topology(data).map_err(|e| anyhow::anyhow!(e))?;
    let (num_layers, hidden_size) =
        result.context("LSTM geometry not detectable (check num_layers and hidden_size)")?;

    match (num_layers, hidden_size) {
        (1, 3) => {
            let model = build_lstm_1layer::<3, 4, 12>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x3(Box::new(model))))
        }
        (1, 8) => {
            let model = build_lstm_1layer::<8, 9, 32>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x8(Box::new(model))))
        }
        (1, 12) => {
            let model = build_lstm_1layer::<12, 13, 48>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x12(Box::new(model))))
        }
        (1, 16) => {
            let model = build_lstm_1layer::<16, 17, 64>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x16(Box::new(model))))
        }
        (1, 24) => {
            let model = build_lstm_1layer::<24, 25, 96>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x24(Box::new(model))))
        }
        (2, 8) => {
            let model = build_lstm_2layer::<8, 9, 16, 32>(data, num_layers, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm2x8(Box::new(model))))
        }
        (2, 12) => {
            let model = build_lstm_2layer::<12, 13, 24, 48>(data, num_layers, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm2x12(Box::new(model))))
        }
        (2, 16) => {
            let model = build_lstm_2layer::<16, 17, 32, 64>(data, num_layers, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm2x16(Box::new(model))))
        }
        (1, 40) => {
            let model = build_lstm_1layer::<40, 41, 160>(data, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm1x40(Box::new(model))))
        }
        (2, 24) => {
            let model = build_lstm_2layer::<24, 25, 48, 96>(data, num_layers, hidden_size)?;
            Ok(Box::new(StaticModel::Lstm2x24(Box::new(model))))
        }
        _ => {
            let model = build_lstm_dynamic(data, num_layers, hidden_size)?;
            Ok(Box::new(StaticModel::LstmDyn(Box::new(model))))
        }
    }
}
