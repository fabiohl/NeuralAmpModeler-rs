// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet Feather builder — const-generic profile `<8, 3, 4>`.
//!
//! 8 channels, 3×1 kernel, head-scale 4. Approximately 1/3 the
//! compute cost of Standard.

use crate::loader::nam_json::{NamModelData, NamWavenetTopology};
use crate::models::wavenet::WaveNetModel;

/// Constructs a WaveNet Feather model (channels=8, kernel_size=3, head_scale=4).
pub(crate) fn build_wavenet_feather(data: &NamModelData) -> anyhow::Result<WaveNetModel<8, 3, 4>> {
    super::standard::build_wavenet_typed::<8, 3, 4>(data, NamWavenetTopology::Feather)
}
