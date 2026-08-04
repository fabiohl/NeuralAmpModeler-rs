// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet Nano builder — const-generic profile `<4, 3, 2>`.
//!
//! 4 channels, 3×1 kernel, head-scale 2. Tiniest variant, ~1/6 cost of Standard.

use crate::loader::nam_json::{NamModelData, NamWavenetTopology};
use crate::models::wavenet::WaveNetModel;

/// Constructs a WaveNet Nano model (channels=4, kernel_size=3, head_scale=2).
pub(crate) fn build_wavenet_nano(data: &NamModelData) -> anyhow::Result<WaveNetModel<4, 3, 2>> {
    super::standard::build_wavenet_typed::<4, 3, 2>(data, NamWavenetTopology::Nano)
}
