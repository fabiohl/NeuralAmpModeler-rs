// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet Lite builder — const-generic profile `<12, 3, 6>`.
//!
//! 12 channels, 3×1 kernel, head-scale 6. Approximately 1/2 the
//! compute cost of Standard.

use crate::loader::nam_json::{NamModelData, NamWavenetTopology};
use crate::models::wavenet::WaveNetModel;

/// Constructs a WaveNet Lite model (channels=12, kernel_size=3, head_scale=6).
pub(crate) fn build_wavenet_lite(data: &NamModelData) -> anyhow::Result<WaveNetModel<12, 3, 6>> {
    super::standard::build_wavenet_typed::<12, 3, 6>(data, NamWavenetTopology::Lite)
}
