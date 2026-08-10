// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::loader::nam_json::{NamModelData, is_a2_shape};
use crate::models::StaticModel;
use anyhow::bail;

pub(crate) mod dynamic;
pub(crate) mod feather;
pub(crate) mod layout;
pub(crate) mod lite;
pub(crate) mod nano;
pub(crate) mod standard;
pub(crate) mod static_factory;
pub(crate) mod traits;

pub use layout::select_interleave_width;
pub use layout::transpose_conv1d_interleaved_4wide;
pub use layout::transpose_conv1d_interleaved_8wide;
pub use layout::transpose_conv1d_interleaved_16wide;

/// Detects the WaveNet topology and branches to the correct const-generic builder.
pub(crate) fn build_wavenet(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    // ── A2: first-class branch (detected by shape) ──
    if let Some(topo) = is_a2_shape(data) {
        return static_factory::build_wavenet_a2(data, topo);
    }

    // ── A2: activation-based detection (secondary) — reject before A1 validation ──
    if data.is_wavenet_a2() {
        let layer_info: Vec<(usize, usize)> = data
            .config
            .layers
            .iter()
            .map(|l| {
                let ch = l.channels.unwrap_or(0);
                let k = l.kernel_size.unwrap_or(0);
                (ch, k)
            })
            .collect();
        bail!(
            "WaveNet A2 model detected but architecture shape not recognized — \
             channels or dilations do not match any known A2 topology. \
             Real A2 inference requires channels=3 (Lite) or 8 (Full) with the \
             canonical 23-layer dilation pattern. \
             Geometry: {:?}",
            layer_info
        );
    }

    // ── A1: topology detection (3-way) ──
    static_factory::build_wavenet_a1(data)
}
