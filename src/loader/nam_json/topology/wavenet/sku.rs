// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Catalog SKU matching for WaveNet standard topologies.

use super::NamWavenetTopology;

/// Canonical dilations for Standard WaveNet models: `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]`.
pub(crate) static STD_DILATIONS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
/// Dilations for the first layer array of Lite/Feather/Nano WaveNet models: `[1, ..., 64]`.
pub(crate) static LITE_DILATIONS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
/// Dilations for the second layer array of Lite/Feather/Nano WaveNet models: `[128, ..., 512, 1, ..., 512]`.
pub(crate) static LITE_DILATIONS_2: &[usize] =
    &[128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

/// Attempts to match dilation and channel patterns against known catalog SKUs.
pub fn try_match_catalog_sku(
    first_channels: usize,
    dilations: &[Vec<usize>],
) -> Option<NamWavenetTopology> {
    let dils_0 = &dilations[0];
    let dils_1 = &dilations[1];
    match first_channels {
        16 if dils_0 == STD_DILATIONS && dils_1 == STD_DILATIONS => {
            Some(NamWavenetTopology::Standard)
        }
        12 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Lite)
        }
        8 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Feather)
        }
        4 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Nano)
        }
        _ => None,
    }
}
