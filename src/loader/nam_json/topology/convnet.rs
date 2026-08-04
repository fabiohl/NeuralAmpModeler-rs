// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of ConvNet topologies from model data.

use super::super::data::NamModelData;
use super::super::model::HeadConfig;
use super::super::validation::{
    MAX_CONVNET_CHANNELS, MAX_CONVNET_KERNEL_SIZE, MAX_DILATION, MAX_DILATIONS_PER_ARRAY,
};

/// ConvNet configuration format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvNetFormat {
    /// Per-block config via `layers` array (nam-rs native format).
    Layers,
    /// Flat C++ config: scalar `channels`, global `dilations`, `batchnorm` bool.
    FlatCpp,
}

/// Detected ConvNet topology.
///
/// Extracted from the model data when `architecture == "ConvNet"`.
/// ConvNet is a feed-forward stack of Conv1D → BatchNorm → Activation blocks
/// with an optional post-stack head, analogous to WaveNet without the
/// recurrent gating.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvNetTopology {
    /// Number of ConvNet blocks.
    pub num_blocks: usize,
    /// Per-block channel count.
    pub channels: Vec<usize>,
    /// Per-block kernel size.
    pub kernel_sizes: Vec<usize>,
    /// Per-block dilations.
    pub dilations: Vec<Vec<usize>>,
    /// Optional post-stack head configuration (Conv1D + activation).
    pub head: Option<HeadConfig>,
    /// Config format (Layers or FlatCpp).
    pub format: ConvNetFormat,
    /// Batchnorm flag (FlatCpp format only).
    pub batchnorm: Option<bool>,
    /// Activation string from C++ flat config (FlatCpp format only).
    pub activation: Option<String>,
}

/// Checks and returns the ConvNet geometry.
///
/// Returns `None` if the architecture is not `"ConvNet"` or if required
/// fields are missing for both formats.
pub fn get_convnet_topology(data: &NamModelData) -> Option<ConvNetTopology> {
    if data.architecture != "ConvNet" {
        return None;
    }

    // ── Detect C++ flat format: scalar `channels` + array `dilations` + `batchnorm` bool ──
    if let (Some(ch), Some(dils), Some(bn)) = (
        data.config.conv_channels,
        data.config.conv_dilations.as_deref(),
        data.config.conv_batchnorm,
    ) {
        if ch == 0 || dils.is_empty() {
            log::warn!("ConvNet flat format: channels=0 or empty dilations");
            return None;
        }
        if ch > MAX_CONVNET_CHANNELS {
            log::warn!(
                "ConvNet flat format: channels ({ch}) exceeds maximum {MAX_CONVNET_CHANNELS} \
                 — OOM/DoS protection"
            );
            return None;
        }
        if dils.len() > MAX_DILATIONS_PER_ARRAY {
            log::warn!(
                "ConvNet flat format: {} dilations exceeds maximum {MAX_DILATIONS_PER_ARRAY} \
                 — OOM/DoS protection",
                dils.len()
            );
            return None;
        }
        for (j, &d) in dils.iter().enumerate() {
            if d > MAX_DILATION {
                log::warn!(
                    "ConvNet flat format dilation[{j}] ({d}) exceeds maximum \
                     {MAX_DILATION} — OOM/DoS protection"
                );
                return None;
            }
        }

        let activation = data.config.conv_activation.as_deref().map(String::from);

        // Synthesize: one block per dilation, kernel_size=2 (C++ hardcoded)
        let num_blocks = dils.len();
        let channels_vec = vec![ch; num_blocks];
        let kernel_sizes_vec = vec![2; num_blocks];
        let dilations_vec: Vec<Vec<usize>> = dils.iter().map(|&d| vec![d]).collect();

        return Some(ConvNetTopology {
            num_blocks,
            channels: channels_vec,
            kernel_sizes: kernel_sizes_vec,
            dilations: dilations_vec,
            head: None, // C++ flat format uses separate linear head, not PostStackHead
            format: ConvNetFormat::FlatCpp,
            batchnorm: Some(bn),
            activation,
        });
    }

    // ── Per-layer `layers` format ──
    let layers = &data.config.layers;
    if layers.is_empty() {
        return None;
    }

    let mut channels = Vec::with_capacity(layers.len());
    let mut kernel_sizes = Vec::with_capacity(layers.len());
    let mut dilations = Vec::with_capacity(layers.len());

    for (i, layer) in layers.iter().enumerate() {
        let ch = match layer.channels {
            Some(c) if c > 0 => {
                if c > MAX_CONVNET_CHANNELS {
                    log::warn!(
                        "ConvNet block {i}: channels ({c}) exceeds maximum {MAX_CONVNET_CHANNELS} \
                         — OOM/DoS protection"
                    );
                    return None;
                }
                c
            }
            _ => {
                log::warn!("ConvNet block {i}: missing or invalid 'channels'");
                return None;
            }
        };
        let k = match layer.kernel_size {
            Some(k) if k > 0 => {
                if k > MAX_CONVNET_KERNEL_SIZE {
                    log::warn!(
                        "ConvNet block {i}: kernel_size ({k}) exceeds maximum \
                         {MAX_CONVNET_KERNEL_SIZE} — OOM/DoS protection"
                    );
                    return None;
                }
                k
            }
            _ => {
                log::warn!("ConvNet block {i}: missing or invalid 'kernel_size'");
                return None;
            }
        };
        let d = match layer.dilations.as_deref() {
            Some(d) if !d.is_empty() => {
                if d.len() > MAX_DILATIONS_PER_ARRAY {
                    log::warn!(
                        "ConvNet block {i}: {} dilations exceeds maximum \
                         {MAX_DILATIONS_PER_ARRAY} — OOM/DoS protection",
                        d.len()
                    );
                    return None;
                }
                for (j, &dil) in d.iter().enumerate() {
                    if dil > MAX_DILATION {
                        log::warn!(
                            "ConvNet block {i} dilation[{j}] ({dil}) exceeds maximum \
                             {MAX_DILATION} — OOM/DoS protection"
                        );
                        return None;
                    }
                }
                d.to_vec()
            }
            _ => {
                log::warn!("ConvNet block {i}: missing or invalid 'dilations'");
                return None;
            }
        };
        channels.push(ch);
        kernel_sizes.push(k);
        dilations.push(d);
    }

    Some(ConvNetTopology {
        num_blocks: layers.len(),
        channels,
        kernel_sizes,
        dilations,
        head: data.config.parse_head(),
        format: ConvNetFormat::Layers,
        batchnorm: None,
        activation: None,
    })
}
