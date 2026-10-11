// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of ConvNet topologies from model data.

use super::super::data::NamModelData;
use super::super::model::HeadConfig;
use super::super::validation::{
    MAX_CONVNET_CHANNELS, MAX_CONVNET_KERNEL_SIZE, MAX_DILATION, MAX_DILATIONS_PER_ARRAY,
};
use super::wavenet::TopologyError;
use crate::common::diagnostics::NamErrorCode;

/// ConvNet configuration format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvNetFormat {
    /// Per-block config via `layers` array (NeuralAmpModeler-rs native format).
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
    /// Activation specification from C++ flat config (FlatCpp format only).
    pub activation: Option<serde_json::Value>,
    /// Number of input audio channels (default 1).
    pub in_channels: usize,
    /// Number of output audio channels (default 1).
    pub out_channels: usize,
    /// Convolution groups (default 1).
    pub groups: usize,
}

impl ConvNetTopology {
    /// Calculates expected number of weights based on topology.
    pub fn expected_weights(&self) -> Result<usize, NamErrorCode> {
        let mut count = 0usize;
        match self.format {
            ConvNetFormat::FlatCpp => {
                let batchnorm = self.batchnorm.unwrap_or(false);
                let groups = self.groups;
                for i in 0..self.num_blocks {
                    let in_ch = if i == 0 {
                        self.in_channels
                    } else {
                        self.channels[i - 1]
                    };
                    let out_ch = self.channels[i];
                    let kernel = self.kernel_sizes[i];
                    let out_per_group = out_ch / groups;
                    let in_per_group = in_ch / groups;
                    let conv_weights = groups
                        .checked_mul(out_per_group)
                        .and_then(|v| v.checked_mul(in_per_group))
                        .and_then(|v| v.checked_mul(kernel))
                        .ok_or(NamErrorCode::WeightCountMismatch)?;
                    count = count
                        .checked_add(conv_weights)
                        .ok_or(NamErrorCode::WeightCountMismatch)?;
                    if batchnorm {
                        let bn_count = (4usize)
                            .checked_mul(out_ch)
                            .and_then(|v| v.checked_add(1))
                            .ok_or(NamErrorCode::WeightCountMismatch)?;
                        count = count
                            .checked_add(bn_count)
                            .ok_or(NamErrorCode::WeightCountMismatch)?;
                    } else {
                        count = count
                            .checked_add(out_ch)
                            .ok_or(NamErrorCode::WeightCountMismatch)?;
                    }
                }
                let last_out_ch = *self.channels.last().unwrap_or(&self.in_channels);
                let head_out_ch = self.out_channels;
                let head_weights = last_out_ch
                    .checked_mul(head_out_ch)
                    .ok_or(NamErrorCode::WeightCountMismatch)?;
                count = count
                    .checked_add(head_weights)
                    .ok_or(NamErrorCode::WeightCountMismatch)?;
                count = count
                    .checked_add(head_out_ch)
                    .ok_or(NamErrorCode::WeightCountMismatch)?;
                Ok(count)
            }
            ConvNetFormat::Layers => {
                for i in 0..self.num_blocks {
                    let in_ch = if i == 0 {
                        self.in_channels
                    } else {
                        self.channels[i - 1]
                    };
                    let out_ch = self.channels[i];
                    let kernel = self.kernel_sizes[i];
                    let conv_total = out_ch
                        .checked_mul(in_ch)
                        .and_then(|v| v.checked_mul(kernel))
                        .ok_or(NamErrorCode::WeightCountMismatch)?;
                    let block_total = conv_total
                        .checked_add(3 * out_ch)
                        .ok_or(NamErrorCode::WeightCountMismatch)?;
                    count = count
                        .checked_add(block_total)
                        .ok_or(NamErrorCode::WeightCountMismatch)?;
                }
                if let Some(ref head) = self.head {
                    let in_ch = *self.channels.last().unwrap_or(&self.in_channels);
                    let out_ch = head.out_channels.unwrap_or(1);
                    let kernel = head.kernel_size.unwrap_or(3);
                    let conv_total = out_ch
                        .checked_mul(in_ch)
                        .and_then(|v| v.checked_mul(kernel))
                        .ok_or(NamErrorCode::WeightCountMismatch)?;
                    let bias_count = if head.bias.unwrap_or(true) { out_ch } else { 0 };
                    count = count
                        .checked_add(conv_total + bias_count)
                        .ok_or(NamErrorCode::WeightCountMismatch)?;
                }
                count = count
                    .checked_add(1)
                    .ok_or(NamErrorCode::WeightCountMismatch)?;
                Ok(count)
            }
        }
    }
}

/// Checks and returns the ConvNet geometry.
///
/// Returns `Ok(Some(...))` for valid ConvNet topologies that can be dispatched.
/// Returns `Ok(None)` if the architecture is not `"ConvNet"` or if required fields
/// are missing for both formats.
/// Returns `Err` with a typed [`TopologyError`] for invalid structural parameters:
/// - `in_channels % groups != 0` or `channels % groups != 0` (E1316 [`NamErrorCode::InvalidFeatureTopology`])
/// - Dimension violations, ceilings, zeros (E1300 [`NamErrorCode::InvalidModelTopology`])
pub fn get_convnet_topology(data: &NamModelData) -> Result<Option<ConvNetTopology>, TopologyError> {
    if data.architecture != "ConvNet" {
        return Ok(None);
    }

    let in_channels = data.config.in_channels.unwrap_or(1);
    let out_channels = data.config.out_channels.unwrap_or(1);
    let groups = data.config.groups.unwrap_or(1);

    if in_channels == 0 {
        return Err(TopologyError {
            code: NamErrorCode::InvalidModelTopology,
            reason: "in_channels=0 (must have at least 1 input channel)".into(),
        });
    }
    if in_channels > MAX_CONVNET_CHANNELS {
        return Err(TopologyError {
            code: NamErrorCode::InvalidModelTopology,
            reason: format!("in_channels ({in_channels}) exceeds maximum {MAX_CONVNET_CHANNELS}"),
        });
    }
    if out_channels == 0 {
        return Err(TopologyError {
            code: NamErrorCode::InvalidModelTopology,
            reason: "out_channels=0 (must have at least 1 output channel)".into(),
        });
    }
    if out_channels > MAX_CONVNET_CHANNELS {
        return Err(TopologyError {
            code: NamErrorCode::InvalidModelTopology,
            reason: format!("out_channels ({out_channels}) exceeds maximum {MAX_CONVNET_CHANNELS}"),
        });
    }
    if groups == 0 {
        return Err(TopologyError {
            code: NamErrorCode::InvalidModelTopology,
            reason: "groups=0 (must have at least 1 group)".into(),
        });
    }
    if groups > MAX_CONVNET_CHANNELS {
        return Err(TopologyError {
            code: NamErrorCode::InvalidModelTopology,
            reason: format!("groups ({groups}) exceeds maximum {MAX_CONVNET_CHANNELS}"),
        });
    }

    if !in_channels.is_multiple_of(groups) {
        return Err(TopologyError {
            code: NamErrorCode::InvalidFeatureTopology,
            reason: format!("in_channels ({in_channels}) must be divisible by groups ({groups})"),
        });
    }

    // ── Detect C++ flat format: scalar `channels` + array `dilations` + `batchnorm` bool ──
    if let (Some(ch), Some(dils), Some(bn)) = (
        data.config.conv_channels,
        data.config.conv_dilations.as_deref(),
        data.config.conv_batchnorm,
    ) {
        if ch == 0 {
            return Err(TopologyError {
                code: NamErrorCode::InvalidModelTopology,
                reason: "ConvNet flat format: channels=0".into(),
            });
        }
        if dils.is_empty() {
            return Err(TopologyError {
                code: NamErrorCode::InvalidModelTopology,
                reason: "ConvNet flat format: empty dilations".into(),
            });
        }
        if ch > MAX_CONVNET_CHANNELS {
            return Err(TopologyError {
                code: NamErrorCode::InvalidModelTopology,
                reason: format!(
                    "ConvNet flat format: channels ({ch}) exceeds maximum {MAX_CONVNET_CHANNELS} \
                     — OOM/DoS protection"
                ),
            });
        }
        if !ch.is_multiple_of(groups) {
            return Err(TopologyError {
                code: NamErrorCode::InvalidFeatureTopology,
                reason: format!("channels ({ch}) must be divisible by groups ({groups})"),
            });
        }
        if dils.len() > MAX_DILATIONS_PER_ARRAY {
            return Err(TopologyError {
                code: NamErrorCode::InvalidModelTopology,
                reason: format!(
                    "ConvNet flat format: {} dilations exceeds maximum {MAX_DILATIONS_PER_ARRAY} \
                     — OOM/DoS protection",
                    dils.len()
                ),
            });
        }
        for (j, &d) in dils.iter().enumerate() {
            if d == 0 {
                return Err(TopologyError {
                    code: NamErrorCode::InvalidModelTopology,
                    reason: format!("ConvNet flat format dilation[{j}] is 0 — invalid dilation"),
                });
            }
            if d > MAX_DILATION {
                return Err(TopologyError {
                    code: NamErrorCode::InvalidModelTopology,
                    reason: format!(
                        "ConvNet flat format dilation[{j}] ({d}) exceeds maximum \
                         {MAX_DILATION} — OOM/DoS protection"
                    ),
                });
            }
        }

        let activation = data.config.conv_activation.clone();

        // Synthesize: one block per dilation, kernel_size=2 (C++ hardcoded)
        let num_blocks = dils.len();
        let channels_vec = vec![ch; num_blocks];
        let kernel_sizes_vec = vec![2; num_blocks];
        let dilations_vec: Vec<Vec<usize>> = dils.iter().map(|&d| vec![d]).collect();

        return Ok(Some(ConvNetTopology {
            num_blocks,
            channels: channels_vec,
            kernel_sizes: kernel_sizes_vec,
            dilations: dilations_vec,
            head: None, // C++ flat format uses separate linear head, not PostStackHead
            format: ConvNetFormat::FlatCpp,
            batchnorm: Some(bn),
            activation,
            in_channels,
            out_channels,
            groups,
        }));
    }

    // ── Per-layer `layers` format ──
    let layers = &data.config.layers;
    if layers.is_empty() {
        return Ok(None);
    }

    let mut channels = Vec::with_capacity(layers.len());
    let mut kernel_sizes = Vec::with_capacity(layers.len());
    let mut dilations = Vec::with_capacity(layers.len());

    for (i, layer) in layers.iter().enumerate() {
        let ch = match layer.channels {
            Some(c) if c > 0 => {
                if c > MAX_CONVNET_CHANNELS {
                    return Err(TopologyError {
                        code: NamErrorCode::InvalidModelTopology,
                        reason: format!(
                            "ConvNet block {i}: channels ({c}) exceeds maximum {MAX_CONVNET_CHANNELS} \
                             — OOM/DoS protection"
                        ),
                    });
                }
                if c % groups != 0 {
                    return Err(TopologyError {
                        code: NamErrorCode::InvalidFeatureTopology,
                        reason: format!(
                            "ConvNet block {i}: channels ({c}) must be divisible by groups ({groups})"
                        ),
                    });
                }
                c
            }
            _ => {
                return Err(TopologyError {
                    code: NamErrorCode::InvalidModelTopology,
                    reason: format!("ConvNet block {i}: missing or invalid 'channels'"),
                });
            }
        };
        let k = match layer.kernel_size {
            Some(k) if k > 0 => {
                if k > MAX_CONVNET_KERNEL_SIZE {
                    return Err(TopologyError {
                        code: NamErrorCode::InvalidModelTopology,
                        reason: format!(
                            "ConvNet block {i}: kernel_size ({k}) exceeds maximum \
                             {MAX_CONVNET_KERNEL_SIZE} — OOM/DoS protection"
                        ),
                    });
                }
                k
            }
            _ => {
                return Err(TopologyError {
                    code: NamErrorCode::InvalidModelTopology,
                    reason: format!("ConvNet block {i}: missing or invalid 'kernel_size'"),
                });
            }
        };
        let d = match layer.dilations.as_deref() {
            Some(d) if !d.is_empty() => {
                if d.len() > MAX_DILATIONS_PER_ARRAY {
                    return Err(TopologyError {
                        code: NamErrorCode::InvalidModelTopology,
                        reason: format!(
                            "ConvNet block {i}: {} dilations exceeds maximum \
                             {MAX_DILATIONS_PER_ARRAY} — OOM/DoS protection",
                            d.len()
                        ),
                    });
                }
                for (j, &dil) in d.iter().enumerate() {
                    if dil == 0 {
                        return Err(TopologyError {
                            code: NamErrorCode::InvalidModelTopology,
                            reason: format!(
                                "ConvNet block {i} dilation[{j}] is 0 — invalid dilation"
                            ),
                        });
                    }
                    if dil > MAX_DILATION {
                        return Err(TopologyError {
                            code: NamErrorCode::InvalidModelTopology,
                            reason: format!(
                                "ConvNet block {i} dilation[{j}] ({dil}) exceeds maximum \
                                 {MAX_DILATION} — OOM/DoS protection"
                            ),
                        });
                    }
                }
                d.to_vec()
            }
            _ => {
                return Err(TopologyError {
                    code: NamErrorCode::InvalidModelTopology,
                    reason: format!("ConvNet block {i}: missing or invalid 'dilations'"),
                });
            }
        };
        channels.push(ch);
        kernel_sizes.push(k);
        dilations.push(d);
    }

    // ── Post-stack head ceilings (F-RES2-02) ──
    let head_in = channels.last().copied().unwrap_or(in_channels);
    let head = match data.config.validate_head(head_in) {
        Ok(head) => head,
        Err(e) => {
            return Err(TopologyError {
                code: NamErrorCode::InvalidModelTopology,
                reason: e.to_string(),
            });
        }
    };

    Ok(Some(ConvNetTopology {
        num_blocks: layers.len(),
        channels,
        kernel_sizes,
        dilations,
        head,
        format: ConvNetFormat::Layers,
        batchnorm: None,
        activation: None,
        in_channels,
        out_channels,
        groups,
    }))
}
