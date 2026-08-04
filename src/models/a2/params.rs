// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Configuration parameters for the WaveNet A2 architecture.
//!
//! This module contains the structures that describe the topology of a WaveNet A2 model,
//! enabling construction and validation of the inference layers.
//!
//! ## Escopo atual (fast-path A2-Full/Lite — `a2_fast.cpp`)
//!
//! Os seguintes campos/structs são necessários ao porte fiel do fast-path:
//! - Constantes arquiteturais (`A2_NUM_LAYERS`, `A2_KERNEL_SIZES`, `A2_DILATIONS`, etc.)
//! - `LayerParamsA2` e `LayerArrayParamsA2` com `bottleneck == channels`,
//!   `groups_* == 1`, `GatingMode::None`, sem `head1x1` e sem FiLM.
//!
//! ## Reservado p/ motor A2 geral (futuro)
//!
//! Os structs/campos abaixo **não** são exercitados pelo fast-path e estão
//! preservados para o motor A2 completo (FiLM, gating, grouped conv,
//! `head1x1`, `bottleneck ≠ channels`, ativações heterogêneas):
//! - `Head1x1Params`, `HeadParams`
//! - Todos os campos `*_film: FiLMConfig`
//! - `secondary_activation` / `secondary_activations`
//! - `groups_input`, `groups_input_mixin`, e `groups` em `Layer1x1Params`/`Head1x1Params`
//! - Variantes `GatingMode::Gated`/`GatingMode::Blended`
//! - `ActivationType`s diferentes de `LeakyReLU`

// =============================================================================
// A2 Architectural Constants — mirroring github.com/NeuralAmpModelerCore/NAM/wavenet/a2_fast.h
// =============================================================================

/// Number of layers in an A2 layer array.
pub const A2_NUM_LAYERS: usize = 23;
/// Kernel size of the layer-array head rechannel convolution.
pub const A2_HEAD_KERNEL_SIZE: usize = 16;
/// LeakyReLU negative-slope used by every layer.
pub const A2_LEAKY_SLOPE: f32 = 0.01;
/// Per-layer kernel sizes (fixed pattern shared by A2 standard + nano).
pub const A2_KERNEL_SIZES: [usize; 23] = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6,
];
/// Per-layer dilations (fixed pattern shared by A2 standard + nano).
pub const A2_DILATIONS: [usize; 23] = [
    1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239,
];
/// Valid channel counts for A2 architectures (nano = 3, standard = 8).
pub const A2_VALID_CHANNELS: [u8; 2] = [3, 8];

use super::activations::ActivationType;
use super::film::FiLMConfig;
use super::gating::GatingMode;

/// Parameters for Head 1x1 configuration.
///
/// Configures an optional 1x1 convolution that sends output directly to the head
/// (skip connection) instead of using the activation output directly.
///
/// NOTE: reservado p/ motor A2 geral (futuro). O fast-path A2-Full/Lite
/// não utiliza head 1x1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Head1x1Params {
    /// Whether the head 1x1 convolution is active.
    pub active: bool,
    /// Number of output channels for the head 1x1 convolution.
    pub out_channels: usize,
    /// Number of groups for the grouped convolution.
    pub groups: u32,
}

/// Parameters for Layer 1x1 configuration.
///
/// Configures an optional 1x1 convolution that processes the activation output
/// for the residual connection to the next layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layer1x1Params {
    /// Whether the layer 1x1 convolution is active.
    pub active: bool,
    /// Number of groups for the grouped convolution.
    pub groups: u32,
}

/// Parameters for constructing a single WaveNet A2 layer.
///
/// Contains all configuration needed to instantiate a detailed layer.
///
/// Para o fast-path A2-Full/Lite (`a2_fast.cpp`), os seguintes campos assumem
/// valores fixos: `bottleneck == channels`, `gating_mode == GatingMode::None`,
/// `groups_* == 1`, `head1x1.active == false`, todos `*_film.active == false`.
/// Os campos `secondary_activation`, `head1x1`, `groups_*` e `*_film` estão
/// reservados p/ motor A2 geral (futuro).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerParamsA2 {
    /// Conditioning input size.
    pub condition_size: usize,
    /// Number of input/output channels between layers.
    pub channels: usize,
    /// Number of internal channels (bottleneck).
    pub bottleneck: usize,
    /// Kernel size for the dilated convolution.
    pub kernel_size: usize,
    /// Dilation factor for the convolution.
    pub dilation: usize,
    /// Primary activation function configuration.
    pub activation: ActivationType,
    /// Gating mode (None, Gated, or Blended).
    pub gating_mode: GatingMode,
    /// Number of groups for the input convolution.
    pub groups_input: u32,
    /// Number of groups for the input mixin convolution.
    pub groups_input_mixin: u32,
    /// Optional layer 1x1 convolution configuration.
    pub layer1x1: Layer1x1Params,
    /// Optional head 1x1 convolution configuration.
    pub head1x1: Head1x1Params,
    /// Secondary activation (used for gating/blending).
    pub secondary_activation: ActivationType,
    /// FiLM parameters before the input convolution.
    pub conv_pre_film: FiLMConfig,
    /// FiLM parameters after the input convolution.
    pub conv_post_film: FiLMConfig,
    /// FiLM parameters before the input mixin.
    pub input_mixin_pre_film: FiLMConfig,
    /// FiLM parameters after the input mixin.
    pub input_mixin_post_film: FiLMConfig,
    /// FiLM parameters before activation.
    pub activation_pre_film: FiLMConfig,
    /// FiLM parameters after activation.
    pub activation_post_film: FiLMConfig,
    /// FiLM parameters after the layer 1x1 convolution.
    pub layer1x1_post_film: FiLMConfig,
    /// FiLM parameters after the head 1x1 convolution.
    pub head1x1_post_film: FiLMConfig,
}

/// Parameters for constructing a WaveNet A2 Layer Array.
///
/// Configures multiple layers that share the same channel count
/// and kernel size, but can have distinct dilations and activations.
///
/// Para o fast-path A2-Full/Lite (`a2_fast.cpp`), os seguintes campos assumem
/// valores fixos: `bottleneck == channels`, todos `gating_modes == GatingMode::None`,
/// `groups_* == 1`, `head1x1.active == false`, todos `*_film.active == false`.
/// Os campos `secondary_activations`, `head1x1`, `groups_*` e `*_film` estão
/// reservados p/ motor A2 geral (futuro).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerArrayParamsA2 {
    /// Input size (number of channels).
    pub input_size: usize,
    /// Conditioning input size.
    pub condition_size: usize,
    /// Head output size (after rechannel).
    pub head_size: usize,
    /// Head rechannel convolution kernel size (>= 1).
    pub head_kernel_size: usize,
    /// Number of channels in each layer.
    pub channels: usize,
    /// Bottleneck size (internal channel count).
    pub bottleneck: usize,
    /// Kernel sizes per layer.
    pub kernel_sizes: Vec<usize>,
    /// Vector of dilation factors, one per layer.
    pub dilations: Vec<usize>,
    /// Vector of primary activation configurations, one per layer.
    pub activations: Vec<ActivationType>,
    /// Vectors of gating modes, one per layer.
    pub gating_modes: Vec<GatingMode>,
    /// Whether to use bias in the head rechannel.
    pub head_bias: bool,
    /// Number of groups for input convolutions.
    pub groups_input: u32,
    /// Number of groups for input mixin convolutions.
    pub groups_input_mixin: u32,
    /// Parameters for optional layer 1x1 convolutions.
    pub layer1x1: Layer1x1Params,
    /// Parameters for optional head 1x1 convolutions.
    pub head1x1: Head1x1Params,
    /// Vector of secondary activation configurations for gating/blending.
    pub secondary_activations: Vec<ActivationType>,
    /// FiLM parameters before input convolutions.
    pub conv_pre_film: FiLMConfig,
    /// FiLM parameters after input convolutions.
    pub conv_post_film: FiLMConfig,
    /// FiLM parameters before input mixins.
    pub input_mixin_pre_film: FiLMConfig,
    /// FiLM parameters after input mixins.
    pub input_mixin_post_film: FiLMConfig,
    /// FiLM parameters before activation.
    pub activation_pre_film: FiLMConfig,
    /// FiLM parameters after activation.
    pub activation_post_film: FiLMConfig,
    /// FiLM parameters after layer 1x1 convolutions.
    pub layer1x1_post_film: FiLMConfig,
    /// FiLM parameters after head 1x1 convolutions.
    pub head1x1_post_film: FiLMConfig,
}

/// Parameters for the optional post-stack Head.
///
/// Corresponds to the `Head` component of WaveNet in NAM.
///
/// NOTE: reservado p/ motor A2 geral (futuro). O fast-path A2-Full/Lite
/// utiliza apenas o *head conv* (`Conv1D(k=16, bias) × head_scale`),
/// não um post-stack Head multicamada.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadParams {
    /// Input channels (usually inherited from the last layer).
    pub in_channels: usize,
    /// Head internal channels.
    pub channels: usize,
    /// Final output channels.
    pub out_channels: usize,
    /// Kernel sizes of the head convolutions.
    pub kernel_sizes: Vec<usize>,
    /// Head activation configuration.
    pub activation: ActivationType,
}

#[cfg(test)]
#[path = "params_test.rs"]
mod tests;
