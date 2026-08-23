// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Data structures for the `.nam` format (JSON).
//!
//! Contains the structs that model the neural model file.

use serde::{Deserialize, Serialize};

use super::validation::{
    deserialize_sample_rate, deserialize_submodels, deserialize_training, deserialize_weights,
};

/// Slimmable configuration extracted from per-layer metadata.
///
/// Mirrors the C++ NAMCore `slimmable` key in `.nam` layer objects.
/// Supports `method = "slice_channels_uniform"` with a `.kwargs.allowed_channels`
/// array of valid channel breakpoints.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub struct SlimmableConfig {
    /// The slimmable method — currently only `"slice_channels_uniform"` is supported.
    pub method: Option<String>,
    /// Keyword arguments for the slimmable method.
    pub kwargs: Option<SlimmableKwargs>,
}

/// Keyword arguments for a slimmable method.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub struct SlimmableKwargs {
    /// Valid channel breakpoints (must be strictly descending, non-zero).
    pub allowed_channels: Option<Vec<usize>>,
}

/// Structure representing a date and time associated with the model's metadata.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct NamDate {
    /// Year.
    pub year: Option<i32>,
    /// Month.
    pub month: Option<i32>,
    /// Day.
    pub day: Option<i32>,
    /// Hour.
    pub hour: Option<i32>,
    /// Minute.
    pub minute: Option<i32>,
    /// Second.
    pub second: Option<i32>,
}

/// Optional metadata contained at the end of the `.nam` format.
#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct NamMetadata {
    /// Model authorship or export date.
    pub date: Option<NamDate>,
    /// The model name.
    pub name: Option<String>,
    /// Who made/trained the model.
    pub modeled_by: Option<String>,
    /// Manufacturer of the original equipment (e.g. Fender).
    pub gear_make: Option<String>,
    /// The model of the original equipment (e.g. Deluxe Reverb).
    pub gear_model: Option<String>,
    /// What type of equipment is this? Options: "amp", "pedal", "pedal_amp", "amp_cab", "amp_pedal_cab", "preamp", and "studio".
    pub gear_type: Option<String>,
    /// What style of equipment? Options: "clean", "overdrive", "crunch", "hi_gain", and "fuzz".
    pub tone_type: Option<String>,
    /// Optional documentation about Pydantic training configuration.
    #[serde(default, deserialize_with = "deserialize_training")]
    pub training: Option<serde_json::Value>,
    /// Expected input level for the model (dBu). Used in input gain staging.
    pub input_level_dbu: Option<f32>,
    /// Expected output level for the model (dBu). Used in output gain staging.
    pub output_level_dbu: Option<f32>,
    /// Overall recorded loudness.
    pub loudness: Option<f32>,
}

/// Helper struct for deserializing `NamLayerConfig` — mirrors the JSON shape
/// and captures all typed fields while also storing the raw JSON value for
/// complex nested checks (activation arrays, FiLM keys, etc.).
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct NamLayerConfigHelper {
    input_size: Option<usize>,
    condition_size: Option<usize>,
    head_size: Option<usize>,
    channels: Option<usize>,
    kernel_size: Option<usize>,
    kernel_sizes: Option<Vec<usize>>,
    dilations: Option<Vec<usize>>,
    activation: Option<serde_json::Value>,
    gated: Option<bool>,
    head_bias: Option<bool>,
    bottleneck: Option<usize>,
    slimmable: Option<SlimmableConfig>,
}

/// The structural configuration of a single layer of the network (whether WaveNet or LSTM).
#[derive(Serialize, Debug, Clone, Default)]
pub struct NamLayerConfig {
    /// Optional: Input tensor size.
    pub input_size: Option<usize>,
    /// Optional: Conditioning tensor size (e.g. external parameters).
    pub condition_size: Option<usize>,
    /// Optional: Output tensor size (head size).
    pub head_size: Option<usize>,
    /// Optional: Number of internal channels (e.g. 16 or 24).
    pub channels: Option<usize>,
    /// Optional: Convolutional kernel size.
    pub kernel_size: Option<usize>,
    /// Optional: Per-layer kernel sizes (A2 architecture — 23 elements).
    pub kernel_sizes: Option<Vec<usize>>,
    /// Optional: Array of dilation factors.
    pub dilations: Option<Vec<usize>>,
    /// Optional: Activation function (e.g. "Tanh" for A1, array of objects for A2 LeakyReLU).
    pub activation: Option<String>,
    /// Optional: Whether the architecture uses gating.
    pub gated: Option<bool>,
    /// Optional: Whether the processing head has bias.
    pub head_bias: Option<bool>,
    /// Optional: Bottleneck size (internal channel count for A2).
    pub bottleneck: Option<usize>,
    /// Optional: Slimmable configuration per layer ("method", "kwargs.allowed_channels").
    pub slimmable: Option<SlimmableConfig>,
    /// Raw JSON value for this layer, preserved for complex shape
    /// checks (activation arrays, FiLM keys, condition_dsp, etc.).
    /// `None` when the struct is constructed directly (not via JSON deserialization).
    #[serde(skip)]
    pub layer_raw: Option<serde_json::Value>,
}

impl NamLayerConfig {
    /// Parses the activation configuration from the preserved raw JSON.
    ///
    /// Extracts per-layer `activation`, `gating_mode`, and `secondary_activation`
    /// arrays for `num_layers` expected entries (23 for standard A2).
    /// Returns `None` when `layer_raw` is absent or the `activation` array
    /// is missing/invalid.
    ///
    /// This is the primary entry point for the dynamic A2 engine
    /// to obtain the heterogeneous activation configuration at model build time.
    pub fn parse_activation_config(
        &self,
        num_layers: usize,
    ) -> Option<super::activation_parser::LayerActivationConfig> {
        let raw = self.layer_raw.as_ref()?;
        super::activation_parser::parse_layer_activations(raw, num_layers)
    }
}

impl<'de> Deserialize<'de> for NamLayerConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw_value = serde_json::Value::deserialize(deserializer)?;
        let helper: NamLayerConfigHelper =
            serde_json::from_value(raw_value.clone()).map_err(serde::de::Error::custom)?;

        let activation = match helper.activation {
            Some(serde_json::Value::String(s)) => Some(s),
            Some(serde_json::Value::Array(_)) => None,
            Some(serde_json::Value::Object(_)) => None,
            Some(serde_json::Value::Null) | None => None,
            _ => {
                return Err(serde::de::Error::custom(
                    "unsupported activation format: expected a string, array, object, or null",
                ));
            }
        };

        Ok(NamLayerConfig {
            input_size: helper.input_size,
            condition_size: helper.condition_size,
            head_size: helper.head_size,
            channels: helper.channels,
            kernel_size: helper.kernel_size,
            kernel_sizes: helper.kernel_sizes,
            dilations: helper.dilations,
            activation,
            gated: helper.gated,
            head_bias: helper.head_bias,
            bottleneck: helper.bottleneck,
            slimmable: helper.slimmable,
            layer_raw: Some(raw_value),
        })
    }
}

/// Implementation mode for Linear architecture convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinearImplementation {
    /// Auto-select based on receptive field size.
    #[default]
    Auto,
    /// Direct time-domain convolution (full receptive field dot product).
    Direct,
    /// FFT partitioned convolution (zero-latency hybrid: direct head + FFT tail).
    Fft,
}

impl LinearImplementation {}

impl std::str::FromStr for LinearImplementation {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        match lower.as_str() {
            "auto" => Ok(Self::Auto),
            "direct" => Ok(Self::Direct),
            "fft" | "partitioned_fft" | "partitioned-fft" => Ok(Self::Fft),
            "legacy" | "old" => Ok(Self::Auto),
            _ => Err(()),
        }
    }
}

/// Weight layout options supported in the `.namb` format.
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum WeightsLayout {
    /// Original layout (standard NAM): `[Gate][H][IH]` for LSTM, `[OUT][IN][K]` for Conv1D.
    #[default]
    Original = 0,
    /// Layout optimized for LSTM: `[Gate][IH][H]`.
    GateMajorLstm = 1,
    /// Layout optimized for WaveNet: Interleaved 4-Wide (`[OUT/4][K][IN][4]`).
    Interleaved4WaveNet = 2,
}

/// Configuration for the post-stack head sub-object (WaveNet / ConvNet).
///
/// Mirrors the `_Head` structure in NAMCore's `convnet.h:108-118`.
/// Contains a Conv1D + activation that processes the signal after the
/// stack of blocks, before the output.
#[derive(Serialize, Debug, Clone, PartialEq, Default)]
pub struct HeadConfig {
    /// Number of internal channels for the head Conv1D.
    pub channels: Option<usize>,
    /// Whether the head Conv1D has a bias term.
    pub bias: Option<bool>,
    /// Number of output channels (typically 1 for mono).
    pub out_channels: Option<usize>,
    /// Activation function name (e.g. "Tanh", "ReLU").
    pub activation: Option<String>,
    /// Kernel size for the head Conv1D.
    pub kernel_size: Option<usize>,
}

/// The internal configuration of the architecture node in the JSON.
#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct NamConfig {
    /// List of stacked layer configurations (present in WaveNet, ConvNet; absent in LSTM).
    #[serde(default)]
    pub layers: Vec<NamLayerConfig>,
    /// Number of input channels (WaveNet A2). Defaults to 1 when absent.
    #[serde(default)]
    pub in_channels: Option<usize>,
    /// Number of output channels. Defaults to 1 when absent (NeuralAmpModeler-rs only supports mono).
    #[serde(default)]
    pub out_channels: Option<usize>,
    /// Post-stack head sub-object (WaveNet with head / ConvNet).
    /// `null` in JSON becomes `Some(Value::Null)` (head present but empty).
    /// Absent in JSON becomes `None`.
    /// Non-null values contain the head configuration object.
    pub head: Option<serde_json::Value>,
    /// Fine scale over the network summation (WaveNet / ConvNet).
    pub head_scale: Option<f32>,
    /// Number of layers (for LSTMs in C++ it is the layer count, or explicit)
    pub num_layers: Option<usize>,
    /// Hidden size of the LSTM cell
    pub hidden_size: Option<usize>,
    /// Receptive field size (Linear architecture only).
    pub receptive_field: Option<usize>,
    /// Whether a bias scalar is included (Linear architecture only).
    pub bias: Option<bool>,
    /// Submodel entries for SlimmableContainer architecture.
    /// Each entry is `{"max_value": f32, "model": { ... full inner .nam JSON ... }}`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_submodels"
    )]
    pub submodels: Option<Vec<serde_json::Value>>,
    /// Nested condition DSP sub-model (raw JSON).
    ///
    /// Self-contained `.nam` model with its own `version`, `architecture`,
    /// `config`, `weights`, and `sample_rate`. Parsed and built independently
    /// before the main model's layer arrays are processed. The sub-model's
    /// output channels serve as the condition input for the main model's layers
    /// (C++ `_process_condition()` mirror, wavenet/model.cpp:692-722).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_dsp: Option<serde_json::Value>,
    /// Linear convolution implementation mode ("Auto", "Direct", "Fft").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation: Option<String>,
    /// ConvNet C++ flat format: scalar channel count (config root level).
    #[serde(default, rename = "channels")]
    pub conv_channels: Option<usize>,
    /// ConvNet C++ flat format: global dilations array (one per block).
    #[serde(default, rename = "dilations")]
    pub conv_dilations: Option<Vec<usize>>,
    /// ConvNet C++ flat format: global batchnorm flag.
    #[serde(default, rename = "batchnorm")]
    pub conv_batchnorm: Option<bool>,
    /// ConvNet C++ flat format: per-block activation function string (config root level).
    #[serde(default, rename = "activation")]
    pub conv_activation: Option<String>,
}

impl NamConfig {
    /// Extracts a typed `HeadConfig` from the raw `head` JSON value.
    ///
    /// Returns `None` if the head field is absent, `null`, or not an object.
    pub fn parse_head(&self) -> Option<HeadConfig> {
        let val = self.head.as_ref()?;
        if val.is_null() || !val.is_object() {
            return None;
        }
        let channels = val
            .get("channels")
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok());
        let bias = val.get("bias").and_then(|v| v.as_bool());
        let out_channels = val
            .get("out_channels")
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok());
        let activation = val
            .get("activation")
            .and_then(|v| v.as_str())
            .map(String::from);
        let kernel_size = val
            .get("kernel_size")
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok());
        Some(HeadConfig {
            channels,
            bias,
            out_channels,
            activation,
            kernel_size,
        })
    }
}

/// Root mapping structure for `.nam` files.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NamModelData {
    /// Version in the JSON header (e.g. "0.5.4")
    pub version: Option<String>,
    /// Declared architecture type ("WaveNet", "LSTM", "ConvNet", "Linear", "SlimmableContainer")
    pub architecture: String,
    /// Structural configuration of hyperparameters
    pub config: NamConfig,
    /// The huge Float32 tensors flattened in SoA format.
    #[serde(deserialize_with = "deserialize_weights")]
    pub weights: Vec<f32>,
    /// Original sample rate projected by the modeling (always ideal reference 48 kHz).
    #[serde(default, deserialize_with = "deserialize_sample_rate")]
    pub sample_rate: Option<f32>,
    /// Extra physical-acoustic properties associated.
    pub metadata: Option<NamMetadata>,
    /// Weight layout (used only in the .namb v2+ binary format).
    #[serde(skip)]
    pub weights_layout: WeightsLayout,
}

impl NamModelData {
    /// Returns whether the model explicitly declares slimmability support in its architecture or metadata.
    pub fn is_slimmable_capable(&self) -> bool {
        if self.architecture == "SlimmableContainer" {
            return true;
        }
        if self.architecture == "WaveNet"
            && self.config.layers.iter().any(|l| l.slimmable.is_some())
        {
            return true;
        }
        if let Some(ref meta) = self.metadata
            && let Some(ref training) = meta.training
            && training
                .get("slimmable")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        {
            return true;
        }
        false
    }
}

#[cfg(test)]
#[path = "model_test.rs"]
mod tests;
