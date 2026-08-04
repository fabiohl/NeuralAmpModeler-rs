// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::dsp::adaptive::SlimOverride;

/// SPSC payload sent from the Host (CLI/UI) to the DSP Thread.
/// Aligned to 128 bytes to mitigate False Sharing.
#[repr(align(128))]
pub enum ParamPayload {
    /// Injects the input gain as a linear multiplier.
    InputGain(f32),
    /// Injects the output gain as a linear multiplier.
    OutputGain(f32),
    /// Loads the decoded mathematical topology, also informing the thresholds
    /// expected by the model creator (resolved from input_level_dbu and loudness tags).
    /// The pointer ensures zero-allocation (no-heap) and deterministic initialization.
    LoadModel {
        /// The encapsulated model for neural inference (Left Channel)
        model_l: Option<Box<crate::models::StaticModel>>,
        /// The encapsulated model for neural inference (Right Channel)
        model_r: Option<Box<crate::models::StaticModel>>,
        /// Expected input gain adjustment as a linear multiplier.
        input_mult_adj: f32,
        /// Expected output gain adjustment as a linear multiplier.
        output_mult_adj: f32,
        /// Sample rate required by the model (usually 48000).
        sample_rate: u32,
    },
    /// Injects the Silence/Mono Gate settings.
    GateConfig(crate::dsp::gate::GateParams),
    /// Sets the manual slim override quality level.
    SlimOverride(SlimOverride),
    /// Sets the oversampling factor for the neural stage.
    SetOversample(crate::dsp::oversample::OversampleFactor),
}
