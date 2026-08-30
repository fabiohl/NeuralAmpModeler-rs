// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::dsp::adaptive::SlimOverride;

/// Versioned envelope for a pre-built `NamResampler` transported over the
/// dedicated Main→RT SPSC channel (F-RB-004).
///
/// `generation` is the `requested_rate_generation` captured by the main thread
/// at build time. The RT drain installs the payload only when its generation
/// still matches the current request; stale envelopes are sent to the GC
/// cascade without unmuting the callback, so a rate renegotiation published
/// during a rebuild can never be silently lost.
pub struct ResamplerSwapPayload {
    /// Request generation this resampler was built for.
    pub generation: u64,
    /// The freshly built resampler (owned; transferred to the RT side).
    pub resampler: Box<crate::dsp::resampler::NamResampler>,
    /// Pre-allocated streaming adapter (strict cardinality) built for the target rate.
    pub stream: Box<crate::dsp::resampling::StreamingResampleBuffer>,
}

/// Versioned envelope for a pre-built `CabSimPair` (or bypass `None`) transported over the
/// dedicated Main→RT SPSC channel (F-RB-004 / T7.1).
pub struct CabSimSwapPayload {
    /// Request generation this cabsim pair was built for (from `RtStatusFlags::requested_cabsim_generation`).
    pub generation: u64,
    /// Cabsim pair (Left and Right adapters) or `None` to bypass/clear cabsim.
    pub pair: Option<Box<crate::dsp::cabsim::adapter::CabSimPair>>,
}

/// Atomic stereo/mono bundle of slimmable WaveNet channel models transported
/// over the dedicated Main→RT SPSC channel (F-RB-005).
///
/// L and R are sliced, prewarmed, and pushed **together** in a single envelope
/// so the RT drain can perform an all-or-nothing swap: `active_model_l` and
/// `active_model_r` always belong to the same generation and channel count.
/// `r` is `None` for mono configurations. If the channel is full, neither
/// channel is delivered and the rebuild flag stays armed for a full retry.
pub struct SlimModelPair {
    /// Slim rebuild generation this pair was built for (from
    /// `RtStatusFlags::requested_slimmable_generation`).
    pub generation: u64,
    /// Channel count the L/R models were sliced to.
    pub channels: usize,
    /// Left-channel model (`None` when retired/drained to GC).
    pub l: Option<Box<crate::models::StaticModel>>,
    /// Right-channel model (`None` for mono configurations or when retired/drained to GC).
    pub r: Option<Box<crate::models::StaticModel>>,
}

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
