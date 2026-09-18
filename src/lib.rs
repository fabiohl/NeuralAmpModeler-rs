// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
// Every `unsafe` block must carry a `// SAFETY:` justification (enforced by clippy::undocumented_unsafe_blocks).
#![warn(clippy::undocumented_unsafe_blocks)]
#![doc = include_str!("../README.md")]

#[cfg(not(target_arch = "x86_64"))]
compile_error!("NeuralAmpModeler-rs requires x86_64 architecture");

#[cfg(not(any(
    doc,
    all(
        target_feature = "avx",
        target_feature = "avx2",
        target_feature = "bmi1",
        target_feature = "bmi2",
        target_feature = "f16c",
        target_feature = "fma",
        target_feature = "lzcnt",
        target_feature = "movbe"
    )
)))]
compile_error!(
    "NeuralAmpModeler-rs requires full x86-64-v3 target support \
     (avx, avx2, bmi1, bmi2, f16c, fma, lzcnt, movbe). \
     Compile with RUSTFLAGS=\"-Ctarget-cpu=x86-64-v3\""
);

/// Host-agnostic infrastructure: diagnostics, SPSC protocol, alloc audit, panic hooks.
pub mod common;

// API Surface Policy:
// Only deliberately chosen types are re-exported at the crate root. Internal
// infrastructure (SPSC protocol, RT status flags, alloc-audit) is
// accessible via its qualified path (neural_amp_modeler_rs::common::spsc::*).
// Do NOT add glob re-exports (pub use common::*) to this file.

/// Diagnostic and system support reporting for host applications.
pub use common::diagnostics::{DiagnosticBundle, SystemSnapshot};
/// Zero-allocation panic dump hook facility for crash reporting.
pub use common::panic_hook::install_panic_hook;
/// Global processing parameters and configuration mode enums.
pub use common::params::{
    ActivationPrecision, AdaptiveComputeMode, ProcessingParams, RtProcessingParams, SlimOverride,
};

/// Digital Signal Processing engine: oversampling, gate, resampler, cab-sim, pipelines.
pub mod dsp;
/// Model loader: parser and builder for `.nam` (JSON) and `.namb` (binary) formats.
pub mod loader;
/// Strongly-typed error returned by model loading operations.
pub use loader::LoadError;
/// Mathematical primitives: SIMD kernels, activations, GEMM, FFT, DSP utilities.
pub mod math;
/// Neural network architectures (WaveNet A1/A2, LSTM, ConvNet, Linear) and runtime dispatch.
pub mod models;
/// Opt-in real-time host hardening (Linux-only, never default).
#[cfg(all(feature = "rt-hardening", target_os = "linux"))]
#[cfg_attr(docsrs, doc(cfg(feature = "rt-hardening")))]
pub mod rt_hardening;

/// Convenience re-exports for the common inference and DSP pipeline.
///
/// Host applications can `use neural_amp_modeler_rs::prelude::*;` to bring the
/// core inference and signal processing types into scope without deep paths:
///
/// - **Diagnostics & Telemetry**: [`crate::SystemSnapshot`]
/// - **Processing Parameters**: [`crate::ActivationPrecision`], [`crate::ProcessingParams`], [`crate::RtProcessingParams`], [`crate::AdaptiveComputeMode`], [`crate::SlimOverride`]
/// - **Model Loading & Errors**: [`crate::loader::load_and_build_model`], [`crate::loader::LoadOptions`], [`crate::loader::LoadError`], [`crate::loader::NambError`], [`crate::loader::JsonError`], [`crate::loader::LoadedModelPair`]
/// - **Neural Models**: [`crate::models::NamModel`], [`crate::models::StaticModel`]
/// - **Noise Gate**: [`crate::dsp::gate::GateParams`], [`crate::dsp::gate::GateParamsBuilder`]
/// - **Cabinet Simulation**: [`crate::dsp::cabsim::adapter::CabSimAdapter`], [`crate::dsp::cabsim::conv::ConvEngine`], [`crate::dsp::cabsim::loader::CabSimIr`]
/// - **Oversampling Engine**: [`crate::dsp::oversample::OversampleEngine`], [`crate::dsp::oversample::OversampleFactor`]
/// - **Sample-Rate Resampling**: [`crate::dsp::resampler::NamResampler`]
/// - **Generic DSP Utilities**: [`crate::dsp::utils::DelayLine`]
///
/// The deep module paths remain available and unchanged; this module is purely additive.
///
/// # Examples
///
/// ```
/// use neural_amp_modeler_rs::prelude::*;
///
/// // Full pipeline components available directly from prelude:
/// let _: Option<ConvEngine> = None;
/// let _: Option<CabSimAdapter> = None;
/// let _: Option<CabSimIr> = None;
/// let _: Option<GateParams> = None;
/// let _: Option<GateParamsBuilder> = None;
/// let _: Option<SystemSnapshot> = None;
/// let _: Option<ActivationPrecision> = None;
/// let _: Option<AdaptiveComputeMode> = None;
/// let _: Option<SlimOverride> = None;
/// let _: Option<ProcessingParams> = None;
/// let _: Option<RtProcessingParams> = None;
/// let _: Option<LoadError> = None;
/// let _: Option<NambError> = None;
/// let _: Option<JsonError> = None;
/// let _: Option<LoadedModelPair> = None;
/// let _: Option<LoadOptions> = None;
/// let _: Option<OversampleEngine> = None;
/// let _: Option<OversampleFactor> = None;
/// let _: Option<NamResampler> = None;
/// let _: Option<DelayLine<f32>> = None;
/// let _: Option<Box<dyn NamModel>> = None;
/// let _: Option<StaticModel> = None;
/// let _ = load_and_build_model;
/// ```
pub mod prelude {
    pub use crate::common::diagnostics::SystemSnapshot;
    pub use crate::common::params::{
        ActivationPrecision, AdaptiveComputeMode, ProcessingParams, RtProcessingParams,
        SlimOverride,
    };
    pub use crate::dsp::cabsim::adapter::CabSimAdapter;
    pub use crate::dsp::cabsim::conv::ConvEngine;
    pub use crate::dsp::cabsim::loader::CabSimIr;
    pub use crate::dsp::gate::{GateParams, GateParamsBuilder};
    pub use crate::dsp::oversample::{OversampleEngine, OversampleFactor};
    pub use crate::dsp::resampler::NamResampler;
    pub use crate::dsp::utils::DelayLine;
    pub use crate::loader::{
        JsonError, LoadError, LoadOptions, LoadedModelPair, NambError, load_and_build_model,
    };
    pub use crate::models::{NamModel, StaticModel};
}

#[cfg(any(test, feature = "testing"))]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
/// Off-RT test utilities, perceptual metrics, and signal generators. Requires `testing` feature.
pub mod testing;

// Backward compatibility with older GLIBC versions (e.g. for enterprise or containerized Linux environments).
// Redirects math symbols to the stable GLIBC_2.2.5 version.
// Since external dependencies use these symbols, we declare global wrappers
// that intercept calls and jump (jmp) via PLT to the compatible versions.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
core::arch::global_asm!(
    ".global log10f",
    ".hidden log10f",
    ".type log10f, @function",
    "log10f:",
    "    jmp log10f_compat@PLT",
    ".symver log10f_compat, log10f@GLIBC_2.2.5",
    ".global atan2f",
    ".hidden atan2f",
    ".type atan2f, @function",
    "atan2f:",
    "    jmp atan2f_compat@PLT",
    ".symver atan2f_compat, atan2f@GLIBC_2.2.5",
    ".global acosf",
    ".hidden acosf",
    ".type acosf, @function",
    "acosf:",
    "    jmp acosf_compat@PLT",
    ".symver acosf_compat, acosf@GLIBC_2.2.5",
    // --- Group 2: Hidden-only symbols ---
    // cbrt, cbrtf, fma, fmod do not have problematic GLIBC version tags
    // (they resolve to GLIBC_2.2.5 without a versioned symbol conflict).
    // `.hidden` is sufficient to prevent the linker from emitting a versioned
    // dependency; no jmp trampoline or .symver redirect is needed.
    // If a future GLIBC version introduces a versioned variant of these
    // symbols, promote them to Group 1 (full jmp + .symver redirect).
    ".global cbrt",
    ".hidden cbrt",
    ".global cbrtf",
    ".hidden cbrtf",
    ".global fma",
    ".hidden fma",
    ".global fmod",
    ".hidden fmod"
);

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_prelude_exports_cabsim_and_all_core_types() {
        let _: Option<ConvEngine> = None;
        let _: Option<CabSimAdapter> = None;
        let _: Option<CabSimIr> = None;
        let _: Option<GateParams> = None;
        let _: Option<GateParamsBuilder> = None;
        let _: Option<SystemSnapshot> = None;
        let _: Option<ActivationPrecision> = None;
        let _: Option<AdaptiveComputeMode> = None;
        let _: Option<SlimOverride> = None;
        let _: Option<ProcessingParams> = None;
        let _: Option<RtProcessingParams> = None;
        let _: Option<LoadError> = None;
        let _: Option<NambError> = None;
        let _: Option<JsonError> = None;
        let _: Option<LoadedModelPair> = None;
        let _: Option<LoadOptions> = None;
        let _: Option<OversampleEngine> = None;
        let _: Option<OversampleFactor> = None;
        let _: Option<NamResampler> = None;
        let _: Option<DelayLine<f32>> = None;
        let _: Option<Box<dyn NamModel>> = None;
        let _: Option<StaticModel> = None;
        let _ = load_and_build_model;
    }
}
