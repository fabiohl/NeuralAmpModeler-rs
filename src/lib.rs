// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![warn(missing_docs)]
// Every `unsafe` block must carry a `// SAFETY:` justification (T6.1/H-02).
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

/// Convenience re-exports for the common inference pipeline.
///
/// Host applications can `use neural_amp_modeler_rs::prelude::*;` to bring the
/// core inference types into scope without deep paths: [`crate::SystemSnapshot`],
/// [`crate::loader::load_and_build_model`], [`crate::loader::LoadOptions`],
/// [`crate::models::NamModel`], [`crate::models::StaticModel`],
/// [`crate::dsp::oversample::OversampleEngine`],
/// [`crate::dsp::oversample::OversampleFactor`],
/// [`crate::dsp::resampler::NamResampler`], and
/// [`crate::dsp::cabsim::loader::CabSimIr`]. The deep module paths remain
/// available and unchanged; this module is purely additive.
pub mod prelude {
    pub use crate::common::diagnostics::SystemSnapshot;
    pub use crate::dsp::cabsim::loader::CabSimIr;
    pub use crate::dsp::oversample::{OversampleEngine, OversampleFactor};
    pub use crate::dsp::resampler::NamResampler;
    pub use crate::loader::{LoadError, LoadOptions, load_and_build_model};
    pub use crate::models::{NamModel, StaticModel};
}

#[cfg(any(test, feature = "testing"))]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
/// Off-RT test utilities, perceptual metrics, and signal generators. Requires `testing` feature.
pub mod testing;

// Backward compatibility with older GLIBC versions (e.g. for Flatpak/Bitwig).
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
    ".global cbrt",
    ".hidden cbrt",
    ".global cbrtf",
    ".hidden cbrtf",
    ".global fma",
    ".hidden fma",
    ".global fmod",
    ".hidden fmod"
);
