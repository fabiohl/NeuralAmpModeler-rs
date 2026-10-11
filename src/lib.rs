// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
// Every `unsafe` block must carry a `// SAFETY:` justification (enforced by clippy::undocumented_unsafe_blocks).
#![warn(clippy::undocumented_unsafe_blocks)]
// Rustdoc resolves intra-doc links in a module's inner docs against the parent
// module scope on the stable toolchain, but against the module's own scope on
// nightly (docs.rs). A short label that references an imported item therefore
// needs an explicit target to resolve on stable, while the same target is
// flagged as redundant on nightly. Keep a single source that is warning-free on
// both toolchains by allowing the nightly-only redundancy lint; explicit
// targets remain so stable documentation keeps resolving these links.
#![allow(rustdoc::redundant_explicit_links)]
//! # NeuralAmpModeler-rs
//!
//! `NeuralAmpModeler-rs` is a high-performance, low-latency real-time neural
//! inference DSP engine written in pure Rust. It loads, builds, and executes
//! [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) models —
//! WaveNet (A1/A2), LSTM, ConvNet, Linear, and Sequential — together with
//! speaker-cabinet impulse-response convolution, multi-rate sinc resampling,
//! and polyphase anti-aliasing oversampling.
//!
//! The engine is host-agnostic: its public API is decoupled from any plugin
//! wrapper or audio server, and it guarantees strict zero heap allocation,
//! zero mutex locks, and zero blocking system calls on the real-time audio
//! thread.
//!
//! ## Feature highlights
//!
//! * **Strict zero-allocation real-time safety** — deterministic audio
//!   callbacks with zero heap drops, locks, or blocking syscalls, audited via a
//!   counting allocator. Parameter and model hot-swaps travel through lock-free
//!   SPSC channels backed by a three-tier garbage-collection cascade.
//! * **`x86-64-v3` SIMD baseline** — mandatory AVX2/FMA/BMI2 vectorization with
//!   hand-written SIMD kernels, tap-major memory layouts, and compile-time
//!   monomorphized dispatch with no vtables or indirect calls.
//! * **Dual-oracle numerical parity** — validated against canonical C++ NAMCore
//!   and a double-precision `f64` oracle; per-model baselines are pinned in
//!   [`docs/quality-contract.json`](https://github.com/fabiohl/NeuralAmpModeler-rs/blob/main/docs/quality-contract.json).
//! * **Complete native DSP stack** — polyphase FIR sinc resampler, UPOLS FFT
//!   cabinet convolution, and multi-stage half-band anti-aliasing oversampling,
//!   with no external audio crates.
//!
//! ## Quick start
//!
//! Add the crate to `Cargo.toml`. The published package is
//! `NeuralAmpModeler-rs`, but the imported library name is
//! `neural_amp_modeler_rs`:
//!
//! ```toml
//! [dependencies]
//! NeuralAmpModeler-rs = "1.0"
//! ```
//!
//! For test harnesses, audio generators, and perceptual fidelity tooling:
//!
//! ```toml
//! [dependencies]
//! NeuralAmpModeler-rs = { version = "1.0", features = ["testing"] }
//! ```
//!
//! ```no_run
//! use std::path::Path;
//! use neural_amp_modeler_rs::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 1. Capture the host hardware capabilities (SIMD features, CPU topology).
//!     let sys = SystemSnapshot::capture();
//!
//!     // 2. Load a `.nam` (JSON) or `.namb` (binary) model.
//!     let mut model_pair = load_and_build_model(
//!         Path::new("path/to/model.nam"),
//!         &sys,
//!         false, // dual_mono: false = mono, left-channel model only
//!         LoadOptions::default(),
//!     )?;
//!
//!     // 3. Process an audio block on the real-time thread (zero allocations, zero locks).
//!     let input_buffer = [0.0_f32; 64];
//!     let mut output_buffer = [0.0_f32; 64];
//!     if let Some(ref mut model) = model_pair.model_l {
//!         model.process(&input_buffer, &mut output_buffer);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! Model files are distributed by the community (see the official
//! Neural Amp Modeler site and TONE3000). To run the engine without any file
//! on disk, the crate ships a fully in-memory
//! [`synthetic_model`](https://github.com/fabiohl/NeuralAmpModeler-rs/blob/main/examples/synthetic_model.rs)
//! example (`cargo run --example synthetic_model`) that builds a deterministic
//! LSTM model through the public API.
//!
//! ## Feature flags
//!
//! | Feature | Default | Description |
//! |:------- |:------- |:----------- |
//! | `dual-mono` | **Enabled** | Independent per-channel neural inference for stereo/dual-mono signals. Disable for lean single-channel mode. |
//! | `heap-audit` | Disabled | Real-time allocation tracking via a counting allocator to enforce zero-heap invariants. |
//! | `testing` | Disabled | Off-RT test utilities, signal generators, synthetic fixtures, and perceptual fidelity oracles. |
//! | `fft-radix4-planner` | Disabled | Radix-4 DIT FFT execution planning routines and benchmarks. |
//! | `rt-hardening` | Disabled | Opt-in Linux real-time host hardening (THP, `mlockall`, `SCHED_FIFO`, MXCSR DAZ/FTZ, CPU affinity). |
//! | `avx512` | Disabled | Experimental AVX-512 kernels and dispatch for research and cross-ISA validation. |
//!
//! ## Module map
//!
//! | Module | Purpose |
//! |:------ |:------- |
//! | [`loader`] | `.nam` (JSON) and `.namb` (binary) model parsing and construction. |
//! | [`math`] | SIMD math primitives, activation approximations, GEMV/FFT kernels. |
//! | [`models`] | Neural network architectures and [`StaticModel`](crate::models::StaticModel) dispatch. |
//! | [`dsp`] | Resampler, noise gate, oversampling, and cabinet simulation. |
//! | [`common`] | Telemetry, diagnostics, panic hooks, and lock-free SPSC channels. |
//! | [`prelude`] | Convenience re-exports for the common inference and DSP pipeline. |
//! | `testing` | Off-RT test utilities and perceptual metrics (feature `testing`). |
//! | `rt_hardening` | Linux real-time host hardening (feature `rt-hardening`). |
//!
//! ## Platform support
//!
//! The crate targets **`x86_64` with the `x86-64-v3` baseline** (AVX2, FMA,
//! BMI2, and the rest of the v3 feature set) and refuses to compile otherwise
//! via `compile_error!`. Other architectures (for example macOS on Apple
//! silicon/aarch64) are unsupported by design. When building documentation
//! locally without the full feature set, prefer `cargo doc --lib`.
//!
//! ## Further documentation
//!
//! * [Architecture and SIMD kernels](https://github.com/fabiohl/NeuralAmpModeler-rs/blob/main/docs/architecture.md)
//! * [Audio fidelity map and quality modes](https://github.com/fabiohl/NeuralAmpModeler-rs/blob/main/docs/audio_fidelity_map.md)
//! * [Benchmarks and regression envelopes](https://github.com/fabiohl/NeuralAmpModeler-rs/blob/main/docs/benchmarks.md)
//! * [Testing and verification policies](https://github.com/fabiohl/NeuralAmpModeler-rs/blob/main/docs/testing.md)

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("NeuralAmpModeler-rs requires the x86_64 or aarch64 architecture");

#[cfg(all(target_arch = "aarch64", not(any(doc, target_feature = "neon"))))]
compile_error!("NeuralAmpModeler-rs requires NEON on aarch64");

#[cfg(all(
    target_arch = "x86_64",
    not(any(
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
    ))
))]
compile_error!(
    "NeuralAmpModeler-rs requires full x86-64-v3 target support \
     (avx, avx2, bmi1, bmi2, f16c, fma, lzcnt, movbe). \
     Compile with RUSTFLAGS=\"-Ctarget-cpu=x86-64-v3\""
);

#[cfg(all(feature = "avx512", not(target_arch = "x86_64")))]
compile_error!("the `avx512` feature is only available on x86_64");

/// Architecture-neutral SIMD intrinsic surface (x86_64 native, aarch64 NEON).
mod arch;

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
/// - **Diagnostics & Telemetry**: [`crate::SystemSnapshot`], [`crate::common::diagnostics::NamErrorCode`]
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
/// let _: Option<NamErrorCode> = None;
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
/// let _ = load_and_build_model_from_bytes;
/// let _ = load_and_build_model_from_bytes_named;
/// ```
///
/// # Error propagation
///
/// [`LoadError`] and [`NamErrorCode`](crate::common::diagnostics::NamErrorCode)
/// both implement [`std::error::Error`], so
/// they propagate through `?` into a boxed error without manual mapping (the
/// same holds for `anyhow::Error`):
///
/// ```
/// use neural_amp_modeler_rs::prelude::*;
///
/// fn load(bytes: &[u8]) -> Result<LoadedModelPair, Box<dyn std::error::Error>> {
///     let sys = SystemSnapshot::capture();
///     // `LoadError` is converted into the boxed trait object by `?`.
///     let pair = load_and_build_model_from_bytes(bytes, &sys, false, LoadOptions::default())?;
///     Ok(pair)
/// }
///
/// assert!(load(br#"{"version":"0.7.0","architecture":"Linear",
///     "config":{"receptive_field":1,"bias":false},"weights":[1.0]}"#).is_ok());
/// assert!(load(b"not a model").is_err());
/// ```
pub mod prelude {
    pub use crate::common::diagnostics::{NamErrorCode, SystemSnapshot};
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
        load_and_build_model_from_bytes, load_and_build_model_from_bytes_named,
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
#[path = "lib_test.rs"]
mod lib_test;
