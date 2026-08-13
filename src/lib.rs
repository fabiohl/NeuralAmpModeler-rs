// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![warn(missing_docs)]

//! # Neural Amp Modeler (NAM-rs) — DSP Core Library
//!
//! **NeuralAmpModeler-rs** is the pure DSP kernel for high-performance,
//! real-time inference of [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/)
//! neural network models (WaveNet A1 and A2, LSTM, ConvNet, Linear) and
//! impulse response (.wav) convolutions.
//!
//! This is an independent public library for the wider audio and Rust
//! communities. Its APIs and policies are host-agnostic: no particular
//! application, audio backend, plugin format, or downstream consumer owns
//! its architecture. Integration-specific behavior belongs in downstream
//! crates.
//!
//! The library provides:
//! - **SIMD-Accelerated Inference Kernels**: Native `x86-64-v3` (AVX2 + FMA)
//!   and `AVX-512` math routines.
//! - **Flexible Model Loader**: Parser and builder for `.nam` (JSON) and
//!   `.namb` (binary profile) files.
//! - **Lock-Free DSP Engine**: Zero heap allocations on the audio
//!   processing hot-path.
//!
//! ---
//!
//! ## 🛠 Feature Flags & Recommended Dependency Setup
//!
//! > ⚠️ **Important for Library Integrators:**
//! > By default no feature flags are enabled. Enable only what you need.
//!
//! ### `Cargo.toml` Configuration Examples
//!
//! **Pure Core DSP & Model Inference:**
//! ```toml
//! [dependencies]
//! NeuralAmpModeler-rs = "x.y.z"
//! ```
//!
//! **Adding Off-RT Testing & Audio Signal Generators:**
//! ```toml
//! [dependencies]
//! NeuralAmpModeler-rs = { version = "x.y.z", features = ["testing"] }
//! ```
//!
//! ### Feature Flags Summary Table
//!
//! | Feature Flag     | Default | Description                                                        |
//! |:---------------- |:-------:|:------------------------------------------------------------------ |
//! | `stereo`         | No      | Enables multi-channel / stereo dual-model loader support.          |
//! | `testing`        | No      | Exposes off-RT test utilities, audio signal generators, and        |
//! |                  |         | perceptual metrics (`testing` module).                             |
//! | `heap-audit`     | No      | Enables heap-allocation auditing infrastructure.                   |
//! | `long_bench`     | No      | Enables long-form inference benchmarks.                            |
//! | `dynamic-engine` | No      | Enables generic dynamic-dimension fallback execution paths for     |
//! |                  |         | arbitrary non-standard model topologies.                           |
//!
//! ---
//!
//! ## 🚀 Quick Start — Loading & Building a Model
//!
//! The [`loader::load_and_build_model`] function reads `.nam` (JSON) or
//! `.namb` (binary) files and constructs optimized [`models::StaticModel`]
//! instances ready for real-time execution.
//!
//! ```no_run
//! use std::path::Path;
//! use neural_amp_modeler_rs::loader::{load_and_build_model, LoadOptions};
//! use neural_amp_modeler_rs::SystemSnapshot;
//!
//! // Capture system capabilities (SIMD feature set, CPU topology)
//! let sys = SystemSnapshot::capture();
//!
//! // Load a model file (.nam or .namb)
//! let model_pair = load_and_build_model(
//!     Path::new("path/to/model.nam"),
//!     &sys,
//!     false, // mono execution (set true for stereo)
//!     LoadOptions::default(),
//! )
//! .expect("Failed to load model");
//!
//! println!("Loaded {} model", model_pair.architecture);
//! assert!(model_pair.model_l.is_some());
//! assert!(model_pair.model_r.is_none()); // Mono load: right channel is None
//! assert!(model_pair.sample_rate > 0);
//! ```
//!
//! ---
//!
//! ## ⚡ FastMath & SIMD Vector Activations
//!
//! High-performance scalar activation functions are available directly in
//! [`math::activations`]:
//!
//! ```rust
//! use neural_amp_modeler_rs::math::activations::{tanh, sigmoid};
//!
//! // Padé [5,4] rational approximant, clamped to [-1.0, 1.0]
//! let t = tanh(1.0);
//! assert!((t - 0.761594).abs() < 1e-3);
//! assert!(tanh(10.0) <= 1.0);
//! assert!(tanh(-10.0) >= -1.0);
//!
//! // Degree-17 minimax polynomial, clamped to [0.0, 1.0]
//! let s = sigmoid(0.0);
//! assert!((s - 0.5).abs() < 1e-2);
//! assert!(sigmoid(10.0) > 0.999);
//! assert!(sigmoid(-10.0) < 0.001);
//! ```
//!
//! For slice-based processing that automatically selects vectorized SIMD
//! kernels (AVX2/AVX-512), use `tanh_slice` and `sigmoid_slice` from
//! [`math::activations`].
//!
//! ---
//!
//! ## 🗺 Crate Module Map
//!
//! | Module     | Purpose                                            | Key Entry Points & Types                               |
//! |:---------- |:-------------------------------------------------- |:------------------------------------------------------ |
//! | [`loader`] | Model deserialization & construction (`.nam`,      | [`loader::load_and_build_model`],                      |
//! |            | `.namb`)                                           | [`loader::LoadOptions`]                                |
//! | [`math`]   | Mathematical primitives, SIMD kernels, &           | [`math::activations`]                                  |
//! |            | activations                                        |                                                        |
//! | [`models`] | Neural network architectures & static topologies   | [`models::StaticModel`], WaveNet, LSTM, ConvNet        |
//! | [`dsp`]    | Digital signal processing engine & oversampling    | [`dsp::gate::GateParams`],                             |
//! |            |                                                    | [`dsp::oversample::OversampleEngine`]                  |
//! | [`common`] | Host-agnostic infrastructure & SPSC protocol     | [`common::spsc::RtStatusFlags`],                   |
//! |            | *(Advanced API — qualified paths)*                 | [`common::alloc_audit`]                                |
//! | `testing`  | Off-RT test utilities & perceptual metrics         | Audio validation and f64 Oracles                       |
//! |            | (requires `testing`)                               |                                                        |
//!
//! ---
//!
//! ## 🛡 Real-Time Safety & Performance Guarantees
//!
//! NeuralAmpModeler-rs is engineered for **absolute real-time safety** on
//! the audio processing thread (`SCHED_FIFO`). The following guarantees
//! are enforced at the architecture level:
//!
//! ### 1. Zero Heap Allocations on Hot-Path
//! Heap objects (`Box`, `Vec`, `Arc`, `String`) are **never** allocated or
//! dropped on the real-time audio thread. All dynamic resources are
//! allocated off-RT and swapped via lock-free SPSC channels
//! ([`common::spsc`]). Compile-time allocation auditing is available via
//! [`common::alloc_audit`].
//!
//! ### 2. Zero Blocking I/O
//! No `println!`, `eprintln!`, `format!`, file I/O, or blocking
//! synchronization primitives are permitted on the RT thread. State
//! transitions are signaled atomically via [`common::spsc::RtStatusFlags`].
//!
//! ### 3. Denormal Protection (FTZ + DAZ)
//! Subnormal (denormal) floating-point numbers cause severe performance
//! degradation (up to 100× slowdown). The audio processing entry point
//! ([`dsp::pipeline::capture_dsp_pipeline`]) reasserts **Flush-To-Zero (FTZ)**
//! and **Denormals-Are-Zero (DAZ)** in the MXCSR register at the start of
//! every processing call ([`math::common::set_daz_ftz`]) — a fixed
//! `stmxcsr`/`ldmxcsr` pair outside any sample loop, with no measurable
//! performance cost.
//!
//! Because MXCSR is **per-thread**, integrators that bypass the pipeline
//! (for example, calling [`models::NamModel::process`] directly) or that run
//! additional DSP on their own audio threads must call
//! [`math::common::set_daz_ftz`] once at the start of each audio thread.
//!
//! ### 4. Panic-Free Hot Path
//! Stack unwinding (panics) breaks hard real-time determinism. Processing
//! hot paths avoid `unwrap()`/`expect()` in favor of explicit fallback
//! bounds checks.
//!
//! ### 5. Lock-Free Cache-Isolated Concurrency
//! Shared structures RT ↔ Main use `#[repr(align(128))]` to eliminate
//! false sharing on CPU cache lines. Inter-thread SPSC buffers use
//! `Acquire`/`Release` atomic ordering.
//!
//! ---
//!
//! ## 📜 License
//!
//! Licensed under the **Apache License, Version 2.0**.
//! Official repository: <https://github.com/fabiohl/NeuralAmpModeler-rs>

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
/// Mathematical primitives: SIMD kernels, activations, GEMM, FFT, DSP utilities.
pub mod math;
/// Neural network architectures (WaveNet A1/A2, LSTM, ConvNet, Linear) and runtime dispatch.
pub mod models;

#[cfg(any(test, feature = "testing"))]
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
