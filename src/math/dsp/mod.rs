// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Module for digital signal processing (DSP) operations.
//!
//! Contains utilities for audio buffer manipulation, optimized for
//! stereo operation and clipping safety.
//!
//! # Features
//! - **Gain and Ramp**: Linear gain application and SIMD ramps for smooth transitions.
//! - **Clipping Detection**: Detection of peaks above 0dBFS integrated into gain application.
//! - **Stereo Processing**: Kernels that operate simultaneously on L/R channels for better cache locality.
//! - **Energy Computation**: RMS/Peak energy calculation for signal telemetry.
//!
//! Contains implementations of optimized audio algorithms, including
//! energy calculations, correlations, and filters.

/// Radix-2 Decimation-in-Time (DIT) FFT with SIMD acceleration.
pub mod fft;
/// Radix-4 DIT FFT — research prototype preserved for reference. Radix-2 SIMD is canonical for production.
/// See module-level docs in `fft_radix4.rs` for decision rationale and benchmarks.
pub mod fft_radix4;
/// Linear gain application and SIMD ramp kernels for smooth audio transitions.
pub mod gain;
/// Pre-computed dB-to-linear gain lookup table.
pub mod gain_lut;
/// Real-valued FFT (RFFT) for spectrum analysis.
pub mod rfft;
/// Stereo audio utilities: convolution, energy computation, peak detection.
pub mod stereo;
