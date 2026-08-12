// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Cab Sim — IR convolution engine.
//!
//! Loads `.wav` impulse responses (mono, PCM16/24/float32),
//! resamples them to the active sample rate, and delivers them
//! to the DSP thread via lock-free SPSC transfer.

/// SPSC adapter: bridges the off-RT loader to the RT convolution engine.
pub mod adapter;
/// Frequency-domain block convolution kernel: partitioned overlap-save FFT.
pub mod conv;
pub(crate) mod ir_parse;
pub(crate) mod ir_resample;
/// IR loader: parses WAV files, resamples to target rate, builds convolution buffers.
pub mod loader;
