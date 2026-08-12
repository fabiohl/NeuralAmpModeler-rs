// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Digital Signal Processing (DSP) module for generic operations
//! before and after the neural engine in NAM-rs.

/// Adaptive compute mode: quality-tier hysteresis FSM for dynamic power scaling.
pub mod adaptive;
/// Cabinet simulator: impulse response convolution engine (mono/stereo, variable-length).
pub mod cabsim;
/// Noise gate: envelope-based gate with configurable hysteresis and fade curve.
pub mod gate;
/// Gate runtime state flags: per-sample clipping, gain applied, gate open/closed signaling.
pub mod gate_flags;
/// Mirror buffer: zero-copy virtual-memory ring buffer with automatic GPU/DMA mirroring.
pub mod mirror_buf;
/// Oversampling engine: 2× UPSAMPLE → PROCESS → 2× DOWNSAMPLE via half-band FIR stages.
pub mod oversample;
/// DSP processing pipeline: staged audio graph with input, inference, output, and telemetry.
pub mod pipeline;
/// Synchronous sample-rate converter with linear or sinc-based interpolation.
pub mod resampler;
/// Windowed-sinc kernel tables for high-quality FIR filter generation.
pub mod sinc_kernel;
/// Exponential parameter smoother: 1-pole IIR envelope for click-free parameter changes.
pub mod smoother;
/// Half-band FIR upsampling/downsampling stage with SIMD acceleration.
pub mod stage;
/// RT-safe telemetry accumulator: peak/RMS/counter aggregates for the DSP monitoring loop.
pub mod telemetry;
