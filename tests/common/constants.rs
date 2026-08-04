// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(dead_code)]

/// Default number of samples for golden vector validation.
pub const GOLDEN_NUM_SAMPLES: usize = 2048;
/// Block size for golden vector processing.
pub const GOLDEN_BLOCK_SIZE: usize = 64;
/// Test block size.
pub const TEST_BLOCK_SIZE: usize = 64;
/// Number of blocks for stress test signals.
pub const TEST_NUM_BLOCKS: usize = 4096;
/// Sample rate used for v1 stress test signals.
pub const STRESS_SAMPLE_RATE: u32 = 48000;

/// Number of prewarm samples used in v2 stress signal processing.
pub const V2_PREWARM_SAMPLES: usize = 2048;
/// Block size for v2 test signal processing.
pub const V2_TEST_BLOCK_SIZE: usize = 64;
/// Duration of v2 stress signals in seconds.
pub const V2_STRESS_DURATION_SECS: f64 = 5.0;

// ── Re-derived fidelity gates (post-weight-dequantization) ──
// Moved to common module for cross-test access.
// Recalibrated post-removal of f16c weight quantization.
//
// Methodology: 24k prewarm + 256-sample sweep @ 48 kHz; both production (f32)
// and oracle (f64 ideal) fed the same signal; ESR measured on last 256 samples.
// Limit = measured ESR × 2 (conservative margin).
//
// Measured (post-dequantization, post-Standard-engine default, f32 weights, prewarm-paired, 256-sample sweep @ 48 kHz):
//   WaveNet: ESR = 6.13e-14  →  WAVENET_ESR_LIMIT = 6.13e-14 × 2  →  1e-12 (numerical floor)
//   LSTM:    ESR = 2.71e-12  →  LSTM_ESR_LIMIT    = 2.71e-12 × 2  →  1e-11 (conservative, post-Standard exact activations)
//   A2:      ESR = 2.22e-14  →  A2_ESR_LIMIT      = 2.22e-14 × 2  →  1e-12 (numerical floor)
//   ConvNet: ESR = 1.83e-14  →  CONVNET_ESR_LIMIT = 1e-12 (numerical floor)
//   A2-FiLM-Lite:  ESR = 9.52e-15  →  A2_FILM_ESR_LIMIT = 1e-12 (numerical floor)
//   A2-FiLM-Full:  ESR = 1.15e-14  →  A2_FILM_ESR_LIMIT = 1e-12 (numerical floor)

/// Calibrated ESR limit for WaveNet model oracle parity (see methodology above).
pub const WAVENET_ESR_LIMIT: f64 = 1e-12;
/// Calibrated ESR limit for LSTM model oracle parity (see methodology above).
pub const LSTM_ESR_LIMIT: f64 = 1e-11;
/// Calibrated ESR limit for WaveNet A2 model oracle parity (see methodology above).
pub const A2_ESR_LIMIT: f64 = 1e-12;
/// Calibrated ESR limit for ConvNet model oracle parity (see methodology above).
pub const CONVNET_ESR_LIMIT: f64 = 1e-12;
/// Calibrated ESR limit for WaveNet A2-FiLM model oracle parity (see methodology above).
pub const A2_FILM_ESR_LIMIT: f64 = 1e-9;
// A2 Generic (wavenet_a2_max.nam) — disabled §7.1, dead threshold retained for meta-test calibration
/// Calibrated ESR limit for WaveNet A2 Generic (disabled model) — dead threshold.
pub const A2_GENERIC_ESR_LIMIT: f64 = 1e-9;

// ── Recurrent State Drift Diagnostic Limits ──
// Methodology: 5.0s of stress signal v2 (240k samples) comparing production f32 vs oracle f64.
//   Legacy:    Full 240k, prewarm(24_000) on zeroes. Measured: ESR = 2.611684e-2. Limit = ESR × 1.5.
//   Paired:    Last 216k, prewarm-paired (no zeroes). Measured: ESR = 2.589060e-2. Limit = ESR × 2.0.
//              For LSTM 2x8 paired: Measured: ESR = 4.062433e-3. Limit = ESR × 2.0.
/// Legacy drift limit for LSTM 1×16 (full 240k sweep, see methodology above).
pub const LSTM_1X16_DRIFT_LEGACY_ESR_LIMIT: f64 = 3.92e-2;
/// Paired-prewarm drift limit for LSTM 1×16 (see methodology above).
pub const LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT: f64 = 5.18e-2;
/// Paired-prewarm drift limit for LSTM 2×8 (see methodology above).
pub const LSTM_2X8_DRIFT_PAIRED_ESR_LIMIT: f64 = 8.13e-3;
