// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Cab Sim — IR convolution engine.
//!
//! Loads `.wav` impulse responses (mono, PCM16/24/float32),
//! resamples them to the active sample rate, and delivers them
//! to the DSP thread via lock-free SPSC transfer.

pub mod adapter;
pub mod conv;
pub(crate) mod ir_parse;
pub(crate) mod ir_resample;
pub mod loader;
