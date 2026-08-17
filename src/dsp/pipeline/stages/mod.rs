// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP pipeline stages (1–4): Input → Inference → Output → Bridge.
//!
//! All functions in this module follow the RT callback rules:
//! - Zero heap allocation
//! - Zero I/O
//! - Zero mutexes

mod adaptive_ctrl;
mod bridge;
mod inference;
mod input;
mod output;
mod routing;

/// Writes processed output samples into the inter-stage bridge buffer.
pub use bridge::write_bridge;
/// Dispatches model inference across architecture-specific process methods.
pub use inference::run_inference;
/// Denormal dither offset constant for FTZ/DAZ maintenance on the audio hot path.
pub use input::DENORMAL_DITHER_OFFSET;
#[cfg(feature = "testing")]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
pub use input::DISABLE_GATE;
/// Applies input DSP stage: gate, denormal dither, and silence detection.
pub use input::apply_input_stage;
pub(crate) use input::apply_input_stage_inner;
/// Detects silence at the input and bypasses the entire DSP chain for idle blocks.
pub use input::handle_silence_bypass;
/// Applies output DSP stage: final volume, clipping detection, post-DSP smoothing.
pub use output::apply_output_stage;
pub(crate) use output::apply_output_stage_inner;
