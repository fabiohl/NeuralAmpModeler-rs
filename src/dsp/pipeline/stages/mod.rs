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

pub use bridge::write_bridge;
pub use inference::run_inference;
pub use input::DENORMAL_DITHER_OFFSET;
#[cfg(feature = "testing")]
pub use input::DISABLE_GATE;
pub use input::apply_input_stage;
pub(crate) use input::apply_input_stage_inner;
pub use input::handle_silence_bypass;
pub use output::apply_output_stage;
pub(crate) use output::apply_output_stage_inner;
