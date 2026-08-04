// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Binary loader for `.namb` models.
//!
//! Performs direct, deterministic, lock-free analysis and deserialization
//! from a binary block into the `NamModelData` structure.

mod error;
mod fallback;
mod header;
mod parse;

pub use super::nam_json::WeightsLayout;
pub use error::NambError;
pub use fallback::{make_fallback_model_data, make_standard_wavenet_config};
pub use header::{FLAG_HAS_CRC32, NambHeader, check_crc, crc32_ieee, crc32_ieee_update};
pub use parse::parse_namb;

/// Maximum number of floats in the weight section (MAX_MODEL_BYTES / 4).
pub(crate) const MAX_FLOAT_COUNT: usize = (256 * 1024 * 1024 / 4) as usize;

#[cfg(test)]
#[path = "../namb_test.rs"]
mod namb_test;
