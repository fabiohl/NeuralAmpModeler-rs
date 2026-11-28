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
pub use header::{
    FLAG_HAS_CRC32, NAMB_HEADER_LEN, NambHeader, check_crc, crc32_ieee, crc32_ieee_update,
    layout_name, v1_crc_is_missing,
};
pub use parse::{parse_namb, parse_namb_typed};

/// Maximum number of floats in the weight section (MAX_MODEL_BYTES / 4).
pub(crate) const MAX_FLOAT_COUNT: usize = (256 * 1024 * 1024 / 4) as usize;

/// Test-only: recomputes the v2 whole-file CRC after mutating header or
/// metadata bytes. Uses the real header offset instead of hardcoded slices so
/// fixture helpers cannot drift from `NambHeader`.
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
pub fn test_fix_v2_crc(data: &mut [u8]) {
    let crc_offset = std::mem::offset_of!(NambHeader, crc32);
    let crc = crc32_ieee_update(0xFFFFFFFFu32, &data[..crc_offset]);
    let crc = crc32_ieee_update(crc, &data[crc_offset + 4..]);
    data[crc_offset..crc_offset + 4].copy_from_slice(&(crc ^ 0xFFFFFFFFu32).to_le_bytes());
}

#[cfg(test)]
#[path = "../namb_test.rs"]
mod namb_test;
