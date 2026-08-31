// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Binary header and CRC32 utilities for `.namb` files.

use super::super::nam_json::WeightsLayout;
use super::error::NambError;

/// Updates the CRC32 (IEEE 802.3) checksum with the given data.
pub fn crc32_ieee_update(mut crc: u32, data: &[u8]) -> u32 {
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB88320u32 & mask);
        }
    }
    crc
}

/// Computes the CRC32 (IEEE 802.3) of a byte slice.
/// Replaces the external `crc32fast` dependency with a lightweight software version.
pub fn crc32_ieee(data: &[u8]) -> u32 {
    crc32_ieee_update(0xFFFFFFFFu32, data) ^ 0xFFFFFFFFu32
}

/// Validates the CRC32 checksum of the file (or weight section depending on version).
pub fn check_crc(
    data: &[u8],
    version: u16,
    weights_offset: usize,
    expected: u32,
) -> Result<(), NambError> {
    let calculated = if version >= 2 {
        // v2+ covers header (except crc field) + JSON + weights.
        // CRC32 field is at offset 24..28.
        let crc = crc32_ieee_update(0xFFFFFFFFu32, &data[..24]);
        let crc = crc32_ieee_update(crc, &data[28..]);
        crc ^ 0xFFFFFFFFu32
    } else {
        crc32_ieee(&data[weights_offset..])
    };

    if calculated != expected {
        // T5.1: structured rejection diagnostic (CRC integrity policy).
        // `offset_of!` yields the byte offset of the `crc32` field within the
        // packed header (also the absolute file offset, since the header
        // starts at file byte 0).
        log::warn!(
            "[Loader] Invalid CRC rejected: field='crc32', got=0x{:08X}, expected=0x{:08X}, offset_bytes={}",
            calculated,
            expected,
            std::mem::offset_of!(NambHeader, crc32)
        );
        return Err(NambError::CrcMismatch {
            got: calculated,
            expected,
        });
    }
    Ok(())
}

/// Flag bitmask for the NAMB header `flags` field.
pub const FLAG_HAS_CRC32: u8 = 0x01;

/// Fixed binary header of the `.namb` format.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct NambHeader {
    /// Magic number `0x4E414D42` ("NAMB" in ASCII).
    pub magic: u32,
    /// Format version (1 = legacy, 2 = with pre-transposed layout).
    pub version: u16,
    /// Weight layout (only if version >= 2). Offset: 6.
    pub layout_type: u8,
    /// Feature flags (bit 0 = FLAG_HAS_CRC32). Offset: 7.
    pub flags: u8,
    /// Reserved for future expansion. Offset: 8.
    pub reserved_v2: [u8; 4],
    /// Offset (in bytes) from the beginning of the file to the start of the weight section.
    pub weights_offset: u32,
    /// Reserved for future expansion.
    pub reserved1: [u32; 2],
    /// CRC32 checksum of the weight block (optional).
    pub crc32: u32,
    /// Reserved for future expansion.
    pub reserved2: u32,
    /// Informational version string (e.g. "NAMB 2.0.0").
    pub version_str: [u8; 32],
    /// Default sample rate (e.g. 48000.0).
    pub sample_rate: f32,
    /// Default input level dBu (e.g. 12.0).
    pub input_level_dbu: f32,
    /// Default output level dBu (e.g. 12.0).
    pub output_level_dbu: f32,
    /// Reserved (total header size must be at least 80 bytes).
    pub reserved3: [u32; 1],
}

impl NambHeader {
    /// Validates whether the header has the magic number and a supported version.
    pub fn validate(&self) -> Result<(), NambError> {
        let magic = self.magic;
        let version = self.version;
        if magic != 0x4E414D42 {
            return Err(NambError::InvalidMagic(magic));
        }
        if version != 1 && version != 2 {
            return Err(NambError::InvalidVersion(version));
        }
        Ok(())
    }

    /// Returns the weight layout based on the version and the flag.
    pub fn get_layout(&self) -> WeightsLayout {
        let version = self.version;
        if version < 2 {
            return WeightsLayout::Original;
        }
        match self.layout_type {
            1 => WeightsLayout::GateMajorLstm,
            2 => WeightsLayout::Interleaved4WaveNet,
            _ => WeightsLayout::Original,
        }
    }
}
