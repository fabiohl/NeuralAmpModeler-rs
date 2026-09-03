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
    /// Parses a `NambHeader` from a byte slice in a fully alignment-safe manner.
    ///
    /// The header is copied by value with [`std::ptr::read_unaligned`], so the
    /// source slice may start at any memory address (including odd addresses),
    /// and the magic number and supported version are validated before
    /// returning. This function performs zero heap allocations.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, NambError> {
        let header_size = std::mem::size_of::<Self>();
        if bytes.len() < header_size {
            return Err(NambError::Truncated {
                got: bytes.len(),
                need: header_size,
            });
        }
        // SAFETY: `bytes.len() >= size_of::<Self>()` was validated above, so the
        // pointer is valid for reads of `size_of::<Self>()` bytes. `NambHeader`
        // is `repr(C, packed)` (no padding, no uninitialized bytes), and
        // `read_unaligned` tolerates any address alignment. The byte slice
        // outlives this call.
        let header = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<Self>()) };
        header.validate()?;
        Ok(header)
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a byte buffer containing a valid NAMB v2 header (80 bytes,
    /// native endianness, matching how the encoder writes the header).
    fn valid_header_bytes() -> Vec<u8> {
        let mut b = Vec::with_capacity(80);
        b.extend_from_slice(&0x4E414D42u32.to_ne_bytes()); // magic
        b.extend_from_slice(&2u16.to_ne_bytes()); // version
        b.push(0); // layout_type
        b.push(0); // flags
        b.extend_from_slice(&[0u8; 4]); // reserved_v2
        b.extend_from_slice(&80u32.to_ne_bytes()); // weights_offset
        b.extend_from_slice(&[0u8; 8]); // reserved1
        b.extend_from_slice(&0xDEADBEEFu32.to_ne_bytes()); // crc32
        b.extend_from_slice(&[0u8; 4]); // reserved2
        b.extend_from_slice(b"NAMB 2.0.0"); // version_str (10 bytes)
        b.extend_from_slice(&[0u8; 22]); // version_str padding to 32
        b.extend_from_slice(&48000.0f32.to_ne_bytes()); // sample_rate
        b.extend_from_slice(&12.0f32.to_ne_bytes()); // input_level_dbu
        b.extend_from_slice(&12.0f32.to_ne_bytes()); // output_level_dbu
        b.extend_from_slice(&[0u8; 4]); // reserved3
        debug_assert_eq!(b.len(), std::mem::size_of::<NambHeader>());
        b
    }

    #[test]
    fn from_slice_short_slice_returns_truncated() {
        let bytes = valid_header_bytes();
        let short = &bytes[..bytes.len() - 1];
        let err = NambHeader::from_slice(short).unwrap_err();
        assert!(matches!(err, NambError::Truncated { got: 79, need: 80 }));
    }

    #[test]
    fn from_slice_unaligned_buffer_ok() {
        // Misalign the header by one byte so the pointer is not 4-byte aligned.
        let mut data = Vec::with_capacity(81);
        data.push(0u8);
        data.extend_from_slice(&valid_header_bytes());
        let header = NambHeader::from_slice(&data[1..]).unwrap();
        let magic = header.magic;
        let version = header.version;
        let sample_rate = header.sample_rate;
        let weights_offset = header.weights_offset;
        assert_eq!(magic, 0x4E414D42);
        assert_eq!(version, 2);
        assert_eq!(sample_rate, 48000.0);
        assert_eq!(weights_offset, 80);
    }

    #[test]
    fn from_slice_invalid_magic_returns_invalid_magic() {
        let mut bytes = valid_header_bytes();
        bytes[0..4].copy_from_slice(&0xDEADBEEFu32.to_ne_bytes());
        let err = NambHeader::from_slice(&bytes).unwrap_err();
        assert!(matches!(err, NambError::InvalidMagic(0xDEADBEEF)));
    }

    #[test]
    fn from_slice_invalid_version_returns_invalid_version() {
        let mut bytes = valid_header_bytes();
        bytes[4..6].copy_from_slice(&99u16.to_ne_bytes());
        let err = NambHeader::from_slice(&bytes).unwrap_err();
        assert!(matches!(err, NambError::InvalidVersion(99)));
    }
}
