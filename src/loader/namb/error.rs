// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Typed errors for `.namb` binary file parsing.

/// Typed error for `.namb` file parsing.
///
/// Each variant corresponds to a specific integrity or format failure
/// of the binary file, enabling precise diagnosis via
/// `downcast_ref` in the `loader` module.
#[derive(Debug, thiserror::Error)]
pub enum NambError {
    /// Truncated file: insufficient bytes for the minimum header.
    #[error("file truncated: got {got} bytes, need at least {need}")]
    Truncated {
        /// Bytes available in the file.
        got: usize,
        /// Minimum bytes needed.
        need: usize,
    },

    /// Invalid magic number (not 0x4E414D42).
    #[error("invalid magic number: 0x{0:08X} (expected 0x4E414D42)")]
    InvalidMagic(u32),

    /// Unsupported `.namb` format version.
    #[error("unsupported .namb version: {0}")]
    InvalidVersion(u16),

    /// Weight section offset beyond file size.
    #[error("weights offset {offset} out of file bounds (file size: {file_len})")]
    WeightsOffsetOutOfBounds {
        /// Offset declared in the header.
        offset: usize,
        /// Total file size in bytes.
        file_len: usize,
    },

    /// Weight section offset smaller than the header size.
    #[error("invalid weights offset {offset} (smaller than header size {header_size})")]
    InvalidWeightsOffset {
        /// Offset declared in the header.
        offset: usize,
        /// Expected header size.
        header_size: usize,
    },

    /// CRC32 checksum of the weight section does not match.
    #[error("CRC32 mismatch: got 0x{got:08X}, expected 0x{expected:08X}")]
    CrcMismatch {
        /// CRC calculated from the data.
        got: u32,
        /// CRC declared in the header.
        expected: u32,
    },

    /// CRC32 missing in NAMB v2+ file (FLAG_HAS_CRC32 flag not set).
    #[error("CRC32 flag missing in NAMB v{version} file (FLAG_HAS_CRC32 not set)")]
    CrcMissing {
        /// NAMB file version.
        version: u16,
    },

    /// Number of floats in the weight section exceeds the maximum (defense-in-depth).
    #[error("weight section too large: {got} floats, maximum {max}")]
    WeightsTooLarge {
        /// Floating-point counts read from the file.
        got: usize,
        /// Maximum allowed number of floats (MAX_MODEL_BYTES/4).
        max: usize,
    },

    /// A weight value is non-finite (NaN, +Inf, or -Inf).
    #[error("weight at index {index} is not finite (value: {value:e})")]
    NonFiniteWeight {
        /// Index (0-based) of the non-finite float in the weights array.
        index: usize,
        /// The non-finite value.
        value: f32,
    },

    /// Header field contains an invalid value (non-finite, <= 0, etc.).
    #[error("invalid header field '{field}': {value:e} ({reason})")]
    InvalidHeaderField {
        /// Name of the header field.
        field: &'static str,
        /// The invalid value read.
        value: f32,
        /// Explanation of why the value is rejected.
        reason: &'static str,
    },
}
