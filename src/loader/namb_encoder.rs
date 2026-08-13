// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Utility for exporting models in the `.namb` v2 binary format.
//!
//! Allows converting JSON models or in-memory models to the optimized
//! binary format with pre-transposed weights.

use super::nam_json::{NamModelData, WeightsLayout};
use super::namb::{FLAG_HAS_CRC32, NambHeader, crc32_ieee, crc32_ieee_update};
use super::transpose::{lstm::transpose_lstm_gate_major, wavenet::transpose_wavenet_interleaved4};
use anyhow::Result;
use std::io::Write;

/// Encodes a [`NamModelData`] into the `.namb` v2 binary format.
///
/// Accepts a JSON-parsed model data structure, a target binary format version,
/// and a requested weight layout. Transposes weights to the target layout
/// (if different from `Original`), builds a V2 header, computes a CRC-32
/// checksum over the weight stream, and returns the complete `.namb` byte buffer.
///
/// # Arguments
///
/// * `data` — Parsed model data (architecture, layers, weights, metadata).
/// * `version` — `.namb` format version number (currently v2).
/// * `target_layout` — Weight matrix memory layout (`Original`, `Interleaved4`, etc.).
///
/// # Returns
///
/// Complete `.namb` byte buffer ready for writing to disk.
pub fn encode_namb(
    data: &NamModelData,
    version: u16,
    target_layout: WeightsLayout,
) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();

    // 1. Prepares the transposed weights
    let transposed_weights = if target_layout != WeightsLayout::Original {
        transpose_weights(data, target_layout)?
    } else {
        data.weights.clone()
    };

    let weights_bytes: Vec<u8> = transposed_weights
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    // 2. Prepares the metadata JSON
    let mut metadata_only = data.clone();
    metadata_only.weights = Vec::new();
    let json_str = serde_json::to_string(&metadata_only)?;
    let json_bytes = json_str.as_bytes();

    let header_flags = if version >= 2 { FLAG_HAS_CRC32 } else { 0 };

    // 3. Builds the Header
    let mut header = NambHeader {
        magic: 0x4E414D42,
        version,
        layout_type: target_layout as u8,
        flags: header_flags,
        reserved_v2: [0; 4],
        weights_offset: (std::mem::size_of::<NambHeader>() + json_bytes.len()) as u32,
        reserved1: [0; 2],
        crc32: if version < 2 {
            crc32_ieee(&weights_bytes)
        } else {
            0
        },
        reserved2: 0,
        version_str: [0; 32],
        sample_rate: data
            .sample_rate
            .unwrap_or(super::loaded_model_pair::DEFAULT_SAMPLE_RATE),
        input_level_dbu: data
            .metadata
            .as_ref()
            .and_then(|m| m.input_level_dbu)
            .unwrap_or(super::loaded_model_pair::DEFAULT_INPUT_LEVEL_DBU),
        output_level_dbu: data
            .metadata
            .as_ref()
            .and_then(|m| m.output_level_dbu)
            .unwrap_or(-6.0),
        reserved3: [0; 1],
    };

    let ver_str = format!("NAMB v{} ({:?})", version, target_layout);
    let ver_bytes = ver_str.as_bytes();
    let copy_len = ver_bytes.len().min(32);
    header.version_str[..copy_len].copy_from_slice(&ver_bytes[..copy_len]);

    // 4. Writes everything into the buffer
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            (&header as *const NambHeader) as *const u8,
            std::mem::size_of::<NambHeader>(),
        )
    };

    buffer.write_all(header_bytes)?;
    buffer.write_all(json_bytes)?;
    buffer.write_all(&weights_bytes)?;

    // 5. Updates the CRC32 checksum for version >= 2
    if version >= 2 {
        let crc = crc32_ieee_update(0xFFFFFFFFu32, &buffer[..24]);
        let crc = crc32_ieee_update(crc, &buffer[28..]);
        let final_crc = crc ^ 0xFFFFFFFFu32;
        buffer[24..28].copy_from_slice(&final_crc.to_le_bytes());
    }

    Ok(buffer)
}

/// Rearranges the "weights" so that the processor can
/// read them more efficiently during audio execution.
fn transpose_weights(data: &NamModelData, layout: WeightsLayout) -> Result<Vec<f32>> {
    match layout {
        // For LSTM, we use an organization where the logical "gates" are grouped.
        WeightsLayout::GateMajorLstm => transpose_lstm_gate_major(data),
        // For WaveNet, we interleave the data to facilitate batch calculations (SIMD).
        WeightsLayout::Interleaved4WaveNet => transpose_wavenet_interleaved4(data),
        // Default case: simply copies the original weights without changes.
        _ => Ok(data.weights.clone()),
    }
}

/// Safety function: ensures we won't try to read data beyond what exists in the file.
/// If the model is corrupted or incomplete, the program warns instead of crashing.
pub(crate) fn ensure_capacity(
    weights: &[f32],
    cursor: usize,
    needed: usize,
    label: String,
) -> Result<()> {
    if cursor + needed > weights.len() {
        anyhow::bail!(
            "Insufficient weights for {}: needs index {}..{}, but total length is {}",
            label,
            cursor,
            cursor + needed,
            weights.len()
        );
    }
    Ok(())
}
