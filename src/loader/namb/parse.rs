// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Main parser for `.namb` binary files.

use super::super::nam_json::{NamMetadata, NamModelData};
use super::error::NambError;
use super::fallback::make_fallback_model_data;
use super::header::{FLAG_HAS_CRC32, NambHeader, check_crc};
use anyhow::Result;
use log::{debug, info};

/// Loads a model in the `.namb` binary format.
pub fn parse_namb(data: &[u8]) -> Result<NamModelData> {
    let header_size = std::mem::size_of::<NambHeader>();
    if data.len() < header_size {
        return Err(NambError::Truncated {
            got: data.len(),
            need: header_size,
        }
        .into());
    }

    // 1. Reads the header
    // SAFETY: `data.len() >= header_size` was validated in lines 16-22. `NambHeader` is
    // `Copy` + `repr(C, packed)`, so a byte-level read of the header prefix is well-defined
    // (no padding, no uninit bytes, correct alignment via `read_unaligned`).
    let header = unsafe { core::ptr::read_unaligned(data.as_ptr().cast::<NambHeader>()) };
    header.validate()?;

    let hdr_version = header.version;
    let hdr_weights_offset = header.weights_offset as usize;
    let hdr_flags = header.flags;
    let file_size = data.len();
    info!(
        "[Loader] .namb header v{} — weights_offset={}, flags=0x{:02X}, file_size={}",
        hdr_version, hdr_weights_offset, hdr_flags, file_size
    );

    // 2. Reads the JSON metadata section (optional in .namb, but common)
    // If weights_offset > header_size, there is a JSON between them.
    let weights_offset = header.weights_offset as usize;
    if weights_offset > data.len() {
        return Err(NambError::WeightsOffsetOutOfBounds {
            offset: weights_offset,
            file_len: data.len(),
        }
        .into());
    }
    if weights_offset < header_size {
        return Err(NambError::InvalidWeightsOffset {
            offset: weights_offset,
            header_size,
        }
        .into());
    }

    let mut model_data = if weights_offset > header_size {
        let json_bytes = &data[header_size..weights_offset];
        // Truncate nulls if present (the NAMB buffer is usually padded)
        let actual_json = if let Some(pos) = json_bytes.iter().position(|&b| b == 0) {
            &json_bytes[..pos]
        } else {
            json_bytes
        };

        if !actual_json.is_empty() {
            debug!(
                "[Loader] .namb JSON metadata section: {} bytes",
                actual_json.len()
            );
            crate::loader::nam_json::parse_nam_json(std::str::from_utf8(actual_json)?)?
        } else {
            debug!("[Loader] .namb has no JSON metadata — using fallback defaults");
            make_fallback_model_data()
        }
    } else {
        debug!("[Loader] .namb has no JSON metadata gap — using fallback defaults");
        make_fallback_model_data()
    };

    // 3. Integrity Validation (CRC32)
    let version = header.version;
    let crc32_header = header.crc32;

    if version >= 2 && hdr_flags & FLAG_HAS_CRC32 == 0 {
        return Err(NambError::CrcMissing { version }.into());
    }

    if version == 1 && crc32_header == 0 {
        return Err(NambError::CrcMissingV1.into());
    }

    check_crc(data, version, weights_offset, crc32_header)?;
    debug!(
        "[Loader] .namb CRC32 validated (v{}, crc=0x{:08X})",
        version, crc32_header
    );

    // 4. Reads the binary weights
    let pesos_raw = &data[weights_offset..];
    if !pesos_raw.len().is_multiple_of(4) {
        let expected_len = weights_offset + pesos_raw.len() + (4 - pesos_raw.len() % 4);
        return Err(NambError::Truncated {
            got: data.len(),
            need: expected_len,
        }
        .into());
    }
    let float_count = pesos_raw.len() / 4;
    // Defense-in-depth: cap to MAX_MODEL_BYTES / 4 (already protected by build.rs, duplicated here)
    if float_count > super::MAX_FLOAT_COUNT {
        return Err(NambError::WeightsTooLarge {
            got: float_count,
            max: super::MAX_FLOAT_COUNT,
        }
        .into());
    }
    let mut weights = Vec::with_capacity(float_count);

    for (i, chunk) in pesos_raw.chunks_exact(4).enumerate() {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !val.is_finite() {
            return Err(NambError::NonFiniteWeight {
                index: i,
                value: val,
            }
            .into());
        }
        weights.push(val);
    }

    // Populates header metadata into the final NamModelData
    let sample_rate_header = header.sample_rate;
    let input_level_header = header.input_level_dbu;
    let output_level_header = header.output_level_dbu;

    debug!(
        "[Loader] .namb weights read: {} floats, sample_rate={:.0}, in_level={:.1}, out_level={:.1}",
        float_count, sample_rate_header, input_level_header, output_level_header
    );

    if !sample_rate_header.is_finite() {
        return Err(NambError::InvalidHeaderField {
            field: "sample_rate",
            value: sample_rate_header,
            reason: "must be finite",
        }
        .into());
    }
    if sample_rate_header <= 0.0 {
        return Err(NambError::InvalidHeaderField {
            field: "sample_rate",
            value: sample_rate_header,
            reason: "must be > 0.0",
        }
        .into());
    }
    if !input_level_header.is_finite() {
        return Err(NambError::InvalidHeaderField {
            field: "input_level_dbu",
            value: input_level_header,
            reason: "must be finite",
        }
        .into());
    }
    if !output_level_header.is_finite() {
        return Err(NambError::InvalidHeaderField {
            field: "output_level_dbu",
            value: output_level_header,
            reason: "must be finite",
        }
        .into());
    }
    let version_header = header.version;

    model_data.weights = weights;
    model_data.sample_rate = Some(sample_rate_header);
    model_data.weights_layout = header.get_layout();

    // Updates metadata if it exists
    if let Some(ref mut metadata) = model_data.metadata {
        metadata.input_level_dbu = Some(input_level_header);
        metadata.output_level_dbu = Some(output_level_header);
    } else {
        model_data.metadata = Some(NamMetadata {
            date: None,
            name: None,
            modeled_by: None,
            gear_make: None,
            gear_model: None,
            gear_type: None,
            tone_type: None,
            training: None,
            input_level_dbu: Some(input_level_header),
            output_level_dbu: Some(output_level_header),
            loudness: Some(-18.0),
        });
    }

    // If the version is null (fallback), gets it from the header string
    if model_data.version.is_none() {
        let end_idx = header
            .version_str
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(32);
        let version_str = String::from_utf8_lossy(&header.version_str[..end_idx]).into_owned();
        model_data.version = Some(version_str);
    }

    let arch = model_data.architecture.as_str();
    info!(
        "[Loader] .namb v{} loaded (arch={}, {} weights, layout={:?})",
        version_header, arch, float_count, model_data.weights_layout
    );

    Ok(model_data)
}
