// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! High-level parsing function for the `.nam` format (JSON).

use super::data::{JsonError, NamModelData};
use super::validation::{MAX_HIDDEN_SIZE, MAX_LAYERS};

/// Raw universal deserialization of the JSON string via `serde_json`.
///
/// After deserialization, performs post-parse OOM-safety validation of the
/// declared topology: layer count must not exceed [`MAX_LAYERS`] and
/// `hidden_size` must not exceed [`MAX_HIDDEN_SIZE`]. Returns
/// `JsonError::UnsupportedTopology` immediately if limits are breached.
pub fn parse_nam_json(json_str: &str) -> Result<NamModelData, JsonError> {
    let data: NamModelData = serde_json::from_str(json_str)?;
    validate_version(&data)?;
    validate_topology_limits(&data)?;
    Ok(data)
}

/// Validates the `.nam` version field against NAMcore-compatible SemVer
/// ranges.
///
/// - Missing version → error (required for forward-compat enforcement).
/// - Unparseable version → `InvalidVersionFormat`.
/// - Version < 0.5.0 → `UnsupportedVersion` (below minimum).
/// - Version > 0.7.x (major > 0 or minor > 7) → `UnsupportedVersion`
///   (exceeds maximum supported).
/// - Version 0.7.x with patch > 0 → accepted with a `warn!` log
///   (partial compatibility, mirrors NAMcore PARTIAL).
fn validate_version(data: &NamModelData) -> Result<(), JsonError> {
    use super::topology::parse_semver;
    use log::warn;

    let version_str = data
        .version
        .as_deref()
        .ok_or_else(|| JsonError::UnsupportedVersion {
            raw: "(missing)".to_string(),
            reason: "version field is required; .nam files without a version are not supported"
                .to_string(),
        })?;

    let (major, minor, patch) =
        parse_semver(version_str).ok_or_else(|| JsonError::InvalidVersionFormat {
            raw: version_str.to_string(),
        })?;

    if major > 0 || minor > 7 {
        return Err(JsonError::UnsupportedVersion {
            raw: version_str.to_string(),
            reason: format!("version {version_str} exceeds maximum supported 0.7.x"),
        });
    }

    if major == 0 && minor < 5 {
        return Err(JsonError::UnsupportedVersion {
            raw: version_str.to_string(),
            reason: format!("version {version_str} is below minimum supported 0.5.0"),
        });
    }

    if major == 0 && minor == 7 && patch > 0 {
        warn!(
            "NAM file version {version_str} exceeds 0.7.0 — \
             loading with partial compatibility. Some features may not be \
             available or may behave differently from the reference NAMcore."
        );
    }

    Ok(())
}

fn validate_topology_limits(data: &NamModelData) -> Result<(), JsonError> {
    let layers = data.config.layers.len();
    if layers > MAX_LAYERS {
        return Err(JsonError::UnsupportedTopology {
            architecture: data.architecture.clone(),
            issue: format!(
                "config.layers has {} entries (max is {})",
                layers, MAX_LAYERS
            ),
            limit: MAX_LAYERS,
        });
    }

    if let Some(hidden_size) = data.config.hidden_size
        && hidden_size > MAX_HIDDEN_SIZE
    {
        return Err(JsonError::UnsupportedTopology {
            architecture: data.architecture.clone(),
            issue: format!(
                "config.hidden_size={} exceeds maximum {}",
                hidden_size, MAX_HIDDEN_SIZE
            ),
            limit: MAX_HIDDEN_SIZE,
        });
    }

    Ok(())
}
