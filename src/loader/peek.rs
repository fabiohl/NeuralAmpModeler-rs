// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Host-neutral "metadata peek" for `.nam` (JSON) and `.namb` (binary) models.
//!
//! Preset browsers and CLIs need the small `metadata` object (name, author,
//! gear) of a model without paying the cost of parsing the weight tensors —
//! and without the false positives of a textual `"metadata"` search (a string
//! value or a nested object can contain the literal). This module parses the
//! model as a real document, streaming over the heavy fields with
//! [`IgnoredAny`](serde::de::IgnoredAny) so no tensor is ever materialized.
//!
//! Two invariants hold:
//! - **Neutrality**: no host/format policy leaks in; only the crate's own
//!   [`NamMetadata`](crate::loader::nam_json::NamMetadata) type is produced.
//! - **Bounded work**: only the `metadata` object is deserialized, and the
//!   whole read is capped by the caller's `max_bytes` budget. The scan over
//!   the surrounding document allocates nothing (weights are skipped, not
//!   collected), so a large budget is cheap and never materializes tensors.
//!
//! `metadata` is resolved as a *top-level key of the model object*, so a
//! `"metadata"` literal inside a string or a nested object is never matched.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use serde::Deserializer;
use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, Visitor};

use super::error::LoadError;
use super::loaded_model_pair::MAX_MODEL_BYTES;
use super::nam_json::{JsonError, NamMetadata};
use super::namb::{NAMB_HEADER_LEN, NambError, NambHeader};

/// Default read budget for [`peek_metadata`]: the engine's model-size limit
/// (`MAX_MODEL_BYTES`).
///
/// Because the scan skips weight arrays without allocating, a budget this
/// large is cheap and guarantees that any model the loader would accept is
/// fully scannable — including models whose `metadata` key follows megabytes
/// of weights (a fixed 1 MiB read cap would silently miss those).
pub const DEFAULT_METADATA_PEEK_BYTES: u64 = MAX_MODEL_BYTES;

/// Extracts the `metadata` object from a `.nam` or `.namb` model file without
/// materializing its weight tensors.
///
/// The format is selected from the file extension (case-insensitive):
/// - `.nam`: the JSON document is streamed; the top-level `metadata` value is
///   deserialized into [`NamMetadata`].
/// - `.namb`: the fixed header is read (see [`NAMB_HEADER_LEN`]) and the JSON
///   section between the header and `weights_offset` is streamed the same way.
///
/// `max_bytes` bounds the number of bytes read from the source. Weight arrays
/// are skipped with `IgnoredAny` (no allocation), so the budget bounds work
/// and allocation without requiring the caller to know where `metadata` sits.
/// A document whose `metadata` falls beyond the budget fails closed
/// (an error, never a silent "no metadata").
///
/// # Errors
///
/// Returns [`LoadError::UnsupportedExtension`] for anything other than
/// `.nam`/`.namb`, [`LoadError::Io`] on read failure,
/// [`LoadError::ModelTooLarge`] when the file exceeds the engine's model-size
/// limit, [`LoadError::Namb`]/[`LoadError::Json`] for malformed or truncated
/// input, and [`LoadError::MetadataNotFound`] when the document carries no
/// usable `metadata` object (absent or JSON `null`).
pub fn peek_metadata(path: &Path, max_bytes: u64) -> Result<NamMetadata, LoadError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("nam") => {
            let file = File::open(path)?;
            let file_len = file.metadata()?.len();
            if file_len > MAX_MODEL_BYTES {
                return Err(LoadError::ModelTooLarge);
            }
            match peek_json_metadata(BufReader::new(file), max_bytes)? {
                Some(meta) => Ok(meta),
                None => Err(LoadError::MetadataNotFound { format: "nam" }),
            }
        }
        Some("namb") => peek_namb(path, max_bytes),
        _ => Err(LoadError::UnsupportedExtension(ext.unwrap_or_default())),
    }
}

/// Streams the top-level `metadata` object of a `.nam` JSON document from a
/// reader, without materializing any other field.
///
/// See [`peek_metadata`] for the budget semantics. Useful when the document is
/// already in memory (or piped) rather than a file on disk.
///
/// # Errors
///
/// As [`peek_metadata`], minus the `.namb`/extension domain:
/// [`LoadError::MetadataNotFound`] when no top-level `metadata` object is
/// present.
pub fn peek_metadata_from_reader<R: Read>(
    reader: R,
    max_bytes: u64,
) -> Result<NamMetadata, LoadError> {
    match peek_json_metadata(reader, max_bytes)? {
        Some(meta) => Ok(meta),
        None => Err(LoadError::MetadataNotFound { format: "nam" }),
    }
}

/// Reads the `.namb` header and streams the JSON section that precedes the
/// weight block.
fn peek_namb(path: &Path, max_bytes: u64) -> Result<NamMetadata, LoadError> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    if file_len > MAX_MODEL_BYTES {
        return Err(LoadError::ModelTooLarge);
    }
    if (file_len as usize) < NAMB_HEADER_LEN {
        return Err(NambError::Truncated {
            got: file_len as usize,
            need: NAMB_HEADER_LEN,
        }
        .into());
    }

    let mut reader = BufReader::new(file);
    let mut header_bytes = [0u8; NAMB_HEADER_LEN];
    reader.read_exact(&mut header_bytes)?;
    let header = NambHeader::from_slice(&header_bytes)?;

    let weights_offset = header.weights_offset as u64;
    if weights_offset < NAMB_HEADER_LEN as u64 {
        return Err(NambError::InvalidWeightsOffset {
            offset: header.weights_offset as usize,
            header_size: NAMB_HEADER_LEN,
        }
        .into());
    }
    if weights_offset > file_len {
        return Err(NambError::WeightsOffsetOutOfBounds {
            offset: header.weights_offset as usize,
            file_len: file_len as usize,
        }
        .into());
    }

    let section_len = weights_offset - NAMB_HEADER_LEN as u64;
    match peek_json_metadata(reader.take(section_len), max_bytes)? {
        Some(meta) => Ok(meta),
        None => Err(LoadError::MetadataNotFound { format: "namb" }),
    }
}

/// Streams the top-level `metadata` value (if any) from a JSON reader, with a
/// hard read budget.
///
/// Returns `Ok(None)` when the document is a valid object without a
/// `metadata` key, or with a JSON `null` value.
fn peek_json_metadata<R: Read>(
    reader: R,
    max_bytes: u64,
) -> Result<Option<NamMetadata>, LoadError> {
    let mut deserializer = serde_json::Deserializer::from_reader(reader.take(max_bytes));
    MetadataSeed
        .deserialize(&mut deserializer)
        .map_err(|e| LoadError::Json(JsonError::Serde(e)))
}

/// Seed resolving the top-level `metadata` value, skipping every other field.
struct MetadataSeed;

impl<'de> DeserializeSeed<'de> for MetadataSeed {
    type Value = Option<NamMetadata>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MetadataVisitor)
    }
}

/// Visits the model object map, ignoring every value except `metadata`.
struct MetadataVisitor;

impl<'de> Visitor<'de> for MetadataVisitor {
    type Value = Option<NamMetadata>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a NAM model JSON object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        // The whole object must be consumed: serde_json's reader-based
        // `deserialize_map` calls `end_map()` after `visit_map` returns and
        // rejects a map left half-read. Skipping the remaining values with
        // `IgnoredAny` walks them without allocating (the weights are never
        // collected), so this stays cheap even for large models.
        let mut found: Option<NamMetadata> = None;
        let mut seen = false;
        while let Some(key) = map.next_key::<String>()? {
            if key == "metadata" && !seen {
                seen = true;
                // `Option<NamMetadata>` maps a JSON `null` to `None`; the
                // caller turns `None` into `MetadataNotFound`.
                found = map.next_value::<Option<NamMetadata>>()?;
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(found)
    }
}

#[cfg(test)]
#[path = "peek_test.rs"]
mod peek_test;
