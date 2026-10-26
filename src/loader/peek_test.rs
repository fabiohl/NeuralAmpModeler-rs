// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for [`peek_metadata`](super::peek_metadata).

use super::*;
use crate::loader::nam_json::{NamModelData, WeightsLayout, parse_nam_json};
use crate::loader::namb_encoder::encode_namb;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic suffix so parallel tests never share a temporary file name.
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Writes `bytes` to a uniquely named temporary file with `ext` and returns
/// its path. The caller is responsible for cleanup through [`TempFile`].
struct TempFile(PathBuf);

impl TempFile {
    fn write(ext: &str, bytes: &[u8]) -> Self {
        let name = format!(
            "nam-peek-test-{}-{:x}.{ext}",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, bytes).expect("write temp model");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

use std::path::Path;

/// Serialized metadata equality: `NamMetadata` is not `PartialEq`, so compare
/// its canonical JSON rendering.
fn metadata_json(meta: &NamMetadata) -> serde_json::Value {
    serde_json::to_value(meta).expect("metadata serializes")
}

#[test]
fn namb_header_len_matches_struct_size() {
    assert_eq!(NAMB_HEADER_LEN, std::mem::size_of::<NambHeader>());
    // The on-disk header was documented as 80 bytes; pin it so a header layout
    // change cannot silently desync the constant from the spec.
    assert_eq!(NAMB_HEADER_LEN, 80);
}

#[test]
#[ignore = "long: parse e verificação de todos os fixtures NAM"]
fn peek_matches_full_parse_for_all_nam_fixtures() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models");
    let mut checked = 0usize;
    for entry in std::fs::read_dir(&dir).expect("fixtures/models readable") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("nam") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read fixture");
        let parsed: NamModelData = parse_nam_json(&text)
            .unwrap_or_else(|e| panic!("fixture {} failed to parse: {e}", path.display()));
        let peeked = peek_metadata(&path, DEFAULT_METADATA_PEEK_BYTES);
        match parsed.metadata {
            Some(expected) => {
                let got = peeked.unwrap_or_else(|e| {
                    panic!("peek failed for {}: {e}", path.display());
                });
                assert_eq!(
                    metadata_json(&got),
                    metadata_json(&expected),
                    "metadata mismatch for {}",
                    path.display()
                );
            }
            None => {
                assert!(
                    matches!(peeked, Err(LoadError::MetadataNotFound { .. })),
                    "expected MetadataNotFound for {} (got {peeked:?})",
                    path.display()
                );
            }
        }
        checked += 1;
    }
    assert!(checked > 0, "no .nam fixtures were exercised");
}

#[test]
fn peek_ignores_metadata_literal_in_nested_object() {
    let json = r#"{"version":"0.6.0","config":{"nested":{"metadata":{"name":"WRONG"}}},"metadata":{"name":"RIGHT"}}"#;
    let file = TempFile::write("nam", json.as_bytes());
    let meta = peek_metadata(file.path(), DEFAULT_METADATA_PEEK_BYTES).expect("peek");
    assert_eq!(meta.name.as_deref(), Some("RIGHT"));
}

#[test]
fn peek_ignores_metadata_literal_inside_string() {
    let json = r#"{"version":"0.6.0","config":{"note":"\"metadata\":{\"name\":\"WRONG\"}"},"metadata":{"name":"RIGHT"}}"#;
    let file = TempFile::write("nam", json.as_bytes());
    let meta = peek_metadata(file.path(), DEFAULT_METADATA_PEEK_BYTES).expect("peek");
    assert_eq!(meta.name.as_deref(), Some("RIGHT"));
}

#[test]
fn peek_finds_metadata_after_more_than_one_mib_of_weights() {
    // ~1.2 MiB of weights (each `0.0,` is 4 bytes) placed *before* metadata.
    let json = big_weights_doc(300_000, "after", "tester");
    let file = TempFile::write("nam", json.as_bytes());
    assert!(file.path().metadata().unwrap().len() > 1_048_576);

    // The default budget is the engine's model-size limit; the scan skips the
    // weight array without allocating and resolves the trailing metadata — a
    // fixed 1 MiB read cap would have silently missed it.
    let meta = peek_metadata(file.path(), DEFAULT_METADATA_PEEK_BYTES).expect("peek past 1 MiB");
    assert_eq!(meta.name.as_deref(), Some("after"));
    assert_eq!(meta.modeled_by.as_deref(), Some("tester"));
}

#[test]
fn peek_fails_closed_when_metadata_is_beyond_budget() {
    // Same >1 MiB document under a legacy-sized (1 MiB) budget: the reader is
    // truncated before the metadata, so the peek must error — never silently
    // report "no metadata".
    let json = big_weights_doc(300_000, "after", "tester");
    let file = TempFile::write("nam", json.as_bytes());
    let err = peek_metadata(file.path(), 1_048_576).expect_err("truncated budget fails closed");
    assert!(matches!(err, LoadError::Json(_)));
}

/// Builds a `.nam` JSON document whose `weights` array (`count` elements)
/// precedes the metadata object.
fn big_weights_doc(count: usize, name: &str, modeled_by: &str) -> String {
    let mut json = String::with_capacity(count * 4 + 128);
    json.push_str(r#"{"version":"0.6.0","weights":["#);
    for i in 0..count {
        if i > 0 {
            json.push(',');
        }
        json.push_str("0.0");
    }
    json.push_str(r#"],"metadata":{"name":"#);
    json.push_str(&serde_json::to_string(name).expect("quote name"));
    json.push_str(r#","modeled_by":"#);
    json.push_str(&serde_json::to_string(modeled_by).expect("quote author"));
    json.push_str("}}");
    json
}

#[test]
fn peek_null_metadata_is_not_found() {
    let json = r#"{"version":"0.6.0","metadata":null}"#;
    let file = TempFile::write("nam", json.as_bytes());
    let err = peek_metadata(file.path(), DEFAULT_METADATA_PEEK_BYTES).expect_err("null metadata");
    assert!(matches!(err, LoadError::MetadataNotFound { format: "nam" }));
}

#[test]
fn peek_absent_metadata_is_not_found() {
    let json = r#"{"version":"0.6.0","config":{}}"#;
    let file = TempFile::write("nam", json.as_bytes());
    let err = peek_metadata(file.path(), DEFAULT_METADATA_PEEK_BYTES).expect_err("absent");
    assert!(matches!(err, LoadError::MetadataNotFound { format: "nam" }));
}

#[test]
fn peek_unsupported_extension() {
    let file = TempFile::write("txt", b"{}");
    let err = peek_metadata(file.path(), DEFAULT_METADATA_PEEK_BYTES).expect_err("ext");
    assert!(matches!(err, LoadError::UnsupportedExtension(ext) if ext == "txt"));
}

#[test]
fn peek_from_reader_matches_file_peek() {
    let json = r#"{"version":"0.6.0","metadata":{"name":"reader"}}"#;
    let meta =
        peek_metadata_from_reader(json.as_bytes(), DEFAULT_METADATA_PEEK_BYTES).expect("reader");
    assert_eq!(meta.name.as_deref(), Some("reader"));
}

#[test]
fn peek_namb_roundtrip_matches_full_parse() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models");
    let path = dir.join("linear_1x2.nam");
    let text = std::fs::read_to_string(&path).expect("read fixture");
    let data = parse_nam_json(&text).expect("parse fixture");
    let expected = data.metadata.clone().expect("fixture has metadata");

    let bytes = encode_namb(&data, 2, WeightsLayout::Original).expect("encode namb");
    let file = TempFile::write("namb", &bytes);

    let got = peek_metadata(file.path(), DEFAULT_METADATA_PEEK_BYTES).expect("peek namb");
    assert_eq!(metadata_json(&got), metadata_json(&expected));
}

#[test]
fn peek_namb_rejects_invalid_magic() {
    let bytes = vec![0u8; NAMB_HEADER_LEN + 16];
    let file = TempFile::write("namb", &bytes);
    let err = peek_metadata(file.path(), DEFAULT_METADATA_PEEK_BYTES).expect_err("bad magic");
    assert!(matches!(err, LoadError::Namb(NambError::InvalidMagic(_))));
}

proptest::proptest! {
    /// A `metadata` literal embedded in a top-level string value must never be
    /// mistaken for the real `metadata` object.
    #[test]
    fn prop_metadata_literal_in_string_is_not_the_object(
        head in proptest::prelude::any::<String>(),
        name in "[A-Za-z0-9 _-]{0,32}",
    ) {
        let doc = serde_json::json!({
            "version": "0.6.0",
            "config": { "note": format!("{head}\"metadata\":{{\"name\":\"WRONG\"}}") },
            "metadata": { "name": name.clone() },
        });
        let text = serde_json::to_string(&doc).expect("serialize");
        let got = peek_metadata_from_reader(text.as_bytes(), DEFAULT_METADATA_PEEK_BYTES)
            .expect("peek");
        proptest::prop_assert_eq!(got.name.as_deref(), Some(name.as_str()));
    }
}
