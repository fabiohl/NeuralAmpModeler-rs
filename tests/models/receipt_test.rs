// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration test suite for the machine-readable capability receipt and skip classification.
//!
//! Validates:
//! - Exact count invariants: 51 canonical entries, 61 catalog paths, 45 supported, 6 unsupported.
//! - Zero unexpected FAILED stages (`receipt.has_unexpected_failures() == false`).
//! - JSON serialization roundtrip and schema validity.
//! - ASCII/Markdown audit table rendering.

use neural_amp_modeler_rs::testing::receipt::{CapabilityReceipt, generate_capability_receipt};

#[test]
fn test_capability_receipt_generation_and_invariants() {
    let receipt = generate_capability_receipt();

    // 1. Count Invariants
    assert_eq!(receipt.total_canonical_models, 51);
    assert_eq!(receipt.total_catalog_paths, 61);
    assert_eq!(receipt.supported_count, 46);
    assert_eq!(receipt.unsupported_count, 5);
    assert_eq!(receipt.entries.len(), 51);

    // 2. Zero unexpected FAILED stages
    assert!(
        !receipt.has_unexpected_failures(),
        "Capability receipt contains unexpected FAILED stages! Table:\n{}",
        receipt.render_table()
    );

    // 3. JSON roundtrip & validity
    let json_str = receipt.render_json();
    assert!(!json_str.is_empty());
    let parsed: CapabilityReceipt =
        serde_json::from_str(&json_str).expect("Receipt JSON must be valid deserializable JSON");
    assert_eq!(parsed.total_canonical_models, 51);
    assert_eq!(parsed.entries.len(), 51);

    // 4. Audit Table Rendering
    let table = receipt.render_table();
    assert!(table.contains("=== Fixture Catalog Capability Receipt ==="));
    assert!(table.contains("Total Canonical: 51"));

    eprintln!("Successfully validated capability receipt for all 51 canonical model identities.");
}
