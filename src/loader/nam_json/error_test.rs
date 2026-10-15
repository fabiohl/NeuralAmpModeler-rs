// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Verifies that `Error::source()` of a `JsonError::Serde`
/// exposes the original `serde_json::Error` (previously discarded).
#[test]
fn serde_variant_preserves_source() {
    let serde_err = serde_json::from_str::<serde_json::Value>("{invalid")
        .expect_err("invalid JSON must fail to parse");
    let err = JsonError::from(serde_err);

    assert!(
        std::error::Error::source(&err).is_some(),
        "JsonError::Serde must retain its source error"
    );
    assert!(format!("{}", err).starts_with("JSON parse error:"));
}

/// PartialEq compares the rendered serde message (serde_json::Error is not PartialEq).
#[test]
fn serde_variant_partial_eq_compares_messages() {
    let a = JsonError::from(
        serde_json::from_str::<serde_json::Value>("{invalid").expect_err("must fail"),
    );
    let b = JsonError::from(
        serde_json::from_str::<serde_json::Value>("{invalid").expect_err("must fail"),
    );
    assert_eq!(a, b);

    let other = JsonError::from(
        serde_json::from_str::<serde_json::Value>("[also invalid").expect_err("must fail"),
    );
    assert_ne!(a, other);
}
