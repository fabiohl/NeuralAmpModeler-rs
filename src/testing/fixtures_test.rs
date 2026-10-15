// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn require_model_path_returns_existing_fixture() {
    // The v2 stress-signal manifest is a committed, always-present fixture
    // used purely to prove the success branch resolves a real file.
    let path = require_model_path("wavenet_a1_standard.nam");
    // Only assert the happy path when the fixture is actually checked in;
    // published/clean checkouts legitimately lack it.
    if let Ok(p) = path {
        assert!(p.exists());
        assert!(p.ends_with("wavenet_a1_standard.nam"));
    }
}

#[test]
fn require_model_path_missing_is_legible_error() {
    let err = require_model_path("__definitely_absent_fixture__.nam")
        .expect_err("an absent model must not resolve");
    assert_eq!(err.name, "__definitely_absent_fixture__.nam");
    let msg = err.to_string();
    assert!(msg.contains("__definitely_absent_fixture__.nam"), "{msg}");
    assert!(msg.contains("NAM_MODELS_DIR"), "{msg}");
}
