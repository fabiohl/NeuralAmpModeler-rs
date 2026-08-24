// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::loader::dispatcher::build_model;
use crate::loader::nam_json::parse_nam_json;
use std::fs;

const A2_MAX_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/models/wavenet_a2_max.nam"
);

#[test]
fn test_a2_max_flag_controlled() {
    let json = fs::read_to_string(A2_MAX_FIXTURE).expect("Fixture wavenet_a2_max.nam not found");
    let data = parse_nam_json(&json).expect("Failed to parse fixture");

    // SAFETY: this test runs single-threaded and no other thread reads
    // `NAM_A2_MAX_UNLOCK`, so mutating the process environment cannot race.
    unsafe {
        std::env::remove_var("NAM_A2_MAX_UNLOCK");
    }
    let result = build_model(&data);
    match result {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("KB-A2-MAX") || msg.contains("parity gap"),
                "Error must cite KB-A2-MAX / parity gap, got: {msg}"
            );
            assert!(
                msg.contains("fail-closed"),
                "Error must cite fail-closed, got: {msg}"
            );
        }
        Ok(_) => panic!("A2 Max must be rejected by default (no unlock flag set)"),
    }

    // SAFETY: this test runs single-threaded and no other thread reads
    // `NAM_A2_MAX_UNLOCK`, so mutating the process environment cannot race.
    unsafe {
        std::env::set_var("NAM_A2_MAX_UNLOCK", "1");
    }
    let model = build_model(&data).expect("A2 Max must build under NAM_A2_MAX_UNLOCK=1");
    assert!(
        matches!(*model, StaticModel::WavenetA2Dyn(_)),
        "Expected WavenetA2Dyn variant under unlock"
    );

    // SAFETY: this test runs single-threaded and no other thread reads
    // `NAM_A2_MAX_UNLOCK`, so mutating the process environment cannot race.
    unsafe {
        std::env::remove_var("NAM_A2_MAX_UNLOCK");
    }
}
