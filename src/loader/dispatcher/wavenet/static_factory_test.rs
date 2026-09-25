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

const CONDITION_DSP_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/models/wavenet_condition_dsp.nam"
);

const A2_DYNAMIC_BLENDED_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/models/a2_dynamic_blended_ch3.nam"
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

/// A nested `condition_dsp` sub-model is deserialized via raw
/// `serde_json::from_value`, which bypasses `parse_nam_json` — the dispatcher
/// must re-apply the root version-range validation to the sub-model itself.
/// A sub-model declaring a version below the supported minimum must fail the
/// whole build, mirroring what the root parser would do for a standalone file.
#[test]
fn test_condition_dsp_version_below_minimum_rejected() {
    let json = fs::read_to_string(CONDITION_DSP_FIXTURE)
        .expect("Fixture wavenet_condition_dsp.nam not found");
    let mut root: serde_json::Value = serde_json::from_str(&json).expect("Fixture must be JSON");
    root["config"]["condition_dsp"]["version"] = serde_json::json!("0.4.0");
    let data: NamModelData =
        serde_json::from_value(root).expect("Root model must still deserialize");

    let err = build_model(&data)
        .err()
        .expect("Sub-model version below minimum 0.5.0 must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("below minimum supported 0.5.0"),
        "Error must cite the sub-model version range violation, got: {msg}"
    );
}

/// Same invariant as `test_condition_dsp_version_below_minimum_rejected`, but
/// exercised through the A2-Dynamic condition_dsp site: a valid A2-Dynamic
/// host model with a grafted `condition_dsp` sub-model whose version exceeds
/// the supported maximum must be rejected before any sub-model construction.
#[test]
fn test_condition_dsp_version_above_maximum_rejected_a2_dynamic() {
    let json = fs::read_to_string(A2_DYNAMIC_BLENDED_FIXTURE)
        .expect("Fixture a2_dynamic_blended_ch3.nam not found");
    let sub_json = fs::read_to_string(CONDITION_DSP_FIXTURE)
        .expect("Fixture wavenet_condition_dsp.nam not found");

    let mut root: serde_json::Value =
        serde_json::from_str(&json).expect("A2-Dynamic fixture must be valid JSON");
    let mut sub: serde_json::Value =
        serde_json::from_str(&sub_json).expect("condition_dsp fixture must be valid JSON");
    sub["version"] = serde_json::json!("1.0.0");
    root["config"]["condition_dsp"] = sub;

    let data: NamModelData =
        serde_json::from_value(root).expect("Grafted model must still deserialize");

    let err = build_model(&data)
        .err()
        .expect("Sub-model version above maximum 0.7.x must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("exceeds maximum supported 0.7.x"),
        "Error must cite the sub-model version range violation, got: {msg}"
    );
}
