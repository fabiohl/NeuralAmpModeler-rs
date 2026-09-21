// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::loader::nam_json::NamModelData;

/// Verifies that is_a2_shape routes FiLM-active models to Dynamic.
#[test]
fn test_a2_film_routes_to_dynamic() {
    let json = r#"{
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "in_channels": 1,
            "head_scale": 0.02,
            "head": null,
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "channels": 3,
                "bottleneck": 3,
                "head": {"out_channels": 1, "kernel_size": 16, "bias": true},
                "kernel_sizes": [6,6,6,6,6,6,6,6,6,6,6,6,6,6,15,15,6,6,6,6,6,6,6],
                "dilations": [1,3,7,17,41,101,239,1,3,7,17,41,101,239,1,13,1,3,7,17,41,101,239],
                "activation": [{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01}],
                "gating_mode": ["none","none","none","none","none","none","none","none","none","none","none","none","none","none","none","none","none","none","none","none","none","none","none"],
                "head1x1": {"active": false, "out_channels": 3, "groups": 1},
                "layer1x1": {"active": true, "groups": 1},
                "conv_post_film": {"active": true, "shift": true, "groups": 1},
                "input_mixin_post_film": {"active": true, "shift": true, "groups": 1},
                "activation_post_film": {"active": true, "shift": true, "groups": 1},
                "layer1x1_post_film": {"active": true, "shift": true, "groups": 1},
                "groups_input": 1,
                "groups_input_mixin": 1
            }]
        },
        "weights": [],
        "sample_rate": 48000
    }"#;

    let data: NamModelData = serde_json::from_str(json).expect("parse FiLM model JSON");
    match is_a2_shape(&data) {
        Some(A2TopologyResult::Dynamic) => {}
        Some(A2TopologyResult::KnownFastPath(ch)) => {
            panic!("FiLM model was routed to KnownFastPath({ch}) instead of Dynamic");
        }
        None => {
            panic!("FiLM model was not recognized as A2 (returned None)");
        }
    }
}

/// Full fast-path candidate JSON with a parameterizable `channels`/`bottleneck`
/// pair; every other strict A2 shape criterion (kernel sizes, dilations,
/// LeakyReLU, gating, layer1x1, layer-array head, groups) is satisfied.
fn a2_fastpath_candidate_json(channels: u64, bottleneck: u64) -> String {
    let leaky: Vec<String> = std::iter::repeat_n(
        r#"{"type":"LeakyReLU","negative_slope":0.01}"#.to_string(),
        23,
    )
    .collect();
    let gating: Vec<String> = std::iter::repeat_n(r#""none""#.to_string(), 23).collect();
    let kernel_sizes = "[6,6,6,6,6,6,6,6,6,6,6,6,6,6,15,15,6,6,6,6,6,6,6]";
    let dilations = "[1,3,7,17,41,101,239,1,3,7,17,41,101,239,1,13,1,3,7,17,41,101,239]";
    format!(
        r#"{{
            "version": "0.6.0",
            "architecture": "WaveNet",
            "config": {{
                "in_channels": 1,
                "head_scale": 0.02,
                "head": null,
                "layers": [{{
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": {channels},
                    "bottleneck": {bottleneck},
                    "head": {{"out_channels": 1, "kernel_size": 16, "bias": true}},
                    "kernel_sizes": {kernel_sizes},
                    "dilations": {dilations},
                    "activation": [{}],
                    "gating_mode": [{}],
                    "head1x1": {{"active": false, "out_channels": 3, "groups": 1}},
                    "layer1x1": {{"active": true, "groups": 1}},
                    "groups_input": 1,
                    "groups_input_mixin": 1
                }}]
            }},
            "weights": [],
            "sample_rate": 48000
        }}"#,
        leaky.join(","),
        gating.join(",")
    )
}

/// Control: a canonical CH=3 candidate still routes to `KnownFastPath(3)`.
#[test]
fn test_a2_fastpath_candidate_ch3_routes_to_known_fast_path() {
    let json = a2_fastpath_candidate_json(3, 3);
    let data: NamModelData = serde_json::from_str(&json).expect("parse A2 candidate JSON");
    assert_eq!(
        is_a2_shape(&data),
        Some(A2TopologyResult::KnownFastPath(3)),
        "control: channels=3/bottleneck=3 must match the CH=3 fast-path"
    );
}

/// F-RES2-05: `channels` above `u8::MAX` whose low byte lands on a valid
/// channel (259 as u8 == 3) must NOT be truncated into the CH=3 fast-path —
/// the declared value is compared in its own width before any narrow cast.
#[test]
fn test_a2_channels_truncation_does_not_route_to_fast_path() {
    for (channels, bottleneck) in [(259u64, 3u64), (264u64, 8u64)] {
        let json = a2_fastpath_candidate_json(channels, bottleneck);
        let data: NamModelData = serde_json::from_str(&json).expect("parse hostile A2 JSON");
        match is_a2_shape(&data) {
            Some(A2TopologyResult::Dynamic) => {}
            other => panic!(
                "channels={channels} (low byte truncated to a valid lane, bottleneck={bottleneck}) \
                 must route to Dynamic, got {other:?}"
            ),
        }
    }
}
