// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::loader::nam_json::NamModelData;

/// B.1.1: Verifies that is_a2_shape routes FiLM-active models to Dynamic.
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
