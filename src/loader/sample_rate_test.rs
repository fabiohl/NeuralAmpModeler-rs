// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::loader::nam_json::parse_nam_json;

fn inner_wavenet(rate: Option<f32>) -> String {
    let sr = match rate {
        Some(r) => format!(r#", "sample_rate": {r}"#),
        None => String::new(),
    };
    format!(
        r#"{{
                "version": "0.5.4",
                "architecture": "WaveNet",
                "config": {{
                    "layers": [{{
                        "input_size": 1, "condition_size": 1, "head_size": 4,
                        "channels": 8, "kernel_size": 3, "dilations": [1,2,4,8],
                        "activation": "Tanh", "gated": false, "head_bias": false
                    }}],
                    "head": null
                }},
                "weights": [0.0]{sr}
            }}"#
    )
}

fn linear_node(rate: Option<f32>) -> String {
    let sr = match rate {
        Some(r) => format!(r#", "sample_rate": {r}"#),
        None => String::new(),
    };
    format!(
        r#"{{
                "version": "0.7.0",
                "architecture": "Linear",
                "config": {{ "receptive_field": 1, "bias": false }},
                "weights": [1.0]{sr}
            }}"#
    )
}

fn sequential_envelope(rate: Option<f32>, children_csv: &str) -> String {
    let sr = match rate {
        Some(r) => format!(r#", "sample_rate": {r}"#),
        None => String::new(),
    };
    format!(
        r#"{{
                "version": "0.7.0",
                "architecture": "Sequential",
                "config": {{ "models": [{children_csv}] }},
                "weights": []{sr}
            }}"#
    )
}

fn rate_less_container(child_a: &str, child_b: &str) -> String {
    container_envelope(None, child_a, child_b)
}

fn container_envelope(rate: Option<f32>, child_a: &str, child_b: &str) -> String {
    let sr = match rate {
        Some(r) => format!(r#", "sample_rate": {r}"#),
        None => String::new(),
    };
    format!(
        r#"{{
                "version": "0.7.0",
                "architecture": "SlimmableContainer",
                "config": {{
                    "layers": [], "head": null,
                    "submodels": [
                        {{"max_value": 0.5, "model": {child_a}}},
                        {{"max_value": 1.0, "model": {child_b}}}
                    ]
                }},
                "weights": []{sr}
            }}"#
    )
}

/// DEC-01c: a rate-less container must reject a child-vs-child known-rate
/// conflict (stricter than C++ `container.cpp:35-46`, which never
/// cross-checks submodels).
#[test]
fn container_without_rate_rejects_child_rate_conflict() {
    let json = rate_less_container(&inner_wavenet(Some(44100.0)), &inner_wavenet(Some(48000.0)));
    let data = parse_nam_json(&json).expect("parse container");
    assert_eq!(
        effective_known_rate(&data),
        Err(NamErrorCode::SequentialSampleRateMismatch)
    );
}

/// DEC-01c: matching known rates resolve; all-unknown resolves to `None`
/// (the 48 kHz default is materialized only at the top-level boundary).
#[test]
fn container_without_rate_resolves_matching_or_unknown() {
    let matching =
        rate_less_container(&inner_wavenet(Some(44100.0)), &inner_wavenet(Some(44100.0)));
    let data = parse_nam_json(&matching).expect("parse container");
    assert_eq!(effective_known_rate(&data), Ok(Some(44100.0)));

    let unknown = rate_less_container(&inner_wavenet(None), &inner_wavenet(None));
    let data = parse_nam_json(&unknown).expect("parse container");
    assert_eq!(effective_known_rate(&data), Ok(None));
}

/// Table-driven test reproducing the complete DEC-01 decision matrix
/// from `docs/cpp_parity_map.md` §5.1 (L408-415).
#[test]
fn dec01_table_matrix() {
    struct TestCase {
        child1_rate: Option<f32>,
        child2_rate: Option<f32>,
        top_rate: Option<f32>,
        expected: Result<Option<f32>, NamErrorCode>,
        label: &'static str,
    }

    let cases = [
        TestCase {
            child1_rate: None,
            child2_rate: None,
            top_rate: None,
            expected: Ok(None),
            label: "All rates undeclared; resolves to None (48000 applied at root)",
        },
        TestCase {
            child1_rate: None,
            child2_rate: Some(44100.0),
            top_rate: None,
            expected: Ok(Some(44100.0)),
            label: "Child 1 unknown ignored; Child 2 establishes 44.1 kHz chain",
        },
        TestCase {
            child1_rate: Some(44100.0),
            child2_rate: Some(48000.0),
            top_rate: None,
            expected: Err(NamErrorCode::SequentialSampleRateMismatch),
            label: "Conflict between known child rates (SampleRateMismatch)",
        },
        TestCase {
            child1_rate: None,
            child2_rate: None,
            top_rate: Some(48000.0),
            expected: Ok(Some(48000.0)),
            label: "Top-level dictates 48 kHz; unknown children conform",
        },
        TestCase {
            child1_rate: Some(44100.0),
            child2_rate: None,
            top_rate: Some(48000.0),
            expected: Err(NamErrorCode::SequentialSampleRateMismatch),
            label: "Conflict between Child 1 (44.1 kHz) and Top-Level (48 kHz)",
        },
        TestCase {
            child1_rate: Some(48000.0),
            child2_rate: Some(48000.0),
            top_rate: Some(48000.0),
            expected: Ok(Some(48000.0)),
            label: "All matching 48 kHz resolves to 48 kHz",
        },
        TestCase {
            child1_rate: Some(-1.0),
            child2_rate: Some(44100.0),
            top_rate: None,
            expected: Ok(Some(44100.0)),
            label: "RFC 7.1 C++ sentinel -1.0 treated as unknown",
        },
        TestCase {
            child1_rate: Some(0.0),
            child2_rate: Some(44100.0),
            top_rate: None,
            expected: Ok(Some(44100.0)),
            label: "Non-positive sentinel 0.0 treated as unknown",
        },
    ];

    for c in &cases {
        let c1 = linear_node(c.child1_rate);
        let c2 = linear_node(c.child2_rate);
        let json = sequential_envelope(c.top_rate, &format!("{c1}, {c2}"));
        let data = parse_nam_json(&json).expect("parse sequential");
        let got = effective_known_rate(&data);
        assert_eq!(got, c.expected, "DEC-01 case failed: {}", c.label);
    }
}

/// Table-driven test reproducing the complete DEC-01c decision matrix
/// from `docs/cpp_parity_map.md` §5.1 (L467-472).
#[test]
fn dec01c_container_table_matrix() {
    struct TestCase {
        container_rate: Option<f32>,
        sub_a_rate: Option<f32>,
        sub_b_rate: Option<f32>,
        expected: Result<Option<f32>, NamErrorCode>,
        label: &'static str,
    }

    let cases = [
        TestCase {
            container_rate: None,
            sub_a_rate: Some(44100.0),
            sub_b_rate: Some(48000.0),
            expected: Err(NamErrorCode::SequentialSampleRateMismatch),
            label: "Submodels conflict with absent container rate -> REJECTED (E1309)",
        },
        TestCase {
            container_rate: None,
            sub_a_rate: Some(44100.0),
            sub_b_rate: None,
            expected: Ok(Some(44100.0)),
            label: "One known submodel with absent container rate -> 44100 Hz",
        },
        TestCase {
            container_rate: None,
            sub_a_rate: None,
            sub_b_rate: None,
            expected: Ok(None),
            label: "All unknown -> None (defaults to 48000 Hz at root)",
        },
        TestCase {
            container_rate: Some(48000.0),
            sub_a_rate: Some(44100.0),
            sub_b_rate: None,
            expected: Err(NamErrorCode::SequentialSampleRateMismatch),
            label: "Submodel 44.1k conflicts with container 48k -> REJECTED (E1309)",
        },
        TestCase {
            container_rate: Some(48000.0),
            sub_a_rate: None,
            sub_b_rate: None,
            expected: Ok(Some(48000.0)),
            label: "Submodels unknown conform to container 48k -> 48000 Hz",
        },
        TestCase {
            container_rate: Some(48000.0),
            sub_a_rate: Some(48000.0),
            sub_b_rate: None,
            expected: Ok(Some(48000.0)),
            label: "Matching container and submodel rates -> 48000 Hz",
        },
    ];

    for c in &cases {
        let sub_a = linear_node(c.sub_a_rate);
        let sub_b = linear_node(c.sub_b_rate);
        let json = container_envelope(c.container_rate, &sub_a, &sub_b);
        let data = parse_nam_json(&json).expect("parse container");
        let got = effective_known_rate(&data);
        assert_eq!(got, c.expected, "DEC-01c case failed: {}", c.label);
    }
}

/// Hardening: Sequential nesting up to MAX_SEQUENTIAL_DEPTH (8) is accepted;
/// depth 9 is rejected with SequentialRecursionDepthExceeded (E1311) without
/// cloning weights or deserializing model data.
#[test]
fn sequential_recursion_depth_limit() {
    fn build_nested_sequential(depth: usize) -> String {
        if depth == 1 {
            sequential_envelope(None, &linear_node(Some(48000.0)))
        } else {
            let inner = build_nested_sequential(depth - 1);
            sequential_envelope(None, &inner)
        }
    }

    // Depth 8: accepted
    let json_8 = build_nested_sequential(MAX_SEQUENTIAL_DEPTH);
    let data_8 = parse_nam_json(&json_8).expect("parse depth 8");
    assert_eq!(effective_known_rate(&data_8), Ok(Some(48000.0)));

    // Depth 9: rejected with E1311
    let json_9 = build_nested_sequential(MAX_SEQUENTIAL_DEPTH + 1);
    let data_9 = parse_nam_json(&json_9).expect("parse depth 9");
    assert_eq!(
        effective_known_rate(&data_9),
        Err(NamErrorCode::SequentialRecursionDepthExceeded)
    );
}

/// Hardening: Total sequential children across the tree is bounded by
/// MAX_SEQUENTIAL_TOTAL_CHILDREN (64). 64 is accepted; 65 is rejected with E1314.
#[test]
fn sequential_total_children_limit() {
    // 64 total children across the tree:
    // Root has 2 Sequential children.
    // Child A has 31 leaves.
    // Child B has 31 leaves.
    // Total sequential children = 2 (in root) + 31 (in A) + 31 (in B) = 64 (accepted)
    let leaves_31a = (0..31)
        .map(|_| linear_node(Some(44100.0)))
        .collect::<Vec<_>>()
        .join(", ");
    let leaves_31b = (0..31)
        .map(|_| linear_node(Some(44100.0)))
        .collect::<Vec<_>>()
        .join(", ");
    let child_a = sequential_envelope(None, &leaves_31a);
    let child_b = sequential_envelope(None, &leaves_31b);
    let json_64 = sequential_envelope(None, &format!("{child_a}, {child_b}"));
    let data_64 = parse_nam_json(&json_64).expect("parse 64 children");
    assert_eq!(effective_known_rate(&data_64), Ok(Some(44100.0)));

    // 65 total children across the tree:
    // Child B has 32 leaves.
    // Total = 2 + 31 + 32 = 65 > 64 (rejected with E1314)
    let leaves_32b = (0..32)
        .map(|_| linear_node(Some(44100.0)))
        .collect::<Vec<_>>()
        .join(", ");
    let child_b_32 = sequential_envelope(None, &leaves_32b);
    let json_65 = sequential_envelope(None, &format!("{child_a}, {child_b_32}"));
    let data_65 = parse_nam_json(&json_65).expect("parse 65 children");
    assert_eq!(
        effective_known_rate(&data_65),
        Err(NamErrorCode::SequentialChildrenExceedLimit)
    );
}

/// Hardening: Container nesting depth up to MAX_CONTAINER_DEPTH (4) is accepted;
/// depth 5 is rejected with NamJsonSubmodelsTooDeep (E1210).
#[test]
fn container_depth_limit() {
    fn build_nested_container(depth: usize) -> String {
        if depth == 0 {
            linear_node(Some(44100.0))
        } else {
            let inner = build_nested_container(depth - 1);
            container_envelope(None, &inner, &linear_node(Some(44100.0)))
        }
    }

    // Root + 4 nested containers = depth 4 (accepted)
    let json_4 = build_nested_container(MAX_CONTAINER_DEPTH + 1);
    let data_4 = parse_nam_json(&json_4).expect("parse container depth 4");
    assert_eq!(effective_known_rate(&data_4), Ok(Some(44100.0)));

    // Root + 5 nested containers = depth 5 (rejected with E1210)
    let json_5 = build_nested_container(MAX_CONTAINER_DEPTH + 2);
    let data_5 = parse_nam_json(&json_5).expect("parse container depth 5");
    assert_eq!(
        effective_known_rate(&data_5),
        Err(NamErrorCode::NamJsonSubmodelsTooDeep)
    );
}

/// Hardening: Submodels per container bounded by MAX_SUBMODELS (8).
/// 8 is accepted; 9 is rejected with NamJsonSubmodelsExceedLimit (E1209).
#[test]
fn container_submodels_count_limit() {
    fn container_with_n_submodels(n: usize) -> String {
        let entries = (0..n)
            .map(|i| {
                let max_val = (i + 1) as f32 / n as f32;
                let model = linear_node(Some(48000.0));
                format!(r#"{{"max_value": {max_val}, "model": {model}}}"#)
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            r#"{{
                    "version": "0.7.0",
                    "architecture": "SlimmableContainer",
                    "config": {{ "layers": [], "head": null, "submodels": [{entries}] }},
                    "weights": []
                }}"#
        )
    }

    let json_8 = container_with_n_submodels(MAX_SUBMODELS);
    let data_8 = parse_nam_json(&json_8).expect("parse 8 submodels");
    assert_eq!(effective_known_rate(&data_8), Ok(Some(48000.0)));

    // Nested container with 9 submodels (bypasses top-level serde schema visitor)
    let inner_9 = container_with_n_submodels(MAX_SUBMODELS + 1);
    let json_outer = format!(
        r#"{{
                "version": "0.7.0",
                "architecture": "SlimmableContainer",
                "config": {{
                    "layers": [], "head": null,
                    "submodels": [
                        {{"max_value": 1.0, "model": {inner_9}}}
                    ]
                }},
                "weights": []
            }}"#
    );
    let data_outer = parse_nam_json(&json_outer).expect("parse outer container");
    assert_eq!(
        effective_known_rate(&data_outer),
        Err(NamErrorCode::NamJsonSubmodelsExceedLimit)
    );
}

/// DEC-01b: Nested Sequential inside Container and Container inside Sequential.
#[test]
fn nested_sequential_and_container_cross_resolution() {
    // Container holding a Sequential child with 44100 Hz
    let seq = sequential_envelope(None, &linear_node(Some(44100.0)));
    let container_holding_seq = rate_less_container(&seq, &linear_node(Some(44100.0)));
    let data = parse_nam_json(&container_holding_seq).expect("parse");
    assert_eq!(effective_known_rate(&data), Ok(Some(44100.0)));

    // Sequential holding a Container child with 44100 Hz
    let cont = rate_less_container(&linear_node(Some(44100.0)), &linear_node(Some(44100.0)));
    let seq_holding_cont = sequential_envelope(None, &cont);
    let data_seq = parse_nam_json(&seq_holding_cont).expect("parse");
    assert_eq!(effective_known_rate(&data_seq), Ok(Some(44100.0)));
}
