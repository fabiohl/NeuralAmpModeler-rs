// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::AlignedVec;
use crate::models::a2::conv1d::A2Conv1d;
use crate::models::a2::film::FiLMConfig;
use crate::models::a2::film::FiLMLayer;
use crate::models::a2::layer::A2Layer;

fn make_minimal_layer(channels: usize) -> A2Layer {
    let kernel_size = 6;
    let num_blocks = channels.div_ceil(4);
    let conv_w_padded = num_blocks * 4 * channels * kernel_size;
    let conv_w = AlignedVec::new(conv_w_padded, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    let conv_b = AlignedVec::new(channels, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    let conv = A2Conv1d::new(conv_w, conv_b, true, 1, channels, channels, kernel_size);
    let mixin_w = AlignedVec::new(channels, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    let l1x1_w = AlignedVec::new(channels * channels, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    let l1x1_b = AlignedVec::new(channels, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b)
}

// =============================================================================
// read_slice tests
// =============================================================================

#[test]
fn test_read_slice_normal() {
    let weights = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mut pos = 1;
    let result = read_slice(&weights, &mut pos, 2, weights.len(), "test");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), &[2.0, 3.0]);
    assert_eq!(pos, 3);
}

#[test]
fn test_read_slice_exact_boundary() {
    let weights = [1.0f32, 2.0, 3.0];
    let mut pos = 0;
    let result = read_slice(&weights, &mut pos, 3, weights.len(), "full");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), &[1.0, 2.0, 3.0]);
    assert_eq!(pos, 3);
}

#[test]
fn test_read_slice_exhausted() {
    let weights = [1.0f32, 2.0, 3.0];
    let mut pos = 2;
    let result = read_slice(&weights, &mut pos, 2, weights.len(), "over");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("stream exhausted"));
    assert!(err.contains("\"over\""));
}

#[test]
fn test_read_slice_zero_elements() {
    let weights = [1.0f32, 2.0, 3.0];
    let mut pos = 1;
    let result = read_slice(&weights, &mut pos, 0, weights.len(), "zero");
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
    assert_eq!(pos, 1);
}

#[test]
fn test_read_slice_from_empty() {
    let weights: [f32; 0] = [];
    let mut pos = 0;
    let result = read_slice(&weights, &mut pos, 1, weights.len(), "empty");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("stream exhausted"));
}

#[test]
fn test_read_slice_label_in_error() {
    let weights = [1.0f32];
    let mut pos = 0;
    let result = read_slice(&weights, &mut pos, 3, weights.len(), "my_label");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("\"my_label\""));
}

// =============================================================================
// transpose_dense_f32 tests
// =============================================================================

#[test]
fn test_transpose_dense_f32_2x2() {
    // Row-major: [[1, 2], [3, 4]]
    let raw = [1.0f32, 2.0, 3.0, 4.0];
    let mut out = [0.0f32; 4];
    transpose_dense_f32(&raw, &mut out, 2, 2);
    // Col-major: [[1, 3], [2, 4]]
    assert_eq!(out, [1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_transpose_dense_f32_3x4() {
    // Row-major 3x4:
    // [[0, 1, 2, 3],
    //  [4, 5, 6, 7],
    //  [8, 9, 10, 11]]
    let raw: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let mut out = vec![0.0f32; 12];
    transpose_dense_f32(&raw, &mut out, 3, 4);
    // Col-major 4x3 (out_size cols, each of in_size rows):
    // col=0: raw[0]=0, raw[1]=1, raw[2]=2 → w[0]=0, w[4]=1, w[8]=2
    // col=1: raw[3]=3, raw[4]=4, raw[5]=5 → w[1]=3, w[5]=4, w[9]=5
    // col=2: raw[6]=6, raw[7]=7, raw[8]=8 → w[2]=6, w[6]=7, w[10]=8
    // col=3: raw[9]=9, raw[10]=10, raw[11]=11 → w[3]=9, w[7]=10, w[11]=11
    let expected: Vec<f32> = vec![0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0];
    assert_eq!(out, expected);
}

#[test]
fn test_transpose_dense_f32_single_column() {
    // Row-major 1x4
    let raw = [10.0f32, 20.0, 30.0, 40.0];
    let mut out = [0.0f32; 4];
    transpose_dense_f32(&raw, &mut out, 1, 4);
    assert_eq!(out, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_transpose_dense_f32_single_row() {
    // Row-major 4x1
    let raw = [5.0f32, 15.0, 25.0, 35.0];
    let mut out = [0.0f32; 4];
    transpose_dense_f32(&raw, &mut out, 4, 1);
    assert_eq!(out, [5.0, 15.0, 25.0, 35.0]);
}

#[test]
fn test_transpose_dense_f32_identity() {
    // Row-major 3x3 identity
    let raw = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut out = [0.0f32; 9];
    transpose_dense_f32(&raw, &mut out, 3, 3);
    assert_eq!(out, raw);
}

// =============================================================================
// transpose_conv1d_interleaved_4wide tests
// =============================================================================

#[test]
fn test_transpose_conv1d_4wide_ch4_k2() {
    // in_ch=4, out_ch=4, kernel=2 → raw has 4*4*2=32 elements
    // Row-major: raw[out_c * in_ch * kernel + in_c * kernel + k]
    let mut raw = vec![0.0f32; 32];
    for out_c in 0..4 {
        for in_c in 0..4 {
            for k in 0..2 {
                let idx = (out_c * 4 + in_c) * 2 + k;
                raw[idx] = (out_c * 100 + in_c * 10 + k) as f32;
            }
        }
    }
    let num_blocks = 4usize.div_ceil(4);
    let padded = num_blocks * 4 * 4 * 2; // = 1 * 4 * 4 * 2 = 32
    let mut out = vec![0.0f32; padded];
    transpose_conv1d_interleaved_4wide(&raw, &mut out, 4, 4, 2);

    // Verify non-zero: at least one position should be set
    let any_nonzero = out.iter().any(|&v| v != 0.0);
    assert!(any_nonzero, "transpose should fill some positions");

    // Spot-check: known mapping
    // target = b * (kernel * in_ch * 4) + k * (in_ch * 4) + in_c * 4 + lane
    // For b=0, k=0, in_c=0, lane=0, out_c=0:
    //   target = 0
    //   raw_idx = (0 * 4 + 0) * 2 + 0 = 0 → raw[0]
    assert_eq!(out[0], raw[0]);
}

#[test]
fn test_transpose_conv1d_4wide_ch5_k2() {
    // in_ch=5, out_ch=5, kernel=2 → raw has 5*5*2=50 elements
    let raw: Vec<f32> = (0..50).map(|i| (i + 1) as f32).collect();
    let num_blocks = 5usize.div_ceil(4); // = 2
    let padded = num_blocks * 4 * 5 * 2; // = 2 * 4 * 5 * 2 = 80
    let mut out = vec![0.0f32; padded];
    transpose_conv1d_interleaved_4wide(&raw, &mut out, 5, 5, 2);

    // Verify padded entries are non-zero in first block, zero in last lane of block 1
    let any_padded = out[0..32].iter().any(|&v| v != 0.0);
    assert!(any_padded, "first block should have non-zero entries");

    // Block 1, lane 4 would be out_c=5 which exceeds out_ch=5, so those stay zero.
    // All lane=3 positions in block 1 should be zero (out_c=7 > 4).
    for b1_base in (40..80).step_by(20) {
        for in_c in 0..5 {
            let idx = b1_base + in_c * 4 + 3; // lane=3 should have data (out_c=7, but out_ch=5)
            if idx < padded {
                assert_eq!(
                    out[idx], 0.0,
                    "lane 3 in block 1 should be zero (out_c={} > 4)",
                    7
                );
            }
        }
    }
}

#[test]
fn test_transpose_conv1d_4wide_ch3_k2() {
    // in_ch=3, out_ch=3, kernel=2 → raw has 18 elements
    let raw: Vec<f32> = (0..18).map(|i| (i + 10) as f32).collect();
    let num_blocks = 3usize.div_ceil(4); // = 1
    let padded = num_blocks * 4 * 3 * 2; // = 1 * 4 * 3 * 2 = 24
    let mut out = vec![0.0f32; padded];
    transpose_conv1d_interleaved_4wide(&raw, &mut out, 3, 3, 2);

    // First 3 lanes should be non-zero; lane 3 (index 3 in each group of 4) should be zero
    for b in 0..num_blocks {
        for k in 0..2 {
            for in_c in 0..3 {
                for lane in 0..3 {
                    let target = b * (2 * 3 * 4) + k * (3 * 4) + in_c * 4 + lane;
                    assert_ne!(
                        out[target], 0.0,
                        "position ({b},{k},{in_c},{lane}) should be non-zero"
                    );
                }
                // lane=3: out_c=3 >= out_ch=3 → should be zero
                let zero_target = b * (2 * 3 * 4) + k * (3 * 4) + in_c * 4 + 3;
                assert_eq!(out[zero_target], 0.0);
            }
        }
    }
}

// =============================================================================
// transpose_head_w tests
// =============================================================================

#[test]
fn test_transpose_head_w_ch3_k4() {
    // channels=3, kernel=4
    // raw: [ch*tap layout] ch0:t0,t1,t2,t3 ch1:t0,t1,t2,t3 ch2:t0,t1,t2,t3
    let raw: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let mut head = vec![0.0f32; 12];
    transpose_head_w(&raw, &mut head, 3, 4);
    // head: [tap*ch layout] t0:c0,c1,c2 t1:c0,c1,c2 t2:c0,c1,c2 t3:c0,c1,c2
    let expected = [
        0.0, 4.0, 8.0, // t0: ch0=raw[0], ch1=raw[4], ch2=raw[8]
        1.0, 5.0, 9.0, // t1
        2.0, 6.0, 10.0, // t2
        3.0, 7.0, 11.0, // t3
    ];
    assert_eq!(head, expected);
}

#[test]
fn test_transpose_head_w_ch8_k2() {
    // channels=8, kernel=2
    let raw: Vec<f32> = (0..16).map(|i| (i * 10) as f32).collect();
    let mut head = vec![0.0f32; 16];
    transpose_head_w(&raw, &mut head, 8, 2);
    // head[tap * 8 + ch] = raw[ch * 2 + tap]
    // t0: c0=raw[0]=0, c1=raw[2]=20, c2=40, c3=60, c4=80, c5=100, c6=120, c7=140
    // t1: c0=raw[1]=10, c1=raw[3]=30, ...
    let expected = [
        0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 10.0, 30.0, 50.0, 70.0, 90.0, 110.0,
        130.0, 150.0,
    ];
    assert_eq!(head, expected);
}

#[test]
fn test_transpose_head_w_ch1_k1() {
    let raw = [7.0f32];
    let mut head = [0.0f32; 1];
    transpose_head_w(&raw, &mut head, 1, 1);
    assert_eq!(head, [7.0]);
}

// =============================================================================
// FILM_KEYS constant tests
// =============================================================================

#[test]
fn test_film_keys_length() {
    assert_eq!(FILM_KEYS.len(), 8);
}

#[test]
fn test_film_keys_indices_unique() {
    let mut seen = [false; 8];
    for &(_, idx) in FILM_KEYS {
        assert!(!seen[idx], "FiLM slot index {idx} appears more than once");
        seen[idx] = true;
    }
    assert!(
        seen.iter().all(|&x| x),
        "all FiLM slots 0-7 should be covered"
    );
}

#[test]
fn test_film_keys_ordered() {
    // FILM_KEYS tuples must map indices 0-7 in order
    for (i, &(_, idx)) in FILM_KEYS.iter().enumerate() {
        assert_eq!(idx, i, "FILM_KEYS entry {i} has index {idx}");
    }
}

// =============================================================================
// film_weight_count tests
// =============================================================================

#[test]
fn test_film_weight_count_shift_true() {
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    // ch=8, cond=4 → g=1, ch_per_group=8, cond_per_group=4, out_per_group=16
    // w = 1 * 16 * 4 = 64
    assert_eq!(film_weight_count_cfg(&config, 4, 8), 64);
}

#[test]
fn test_film_weight_count_shift_false() {
    let config = FiLMConfig {
        active: true,
        shift: false,
        groups: 1,
    };
    // ch=8, cond=4 → out_per_group=8, w = 1 * 8 * 4 = 32
    assert_eq!(film_weight_count_cfg(&config, 4, 8), 32);
}

#[test]
fn test_film_weight_count_groups() {
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 2,
    };
    // ch=8, cond=4 → g=2, ch_per_group=4, cond_per_group=2, out_per_group=8
    // w = 2 * 8 * 2 = 32
    assert_eq!(film_weight_count_cfg(&config, 4, 8), 32);
}

#[test]
fn test_film_weight_count_inactive() {
    // Weight count is independent of active flag
    let config = FiLMConfig {
        active: false,
        shift: true,
        groups: 1,
    };
    assert_eq!(film_weight_count_cfg(&config, 4, 8), 64);
}

// =============================================================================
// film_bias_count tests
// =============================================================================

#[test]
fn test_film_bias_count_shift_true() {
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    assert_eq!(film_bias_count_cfg(&config, 8), 16); // channels * 2
}

#[test]
fn test_film_bias_count_shift_false() {
    let config = FiLMConfig {
        active: true,
        shift: false,
        groups: 1,
    };
    assert_eq!(film_bias_count_cfg(&config, 8), 8); // channels
}

#[test]
fn test_film_bias_count_odd_channels() {
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    assert_eq!(film_bias_count_cfg(&config, 3), 6);
}

#[test]
fn test_film_bias_count_inactive() {
    let config = FiLMConfig {
        active: false,
        shift: true,
        groups: 1,
    };
    assert_eq!(film_bias_count_cfg(&config, 8), 16);
}

// =============================================================================
// parse_single_film_config tests
// =============================================================================

#[test]
fn test_parse_single_film_config_default() {
    let raw = serde_json::json!({});
    let config = parse_single_film_config(&raw, "missing_key");
    assert_eq!(config, FiLMConfig::default());
}

#[test]
fn test_parse_single_film_config_active() {
    let raw = serde_json::json!({
        "my_film": {
            "active": true,
            "shift": true,
            "groups": 1
        }
    });
    let config = parse_single_film_config(&raw, "my_film");
    assert!(config.active);
    assert!(config.shift);
    assert_eq!(config.groups, 1);
}

#[test]
fn test_parse_single_film_config_shift_false() {
    let raw = serde_json::json!({
        "my_film": {
            "active": true,
            "shift": false,
            "groups": 1
        }
    });
    let config = parse_single_film_config(&raw, "my_film");
    assert!(config.active);
    assert!(!config.shift);
    assert_eq!(config.groups, 1);
}

#[test]
fn test_parse_single_film_config_groups() {
    let raw = serde_json::json!({
        "my_film": {
            "active": true,
            "shift": true,
            "groups": 4
        }
    });
    let config = parse_single_film_config(&raw, "my_film");
    assert_eq!(config.groups, 4);
}

#[test]
fn test_parse_single_film_config_inactive() {
    let raw = serde_json::json!({
        "my_film": {
            "active": false,
            "shift": true,
            "groups": 1
        }
    });
    let config = parse_single_film_config(&raw, "my_film");
    assert!(!config.active);
}

#[test]
fn test_parse_single_film_config_not_object() {
    let raw = serde_json::json!({
        "my_film": "not an object"
    });
    let config = parse_single_film_config(&raw, "my_film");
    assert_eq!(config, FiLMConfig::default());
}

// =============================================================================
// parse_film_configs tests
// =============================================================================

#[test]
fn test_parse_film_configs_all_default() {
    let raw = serde_json::json!({});
    let configs = parse_film_configs(&raw);
    assert_eq!(configs.len(), 8);
    for cfg in &configs {
        assert_eq!(*cfg, FiLMConfig::default());
    }
}

#[test]
fn test_parse_film_configs_one_active() {
    let raw = serde_json::json!({
        "conv_pre_film": {
            "active": true,
            "shift": false,
            "groups": 2
        }
    });
    let configs = parse_film_configs(&raw);
    assert_eq!(configs.len(), 8);
    assert!(configs[0].active);
    assert!(!configs[0].shift);
    assert_eq!(configs[0].groups, 2);
    // Others remain default
    for config in configs.iter().skip(1) {
        assert_eq!(*config, FiLMConfig::default());
    }
}

#[test]
fn test_parse_film_configs_multiple_active() {
    let raw = serde_json::json!({
        "conv_pre_film": { "active": true, "shift": true, "groups": 1 },
        "head1x1_post_film": { "active": true, "shift": false, "groups": 4 }
    });
    let configs = parse_film_configs(&raw);
    assert!(configs[0].active);
    assert!(configs[0].shift);
    assert_eq!(configs[0].groups, 1);
    assert!(configs[7].active);
    assert!(!configs[7].shift);
    assert_eq!(configs[7].groups, 4);
    for config in configs.iter().skip(1).take(6) {
        assert_eq!(*config, FiLMConfig::default());
    }
}

// =============================================================================
// set_layer_film tests
// =============================================================================

#[test]
fn test_set_layer_film_slot_0() {
    let mut layer = make_minimal_layer(4);
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    let film = FiLMLayer::load(config, 1, 4, vec![0.0f32; 8], vec![0.0f32; 8]).unwrap();
    assert!(layer.conv_pre_film.is_none());
    set_layer_film(&mut layer, &config, 0, film).unwrap();
    assert!(layer.conv_pre_film.is_some());
}

#[test]
fn test_set_layer_film_slot_1() {
    let mut layer = make_minimal_layer(4);
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    let film = FiLMLayer::load(config, 1, 4, vec![0.0f32; 8], vec![0.0f32; 8]).unwrap();
    assert!(layer.conv_post_film.is_none());
    set_layer_film(&mut layer, &config, 1, film).unwrap();
    assert!(layer.conv_post_film.is_some());
}

#[test]
fn test_set_layer_film_slot_2() {
    let mut layer = make_minimal_layer(4);
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    let film = FiLMLayer::load(config, 1, 4, vec![0.0f32; 8], vec![0.0f32; 8]).unwrap();
    set_layer_film(&mut layer, &config, 2, film).unwrap();
    assert!(layer.input_mixin_pre_film.is_some());
}

#[test]
fn test_set_layer_film_slot_3() {
    let mut layer = make_minimal_layer(4);
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    let film = FiLMLayer::load(config, 1, 4, vec![0.0f32; 8], vec![0.0f32; 8]).unwrap();
    set_layer_film(&mut layer, &config, 3, film).unwrap();
    assert!(layer.input_mixin_post_film.is_some());
}

#[test]
fn test_set_layer_film_slot_4() {
    let mut layer = make_minimal_layer(4);
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    let film = FiLMLayer::load(config, 1, 4, vec![0.0f32; 8], vec![0.0f32; 8]).unwrap();
    set_layer_film(&mut layer, &config, 4, film).unwrap();
    assert!(layer.activation_pre_film.is_some());
}

#[test]
fn test_set_layer_film_slot_5() {
    let mut layer = make_minimal_layer(4);
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    let film = FiLMLayer::load(config, 1, 4, vec![0.0f32; 8], vec![0.0f32; 8]).unwrap();
    set_layer_film(&mut layer, &config, 5, film).unwrap();
    assert!(layer.activation_post_film.is_some());
}

#[test]
fn test_set_layer_film_slot_6() {
    let mut layer = make_minimal_layer(4);
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    let film = FiLMLayer::load(config, 1, 4, vec![0.0f32; 8], vec![0.0f32; 8]).unwrap();
    set_layer_film(&mut layer, &config, 6, film).unwrap();
    assert!(layer.layer1x1_post_film.is_some());
}

#[test]
fn test_set_layer_film_slot_7() {
    let mut layer = make_minimal_layer(4);
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    let film = FiLMLayer::load(config, 1, 4, vec![0.0f32; 8], vec![0.0f32; 8]).unwrap();
    set_layer_film(&mut layer, &config, 7, film).unwrap();
    assert!(layer.head1x1_post_film.is_some());
}

#[test]
fn test_set_layer_film_out_of_range() {
    let mut layer = make_minimal_layer(4);
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    let film = FiLMLayer::load(config, 1, 4, vec![0.0f32; 8], vec![0.0f32; 8]).unwrap();
    let result = set_layer_film(&mut layer, &config, 8, film);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("out of range"));
}

#[test]
fn test_set_layer_film_overwrite() {
    let mut layer = make_minimal_layer(4);
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    let film1 = FiLMLayer::load(config, 1, 4, vec![0.0f32; 8], vec![1.0f32; 8]).unwrap();
    let film2 = FiLMLayer::load(config, 1, 4, vec![0.0f32; 8], vec![2.0f32; 8]).unwrap();
    set_layer_film(&mut layer, &config, 0, film1).unwrap();
    set_layer_film(&mut layer, &config, 0, film2).unwrap();
    let film = layer.conv_pre_film.as_ref().unwrap();
    assert_eq!(film.bias[0], 2.0);
}

// =============================================================================
// load_film_for_layer tests
// =============================================================================

#[test]
fn test_load_film_no_active_configs() {
    let mut layer = make_minimal_layer(8);
    let configs = [FiLMConfig::default(); 8];
    let weights = [1.0f32; 100];
    let mut pos = 5;
    let total = weights.len();
    let result = load_film_for_layer(&mut layer, &configs, 8, 1, 1, &weights, &mut pos, total, 0);
    assert!(result.is_ok());
    assert_eq!(
        pos, 5,
        "no weights consumed when no FiLM configs are active"
    );
    assert!(layer.conv_pre_film.is_none());
}

#[test]
fn test_load_film_one_active_config() {
    let mut layer = make_minimal_layer(8);
    let mut configs = [FiLMConfig::default(); 8];
    configs[0] = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    // ch=8, cond=1, shift=true → weight_count = 1*16*1 = 16, bias_count = 16
    let w_count = 16;
    let b_count = 16;
    let total = 100;
    let mut weights = vec![0.0f32; total];
    // Fill known values
    for (i, w) in weights.iter_mut().enumerate().take(w_count) {
        *w = (i + 1) as f32;
    }
    for (i, w) in weights.iter_mut().enumerate().skip(w_count).take(b_count) {
        *w = (i - w_count + 50) as f32;
    }

    let mut pos = 0;
    let result = load_film_for_layer(&mut layer, &configs, 8, 1, 1, &weights, &mut pos, total, 0);
    assert!(result.is_ok());
    assert_eq!(pos, w_count + b_count);
    assert!(layer.conv_pre_film.is_some());

    let film = layer.conv_pre_film.as_ref().unwrap();
    assert_eq!(film.weights.len(), w_count);
    assert_eq!(film.bias.len(), b_count);
    assert_eq!(film.weights[0], 1.0);
    assert_eq!(film.bias[0], 50.0);
}

#[test]
fn test_load_film_two_active_configs() {
    let mut layer = make_minimal_layer(8);
    let mut configs = [FiLMConfig::default(); 8];
    configs[0] = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    }; // conv_pre_film
    configs[1] = FiLMConfig {
        active: true,
        shift: false,
        groups: 1,
    }; // conv_post_film (scale only)

    // config 0: w=16, b=16
    // config 1: w=8, b=8
    let c0_w = 16;
    let c0_b = 16;
    let c1_w = 8;
    let c1_b = 8;
    let total = c0_w + c0_b + c1_w + c1_b;

    let mut weights = vec![0.0f32; total];
    for (i, w) in weights.iter_mut().enumerate() {
        *w = i as f32;
    }

    let mut pos = 0;
    let result = load_film_for_layer(&mut layer, &configs, 8, 1, 1, &weights, &mut pos, total, 0);
    assert!(result.is_ok());
    assert_eq!(pos, total);
    assert!(layer.conv_pre_film.is_some());
    assert!(layer.conv_post_film.is_some());

    let pre = layer.conv_pre_film.as_ref().unwrap();
    assert_eq!(pre.weights.len(), c0_w);
    assert_eq!(pre.bias.len(), c0_b);
    assert_eq!(pre.weights[0], 0.0);

    let post = layer.conv_post_film.as_ref().unwrap();
    assert_eq!(post.weights.len(), c1_w);
    assert_eq!(post.bias.len(), c1_b);
    // config 1 weights start at offset c0_w + c0_b = 32
    assert_eq!(post.weights[0], 32.0);
}

#[test]
fn test_load_film_exhausted_mid_stream() {
    let mut layer = make_minimal_layer(8);
    let mut configs = [FiLMConfig::default(); 8];
    configs[0] = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    // Needs w=16 + b=16 = 32 elements, but only 10 available
    let weights = vec![0.0f32; 10];
    let mut pos = 0;
    let result = load_film_for_layer(
        &mut layer,
        &configs,
        8,
        1,
        1,
        &weights,
        &mut pos,
        weights.len(),
        0,
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("stream exhausted"));
}

#[test]
fn test_load_film_with_groups() {
    let mut layer = make_minimal_layer(8);
    let mut configs = [FiLMConfig::default(); 8];
    configs[5] = FiLMConfig {
        active: true,
        shift: true,
        groups: 2,
    };
    // ch=8, g=2, cond=4, shift=true → generic path (cond_size > 1)
    // w = channels * mult * cond_size / g = 8*2*4/2 = 32
    // b = film_bias_count_generic(channels, shift=true) = 8*2 = 16
    let w_count = 32;
    let b_count = 16;
    let total = w_count + b_count;
    let weights = vec![0.0f32; total];
    let mut pos = 0;
    let result = load_film_for_layer(&mut layer, &configs, 8, 4, 1, &weights, &mut pos, total, 0);
    assert!(result.is_ok());
    assert_eq!(pos, total);
    assert!(layer.activation_post_film.is_some());
}
