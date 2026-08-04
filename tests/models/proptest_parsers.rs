// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Property-based testing and fuzzing for model JSON and binary parsers.

use neural_amp_modeler_rs::loader::nam_json::{
    A2TopologyResult, NamConfig, NamDate, NamLayerConfig, NamMetadata, NamModelData, WeightsLayout,
    get_lstm_topology, get_wavenet_topology, is_a2_shape, parse_nam_json,
};
use neural_amp_modeler_rs::loader::namb::{FLAG_HAS_CRC32, crc32_ieee_update, parse_namb};
use proptest::prelude::*;
use std::fs;

use neural_amp_modeler_rs::loader::nam_json::{
    MAX_A2_DYN_BOTTLENECK, MAX_A2_DYN_CHANNELS, MAX_CONVNET_CHANNELS, MAX_CONVNET_KERNEL_SIZE,
    MAX_DILATION, MAX_DILATIONS_PER_ARRAY, MAX_HEAD_SIZE, MAX_HIDDEN_SIZE, MAX_KERNEL_SIZE,
    MAX_LAYERS, MAX_LSTM_HIDDEN_SIZE, MAX_LSTM_LAYERS, MAX_RECEPTIVE_FIELD, MAX_TOTAL_STATE_FRAMES,
    MAX_WAVENET_ARRAYS, MAX_WAVENET_FREE_CHANNELS,
};

// Fuzz 1: Sends fully arbitrary bytes to the JSON parser.
// Ensures the parser handles data that is not valid UTF-8 without panicking.
proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]
    #[test]
    #[ignore]
    fn prop_fuzz_nam_json_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..4096)) {
        let json_str = String::from_utf8_lossy(&bytes);
        // The goal here is only to ensure absence of panics (memory safety).
        let _ = parse_nam_json(&json_str);
    }
}

/// Strategy: Generates a "near-valid" JSON.
/// Contains required fields but with mixed data types or corrupted structures.
/// Tests whether the parser correctly rejects semantically invalid data.
fn near_valid_json_strategy() -> impl Strategy<Value = String> {
    (
        any::<Option<String>>(),                     // version (optional)
        any::<String>(),                             // architecture (arbitrary string)
        prop::collection::vec(any::<f32>(), 0..100), // weights (arbitrary amount)
        any::<bool>(),                               // flag to decide whether to corrupt the JSON
    )
        .prop_map(|(ver, arch, weights, scramble)| {
            let mut json = serde_json::json!({
                "architecture": arch,
                "weights": weights,
                "config": {
                    "layers": [],
                    "head": null,
                    "head_scale": 0.02
                }
            });

            if let Some(v) = ver {
                json["version"] = serde_json::Value::String(v);
            }

            if scramble {
                // Forces type error: transforms string into number and object into array
                json["architecture"] = serde_json::Value::Number(123.into());
                json["config"] = serde_json::Value::Array(vec![]);
            }

            serde_json::to_string(&json).unwrap()
        })
}

// Fuzz 2: Near valid but randomized values
proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]
    #[test]
    #[ignore]
    fn prop_fuzz_nam_json_near_valid(json_str in near_valid_json_strategy()) {
        let _ = parse_nam_json(&json_str);
    }
}

// Fuzz 3: Truncation of valid JSONs.
// Simulates an incomplete download or corrupted file on disk.
proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]
    #[test]
    #[ignore]
    fn prop_fuzz_nam_json_truncated(cut_idx in 0usize..400_000) {
        // Uses a real fixture as the base for truncation
        let fixture_content = fs::read_to_string("tests/fixtures/models-nondist/EVH-5150-Lite.nam")
            .unwrap_or_else(|_| "{}".to_string());

        let mut idx = cut_idx;
        if idx > fixture_content.len() {
            idx = fixture_content.len();
        }

        // We truncate raw bytes and convert lossy to UTF-8
        // so as not to break multi-byte characters.
        let bytes = fixture_content.as_bytes();
        let truncated = &bytes[0..idx];
        let json_str = String::from_utf8_lossy(truncated);

        let _ = parse_nam_json(&json_str);
    }
}

/// Strategy: Generates JSONs with extreme or invalid numeric values (NaN/Inf).
/// Tests the system's resilience against mathematical instabilities originating from the weights file.
fn weight_overflow_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            Just(f32::MAX),
            Just(f32::MIN),
            Just(f32::INFINITY),
            Just(f32::NEG_INFINITY),
            Just(f32::NAN),
            Just(f32::MIN_POSITIVE),
            any::<f32>(),
        ],
        0..200
    ).prop_map(|weights| {
        // Builds a valid WaveNet topology, but populated with numeric "garbage".
        let json = serde_json::json!({
            "version": "0.5.4",
            "architecture": "WaveNet",
            "config": {
                "layers": [
                    {
                        "input_size": 1, "condition_size": 1, "head_size": 4,
                        "channels": 12, "kernel_size": 3, "dilations": [1,2,4,8,16,32,64],
                        "activation": "Tanh", "gated": false, "head_bias": false
                    },
                    {
                        "input_size": 1, "condition_size": 1, "head_size": 4,
                        "channels": 12, "kernel_size": 3, "dilations": [128,256,512,1,2,4,8,16,32,64,128,256,512],
                        "activation": "Tanh", "gated": false, "head_bias": true
                    }
                ],
                "head": null,
                "head_scale": 0.02
            },
            "weights": weights,
            "sample_rate": 48000
        });

        serde_json::to_string(&json).unwrap()
    })
}

// Fuzz 4: Weight overflow / weird f32s
proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]
    #[test]
    #[ignore]
    fn prop_fuzz_nam_json_weight_overflow(json_str in weight_overflow_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            // Just ensure accessing topology doesn't crash
            let _topo = get_wavenet_topology(&parsed);
        }
    }
}

// Fuzz 5: Arbitrary bytes in the binary parser (NAMB).
// The NAMB parser must be extremely resilient to malformed binary files.
proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]
    #[test]
    #[ignore]
    fn prop_fuzz_namb_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..8192)) {
        let _ = parse_namb(&bytes);
    }
}

// Strategy: Generates a valid synthetic NAMB file.
// Allows corrupting specific fields (Magic, CRC, Offsets) in the tests below.
prop_compose! {
    fn valid_namb_strategy()(weights in prop::collection::vec(any::<f32>(), 0..200)) -> Vec<u8> {
        let weights_offset: usize = 80;
        let total_size = weights_offset + weights.len() * 4;
        let mut sim_data = vec![0u8; total_size];

        // Header: Magic 'NAMB'
        sim_data[0..4].copy_from_slice(&0x4E414D42u32.to_le_bytes());
        // Version: 1
        sim_data[4..6].copy_from_slice(&1u16.to_le_bytes());
        // Weights offset
        sim_data[12..16].copy_from_slice(&(weights_offset as u32).to_le_bytes());
        // Short metadata
        sim_data[32..37].copy_from_slice(b"1.0.0");
        // DSP Params
        sim_data[64..68].copy_from_slice(&48000.0f32.to_le_bytes());
        sim_data[68..72].copy_from_slice(&12.0f32.to_le_bytes());
        sim_data[72..76].copy_from_slice(&(-6.0f32).to_le_bytes());

        // Write weights in Little Endian
        for (i, float_val) in weights.iter().enumerate() {
            let off = weights_offset + i * 4;
            sim_data[off..off + 4].copy_from_slice(&float_val.to_le_bytes());
        }

        // Compute actual CRC32 IEEE to ensure the file starts valid
        let crc = neural_amp_modeler_rs::loader::namb::crc32_ieee(&sim_data[weights_offset..]);
        sim_data[24..28].copy_from_slice(&crc.to_le_bytes());

        sim_data
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]

    // Tests rejection of files with invalid 'Magic Bytes'.
    #[test]
    #[ignore]
    fn prop_fuzz_namb_bad_magic(mut namb in valid_namb_strategy(), bad_magic in any::<u32>()) {
        prop_assume!(bad_magic != 0x4E414D42);
        let bad_bytes = bad_magic.to_le_bytes();
        prop_assume!(&bad_bytes != b"NAMB" && &bad_bytes != b"BMAN");
        namb[0..4].copy_from_slice(&bad_bytes);
        assert!(parse_namb(&namb).is_err(), "Parser accepted invalid Magic Byte!");
    }

    // Tests integrity via CRC. Flipping 1 bit should cause a parsing error.
    #[test]
    #[ignore]
    fn prop_fuzz_namb_bad_crc(mut namb in valid_namb_strategy(), bit_flip in 0..32usize) {
        let byte_idx = 24 + (bit_flip / 8);
        let bit_idx = bit_flip % 8;
        namb[byte_idx] ^= 1 << bit_idx;
        assert!(parse_namb(&namb).is_err(), "Parser accepted file with corrupted CRC!");
    }

    // Tests resilience to prematurely truncated files.
    #[test]
    #[ignore]
    fn prop_fuzz_namb_truncated(namb in valid_namb_strategy(), truncate_idx in any::<usize>()) {
        let idx = std::cmp::min(truncate_idx, namb.len().saturating_sub(1));
        let truncated = &namb[0..idx];
        assert!(parse_namb(truncated).is_err(), "Parser accepted truncated binary file!");
    }

    // Tests out-of-bounds offset attacks (Oversized Offset).
    #[test]
    #[ignore]
    fn prop_fuzz_namb_oversized_offset(mut namb in valid_namb_strategy(), offset_add in 1..10000u32) {
        let new_offset = namb.len() as u32 + offset_add;
        namb[12..16].copy_from_slice(&new_offset.to_le_bytes());
        assert!(parse_namb(&namb).is_err(), "Parser accepted out-of-bounds weights offset!");
    }
}

// ---------------------------------------------------------------------------
// S13.T02 — Shrinking strategy for NamModelData (via strategy functions)
// ---------------------------------------------------------------------------

/// Strategy for `NamLayerConfig` with shrinking.
fn arbitrary_layer_config() -> impl Strategy<Value = NamLayerConfig> {
    let channels_s = prop_oneof![
        Just(8usize),
        Just(12),
        Just(16),
        Just(20),
        Just(24),
        any::<usize>(),
    ];
    let kernel_s = prop_oneof![Just(1usize), Just(2), Just(3), any::<usize>()];
    let dilation_s = prop_oneof![
        Just(1usize),
        Just(2),
        Just(4),
        Just(8),
        Just(16),
        Just(32),
        Just(64),
        Just(128),
        Just(256),
        Just(512),
        any::<usize>(),
    ];
    let activation_s = prop_oneof![
        Just("Tanh".to_string()),
        Just("ReLU".to_string()),
        any::<String>()
    ];

    (
        any::<Option<usize>>(),
        any::<Option<usize>>(),
        any::<Option<usize>>(),
        channels_s,
        kernel_s,
        prop::collection::vec(dilation_s, 1..20),
        activation_s,
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(
            |(
                input_size,
                condition_size,
                head_size,
                channels,
                kernel_size,
                dilations,
                activation,
                gated,
                head_bias,
            )| NamLayerConfig {
                input_size,
                condition_size,
                head_size,
                channels: Some(channels),
                kernel_size: Some(kernel_size),
                dilations: Some(dilations),
                activation: Some(activation),
                gated: Some(gated),
                head_bias: Some(head_bias),
                ..Default::default()
            },
        )
}

/// Strategy for `NamConfig` head field.
/// Generates `None` (absent), `Some(Value::Null)` (null),
/// or `Some(Value::Object)` (head config with fields).
fn arbitrary_head_value() -> impl Strategy<Value = Option<serde_json::Value>> {
    prop::option::of(prop_oneof![
        Just(serde_json::Value::Null),
        (any::<u16>(), any::<bool>(), any::<u16>(), any::<u16>(),).prop_map(
            |(ch, bias, out_ch, ks)| {
                let mut map = serde_json::Map::new();
                map.insert(
                    "channels".into(),
                    serde_json::Value::Number((ch as u64).into()),
                );
                map.insert("bias".into(), serde_json::Value::Bool(bias));
                map.insert(
                    "out_channels".into(),
                    serde_json::Value::Number((out_ch as u64).into()),
                );
                map.insert(
                    "activation".into(),
                    serde_json::Value::String("Tanh".into()),
                );
                map.insert(
                    "kernel_size".into(),
                    serde_json::Value::Number((ks as u64).into()),
                );
                serde_json::Value::Object(map)
            }
        ),
    ])
}

/// Strategy for `NamConfig` with shrinking.
fn arbitrary_nam_config() -> impl Strategy<Value = NamConfig> {
    (
        prop::collection::vec(arbitrary_layer_config(), 1..6),
        arbitrary_head_value(),
        prop_oneof![Just(Some(0.02f32)), any::<Option<f32>>()],
        any::<Option<usize>>(),
        any::<Option<usize>>(),
    )
        .prop_map(
            |(layers, head, head_scale, num_layers, hidden_size)| NamConfig {
                layers,
                head,
                head_scale,
                num_layers,
                hidden_size,
                receptive_field: None,
                bias: None,
                submodels: None,
                ..Default::default()
            },
        )
}

/// Strategy for `NamDate` with shrinking.
fn arbitrary_nam_date() -> impl Strategy<Value = NamDate> {
    (
        any::<Option<i32>>(),
        any::<Option<i32>>(),
        any::<Option<i32>>(),
        any::<Option<i32>>(),
        any::<Option<i32>>(),
        any::<Option<i32>>(),
    )
        .prop_map(|(year, month, day, hour, minute, second)| NamDate {
            year,
            month,
            day,
            hour,
            minute,
            second,
        })
}

/// Strategy for `NamMetadata` with shrinking.
fn arbitrary_nam_metadata() -> impl Strategy<Value = NamMetadata> {
    (
        arbitrary_nam_date().prop_map(Some),
        any::<Option<String>>(),
        any::<Option<String>>(),
        any::<Option<String>>(),
        any::<Option<String>>(),
        any::<Option<String>>(),
        any::<Option<String>>(),
        any::<Option<bool>>(), // training presence flag
        any::<Option<f32>>(),
        any::<Option<f32>>(),
        any::<Option<f32>>(),
    )
        .prop_map(
            |(
                date,
                name,
                modeled_by,
                gear_make,
                gear_model,
                gear_type,
                tone_type,
                training_has,
                input_level_dbu,
                output_level_dbu,
                loudness,
            )| {
                let training = training_has.and_then(|has| {
                    if has {
                        Some(serde_json::json!({"epochs": 100, "lr": 0.001}))
                    } else {
                        None
                    }
                });
                NamMetadata {
                    date,
                    name,
                    modeled_by,
                    gear_make,
                    gear_model,
                    gear_type,
                    tone_type,
                    training,
                    input_level_dbu,
                    output_level_dbu,
                    loudness,
                }
            },
        )
}

/// Shrinking strategy for `NamModelData`.
///
/// Generates synthetic models (WaveNet or LSTM) with random weights, metadata,
/// and configuration. Proptest shrinking automatically reduces the model to the
/// smallest counter-example when an assertion fails.
pub fn arbitrary_nam_model_data() -> impl Strategy<Value = NamModelData> {
    let arch = prop_oneof![Just("WaveNet".to_string()), Just("LSTM".to_string()),];

    let layout = prop_oneof![
        Just(WeightsLayout::Original),
        Just(WeightsLayout::GateMajorLstm),
        Just(WeightsLayout::Interleaved4WaveNet),
    ];

    let version = prop_oneof![
        Just(Some("0.5.4".to_string())),
        Just(Some("0.6.0".to_string())),
        Just(Some("0.7.14".to_string())),
        Just(Some("0.3.2".to_string())),
    ];

    (
        version,
        arch,
        arbitrary_nam_config(),
        prop::collection::vec(any::<f32>(), 0..500),
        prop_oneof![
            Just(None),
            Just(Some(44100.0)),
            Just(Some(48000.0)),
            Just(Some(96000.0))
        ],
        arbitrary_nam_metadata().prop_map(Some),
        layout,
    )
        .prop_map(
            |(version, architecture, config, weights, sample_rate, metadata, weights_layout)| {
                NamModelData {
                    version,
                    architecture,
                    config,
                    weights,
                    sample_rate,
                    metadata,
                    weights_layout,
                }
            },
        )
}

// ---------------------------------------------------------------------------
// S16.T03 — JSON serialization roundtrip for NamModelData
// ---------------------------------------------------------------------------
//
// prop_model_data_serialization_roundtrip is executed in Fase 3
// of utils/tests-quick.sh (proptest suite, #[ignore]d by default,
// invoked via -- --ignored). Not orphaned: the ignore attribute is
// intentional because the 200-case sweep would slow down the fast
// pass. CI and pre-release workflows run the full suite including
// ignored proptest cases.

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(200)
    })]

    #[test]
    #[ignore]
    fn prop_model_data_serialization_roundtrip(data in arbitrary_nam_model_data()) {
        let json_str =
            serde_json::to_string(&data).expect("Failed to serialize NamModelData to JSON");
        let parsed = match parse_nam_json(&json_str) {
            Ok(d) => d,
            Err(_) => return Ok(()), // parser correctly rejected invalid data
        };
        let original =
            serde_json::to_value(&data).expect("Failed to convert original to serde_json::Value");
        let roundtripped = serde_json::to_value(&parsed)
            .expect("Failed to convert roundtripped to serde_json::Value");
        assert_eq!(
            original, roundtripped,
            "JSON roundtrip serialization mismatch"
        );
    }
}

// ---------------------------------------------------------------------------
// S13.T02 — 100k iterations: valid NAMB header + random body
// ---------------------------------------------------------------------------

/// Generates a byte-array with a syntactically valid NAMB v2 header
/// (magic, version, flags, offset, crc) followed by a completely random
/// weights body.
fn arbitrary_namb_bytes_strategy() -> impl Strategy<Value = Vec<u8>> {
    const HEADER_SIZE: usize = 80;

    (
        any::<u32>(),      // sample_rate (reinterpreted as bytes)
        any::<u32>(),      // input_level_dbu
        any::<u32>(),      // output_level_dbu
        any::<[u8; 32]>(), // version_str
        any::<u8>(),       // layout_type
        prop::collection::vec(any::<u8>(), 0..16384),
    )
        .prop_map(
            |(
                sample_rate_raw,
                input_level_raw,
                output_level_raw,
                version_str,
                layout_type,
                body,
            )| {
                let offset = HEADER_SIZE;
                let total_len = offset + body.len();
                let mut data = vec![0u8; total_len];

                // Magic 'NAMB'
                data[0..4].copy_from_slice(&0x4E414D42u32.to_le_bytes());
                // Version 2
                data[4..6].copy_from_slice(&2u16.to_le_bytes());
                // Layout type (0=Original, 1=GateMajorLstm, 2=Interleaved4WaveNet)
                data[6] = layout_type % 3;
                // Flags: FLAG_HAS_CRC32 set
                data[7] = FLAG_HAS_CRC32;
                // Weights offset = 80 (header only, no JSON)
                data[12..16].copy_from_slice(&(offset as u32).to_le_bytes());
                // Version string
                data[32..64].copy_from_slice(&version_str);
                // Sample rate
                data[64..68].copy_from_slice(&sample_rate_raw.to_le_bytes());
                // Input level dBu
                data[68..72].copy_from_slice(&input_level_raw.to_le_bytes());
                // Output level dBu
                data[72..76].copy_from_slice(&output_level_raw.to_le_bytes());

                // Copy body into weights section
                data[offset..].copy_from_slice(&body);

                // Compute CRC32 of the entire file excluding CRC32 field itself
                let crc = {
                    let mut crc_val = crc32_ieee_update(0xFFFFFFFFu32, &data[..24]);
                    crc_val = crc32_ieee_update(crc_val, &data[28..]);
                    crc_val ^ 0xFFFFFFFFu32
                };
                data[24..28].copy_from_slice(&crc.to_le_bytes());

                data
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(100_000)
    })]

    /// Fuzz 6: Valid NAMB v2 header + completely random weights body.
    ///
    /// Ensures the parser never panics with 100k combinations of arbitrary
    /// body, even when the header is syntactically correct.
    /// — Triggered exclusively via `utils/tests-long.sh`.
    #[test]
    #[ignore]
    fn prop_fuzz_namb_arbitrary_valid_header(bytes in arbitrary_namb_bytes_strategy()) {
        let _ = parse_namb(&bytes);
    }
}

// ---------------------------------------------------------------------------
// Adversarial dimension proptest: kernel_size, dilations, channels,
//      head_size, receptive_field — ensure Err, never abort/panic.
// ---------------------------------------------------------------------------

/// Strategy: generates a WaveNet model JSON with one or more adversarial
/// dimension fields (kernel_size, dilations count/value, head_size, array count).
fn adversarial_wavenet_json_strategy() -> impl Strategy<Value = String> {
    (
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
    )
        .prop_map(
            |(_ch_raw, k_raw, d_cnt_raw, d_val_raw, head_raw, arrays_raw, _pattern)| {
                // Select an attack pattern based on _pattern
                let pattern = _pattern % 4;

                let (ch, k, dil_cnt, dil_val, head, arrays) = match pattern {
                    0 => (8, MAX_KERNEL_SIZE + 1 + (k_raw % 1024), 10, 2, 4, 2), // extreme kernel
                    1 => (
                        8,
                        3,
                        MAX_DILATIONS_PER_ARRAY + 1 + (d_cnt_raw % 64),
                        2,
                        4,
                        2,
                    ), // many dilations
                    2 => (8, 3, 10, MAX_DILATION + 1 + (d_val_raw % 1024), 4, 2), // extreme dilation value
                    3 => (8, 3, 10, 2, MAX_HEAD_SIZE + 1 + (head_raw % 256), 2), // extreme head_size
                    _ => (8, 3, 10, 2, 4, MAX_WAVENET_ARRAYS + 1 + (arrays_raw % 4)), // many arrays
                };

                let dil_vec: Vec<usize> = if dil_cnt == 0 {
                    vec![1]
                } else {
                    let cnt = dil_cnt.min(256);
                    vec![dil_val.min(usize::MAX - 1); cnt]
                };
                let k_val = k.min(usize::MAX / 4096);
                let head_val = head.min(usize::MAX / 4096);

                let mut layers = Vec::new();
                let n = arrays.min(16);
                for i in 0..n {
                    let layer = serde_json::json!({
                        "channels": ch.min(MAX_WAVENET_FREE_CHANNELS),
                        "kernel_size": k_val,
                        "dilations": dil_vec.clone(),
                        "head_size": head_val,
                        "activation": "Tanh",
                        "gated": false,
                        "head_bias": i == n - 1,
                    });
                    layers.push(layer);
                }

                let json = serde_json::json!({
                    "version": "0.5.4",
                    "architecture": "WaveNet",
                    "config": {
                        "layers": layers,
                        "head": null,
                        "head_scale": 0.02
                    },
                    "weights": vec![0.0f32; 16],
                    "sample_rate": 48000
                });
                serde_json::to_string(&json).unwrap()
            },
        )
}

/// Strategy: generates a ConvNet model JSON with adversarial channels/kernel/dilations.
fn adversarial_convnet_json_strategy() -> impl Strategy<Value = String> {
    (
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
    )
        .prop_map(|(ch_raw, k_raw, d_raw, pattern)| {
            let (ch, k, dil_val, dil_cnt) = match pattern % 3 {
                0 => (MAX_CONVNET_CHANNELS + 1 + (ch_raw % 64), 3, 2, 1), // extreme channels
                1 => (8, MAX_CONVNET_KERNEL_SIZE + 1 + (k_raw % 64), 2, 1), // extreme kernel
                _ => (
                    8,
                    3,
                    MAX_DILATION + 1 + (d_raw % 1024),
                    MAX_DILATIONS_PER_ARRAY + 1,
                ), // extreme dilation
            };

            let dil_vec: Vec<usize> = {
                let cnt = dil_cnt.min(64);
                vec![dil_val.min(usize::MAX / 4096); cnt]
            };

            let json = serde_json::json!({
                "version": "0.7.0",
                "architecture": "ConvNet",
                "config": {
                    "layers": [{
                        "channels": ch.min(usize::MAX / 4096),
                        "kernel_size": k.min(usize::MAX / 4096),
                        "dilations": dil_vec,
                        "activation": "Tanh",
                    }],
                    "head": null,
                    "head_scale": 0.02
                },
                "weights": vec![0.0f32; 16],
                "sample_rate": 48000
            });
            serde_json::to_string(&json).unwrap()
        })
}

/// Strategy: generates a Linear model JSON with adversarial receptive_field.
fn adversarial_linear_json_strategy() -> impl Strategy<Value = String> {
    any::<usize>().prop_map(|rf_raw| {
        let rf = MAX_RECEPTIVE_FIELD + 1 + (rf_raw % 1024);
        let weights: Vec<f32> = vec![0.0f32; 16.min(rf)];
        let json = serde_json::json!({
            "version": "0.5.4",
            "architecture": "Linear",
            "config": {
                "layers": [],
                "head": null,
                "receptive_field": rf.min(usize::MAX / 4096),
                "bias": false
            },
            "weights": weights,
            "sample_rate": 48000
        });
        serde_json::to_string(&json).unwrap()
    })
}

// ---------------------------------------------------------------------------
// T23.2 — Adversarial A2-Dynamic dimensions: channels, bottleneck, kernel
//         sizes, dilations, multi-array cascades — exercise A2 shape detection
//         under stress, never abort/panic.
// ---------------------------------------------------------------------------

/// Strategy: generates a WaveNet model JSON targeting the A2-Dynamic code path
/// with adversarial dimension values (channels, bottleneck, kernel_sizes,
/// dilations, multi-array cascades with A2 features).
fn adversarial_a2_dynamic_json_strategy() -> impl Strategy<Value = String> {
    let a2_kernel_sizes: [usize; 23] = [
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6,
    ];
    let a2_dilations: [usize; 23] = [
        1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239,
    ];

    (
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
    )
        .prop_map(move |(raw, seed, pattern)| {
            match pattern % 8 {
                // 0: channels > MAX_A2_DYN_CHANNELS, canonical kernel_sizes/dilations, LeakyReLU
                0 => {
                    let ch = MAX_A2_DYN_CHANNELS + 1 + (raw % 512);
                    let ch = ch.min(usize::MAX / 4096);
                    let acts: Vec<serde_json::Value> = (0..23)
                        .map(|_| serde_json::json!({"type": "LeakyReLU", "negative_slope": 0.01}))
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.7.0",
                        "architecture": "WaveNet",
                        "config": {
                            "in_channels": 1,
                            "head_scale": 0.02,
                            "layers": [{
                                "input_size": 1,
                                "condition_size": 1,
                                "channels": ch,
                                "bottleneck": ch,
                                "kernel_sizes": a2_kernel_sizes,
                                "dilations": a2_dilations,
                                "head_size": 1,
                                "head_bias": true,
                                "activation": acts,
                                "layer_raw": {
                                    "activation": acts,
                                    "layer1x1": {"active": true, "groups": 1},
                                    "head1x1": {"active": false},
                                    "head": {"out_channels": 1, "kernel_size": 16, "bias": true}
                                }
                            }]
                        },
                        "weights": vec![0.0f32; 1024],
                        "sample_rate": 48000
                    })).unwrap()
                }
                // 1: extreme non-canonical kernel_sizes, canonical dilations, LeakyReLU → Dynamic
                1 => {
                    let mut ks = a2_kernel_sizes;
                    let idx = raw % 23;
                    ks[idx] = MAX_KERNEL_SIZE + 1 + (seed % 64);
                    let acts: Vec<serde_json::Value> = (0..23)
                        .map(|_| serde_json::json!({"type": "LeakyReLU", "negative_slope": 0.01}))
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.7.0",
                        "architecture": "WaveNet",
                        "config": {
                            "in_channels": 1,
                            "head_scale": 0.02,
                            "layers": [{
                                "input_size": 1,
                                "condition_size": 1,
                                "channels": 8,
                                "bottleneck": 8,
                                "kernel_sizes": ks,
                                "dilations": a2_dilations,
                                "head_size": 1,
                                "head_bias": true,
                                "activation": acts,
                                "layer_raw": {
                                    "activation": acts,
                                    "layer1x1": {"active": true, "groups": 1},
                                    "head1x1": {"active": false},
                                    "head": {"out_channels": 1, "kernel_size": 16, "bias": true}
                                }
                            }]
                        },
                        "weights": vec![0.0f32; 1024],
                        "sample_rate": 48000
                    })).unwrap()
                }
                // 2: extreme dilations with non-canonical values → Dynamic
                2 => {
                    let mut dils = a2_dilations;
                    let idx = raw % 23;
                    dils[idx] = MAX_DILATION + 1 + (seed % 4096);
                    let acts: Vec<serde_json::Value> = (0..23)
                        .map(|_| serde_json::json!({"type": "LeakyReLU", "negative_slope": 0.01}))
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.7.0",
                        "architecture": "WaveNet",
                        "config": {
                            "in_channels": 1,
                            "head_scale": 0.02,
                            "layers": [{
                                "input_size": 1,
                                "condition_size": 1,
                                "channels": 8,
                                "bottleneck": 8,
                                "kernel_sizes": a2_kernel_sizes,
                                "dilations": dils,
                                "head_size": 1,
                                "head_bias": true,
                                "activation": acts,
                                "layer_raw": {
                                    "activation": acts,
                                    "layer1x1": {"active": true, "groups": 1},
                                    "head1x1": {"active": false},
                                    "head": {"out_channels": 1, "kernel_size": 16, "bias": true}
                                }
                            }]
                        },
                        "weights": vec![0.0f32; 1024],
                        "sample_rate": 48000
                    })).unwrap()
                }
                // 3: multi-array A2 cascade with head1x1 active + extreme channels
                3 => {
                    let n_arrays = 2usize + (raw % 3);
                    let ch = 16usize + (seed % 240);
                    let acts: Vec<serde_json::Value> = (0..23)
                        .map(|_| serde_json::json!({"type": "LeakyReLU", "negative_slope": 0.01}))
                        .collect();
                    let layers: Vec<serde_json::Value> = (0..n_arrays)
                        .map(|i| {
                            let mut layer = serde_json::json!({
                                "input_size": 1,
                                "condition_size": 1,
                                "channels": ch + i,
                                "bottleneck": ch + i,
                                "kernel_sizes": a2_kernel_sizes,
                                "dilations": a2_dilations,
                                "head_size": 1,
                                "head_bias": true,
                                "activation": acts,
                                "layer_raw": {
                                    "activation": acts,
                                    "layer1x1": {"active": true, "groups": 1},
                                }
                            });
                            if i == n_arrays - 1 {
                                layer["layer_raw"]["head1x1"] =
                                    serde_json::json!({"active": true, "out_channels": ch + i, "groups": 1});
                            } else {
                                layer["layer_raw"]["head1x1"] =
                                    serde_json::json!({"active": false});
                            }
                            layer["layer_raw"]["head"] =
                                serde_json::json!({"out_channels": 1, "kernel_size": 16, "bias": true});
                            layer
                        })
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.7.0",
                        "architecture": "WaveNet",
                        "config": {
                            "in_channels": 1,
                            "head_scale": 0.02,
                            "layers": layers
                        },
                        "weights": vec![0.0f32; 4096],
                        "sample_rate": 48000
                    })).unwrap()
                }
                // 4: bottleneck != channels, valid A2 shape (ch=8, bn > MAX_A2_DYN_BOTTLENECK)
                4 => {
                    let bn = MAX_A2_DYN_BOTTLENECK + 1 + (raw % 512);
                    let bn = bn.min(usize::MAX / 4096);
                    let acts: Vec<serde_json::Value> = (0..23)
                        .map(|_| serde_json::json!({"type": "LeakyReLU", "negative_slope": 0.01}))
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.7.0",
                        "architecture": "WaveNet",
                        "config": {
                            "in_channels": 1,
                            "head_scale": 0.02,
                            "layers": [{
                                "input_size": 1,
                                "condition_size": 1,
                                "channels": 8,
                                "bottleneck": bn,
                                "kernel_sizes": a2_kernel_sizes,
                                "dilations": a2_dilations,
                                "head_size": 1,
                                "head_bias": true,
                                "activation": acts,
                                "layer_raw": {
                                    "activation": acts,
                                    "layer1x1": {"active": true, "groups": 1},
                                    "head1x1": {"active": false},
                                    "head": {"out_channels": 1, "kernel_size": 16, "bias": true}
                                }
                            }]
                        },
                        "weights": vec![0.0f32; 1024],
                        "sample_rate": 48000
                    })).unwrap()
                }
                // 5: non-canonical channels (not 3 or 8) → Dynamic
                5 => {
                    let ch_variants: [usize; 7] = [4, 12, 16, 32, 64, 128, 256];
                    let ch = ch_variants[raw % 7];
                    let acts: Vec<serde_json::Value> = (0..23)
                        .map(|_| serde_json::json!({"type": "LeakyReLU", "negative_slope": 0.01}))
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.7.0",
                        "architecture": "WaveNet",
                        "config": {
                            "in_channels": 1,
                            "head_scale": 0.02,
                            "layers": [{
                                "input_size": 1,
                                "condition_size": 1,
                                "channels": ch,
                                "bottleneck": ch,
                                "kernel_sizes": a2_kernel_sizes,
                                "dilations": a2_dilations,
                                "head_size": 1,
                                "head_bias": true,
                                "activation": acts,
                                "layer_raw": {
                                    "activation": acts,
                                    "layer1x1": {"active": true, "groups": 1},
                                    "head1x1": {"active": false},
                                    "head": {"out_channels": 1, "kernel_size": 16, "bias": true}
                                }
                            }]
                        },
                        "weights": vec![0.0f32; 1024],
                        "sample_rate": 48000
                    })).unwrap()
                }
                // 6: gated mode with canonical shape → Dynamic or None
                6 => {
                    let acts: Vec<serde_json::Value> = (0..23)
                        .map(|_| serde_json::json!({"type": "LeakyReLU", "negative_slope": 0.01}))
                        .collect();
                    let gates: Vec<&str> = vec!["Gated"; 23];
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.7.0",
                        "architecture": "WaveNet",
                        "config": {
                            "in_channels": 1,
                            "head_scale": 0.02,
                            "layers": [{
                                "input_size": 1,
                                "condition_size": 1,
                                "channels": 8,
                                "bottleneck": 8,
                                "kernel_sizes": a2_kernel_sizes,
                                "dilations": a2_dilations,
                                "head_size": 1,
                                "head_bias": true,
                                "activation": acts,
                                "layer_raw": {
                                    "activation": acts,
                                    "gating_mode": gates,
                                    "layer1x1": {"active": true, "groups": 1},
                                    "head1x1": {"active": false},
                                    "head": {"out_channels": 1, "kernel_size": 16, "bias": true}
                                }
                            }]
                        },
                        "weights": vec![0.0f32; 1024],
                        "sample_rate": 48000
                    })).unwrap()
                }
                // 7: scalar kernel_size (non-canonical, no kernel_sizes array) — A2 generic
                _ => {
                    let k = 3usize + (raw % 64);
                    let dils: Vec<usize> = a2_dilations.to_vec();
                    let acts: Vec<serde_json::Value> = (0..23)
                        .map(|_| serde_json::json!({"type": "LeakyReLU", "negative_slope": 0.01}))
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.7.0",
                        "architecture": "WaveNet",
                        "config": {
                            "in_channels": 1,
                            "head_scale": 0.02,
                            "layers": [{
                                "input_size": 1,
                                "condition_size": 1,
                                "channels": 8,
                                "bottleneck": 8,
                                "kernel_size": k,
                                "dilations": dils,
                                "head_size": 1,
                                "head_bias": true,
                                "activation": acts,
                                "layer_raw": {
                                    "activation": acts,
                                    "layer1x1": {"active": true, "groups": 1},
                                    "head1x1": {"active": false},
                                    "head": {"out_channels": 1, "kernel_size": 16, "bias": true}
                                }
                            }]
                        },
                        "weights": vec![0.0f32; 1024],
                        "sample_rate": 48000
                    })).unwrap()
                }
            }
        })
}

// ---------------------------------------------------------------------------
// T23.3 — Adversarial SlimmableContainer: exercises submodel count, recursion
//         depth, top-level topology overflows, missing fields, and version
//         bounds — never abort/panic.
// ---------------------------------------------------------------------------

/// Strategy: generates a SlimmableContainer model JSON with adversarial
/// submodel count, depth, topology, version, and structural properties.
fn adversarial_container_json_strategy() -> impl Strategy<Value = String> {
    (
        any::<usize>(), // raw value for dimension overflows
        any::<usize>(), // pattern selector
        any::<u8>(),    // jitter for submodel count
    )
        .prop_map(|(raw, pattern, jitter)| {
            let inner_model = serde_json::json!({
                "version": "0.7.0",
                "architecture": "WaveNet",
                "config": {
                    "layers": [{
                        "input_size": 1,
                        "condition_size": 1,
                        "channels": 4,
                        "kernel_size": 3,
                        "dilations": [1],
                        "head_size": 1,
                        "activation": "Tanh",
                        "gated": false,
                        "head_bias": true
                    }],
                    "head": null,
                    "head_scale": 0.02
                },
                "weights": vec![0.0f32; 32],
                "sample_rate": 48000
            });

            match pattern % 8 {
                // 0: > MAX_SUBMODELS submodels → Err at parse (SubmodelsExceedLimit)
                0 => {
                    let count = 9usize + (jitter as usize % 32);
                    let submodels: Vec<serde_json::Value> = (0..count)
                        .map(|i| {
                            serde_json::json!({
                                "max_value": (i + 1) as f32 / count as f32,
                                "model": inner_model
                            })
                        })
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "architecture": "SlimmableContainer",
                        "config": {
                            "submodels": submodels,
                            "layers": [],
                            "head": null
                        },
                        "weights": Vec::<f32>::new(),
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
                // 1: deeply nested containers exceeding MAX_CONTAINER_DEPTH (4)
                1 => {
                    let depth = 5usize + (jitter as usize % 4);
                    let innermost = inner_model.clone();
                    let mut nested = innermost;
                    for _ in 0..depth {
                        nested = serde_json::json!({
                            "architecture": "SlimmableContainer",
                            "config": {
                                "submodels": [{
                                    "max_value": 1.0,
                                    "model": nested
                                }],
                                "layers": [],
                                "head": null
                            },
                            "weights": Vec::<f32>::new(),
                            "sample_rate": 48000
                        });
                    }
                    serde_json::to_string(&serde_json::json!({
                        "architecture": "SlimmableContainer",
                        "config": {
                            "submodels": [{
                                "max_value": 1.0,
                                "model": nested
                            }],
                            "layers": [],
                            "head": null
                        },
                        "weights": Vec::<f32>::new(),
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
                // 2: top-level hidden_size > MAX_HIDDEN_SIZE → Err at parse
                2 => {
                    let hs = MAX_HIDDEN_SIZE + 1 + (raw % 2048);
                    serde_json::to_string(&serde_json::json!({
                        "architecture": "SlimmableContainer",
                        "config": {
                            "submodels": [{
                                "max_value": 1.0,
                                "model": inner_model
                            }],
                            "layers": [],
                            "head": null,
                            "hidden_size": hs.min(usize::MAX / 4096)
                        },
                        "weights": Vec::<f32>::new(),
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
                // 3: top-level layers > MAX_LAYERS → Err at parse
                3 => {
                    let n_layers = MAX_LAYERS + 1 + (raw % 16);
                    let layers: Vec<serde_json::Value> = (0..n_layers)
                        .map(|_| serde_json::json!({"input_size": 4, "output_size": 4}))
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "architecture": "SlimmableContainer",
                        "config": {
                            "submodels": [{
                                "max_value": 1.0,
                                "model": inner_model
                            }],
                            "layers": layers,
                            "head": null
                        },
                        "weights": Vec::<f32>::new(),
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
                // 4: unsupported version (< 0.5.0) → Err at parse
                4 => {
                    let ver = match raw % 3 {
                        0 => "0.4.0",
                        1 => "0.4.9",
                        _ => "0.1.0",
                    };
                    serde_json::to_string(&serde_json::json!({
                        "version": ver,
                        "architecture": "SlimmableContainer",
                        "config": {
                            "submodels": [{
                                "max_value": 1.0,
                                "model": inner_model
                            }],
                            "layers": [],
                            "head": null
                        },
                        "weights": Vec::<f32>::new(),
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
                // 5: submodel entry missing 'model' field → passes parse, dispatch fails
                5 => serde_json::to_string(&serde_json::json!({
                    "architecture": "SlimmableContainer",
                    "config": {
                        "submodels": [{
                            "max_value": 1.0
                        }],
                        "layers": [],
                        "head": null
                    },
                    "weights": Vec::<f32>::new(),
                    "sample_rate": 48000
                }))
                .unwrap(),
                // 6: exactly at limit — 8 submodels (edge case that *should* parse)
                6 => {
                    let submodels: Vec<serde_json::Value> = (0..8)
                        .map(|i| {
                            serde_json::json!({
                                "max_value": (i + 1) as f32 / 8.0,
                                "model": inner_model
                            })
                        })
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "architecture": "SlimmableContainer",
                        "config": {
                            "submodels": submodels,
                            "layers": [],
                            "head": null
                        },
                        "weights": Vec::<f32>::new(),
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
                // 7: container with no submodels field → passes parse, dispatch fails
                _ => serde_json::to_string(&serde_json::json!({
                    "architecture": "SlimmableContainer",
                    "config": {
                        "layers": [],
                        "head": null
                    },
                    "weights": Vec::<f32>::new(),
                    "sample_rate": 48000
                }))
                .unwrap(),
            }
        })
}

// ---------------------------------------------------------------------------
// Adversarial state budget. Generates models that combine near-max
//       kernel_size × dilation × channels × layer count to stress the new
//       MAX_TOTAL_STATE_FRAMES bound.
// ---------------------------------------------------------------------------

/// Strategy: generates a WaveNet model JSON with high aggregate state budget.
/// Targets the combination of kernel_size × dilation × channels across many
/// layers — the adversarial state budget DoS vector.
fn adversarial_state_budget_strategy() -> impl Strategy<Value = String> {
    (
        any::<usize>(), // pattern selector
        any::<u8>(),    // jitter for near-max values
    )
        .prop_map(|(pattern, jitter)| {
            let j = (jitter as usize) % 16;

            match pattern % 3 {
                // Case 0: just under the budget — should be Free (or Rejected if channels push over)
                0 => {
                    // Target ~60 Mi frames (under 64 Mi cap)
                    // 8 arrays × 4 dilations × (63 × 256 × 512) ≈ 63 × 256 × 512 × 32 = ~264M
                    // Too high. Let's use: 4 arrays, 8 dilations, k=16, d=1024, ch=128
                    // => 4 × 8 × 15 × 1024 × 128 = 62,914,560 ≈ 60 Mi (under 64 Mi)
                    let dil_vec: Vec<usize> = vec![1024usize.saturating_sub(j); 8];
                    let layers: Vec<serde_json::Value> = (0..4)
                        .map(|i| {
                            serde_json::json!({
                                "channels": 128,
                                "kernel_size": 16,
                                "dilations": dil_vec.clone(),
                                "head_size": 4,
                                "activation": "Tanh",
                                "gated": false,
                                "head_bias": i == 3,
                            })
                        })
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.5.4",
                        "architecture": "WaveNet",
                        "config": { "layers": layers, "head": null, "head_scale": 0.02 },
                        "weights": vec![0.0f32; 16],
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
                // Case 1: over the budget — should be Rejected
                1 => {
                    // Target ~100 Mi frames (over 64 Mi cap)
                    // 8 arrays × 4 dilations × (63 × 512 × 64) = 8 × 4 × 63 × 512 × 64 = ~262M
                    // Too high. Use: 8 arrays, 4 dilations, high k, d, ch combo
                    let cnt = (4 + j / 4).min(16);
                    let dil_val = 2048usize.saturating_add(j * 128).min(MAX_DILATION);
                    let dil_vec: Vec<usize> = vec![dil_val; cnt];
                    let arrays = (3 + j / 8).min(8);
                    let layers: Vec<serde_json::Value> = (0..arrays)
                        .map(|i| {
                            serde_json::json!({
                                "channels": 128usize.saturating_add(j * 8).min(MAX_WAVENET_FREE_CHANNELS),
                                "kernel_size": (32usize + j).min(MAX_KERNEL_SIZE),
                                "dilations": dil_vec.clone(),
                                "head_size": 4,
                                "activation": "Tanh",
                                "gated": false,
                                "head_bias": i == arrays - 1,
                            })
                        })
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.5.4",
                        "architecture": "WaveNet",
                        "config": { "layers": layers, "head": null, "head_scale": 0.02 },
                        "weights": vec![0.0f32; 16],
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
                // Case 2: channels=1 amplification — tiny weight file exploiting receptive field
                _ => {
                    // Worst case from adversarial state budget test: k=64, d=4096, ch=1, 8 arrays × 64 dilations
                    // => 512 layers × ((63 * 4096) + 1600) × 1 = 512 × ~259,648 = ~133 Mi frames
                    let dil_cnt = (MAX_DILATIONS_PER_ARRAY / 2 + j / 2).min(MAX_DILATIONS_PER_ARRAY);
                    let dil_val = (MAX_DILATION / 2 + j * 128).min(MAX_DILATION);
                    let dil_vec: Vec<usize> = vec![dil_val; dil_cnt];
                    let arrays = 2usize.saturating_add(j / 4).min(MAX_WAVENET_ARRAYS);
                    let layers: Vec<serde_json::Value> = (0..arrays)
                        .map(|i| {
                            serde_json::json!({
                                "channels": 1,
                                "kernel_size": (32usize + j).min(MAX_KERNEL_SIZE),
                                "dilations": dil_vec.clone(),
                                "head_size": 4,
                                "activation": "Tanh",
                                "gated": false,
                                "head_bias": i == arrays - 1,
                            })
                        })
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.5.4",
                        "architecture": "WaveNet",
                        "config": { "layers": layers, "head": null, "head_scale": 0.02 },
                        "weights": vec![0.0f32; 16],
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
            }
        })
}

// ---------------------------------------------------------------------------
// T23.1 — Adversarial LSTM dimensions: hidden_size, num_layers, channels,
//         weight count, missing fields — ensure Err/None, never abort/panic.
// ---------------------------------------------------------------------------

/// Strategy: generates an LSTM model JSON with one or more adversarial
/// dimension fields (hidden_size, num_layers exceeding bounds, in/out channels,
/// layers count, missing fields, weight mismatch).
fn adversarial_lstm_json_strategy() -> impl Strategy<Value = String> {
    (any::<usize>(), any::<usize>(), any::<usize>()).prop_map(|(raw, _seed, pattern)| {
        use MAX_LSTM_HIDDEN_SIZE as MHS;
        use MAX_LSTM_LAYERS as MLL;

        // Each arm forges a distinct adversarial LSTM JSON to exercise a different
        // rejection path in the loader (bounds overflow, missing fields, mismatches):
        //   0 — hidden_size exceeds MAX_LSTM_HIDDEN_SIZE
        //   1 — num_layers exceeds MAX_LSTM_LAYERS (weights sized to match)
        //   2 — num_layers = 0 (empty stack)
        //   3 — hidden_size exceeds MAX_HIDDEN_SIZE (WaveNet-bound constant)
        //   4 — explicit `layers` array exceeds MAX_LAYERS
        //   5 — in/out channels > 1 (weight/channel mismatch)
        //   6 — config field removed (num_layers or hidden_size missing)
        //   _ — both num_layers and hidden_size exceed bounds simultaneously
        match pattern % 8 {
            0 => {
                let hs = MHS + 1 + (raw % 2048);
                serde_json::to_string(&serde_json::json!({
                    "version": "0.5.4",
                    "architecture": "LSTM",
                    "config": {
                        "num_layers": 2,
                        "hidden_size": hs.min(usize::MAX / 4096),
                        "in_channels": 1,
                        "out_channels": 1
                    },
                    "weights": vec![0.0f32; 16],
                    "sample_rate": 48000
                }))
                .unwrap()
            }
            1 => {
                let nl = MLL + 1 + (raw % 128);
                let hs = 4usize;
                let wt = neural_amp_modeler_rs::models::lstm::lstm_weight_count(nl.min(256), hs);
                serde_json::to_string(&serde_json::json!({
                    "version": "0.5.4",
                    "architecture": "LSTM",
                    "config": {
                        "num_layers": nl.min(1024),
                        "hidden_size": hs,
                        "in_channels": 1,
                        "out_channels": 1
                    },
                    "weights": vec![0.0f32; wt.min(65536)],
                    "sample_rate": 48000
                }))
                .unwrap()
            }
            2 => serde_json::to_string(&serde_json::json!({
                "version": "0.5.4",
                "architecture": "LSTM",
                "config": {
                    "num_layers": 0,
                    "hidden_size": 4,
                    "in_channels": 1,
                    "out_channels": 1
                },
                "weights": vec![0.0f32; 16],
                "sample_rate": 48000
            }))
            .unwrap(),
            3 => {
                let hs = MAX_HIDDEN_SIZE + 1 + (raw % 2048);
                serde_json::to_string(&serde_json::json!({
                    "version": "0.5.4",
                    "architecture": "LSTM",
                    "config": {
                        "num_layers": 2,
                        "hidden_size": hs.min(usize::MAX / 4096),
                        "in_channels": 1,
                        "out_channels": 1
                    },
                    "weights": vec![0.0f32; 16],
                    "sample_rate": 48000
                }))
                .unwrap()
            }
            4 => {
                let layers: Vec<serde_json::Value> = (0..(MAX_LAYERS + 1 + (raw % 16)))
                    .map(|_| serde_json::json!({"input_size": 4, "output_size": 4}))
                    .collect();
                serde_json::to_string(&serde_json::json!({
                    "version": "0.5.4",
                    "architecture": "LSTM",
                    "config": {
                        "num_layers": 1,
                        "hidden_size": 4,
                        "in_channels": 1,
                        "out_channels": 1,
                        "layers": layers
                    },
                    "weights": vec![0.0f32; 16],
                    "sample_rate": 48000
                }))
                .unwrap()
            }
            5 => {
                let ch = 2usize + (raw % 8);
                serde_json::to_string(&serde_json::json!({
                    "version": "0.5.4",
                    "architecture": "LSTM",
                    "config": {
                        "num_layers": 2,
                        "hidden_size": 4,
                        "in_channels": ch,
                        "out_channels": ch
                    },
                    "weights": vec![0.0f32; 16],
                    "sample_rate": 48000
                }))
                .unwrap()
            }
            6 => {
                let missing = raw % 2 == 0;
                let mut cfg = serde_json::json!({
                    "num_layers": 2,
                    "hidden_size": 4,
                    "in_channels": 1,
                    "out_channels": 1
                });
                if missing {
                    cfg.as_object_mut().unwrap().remove("num_layers");
                } else {
                    cfg.as_object_mut().unwrap().remove("hidden_size");
                }
                serde_json::to_string(&serde_json::json!({
                    "version": "0.5.4",
                    "architecture": "LSTM",
                    "config": cfg,
                    "weights": vec![0.0f32; 16],
                    "sample_rate": 48000
                }))
                .unwrap()
            }
            _ => {
                let nl = MLL + 1 + (raw % 32);
                let hs = MHS + 1 + (_seed % 2048);
                let wt = neural_amp_modeler_rs::models::lstm::lstm_weight_count(
                    nl.min(MLL + 128),
                    hs.min(MHS + 2048),
                );
                serde_json::to_string(&serde_json::json!({
                    "version": "0.5.4",
                    "architecture": "LSTM",
                    "config": {
                        "num_layers": nl.min(1024),
                        "hidden_size": hs.min(usize::MAX / 4096),
                        "in_channels": 1,
                        "out_channels": 1
                    },
                    "weights": vec![0.0f32; wt.min(65536)],
                    "sample_rate": 48000
                }))
                .unwrap()
            }
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(10_000)
    })]

    /// Adversarial dimensions: ensures topology detection rejects
    /// models with dimensions exceeding safe bounds (never abort/panic).
    #[test]
    #[ignore]
    fn prop_fuzz_adversarial_wavenet_dims(json_str in adversarial_wavenet_json_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            let result = get_wavenet_topology(&parsed);
            // Must never match Known — adversarial dims should be Free (if valid)
            // or Rejected (if exceeding bounds).
            match result {
                neural_amp_modeler_rs::loader::nam_json::WavenetTopologyResult::Known(_) => {
                    panic!("adversarial dimensions should not match a catalog SKU");
                }
                neural_amp_modeler_rs::loader::nam_json::WavenetTopologyResult::Free(geom) => {
                    // If accepted as Free, dimensions must be within bounds
                    assert!(
                        geom.channels.iter().all(|&c| c <= MAX_WAVENET_FREE_CHANNELS),
                        "free geometry channels within bounds"
                    );
                    // Verify state budget — no Free model should exceed MAX_TOTAL_STATE_FRAMES
                    let total_state = geom
                        .dilations
                        .iter()
                        .zip(geom.kernel_sizes.iter())
                        .zip(geom.channels.iter())
                        .fold(0usize, |acc, ((dils, &k), &ch)| {
                            let rf = k.saturating_sub(1);
                            acc.wrapping_add(
                                dils
                                    .iter()
                                    .map(|&d| rf.saturating_mul(d).saturating_mul(ch))
                                    .sum::<usize>(),
                            )
                        });
                    assert!(
                        total_state <= MAX_TOTAL_STATE_FRAMES,
                        "Free geometry exceeded state budget: {total_state} > {MAX_TOTAL_STATE_FRAMES}"
                    );
                }
                neural_amp_modeler_rs::loader::nam_json::WavenetTopologyResult::Rejected(_) => {
                    // Expected for adversarial dims — nothing to check
                }
            }
        }
    }

    /// Adversarial ConvNet dimensions.
    #[test]
    #[ignore]
    fn prop_fuzz_adversarial_convnet_dims(json_str in adversarial_convnet_json_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            let topo = neural_amp_modeler_rs::loader::nam_json::get_convnet_topology(&parsed);
            // Adversarial dimensions should be rejected (None)
            assert!(topo.is_none(),
                "ConvNet with adversarial dimensions should be rejected, got: {topo:?}");
        }
    }

    /// Adversarial Linear receptive_field.
    #[test]
    #[ignore]
    fn prop_fuzz_adversarial_linear_dims(json_str in adversarial_linear_json_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            let topo = neural_amp_modeler_rs::loader::nam_json::get_linear_topology(&parsed);
            // Adversarial receptive_field should be rejected (None)
            assert!(topo.is_none(),
                "Linear with adversarial receptive_field should be rejected, got: {topo:?}");
        }
    }

    /// T23.1 — Adversarial LSTM dimensions: ensures topology detection rejects
    /// models with hidden_size, num_layers, channels, layers count, missing
    /// fields, or weight mismatches exceeding safe bounds (never abort/panic).
    #[test]
    #[ignore]
    fn prop_fuzz_adversarial_lstm_dims(json_str in adversarial_lstm_json_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            let result = get_lstm_topology(&parsed);
            match result {
                Ok(Some((nl, hs))) => {
                    panic!(
                        "adversarial LSTM should not resolve valid topology \
                         (num_layers={nl}, hidden_size={hs})"
                    );
                }
                Ok(None) => {
                    // Expected: topology detection rejects the model
                }
                Err(e) => {
                    // Also acceptable: explicit error from validation
                    let _ = e;
                }
            }
        }
    }

    /// T23.2 — Adversarial A2-Dynamic dimensions: ensures A2 shape detection
    /// handles WaveNet models with adversarial channel, bottleneck, kernel,
    /// dilation, multi-array, and gating dimensions (never abort/panic).
    #[test]
    #[ignore]
    fn prop_fuzz_adversarial_a2_dynamic_dims(json_str in adversarial_a2_dynamic_json_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            let result = is_a2_shape(&parsed);
            match result {
                Some(A2TopologyResult::KnownFastPath(ch)) => {
                    panic!(
                        "adversarial A2-Dynamic should not resolve KnownFastPath (ch={ch})"
                    );
                }
                Some(A2TopologyResult::Dynamic) => {
                    // Expected: adversarial dimensions trigger Dynamic path
                }
                None => {
                    // Acceptable: model does not qualify as A2
                }
            }
        }
    }

    /// T23.3 — Adversarial SlimmableContainer: ensures submodel count,
    /// recursion depth, topology overflow, version, and structural
    /// properties are rejected or handled without panic/abort.
    #[test]
    #[ignore]
    fn prop_fuzz_adversarial_container_dims(json_str in adversarial_container_json_strategy()) {
        let result = parse_nam_json(&json_str);
        match result {
            Ok(_data) => {
                // Acceptable: some adversarial patterns pass parsing
                // (e.g. 8 submodels, deeply nested JSON). Dispatch would
                // catch them later.
            }
            Err(_e) => {
                // Also acceptable: parse-time rejection (e.g. >8 submodels,
                // topology overflow, unsupported version).
            }
        }
    }

    /// Adversarial state budget: ensures models exceeding
    /// MAX_TOTAL_STATE_FRAMES are Rejected, while others may be Free or Rejected
    /// but never cause panic or incorrect acceptance.
    #[test]
    #[ignore]
    fn prop_fuzz_adversarial_state_budget(json_str in adversarial_state_budget_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            let result = get_wavenet_topology(&parsed);
            match result {
                neural_amp_modeler_rs::loader::nam_json::WavenetTopologyResult::Known(_) => {
                    panic!("adversarial state budget should not match a catalog SKU");
                }
                neural_amp_modeler_rs::loader::nam_json::WavenetTopologyResult::Free(geom) => {
                    // If accepted, must not exceed budget
                    let total_state = geom
                        .dilations
                        .iter()
                        .zip(geom.kernel_sizes.iter())
                        .zip(geom.channels.iter())
                        .fold(0usize, |acc, ((dils, &k), &ch)| {
                            let rf = k.saturating_sub(1);
                            acc.wrapping_add(
                                dils
                                    .iter()
                                    .map(|&d| rf.saturating_mul(d).saturating_mul(ch))
                                    .sum::<usize>(),
                            )
                        });
                    assert!(
                        total_state <= MAX_TOTAL_STATE_FRAMES,
                        "Free geometry exceeded state budget: {total_state} > {MAX_TOTAL_STATE_FRAMES}"
                    );
                }
                neural_amp_modeler_rs::loader::nam_json::WavenetTopologyResult::Rejected(_) => {
                    // Rejection is acceptable — some combos exceed budget
                }
            }
        }
    }
}
