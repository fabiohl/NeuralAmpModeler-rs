// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::common::diagnostics::SystemSnapshot;
    use crate::loader::{
        LoadError, LoadOptions, MetadataError, load_and_build_model,
        load_and_build_model_from_bytes, load_and_build_model_from_bytes_named,
    };
    use crate::models::NamModel;
    use crate::testing::fixtures::model_path;
    use std::path::Path;
    use std::path::PathBuf;

    /// Writes a hostile-metadata variant of `wavenet.nam` to a unique temp file
    /// and returns its path. `tag` must be unique per test (parallel-safe);
    /// `replace` maps the original JSON snippet to the hostile one
    /// (e.g. `"input_level_dbu": 18.3` → `"input_level_dbu": 1e999`).
    fn write_hostile_metadata_variant(tag: &str, replace: (&str, &str)) -> PathBuf {
        let src = model_path("wavenet.nam");
        let content = std::fs::read_to_string(&src).expect("fixture must be readable");
        assert!(
            content.contains(replace.0),
            "fixture does not contain expected snippet: {}",
            replace.0
        );
        let hostile = content.replace(replace.0, replace.1);

        let path = std::env::temp_dir().join(format!(
            "nam_metadata_hostile_{}_{}.nam",
            std::process::id(),
            tag
        ));
        std::fs::write(&path, hostile).expect("temp file must be writable");
        path
    }

    /// Acceptance: JSON with `input_level_dbu: 1e999` returns a typed `Err`.
    ///
    /// serde_json rejects `1e999` as "number out of range" at parse time
    /// (`JsonError::Serde`). Defense-in-depth: if a future serde version
    /// saturates instead, the post-parse gate (`MetadataError`) also rejects.
    #[test]
    fn test_metadata_input_level_1e999_returns_typed_err() {
        let sys = SystemSnapshot::capture();
        let path = write_hostile_metadata_variant(
            "in_1e999",
            ("\"input_level_dbu\": 18.3", "\"input_level_dbu\": 1e999"),
        );
        let res = load_and_build_model(&path, &sys, false, LoadOptions::default());
        std::fs::remove_file(&path).ok();
        assert!(
            res.is_err(),
            "metadata with input_level_dbu=1e999 must be rejected"
        );
    }

    /// The real saturation vector (F-14): `1e39` is finite in f64 but saturates
    /// to `+Inf` when serde deserializes into `f32`. The post-parse gate must
    /// reject it with the typed `MetadataError::NonFinite`.
    #[test]
    fn test_metadata_input_level_f32_saturation_rejected_typed() {
        let sys = SystemSnapshot::capture();
        let path = write_hostile_metadata_variant(
            "in_1e39",
            ("\"input_level_dbu\": 18.3", "\"input_level_dbu\": 1e39"),
        );
        let res = load_and_build_model(&path, &sys, false, LoadOptions::default());
        std::fs::remove_file(&path).ok();

        let err = res.expect_err("metadata with input_level_dbu=1e39 must be rejected");
        match err {
            LoadError::InvalidMetadata(meta_err) => {
                assert!(
                    matches!(
                        meta_err,
                        MetadataError::NonFinite {
                            field: "input_level_dbu",
                            ..
                        }
                    ),
                    "expected MetadataError::NonFinite for input_level_dbu, got: {:?}",
                    meta_err
                );
            }
            other => panic!("expected LoadError::InvalidMetadata, got: {:?}", other),
        }
    }

    /// Out-of-range dBu metadata (beyond ±60 dBu) is rejected with the typed
    /// `MetadataError::DbOutOfRange`.
    #[test]
    fn test_metadata_db_out_of_range_rejected_typed() {
        let sys = SystemSnapshot::capture();
        let path = write_hostile_metadata_variant(
            "loudness_5000",
            ("\"loudness\": -20.020729064941406", "\"loudness\": -5000.0"),
        );
        let res = load_and_build_model(&path, &sys, false, LoadOptions::default());
        std::fs::remove_file(&path).ok();

        let err = res.expect_err("loudness beyond ±60 dBu must be rejected");
        match err {
            LoadError::InvalidMetadata(meta_err) => {
                assert!(
                    matches!(
                        meta_err,
                        MetadataError::DbOutOfRange {
                            field: "loudness",
                            ..
                        }
                    ),
                    "expected MetadataError::DbOutOfRange for loudness, got: {:?}",
                    meta_err
                );
            }
            other => panic!("expected LoadError::InvalidMetadata, got: {:?}", other),
        }
    }

    /// Hostile `head_scale` (negative or beyond the plausible linear range)
    /// is rejected with the typed `MetadataError::HeadScaleOutOfRange`.
    #[test]
    fn test_metadata_head_scale_out_of_range_rejected_typed() {
        let sys = SystemSnapshot::capture();
        let path = write_hostile_metadata_variant(
            "head_scale_neg",
            ("\"head_scale\": 0.02", "\"head_scale\": -0.02"),
        );
        let res = load_and_build_model(&path, &sys, false, LoadOptions::default());
        std::fs::remove_file(&path).ok();

        let err = res.expect_err("head_scale=-0.02 must be rejected");
        match err {
            LoadError::InvalidMetadata(meta_err) => {
                assert!(
                    matches!(meta_err, MetadataError::HeadScaleOutOfRange { .. }),
                    "expected MetadataError::HeadScaleOutOfRange, got: {:?}",
                    meta_err
                );
            }
            other => panic!("expected LoadError::InvalidMetadata, got: {:?}", other),
        }
    }

    #[test]
    fn test_load_valid_model_mono() {
        let sys = SystemSnapshot::capture();
        let path = model_path("wavenet.nam");
        let pair = load_and_build_model(&path, &sys, false, LoadOptions::default())
            .expect("Valid model should load successfully");
        assert!(
            pair.model_l.is_some(),
            "model_l must be Some for valid mono load"
        );
        assert!(pair.model_r.is_none(), "model_r must be None for mono load");
        assert!(pair.sample_rate > 0);
    }

    #[test]
    fn test_load_valid_model_stereo() {
        let sys = SystemSnapshot::capture();
        let path = model_path("wavenet.nam");
        let pair = load_and_build_model(&path, &sys, true, LoadOptions::default())
            .expect("Valid model should load successfully in stereo mode");
        assert!(
            pair.model_l.is_some(),
            "model_l must be Some for stereo load"
        );
        #[cfg(feature = "dual-mono")]
        assert!(
            pair.model_r.is_some(),
            "model_r must be Some for stereo load when dual-mono is enabled"
        );
        #[cfg(not(feature = "dual-mono"))]
        assert!(
            pair.model_r.is_none(),
            "model_r must be None for stereo load when dual-mono is disabled"
        );
    }

    #[test]
    fn test_load_invalid_model_mock_a2_returns_err() {
        let sys = SystemSnapshot::capture();
        let path = model_path("mock_a2.nam");
        let res = load_and_build_model(&path, &sys, false, LoadOptions::default());
        assert!(
            res.is_err(),
            "mock_a2.nam must fail build and return Err, never Ok with empty channels"
        );
    }

    #[test]
    fn test_load_truncated_json_returns_err() {
        let sys = SystemSnapshot::capture();
        let path = model_path("keras_unsupported.json");
        let res = load_and_build_model(&path, &sys, false, LoadOptions::default());
        assert!(
            res.is_err(),
            "Invalid/unsupported JSON model must return Err"
        );
    }

    #[test]
    fn test_load_nonexistent_file_returns_err() {
        let sys = SystemSnapshot::capture();
        let path = Path::new("non_existent_model_path_12345.nam");
        let res = load_and_build_model(path, &sys, false, LoadOptions::default());
        assert!(res.is_err(), "Nonexistent model path must return Err");
    }

    #[test]
    fn test_build_model_fail_fast_on_valid_system() {
        // On a valid x86-64-v3 host (where avx2 & fma are supported),
        // dispatcher::build_model must succeed in feature check and construct the model.
        let path = model_path("wavenet.nam");
        let content = std::fs::read_to_string(&path).expect("fixture must exist");
        let data = crate::loader::nam_json::parse_nam_json(&content).expect("JSON must parse");
        let model = crate::loader::dispatcher::build_model(&data);
        assert!(model.is_ok(), "build_model must succeed on supported CPU");
    }

    #[test]
    fn test_load_and_build_model_from_bytes_parity() {
        let sys = SystemSnapshot::capture();
        let path = model_path("wavenet.nam");
        let bytes = std::fs::read(&path).expect("fixture must exist");

        let file_pair = load_and_build_model(&path, &sys, false, LoadOptions::default())
            .expect("file load must succeed");
        let bytes_pair =
            load_and_build_model_from_bytes(&bytes, &sys, false, LoadOptions::default())
                .expect("bytes load must succeed");

        assert_eq!(file_pair.architecture, bytes_pair.architecture);
        assert_eq!(file_pair.topology, bytes_pair.topology);
        assert_eq!(file_pair.sample_rate, bytes_pair.sample_rate);
        assert_eq!(file_pair.weights_layout, bytes_pair.weights_layout);
        assert_eq!(file_pair.input_mult_adj, bytes_pair.input_mult_adj);
        assert_eq!(file_pair.output_mult_adj, bytes_pair.output_mult_adj);

        // Run audio block inference through both models and assert bit-exact equality.
        let mut model_file = file_pair.model_l.expect("model_l must exist");
        let mut model_bytes = bytes_pair.model_l.expect("model_l must exist");

        let block_size = 256;
        let mut input = vec![0.0f32; block_size];
        for (i, sample) in input.iter_mut().enumerate() {
            *sample = (i as f32 * 0.05).sin() * 0.5;
        }

        let mut out_file = vec![0.0f32; block_size];
        let mut out_bytes = vec![0.0f32; block_size];

        model_file.process(&input, &mut out_file);
        model_bytes.process(&input, &mut out_bytes);

        for (i, (&f_samp, &b_samp)) in out_file.iter().zip(out_bytes.iter()).enumerate() {
            assert_eq!(
                f_samp.to_bits(),
                b_samp.to_bits(),
                "bit-exact parity mismatch at index {}: file={}, bytes={}",
                i,
                f_samp,
                b_samp
            );
        }
    }

    #[test]
    fn test_load_and_build_model_from_bytes_namb_format() {
        let sys = SystemSnapshot::capture();
        let path = model_path("wavenet.nam");
        let content = std::fs::read_to_string(&path).expect("fixture must exist");
        let data = crate::loader::nam_json::parse_nam_json(&content).expect("JSON must parse");
        let namb_bytes = crate::loader::namb_encoder::encode_namb(
            &data,
            2,
            crate::loader::nam_json::WeightsLayout::Original,
        )
        .expect("namb encode must succeed");

        let pair = load_and_build_model_from_bytes_named(
            &namb_bytes,
            "test_embedded.namb",
            &sys,
            false,
            LoadOptions::default(),
        )
        .expect("namb bytes load must succeed");

        assert!(pair.model_l.is_some());
        assert_eq!(pair.architecture, "WaveNet");
    }

    #[test]
    fn test_load_and_build_model_from_bytes_empty_and_garbage() {
        let sys = SystemSnapshot::capture();
        let empty: [u8; 0] = [];
        let res_empty =
            load_and_build_model_from_bytes(&empty, &sys, false, LoadOptions::default());
        assert!(res_empty.is_err(), "empty bytes must fail load");

        let garbage = b"not a valid nam json or namb binary payload";
        let res_garbage =
            load_and_build_model_from_bytes(garbage, &sys, false, LoadOptions::default());
        assert!(res_garbage.is_err(), "garbage bytes must fail load");
    }
}
