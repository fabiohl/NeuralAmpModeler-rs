// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::common::diagnostics::SystemSnapshot;
    use crate::loader::{LoadOptions, load_and_build_model};
    use crate::testing::fixtures::model_path;
    use std::path::Path;

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
        assert!(
            pair.model_r.is_some(),
            "model_r must be Some for stereo load"
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
}
