// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn output_mode_conversions_roundtrip() {
    for mode in [
        OutputMode::Raw,
        OutputMode::Normalized,
        OutputMode::Calibrated,
    ] {
        assert_eq!(OutputMode::from_u32(mode.to_u32()), mode);
        assert_eq!(OutputMode::from_f32(mode.to_f32()), mode);
        assert_eq!(OutputMode::from_text(mode.as_str()), Some(mode));
    }
    assert_eq!(OutputMode::from_f32(f32::NAN), OutputMode::Normalized);
    assert_eq!(OutputMode::from_u32(99), OutputMode::Normalized);
    assert_eq!(OutputMode::from_text("invalid"), None);
}

#[test]
fn sanitize_calibration_level_clamps_and_handles_non_finite() {
    assert_eq!(
        sanitize_calibration_level_dbu(f32::NAN),
        DEFAULT_CALIBRATION_DBU
    );
    assert_eq!(
        sanitize_calibration_level_dbu(f32::INFINITY),
        DEFAULT_CALIBRATION_DBU
    );
    assert_eq!(sanitize_calibration_level_dbu(-120.0), MIN_CALIBRATION_DBU);
    assert_eq!(sanitize_calibration_level_dbu(120.0), MAX_CALIBRATION_DBU);
    assert_eq!(sanitize_calibration_level_dbu(18.0), 18.0);
}

#[test]
fn compute_calibration_gains_matches_expected_db_math() {
    let meta = ModelCalibrationMeta {
        has_model: true,
        input_level_dbu: Some(6.0),
        output_level_dbu: Some(0.0),
        loudness: Some(-24.0),
    };

    let norm = compute_calibration_gains(meta, true, 12.0, OutputMode::Normalized);
    let expected_in = 10.0_f64.powf(6.0 / 20.0) as f32;
    let expected_out_norm = 10.0_f64.powf(6.0 / 20.0) as f32;
    assert!((norm.input_mult - expected_in).abs() < 1e-6);
    assert!((norm.output_mult - expected_out_norm).abs() < 1e-6);

    let cal = compute_calibration_gains(meta, false, 12.0, OutputMode::Calibrated);
    let expected_out_cal = 10.0_f64.powf(-12.0 / 20.0) as f32;
    assert_eq!(cal.input_mult, 1.0);
    assert!((cal.output_mult - expected_out_cal).abs() < 1e-6);

    let raw = compute_calibration_gains(meta, false, 12.0, OutputMode::Raw);
    assert_eq!(raw, CalibrationGains::default());

    let empty = compute_calibration_gains(
        ModelCalibrationMeta::empty(),
        true,
        12.0,
        OutputMode::Calibrated,
    );
    assert_eq!(empty, CalibrationGains::default());
}

#[test]
fn loudness_outside_sane_bounds_is_rejected() {
    use crate::loader::LoadedModelPair;
    use crate::loader::nam_json::NamMetadata;

    let make_pair = |loudness: f32| LoadedModelPair {
        model_l: None,
        model_r: None,
        input_mult_adj: 1.0,
        output_mult_adj: 1.0,
        sample_rate: 48_000,
        architecture: "Linear".to_string(),
        topology: "RF1".to_string(),
        metadata: Some(NamMetadata {
            loudness: Some(loudness),
            ..Default::default()
        }),
        weights_layout: "Original".to_string(),
    };

    for extreme in [-300.0, -120.1, 60.1, 200.0, f32::NAN, f32::INFINITY] {
        let meta = ModelCalibrationMeta::from_loaded_model(&make_pair(extreme));
        assert_eq!(
            meta.loudness, None,
            "extreme loudness {extreme} must be rejected by from_loaded_model"
        );

        let direct_meta = ModelCalibrationMeta {
            has_model: true,
            input_level_dbu: None,
            output_level_dbu: None,
            loudness: Some(extreme),
        };
        let gains = compute_calibration_gains(direct_meta, false, 12.0, OutputMode::Normalized);
        assert_eq!(
            gains.output_mult, 1.0,
            "extreme loudness {extreme} must yield unity output_mult"
        );
    }

    for valid in [MIN_LOUDNESS_DB, -18.0, 0.0, MAX_LOUDNESS_DB] {
        let meta = ModelCalibrationMeta::from_loaded_model(&make_pair(valid));
        assert_eq!(meta.loudness, Some(valid));
    }
}
