// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use std::path::PathBuf;

#[test]
fn test_params_default() {
    let params = ProcessingParams::default();
    assert_eq!(params.input_gain_db, 0.0);
    assert_eq!(params.output_gain_db, 0.0);
    assert_eq!(params.gate_threshold_db, -70.0);
    assert_eq!(params.model_path, None);
    assert_eq!(params.model_basename, None);
    assert_eq!(params.model_hash, None);
    assert!(params.model_search_paths.is_empty());
    assert!(!params.bypass);
    assert_eq!(params.ir_path, None);
}

#[test]
fn test_params_builder() {
    let params = ProcessingParams::builder()
        .with_input_gain_db(3.5)
        .with_output_gain_db(-2.0)
        .with_gate_threshold_db(-65.0)
        .with_model_path(PathBuf::from("/models/amp.nam"))
        .with_model_basename("amp.nam")
        .with_model_hash("abcd1234efgh5678")
        .with_model_search_path(PathBuf::from("/custom/models"))
        .with_bypass(true)
        .with_adaptive_compute(AdaptiveComputeMode::Conservative)
        .with_slim_override(SlimOverride::ForceFull)
        .with_oversample(OversampleFactor::X2)
        .with_ir_path(PathBuf::from("/irs/cab.wav"))
        .with_ir_hash("feedbeefcafe")
        .with_activation_precision(ActivationPrecision::Fast);

    assert_eq!(params.input_gain_db, 3.5);
    assert_eq!(params.output_gain_db, -2.0);
    assert_eq!(params.gate_threshold_db, -65.0);
    assert_eq!(params.model_path, Some(PathBuf::from("/models/amp.nam")));
    assert_eq!(params.model_basename.as_deref(), Some("amp.nam"));
    assert_eq!(params.model_hash.as_deref(), Some("abcd1234efgh5678"));
    assert_eq!(
        params.model_search_paths,
        vec![PathBuf::from("/custom/models")]
    );
    assert!(params.bypass);
    assert_eq!(params.adaptive_compute, AdaptiveComputeMode::Conservative);
    assert_eq!(params.slim_override, SlimOverride::ForceFull);
    assert_eq!(params.oversample, OversampleFactor::X2);
    assert_eq!(params.ir_path, Some(PathBuf::from("/irs/cab.wav")));
    assert_eq!(params.ir_hash.as_deref(), Some("feedbeefcafe"));
    assert_eq!(params.activation_precision, ActivationPrecision::Fast);
}

#[test]
fn test_rt_params_builder() {
    let params = RtProcessingParams::builder()
        .with_input_gain_db(3.5)
        .with_output_gain_db(-2.0)
        .with_gate_threshold_db(-65.0)
        .with_bypass(true)
        .with_adaptive_compute(AdaptiveComputeMode::Conservative)
        .with_slim_override(SlimOverride::ForceFull)
        .with_oversample(OversampleFactor::X2)
        .with_activation_precision(ActivationPrecision::Fast);

    assert_eq!(params.input_gain_db, 3.5);
    assert_eq!(params.output_gain_db, -2.0);
    assert_eq!(params.gate_threshold_db, -65.0);
    assert!(params.bypass);
    assert_eq!(params.adaptive_compute, AdaptiveComputeMode::Conservative);
    assert_eq!(params.slim_override, SlimOverride::ForceFull);
    assert_eq!(params.oversample, OversampleFactor::X2);
    assert_eq!(params.activation_precision, ActivationPrecision::Fast);
    assert_eq!(RtProcessingParams::new(), RtProcessingParams::default());
}
