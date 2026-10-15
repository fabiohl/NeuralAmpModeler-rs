// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::prelude::*;

#[test]
fn test_prelude_exports_cabsim_and_all_core_types() {
    let _: Option<ConvEngine> = None;
    let _: Option<CabSimAdapter> = None;
    let _: Option<CabSimIr> = None;
    let _: Option<GateParams> = None;
    let _: Option<GateParamsBuilder> = None;
    let _: Option<SystemSnapshot> = None;
    let _: Option<NamErrorCode> = None;
    let _: Option<ActivationPrecision> = None;
    let _: Option<AdaptiveComputeMode> = None;
    let _: Option<SlimOverride> = None;
    let _: Option<ProcessingParams> = None;
    let _: Option<RtProcessingParams> = None;
    let _: Option<LoadError> = None;
    let _: Option<NambError> = None;
    let _: Option<JsonError> = None;
    let _: Option<LoadedModelPair> = None;
    let _: Option<LoadOptions> = None;
    let _: Option<OversampleEngine> = None;
    let _: Option<OversampleFactor> = None;
    let _: Option<NamResampler> = None;
    let _: Option<DelayLine<f32>> = None;
    let _: Option<Box<dyn NamModel>> = None;
    let _: Option<StaticModel> = None;
    let _ = load_and_build_model;
}
