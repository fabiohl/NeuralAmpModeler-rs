// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared helper module for NeuralAmpModeler-rs integration tests.
//!
//! Centralizes signal generation functions, error metrics, and DSP validation
//! to avoid duplication across integration test files.

#![allow(dead_code)]
#![allow(unused_imports)]

pub mod alloc_audit;
pub mod bin_fixtures;
pub mod constants;
pub mod conv_helpers;
pub mod discovery;
pub mod io_helpers;
pub mod manifest;
pub mod metrics;
pub mod model_builders;
pub mod mushra_primitives;
pub mod perceptual;
pub mod precision;
pub mod rt_helpers;
pub mod validation;
pub mod wav;

pub use constants::*;
pub use discovery::*;
pub use io_helpers::*;
pub use manifest::ManifestEntry;
pub use metrics::*;
pub use neural_amp_modeler_rs::testing::aliasing::generate_sine_440hz;
pub use neural_amp_modeler_rs::testing::stress::SUPPORTED_SAMPLE_RATES;
pub use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v1;
pub use neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2_default as generate_stress_signal_v2;
pub use precision::PrecisionGuard;
pub use validation::*;

// Re-export the unified fixture catalog for fail-closed fixture availability.
pub use io_helpers::{
    ApplicableOracle, ExecutionProfile, FIXTURE_CATALOG, FixtureCatalog, FixtureEntry,
    FixtureOrigin, FixtureStatus,
};
