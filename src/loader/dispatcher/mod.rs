// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Model Dispatcher — converts `NamModelData` (JSON/NAMB parsers) into `Box<StaticModel>`.
//!
//! The dispatcher reads the architecture signature from the model data and routes
//! to the appropriate variant-specific builder (WaveNet or LSTM),
//! ensuring the resulting `Box<StaticModel>` is ready for injection into the
//! DSP thread via SPSC without any allocation on the RT path.

use crate::common::diagnostics::NamErrorCode;
use crate::loader::nam_json::NamModelData;
use crate::models::StaticModel;
use anyhow::bail;

// =============================================================================
// Public Entry Point
// =============================================================================

/// Builds a `Box<StaticModel>` from the raw parsed data.
///
/// Branches by architecture (`"WaveNet"` / `"LSTM"` / `"SlimmableContainer"`) and delegates to the
/// specialized builders with const generics.
pub fn build_model(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            log::error!(
                "CPU incompatible: x86-64-v3 baseline required (AVX2 + FMA). \
                 Execution aborted cleanly to prevent SIGILL."
            );
            return Err(anyhow::anyhow!(NamErrorCode::UnsupportedCpuArchitecture));
        }
    }

    match data.architecture.as_str() {
        "WaveNet" => wavenet::build_wavenet(data),
        "LSTM" => lstm::build_lstm(data),
        "SlimmableContainer" => container::build_container(data),
        "Linear" => linear::build_linear(data),
        "ConvNet" => convnet::build_convnet(data),
        other => bail!("Unsupported architecture: '{}'", other),
    }
}

/// Checked arithmetic helpers (F2 — DoS/OOM prevention)
pub(crate) mod checked_arith;
/// SlimmableContainer model builder module
pub mod container;
/// ConvNet model builder module
pub mod convnet;
/// Linear model builder module
pub mod linear;
/// LSTM model builder module
pub mod lstm;
/// WaveNet model builder module
pub mod wavenet;
/// WeightCursor — Deterministic sequential reading of flattened weights
pub mod weight_cursor;

pub(crate) use weight_cursor::WeightCursor;
