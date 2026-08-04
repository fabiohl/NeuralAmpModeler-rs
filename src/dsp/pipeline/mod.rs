// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP processing pipeline (Capture → DSP → Bridge).
//!
//! This module isolates the audio processing logic from host orchestration.
//! It contains the hot-path executed every audio cycle on the real-time thread.

#[cfg(test)]
use crate::models::NamModel;

mod bridge;
mod capture;
mod context;

mod stages;

// Re-exports — preserve the same visibility as the original pipeline.rs.

pub use bridge::{BridgeBuffer, BridgeRef, DspBridge, DspBridgeReader, DspBridgeWriter};
pub use bridge::{MAX_BRIDGE_BUF, MAX_RESAMP_BUF};

pub use context::{DspBuffers, DspPipelineContext};

pub use stages::DENORMAL_DITHER_OFFSET;
#[cfg(feature = "testing")]
pub use stages::DISABLE_GATE;
pub use stages::apply_input_stage;
pub use stages::apply_output_stage;
pub use stages::handle_silence_bypass;
pub use stages::run_inference;
pub use stages::write_bridge;

pub use capture::capture_dsp_pipeline;

#[cfg(any(test, feature = "testing"))]
/// Test utilities for pipeline benchmarks and tests, exposed for plugin integration testing.
pub mod test_util {
    /// Shared test infrastructure (allocators, guards) for pipeline tests.
    pub mod infra {
        #[cfg(test)]
        pub use crate::common::alloc_audit::CountingAllocator;
        pub use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};
    }
}

#[cfg(test)]
#[global_allocator]
static GLOBAL: test_util::infra::CountingAllocator = test_util::infra::CountingAllocator;

#[cfg(test)]
mod pipeline_test;

#[cfg(test)]
mod pipeline_block_test;
