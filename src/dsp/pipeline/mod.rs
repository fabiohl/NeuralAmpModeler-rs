// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP processing pipeline (Capture → DSP → Bridge).
//!
//! This module isolates the audio processing logic from host orchestration.
//! It contains the hot-path executed every audio cycle on the real-time thread.
//!
//! # Construction policy
//!
//! [`DspBuffers`](crate::dsp::pipeline::DspBuffers) and
//! [`DspPipelineContext`](crate::dsp::pipeline::DspPipelineContext) are **not**
//! marked `#[non_exhaustive]` (that would break existing host literals); field
//! additions remain SemVer-breaking for literal construction. Prefer
//! `DspBuffers::from_parts` / `DspBuffers::new` and
//! `DspPipelineContext::from_parts`, which survive future field additions.
//! New public types in this module must be born with a constructor.

#[cfg(test)]
use crate::models::NamModel;

mod bridge;
mod capture;
mod context;

mod stages;

// Re-exports — preserve the same visibility as the original pipeline.rs.

/// Bridge buffer structs and reader/writer interfaces for inter-stage data transfer.
pub use bridge::{BridgeBuffer, BridgeRef, DspBridge, DspBridgeReader, DspBridgeWriter};
/// Maximum bridge and resampler buffer capacities (in samples).
pub use bridge::{MAX_BRIDGE_BUF, MAX_RESAMP_BUF};

/// DSP buffer types and pipeline context for per-call stack allocation.
pub use context::{DspBuffers, DspPipelineContext};

/// Denormal dither offset value for FTZ/DAZ protection on the hot path.
pub use stages::DENORMAL_DITHER_OFFSET;
#[cfg(feature = "testing")]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
pub use stages::DISABLE_GATE;
/// Input stage: gate processing, denormal dither, and silence bypass detection.
pub use stages::apply_input_stage;
/// Output stage: final volume adjustment, clipping detection, and post-DSP smoothing.
pub use stages::apply_output_stage;
/// Silence bypass: zero-fills output buffer when gate is fully closed and signal is silent.
pub use stages::handle_silence_bypass;
/// Model inference dispatcher: routes to the correct architecture-specific process method.
pub use stages::run_inference;
/// Streaming inference pass with strict host cardinality (F-PERF-002).
pub use stages::run_inference_streaming;
/// Bridge write-out: copies processed output into the inter-stage bridge buffer.
pub use stages::write_bridge;

/// Full DSP pipeline capture: input → processing chain → bridge in a single pass.
pub use capture::capture_dsp_pipeline;
/// Full DSP pipeline capture with streaming resampler adapter (strict host cardinality).
pub use capture::capture_dsp_pipeline_streaming;

#[cfg(any(test, feature = "testing"))]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
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
