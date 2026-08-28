// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Bounded, host-agnostic streaming resample buffers with strict cardinality.
//!
//! `StreamingResampleBuffer` couples a host sample rate to a model sample
//! rate through caller-supplied model processing, guaranteeing exactly-N
//! host output extraction with deterministic FIFO capacities and zero heap
//! allocation on the real-time processing path.

mod fifo;
mod streaming_adapter;

pub use fifo::SampleFifo;
pub use streaming_adapter::{
    MAX_STREAM_BLOCK, PullResult, StreamingResampleBuffer, StreamingResult,
};
