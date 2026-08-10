// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Off-RT diagnostic dump harness for intermediate model state.
//!
//! ## Invariant
//! All capture buffers and hooks are gated by `#[cfg(any(test, feature = "testing"))]`.
//! Release builds carry zero dump symbols; the hot path performs no allocation or I/O.
//!
//! ## Usage
//! ```ignore
//! model.enable_diagnostics(DiagnosticConfig { capture_condition_dsp: true, .. });
//! model.process(&input, &mut output);
//! let dump = model.take_diagnostics().unwrap();
//! ```

use crate::math::common::AlignedVec;

/// Controls which intermediate states are captured during `process()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiagnosticConfig {
    /// Capture condition_dsp output after every sub-block (8 ch × N frames).
    pub capture_condition_dsp: bool,
    /// Capture head_accum snapshot after every layer completes its frames.
    pub capture_head_per_layer: bool,
    /// Capture final output after head convolution.
    pub capture_final_output: bool,
}

/// Snapshot of `condition_dsp_output` after one sub-block.
#[derive(Debug, Clone)]
pub struct ConditionDspSnapshot {
    /// Number of channels (= condition_size, 8 for A2 Max).
    pub channels: usize,
    /// Number of frames in this sub-block (≤ WAVENET_MAX_NUM_FRAMES).
    pub num_frames: usize,
    /// Interleaved 2D buffer: `[ch0_f0, ch1_f0, ..., chN_f0, ch0_f1, ...]`.
    pub data: AlignedVec<f32>,
}

/// Snapshot of `head_accum` state immediately after a layer completes
/// processing for a sub-block.
#[derive(Debug, Clone)]
pub struct HeadPerLayerSnapshot {
    /// Layer index (0-based).
    pub layer: usize,
    /// Head accumulator size (channels for head1x1-active path).
    pub accum_size: usize,
    /// Number of frames written in this sub-block.
    pub num_frames: usize,
    /// Head write position *before* this layer started.
    pub head_wp: usize,
    /// Head accumulator ring buffer content covering the written region.
    /// Layout: `[frame_0_chan_0, ..., frame_0_chan_N, frame_1_chan_0, ...]`.
    pub data: AlignedVec<f32>,
}

/// Full diagnostic dump collected during a single `process()` call.
#[derive(Debug, Clone)]
pub struct DiagnosticDump {
    /// Total number of input frames processed.
    pub total_frames: usize,
    /// Condition DSP output snapshots, one per sub-block.
    pub condition_dsp_snapshots: Vec<ConditionDspSnapshot>,
    /// Head-per-layer snapshots, one per (layer × sub-block).
    pub head_per_layer_snapshots: Vec<HeadPerLayerSnapshot>,
    /// Final output buffer as produced by the model.
    /// This matches the `output` slice passed to `process()`.
    pub final_output: Option<AlignedVec<f32>>,
}

impl DiagnosticDump {
    /// Creates an empty dump for the given total frame count.
    pub fn new(total_frames: usize) -> Self {
        Self {
            total_frames,
            condition_dsp_snapshots: Vec::new(),
            head_per_layer_snapshots: Vec::new(),
            final_output: None,
        }
    }

    /// Returns a bit-stable f32 hash of the full dump for deterministic comparison.
    ///
    /// Sums the raw f32 bits (as u32) of all captured data for a bit-exact
    /// identity check — two runs with identical input produce identical hashes.
    pub fn bit_stable_hash(&self) -> u64 {
        let mut h: u64 = self.total_frames as u64;
        for snap in &self.condition_dsp_snapshots {
            h = h.wrapping_add(snap.channels as u64);
            h = h.wrapping_add(snap.num_frames as u64);
            for &v in snap.data.iter() {
                h = h.wrapping_add(v.to_bits() as u64);
            }
        }
        for snap in &self.head_per_layer_snapshots {
            h = h.wrapping_add(snap.layer as u64);
            h = h.wrapping_add(snap.accum_size as u64);
            h = h.wrapping_add(snap.num_frames as u64);
            for &v in snap.data.iter() {
                h = h.wrapping_add(v.to_bits() as u64);
            }
        }
        if let Some(ref out) = self.final_output {
            for &v in out.iter() {
                h = h.wrapping_add(v.to_bits() as u64);
            }
        }
        h
    }
}
