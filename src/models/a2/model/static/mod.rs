// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 static model (`WaveNetA2<const CH: usize>`).
//!
//! Single layer-array of 23 dilated causal layers with skip-connection accumulator
//! and head rechannel convolution, matching the fast-path from `a2_fast.cpp`.
//!
//! ## Architecture
//!
//! 1. Input rechannel: `Conv1x1(1 → CH)` (bias, no activation)
//! 2. 23 layers: dilated conv → input-mixin → LeakyReLU → head_accum += out → layer1x1 → residual
//! 3. Head conv: `Conv1D(CH → 1, K=16, bias)` over head_accum ring → × head_scale
//!
//! Processing is chunked by `WAVENET_MAX_NUM_FRAMES` (64) with zero allocation on the hot-path.
//!
//! ## Ring buffer architecture
//!
//! Each layer's history is a power-of-2 `MirroredBuffer<f32>` that provides
//! branchless reads via virtual-memory mirroring. The `buffer_start` pointer
//! advances through the 2× virtual mapping; when it approaches the 2× boundary,
//! it rewinds by subtracting `ring_size`. Reads at `buffer_start - offset` are
//! always valid because the mirrored mapping maps `[S, 2S)` → `[0, S)`.
//!
//! The head accumulator uses a plain `AlignedVec` with pow2 mask (`& ring_mask`)
//! for branchless ring access — no MirroredBuffer needed since the head reads
//! are already mask-based.
//!
//! ## Source of truth
//! - `a2_fast.cpp`: class `A2FastModel` (members, process, prewarm, reset)
//! - `detail.h`: `LayerArray::Process` (per-layer sequence)

use super::super::head::A2HeadConv;
use super::super::layer::A2Layer;
use super::super::params::{A2_DILATIONS, A2_KERNEL_SIZES, A2_NUM_LAYERS};
use super::a2_receptive_field;
use crate::dsp::mirror_buf::MirroredBuffer;
use crate::math::common::AlignedVec;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;
use serde_json::Value;
use std::sync::Arc;

pub mod prewarm;
pub mod process;

/// Complete WaveNet A2 Model.
///
/// `CH` = channel count (3 for Lite/Nano, 8 for Full/Standard).
pub struct WaveNetA2<const CH: usize> {
    /// 23 A2 layers (one per layer index). Populated by `set_weights`.
    pub layers: Vec<A2Layer>,

    /// Input rechannel weights: `Conv1x1(1 → CH)` (no bias).
    pub rechannel_w_f32: AlignedVec<f32>,

    /// Head convolution (K=16 over skip-connection accumulator, bias, head_scale).
    pub head_conv: Option<A2HeadConv>,

    /// Head accumulator ring buffer (skip-connection sum, column-major, pow2 size).
    pub head_accum: AlignedVec<f32>,

    /// Write position in `head_accum` (in columns, wraps via `head_ring_mask`).
    pub head_write_pos: usize,

    /// Ring mask for `head_accum` (pow2 ring, mask = capacity - 1).
    pub head_ring_mask: usize,

    /// Per-layer history buffers: one MirroredBuffer per layer (23 total).
    /// Each buffer provides 2× virtual mapping for branchless ring access.
    pub layer_buffers: Vec<MirroredBuffer<f32>>,

    /// Per-layer ring sizes in elements (pow2 page-aligned). For rewind: `start -= ring_size`.
    pub layer_ring_sizes: Vec<usize>,

    /// Per-layer maximum dilation lookback = (kernel-1) * dilation.
    pub layer_lookbacks: Vec<usize>,

    /// Per-layer buffer starts (advanced with each written frame, rewound near 2× boundary).
    pub layer_buffer_starts: Vec<usize>,

    /// Inter-layer data buffer: `CH × max_buffer_size` f32, reused across layers.
    /// Each layer reads from it, then writes its l1x1 residual back (in-place update).
    pub layer_in: AlignedVec<f32>,

    /// Total receptive field: sum of `(kernel-1)*dilation` + head kernel - 1.
    pub receptive_field_size: usize,

    /// Maximum frames per processing block (= `WAVENET_MAX_NUM_FRAMES`).
    pub max_buffer_size: usize,

    /// Raw JSON for the single layer array, preserved for FiLM config parsing.
    /// `None` when the model is constructed directly (not via JSON deserialization).
    pub layer_raw: Option<Value>,

    /// Per-frame scratch buffer for the fallback conv path (CH f32).
    /// Allocated once at model creation to avoid heap ops on the hot path.
    pub z_scratch: AlignedVec<f32>,

    /// `RtStatusFlags` for silent RT→Main telemetry (RF8).
    pub rt_status: Option<Arc<crate::common::spsc::RtStatusFlags>>,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,
}

impl<const CH: usize> WaveNetA2<CH> {
    /// Creates a new uninitialized WaveNet A2 model.
    ///
    /// Allocates ring buffers (MirroredBuffer per layer, pow2 head accumulator)
    /// sized for the architecture and computes the receptive field.
    /// Weight-bearing fields start empty and are populated by the weight loader.
    pub fn new() -> anyhow::Result<Self> {
        let rf = a2_receptive_field();
        let max_buf = WAVENET_MAX_NUM_FRAMES;

        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        let head_ring_mask = head_ring_size - 1;

        let mut layer_buffers = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_ring_sizes = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_lookbacks = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_buffer_starts = Vec::with_capacity(A2_NUM_LAYERS);

        for i in 0..A2_NUM_LAYERS {
            let max_lookback = (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            let cap = max_lookback + max_buf + 1;
            let mb = MirroredBuffer::<f32>::new(cap * CH)?;
            let ring_size = mb.size();
            layer_buffers.push(mb);
            layer_ring_sizes.push(ring_size);
            layer_lookbacks.push(max_lookback * CH);
            layer_buffer_starts.push(ring_size);
        }

        Ok(Self {
            layers: Vec::with_capacity(A2_NUM_LAYERS),
            rechannel_w_f32: AlignedVec::new(CH, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            head_conv: None,
            head_accum: AlignedVec::new(head_ring_size * CH, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            head_write_pos: rf,
            head_ring_mask,
            layer_buffers,
            layer_ring_sizes,
            layer_lookbacks,
            layer_buffer_starts,
            layer_in: AlignedVec::new(CH * max_buf, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            receptive_field_size: rf,
            max_buffer_size: max_buf,
            layer_raw: None,
            z_scratch: AlignedVec::new(CH, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            rt_status: None,
            prewarm_on_reset: true,
        })
    }

    /// Returns the channel count.
    #[inline(always)]
    pub fn channels(&self) -> usize {
        CH
    }

    /// Injects `RtStatusFlags` for RT→Main telemetry (RF8).
    pub fn inject_rt_status(&mut self, rt_status: Arc<crate::common::spsc::RtStatusFlags>) {
        self.rt_status = Some(rt_status);
    }

    /// Returns the total receptive field size.
    #[inline(always)]
    pub fn receptive_field(&self) -> usize {
        self.receptive_field_size
    }

    /// Reallocates internal buffers to support the given maximum block size.
    ///
    /// Any block size is accepted: processing is internally chunked into
    /// sub-blocks of ≤ `WAVENET_MAX_NUM_FRAMES` (64) by [`process`], matching
    /// the kernel scratch buffer capacity. Call this to inform the model
    /// of the negotiated host block size so that internal ring
    /// buffers are sized correctly.
    ///
    /// If `max_buf` is smaller than the current capacity, this is a no-op.
    /// If `max_buf` equals the current capacity, state variables are reset
    /// and all buffers are zero-filled in-place — avoiding heap allocation
    /// on the RT thread (F9 / RT-Safety).
    pub fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        if max_buf < self.max_buffer_size {
            return Ok(());
        }
        if max_buf == self.max_buffer_size {
            let rf = self.receptive_field_size;
            let ha_len = self.head_accum.len();
            self.head_accum[..ha_len].fill(0.0);
            self.head_write_pos = rf;
            for buf in self.layer_buffers.iter_mut() {
                let len = buf.size();
                buf[..len].fill(0.0);
            }
            self.layer_buffer_starts
                .copy_from_slice(&self.layer_ring_sizes);
            let li_len = self.layer_in.len();
            self.layer_in[..li_len].fill(0.0);
            return Ok(());
        }
        self.max_buffer_size = max_buf;
        let rf = self.receptive_field_size;

        self.layer_buffers.clear();
        self.layer_ring_sizes.clear();
        self.layer_lookbacks.clear();
        self.layer_buffer_starts.clear();

        for i in 0..A2_NUM_LAYERS {
            let max_lookback = (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            let cap = max_lookback + max_buf + 1;
            let mb = MirroredBuffer::<f32>::new(cap * CH)?;
            let ring_size = mb.size();
            self.layer_buffers.push(mb);
            self.layer_ring_sizes.push(ring_size);
            self.layer_lookbacks.push(max_lookback * CH);
            self.layer_buffer_starts.push(ring_size);
        }

        self.layer_in = AlignedVec::new(CH * max_buf, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");

        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        self.head_ring_mask = head_ring_size - 1;
        self.head_accum = AlignedVec::new(head_ring_size * CH, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        self.head_write_pos = rf;

        Ok(())
    }

    /// Resets internal state for a new sample rate and max buffer size.
    pub fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.set_max_buffer_size(max_buffer_size)?;
        if self.prewarm_on_reset {
            self.prewarm();
        }
        Ok(())
    }

    /// Returns whether weights have been loaded via `set_weights`.
    #[inline(always)]
    pub fn has_weights(&self) -> bool {
        !self.layers.is_empty()
    }

    /// Stores the raw layer JSON for FiLM config parsing during weight loading.
    pub fn set_layer_raw(&mut self, raw: Option<Value>) {
        self.layer_raw = raw;
    }
}
