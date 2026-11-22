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
    pub(crate) layers: Vec<A2Layer>,

    /// Input rechannel weights: `Conv1x1(1 → CH)` (no bias).
    pub(crate) rechannel_w_f32: AlignedVec<f32>,

    /// Head convolution (K=16 over skip-connection accumulator, bias, head_scale).
    pub(crate) head_conv: Option<A2HeadConv>,

    /// Head accumulator ring buffer (skip-connection sum, column-major, pow2 size).
    pub(crate) head_accum: AlignedVec<f32>,

    /// Write position in `head_accum` (in columns, wraps via `head_ring_mask`).
    pub(crate) head_write_pos: usize,

    /// Ring mask for `head_accum` (pow2 ring, mask = capacity - 1).
    pub(crate) head_ring_mask: usize,

    /// Per-layer history buffers: one MirroredBuffer per layer (23 total).
    /// Each buffer provides 2× virtual mapping for branchless ring access.
    pub(crate) layer_buffers: Vec<MirroredBuffer<f32>>,

    /// Per-layer ring sizes in elements (pow2 page-aligned). For rewind: `start -= ring_size`.
    pub(crate) layer_ring_sizes: Vec<usize>,

    /// Per-layer maximum dilation lookback = (kernel-1) * dilation.
    pub(crate) layer_lookbacks: Vec<usize>,

    /// Per-layer buffer starts (advanced with each written frame, rewound near 2× boundary).
    pub(crate) layer_buffer_starts: Vec<usize>,

    /// Inter-layer data buffer: `CH × max_buffer_size` f32, reused across layers.
    /// Each layer reads from it, then writes its l1x1 residual back (in-place update).
    pub(crate) layer_in: AlignedVec<f32>,

    /// Total receptive field: sum of `(kernel-1)*dilation` + head kernel - 1.
    pub(crate) receptive_field_size: usize,

    /// Maximum frames per processing block (= `WAVENET_MAX_NUM_FRAMES`).
    pub(crate) max_buffer_size: usize,
    /// Largest block size this instance has been sized for. Distinct from
    /// [`Self::max_buffer_size`], which stays capped at
    /// `WAVENET_MAX_NUM_FRAMES` because the internal rings never inflate.
    /// Tracking the negotiated maximum makes a later smaller request (e.g.
    /// the host block after the loader's `MAX_RESAMP_BUF` sizing) a true
    /// no-op, so it cannot silently wipe stabilized state.
    pub(crate) negotiated_max_buffer_size: usize,

    /// Raw JSON for the single layer array, preserved for FiLM config parsing.
    /// `None` when the model is constructed directly (not via JSON deserialization).
    pub(crate) layer_raw: Option<Value>,

    /// Per-frame scratch buffer for the fallback conv path (CH f32).
    /// Allocated once at model creation to avoid heap ops on the hot path.
    pub(crate) z_scratch: AlignedVec<f32>,

    /// `RtStatusFlags` for silent RT→Main telemetry (RF8).
    pub(crate) rt_status: Option<Arc<crate::common::spsc::RtStatusFlags>>,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub(crate) prewarm_on_reset: bool,
    /// Zeroed-sample stabilization work still pending for the deferred split
    /// pass armed by [`Self::prewarm_reset`](super::super::NamModel::prewarm_reset).
    /// Always `0` for freshly built models; the integral
    /// [`Self::prewarm`](super::super::NamModel::prewarm) /
    /// [`Self::reset`](super::super::NamModel::reset) paths complete it by
    /// construction and never consult or alter it otherwise.
    pub(crate) prewarm_pending: usize,
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
            rechannel_w_f32: AlignedVec::new(CH, 0.0f32)?,
            head_conv: None,
            head_accum: AlignedVec::new(head_ring_size * CH, 0.0f32)?,
            head_write_pos: rf,
            head_ring_mask,
            layer_buffers,
            layer_ring_sizes,
            layer_lookbacks,
            layer_buffer_starts,
            layer_in: AlignedVec::new(CH * max_buf, 0.0f32)?,
            receptive_field_size: rf,
            max_buffer_size: max_buf,
            negotiated_max_buffer_size: max_buf,
            layer_raw: None,
            z_scratch: AlignedVec::new(CH, 0.0f32)?,
            rt_status: None,
            prewarm_on_reset: true,
            prewarm_pending: 0,
        })
    }

    /// Returns the channel count.
    #[inline(always)]
    pub fn channels(&self) -> usize {
        CH
    }

    /// Injects `RtStatusFlags` for RT→Main telemetry (RF8).
    ///
    /// Off-RT only: replaces (and thus may free) a previous `Arc`.
    pub fn inject_rt_status(&mut self, rt_status: Arc<crate::common::spsc::RtStatusFlags>) {
        self.rt_status = Some(rt_status);
    }

    /// RT-safe `RtStatusFlags` injection.
    ///
    /// The canonical instance is expected to be pre-injected by the off-RT
    /// loader with [`Self::inject_rt_status`]; when the RT thread later hands
    /// the same `Arc`, this is a `ptr_eq` no-op that touches no reference
    /// count. When no status is stored yet, a new strong reference is adopted
    /// (atomic increment only). A different live instance is refused instead of
    /// being replaced, because replacing it would drop the previous strong
    /// reference — and `Arc::drop_slow` on the audio thread is forbidden.
    ///
    /// Zero allocation and zero deallocation.
    #[inline]
    pub fn inject_rt_status_rt(&mut self, rt_status: &Arc<crate::common::spsc::RtStatusFlags>) {
        match self.rt_status.as_ref() {
            Some(current) if Arc::ptr_eq(current, rt_status) => {}
            Some(_) => {}
            None => self.rt_status = Some(Arc::clone(rt_status)),
        }
    }

    /// Test/tooling hook: whether the stored status is exactly `other`.
    ///
    /// Lets a host-agnostic consumer verify, without touching the private
    /// field, that the off-RT pre-injection landed the canonical instance
    /// (the RT hand-off then reduces to a `ptr_eq` no-op).
    #[cfg(any(test, feature = "testing"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
    pub fn rt_status_ptr_eq(&self, other: &Arc<crate::common::spsc::RtStatusFlags>) -> bool {
        self.rt_status
            .as_ref()
            .is_some_and(|stored| Arc::ptr_eq(stored, other))
    }

    /// Returns the total receptive field size.
    #[inline(always)]
    pub fn receptive_field(&self) -> usize {
        self.receptive_field_size
    }

    /// Returns the total receptive field size in samples.
    #[inline(always)]
    pub fn receptive_field_size(&self) -> usize {
        self.receptive_field_size
    }

    /// Returns the A2 layers.
    #[inline(always)]
    pub fn layers(&self) -> &[A2Layer] {
        &self.layers
    }

    /// Returns the input rechannel weight slice.
    #[inline(always)]
    pub fn rechannel_w_f32(&self) -> &[f32] {
        &self.rechannel_w_f32
    }

    /// Returns the optional head convolution.
    #[inline(always)]
    pub fn head_conv(&self) -> Option<&A2HeadConv> {
        self.head_conv.as_ref()
    }

    /// Returns the head accumulator ring buffer.
    #[inline(always)]
    pub fn head_accum(&self) -> &[f32] {
        &self.head_accum
    }

    /// Returns the current head write position in columns.
    #[inline(always)]
    pub fn head_write_pos(&self) -> usize {
        self.head_write_pos
    }

    /// Returns the power-of-2 ring mask for `head_accum`.
    #[inline(always)]
    pub fn head_ring_mask(&self) -> usize {
        self.head_ring_mask
    }

    /// Returns the per-layer history mirrored buffers.
    #[inline(always)]
    pub fn layer_buffers(&self) -> &[MirroredBuffer<f32>] {
        &self.layer_buffers
    }

    /// Returns the per-layer ring sizes in elements.
    #[inline(always)]
    pub fn layer_ring_sizes(&self) -> &[usize] {
        &self.layer_ring_sizes
    }

    /// Returns the per-layer lookbacks in elements.
    #[inline(always)]
    pub fn layer_lookbacks(&self) -> &[usize] {
        &self.layer_lookbacks
    }

    /// Returns the per-layer buffer start cursors.
    #[inline(always)]
    pub fn layer_buffer_starts(&self) -> &[usize] {
        &self.layer_buffer_starts
    }

    /// Returns the inter-layer working buffer.
    #[inline(always)]
    pub fn layer_in(&self) -> &[f32] {
        &self.layer_in
    }

    /// Returns the configured maximum block size in frames.
    #[inline(always)]
    pub fn max_buffer_size(&self) -> usize {
        self.max_buffer_size
    }

    /// Returns the optional raw JSON layer metadata.
    #[inline(always)]
    pub fn layer_raw(&self) -> Option<&Value> {
        self.layer_raw.as_ref()
    }

    /// Returns whether prewarm runs during `reset()`.
    #[inline(always)]
    pub fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    /// Returns whether a deferred prewarm pass is pending.
    #[inline(always)]
    pub fn prewarm_pending(&self) -> bool {
        self.prewarm_pending > 0
    }

    /// Resets internal state without inflating the ring buffers beyond
    /// `WAVENET_MAX_NUM_FRAMES` (64).
    ///
    /// Any block size is accepted by [`process`]: processing is internally
    /// chunked into sub-blocks of ≤ `WAVENET_MAX_NUM_FRAMES` (64), so the
    /// 23 per-layer `MirroredBuffer`s, `layer_in`, and `head_accum` never need
    /// capacity above `max_lookback + WAVENET_MAX_NUM_FRAMES + 1`.
    ///
    /// A request no larger than the largest size this instance was already
    /// sized for is a no-op, including equality: the negotiated size is
    /// idempotent so a host quantum that repeats the loader's `MAX_RESAMP_BUF`
    /// sizing cannot wipe stabilized/prewarmed state. Callers that need a
    /// deliberate reset must call [`Self::reset`] or `clear_state`. Only a
    /// genuine growth re-zeroes state in-place — zero heap allocation
    /// (F9 / RT-Safety).
    pub fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        if max_buf <= self.negotiated_max_buffer_size {
            return Ok(());
        }
        let rf = self.receptive_field_size;
        let ha_len = self.head_accum.len();
        self.head_accum[..ha_len].fill(0.0);
        self.head_write_pos = rf;
        for buf in self.layer_buffers.iter_mut() {
            buf.fill(0.0);
        }
        self.layer_buffer_starts
            .copy_from_slice(&self.layer_ring_sizes);
        let li_len = self.layer_in.len();
        self.layer_in[..li_len].fill(0.0);
        self.negotiated_max_buffer_size = max_buf;
        Ok(())
    }

    /// Resets internal state for a new sample rate and max buffer size.
    ///
    /// Ring and scratch capacities stay capped at `WAVENET_MAX_NUM_FRAMES` (64),
    /// so `reset()` only needs a single buffer clear: via `prewarm()` (which
    /// starts with `prewarm_reset()` -> `a2_prewarm_common`) when
    /// `prewarm_on_reset` is enabled, or `clear_state()` otherwise.
    pub fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) -> anyhow::Result<()> {
        if self.prewarm_on_reset {
            self.prewarm();
        } else {
            self.clear_state();
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

    /// Test-only helper to fill `head_accum`, `layer_buffers`, and `head_write_pos`
    /// with sentinel values when verifying that `reset()` clears submodel state.
    #[cfg(feature = "testing")]
    #[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
    pub fn poison_state_for_reset_test(&mut self, fill_val: f32, write_pos: usize) {
        self.head_accum.fill(fill_val);
        self.head_write_pos = write_pos;
        for buf in &mut self.layer_buffers {
            buf.fill(fill_val);
        }
    }
}
