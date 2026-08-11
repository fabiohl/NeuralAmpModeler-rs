// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Bridge types for lock-free communication between capture and playback.
//!
//! Contains `DspBridge`, `BridgeBuffer`, `BridgeRef`, `DspBridgeWriter`,
//! `DspBridgeReader` and the constants `MAX_BRIDGE_BUF` / `MAX_RESAMP_BUF`.

use std::sync::atomic::Ordering;

/// Maximum intermediate buffer size between the two streams (capture → playback).
/// Sized for the maximum host quantum (8192 frames).
pub const MAX_BRIDGE_BUF: usize = 8192;
/// Maximum buffer size for resampling.
///
/// **RT Safety Contract**: This value determines the size of pre-allocated buffers
/// in `DspPipelineContext`. Increasing this value impacts the size of the processing
/// closure object (which must fit on the RT thread stack or be moved to the heap).
/// Currently fixed at 8192 samples (32 KiB per channel).
///
/// **Ratio-Aware Safety**: The resampler may produce more output than input during
/// upsampling (e.g. 44100→48000 Hz). The inference pipeline handles this via
/// internal chunking: input is sliced into blocks bounded by
/// `NamResampler::max_input_samples(MAX_RESAMP_BUF, host_rate, nam_rate)`,
/// ensuring no buffer overflow even at maximum host quantum.
pub const MAX_RESAMP_BUF: usize = 8192;

/// Individual audio buffer for the DspBridge (double-buffer).
#[repr(align(128))]
pub struct BridgeBuffer {
    /// Processed output buffer, left channel.
    pub buf_l: [f32; MAX_BRIDGE_BUF],
    /// Processed output buffer, right channel.
    pub buf_r: [f32; MAX_BRIDGE_BUF],
    /// Number of valid samples in the current buffer.
    pub n_samples: u32,
}

/// Shared buffer between the capture callback (DSP) and the playback callback.
///
/// The capture callback writes the processed result here with `fence(Release)`;
/// the playback callback reads with `fence(Acquire)`. The atomic `generation` allows
/// the playback to detect whether new data is available without spin-lock.
///
/// Aligned to 128 bytes to avoid false-sharing between the two RT callbacks.
#[repr(align(128))]
pub struct DspBridge {
    /// The two physical buffers (front/back) for double-buffering.
    pub buffers: [BridgeBuffer; 2],
    /// Index of the active buffer for READING (0 or 1). Capture always writes to (1 - active).
    pub active_read_idx: std::sync::atomic::AtomicUsize,
    /// Generation counter — incremented on each write by the capture callback.
    /// Playback compares with its local copy to detect new data.
    pub generation: std::sync::atomic::AtomicU64,
    /// Consumed generation counter — updated by the playback callback.
    pub consumed_gen: std::sync::atomic::AtomicU64,
    /// Counter of dropped frames (overwritten without consumption).
    /// Incremented by RT callbacks, drained via `drain_dropped_frames()` by the main loop.
    pub dropped_frames: std::sync::atomic::AtomicU32,
}

impl DspBridge {
    /// Drains the dropped frames counter, returning the accumulated value and resetting it.
    ///
    /// RT-Safe for the reader: uses atomic `swap` without locks.
    pub fn drain_dropped_frames(&self) -> u32 {
        self.dropped_frames.swap(0, Ordering::Relaxed)
    }
}

#[derive(Clone, Copy)]
/// Safe reference to the DspBridge (shared across threads via pointer).
pub struct BridgeRef(*mut DspBridge);

impl BridgeRef {
    /// Creates a new BridgeRef.
    /// # Safety
    /// The pointer must be valid and non-null.
    #[inline(always)]
    pub unsafe fn new(ptr: *mut DspBridge) -> Self {
        assert!(!ptr.is_null());
        Self(ptr)
    }

    /// Creates a null BridgeRef (for when the bridge is not needed).
    #[inline(always)]
    pub fn null() -> Self {
        Self(std::ptr::null_mut())
    }

    /// Checks whether BridgeRef is null.
    #[inline(always)]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// Returns the internal raw pointer.
    /// # Safety
    /// The caller must ensure the pointer is valid if dereferenced.
    #[inline(always)]
    pub unsafe fn as_ptr(self) -> *mut DspBridge {
        self.0
    }
}

#[derive(Clone, Copy)]
/// Write face of `DspBridge` exposed to the capture thread.
pub struct DspBridgeWriter(std::ptr::NonNull<DspBridge>);

/// SAFETY: DspBridgeWriter owns a `NonNull<DspBridge>` that points to a heap-immortal
/// allocation (Box::leak in standalone mode, or host/plugin lifecycle memory).
/// The capture thread has exclusive write access to the back-buffer; the playback
/// thread only reads the active front-buffer. All synchronization uses atomic
/// ordering (Release/Acquire). Sending between threads for initialization is safe.
unsafe impl Send for DspBridgeWriter {}
/// SAFETY: DspBridgeWriter exposes only immutable reference access to the shared
/// bridge through &self methods. All state transitions are mediated by atomic
/// loads/stores with appropriate Release/Acquire ordering — no data races possible.
unsafe impl Sync for DspBridgeWriter {}

impl DspBridgeWriter {
    /// Creates a `DspBridgeWriter` from a raw pointer to `DspBridge`.
    /// # Safety
    /// The pointer must be valid and non-null.
    #[inline(always)]
    pub unsafe fn new(ptr: *mut DspBridge) -> Self {
        assert!(!ptr.is_null());
        Self(unsafe { std::ptr::NonNull::new_unchecked(ptr) })
    }

    /// Creates a `DspBridgeWriter` from a `BridgeRef`.
    /// Returns `None` if the reference is null.
    #[inline(always)]
    pub fn from_ref(r: BridgeRef) -> Option<Self> {
        std::ptr::NonNull::new(r.0).map(Self)
    }

    /// Writes a stereo sample block into the bridge's active back-buffer.
    ///
    /// Skip-on-overflow: if the reader hasn't consumed the last published generation,
    /// the write is skipped and `dropped_frames` is incremented instead of overwriting
    /// the buffer the reader may be actively reading. This converts potential UB into
    /// deterministic, measurable dropouts.
    pub fn write_block(
        &self,
        resamp_out_l: &[f32],
        resamp_out_r: &[f32],
        n_pw: usize,
        process_mono: bool,
    ) {
        // SAFETY: self.0 is NonNull<DspBridge> into heap-immortal memory. The back-buffer
        // (1 - active_read_idx) is exclusively written here; the reader only accesses the
        // complementary front-buffer. Atomic fences (Release) synchronize visibility.
        unsafe {
            let bridge = self.0.as_ref();

            let current_gen = bridge.generation.load(Ordering::Relaxed);
            let consumed_gen = bridge.consumed_gen.load(Ordering::Acquire);
            if current_gen > consumed_gen {
                let _ =
                    bridge
                        .dropped_frames
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                            Some(v.saturating_add(1))
                        });
                return;
            }

            let back_idx = 1 - bridge.active_read_idx.load(Ordering::Relaxed);
            let back_buf = &mut (*self.0.as_ptr()).buffers[back_idx];

            let n_bridge = n_pw.min(MAX_BRIDGE_BUF);
            core::ptr::copy_nonoverlapping(
                resamp_out_l.as_ptr(),
                back_buf.buf_l.as_mut_ptr(),
                n_bridge,
            );
            if process_mono {
                core::ptr::copy_nonoverlapping(
                    resamp_out_l.as_ptr(),
                    back_buf.buf_r.as_mut_ptr(),
                    n_bridge,
                );
            } else {
                core::ptr::copy_nonoverlapping(
                    resamp_out_r.as_ptr(),
                    back_buf.buf_r.as_mut_ptr(),
                    n_bridge,
                );
            }
            back_buf.n_samples = n_bridge as u32;

            bridge.active_read_idx.store(back_idx, Ordering::Release);
            bridge.generation.store(current_gen + 1, Ordering::Release);
        }
    }

    /// Resets the active back-buffer to indicate silence (0 samples).
    ///
    /// Skip-on-overflow: same prevention as `write_block` — if the reader hasn't consumed
    /// the last published generation, the write is skipped and `dropped_frames` is incremented.
    pub fn write_silence(&self) {
        // SAFETY: Same rationale as write_block: pointer is NonNull into heap-immortal
        // memory, back-buffer is write-exclusive, atomic Release fences synchronize
        // with the reader's Acquire loads.
        unsafe {
            let bridge = self.0.as_ref();

            let current_gen = bridge.generation.load(Ordering::Relaxed);
            let consumed_gen = bridge.consumed_gen.load(Ordering::Acquire);
            if current_gen > consumed_gen {
                let _ =
                    bridge
                        .dropped_frames
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                            Some(v.saturating_add(1))
                        });
                return;
            }

            let back_idx = 1 - bridge.active_read_idx.load(Ordering::Relaxed);
            let back_buf = &mut (*self.0.as_ptr()).buffers[back_idx];
            back_buf.n_samples = 0;

            bridge.active_read_idx.store(back_idx, Ordering::Release);
            bridge.generation.store(current_gen + 1, Ordering::Release);
        }
    }
}

#[derive(Clone, Copy)]
/// Read face of `DspBridge` exposed to the playback thread.
pub struct DspBridgeReader(std::ptr::NonNull<DspBridge>);

/// SAFETY: DspBridgeReader owns a `NonNull<DspBridge>` pointing to a heap-immortal
/// allocation (same lifecycle as DspBridgeWriter — Box::leaked or host/plugin memory).
/// The playback thread has exclusive read access to the active front-buffer (indicated
/// by `active_read_idx`) while the capture thread writes the back-buffer. All
/// synchronization uses atomic ordering. Sending between threads for init is safe.
unsafe impl Send for DspBridgeReader {}
/// SAFETY: DspBridgeReader exposes only &self reads from the bridge's front-buffer
/// (selected atomically via active_read_idx with Acquire ordering). No mutable
/// aliasing occurs — the capture thread writes the complementary back-buffer.
unsafe impl Sync for DspBridgeReader {}

impl DspBridgeReader {
    /// Creates a `DspBridgeReader` from a raw pointer to `DspBridge`.
    /// # Safety
    /// The pointer must be valid and non-null.
    #[inline(always)]
    pub unsafe fn new(ptr: *mut DspBridge) -> Self {
        assert!(!ptr.is_null());
        Self(unsafe { std::ptr::NonNull::new_unchecked(ptr) })
    }

    /// Creates a `DspBridgeReader` from a `BridgeRef`.
    /// Returns `None` if the reference is null.
    #[inline(always)]
    pub fn from_ref(r: BridgeRef) -> Option<Self> {
        std::ptr::NonNull::new(r.0).map(Self)
    }

    /// Attempts to read an audio block from the bridge, passing references to L and R channels to a closure.
    ///
    /// Returns `Some(R)` if a new, valid block is available.
    /// Otherwise, returns `None`.
    pub fn read_block<F, R>(&self, last_bridge_gen: &mut u64, f: F) -> Option<R>
    where
        F: FnOnce(&[f32], &[f32]) -> R,
    {
        // SAFETY: self.0 is NonNull<DspBridge> into heap-immortal memory. The front-buffer
        // (active_read_idx) is exclusively read here; the writer only accesses the
        // complementary back-buffer. Acquire loads synchronize with the writer's Release stores.
        unsafe {
            let bridge = self.0.as_ref();
            let current_gen = bridge.generation.load(Ordering::Acquire);
            if current_gen == *last_bridge_gen {
                return None;
            }
            *last_bridge_gen = current_gen;
            bridge.consumed_gen.store(current_gen, Ordering::Release);

            let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
            let front_buf = &bridge.buffers[read_idx];
            let n_samples = front_buf.n_samples as usize;
            if n_samples == 0 || n_samples > MAX_BRIDGE_BUF {
                return None;
            }

            Some(f(
                &front_buf.buf_l[..n_samples],
                &front_buf.buf_r[..n_samples],
            ))
        }
    }
}
