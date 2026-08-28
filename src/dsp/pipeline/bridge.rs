// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Bridge types for lock-free communication between capture and playback.
//!
//! Contains `DspBridge`, `BridgeBuffer`, `BridgeRef`, `DspBridgeWriter`,
//! `DspBridgeReader` and the constants `MAX_BRIDGE_BUF` / `MAX_RESAMP_BUF`.

use crate::common::atomics::{AtomicU32, AtomicU64, AtomicUsize};
use core::sync::atomic::Ordering;

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
    /// Generation counter corresponding to the published buffer contents.
    pub generation: u64,
}

impl BridgeBuffer {
    /// Creates a zero-initialized `BridgeBuffer`.
    pub const fn new() -> Self {
        Self {
            buf_l: [0.0; MAX_BRIDGE_BUF],
            buf_r: [0.0; MAX_BRIDGE_BUF],
            n_samples: 0,
            generation: 0,
        }
    }
}

impl Default for BridgeBuffer {
    fn default() -> Self {
        Self::new()
    }
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
    pub active_read_idx: AtomicUsize,
    /// Generation counter — incremented on each write by the capture callback.
    /// Playback compares with its local copy to detect new data.
    pub generation: AtomicU64,
    /// Consumed generation counter — updated by the playback callback.
    pub consumed_gen: AtomicU64,
    /// Counter of dropped frames (overwritten without consumption).
    /// Incremented by RT callbacks, drained via `drain_dropped_frames()` by the main loop.
    pub dropped_frames: AtomicU32,
}

impl DspBridge {
    /// Drains the dropped frames counter, returning the accumulated value and resetting it.
    ///
    /// RT-Safe for the reader: uses atomic `swap` without locks.
    pub fn drain_dropped_frames(&self) -> u32 {
        self.dropped_frames.swap(0, Ordering::Relaxed)
    }

    /// Resets the bridge state to silence during host teardown or reconnection.
    ///
    /// Clears buffer lengths to 0, zero-fills sample arrays, and synchronizes
    /// `generation` and `consumed_gen` so that a newly connected playback reader
    /// observes silence instead of replaying stale audio from a previous stream session.
    pub fn reset_to_silence(&self) {
        let curr_gen = self.generation.load(Ordering::Relaxed);
        let next_gen = curr_gen.wrapping_add(1);
        let buf_ptr = self as *const DspBridge as *mut DspBridge;
        // SAFETY: self.buffers accesses internal mutable buffers under main-thread lifecycle/teardown control.
        unsafe {
            (*buf_ptr).buffers[0].n_samples = 0;
            (*buf_ptr).buffers[0].generation = 0;
            (*buf_ptr).buffers[1].n_samples = 0;
            (*buf_ptr).buffers[1].generation = 0;
        }
        self.active_read_idx.store(0, Ordering::Release);
        self.consumed_gen.store(next_gen, Ordering::Release);
        self.generation.store(next_gen, Ordering::Release);
    }
}

#[derive(Clone, Copy)]
/// Safe reference to the DspBridge (shared across threads via pointer).
pub struct BridgeRef(*mut DspBridge);

impl BridgeRef {
    /// Creates a new BridgeRef.
    ///
    /// # Safety
    ///
    /// The pointer must be valid and non-null. This is an initialization-path
    /// constructor; the null check is a `debug_assert!` (loud in dev builds,
    /// compiled out in release) because the lifetime is heap-immortal
    /// (`Box::leak`ed at startup, never freed) — a null pointer here is a
    /// programming error caught by debug tooling, not a runtime-recoverable
    /// condition. Release builds rely on the caller contract documented here.
    #[inline(always)]
    pub unsafe fn new(ptr: *mut DspBridge) -> Self {
        debug_assert!(!ptr.is_null(), "BridgeRef requires a non-null pointer");
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
    ///
    /// # Safety
    ///
    /// The pointer must be valid and non-null, and must reference heap-immortal
    /// memory (leaked `Box`, or host/plugin lifecycle memory that outlives the
    /// writer) — see the `Send`/`Sync` SAFETY comments. The null check is a
    /// `debug_assert!` (loud in dev builds, compiled out in release): a null
    /// pointer here is a programming error on the initialization path, not a
    /// runtime-recoverable condition.
    #[inline(always)]
    pub unsafe fn new(ptr: *mut DspBridge) -> Self {
        debug_assert!(
            !ptr.is_null(),
            "DspBridgeWriter requires a non-null pointer"
        );
        // SAFETY: the caller contract of this `unsafe fn` guarantees `ptr` is
        // non-null (checked above) and points to heap-immortal memory that
        // outlives the writer.
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

            let next_gen = current_gen + 1;
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
            back_buf.generation = next_gen;

            bridge.active_read_idx.store(back_idx, Ordering::Release);
            bridge.generation.store(next_gen, Ordering::Release);
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

            let next_gen = current_gen + 1;
            let back_idx = 1 - bridge.active_read_idx.load(Ordering::Relaxed);
            let back_buf = &mut (*self.0.as_ptr()).buffers[back_idx];
            back_buf.n_samples = 0;
            back_buf.generation = next_gen;

            bridge.active_read_idx.store(back_idx, Ordering::Release);
            bridge.generation.store(next_gen, Ordering::Release);
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
    ///
    /// # Safety
    ///
    /// The pointer must be valid and non-null, and must reference heap-immortal
    /// memory (leaked `Box`, or host/plugin lifecycle memory that outlives the
    /// reader) — see the `Send`/`Sync` SAFETY comments. The null check is a
    /// `debug_assert!` (loud in dev builds, compiled out in release): a null
    /// pointer here is a programming error on the initialization path, not a
    /// runtime-recoverable condition.
    #[inline(always)]
    pub unsafe fn new(ptr: *mut DspBridge) -> Self {
        debug_assert!(
            !ptr.is_null(),
            "DspBridgeReader requires a non-null pointer"
        );
        // SAFETY: the caller contract of this `unsafe fn` guarantees `ptr` is
        // non-null (checked above) and points to heap-immortal memory that
        // outlives the reader.
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
            let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
            let post_gen = bridge.generation.load(Ordering::Acquire);
            if current_gen != post_gen {
                return None;
            }
            let front_buf = &bridge.buffers[read_idx];
            if front_buf.generation != current_gen {
                return None;
            }
            let n_samples = front_buf.n_samples as usize;
            if n_samples == 0 || n_samples > MAX_BRIDGE_BUF {
                *last_bridge_gen = current_gen;
                bridge.consumed_gen.store(current_gen, Ordering::Release);
                return None;
            }

            let result = f(&front_buf.buf_l[..n_samples], &front_buf.buf_r[..n_samples]);

            *last_bridge_gen = current_gen;
            bridge.consumed_gen.store(current_gen, Ordering::Release);

            Some(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_reader_interleaved_writer_race() {
        let mut bridge = Box::new(DspBridge {
            buffers: [
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                    generation: 0,
                },
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                    generation: 0,
                },
            ],
            active_read_idx: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
            consumed_gen: AtomicU64::new(0),
            dropped_frames: AtomicU32::new(0),
        });

        let bridge_ptr = &mut *bridge as *mut DspBridge;
        // SAFETY: Pointer is valid and points to a allocated DspBridge instance.
        let writer = unsafe { DspBridgeWriter::new(bridge_ptr) };
        // SAFETY: Pointer is valid and points to a allocated DspBridge instance.
        let reader = unsafe { DspBridgeReader::new(bridge_ptr) };

        // Publish block 1 (generation 1)
        writer.write_block(&[1.0, 1.0], &[1.0, 1.0], 2, false);
        assert_eq!(bridge.generation.load(Ordering::Relaxed), 1);

        let mut last_gen = 0u64;

        // Simulate reader loading generation=1
        let current_gen = bridge.generation.load(Ordering::Acquire);
        assert_eq!(current_gen, 1);

        // Before reader loads active_read_idx, consumer/writer publishes block 2 (generation 2).
        // Since reader has not updated consumed_gen (still 0), writer skips block 2 if consumed_gen < current_gen.
        // To test writer publication race, update consumed_gen to 1 so writer can publish block 2:
        bridge.consumed_gen.store(1, Ordering::Release);
        writer.write_block(&[2.0, 2.0], &[2.0, 2.0], 2, false);
        assert_eq!(bridge.generation.load(Ordering::Relaxed), 2);

        // Now reader reads active_read_idx and post_gen
        let _read_idx = bridge.active_read_idx.load(Ordering::Acquire);
        let post_gen = bridge.generation.load(Ordering::Acquire);
        assert_ne!(current_gen, post_gen);

        // Reader's post_gen check fails because post_gen (2) != current_gen (1).
        // Calling read_block from scratch now:
        let res = reader.read_block(&mut last_gen, |l, _| l[0]);
        assert_eq!(res, Some(2.0));
        assert_eq!(last_gen, 2);
    }
}
