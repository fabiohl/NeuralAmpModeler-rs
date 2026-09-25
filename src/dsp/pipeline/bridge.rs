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
///
/// # RT ordering invariants (R-6 / A8)
///
/// The lock-free protocol is sound **only** under the following ordering rules,
/// which the reader/writer methods below implement and which host callers must
/// not bypass:
///
/// 1. **Skip-if-not-consumed (writer).** `DspBridgeWriter::write_block` /
///    `write_silence` never overwrite the buffer the reader may still be
///    reading: they compare `generation` (Relaxed) against `consumed_gen`
///    (Acquire) and, when the previous block has not been consumed yet,
///    increment `dropped_frames` and return instead of writing. This converts
///    a potential race into a deterministic, measurable dropout.
/// 2. **Back-buffer write exclusivity.** The writer only ever mutates
///    `buffers[1 - active_read_idx]` (the complement of the buffer the reader
///    selects via `active_read_idx`), so the reader never observes a torn
///    payload. The selected back-buffer is published atomically by storing
///    `active_read_idx` (Release) **after** its `n_samples` / `generation` /
///    sample fields are written.
/// 3. **Generation double-load (reader).** `DspBridgeReader::read_block`
///    loads `generation` with Acquire twice — once before and once after
///    reading `active_read_idx` — and returns `None` when the two differ. This
///    rejects a block that was published concurrently with the read (torn
///    publication window) instead of consuming a partially-written buffer.
/// 4. **Front-buffer generation check (reader).** After selecting the front
///    buffer, `read_block` verifies `front_buf.generation == current_gen`
///    before reading samples. A mismatch means the writer flipped the active
///    index mid-read; the reader skips rather than consuming stale/partial data.
/// 5. **Reset is teardown-only.** `DspBridge::reset_to_silence` requires
///    `&mut self` so it can only be invoked with exclusive access — i.e. after
///    real-time producer/consumer threads have stopped and no `DspBridgeWriter` /
///    `DspBridgeReader` / `&DspBridge` is live. Calling it while a callback may
///    run is a data race on the (non-atomic) `n_samples`/`generation` fields.
/// 6. **Single-writer / single-reader discipline.** Exactly one capture thread
///    may hold or use a `DspBridgeWriter`, and exactly one playback thread a
///    `DspBridgeReader`. This is an invariant of the bridge type, not a
///    runtime-enforced property: the payload writes in `write_block` are
///    non-atomic, so the `unsafe impl Sync` on both faces is sound only under
///    this rule.
#[repr(C, align(128))]
pub struct DspBridge {
    /// The two physical buffers (front/back) for double-buffering.
    pub buffers: [BridgeBuffer; 2],

    // =========================================================================
    // Writer-Owned Line (64 bytes):
    // Written exclusively by the capture callback (DSP producer) during block
    // publication and frame-drop accounting. Main thread drains dropped_frames.
    // =========================================================================
    /// Index of the active buffer for READING (0 or 1). Capture always writes to (1 - active).
    pub active_read_idx: AtomicUsize,
    /// Generation counter — incremented on each write by the capture callback.
    /// Playback compares with its local copy to detect new data.
    pub generation: AtomicU64,
    /// Counter of dropped frames (overwritten without consumption).
    /// Incremented by RT callbacks, drained via `drain_dropped_frames()` by the main loop.
    pub dropped_frames: AtomicU32,
    /// Padding to isolate writer-owned atomic fields to their own 64-byte cache line.
    /// (8 + 8 + 4 = 20 bytes; 64 - 20 = 44 bytes).
    _pad_writer: [u8; 44],

    // =========================================================================
    // Reader-Owned Line (64 bytes):
    // Written exclusively by the playback callback (consumer) upon block consumption.
    // Read by the writer to detect pending (unconsumed) frames.
    // =========================================================================
    /// Consumed generation counter — updated by the playback callback.
    pub consumed_gen: AtomicU64,
    /// Padding to isolate reader-owned atomic fields to their own 64-byte cache line
    /// and maintain 128-byte alignment for the entire struct.
    /// (8 bytes; 64 - 8 = 56 bytes).
    _pad_reader: [u8; 56],
}

impl DspBridge {
    /// Creates a new zero-initialized `DspBridge`.
    pub fn new() -> Self {
        Self {
            buffers: [BridgeBuffer::new(), BridgeBuffer::new()],
            active_read_idx: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
            dropped_frames: AtomicU32::new(0),
            _pad_writer: [0; 44],
            consumed_gen: AtomicU64::new(0),
            _pad_reader: [0; 56],
        }
    }

    /// Allocates and initializes a new heap-allocated `DspBridge`.
    ///
    /// The allocation is performed in-place on the heap using `Box::new_uninit()`
    /// to avoid placing a ~131 KiB temporary on the caller's thread stack.
    pub fn new_boxed() -> Box<Self> {
        let mut boxed = Box::<Self>::new_uninit();
        // SAFETY: `boxed` is freshly allocated memory for `DspBridge`. Zeroing all bytes initializes
        // the 128 KiB sample buffers and padding cleanly, and atomic fields are explicitly constructed.
        unsafe {
            core::ptr::write_bytes(boxed.as_mut_ptr(), 0, 1);
            let b = boxed.as_mut_ptr();
            (*b).active_read_idx = AtomicUsize::new(0);
            (*b).generation = AtomicU64::new(0);
            (*b).dropped_frames = AtomicU32::new(0);
            (*b).consumed_gen = AtomicU64::new(0);
            boxed.assume_init()
        }
    }

    /// Initializes a `DspBridge` in-place at the provided raw pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for writes, properly aligned to `align_of::<DspBridge>()`,
    /// and point to memory of at least `size_of::<DspBridge>()` bytes.
    pub unsafe fn init_in_place(ptr: *mut Self) {
        // SAFETY: Caller guarantees `ptr` is valid for writes, properly aligned to 128 bytes,
        // and points to at least `size_of::<DspBridge>()` bytes.
        unsafe {
            core::ptr::write_bytes(ptr, 0, 1);
            let b = &mut *ptr;
            b.active_read_idx = AtomicUsize::new(0);
            b.generation = AtomicU64::new(0);
            b.dropped_frames = AtomicU32::new(0);
            b.consumed_gen = AtomicU64::new(0);
        }
    }

    /// Drains the dropped frames counter, returning the accumulated value and resetting it.
    ///
    /// RT-Safe for the reader: uses atomic `swap` without locks.
    pub fn drain_dropped_frames(&self) -> u32 {
        self.dropped_frames.swap(0, Ordering::Relaxed)
    }

    /// Resets the bridge state to silence during host teardown or reconnection.
    ///
    /// Clears both buffers' lengths to 0 and synchronizes `generation` and
    /// `consumed_gen` so that a newly connected playback reader observes
    /// silence instead of replaying stale audio from a previous stream session
    /// (readers treat a zero-length block as silence and do not advance their
    /// generation on it).
    ///
    /// # Teardown-only gate (`&mut self`)
    ///
    /// This method requires **exclusive access** (`&mut self`), so it can only be
    /// invoked after real-time producer/consumer threads have stopped and no
    /// `DspBridgeWriter` / `DspBridgeReader` (or `&DspBridge`) is live — see the
    /// RT ordering invariants on [`DspBridge`]. The fields it clears are
    /// non-atomic and shared with callbacks under the release/acquire protocol;
    /// writing them through `&self` would be a data race if any callback were
    /// still running. Requiring `&mut` makes the teardown-only contract hold at
    /// compile time for safe callers (raw-pointer callers retain their existing
    /// exclusive-access obligation).
    pub fn reset_to_silence(&mut self) {
        let curr_gen = self.generation.load(Ordering::Relaxed);
        let next_gen = curr_gen.wrapping_add(1);
        self.buffers[0].n_samples = 0;
        self.buffers[0].generation = 0;
        self.buffers[1].n_samples = 0;
        self.buffers[1].generation = 0;
        self.active_read_idx.store(0, Ordering::Release);
        self.consumed_gen.store(next_gen, Ordering::Release);
        self.generation.store(next_gen, Ordering::Release);
    }
}

impl Default for DspBridge {
    fn default() -> Self {
        Self::new()
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
/// Sending between threads for initialization is safe.
unsafe impl Send for DspBridgeWriter {}
/// SAFETY: `Sync` is sound **only under the single-writer/single-reader
/// discipline declared as an invariant of the bridge type** (DspBridge doc,
/// "RT ordering invariants", rule 6): exactly one capture thread may hold or
/// use a `DspBridgeWriter`, and exactly one playback thread a
/// `DspBridgeReader`. Under that discipline the `&self` writer methods mutate
/// only the back-buffer the reader never touches and publish it with Release
/// stores the reader observes with Acquire — no data race. The atomics alone
/// do NOT provide this: `write_block` writes the payload with non-atomic
/// `copy_nonoverlapping` and updates `n_samples`/`generation` non-atomically,
/// so two threads sharing `&DspBridgeWriter` could legally collide. The
/// discipline is upheld by the bridge lifecycle (the writer is handed to one
/// capture callback, the reader to one playback callback), not enforceable by
/// the type itself without an API redesign (an alternative `&mut`-only wrapper
/// design was considered and deferred to avoid a public-API break).
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
        let n_bridge = n_pw.min(MAX_BRIDGE_BUF);
        debug_assert!(
            resamp_out_l.len() >= n_bridge,
            "resamp_out_l slice smaller than n_bridge"
        );
        debug_assert!(
            process_mono || resamp_out_r.len() >= n_bridge,
            "resamp_out_r slice smaller than n_bridge in stereo mode"
        );

        // SAFETY: self.0 is NonNull<DspBridge> into heap-immortal memory. The back-buffer
        // (1 - active_read_idx) is exclusively written here; the reader only accesses the
        // complementary front-buffer.
        // Bounds & non-overlap: `n_bridge <= MAX_BRIDGE_BUF` by definition; `back_buf.buf_l` and
        // `back_buf.buf_r` have capacity `MAX_BRIDGE_BUF`. `resamp_out_l.len() >= n_bridge` and
        // (in stereo) `resamp_out_r.len() >= n_bridge` by caller contract (verified by debug_assert).
        // Memory regions do not overlap (input is resampler scratch, output is bridge back-buffer).
        // Atomic fences (Release) synchronize visibility.
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
/// Sending between threads for init is safe.
unsafe impl Send for DspBridgeReader {}
/// SAFETY: sound under the bridge's single-writer/single-reader type invariant
/// (see the writer's `Sync` SAFETY note): exactly one playback thread reads,
/// its `&self` method only reads the front-buffer selected atomically via
/// `active_read_idx` (Acquire) and publishes consumption via `consumed_gen`
/// (Release), while the single writer mutates only the complementary
/// back-buffer — no mutable aliasing occurs under that discipline.
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
#[path = "bridge_test.rs"]
mod bridge_test;
