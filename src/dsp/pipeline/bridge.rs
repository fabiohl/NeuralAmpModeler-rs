// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Bridge types for lock-free communication between capture and playback.
//!
//! Contains `DspBridge`, `BridgeBuffer`, `BridgeRef`, `DspBridgeWriter`,
//! `DspBridgeReader` and the constants `MAX_BRIDGE_BUF` / `MAX_RESAMP_BUF`.

use crate::common::atomics::{AtomicExt, AtomicU32, AtomicU64, AtomicUsize};
use core::cell::UnsafeCell;
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
    /// Capture-side start timestamp (nanoseconds, monotonic TSC domain) that
    /// produced this buffer. `0` means "unknown" (silence placeholder or a
    /// writer that did not publish a stamp). Travels with the generation so the
    /// playback thread can compute capture→playback latency for the *delivered*
    /// generation instead of the last global capture timestamp (F8 / T5.2).
    pub capture_tsc: u64,
}

impl BridgeBuffer {
    /// Creates a zero-initialized `BridgeBuffer`.
    pub const fn new() -> Self {
        Self {
            buf_l: [0.0; MAX_BRIDGE_BUF],
            buf_r: [0.0; MAX_BRIDGE_BUF],
            n_samples: 0,
            generation: 0,
            capture_tsc: 0,
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
    ///
    /// Each buffer is wrapped in [`UnsafeCell`] to provide *interior
    /// mutability*: the capture callback mutates the complementary back-buffer
    /// while the playback callback reads the front-buffer, both reached through
    /// raw pointers obtained via [`UnsafeCell::get`]. This lets the lock-free
    /// protocol avoid materializing an aliasing whole-struct `&DspBridge` /
    /// `&mut DspBridge` on the hot path — which is undefined behavior under the
    /// Stacked Borrows / Tree Borrows aliasing models.
    ///
    /// `UnsafeCell` is `#[repr(transparent)]`, so the layout, field offsets and
    /// 128-byte cache-line isolation of `DspBridge` and `BridgeBuffer` are
    /// unchanged; the wrapper compiles to zero runtime cost.
    pub buffers: [UnsafeCell<BridgeBuffer>; 2],

    // =========================================================================
    // Writer-Owned Line (128 bytes):
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
    /// Padding to isolate writer-owned atomic fields to their own 128-byte cache line.
    /// 128 bytes match the Apple P-core line size; a 64-byte block would place the
    /// writer and reader fields below in the same physical line on Apple Silicon.
    /// (8 + 8 + 4 = 20 bytes; 128 - 20 = 108 bytes).
    _pad_writer: [u8; 108],

    // =========================================================================
    // Reader-Owned Line (128 bytes):
    // Written exclusively by the playback callback (consumer) upon block consumption.
    // Read by the writer to detect pending (unconsumed) frames.
    // =========================================================================
    /// Consumed generation counter — updated by the playback callback.
    pub consumed_gen: AtomicU64,
    /// Padding to isolate reader-owned atomic fields to their own 128-byte cache line
    /// and maintain 128-byte alignment for the entire struct.
    /// (8 bytes; 128 - 8 = 120 bytes).
    _pad_reader: [u8; 120],
}

impl DspBridge {
    /// Creates a new zero-initialized `DspBridge`.
    pub fn new() -> Self {
        Self {
            buffers: [
                UnsafeCell::new(BridgeBuffer::new()),
                UnsafeCell::new(BridgeBuffer::new()),
            ],
            active_read_idx: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
            dropped_frames: AtomicU32::new(0),
            _pad_writer: [0; 108],
            consumed_gen: AtomicU64::new(0),
            _pad_reader: [0; 120],
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
        // `&mut self` proves exclusive access (teardown-only contract), so the
        // safe `UnsafeCell::get_mut` accessor reaches each buffer without unsafe.
        let buf0 = self.buffers[0].get_mut();
        buf0.n_samples = 0;
        buf0.generation = 0;
        buf0.capture_tsc = 0;
        let buf1 = self.buffers[1].get_mut();
        buf1.n_samples = 0;
        buf1.generation = 0;
        buf1.capture_tsc = 0;
        self.active_read_idx.store(0, Ordering::Release);
        self.consumed_gen.store(next_gen, Ordering::Release);
        self.generation.store(next_gen, Ordering::Release);
    }

    /// Pre-loads one `frames`-sample silence block during off-RT setup/teardown
    /// windows where the caller holds exclusive access (same `&mut self`
    /// contract as [`reset_to_silence`](Self::reset_to_silence): no live
    /// producer/consumer).
    ///
    /// Without this pre-roll, a fresh session's first playback quantum observes
    /// `generation == last_bridge_gen` (both `0`) and reports bridge
    /// starvation, delivering the muted silence path and paying one quantum of
    /// zero-order-hold delay before the capture producer's first real block
    /// lands. Publishing one real zero-filled block advances the generation so
    /// the first read returns silence through the normal wet path (no
    /// starvation event); the capture producer overwrites it on its first
    /// callback.
    ///
    /// `frames` is clamped to [`MAX_BRIDGE_BUF`]; a `0` is treated as one sample
    /// so the block is never zero-length (which the reader treats as silence
    /// and does not surface as a block). `dropped_frames` is never touched
    /// here; the write is skipped, preserving the skip-on-overflow invariant,
    /// when a previous block is still unconsumed (`generation > consumed_gen`).
    pub fn prime_silence(&mut self, frames: usize) {
        let current_gen = self.generation.load(Ordering::Relaxed);
        let consumed_gen = self.consumed_gen.load(Ordering::Acquire);
        if current_gen > consumed_gen {
            return;
        }

        let n = frames.clamp(1, MAX_BRIDGE_BUF);
        let next_gen = current_gen.wrapping_add(1);
        let back_idx = 1 - self.active_read_idx.load(Ordering::Relaxed);
        // `&mut self` proves exclusive access (off-RT setup/teardown window), so
        // `UnsafeCell::get_mut` is the sound safe accessor for the back buffer.
        let back_buf = self.buffers[back_idx].get_mut();
        back_buf.buf_l[..n].fill(0.0);
        back_buf.buf_r[..n].fill(0.0);
        back_buf.n_samples = n as u32;
        back_buf.generation = next_gen;
        back_buf.capture_tsc = 0;
        self.active_read_idx.store(back_idx, Ordering::Release);
        self.generation.store(next_gen, Ordering::Release);
    }
}

impl Default for DspBridge {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: before `buffers` became `UnsafeCell`, `DspBridge` was automatically
// `Sync` (all fields were `Send + Sync`). `UnsafeCell<T>` suppresses the auto
// `Sync`, so this manual impl preserves the pre-existing public trait surface.
// It is sound because the only interior-mutable state is the audio buffers,
// whose mutation is confined to the single-writer/single-reader discipline
// enforced by `DspBridgeWriter`/`DspBridgeReader`; every safe `&self` method on
// `DspBridge` (`drain_dropped_frames`) touches only atomics, so sharing
// `&DspBridge` across threads cannot introduce a data race.
unsafe impl Sync for DspBridge {}

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

/// Write face of `DspBridge` exposed exclusively to the single SPSC capture thread.
///
/// Intentionally neither [`Clone`], [`Copy`], nor [`Sync`]: at most one
/// producer thread may hold and drive a `DspBridgeWriter` for a given
/// `DspBridge` at any time.
///
/// ```compile_fail
/// use neural_amp_modeler_rs::dsp::pipeline::DspBridgeWriter;
/// fn assert_clone<T: Clone>() {}
/// fn main() { assert_clone::<DspBridgeWriter>(); }
/// ```
///
/// ```compile_fail
/// use neural_amp_modeler_rs::dsp::pipeline::DspBridgeWriter;
/// fn assert_sync<T: Sync>() {}
/// fn main() { assert_sync::<DspBridgeWriter>(); }
/// ```
pub struct DspBridgeWriter(std::ptr::NonNull<DspBridge>);

/// SAFETY: `DspBridgeWriter` owns exclusive producer access to a
/// `NonNull<DspBridge>` pointing to a heap-immortal allocation (`Box::leak` in
/// standalone mode, or host/plugin lifecycle memory). Moving it to the single
/// capture thread during setup is sound. `Sync` is intentionally NOT
/// implemented so `&DspBridgeWriter` cannot be shared across threads.
unsafe impl Send for DspBridgeWriter {}

impl DspBridgeWriter {
    /// Creates a `DspBridgeWriter` from a raw pointer to `DspBridge`.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid, non-null, and reference heap-immortal memory
    ///   (leaked `Box`, or host/plugin lifecycle memory that outlives the
    ///   writer).
    /// - **SPSC single-producer uniqueness**: no other live `DspBridgeWriter`
    ///   may exist concurrently for the same `DspBridge` (two writers writing
    ///   to the back-buffer concurrently is immediate UB).
    ///
    /// The null check is a `debug_assert!` (loud in dev builds, compiled out in
    /// release): a null pointer here is a programming error on the
    /// initialization path, not a runtime-recoverable condition.
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
    ///
    /// # Safety
    ///
    /// - When `r` is non-null, it must point to a valid, heap-immortal
    ///   `DspBridge` allocation that outlives the returned writer.
    /// - **SPSC single-producer uniqueness**: the caller must ensure at most
    ///   one `DspBridgeWriter` exists and is used at any time for the
    ///   underlying `DspBridge`.
    #[inline(always)]
    pub unsafe fn from_ref(r: BridgeRef) -> Option<Self> {
        std::ptr::NonNull::new(r.0).map(Self)
    }

    /// Splits a `BridgeRef` into its exclusive SPSC `(writer, reader)` pair.
    /// Returns `None` if `r` is null.
    ///
    /// # Safety
    ///
    /// - When `r` is non-null, it must point to a valid, heap-immortal
    ///   `DspBridge` allocation that outlives both returned endpoints.
    /// - **SPSC uniqueness**: no other `DspBridgeWriter` or `DspBridgeReader`
    ///   may exist concurrently for the underlying `DspBridge`.
    #[inline(always)]
    pub unsafe fn split_spsc(r: BridgeRef) -> Option<(Self, DspBridgeReader)> {
        std::ptr::NonNull::new(r.0).map(|ptr| (Self(ptr), DspBridgeReader(ptr)))
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
        self.write_block_stamped(resamp_out_l, resamp_out_r, n_pw, process_mono, 0);
    }

    /// Same as [`Self::write_block`], additionally publishing the capture-side
    /// start timestamp (`capture_tsc`, nanoseconds, monotonic TSC domain) with
    /// the block so the reader can compute per-generation capture→playback
    /// latency. `0` means "unknown". No allocation, lock, or I/O is added to
    /// the RT path (one extra 8-byte store inside the back-buffer).
    pub fn write_block_stamped(
        &self,
        resamp_out_l: &[f32],
        resamp_out_r: &[f32],
        n_pw: usize,
        process_mono: bool,
        capture_tsc: u64,
    ) {
        let avail_l = resamp_out_l.len();
        let avail = if process_mono {
            avail_l
        } else {
            avail_l.min(resamp_out_r.len())
        };
        let n_bridge = n_pw.min(MAX_BRIDGE_BUF).min(avail);

        // SAFETY: self.0 is NonNull<DspBridge> into heap-immortal memory. The
        // whole-struct pointer is only ever narrowed to field places and raw
        // buffer pointers — no `&DspBridge` or `&mut DspBridge` is materialized
        // concurrently with a write, so the aliasing rules of Stacked Borrows /
        // Tree Borrows hold. `UnsafeCell::get` yields the exclusively-owned
        // back-buffer (1 - active_read_idx); the reader only accesses the
        // complementary front-buffer.
        // Bounds & non-overlap: `n_bridge <= MAX_BRIDGE_BUF` by definition;
        // `buf_l`/`buf_r` have capacity `MAX_BRIDGE_BUF`; `n_bridge <= resamp_out_l.len()`
        // and, in stereo, `n_bridge <= resamp_out_r.len()` guaranteed by the release clamp
        // above (R-15). The input (resampler scratch) and output (bridge back-buffer)
        // regions do not overlap. Release stores synchronize visibility with the reader's
        // Acquire loads.
        unsafe {
            let bridge = self.0.as_ptr();

            let current_gen = (*bridge).generation.load(Ordering::Relaxed);
            let consumed_gen = (*bridge).consumed_gen.load(Ordering::Acquire);
            if current_gen > consumed_gen {
                let _ = (*bridge)
                    .dropped_frames
                    .saturating_inc(Ordering::Relaxed, Ordering::Relaxed);
                return;
            }

            let next_gen = current_gen + 1;
            let back_idx = 1 - (*bridge).active_read_idx.load(Ordering::Relaxed);
            let back_buf = (*bridge).buffers[back_idx].get();

            if n_bridge > 0 {
                core::ptr::copy_nonoverlapping(
                    resamp_out_l.as_ptr(),
                    (*back_buf).buf_l.as_mut_ptr(),
                    n_bridge,
                );
                if process_mono {
                    core::ptr::copy_nonoverlapping(
                        resamp_out_l.as_ptr(),
                        (*back_buf).buf_r.as_mut_ptr(),
                        n_bridge,
                    );
                } else {
                    core::ptr::copy_nonoverlapping(
                        resamp_out_r.as_ptr(),
                        (*back_buf).buf_r.as_mut_ptr(),
                        n_bridge,
                    );
                }
            }
            (*back_buf).n_samples = n_bridge as u32;
            (*back_buf).generation = next_gen;
            (*back_buf).capture_tsc = capture_tsc;

            (*bridge).active_read_idx.store(back_idx, Ordering::Release);
            (*bridge).generation.store(next_gen, Ordering::Release);
        }
    }

    /// Resets the active back-buffer to indicate silence (0 samples).
    ///
    /// Skip-on-overflow: same prevention as `write_block` — if the reader hasn't consumed
    /// the last published generation, the write is skipped and `dropped_frames` is incremented.
    pub fn write_silence(&self) {
        // SAFETY: Same rationale as `write_block`: the pointer is NonNull into
        // heap-immortal memory and is narrowed only to field places / the
        // exclusive back-buffer via `UnsafeCell::get` (no whole-struct reference
        // is materialized). Release stores synchronize with the reader's
        // Acquire loads.
        unsafe {
            let bridge = self.0.as_ptr();

            let current_gen = (*bridge).generation.load(Ordering::Relaxed);
            let consumed_gen = (*bridge).consumed_gen.load(Ordering::Acquire);
            if current_gen > consumed_gen {
                let _ = (*bridge)
                    .dropped_frames
                    .saturating_inc(Ordering::Relaxed, Ordering::Relaxed);
                return;
            }

            let next_gen = current_gen + 1;
            let back_idx = 1 - (*bridge).active_read_idx.load(Ordering::Relaxed);
            let back_buf = (*bridge).buffers[back_idx].get();
            (*back_buf).n_samples = 0;
            (*back_buf).generation = next_gen;
            (*back_buf).capture_tsc = 0;

            (*bridge).active_read_idx.store(back_idx, Ordering::Release);
            (*bridge).generation.store(next_gen, Ordering::Release);
        }
    }
}

/// Read face of `DspBridge` exposed exclusively to the single SPSC playback thread.
///
/// Intentionally neither [`Clone`], [`Copy`], nor [`Sync`]: at most one
/// consumer thread may hold and drive a `DspBridgeReader` for a given
/// `DspBridge` at any time.
///
/// ```compile_fail
/// use neural_amp_modeler_rs::dsp::pipeline::DspBridgeReader;
/// fn assert_clone<T: Clone>() {}
/// fn main() { assert_clone::<DspBridgeReader>(); }
/// ```
///
/// ```compile_fail
/// use neural_amp_modeler_rs::dsp::pipeline::DspBridgeReader;
/// fn assert_sync<T: Sync>() {}
/// fn main() { assert_sync::<DspBridgeReader>(); }
/// ```
pub struct DspBridgeReader(std::ptr::NonNull<DspBridge>);

/// SAFETY: `DspBridgeReader` owns exclusive consumer access to a
/// `NonNull<DspBridge>` pointing to a heap-immortal allocation (same lifecycle
/// as `DspBridgeWriter` — `Box::leak` or host/plugin memory). Moving it to the
/// single playback thread during setup is sound. `Sync` is intentionally NOT
/// implemented so `&DspBridgeReader` cannot be shared across threads.
unsafe impl Send for DspBridgeReader {}

impl DspBridgeReader {
    /// Creates a `DspBridgeReader` from a raw pointer to `DspBridge`.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid, non-null, and reference heap-immortal memory
    ///   (leaked `Box`, or host/plugin lifecycle memory that outlives the
    ///   reader).
    /// - **SPSC single-consumer uniqueness**: no other live `DspBridgeReader`
    ///   may exist concurrently for the same `DspBridge` (two readers mutating
    ///   `consumed_gen` or advancing past the front-buffer simultaneously breaks
    ///   the SPSC protocol).
    ///
    /// The null check is a `debug_assert!` (loud in dev builds, compiled out in
    /// release): a null pointer here is a programming error on the
    /// initialization path, not a runtime-recoverable condition.
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
    ///
    /// # Safety
    ///
    /// - When `r` is non-null, it must point to a valid, heap-immortal
    ///   `DspBridge` allocation that outlives the returned reader.
    /// - **SPSC single-consumer uniqueness**: the caller must ensure at most
    ///   one `DspBridgeReader` exists and is used at any time for the
    ///   underlying `DspBridge`.
    #[inline(always)]
    pub unsafe fn from_ref(r: BridgeRef) -> Option<Self> {
        std::ptr::NonNull::new(r.0).map(Self)
    }

    /// Attempts to read an audio block from the bridge, passing references to L and R channels to a closure.
    ///
    /// Returns `Some(R)` if a new, valid block is available.
    /// Otherwise, returns `None`.
    ///
    /// # Borrow & Post-Return Stability Contract
    ///
    /// The `&[f32]` slices passed to `f` are valid for the duration of `f`; once
    /// `f` returns, `read_block` acknowledges consumption by storing `current_gen`
    /// into `consumed_gen` (`Ordering::Release`). Under the single-writer /
    /// single-reader invariant, the writer can subsequently publish at most one
    /// block into the complementary back-buffer before blocking on
    /// `current_gen > consumed_gen`, so the previously read front-buffer stays
    /// unmodified until either the next `read_block` call or an exclusive
    /// teardown (`reset_to_silence` / `prime_silence`) after both RT callbacks
    /// are quiescent. Callers that defer copying past `read_block`'s return
    /// rely on non-reentrancy of the single reader and quiescent teardown ordering.
    pub fn read_block<F, R>(&self, last_bridge_gen: &mut u64, f: F) -> Option<R>
    where
        F: FnOnce(&[f32], &[f32]) -> R,
    {
        self.read_block_stamped(last_bridge_gen, f)
            .map(|(_, _, result)| result)
    }

    /// Same as [`Self::read_block`], additionally returning the delivered
    /// generation and the capture-side start timestamp that produced it
    /// (`capture_tsc`, `0` when unknown). The tuple is
    /// `(generation, capture_tsc, R)`. RT-safe: the stamp is copied out of the
    /// front-buffer alongside the samples, no extra atomic and no allocation.
    pub fn read_block_stamped<F, R>(&self, last_bridge_gen: &mut u64, f: F) -> Option<(u64, u64, R)>
    where
        F: FnOnce(&[f32], &[f32]) -> R,
    {
        // SAFETY: self.0 is NonNull<DspBridge> into heap-immortal memory. The
        // pointer is narrowed only to field places and the front-buffer raw
        // pointer (`UnsafeCell::get`) — no whole-struct `&DspBridge` is
        // materialized, so access concurrent with the single writer stays within
        // the Stacked/Tree Borrows aliasing rules. The front-buffer
        // (active_read_idx) is read exclusively here; the writer mutates only the
        // complementary back-buffer. Acquire loads synchronize with the writer's
        // Release stores.
        unsafe {
            let bridge = self.0.as_ptr();
            let current_gen = (*bridge).generation.load(Ordering::Acquire);
            if current_gen == *last_bridge_gen {
                return None;
            }
            let read_idx = (*bridge).active_read_idx.load(Ordering::Acquire);
            let post_gen = (*bridge).generation.load(Ordering::Acquire);
            if current_gen != post_gen {
                return None;
            }
            let front_buf = (*bridge).buffers[read_idx].get();
            if (*front_buf).generation != current_gen {
                return None;
            }
            let n_samples = (*front_buf).n_samples as usize;
            if n_samples == 0 || n_samples > MAX_BRIDGE_BUF {
                *last_bridge_gen = current_gen;
                (*bridge).consumed_gen.store(current_gen, Ordering::Release);
                return None;
            }

            let result = f(
                &(&(*front_buf).buf_l)[..n_samples],
                &(&(*front_buf).buf_r)[..n_samples],
            );

            let capture_tsc = (*front_buf).capture_tsc;
            *last_bridge_gen = current_gen;
            (*bridge).consumed_gen.store(current_gen, Ordering::Release);

            Some((current_gen, capture_tsc, result))
        }
    }
}

#[cfg(test)]
#[path = "bridge_test.rs"]
mod bridge_test;
