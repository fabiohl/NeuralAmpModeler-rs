// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! ## Final drain policy
//!
//! The `GcOverflowBuffer`, the GC consumer, and the RT parking lot must be
//! drained by a non-real-time control thread before the processing pipeline is
//! destroyed. Destruction of real-time shared state cannot perform that drain
//! because the consumer belongs to the control-plane lifecycle.
//!
//! [`drain_gc_channels`] is the canonical drain path: it drains the SPSC
//! channel, the overflow buffer, and the RT parking lot. A host integration
//! must call it periodically during control-plane housekeeping and once more
//! during orderly teardown, after real-time producers have stopped. Any
//! residual item at abrupt process termination is an integration-level
//! shutdown concern, not permission to drop heap-owned values on the audio
//! thread.

use rtrb::Consumer;
use std::sync::atomic::{AtomicU64, Ordering};

/// Unified declaration of GcItem variants and their dispatch methods.
///
/// Each entry maps a variant name to its inner type and a unique `type_id`.
/// The macro generates the `GcItem` enum plus the three dispatch methods
/// (`type_id`, `from_raw_parts`, `into_packed`) from this single source
/// of truth.  To add a new GC item type, only this invocation needs updating —
/// no more duplicated match blocks.
macro_rules! define_gc_item {
    (
        $(
            $(#[$attr:meta])*
            $variant:ident($inner:ty) = $id:literal
        ),+ $(,)?
    ) => {
        /// Represents an item that should be safely disposed outside the audio thread.
        /// Dropping these items may involve heavy memory deallocations.
        #[expect(missing_docs, reason = "Docs for generated enum variants are provided by the macro invocation site — repeating would duplicate the doc contract")]
        pub enum GcItem {
            $(
                $(#[$attr])*
                $variant(Box<$inner>),
            )*
        }

        impl std::fmt::Debug for GcItem {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(
                        $(#[$attr])*
                        GcItem::$variant(_) => f.debug_tuple(stringify!($variant)).finish(),
                    )*
                }
            }
        }

        impl GcItem {
            /// Returns the type ID for the overflow buffer.
            fn type_id(&self) -> u8 {
                match self {
                    $(
                        $(#[$attr])*
                        GcItem::$variant(_) => $id,
                    )*
                }
            }

            /// Reconstructs a GcItem from a raw pointer and a type ID.
            ///
            /// Returns `None` if the type ID is unknown. In that case, the caller
            /// must intentionally leak the pointer to avoid UB from `Box::from_raw`
            /// with a mismatched type.
            ///
            /// # Safety
            /// The pointer must have been generated via `Box::into_raw` of an object
            /// of the corresponding type, validated by the caller against type_id.
            ///
            /// ## Dispatch protocol
            ///
            /// Each variant is assigned a fixed `type_id`.
            /// The ID is validated before `Box::from_raw` — an unknown ID results in
            /// `None`, and the caller must leak the raw pointer. This ensures that
            /// even if the packed u64 is corrupted in the overflow buffer, the RT
            /// thread never calls `Box::from_raw` with a mismatched layout.
            unsafe fn from_raw_parts(ptr: *mut std::ffi::c_void, type_id: u8) -> Option<Self> {
                match type_id {
                    $(
                        $(#[$attr])*
                        $id => Some(GcItem::$variant(unsafe {
                            Box::from_raw(ptr as *mut $inner)
                        })),
                    )*
                    _ => None,
                }
            }

            /// Converts this GcItem into a packed 64-bit representation for the
            /// overflow buffer.
            ///
            /// Layout (little-endian):
            ///   Bits 0-55:  user-space pointer (≤ 56 bits)
            ///   Bits 56-63: type ID
            ///
            /// ## Platform dependency
            ///
            /// This packing scheme relies on **x86-64 canonical addressing**: under
            /// **4-level paging** (LA48, 48-bit) or **5-level paging** (LA57, 57-bit),
            /// the Linux kernel maps the user-space virtual address range so that the
            /// top byte (bits 56–63) of any `malloc`/`Box` pointer on the heap is
            /// guaranteed to be zero.  Consequently, all live heap pointers fit
            /// within 56 bits, and repurposing bits 56–63 for the type ID is safe.
            ///
            /// Should a future x86-64 extension widen the canonical address range
            /// beyond 57 bits or a non-Linux kernel violate this convention, the
            /// `debug_assert!` below will fire and the packing scheme must be revised.
            pub(crate) fn into_packed(self) -> u64 {
                let type_id = self.type_id();
                let ptr = match self {
                    $(
                        $(#[$attr])*
                        GcItem::$variant(b) => Box::into_raw(b) as *mut std::ffi::c_void,
                    )*
                };

                debug_assert!(
                    (ptr as u64) < (1u64 << 56),
                    "GC pointer 0x{:016X} exceeds 56 bits — packing scheme is unsafe \
                     on this system (requires LA57 with 57-bit canonical addresses or less).",
                    ptr as u64
                );

                ((type_id as u64) << 56) | (ptr as u64 & 0x00FF_FFFF_FFFF_FFFF)
            }
        }
    };
}

define_gc_item! {
    Model(crate::models::StaticModel) = 1,
    Resampler(crate::dsp::resampler::NamResampler) = 2,
    CabSimIr(crate::dsp::cabsim::loader::CabSimIr) = 3,
    CabConvAdapter(crate::dsp::cabsim::adapter::CabSimAdapter) = 4,
    Oversample(crate::dsp::oversample::OversampleEngine) = 5,
    #[cfg(test)]
    Test(std::sync::Arc<std::sync::atomic::AtomicU32>) = 255,
}

impl GcItem {
    /// Reconstructs a GcItem from a packed 64-bit value.
    ///
    /// The complement of [`into_packed`](Self::into_packed): unmarshals the
    /// raw pointer (bits 0–55) and type ID (bits 56–63) and dispatches via
    /// [`from_raw_parts`](Self::from_raw_parts).
    ///
    /// Returns `None` if the packed value is zero (empty slot) or the type
    /// ID is unknown. On unknown type ID, the caller must leak the pointer
    /// to avoid UB.
    ///
    /// ## Platform dependency
    ///
    /// See [`into_packed`](Self::into_packed) — the same 56-bit canonical
    /// address assumption applies here, as the pointer is extracted via
    /// `packed & 0x00FF_FFFF_FFFF_FFFF`.  If that assumption does not hold,
    /// the pointer will be silently truncated (caught by the `debug_assert!`
    /// in `into_packed`).
    ///
    /// # Safety
    /// The pointer embedded in `packed` must be valid for the type encoded
    /// in bits 56–63.  A corrupted type ID leads to a `None` return and a
    /// controlled leak by the caller.
    unsafe fn from_packed(packed: u64) -> Option<Self> {
        if packed == 0 {
            return None;
        }
        let type_id = ((packed >> 56) & 0xFF) as u8;
        if type_id == 0 {
            return None;
        }
        let ptr = (packed & 0x00FF_FFFF_FFFF_FFFF) as *mut std::ffi::c_void;
        unsafe { Self::from_raw_parts(ptr, type_id) }
    }
}

/// Circular "final parking" buffer for GC items.
///
/// Each slot is a single `AtomicU64` packing type_id (bits 56-63) and
/// pointer (bits 0-55) into one atomic word, eliminating the torn-read
/// window that existed when type and pointer were swapped independently.
///
/// Used when the main SPSC channel and the thread's parking lot are both full.
/// Ensures no object is dropped on the audio thread at the cost of a controlled leak
/// or overwrite in extreme stress scenarios.
pub struct GcOverflowBuffer {
    pub(crate) slots: Box<[AtomicU64]>,
    write_idx: AtomicU64,
}

impl GcOverflowBuffer {
    #[cold]
    /// Creates a new overflow buffer with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "GcOverflowBuffer: capacity must be greater than 0 to avoid division by zero panic."
        );
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(AtomicU64::new(0));
        }
        Self {
            slots: slots.into_boxed_slice(),
            write_idx: AtomicU64::new(0),
        }
    }

    /// Attempts to park an item in the overflow buffer without RT allocations.
    ///
    /// If the buffer is full, the oldest item is overwritten (leak).
    /// Returns `true` if an overwrite (controlled leak) occurred.
    ///
    /// ## Concurrency note
    ///
    /// `write_idx` uses `Relaxed` because this is an SPSC overflow buffer:
    /// only the producer (RT thread) writes/reads this index. The drain
    /// path (main thread) never touches `write_idx` — it sweeps all slots
    /// via `swap(AcqRel)` on each slot. That `AcqRel` pair (producer at
    /// `swap(packed, AcqRel)` in `push`, consumer at `swap(0, AcqRel)` in
    /// `drain`) provides the happens-before edge for the item payload.
    ///
    /// The window between `fetch_add` and `swap` means a slot may be
    /// collected one cycle later than its write. This is benign: no
    /// double-free or leak can occur because the consumer always replaces
    /// the slot value with 0.
    pub fn push(&self, item: GcItem) -> bool {
        let packed = item.into_packed();

        let len = self.slots.len() as u64;
        let idx = (self.write_idx.fetch_add(1, Ordering::Relaxed) % len) as usize; // Relaxed safe: SPSC — only producer touches write_idx; happens-before via slot swap(AcqRel)

        let old = self.slots[idx].swap(packed, Ordering::AcqRel);

        old != 0
    }

    /// Drains all accumulated items from the overflow buffer.
    /// Should be called periodically by the main thread (Cold Path).
    ///
    /// On corrupted slots (unknown type_id), the pointer is intentionally leaked
    /// and `RT_STATUS_GC_CORRUPTED` is set via the `rt_status` parameter.
    /// Returns ownership of the drained items so the caller can drop them.
    pub fn drain(&self, rt_status: &super::RtStatusFlags) -> Vec<GcItem> {
        let mut items = Vec::with_capacity(self.slots.len());
        for i in 0..self.slots.len() {
            let packed = self.slots[i].swap(0, Ordering::AcqRel);
            if packed == 0 {
                continue;
            }
            unsafe {
                match GcItem::from_packed(packed) {
                    Some(item) => items.push(item),
                    None => {
                        rt_status.set_flag(super::RT_STATUS_GC_CORRUPTED);
                    }
                }
            }
        }
        items
    }
}

impl Default for GcOverflowBuffer {
    fn default() -> Self {
        Self::new(64)
    }
}

/// RT-safe GC cascade: tries the SPSC channel, then a 16-slot parking lot,
/// then the overflow buffer as a last resort.
///
/// # Tier progression and status flags
///
/// - **Tier 1** (SPSC): no flag set — normal fast-path drain.
/// - **Tier 2** (parking lot): no flag set — auxiliary drain.
/// - **Tier 3** (overflow buffer): `RT_STATUS_GC_TIER3` is always set,
///   indicating the cascade reached the overflow buffer.
/// - `RT_STATUS_GC_OVERFLOW` is **only** set on an actual overwrite/leak
///   (i.e. the slot was already occupied), not on every Tier 3 entry.
///
/// Before parking a new item, the cascade opportunistically flushes any item
/// already parked in `parking_lot` back to the SPSC channel while there is
/// free capacity. This keeps the parking lot empty across successive audio
/// cycles so parked items reach the off-RT [`drain_gc_channels`] instead of
/// lingering in real-time state. On exhaustion, the item cascades into the
/// overflow buffer — it is **never** dropped on the audio thread.
///
/// # Parameters
/// - `item`: The `GcItem` to dispose of outside the RT thread.
/// - `gc_producer`: SPSC GC channel.
/// - `parking_lot`: 16-slot fallback array shared across all drainers.
/// - `gc_overflow`: Overflow ring buffer.
/// - `rt_status`: Status flags (sets `RT_STATUS_GC_TIER3` and
///   `RT_STATUS_GC_OVERFLOW` on actual overwrite).
#[inline(always)]
pub fn gc_cascade(
    mut item: Option<GcItem>,
    gc_producer: &mut rtrb::Producer<GcItem>,
    parking_lot: &mut [Option<GcItem>; 16],
    gc_overflow: &GcOverflowBuffer,
    rt_status: &super::RtStatusFlags,
) {
    // 0. Opportunistically flush items parked in a previous cycle back to the
    //    SPSC channel while it has free capacity, so the parking lot tends to
    //    empty every cycle and items reach the off-RT drain.
    for slot in parking_lot.iter_mut() {
        let Some(parked) = slot.take() else { continue };
        match gc_producer.push(parked) {
            Ok(()) => {}
            Err(rtrb::PushError::Full(returned)) => {
                *slot = Some(returned);
                break;
            }
        }
    }

    if let Some(i) = item.take() {
        if let Err(rtrb::PushError::Full(returned)) = gc_producer.push(i) {
            item = Some(returned);
        } else {
            return;
        }
    }

    if let Some(i) = item.take() {
        let mut i_opt = Some(i);
        for slot in parking_lot.iter_mut() {
            if slot.is_none() {
                *slot = i_opt.take();
                return;
            }
        }
        item = i_opt;
    }

    if let Some(i) = item.take() {
        rt_status.set_flag(super::RT_STATUS_GC_TIER3);
        if gc_overflow.push(i) {
            rt_status.set_flag(super::RT_STATUS_GC_OVERFLOW);
        }
    }
}

/// Drains the RT parking lot, dropping every parked item on the calling
/// (non-real-time) thread.
///
/// The parking lot is a 16-slot fixed-size contingency array owned by the
/// real-time producer. It is drained off-RT only after the producer has
/// stopped, so no heap object is ever dropped on the audio thread.
///
/// Returns the number of items dropped.
pub fn drain_parking_lot(parking_lot: &mut [Option<GcItem>; 16]) -> usize {
    let mut count = 0;
    for slot in parking_lot.iter_mut() {
        if let Some(item) = slot.take() {
            drop(item);
            count += 1;
        }
    }
    count
}

/// Aggressively drains the Garbage Collection channels to free memory.
///
/// This function should be called periodically by the main thread (CLI/UI)
/// or by the host event loop. It executes `drop()`
/// on obsolete objects (models, resamplers) outside the RT thread.
///
/// It drains, in order: the SPSC channel, the overflow buffer, and the RT
/// parking lot. The `parking_lot` must be handed over by the real-time side
/// after its producers have stopped (single-owner handoff), so that parked
/// items are also released off-RT.
///
/// Returns the total number of GC items dropped during this call.
pub fn drain_gc_channels(
    gc_consumer: &mut Consumer<GcItem>,
    gc_overflow: &GcOverflowBuffer,
    parking_lot: &mut [Option<GcItem>; 16],
    rt_status: &super::RtStatusFlags,
) -> usize {
    let mut count = 0;
    // 1. Drain the main SPSC channel (Drop-Delegation)
    while let Ok(item) = gc_consumer.pop() {
        drop(item);
        count += 1;
    }

    // 2. Drain the overflow buffer (overwrite ring buffer)
    for item in gc_overflow.drain(rt_status) {
        drop(item);
        count += 1;
    }

    // 3. Drain the RT parking lot (16 slots of last-resort contingency)
    count += drain_parking_lot(parking_lot);
    count
}
