// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Low-level virtual memory mapping invariants are strictly encapsulated within MirroredBuffer.
#![warn(clippy::undocumented_unsafe_blocks)]

//! # Mirrored Buffer (MirroredBuffer) via Mirrored Memory Mapping
//!
//! The `MirroredBuffer` is an advanced memory management technique that solves
//! the classic "contiguity break" problem in circular/delay-line buffers.
//!
//! ## The Problem: The Buffer Boundary
//! In traditional circular buffers, upon reaching the end of allocated space, the pointer
//! wraps back to the start. If a DSP algorithm (such as a Convolution or FFT) needs to read
//! a window of 1024 samples, but the pointer is only 500 samples from the end, the developer
//! would have to handle the read in two parts or perform an expensive copy (`copy_within`).
//! This introduces complex logic (`if/else`) and hurts performance in the "hot-path".
//!
//! ## The Solution: The Mirroring "Trick"
//! This structure leverages the processor's Memory Management Unit (MMU) features
//! to map the **same physical memory block** twice consecutively in the virtual
//! address space:
//!
//! ```text
//! Virtual Space: [ Physical Block (Page 0..N) ] [ Physical Block (Page 0..N) ]
//!                 ^                             ^
//!                 |                             |
//!          Buffer Start                    Mirror of the Start
//! ```
//!
//! Thanks to this mapping, any access that "goes past" the end of the first block will
//! automatically fall into the start of the second block — which is, physically, the buffer
//! start itself.
//!
//! ## Benefits for Real-Time Audio
//! 1. **Linear Access**: Algorithms can read contiguous windows of any size (up to the full buffer size) without worrying about "wrap".
//! 2. **Zero-Copy**: Eliminates the need to copy data to temporary buffers to linearize them.
//! 3. **SIMD Performance**: Enables vector instructions (AVX/SSE) to process data across the buffer boundary without logic interruptions.
//! 4. **Branch-Free**: Removes modulo (`%`) operations and `if` conditions, optimizing processor branch prediction.
//!
//! ## Huge Page Support
//! Attempts 2 MB huge pages (MAP_HUGETLB / MFD_HUGETLB) for the mirror buffer to reduce
//! TLB pressure in the DSP hot-path. Falls back to regular pages + `madvise(MADV_HUGEPAGE)` + `MADV_COLLAPSE` (synchronous THP).
//! Status is tracked via `MIRROR_BUF_HUGEPAGE_STATE` global to avoid inflating the
//! struct layout (which would affect hot-path cache performance).
use libc::{c_void, munmap};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::AtomicU8;

mod alloc;

thread_local! {
    pub(crate) static SIMULATE_FAIL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Huge-page state constants for `MIRROR_BUF_HUGEPAGE_STATE`.
pub const HUGEPAGE_STATE_STANDARD: u8 = 0;
/// State indicating Transparent Huge Pages (madvise MADV_HUGEPAGE + MADV_COLLAPSE) were used.
pub const HUGEPAGE_STATE_THP: u8 = 1;
/// State indicating explicit 2 MB HugeTLB pages (MAP_HUGETLB) are active.
pub const HUGEPAGE_STATE_HUGETLB: u8 = 2;

/// Global flag tracking the highest huge-page mode achieved by any MirroredBuffer.
/// 0 = Standard, 1 = THP (madvise), 2 = HugeTLB (explicit 2 MB pages).
static MIRROR_BUF_HUGEPAGE_STATE: AtomicU8 = AtomicU8::new(HUGEPAGE_STATE_STANDARD);

/// Synchronizes the mirror buffer huge-page status to the RT status flag.
/// Call once during main-thread initialization.
pub fn sync_huge_page_flag(rt_status: &crate::common::spsc::RtStatusFlags) {
    let state = MIRROR_BUF_HUGEPAGE_STATE.load(std::sync::atomic::Ordering::Relaxed);
    match state {
        HUGEPAGE_STATE_HUGETLB => {
            rt_status.set_flag(crate::common::spsc::RT_STATUS_HUGEPAGE_OK);
        }
        HUGEPAGE_STATE_THP => {
            rt_status.set_flag(crate::common::spsc::RT_STATUS_THP_ACTIVE);
        }
        _ => {}
    }
}

/// Tracks whether huge pages were successfully activated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorHugePageStatus {
    /// Explicit 2 MB huge pages via MAP_HUGETLB | MFD_HUGETLB.
    Explicit2MB,
    /// Transparent huge pages via madvise(MADV_HUGEPAGE) + MADV_COLLAPSE (synchronous).
    Transparent,
    /// Standard 4 KB pages (fallback).
    Standard,
}

/// Returns whether any mirror buffer has huge pages active (THP or HugeTLB).
pub fn is_huge_page_active() -> bool {
    let state = MIRROR_BUF_HUGEPAGE_STATE.load(std::sync::atomic::Ordering::Relaxed);
    state == HUGEPAGE_STATE_THP || state == HUGEPAGE_STATE_HUGETLB
}

/// Returns the current `MirrorHugePageStatus` for diagnostics.
pub fn huge_page_status() -> MirrorHugePageStatus {
    match MIRROR_BUF_HUGEPAGE_STATE.load(std::sync::atomic::Ordering::Relaxed) {
        HUGEPAGE_STATE_HUGETLB => MirrorHugePageStatus::Explicit2MB,
        HUGEPAGE_STATE_THP => MirrorHugePageStatus::Transparent,
        _ => MirrorHugePageStatus::Standard,
    }
}

/// A Mirrored Buffer that uses mirrored memory mapping.
///
/// This structure maps the same physical content twice consecutively in the virtual
/// address space. This allows accesses that would "cross" the end of the buffer
/// to be performed linearly and contiguously, eliminating the need for "rewind"
/// or "copy_within" operations in the DSP hot-path.
///
/// # Half-aliasing contract
///
/// The virtual extent exposed by `Deref`/`DerefMut` is `2N` elements (`N =
/// size()`), but the **two halves are NOT independent storage**: they map the
/// same physical pages twice (`MAP_SHARED` on the same fd/offset in
/// `mirror_buf/alloc.rs`). Consequences:
///
/// - A write to `buf[i]` for `i < N` is observable at `buf[N + i]`, and vice
///   versa — both addresses alias the same physical location. The mirror exists
///   precisely so linear reads/writes that cross the `N` boundary behave as a
///   wrap-around on the same physical buffer.
/// - The second half exists so a window that crosses the end of the first half
///   (a ring wrap) can be read — and, in the ring write pattern, written —
///   linearly through a single `&[T]`/`&mut [T]` borrow. Such a wrap write is
///   *intended* to mutate the physical start of the buffer.
/// - Do **not** treat the two halves as disjoint regions you can own
///   simultaneously. `split_at_mut(size())` yields two `&mut [T]` slices that
///   are disjoint in *virtual address space* but physically alias the same
///   pages; using both as if they were independent buffers (e.g. writing
///   different data through each) is not a sound way to create two independent
///   `&mut` views of the same physical data. The supported pattern is a single
///   mutable borrow over the full `2N` view, or a borrow confined to one side
///   of a linear operation that crosses the boundary as a wrap.
/// - Element reads/writes of `T` are `Copy` (f32/f64/i32/…) in every in-crate
///   instantiation, so no `Drop` runs through the aliased mapping; `T` must
///   remain `Copy`-like (`Deref` requires no `Drop` of an aliased slot).
///
pub(crate) mod private {
    /// Private sealing trait preventing external arbitrary implementations of [`super::MirrorSafe`].
    pub trait Sealed {}
}

/// Marker trait for types that are safe to inhabit a [`MirroredBuffer`].
///
/// # Safety
/// Implementing types must satisfy all sound virtual memory mirroring invariants:
/// 1. An all-zero byte representation (`[0u8; size_of::<Self>()]`) is a valid and sound value.
/// 2. `Self: Copy`, ensuring bitwise duplicability without running custom `Drop` glue.
/// 3. `Self` does not rely on address identity or address-derived indirection.
pub unsafe trait MirrorSafe: Copy + private::Sealed {}

macro_rules! impl_mirror_safe {
    ($($t:ty),* $(,)?) => {
        $(
            impl private::Sealed for $t {}
            // SAFETY: Primitive numeric types are Copy, zero-bit patterns are sound representations, and types have no address identity dependencies.
            unsafe impl MirrorSafe for $t {}
        )*
    };
}

impl_mirror_safe!(f32, f64, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

/// A zero-copy circular ring buffer implemented via dual contiguous virtual memory mappings.
///
/// # Element type (`T`) contract and `MirrorSafe` trait
///
/// The mirror mapping makes every physical slot reachable through two virtual
/// addresses, and the storage is zero-filled at creation. Soundness of
/// `Deref`/`DerefMut` — which reinterpret the mapping as a `2N`-element
/// `[T]` — requires every `T` used with this buffer to satisfy:
///
/// 1. **All-bits-zero is a valid `T`.** Storage is created zero-filled
///    (fresh mmap/ftruncate pages), so an all-zero bit pattern must be a
///    valid value of `T` — types with niches forbidding it (e.g.
///    `NonZeroU32`, references `&T`, `NonNull<u8>`) are forbidden.
/// 2. **No `Drop` glue.** A slot is reachable and overwritable through two
///    virtual addresses; memory is unmapped without running element drops.
///    `Self: Copy` guarantees absence of custom destructors.
/// 3. **Aliasing tolerance under `&mut`.** Writes through the mirrored view
///    intentionally create aliased observations of the same physical bytes;
///    `T` must not rely on reference identity or unique addresses.
/// 4. **No address-derived indirection.** `T` must not derive meaning from its
///    own virtual memory address.
///
/// These requirements are statically enforced by the sealed [`MirrorSafe`] trait.
///
/// Struct layout (16 bytes): optimized for cache — the hot-path Deref
/// accesses only the first two fields.
///
/// # Examples
///
/// ```compile_fail
/// use std::num::NonZeroU32;
/// use neural_amp_modeler_rs::dsp::mirror_buf::MirroredBuffer;
///
/// // NonZeroU32 does not implement MirrorSafe (all-zero is invalid), preventing unsound instantiation:
/// let _ = MirroredBuffer::<NonZeroU32>::new(1);
/// ```
///
/// ```compile_fail
/// use neural_amp_modeler_rs::dsp::mirror_buf::MirroredBuffer;
///
/// // Zero-Sized Types () do not implement MirrorSafe:
/// let _ = MirroredBuffer::<()>::new(1024);
/// ```
pub struct MirroredBuffer<T: MirrorSafe> {
    ptr: *mut T,
    size_elements: usize,
    _marker: PhantomData<T>,
}

/// Sets whether the next `MirroredBuffer` creation calls should simulate
/// virtual memory allocation failure.
#[cfg(any(test, feature = "testing"))]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
pub fn set_simulate_fail(fail: bool) {
    SIMULATE_FAIL.with(|f| f.set(fail));
}

impl<T: MirrorSafe> MirroredBuffer<T> {
    /// Returns the physical buffer size (before mirroring) in elements.
    pub fn size(&self) -> usize {
        self.size_elements
    }
}

impl<T: MirrorSafe> std::fmt::Debug for MirroredBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MirroredBuffer")
            .field("ptr", &self.ptr)
            .field("size_elements", &self.size_elements)
            .field("capacity_virtual", &(self.size_elements * 2))
            .finish()
    }
}

impl<T: MirrorSafe> Deref for MirroredBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // SAFETY: self.ptr points to a valid virtual mapping of size_elements*2 elements.
        // The allocation reserves contiguous virtual space for the two mirrored halves,
        // mapping the same physical pages twice. Page alignment (>= 4096 bytes) satisfies
        // align_of::<T>() <= 64 guaranteed at compile time by const assertions in constructors,
        // and align_of::<T>() <= page_size guaranteed at construction by the runtime assert
        // in mirror_buf/alloc.rs (A7 / R-5). The underlying physical pages are zero-filled by the kernel,
        // which is guaranteed to be a valid representation of T by the sealed T: MirrorSafe bound.
        // The two halves physically alias, which is sound for shared reads and Copy elements.
        // ptr is initialized by mmap/ftruncate and remains valid for the full virtual extent until Drop.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size_elements * 2) }
    }
}

impl<T: MirrorSafe> DerefMut for MirroredBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: Same virtual mapping invariants as Deref: ptr covers size_elements*2 valid
        // elements of type T: MirrorSafe. Page alignment satisfies align_of::<T>() <= 64 and
        // align_of::<T>() <= page_size.
        // The &mut self borrow guarantees exclusive access to this MirroredBuffer instance
        // through a single &mut [T] over the 2N virtual extent.
        // Note on half-aliasing: the two virtual halves map the same physical pages. Callers
        // must not attempt to split this slice (e.g. via split_at_mut) into two simultaneously
        // active mutable references to the same physical memory. A wrap write crossing the N
        // boundary through this single continuous slice is the supported pattern and intentionally
        // aliases the physical start of the buffer.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_elements * 2) }
    }
}

impl<T: MirrorSafe> Drop for MirroredBuffer<T> {
    fn drop(&mut self) {
        let element_size = std::mem::size_of::<T>();
        let size_bytes = self.size_elements * element_size;
        // SAFETY: self.ptr was obtained via double-mapping an anonymous file descriptor
        // (memfd with MAP_SHARED) spanning size_bytes * 2 (two contiguous mirrored halves).
        // munmap receives the original base address and the full virtual extent (size_bytes * 2),
        // cleanly unmapping both virtual mirrors. Drop consumes self by value, ensuring
        // this is the exclusive and final use of the pointer. Individual elements are not
        // dropped, which is sound because T: MirrorSafe bounds T to Copy types with no Drop glue.
        unsafe {
            munmap(self.ptr as *mut c_void, size_bytes * 2);
        }
    }
}

impl<T: MirrorSafe> MirroredBuffer<T> {
    /// Fallible clone — returns `Err` on allocation failure instead of panicking.
    ///
    /// Preferred over `Clone::clone()` in host activation paths where panic
    /// would cross the FFI boundary.
    pub fn try_clone(&self) -> std::io::Result<Self> {
        let mut new_buf = Self::new(self.size_elements)?;
        new_buf[..self.size_elements].clone_from_slice(&self[..self.size_elements]);
        Ok(new_buf)
    }
}

/// ## Architectural decision: `impl Clone` for `MirroredBuffer<T>`
///
/// `Clone` is **intentionally infallible** (panics on OOM) to enable `#[derive(Clone)]`
/// on container structs (e.g. `NamModelData`, `ManagedConvolutionEngine`) that own a
/// `MirroredBuffer<T>`. Rust's derive macro requires all fields to implement `Clone`,
/// and an infallible `Clone` satisfies that constraint.
///
/// For **fallible** duplication in host activation paths, use `try_clone()` instead.
/// Panicking across the FFI boundary is undefined behavior, so `Clone::clone()` must
/// only be called in initialization/control paths where allocation failures are fatal.
impl<T: MirrorSafe> Clone for MirroredBuffer<T> {
    #[cold]
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("MirroredBuffer::clone: allocation failed (use try_clone for fallible path)")
    }
}

// SAFETY: ptr points to an exclusive process-local virtual mapping created via an
// unlinked anonymous memfd (MAP_SHARED across the two adjacent mirror halves of this
// buffer only, with no external process sharing or fd leaks). Moving the struct
// across threads is sound because the underlying memory is uniquely owned by this
// MirroredBuffer instance, and T: MirrorSafe + Send.
unsafe impl<T: MirrorSafe + Send> Send for MirroredBuffer<T> {}

// SAFETY: T: MirrorSafe + Sync ensures that shared references to T elements are sound across
// threads. MirroredBuffer's internal ptr is accessed exclusively through Deref/DerefMut which follow
// standard Rust borrowing rules (&self vs &mut self). The exclusive process-local virtual
// mapping prevents data races from external writers.
unsafe impl<T: MirrorSafe + Sync> Sync for MirroredBuffer<T> {}

#[cfg(all(test, target_os = "linux"))]
#[path = "mirror_buf_test.rs"]
mod mirror_buf_test;
