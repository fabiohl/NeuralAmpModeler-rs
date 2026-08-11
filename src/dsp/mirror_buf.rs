// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
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
/// Struct layout (16 bytes): optimized for cache — the hot-path Deref
/// accesses only the first two fields.
pub struct MirroredBuffer<T> {
    ptr: *mut T,
    size_elements: usize,
    _marker: PhantomData<T>,
}

/// Sets whether the next `MirroredBuffer` creation calls should simulate
/// virtual memory allocation failure.
pub fn set_simulate_fail(fail: bool) {
    SIMULATE_FAIL.with(|f| f.set(fail));
}

impl<T> MirroredBuffer<T> {
    /// Returns the physical buffer size (before mirroring) in elements.
    pub fn size(&self) -> usize {
        self.size_elements
    }
}

impl<T> std::fmt::Debug for MirroredBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MirroredBuffer")
            .field("ptr", &self.ptr)
            .field("size_elements", &self.size_elements)
            .field("capacity_virtual", &(self.size_elements * 2))
            .finish()
    }
}

impl<T> Deref for MirroredBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // SAFETY: self.ptr points to a valid virtual mapping of size_elements*2 elements.
        // The allocation reserves contiguous virtual space for the two mirrored halves,
        // mapping the same physical pages twice. ptr is initialized by mmap/ftruncate
        // (or std heap fallback validated by the alloc module) and remains valid for the
        // full virtual extent until Drop.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size_elements * 2) }
    }
}

impl<T> DerefMut for MirroredBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: Same invariants as Deref: ptr covers size_elements*2 valid virtual elements
        // (two mirrored halves mapping the same physical pages). The &mut self reference
        // guarantees exclusive access, so no aliasing of the underlying memory occurs.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_elements * 2) }
    }
}

impl<T> Drop for MirroredBuffer<T> {
    fn drop(&mut self) {
        let element_size = std::mem::size_of::<T>();
        let size_bytes = self.size_elements * element_size;
        // SAFETY: self.ptr was obtained via mmap (or std heap via the alloc module).
        // The virtual mapping spans size_bytes*2 (two mirrored halves). munmap receives
        // the original base address and the full virtual extent, matching the allocation
        // interface. Drop consumes self, so this is the last use of the pointer.
        unsafe {
            munmap(self.ptr as *mut c_void, size_bytes * 2);
        }
    }
}

impl<T: Clone> MirroredBuffer<T> {
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
impl<T: Clone> Clone for MirroredBuffer<T> {
    #[cold]
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("MirroredBuffer::clone: allocation failed (use try_clone for fallible path)")
    }
}

// SAFETY: ptr points to an exclusive virtual mapping (mmap/MAP_PRIVATE or heap) with no
// shared physical aliasing to other processes. Moving the struct across threads is sound
// because the underlying memory is not shared with other MirroredBuffer instances.
unsafe impl<T: Send> Send for MirroredBuffer<T> {}
// SAFETY: T: Sync ensures that shared references to T elements are sound across threads.
// MirroredBuffer's internal ptr is only accessed through Deref/DerefMut which follow Rust's
// borrow rules. The exclusive virtual mapping prevents races from external writers.
unsafe impl<T: Sync> Sync for MirroredBuffer<T> {}

#[cfg(all(test, target_os = "linux"))]
#[path = "mirror_buf_test.rs"]
mod mirror_buf_test;
