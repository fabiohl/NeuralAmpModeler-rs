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
//! ## Platform Backends & Huge Page Support
//! * **POSIX (`posix.rs`):** Uses `memfd_create` (Linux) or unlinked POSIX shared memory
//!   (`shm_open` on macOS) with double `mmap`. On Linux, attempts 2 MB huge pages
//!   (`MAP_HUGETLB` / `MFD_HUGETLB`) for the mirror buffer to reduce TLB pressure in the
//!   DSP hot-path, falling back to regular pages + `madvise(MADV_HUGEPAGE)` +
//!   `MADV_COLLAPSE` (synchronous THP). Status is tracked via `MIRROR_BUF_HUGEPAGE_STATE`
//!   to avoid inflating the struct layout (which would affect hot-path cache performance).
//! * **Windows 11 (`windows.rs`):** Uses modern NT virtual memory placeholders
//!   (`VirtualAlloc2` with `MEM_RESERVE_PLACEHOLDER`, split via `VirtualFree` with
//!   `MEM_PRESERVE_PLACEHOLDER` at 64 KiB allocation granularity, pagefile-backed
//!   `CreateFileMappingW`, dual `MapViewOfFile3` with `MEM_REPLACE_PLACEHOLDER`,
//!   immediate `CloseHandle`, and dual `UnmapViewOfFile` on `Drop`), preserving
//!   `size_of::<MirroredBuffer<T>>() == 16`.
use std::marker::PhantomData;
use std::ops::{Deref, Index, IndexMut, Range};
use std::sync::atomic::AtomicU8;

mod alloc;
#[cfg(unix)]
mod posix;
#[cfg(any(windows, test))]
mod windows;

thread_local! {
    /// Thread-local injection hook for unit tests to simulate allocation failures.
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
#[non_exhaustive]
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
/// The virtual extent exposed by `Deref` (`&[T]`) is `2N` elements (`N =
/// size()`), but the **two halves are NOT independent storage**: they map the
/// same physical pages twice (`MAP_SHARED` on the same fd/offset in
/// `mirror_buf/posix.rs`, or dual section views in `mirror_buf/windows.rs`).
/// Consequences:
///
/// - A write to `buf[i]` for `i < N` is observable at `buf[N + i]`, and vice
///   versa — both addresses alias the same physical location. The mirror exists
///   precisely so linear reads/writes that cross the `N` boundary behave as a
///   wrap-around on the same physical buffer.
/// - `Deref` (`&[T]` over `2N`) is sound for shared reads because `T: MirrorSafe`
///   is `Copy` with no `Drop` glue and no interior mutability (`UnsafeCell` is
///   `!Copy`), and no `&mut` borrow can coexist with `&self`.
/// - `DerefMut` is **intentionally not implemented**: exposing `&mut [T]` over
///   `2N` would let safe code call `split_at_mut(buf.size())` to obtain two
///   simultaneously live `&mut [T]` slices over the same physical memory,
///   violating Rust's exclusive-reference (`noalias`) guarantees and enabling
///   safe data races.
/// - Mutable access is exposed exclusively through operations that cannot
///   create overlapping `&mut` borrows: single-element indexing (`IndexMut<usize>`),
///   bounded mutable windows of at most `N` elements (`window_mut` /
///   `try_window_mut`, where `len <= N` guarantees every element maps to a
///   distinct physical slot), `write_slice_at`, `fill`, and ring-aware
///   `copy_within`.
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
/// `Deref` (shared `2N` view) and bounded mutable windows (`len <= N`) requires
/// every `T` used with this buffer to satisfy:
///
/// 1. **All-bits-zero is a valid `T`.** Storage is created zero-filled
///    (fresh mmap/ftruncate or pagefile section pages), so an all-zero bit
///    pattern must be a valid value of `T` — types with niches forbidding it
///    (e.g. `NonZeroU32`, references `&T`, `NonNull<u8>`) are forbidden.
/// 2. **No `Drop` glue and no interior mutability.** A slot is reachable and
///    overwritable through two virtual addresses; memory is unmapped without
///    running element drops. `Self: Copy` guarantees absence of custom
///    destructors and forbids `UnsafeCell`.
/// 3. **Aliasing tolerance across borrows.** Writes through a bounded mutable
///    borrow intentionally update the mirrored observation at the partner
///    virtual offset once the mutable borrow ends; `T` must not rely on
///    reference identity or unique addresses.
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
///
/// ```compile_fail
/// use neural_amp_modeler_rs::dsp::mirror_buf::MirroredBuffer;
///
/// // DerefMut is absent: safe code cannot split the 2N view into two aliasing `&mut [T]` halves:
/// let mut buf = MirroredBuffer::<f32>::new(16).unwrap();
/// let n = buf.size();
/// let (_a, _b) = buf.split_at_mut(n);
/// ```
///
/// ```compile_fail
/// use neural_amp_modeler_rs::dsp::mirror_buf::MirroredBuffer;
///
/// // Mutable range indexing across > N elements is rejected at compile time:
/// let mut buf = MirroredBuffer::<f32>::new(16).unwrap();
/// let _s: &mut [f32] = &mut buf[..];
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
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size_elements
    }

    /// Returns a bounded mutable slice covering `offset..offset + len` if
    /// `len <= self.size()` and `offset + len <= 2 * self.size()`.
    ///
    /// Enforcing `len <= self.size()` guarantees that no two elements in the
    /// returned `&mut [T]` alias the same physical memory, even when the
    /// window crosses the `N` mirror seam.
    #[inline(always)]
    pub fn try_window_mut(&mut self, offset: usize, len: usize) -> Option<&mut [T]> {
        let n = self.size_elements;
        if len > n {
            return None;
        }
        let end = offset.checked_add(len)?;
        if end > n * 2 {
            return None;
        }
        // SAFETY: `offset + len <= 2 * size_elements` keeps the range within the valid
        // `2N` virtual mapping (`self.ptr` provenance spans `2N` elements, page-aligned >= align_of::<T>()).
        // `len <= size_elements` guarantees that for any distinct `i != j` in `offset..offset + len`,
        // `0 < |i - j| < N`, so `i % N != j % N`: every slot in the returned `&mut [T]` maps
        // to a distinct physical location (zero internal aliasing). `&mut self` guarantees exclusive access.
        Some(unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), len) })
    }

    /// Returns a bounded mutable slice covering `offset..offset + len`.
    ///
    /// # Panics
    /// Panics if `len > self.size()` (which would allow internal physical
    /// aliasing) or if `offset + len > 2 * self.size()`.
    #[inline(always)]
    pub fn window_mut(&mut self, offset: usize, len: usize) -> &mut [T] {
        let n = self.size_elements;
        assert!(
            len <= n,
            "MirroredBuffer::window_mut: window length {len} exceeds physical size {n}"
        );
        let end = offset
            .checked_add(len)
            .expect("MirroredBuffer::window_mut: offset + len overflowed usize");
        assert!(
            end <= n * 2,
            "MirroredBuffer::window_mut: window {offset}..{end} out of bounds (2N = {})",
            n * 2
        );
        // SAFETY: Same invariants verified above as `try_window_mut`: `offset + len <= 2 * N`
        // within the valid `2N` mapping and `len <= N` ensures every slot is physically distinct.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(offset), len) }
    }

    /// Copies `src` into the bounded window starting at `offset`.
    ///
    /// # Panics
    /// Panics if `src.len() > self.size()` or `offset + src.len() > 2 * self.size()`.
    #[inline(always)]
    pub fn write_slice_at(&mut self, offset: usize, src: &[T]) {
        self.window_mut(offset, src.len()).copy_from_slice(src);
    }

    /// Fills all `N = self.size()` physical elements with `value`.
    ///
    /// Because both virtual halves map the same physical pages, writing `[0..N)`
    /// updates `[N..2N)` in one pass.
    #[inline]
    pub fn fill(&mut self, value: T) {
        // SAFETY: `self.ptr` is valid for `size_elements * 2` elements, so the first
        // `size_elements` physical elements `[0..N)` are valid, aligned, and mutually disjoint.
        // Exclusive `&mut self` ensures no other reference exists during the fill.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_elements) }.fill(value);
    }

    /// Copies `src` to `dest` within the `2N` mirrored view, handling both
    /// virtual and physical wrap-around overlap safely without heap allocation.
    ///
    /// # Panics
    /// Panics if `src.start > src.end`, `src.end > 2 * self.size()`,
    /// `src.len() > self.size()`, or `dest + src.len() > 2 * self.size()`.
    pub fn copy_within(&mut self, src: Range<usize>, dest: usize) {
        let n = self.size_elements;
        let total = n * 2;
        assert!(
            src.start <= src.end && src.end <= total,
            "MirroredBuffer::copy_within: source range {}..{} out of bounds (2N = {total})",
            src.start,
            src.end
        );
        let len = src.end - src.start;
        assert!(
            len <= n,
            "MirroredBuffer::copy_within: copy length {len} exceeds physical size {n}"
        );
        assert!(
            dest.checked_add(len).is_some_and(|end| end <= total),
            "MirroredBuffer::copy_within: destination range {dest}..+{len} out of bounds (2N = {total})"
        );
        if len == 0 {
            return;
        }

        let src_p = if src.start >= n {
            src.start - n
        } else {
            src.start
        };
        let dst_p = if dest >= n { dest - n } else { dest };
        if src_p == dst_p {
            return;
        }

        let fwd_dist = if dst_p >= src_p {
            dst_p - src_p
        } else {
            n - (src_p - dst_p)
        };
        let bwd_dist = n - fwd_dist;

        if fwd_dist >= len && bwd_dist >= len {
            // SAFETY: `src_p < n`, `dst_p < n`, `len <= n` keep both ranges inside `[0..2N)`.
            // `fwd_dist >= len && bwd_dist >= len` proves physical disjointness on the ring.
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.add(src_p), self.ptr.add(dst_p), len);
            }
        } else if fwd_dist >= len {
            if dst_p < src_p {
                // SAFETY: Both ranges are inside `[0..2N)`, and `dst_p < src_p` ensures
                // `ptr::copy` copies forward, matching `fwd_dist >= len` physical safety.
                unsafe { std::ptr::copy(self.ptr.add(src_p), self.ptr.add(dst_p), len) };
            } else if src_p + len <= n {
                // SAFETY: `src_p + n + len <= 2N` and `dst_p < src_p + n`, so `ptr::copy`
                // copies forward across the `[src_p + n ..]` mirror alias safely.
                unsafe { std::ptr::copy(self.ptr.add(src_p + n), self.ptr.add(dst_p), len) };
            } else {
                for i in 0..len {
                    // SAFETY: `src.start + i < 2N` and `dest + i < 2N`; `fwd_dist >= len`
                    // guarantees forward order never overwrites an unread source element.
                    unsafe {
                        let v = std::ptr::read(self.ptr.add(src.start + i));
                        std::ptr::write(self.ptr.add(dest + i), v);
                    }
                }
            }
        } else if bwd_dist >= len {
            if src_p < dst_p {
                // SAFETY: Both ranges are inside `[0..2N)`, and `src_p < dst_p` ensures
                // `ptr::copy` copies backward, matching `bwd_dist >= len` physical safety.
                unsafe { std::ptr::copy(self.ptr.add(src_p), self.ptr.add(dst_p), len) };
            } else if dst_p + len <= n {
                // SAFETY: `dst_p + n + len <= 2N` and `src_p < dst_p + n`, so `ptr::copy`
                // copies backward across the `[dst_p + n ..]` mirror alias safely.
                unsafe { std::ptr::copy(self.ptr.add(src_p), self.ptr.add(dst_p + n), len) };
            } else {
                for i in (0..len).rev() {
                    // SAFETY: `src.start + i < 2N` and `dest + i < 2N`; `bwd_dist >= len`
                    // guarantees backward order never overwrites an unread source element.
                    unsafe {
                        let v = std::ptr::read(self.ptr.add(src.start + i));
                        std::ptr::write(self.ptr.add(dest + i), v);
                    }
                }
            }
        } else {
            // Both ends wrap-overlap (`len > n / 2`). Perform an exact allocation-free
            // ring shift over the single physical view `[0..n)`.
            // SAFETY: `self.ptr` is valid for `n` disjoint physical elements `[0..n)` under `&mut self`.
            let phys = unsafe { std::slice::from_raw_parts_mut(self.ptr, n) };
            let ring_idx = |offset: usize| -> usize {
                let sum = src_p + offset;
                if sum >= n { sum - n } else { sum }
            };
            for sink in len..n {
                let mut cur = sink;
                loop {
                    let prev = if cur >= fwd_dist {
                        cur - fwd_dist
                    } else {
                        cur + bwd_dist
                    };
                    if prev >= len {
                        break;
                    }
                    phys[ring_idx(cur)] = phys[ring_idx(prev)];
                    cur = prev;
                }
            }
            let g = alloc::gcd(fwd_dist, n);
            if n - len < g {
                for r in 0..g {
                    let rem_dist = (r + g - (len % g)) % g;
                    if rem_dist >= n - len {
                        let saved = phys[ring_idx(r)];
                        let mut cur = r;
                        loop {
                            let prev = if cur >= fwd_dist {
                                cur - fwd_dist
                            } else {
                                cur + bwd_dist
                            };
                            if prev == r {
                                phys[ring_idx(cur)] = saved;
                                break;
                            }
                            phys[ring_idx(cur)] = phys[ring_idx(prev)];
                            cur = prev;
                        }
                    }
                }
            }
        }
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

impl<T: MirrorSafe, I: std::slice::SliceIndex<[T]>> Index<I> for MirroredBuffer<T> {
    type Output = I::Output;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        &(**self)[index]
    }
}

impl<T: MirrorSafe> IndexMut<usize> for MirroredBuffer<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let virtual_len = self.size_elements * 2;
        assert!(
            index < virtual_len,
            "MirroredBuffer::index_mut: index {index} out of bounds (2N = {virtual_len})"
        );
        // SAFETY: `index < 2 * size_elements` lies within the valid `2N` mapping,
        // `self.ptr` is page-aligned (`>= align_of::<T>()`), and `&mut self` grants
        // exclusive access to this single element (cannot alias another `&mut T`).
        unsafe { &mut *self.ptr.add(index) }
    }
}

impl<T: MirrorSafe> Drop for MirroredBuffer<T> {
    fn drop(&mut self) {
        let element_size = std::mem::size_of::<T>();
        let size_bytes = self.size_elements * element_size;
        #[cfg(unix)]
        // SAFETY: self.ptr was obtained via posix::alloc_mirrored / try_alloc_huge_mirrored
        // spanning size_bytes * 2 (two contiguous mirrored halves). Drop consumes self by
        // value, ensuring this is the exclusive and final use of the pointer. Individual
        // elements are not dropped, which is sound because T: MirrorSafe bounds T to Copy
        // types with no Drop glue.
        unsafe {
            posix::unmap_posix_views(self.ptr as *mut u8, size_bytes);
        }
        #[cfg(all(not(miri), windows))]
        // SAFETY: self.ptr is the base of two contiguous Win32 section views spanning
        // size_bytes each ([0..size_bytes) and [size_bytes..2*size_bytes)). Both views
        // are unmapped individually; Drop consumes self by value and T: MirrorSafe
        // guarantees Copy elements with no Drop glue.
        unsafe {
            windows::unmap_win32_views(self.ptr as *mut u8, size_bytes);
        }
        #[cfg(not(any(unix, all(not(miri), windows))))]
        let _ = size_bytes;
    }
}

impl<T: MirrorSafe> MirroredBuffer<T> {
    /// Fallible clone — returns `Err` on allocation failure instead of panicking.
    ///
    /// Preferred over `Clone::clone()` in host activation paths where panic
    /// would cross the FFI boundary.
    pub fn try_clone(&self) -> std::io::Result<Self> {
        let mut new_buf = Self::new(self.size_elements)?;
        new_buf.write_slice_at(0, &self[..self.size_elements]);
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
// threads. Shared access (`Deref` / `Index<usize>`) only exposes immutable `&[T]`/`&T` views
// over `Copy` non-`UnsafeCell` elements, while all mutation (`IndexMut<usize>`, `window_mut`,
// `try_window_mut`, `write_slice_at`, `fill`, `copy_within`) requires `&mut self` and never
// exposes two aliasing `&mut` references. The exclusive process-local virtual mapping prevents
// data races from external writers.
unsafe impl<T: MirrorSafe + Sync> Sync for MirroredBuffer<T> {}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[path = "mirror_buf_test.rs"]
mod mirror_buf_test;
