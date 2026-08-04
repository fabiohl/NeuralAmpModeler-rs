// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Low-level mmap/munmap/madvise syscalls with size/alignment validation.

//! Huge Page-aware memory allocation for large, RT-critical buffers.
//!
//! TLB (Translation Lookaside Buffer) misses in the DSP hot-path cost ~100 cycles.
//! Standard 4 KB pages cause ~20 TLB entries for an 80 KB weight buffer;
//! a single 2 MB huge page covers the entire buffer with 1 TLB entry.
//!
//! # Strategy (best-effort, zero regression risk)
//! 1. `mmap(MAP_HUGETLB | MAP_HUGE_2MB)` — explicit huge pages (requires admin setup).
//! 2. `mmap(MAP_ANONYMOUS | MAP_PRIVATE)` + `madvise(MADV_HUGEPAGE)` + `madvise(MADV_COLLAPSE)` — synchronous THP.
//! 3. `std::alloc::alloc` — fallback (existing behaviour).
//!
//! Callers must use the returned `AllocInfo` to choose the correct deallocation path.

use std::alloc::{Layout, alloc, dealloc};
use std::ptr;

use crate::common::diagnostics::NamErrorCode;

/// Tracks which allocation strategy was used, to pick the right deallocation path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocInfo {
    /// Standard global-allocator allocation (`std::alloc::dealloc`).
    Heap,
    /// Anonymous `mmap`: deallocate via `munmap(ptr, size_bytes)`.
    MmapAnon {
        /// The exact byte size passed to mmap (page-aligned).
        size_bytes: usize,
    },
    /// Explicit 2 MB huge-page `mmap`: deallocate via `munmap(ptr, size_bytes)`.
    HugeTlb2M {
        /// The exact byte size passed to mmap (2 MB-aligned).
        size_bytes: usize,
    },
}

/// Result of a huge-page allocation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HugePageStatus {
    /// Explicit 2 MB huge pages via MAP_HUGETLB succeeded.
    Explicit2MB,
    /// Transparent huge pages via madvise(MADV_HUGEPAGE) + MADV_COLLAPSE on anonymous mmap.
    Transparent,
    /// Fallback to standard heap allocation (global allocator).
    Heap,
}

/// Minimum allocation size (in bytes) that triggers a huge-page attempt.
/// Below this threshold, standard `alloc` is used directly (no syscall overhead).
pub const HUGE_PAGE_THRESHOLD: usize = 1 << 20; // 1 MiB

/// Page size for transparent huge pages (4 KB standard pages, kernel promotes to 2 MB).
const PAGE_4K: usize = 4096;

/// Huge page size (2 MB on x86-64). Must divide evenly into the allocation size.
pub const HUGE_PAGE_2M: usize = 2 * 1024 * 1024;

/// Rounds `size` up to the next multiple of `alignment`.
const fn align_up(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

/// Attempts to allocate `size_bytes` with huge-page preference.
///
/// # Returns
/// `Ok((ptr, AllocInfo, HugePageStatus))` — the caller must use `AllocInfo` to
/// select the correct deallocation path.
///
/// # Errors
/// Returns `NamErrorCode::OutOfMemory` if the layout could not be constructed
/// (size overflows `isize::MAX`) or if the global allocator is exhausted.
///
/// # Safety
/// The caller must eventually deallocate using `deallocate_huge()` with the matching `AllocInfo`.
/// The returned pointer is guaranteed to be at least 64-byte aligned for AVX-512.
pub fn allocate_huge_pages(
    size_bytes: usize,
) -> Result<(*mut u8, AllocInfo, HugePageStatus), NamErrorCode> {
    if size_bytes < HUGE_PAGE_THRESHOLD {
        let layout =
            Layout::from_size_align(size_bytes, 64).map_err(|_| NamErrorCode::OutOfMemory)?;
        // SAFETY: standard alloc with valid layout.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(NamErrorCode::OutOfMemory);
        }
        return Ok((ptr, AllocInfo::Heap, HugePageStatus::Heap));
    }

    let huge_2m_size = align_up(size_bytes, HUGE_PAGE_2M);

    // Strategy 1: explicit 2 MB huge pages via MAP_HUGETLB.
    // SAFETY: try_mmap_huge with validated size, no aliasing violations.
    let ptr = unsafe { try_mmap_huge(ptr::null_mut(), huge_2m_size, -1, 0, true) };

    if ptr != libc::MAP_FAILED {
        return Ok((
            ptr as *mut u8,
            AllocInfo::HugeTlb2M {
                size_bytes: huge_2m_size,
            },
            HugePageStatus::Explicit2MB,
        ));
    }

    // Strategy 2: anonymous mmap + madvise(MADV_HUGEPAGE) + MADV_COLLAPSE for synchronous THP.
    let thp_size = align_up(size_bytes, PAGE_4K);
    // SAFETY: try_mmap_huge with validated size.
    let ptr = unsafe { try_mmap_huge(ptr::null_mut(), thp_size, -1, 0, false) };

    if ptr != libc::MAP_FAILED {
        // Hint the kernel to promote these pages to huge pages.
        // SAFETY: ptr and thp_size are valid from the successful mmap above.
        let _madvise_rc = unsafe { libc::madvise(ptr, thp_size, libc::MADV_HUGEPAGE) };
        // Synchronous collapse (Linux 6.1+). Mandatory: no kernel-version fallback checks.
        // SAFETY: ptr and thp_size are valid from the successful mmap above.
        let collapse_rc = unsafe { libc::madvise(ptr, thp_size, libc::MADV_COLLAPSE) };
        let status = if collapse_rc == 0 {
            HugePageStatus::Transparent
        } else {
            HugePageStatus::Heap
        };

        return Ok((
            ptr as *mut u8,
            AllocInfo::MmapAnon {
                size_bytes: thp_size,
            },
            status,
        ));
    }

    // Strategy 3: fallback to standard allocator.
    let layout = Layout::from_size_align(size_bytes, 64).map_err(|_| NamErrorCode::OutOfMemory)?;
    // SAFETY: standard alloc with valid layout.
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        return Err(NamErrorCode::OutOfMemory);
    }
    Ok((ptr, AllocInfo::Heap, HugePageStatus::Heap))
}

/// Deallocates memory previously obtained from `allocate_huge_pages()`.
///
/// # Safety
/// - `ptr` must have been returned by `allocate_huge_pages()`.
/// - `info` must be the exact `AllocInfo` returned by that call.
/// - `size_bytes` must be the original request size (un-rounded) for Heap deallocs;
///   for mmap-based deallocs, the alloc info already stores the correct size.
pub unsafe fn deallocate_huge(ptr: *mut u8, info: AllocInfo, size_bytes: usize) {
    match info {
        AllocInfo::Heap => {
            // Layout construction cannot fail here: size_bytes was validated
            // during allocation when the same size+align produced a valid layout.
            if let Ok(layout) = Layout::from_size_align(size_bytes, 64) {
                // SAFETY: ptr and layout match the original allocation.
                unsafe { dealloc(ptr, layout) };
            }
        }
        AllocInfo::MmapAnon {
            size_bytes: mmap_size,
        }
        | AllocInfo::HugeTlb2M {
            size_bytes: mmap_size,
        } => {
            // SAFETY: ptr and mmap_size match the original mmap.
            unsafe { libc::munmap(ptr as *mut libc::c_void, mmap_size) };
        }
    }
}

// ── Shared helpers for fd-backed huge-page allocation ────────────────────────

/// Creates a backing file descriptor for huge-page or standard allocation.
///
/// Uses `memfd_create` with optional `MFD_HUGETLB` (Linux 5.14+) flag.
/// When `use_huge` is true, first attempts MFD_HUGETLB, falling back to
/// regular memfd if huge pages are unavailable. The file is truncated
/// to `size` bytes.
///
/// Returns the file descriptor on success, or an IO error.
#[cfg(target_os = "linux")]
pub(crate) unsafe fn create_backing_fd(
    size: usize,
    use_huge: bool,
) -> std::io::Result<std::os::raw::c_int> {
    if use_huge {
        const MFD_HUGETLB: libc::c_uint = 0x0004;
        const MFD_NOEXEC_SEAL: libc::c_uint = 0x0008;
        // SAFETY: C string pointer is valid and null-terminated; flags are valid (Linux 6.3+).
        let fd = unsafe {
            libc::memfd_create(
                c"nam_huge_buf".as_ptr(),
                libc::MFD_CLOEXEC | MFD_HUGETLB | MFD_NOEXEC_SEAL,
            )
        };
        if fd != -1 {
            // SAFETY: fd is a valid file descriptor from a successful memfd_create above.
            if unsafe { libc::ftruncate(fd, size as libc::off_t) } != -1 {
                return Ok(fd);
            }
            let err = std::io::Error::last_os_error();
            // SAFETY: fd is valid and no longer needed after ftruncate failure.
            unsafe { libc::close(fd) };
            return Err(err);
        }
        // Fall through to regular memfd
    }

    const MFD_NOEXEC_SEAL: libc::c_uint = 0x0008;
    // SAFETY: C string pointer is valid and null-terminated; flags are valid (Linux 6.3+).
    let fd =
        unsafe { libc::memfd_create(c"nam_buf".as_ptr(), libc::MFD_CLOEXEC | MFD_NOEXEC_SEAL) };
    if fd == -1 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: fd is a valid file descriptor from a successful memfd_create above.
    if unsafe { libc::ftruncate(fd, size as libc::off_t) } == -1 {
        let err = std::io::Error::last_os_error();
        // SAFETY: fd is valid and no longer needed after ftruncate failure.
        unsafe { libc::close(fd) };
        return Err(err);
    }
    Ok(fd)
}

#[cfg(not(target_os = "linux"))]
pub(crate) unsafe fn create_backing_fd(
    _size: usize,
    _use_huge: bool,
) -> std::io::Result<std::os::raw::c_int> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "create_backing_fd is only supported on Linux",
    ))
}

/// mmap with optional huge-page flags (MAP_HUGETLB | MAP_HUGE_2MB).
///
/// When `fd == -1`, uses `MAP_PRIVATE | MAP_ANONYMOUS` (anonymous mapping).
/// When `fd != -1`, uses `MAP_SHARED` (file-backed mapping).
/// When `use_huge` is true, adds `MAP_HUGETLB | MAP_HUGE_2MB` to the flags.
///
/// Uses `PROT_READ | PROT_WRITE` protection.
///
/// Returns `MAP_FAILED` on failure (caller checks against `libc::MAP_FAILED`).
pub(crate) unsafe fn try_mmap_huge(
    addr: *mut libc::c_void,
    len: usize,
    fd: std::os::raw::c_int,
    offset: libc::off_t,
    use_huge: bool,
) -> *mut libc::c_void {
    let mut flags = if fd == -1 {
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS
    } else {
        libc::MAP_SHARED
    };

    if use_huge {
        flags |= libc::MAP_HUGETLB | libc::MAP_HUGE_2MB;
    }

    // SAFETY: caller guarantees addr/len/fd/offset are valid for the requested mapping.
    unsafe {
        libc::mmap(
            addr,
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            flags,
            fd,
            offset,
        )
    }
}

// ── HugePageVec: AlignedVec-like wrapper with huge-page deallocation ──────

use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// A 64-byte aligned buffer backed by huge-page allocation (best-effort).
///
/// Unlike `AlignedVec`, this type carries the deallocation metadata (mmap size)
/// and is intended for allocations ≥ 1 MiB where TLB pressure matters.
///
/// Layout (24 bytes): larger than `AlignedVec` (16 bytes), but only used for
/// large allocations where the overhead is negligible.
#[derive(Debug)]
pub struct HugePageVec<T> {
    ptr: NonNull<T>,
    len: usize,
    alloc_info: AllocInfo,
}

impl<T> HugePageVec<T> {
    /// Creates a new huge-page-backed buffer filled with `default`.
    ///
    /// # Errors
    /// Returns `NamErrorCode::OutOfMemory` if allocation fails.
    pub fn new(len: usize, default: T) -> Result<(Self, HugePageStatus), NamErrorCode>
    where
        T: Copy,
    {
        let (mut vec, status) = Self::with_capacity(len)?;
        // SAFETY: Inner safety guarantees are upheld by caller invariants.
        unsafe {
            for i in 0..len {
                vec.ptr.as_ptr().add(i).write(default);
            }
        }
        vec.len = len;
        Ok((vec, status))
    }

    /// Reserves capacity with huge-page preference.
    ///
    /// # Errors
    /// Returns `NamErrorCode::OutOfMemory` if allocation fails.
    pub fn with_capacity(capacity: usize) -> Result<(Self, HugePageStatus), NamErrorCode> {
        if capacity == 0 {
            return Ok((
                Self {
                    ptr: NonNull::dangling(),
                    len: 0,
                    alloc_info: AllocInfo::Heap,
                },
                HugePageStatus::Heap,
            ));
        }
        let size_bytes = capacity * std::mem::size_of::<T>();
        let (ptr, alloc_info, status) = allocate_huge_pages(size_bytes)?;
        Ok((
            Self {
                ptr: NonNull::new(ptr as *mut T).unwrap(),
                len: 0,
                alloc_info,
            },
            status,
        ))
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the huge-page allocation status.
    pub fn huge_page_status(&self) -> HugePageStatus {
        match self.alloc_info {
            AllocInfo::Heap => HugePageStatus::Heap,
            AllocInfo::MmapAnon { .. } => HugePageStatus::Transparent,
            AllocInfo::HugeTlb2M { .. } => HugePageStatus::Explicit2MB,
        }
    }
}

impl<T> Deref for HugePageVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: self.ptr is non-null (allocated via allocate_huge_pages or fallback
            // alloc; empty case guarded by len==0 above). self.len ≤ capacity, determined
            // at allocation time, and elements [0..len) are properly initialized by the
            // HugePageVec construction and push operations.
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl<T> DerefMut for HugePageVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: Same pointer/size invariants as Deref. &mut self ensures exclusive
            // access to the allocation, so no aliasing of any element occurs.
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl<T> Drop for HugePageVec<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: self.ptr was allocated via allocate_huge_pages (or the fallback heap
            // path), tracked by self.alloc_info. The deallocation size self.len * size_of::<T>()
            // matches the allocation size modulo page alignment rounding (deallocate_huge uses
            // the stored size_bytes from allocation). Drop consumes self; this is the final
            // use of the pointer.
            unsafe {
                deallocate_huge(
                    self.ptr.as_ptr() as *mut u8,
                    self.alloc_info,
                    self.len * std::mem::size_of::<T>(),
                );
            }
        }
    }
}

// SAFETY: HugePageVec owns its allocation exclusively (mmap/MAP_PRIVATE or heap with no
// external aliasing). Sending across threads is sound because the underlying memory is not
// shared with other HugePageVec instances, and T: Send ensures elements are thread-transferable.
unsafe impl<T: Send> Send for HugePageVec<T> {}
// SAFETY: T: Sync guarantees shared references to elements are sound across threads. The
// allocation is exclusively owned and accessed through Deref/DerefMut which enforce Rust's
// aliasing rules.
unsafe impl<T: Sync> Sync for HugePageVec<T> {}
