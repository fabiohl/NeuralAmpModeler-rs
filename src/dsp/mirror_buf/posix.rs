// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Virtual memory mappings (mmap, munmap, madvise) and file descriptor operations
// in this module adhere to strict kernel invariants: validated page alignment, leak-free error paths,
// and bounds-checked pointer offsets.
#![warn(clippy::undocumented_unsafe_blocks)]

use super::alloc::{lcm, round_up_to_multiple};
use super::{HUGEPAGE_STATE_HUGETLB, HUGEPAGE_STATE_THP, MIRROR_BUF_HUGEPAGE_STATE, SIMULATE_FAIL};
use crate::math::common::huge_alloc::{HUGE_PAGE_2M, create_backing_fd, set_errno, try_mmap_huge};
use libc::{
    MAP_FAILED, MAP_FIXED, MAP_SHARED, PROT_READ, PROT_WRITE, c_void, mmap, munmap, sysconf,
};
use std::ptr;
use std::sync::atomic::Ordering;

/// Returns the POSIX system page size (`_SC_PAGESIZE`), verified to be a power of two.
pub(super) fn allocation_granularity() -> std::io::Result<usize> {
    // SAFETY: `sysconf` is a standard POSIX runtime system configuration query with no
    // pointer dereferences or memory safety side-effects.
    let raw = unsafe { sysconf(libc::_SC_PAGESIZE) };
    if raw <= 0 {
        return Err(std::io::Error::last_os_error());
    }
    let page_size = raw as usize;
    if !page_size.is_power_of_two() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("system page size {page_size} is not a power of two"),
        ));
    }
    Ok(page_size)
}

/// Allocates two contiguous mirrored virtual views of `size_bytes` backed by an anonymous fd.
#[cold]
pub(super) fn alloc_mirrored(size_bytes: usize) -> std::io::Result<*mut u8> {
    // 1. Create backing store (memfd on Linux, unlinked POSIX shm on macOS)
    // SAFETY: set_errno writes the thread-local errno; create_backing_fd creates
    // an anonymous memory file descriptor sized to size_bytes.
    let fd = unsafe {
        if SIMULATE_FAIL.with(|f| f.get()) {
            set_errno(libc::ENOMEM);
            return Err(std::io::Error::last_os_error());
        }
        create_backing_fd(size_bytes, false)?
    };

    // 2. Reserve contiguous virtual space (2x size)
    let total_size = size_bytes.checked_mul(2).ok_or_else(|| {
        // SAFETY: fd is open and must be closed before returning overflow error.
        unsafe { libc::close(fd) };
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "size_bytes * 2 overflowed",
        )
    })?;

    // SAFETY: try_mmap_huge with null address requests a contiguous unmapped address reservation
    // of total_size (2 * size_bytes) from the kernel; return value is checked against MAP_FAILED.
    let base_ptr = unsafe {
        if SIMULATE_FAIL.with(|f| f.get()) {
            set_errno(libc::ENOMEM);
            MAP_FAILED
        } else {
            try_mmap_huge(ptr::null_mut(), total_size, -1, 0, false)
        }
    };
    if base_ptr == MAP_FAILED {
        let err = std::io::Error::last_os_error();
        // SAFETY: fd is the open file descriptor returned by create_backing_fd; closing it prevents descriptor leaks.
        unsafe { libc::close(fd) };
        return Err(err);
    }

    // 3. Map the first half
    // SAFETY: base_ptr is the base of the reserved virtual address range of size total_size >= size_bytes;
    // MAP_FIXED | MAP_SHARED maps fd (valid descriptor of length size_bytes) at offset 0 into the first half.
    let ptr1 = unsafe {
        mmap(
            base_ptr,
            size_bytes,
            PROT_READ | PROT_WRITE,
            MAP_FIXED | MAP_SHARED,
            fd,
            0,
        )
    };
    if ptr1 != base_ptr {
        let err = std::io::Error::last_os_error();
        // SAFETY: Roll back on failure by unmapping the full total_size virtual reservation and closing fd.
        unsafe {
            munmap(base_ptr, total_size);
            libc::close(fd);
        }
        return Err(err);
    }

    // 4. Map the second half (mirror)
    // SAFETY: base_ptr.add(size_bytes) points to the exact start of the second half within the
    // contiguous total_size reservation; MAP_FIXED | MAP_SHARED maps the same fd at offset 0.
    let ptr2 = unsafe {
        mmap(
            (base_ptr as *mut u8).add(size_bytes) as *mut c_void,
            size_bytes,
            PROT_READ | PROT_WRITE,
            MAP_FIXED | MAP_SHARED,
            fd,
            0,
        )
    };
    // SAFETY: Invariant check: base_ptr.add(size_bytes) computes the exact expected fixed address for the mirror.
    if ptr2 != unsafe { (base_ptr as *mut u8).add(size_bytes) as *mut c_void } {
        let err = std::io::Error::last_os_error();
        // SAFETY: Roll back by unmapping the entire reserved virtual range (total_size) and closing fd.
        unsafe {
            munmap(base_ptr, total_size);
            libc::close(fd);
        }
        return Err(err);
    }

    // Hint THP promotion for the data regions, then force synchronous collapse.
    // Only report THP active if the kernel confirms success (return 0).
    // SAFETY: base_ptr points to the valid page-aligned mapped region of size_bytes; madvise provides VM advice.
    #[cfg(target_os = "linux")]
    let collapse_rc = unsafe {
        libc::madvise(base_ptr, size_bytes, libc::MADV_HUGEPAGE);
        libc::madvise(base_ptr, size_bytes, libc::MADV_COLLAPSE)
    };
    // No transparent huge pages outside Linux.
    #[cfg(not(target_os = "linux"))]
    let collapse_rc = -1;
    if collapse_rc == 0 {
        MIRROR_BUF_HUGEPAGE_STATE.fetch_max(HUGEPAGE_STATE_THP, Ordering::Relaxed);
    }

    // SAFETY: Both virtual mirrors maintain open references to the backing file in kernel VM state;
    // closing the user-space fd avoids descriptor leaks without unmapping the memory.
    unsafe { libc::close(fd) };

    Ok(base_ptr as *mut u8)
}

/// Attempts creation with explicit 2 MB huge pages, honouring `elem_stride` alignment.
/// Returns `(base_ptr, size_bytes_per_half)` on success.
#[cold]
pub(super) fn try_alloc_huge_mirrored(
    requested_bytes: usize,
    elem_stride: usize,
) -> std::io::Result<(*mut u8, usize)> {
    let align_bytes = lcm(HUGE_PAGE_2M, elem_stride).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "lcm(huge_page_size, elem_stride) overflowed",
        )
    })?;
    let size_bytes = round_up_to_multiple(requested_bytes, align_bytes).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "size_bytes calculation overflowed",
        )
    })?;

    // 1. Create HugeTLB-backed memfd (falls back to regular memfd)
    // SAFETY: set_errno writes the thread-local errno; create_backing_fd creates
    // an anonymous HugeTLB-backed memfd sized to size_bytes.
    let fd = unsafe {
        if SIMULATE_FAIL.with(|f| f.get()) {
            set_errno(libc::ENOMEM);
            return Err(std::io::Error::last_os_error());
        }
        create_backing_fd(size_bytes, true)?
    };

    let total_size = size_bytes.checked_mul(2).ok_or_else(|| {
        // SAFETY: fd is open and must be closed before returning overflow error.
        unsafe { libc::close(fd) };
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "size_bytes * 2 overflowed",
        )
    })?;

    // 2. Reserve 2x virtual space with MAP_HUGETLB
    // SAFETY: try_mmap_huge with null address requests a contiguous unmapped huge-page address reservation
    // of total_size (2 * size_bytes) from the kernel; return value is checked against MAP_FAILED.
    let base_ptr = unsafe { try_mmap_huge(ptr::null_mut(), total_size, -1, 0, true) };
    if base_ptr == MAP_FAILED {
        let err = std::io::Error::last_os_error();
        // SAFETY: fd is the open file descriptor returned by create_backing_fd; closing it prevents descriptor leaks.
        unsafe { libc::close(fd) };
        return Err(err);
    }

    // 3. Map the first half
    let map_flags = MAP_FIXED | MAP_SHARED;
    // SAFETY: base_ptr is the base of the reserved huge-page virtual address range of size total_size >= size_bytes;
    // MAP_FIXED | MAP_SHARED maps fd (valid descriptor of length size_bytes) at offset 0 into the first half.
    let ptr1 = unsafe {
        mmap(
            base_ptr,
            size_bytes,
            PROT_READ | PROT_WRITE,
            map_flags,
            fd,
            0,
        )
    };
    if ptr1 != base_ptr {
        let err = std::io::Error::last_os_error();
        // SAFETY: Roll back on failure by unmapping the full total_size huge-page virtual reservation and closing fd.
        unsafe {
            munmap(base_ptr, total_size);
            libc::close(fd);
        }
        return Err(err);
    }

    // 4. Map the second half (mirror)
    // SAFETY: base_ptr.add(size_bytes) points to the exact start of the second half within the
    // contiguous total_size reservation; MAP_FIXED | MAP_SHARED maps the same fd at offset 0.
    let ptr2 = unsafe {
        mmap(
            (base_ptr as *mut u8).add(size_bytes) as *mut c_void,
            size_bytes,
            PROT_READ | PROT_WRITE,
            map_flags,
            fd,
            0,
        )
    };
    // SAFETY: Invariant check: base_ptr.add(size_bytes) computes the exact expected fixed address for the mirror.
    if ptr2 != unsafe { (base_ptr as *mut u8).add(size_bytes) as *mut c_void } {
        let err = std::io::Error::last_os_error();
        // SAFETY: Roll back by unmapping the entire reserved virtual range (total_size) and closing fd.
        unsafe {
            munmap(base_ptr, total_size);
            libc::close(fd);
        }
        return Err(err);
    }

    // SAFETY: Both huge-page virtual mirrors maintain open references to the backing file in kernel VM state;
    // closing the user-space fd avoids descriptor leaks without unmapping the memory.
    unsafe { libc::close(fd) };

    MIRROR_BUF_HUGEPAGE_STATE.fetch_max(HUGEPAGE_STATE_HUGETLB, Ordering::Relaxed);
    Ok((base_ptr as *mut u8, size_bytes))
}

/// Unmaps both contiguous POSIX mirrored views (`[0..size_bytes * 2)`).
///
/// # Safety
/// `base` must be the base address returned by [`alloc_mirrored`] or [`try_alloc_huge_mirrored`]
/// for a mirrored buffer of `size_bytes` bytes per half, and this must be called at most once
/// (on `Drop`).
pub(super) unsafe fn unmap_posix_views(base: *mut u8, size_bytes: usize) {
    // SAFETY: `base` spans `size_bytes * 2` contiguous mirrored bytes mapped over the same
    // anonymous backing fd; unmapping the full `size_bytes * 2` range releases both halves.
    unsafe {
        munmap(base as *mut c_void, size_bytes * 2);
    }
}
