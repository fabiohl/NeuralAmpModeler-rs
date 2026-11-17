// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Virtual memory mappings (mmap, munmap, madvise) and file descriptor operations
// in this module adhere to strict kernel invariants: validated page alignment, leak-free error paths,
// and bounds-checked pointer offsets.
#![warn(clippy::undocumented_unsafe_blocks)]

#[cfg(target_os = "linux")]
use super::HUGEPAGE_STATE_HUGETLB;
#[cfg(target_os = "linux")]
use super::HUGEPAGE_STATE_THP;
#[cfg(target_os = "linux")]
use super::MIRROR_BUF_HUGEPAGE_STATE;
use super::SIMULATE_FAIL;
#[cfg(target_os = "linux")]
use super::alloc::{lcm, round_up_to_multiple};
#[cfg(target_os = "linux")]
use crate::math::common::huge_alloc::HUGE_PAGE_2M;
use libc::{
    MAP_ANONYMOUS, MAP_FAILED, MAP_FIXED, MAP_PRIVATE, MAP_SHARED, PROT_READ, PROT_WRITE, c_void,
    mmap, munmap, sysconf,
};
use std::ptr;
#[cfg(target_os = "linux")]
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

    // SAFETY: mmap with null address requests a contiguous unmapped address reservation
    // of total_size (2 * size_bytes) from the kernel; return value is checked against MAP_FAILED.
    let base_ptr = unsafe {
        if SIMULATE_FAIL.with(|f| f.get()) {
            set_errno(libc::ENOMEM);
            MAP_FAILED
        } else {
            mmap(
                ptr::null_mut(),
                total_size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
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
    #[cfg(target_os = "linux")]
    {
        // SAFETY: base_ptr points to the valid page-aligned mapped region of size_bytes; madvise provides VM advice.
        let collapse_rc = unsafe {
            libc::madvise(base_ptr, size_bytes, libc::MADV_HUGEPAGE);
            libc::madvise(base_ptr, size_bytes, libc::MADV_COLLAPSE)
        };
        if collapse_rc == 0 {
            MIRROR_BUF_HUGEPAGE_STATE.fetch_max(HUGEPAGE_STATE_THP, Ordering::Relaxed);
        }
    }

    // SAFETY: Both virtual mirrors maintain open references to the backing file in kernel VM state;
    // closing the user-space fd avoids descriptor leaks without unmapping the memory.
    unsafe { libc::close(fd) };

    Ok(base_ptr as *mut u8)
}

/// Attempts creation with explicit 2 MB huge pages, honouring `elem_stride` alignment.
/// Returns `(base_ptr, size_bytes_per_half)` on success.
#[cfg(target_os = "linux")]
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
    // SAFETY: mmap with null address requests a contiguous unmapped huge-page address reservation
    // of total_size (2 * size_bytes) from the kernel; return value is checked against MAP_FAILED.
    let base_ptr = unsafe {
        mmap(
            ptr::null_mut(),
            total_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | libc::MAP_HUGETLB | libc::MAP_HUGE_2MB,
            -1,
            0,
        )
    };
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

/// Sets the calling thread's `errno`, so callers that report failures with
/// `std::io::Error::last_os_error()` see the intended cause.
///
/// # Safety
/// Writes through the libc thread-local errno pointer, which is always valid.
unsafe fn set_errno(code: libc::c_int) {
    // SAFETY: the libc errno accessor returns a valid pointer to this thread's errno.
    unsafe {
        #[cfg(target_os = "linux")]
        {
            *libc::__errno_location() = code;
        }
        #[cfg(target_vendor = "apple")]
        {
            *libc::__error() = code;
        }
        #[cfg(not(any(target_os = "linux", target_vendor = "apple")))]
        {
            let _ = code;
        }
    }
}

/// Creates a backing file descriptor for huge-page or standard mirrored allocation.
///
/// Uses `memfd_create` with optional `MFD_HUGETLB` (Linux 5.14+) flag.
/// When `use_huge` is true, first attempts MFD_HUGETLB, falling back to
/// regular memfd if huge pages are unavailable. The file is truncated
/// to `size` bytes.
#[cfg(target_os = "linux")]
unsafe fn create_backing_fd(size: usize, use_huge: bool) -> std::io::Result<std::os::raw::c_int> {
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

/// Sequence number for macOS shared-memory object names (`/nam.<pid>.<seq>`).
#[cfg(target_vendor = "apple")]
static SHM_NAME_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// macOS equivalent of the anonymous `memfd`: a POSIX shared-memory object
/// that is unlinked immediately, so the returned descriptor is its only
/// reference. Huge pages are not available; `use_huge` is ignored. The object
/// is truncated to `size` bytes.
#[cfg(target_vendor = "apple")]
unsafe fn create_backing_fd(size: usize, _use_huge: bool) -> std::io::Result<std::os::raw::c_int> {
    // SAFETY: forwards the caller's contract; the global counter is the
    // production name sequence.
    unsafe { create_backing_fd_with_seq(size, &SHM_NAME_SEQ) }
}

/// [`create_backing_fd`] with an explicit name-sequence counter, so tests can
/// drive the name collision path deterministically.
#[cfg(target_vendor = "apple")]
unsafe fn create_backing_fd_with_seq(
    size: usize,
    seq: &std::sync::atomic::AtomicU64,
) -> std::io::Result<std::os::raw::c_int> {
    use std::sync::atomic::Ordering;

    /// Appends the decimal digits of `n` at `buf[pos..]`, returning the new end.
    fn push_decimal(buf: &mut [u8], mut pos: usize, mut n: u64) -> usize {
        let mut digits = [0u8; 20];
        let mut count = 0;
        loop {
            digits[count] = b'0' + (n % 10) as u8;
            count += 1;
            n /= 10;
            if n == 0 {
                break;
            }
        }
        while count > 0 {
            count -= 1;
            buf[pos] = digits[count];
            pos += 1;
        }
        pos
    }

    // An object left behind by a process killed between shm_open and
    // shm_unlink can collide with a reused pid; on EEXIST try the next name.
    const MAX_NAME_ATTEMPTS: usize = 64;
    let mut attempt = 0;
    let (fd, name_buf) = loop {
        // "/nam.<pid>.<seq>" NUL-terminated on the stack, no allocation. macOS
        // limits shm names to 31 bytes (PSHMNAMLEN): 5 + 10 (u32 pid) + 1 + 9
        // (seq mod 1e9) = 25 bytes, plus the zero terminator already in `name`.
        let mut name = [0u8; 32];
        name[..5].copy_from_slice(b"/nam.");
        let mut end = push_decimal(&mut name, 5, u64::from(std::process::id()));
        name[end] = b'.';
        end = push_decimal(
            &mut name,
            end + 1,
            seq.fetch_add(1, Ordering::Relaxed) % 1_000_000_000,
        );
        debug_assert!(end < name.len());
        // SAFETY: `name` is a valid NUL-terminated C string; O_EXCL guarantees a fresh object.
        let fd = unsafe {
            libc::shm_open(
                name.as_ptr().cast::<libc::c_char>(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                0o600 as libc::c_uint,
            )
        };
        if fd != -1 {
            break (fd, name);
        }
        let err = std::io::Error::last_os_error();
        attempt += 1;
        if err.raw_os_error() != Some(libc::EEXIST) || attempt == MAX_NAME_ATTEMPTS {
            return Err(err);
        }
    };
    // SAFETY: `name_buf` is valid; unlinking leaves the open descriptor as the only reference.
    unsafe { libc::shm_unlink(name_buf.as_ptr().cast::<libc::c_char>()) };
    // SAFETY: fd is a valid descriptor from the successful shm_open above.
    let ok = unsafe {
        libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) != -1
            && libc::ftruncate(fd, size as libc::off_t) != -1
    };
    if !ok {
        let err = std::io::Error::last_os_error();
        // SAFETY: fd is valid and no longer needed after the failure.
        unsafe { libc::close(fd) };
        return Err(err);
    }
    Ok(fd)
}

#[cfg(all(unix, not(any(target_os = "linux", target_vendor = "apple"))))]
unsafe fn create_backing_fd(_size: usize, _use_huge: bool) -> std::io::Result<std::os::raw::c_int> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "create_backing_fd is only supported on Linux and macOS",
    ))
}

#[cfg(all(test, target_vendor = "apple"))]
#[path = "posix_test.rs"]
mod tests;
