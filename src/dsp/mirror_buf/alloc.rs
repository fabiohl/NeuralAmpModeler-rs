// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Virtual memory mappings (mmap, munmap, madvise) and file descriptor operations
// in this module adhere to strict kernel invariants: validated page alignment, leak-free error paths,
// and bounds-checked pointer offsets.
#![warn(clippy::undocumented_unsafe_blocks)]

use super::{
    HUGEPAGE_STATE_HUGETLB, HUGEPAGE_STATE_THP, MIRROR_BUF_HUGEPAGE_STATE, MirroredBuffer,
    SIMULATE_FAIL,
};
use crate::math::common::huge_alloc::{HUGE_PAGE_2M, create_backing_fd, try_mmap_huge};
use libc::{
    MADV_HUGEPAGE, MAP_FAILED, MAP_FIXED, MAP_SHARED, PROT_READ, PROT_WRITE, c_void, mmap, munmap,
    sysconf,
};
use std::marker::PhantomData;
use std::ptr;

/// Computes the greatest common divisor (GCD) of two numbers using Euclid's algorithm.
const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Computes the least common multiple (LCM) of two numbers.
/// Returns `None` if the operation overflows `usize`.
const fn lcm(a: usize, b: usize) -> Option<usize> {
    let g = gcd(a, b);
    let a_div_g = a / g;
    a_div_g.checked_mul(b)
}

/// Rounds `value` up to the next multiple of `align`.
/// Returns `None` if arithmetic overflow occurs.
fn round_up_to_multiple(value: usize, align: usize) -> Option<usize> {
    let padded = value.checked_add(align - 1)?;
    Some(padded - (padded % align))
}

impl<T> MirroredBuffer<T> {
    /// Creates a new mirrored buffer with huge-page preference.
    ///
    /// The `requested_size` (in elements) will be rounded up to the next
    /// multiple of the system page size (2 MB for huge pages, 4 KB for standard).
    /// Equivalent to `new_aligned(requested_size, 1)`.
    #[cold]
    pub fn new(requested_size: usize) -> std::io::Result<Self> {
        const {
            assert!(
                std::mem::align_of::<T>() <= 64,
                "MirroredBuffer element alignment must not exceed 64 bytes"
            );
        };
        Self::new_aligned(requested_size, 1)
    }

    /// Creates a mirrored buffer guaranteeing `size_elements % elem_multiple == 0`.
    ///
    /// Rounds `size_bytes` up to the least common multiple of the system page
    /// size and `elem_multiple * size_of::<T>()`. This ensures both the mmap
    /// mirror invariant (size is page-aligned) and divisibility by `elem_multiple`
    /// — essential for ring-buffer wrap arithmetic in multi-channel DSP.
    ///
    /// For huge-page path, rounds to `lcm(2 MiB, elem_multiple * size_of::<T>())`.
    ///
    /// Cost is negligible: e.g. CH=12 (48 B stride) adds at most 12 KiB on 4 KB
    /// pages, or up to 6 MiB on huge pages — all cold, during model load.
    #[cold]
    pub fn new_aligned(requested_size: usize, elem_multiple: usize) -> std::io::Result<Self> {
        const {
            assert!(
                std::mem::align_of::<T>() <= 64,
                "MirroredBuffer element alignment must not exceed 64 bytes"
            );
        };
        // SAFETY: `sysconf` is a standard POSIX runtime system configuration query with no
        // pointer dereferences or memory safety side-effects.
        let page_size = unsafe { sysconf(libc::_SC_PAGESIZE) } as usize;
        // R-5 / A7: the `mmap` base (and therefore every element address produced
        // by `Deref`/`DerefMut` over the 2N virtual extent) is aligned only to the
        // system page size. `from_raw_parts` additionally requires alignment to
        // `align_of::<T>()`, so construction must prove `align_of::<T>() <=
        // page_size`. This is a runtime net on top of the compile-time
        // `align_of::<T>() <= 64` const assert (both are satisfied on every
        // supported platform, where page_size >= 4096).
        let align = std::mem::align_of::<T>();
        if align > page_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "MirroredBuffer element alignment {align} exceeds the system page size {page_size}"
                ),
            ));
        }
        let element_size = std::mem::size_of::<T>();

        if element_size == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "MirroredBuffer does not support Zero Sized Types",
            ));
        }

        if requested_size == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "requested_size must be greater than zero",
            ));
        }

        if elem_multiple == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "elem_multiple must be greater than zero",
            ));
        }

        let requested_bytes = match requested_size.checked_mul(element_size) {
            Some(val) => val,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "requested_size * element_size overflowed",
                ));
            }
        };

        let total_chunk = match requested_bytes.checked_mul(2) {
            Some(val) => val,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "requested_bytes * 2 overflowed",
                ));
            }
        };

        let elem_stride = match elem_multiple.checked_mul(element_size) {
            Some(val) => val,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "elem_multiple * element_size overflowed",
                ));
            }
        };

        if total_chunk >= HUGE_PAGE_2M {
            let huge_res = Self::try_new_huge_aligned(requested_bytes, HUGE_PAGE_2M, elem_stride);
            if let Ok(buf) = huge_res {
                MIRROR_BUF_HUGEPAGE_STATE
                    .fetch_max(HUGEPAGE_STATE_HUGETLB, std::sync::atomic::Ordering::Relaxed);
                return Ok(buf);
            }
        }

        let align_bytes = match lcm(page_size, elem_stride) {
            Some(val) => val,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "lcm(page_size, elem_stride) overflowed",
                ));
            }
        };
        let size_bytes = match round_up_to_multiple(requested_bytes, align_bytes) {
            Some(v) => v,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "size_bytes calculation overflowed",
                ));
            }
        };
        let size_elements = size_bytes / element_size;

        // 1. Create backing store (memfd on Linux, stub fallback on other platforms)
        // SAFETY: __errno_location() returns a valid thread-local errno pointer; create_backing_fd creates
        // an anonymous, sealed memory file descriptor sized to size_bytes.
        let fd = unsafe {
            #[cfg(target_os = "linux")]
            {
                if SIMULATE_FAIL.with(|f| f.get()) {
                    *libc::__errno_location() = libc::ENOMEM;
                    return Err(std::io::Error::last_os_error());
                }
            }
            create_backing_fd(size_bytes, false)?
        };

        // 2. Reserve contiguous virtual space (2x size)
        let total_size = size_bytes.checked_mul(2).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "size_bytes * 2 overflowed",
            )
        })?;

        // SAFETY: try_mmap_huge with null address requests a contiguous unmapped address reservation
        // of total_size (2 * size_bytes) from the kernel; return value is checked against MAP_FAILED.
        let base_ptr = unsafe {
            if SIMULATE_FAIL.with(|f| f.get()) {
                *libc::__errno_location() = libc::ENOMEM;
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
        let collapse_rc = unsafe {
            libc::madvise(base_ptr, size_bytes, MADV_HUGEPAGE);
            libc::madvise(base_ptr, size_bytes, libc::MADV_COLLAPSE)
        };
        if collapse_rc == 0 {
            MIRROR_BUF_HUGEPAGE_STATE
                .fetch_max(HUGEPAGE_STATE_THP, std::sync::atomic::Ordering::Relaxed);
        }

        // SAFETY: Both virtual mirrors maintain open references to the backing file in kernel VM state;
        // closing the user-space fd avoids descriptor leaks without unmapping the memory.
        unsafe { libc::close(fd) };

        Ok(Self {
            ptr: base_ptr as *mut T,
            size_elements,
            _marker: PhantomData,
        })
    }

    /// Attempts creation with explicit 2 MB huge pages, honouring element alignment.
    #[cold]
    fn try_new_huge_aligned(
        requested_bytes: usize,
        huge_page_size: usize,
        elem_stride: usize,
    ) -> std::io::Result<Self> {
        let element_size = std::mem::size_of::<T>();

        let align_bytes = match lcm(huge_page_size, elem_stride) {
            Some(val) => val,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "lcm(huge_page_size, elem_stride) overflowed",
                ));
            }
        };
        let size_bytes = match round_up_to_multiple(requested_bytes, align_bytes) {
            Some(v) => v,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "size_bytes calculation overflowed",
                ));
            }
        };
        let size_elements = size_bytes / element_size;

        // 1. Create HugeTLB-backed memfd (falls back to regular memfd)
        // SAFETY: __errno_location() returns a valid thread-local errno pointer; create_backing_fd creates
        // an anonymous HugeTLB-backed memfd sized to size_bytes.
        let fd = unsafe {
            if SIMULATE_FAIL.with(|f| f.get()) {
                *libc::__errno_location() = libc::ENOMEM;
                return Err(std::io::Error::last_os_error());
            }
            create_backing_fd(size_bytes, true)?
        };

        let total_size = size_bytes.checked_mul(2).ok_or_else(|| {
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

        Ok(Self {
            ptr: base_ptr as *mut T,
            size_elements,
            _marker: PhantomData,
        })
    }
}
