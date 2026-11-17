// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Shared alignment math and compile-time backend dispatch for MirroredBuffer;
// OS virtual memory operations reside in private `posix` and `windows` submodules.
#![warn(clippy::undocumented_unsafe_blocks)]

use super::{MirrorSafe, MirroredBuffer};
#[cfg(unix)]
#[cfg(target_os = "linux")]
use crate::math::common::huge_alloc::HUGE_PAGE_2M;
use std::marker::PhantomData;

pub(super) const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Computes the least common multiple (LCM) of `a` and `b`, or `None` on overflow.
pub(super) const fn lcm(a: usize, b: usize) -> Option<usize> {
    (a / gcd(a, b)).checked_mul(b)
}

/// Rounds `value` up to the next multiple of `align`, or `None` on overflow.
pub(super) fn round_up_to_multiple(value: usize, align: usize) -> Option<usize> {
    let padded = value.checked_add(align - 1)?;
    Some(padded - (padded % align))
}

#[inline]
fn invalid_input(msg: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, msg)
}

/// Returns the OS virtual memory allocation granularity (`_SC_PAGESIZE` / `dwAllocationGranularity`).
pub(super) fn allocation_granularity() -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        super::posix::allocation_granularity()
    }
    #[cfg(all(not(miri), windows))]
    {
        super::windows::allocation_granularity()
    }
    #[cfg(not(any(unix, all(not(miri), windows))))]
    {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "MirroredBuffer is only supported on Unix and Windows",
        ))
    }
}

fn effective_granularity() -> std::io::Result<(usize, bool)> {
    #[cfg(all(test, target_os = "linux"))]
    {
        let override_val = super::mirror_buf_test::granularity_override();
        if override_val > 0 {
            return Ok((override_val, true));
        }
    }
    Ok((allocation_granularity()?, false))
}

impl<T: MirrorSafe> MirroredBuffer<T> {
    /// Creates a new mirrored buffer with huge-page preference (equivalent to `new_aligned(requested_size, 1)`).
    #[cold]
    pub fn new(requested_size: usize) -> std::io::Result<Self> {
        Self::new_aligned(requested_size, 1)
    }

    /// Creates a mirrored buffer guaranteeing `size_elements % elem_multiple == 0`.
    ///
    /// Rounds `size_bytes` up to `lcm(granularity, elem_multiple * size_of::<T>())`
    /// (or `lcm(2 MiB, ...)` for Linux explicit huge pages) during cold model load.
    #[cold]
    pub fn new_aligned(requested_size: usize, elem_multiple: usize) -> std::io::Result<Self> {
        const {
            assert!(
                std::mem::align_of::<T>() <= 64,
                "MirroredBuffer element alignment must not exceed 64 bytes"
            );
        };
        let (page_size, granularity_overridden) = effective_granularity()?;
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
            return Err(invalid_input(
                "MirroredBuffer does not support Zero Sized Types",
            ));
        }
        if requested_size == 0 {
            return Err(invalid_input("requested_size must be greater than zero"));
        }
        if elem_multiple == 0 {
            return Err(invalid_input("elem_multiple must be greater than zero"));
        }
        let requested_bytes = requested_size
            .checked_mul(element_size)
            .ok_or_else(|| invalid_input("requested_size * element_size overflowed"))?;
        let total_chunk = requested_bytes
            .checked_mul(2)
            .ok_or_else(|| invalid_input("requested_bytes * 2 overflowed"))?;
        let elem_stride = elem_multiple
            .checked_mul(element_size)
            .ok_or_else(|| invalid_input("elem_multiple * element_size overflowed"))?;

        #[cfg(target_os = "linux")]
        if !cfg!(miri)
            && !granularity_overridden
            && total_chunk >= HUGE_PAGE_2M
            && let Ok((ptr, size_bytes)) =
                super::posix::try_alloc_huge_mirrored(requested_bytes, elem_stride)
        {
            #[cfg(test)]
            super::mirror_buf_test::record_test_alloc(requested_bytes, size_bytes);
            return Ok(Self {
                ptr: ptr as *mut T,
                size_elements: size_bytes / element_size,
                _marker: PhantomData,
            });
        }
        #[cfg(not(target_os = "linux"))]
        let _ = (granularity_overridden, total_chunk);

        let align_bytes = lcm(page_size, elem_stride)
            .ok_or_else(|| invalid_input("lcm(page_size, elem_stride) overflowed"))?;
        let size_bytes = round_up_to_multiple(requested_bytes, align_bytes)
            .ok_or_else(|| invalid_input("size_bytes calculation overflowed"))?;
        let size_elements = size_bytes / element_size;

        if cfg!(miri) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "MirroredBuffer allocation is not supported under Miri",
            ));
        }

        #[cfg(unix)]
        let ptr = {
            let p = super::posix::alloc_mirrored(size_bytes)?;
            #[cfg(all(test, target_os = "linux"))]
            super::mirror_buf_test::record_test_alloc(requested_bytes, size_bytes);
            p
        };
        #[cfg(all(not(miri), windows))]
        let ptr = super::windows::alloc_mirrored(size_bytes)?;
        #[cfg(not(any(unix, all(not(miri), windows))))]
        let ptr: *mut u8 = {
            let _ = size_elements;
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "MirroredBuffer is only supported on Unix and Windows",
            ));
        };

        Ok(Self {
            ptr: ptr as *mut T,
            size_elements,
            _marker: PhantomData,
        })
    }
}
