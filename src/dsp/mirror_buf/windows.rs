// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Win32 virtual memory placeholder reservation, splitting, mapping, and cleanup
// adhere to transactional rollback invariants with zero handle/view/placeholder leaks.
#![warn(clippy::undocumented_unsafe_blocks)]

#[cfg(all(not(miri), windows))]
use super::SIMULATE_FAIL;

/// Returns the Win32 virtual memory allocation granularity (`dwAllocationGranularity`),
/// verified to be a non-zero power of two.
#[cfg(all(not(miri), windows))]
pub(super) fn allocation_granularity() -> std::io::Result<usize> {
    let mut sys_info = windows_sys::Win32::System::SystemInformation::SYSTEM_INFO::default();
    // SAFETY: `sys_info` is a stack-allocated, zero-initialized `SYSTEM_INFO` struct;
    // `GetSystemInfo` writes only into its fields and never fails.
    unsafe {
        windows_sys::Win32::System::SystemInformation::GetSystemInfo(&mut sys_info);
    }
    let granularity = sys_info.dwAllocationGranularity as usize;
    if granularity == 0 || !granularity.is_power_of_two() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Win32 allocation granularity {granularity} is not a non-zero power of two"),
        ));
    }
    Ok(granularity)
}

/// Allocates two contiguous mirrored Win32 section views of `size_bytes` each.
#[cfg(all(not(miri), windows))]
#[cold]
pub(super) fn alloc_mirrored(size_bytes: usize) -> std::io::Result<*mut u8> {
    let mut ops = RealWin32VmOps::default();
    create_win32_mirrored_with(&mut ops, size_bytes)
}

/// Unmaps both contiguous Win32 section views (`[0..size_bytes)` and `[size_bytes..2*size_bytes)`).
///
/// # Safety
/// `base` must be the base address returned by [`create_win32_mirrored_with`] for a
/// mirrored buffer of `size_bytes` bytes per half, and this must be called at most once (on `Drop`).
#[cfg(all(not(miri), windows))]
pub(super) unsafe fn unmap_win32_views(base: *mut u8, size_bytes: usize) {
    use core::ffi::c_void;
    use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};

    // SAFETY: `base` and `base.add(size_bytes)` are the two distinct mapped views created by
    // `MapViewOfFile3` in `create_win32_mirrored_with`. Each view must be unmapped individually.
    unsafe {
        UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
            Value: base as *mut c_void,
        });
        UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
            Value: base.add(size_bytes) as *mut c_void,
        });
    }
}

/// Stages of the Win32 placeholder/mapping initialization transaction (`S0`–`S4`).
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Win32Stage {
    /// S0: `VirtualAlloc2` 2N placeholder reservation.
    S0ReservePlaceholder,
    /// S1: `VirtualFree(..., MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)` split.
    S1SplitPlaceholder,
    /// S2: `CreateFileMappingW` anonymous pagefile-backed section.
    S2CreateMapping,
    /// S3: 1st `MapViewOfFile3(..., MEM_REPLACE_PLACEHOLDER)` at `[0..N)`.
    S3MapFirstView,
    /// S4: 2nd `MapViewOfFile3(..., MEM_REPLACE_PLACEHOLDER)` at `[N..2N)`.
    S4MapSecondView,
}

#[cfg(all(test, not(miri), windows))]
thread_local! {
    pub(super) static WIN32_FAULT_STAGE: std::cell::Cell<Option<Win32Stage>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(all(test, not(miri), windows))]
pub(super) fn set_win32_fault_stage(stage: Option<Win32Stage>) {
    WIN32_FAULT_STAGE.with(|c| c.set(stage));
}

/// Abstract Win32 virtual-memory operations enabling both real kernel calls
/// and deterministic transactional fault-injection verification.
pub(super) trait Win32VmOps {
    fn alloc_placeholder_2n(&mut self, total_size: usize) -> std::io::Result<*mut u8>;
    fn split_placeholder(&mut self, base: *mut u8, half_size: usize) -> std::io::Result<()>;
    fn free_placeholder(&mut self, addr: *mut u8);
    fn create_section(&mut self, size_bytes: usize) -> std::io::Result<usize>;
    fn close_section(&mut self, handle: usize);
    fn map_view(
        &mut self,
        handle: usize,
        base: *mut u8,
        size_bytes: usize,
    ) -> std::io::Result<*mut u8>;
    fn unmap_view(&mut self, base: *mut u8);
}

/// Executes the 6-stage transactional Win32 mirror setup (`S0`–`S5`) with
/// immediate error capture and zero-leak rollback on partial failures.
pub(super) fn create_win32_mirrored_with<O: Win32VmOps>(
    ops: &mut O,
    size_bytes: usize,
) -> std::io::Result<*mut u8> {
    let total_size = size_bytes.checked_mul(2).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "size_bytes * 2 overflowed",
        )
    })?;

    // S0: Reserve contiguous 2N placeholder range.
    let base = ops.alloc_placeholder_2n(total_size)?;

    // S1: Split 2N placeholder into [0..N) and [N..2N).
    if let Err(err) = ops.split_placeholder(base, size_bytes) {
        ops.free_placeholder(base);
        return Err(err);
    }

    let second_half = base.wrapping_add(size_bytes);

    // S2: Create anonymous pagefile-backed section of size N.
    let section = match ops.create_section(size_bytes) {
        Ok(h) => h,
        Err(err) => {
            ops.free_placeholder(base);
            ops.free_placeholder(second_half);
            return Err(err);
        }
    };

    // S3: Map 1st view at [0..N).
    if let Err(err) = ops.map_view(section, base, size_bytes) {
        ops.close_section(section);
        ops.free_placeholder(base);
        ops.free_placeholder(second_half);
        return Err(err);
    }

    // S4: Map 2nd view (mirror) at [N..2N).
    if let Err(err) = ops.map_view(section, second_half, size_bytes) {
        ops.unmap_view(base);
        ops.free_placeholder(second_half);
        ops.close_section(section);
        return Err(err);
    }

    // S5: Close section handle immediately; mapped views retain kernel reference.
    ops.close_section(section);
    Ok(base)
}

#[cfg(all(not(miri), windows))]
#[derive(Default)]
struct RealWin32VmOps {
    #[cfg(test)]
    map_call_index: u8,
}

#[cfg(all(not(miri), windows))]
impl RealWin32VmOps {
    #[cfg(test)]
    fn check_fault(stage: Win32Stage) -> std::io::Result<()> {
        if WIN32_FAULT_STAGE.with(|c| c.get()) == Some(stage) {
            let synthetic_code = match stage {
                Win32Stage::S0ReservePlaceholder => 8, // ERROR_NOT_ENOUGH_MEMORY
                Win32Stage::S1SplitPlaceholder => 87,  // ERROR_INVALID_PARAMETER
                Win32Stage::S2CreateMapping => 1455,   // ERROR_COMMITMENT_LIMIT
                Win32Stage::S3MapFirstView => 487,     // ERROR_INVALID_ADDRESS
                Win32Stage::S4MapSecondView => 14,     // ERROR_OUTOFMEMORY
            };
            return Err(std::io::Error::from_raw_os_error(synthetic_code));
        }
        Ok(())
    }
}

#[cfg(all(not(miri), windows))]
impl Win32VmOps for RealWin32VmOps {
    fn alloc_placeholder_2n(&mut self, total_size: usize) -> std::io::Result<*mut u8> {
        use std::ptr;
        use windows_sys::Win32::System::Memory::{
            MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, VirtualAlloc2,
        };

        if SIMULATE_FAIL.with(|f| f.get()) {
            return Err(std::io::Error::from_raw_os_error(8));
        }
        #[cfg(test)]
        Self::check_fault(Win32Stage::S0ReservePlaceholder)?;

        // SAFETY: `VirtualAlloc2` with null process/base reserves a fresh 2N placeholder
        // range in the current process address space; null return is checked below.
        let base = unsafe {
            VirtualAlloc2(
                ptr::null_mut(),
                ptr::null(),
                total_size,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                ptr::null_mut(),
                0,
            )
        };
        if base.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        Ok(base as *mut u8)
    }

    fn split_placeholder(&mut self, base: *mut u8, half_size: usize) -> std::io::Result<()> {
        use core::ffi::c_void;
        use windows_sys::Win32::System::Memory::{
            MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, VirtualFree,
        };

        #[cfg(test)]
        Self::check_fault(Win32Stage::S1SplitPlaceholder)?;

        // SAFETY: `base` is an active 2N placeholder reservation returned by `VirtualAlloc2`;
        // `MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER` splits it at `half_size` into `[0..N)` and `[N..2N)`.
        let ok = unsafe {
            VirtualFree(
                base as *mut c_void,
                half_size,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    fn free_placeholder(&mut self, addr: *mut u8) {
        use core::ffi::c_void;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};

        // SAFETY: `addr` is an unmapped placeholder base reserved by `VirtualAlloc2`;
        // size `0` with `MEM_RELEASE` releases the entire placeholder region.
        unsafe {
            VirtualFree(addr as *mut c_void, 0, MEM_RELEASE);
        }
    }

    fn create_section(&mut self, size_bytes: usize) -> std::io::Result<usize> {
        use std::ptr;
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, PAGE_READWRITE};

        #[cfg(test)]
        Self::check_fault(Win32Stage::S2CreateMapping)?;

        let high = ((size_bytes as u64) >> 32) as u32;
        let low = size_bytes as u32;
        // SAFETY: `INVALID_HANDLE_VALUE` requests an anonymous pagefile-backed section
        // with null security attributes and null name; return value is checked against null.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                high,
                low,
                ptr::null(),
            )
        };
        if handle.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        Ok(handle as usize)
    }

    fn close_section(&mut self, handle: usize) {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};

        // SAFETY: `handle` is an open section handle returned by `CreateFileMappingW`.
        unsafe {
            CloseHandle(handle as HANDLE);
        }
    }

    fn map_view(
        &mut self,
        handle: usize,
        base: *mut u8,
        size_bytes: usize,
    ) -> std::io::Result<*mut u8> {
        use core::ffi::c_void;
        use std::ptr;
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::System::Memory::{
            MEM_REPLACE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3, PAGE_READWRITE,
            UnmapViewOfFile,
        };

        #[cfg(test)]
        {
            let current_stage = if self.map_call_index == 0 {
                Win32Stage::S3MapFirstView
            } else {
                Win32Stage::S4MapSecondView
            };
            self.map_call_index = self.map_call_index.saturating_add(1);
            Self::check_fault(current_stage)?;
        }

        // SAFETY: `handle` is a valid section of `size_bytes`, and `base` is an unmapped
        // placeholder of exact length `size_bytes` ready to be replaced by `MapViewOfFile3`.
        let view = unsafe {
            MapViewOfFile3(
                handle as HANDLE,
                ptr::null_mut(),
                base as *const c_void,
                0,
                size_bytes,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                ptr::null_mut(),
                0,
            )
        };
        if view.Value.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        if view.Value != base as *mut c_void {
            // SAFETY: defensive cleanup if the kernel returned an unexpected address.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            }
            return Err(std::io::Error::other(
                "MapViewOfFile3 returned unexpected view base address",
            ));
        }
        Ok(view.Value as *mut u8)
    }

    fn unmap_view(&mut self, base: *mut u8) {
        use core::ffi::c_void;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};

        // SAFETY: `base` is a valid mapped view address returned by `MapViewOfFile3`.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: base as *mut c_void,
            });
        }
    }
}
