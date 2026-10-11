// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! macOS backing-store tests for `create_backing_fd`.

use super::{SHM_NAME_SEQ, create_backing_fd};
use std::ffi::CString;
use std::sync::atomic::Ordering;

/// A stale shared-memory object occupying the next names (a pid reused after
/// a process was killed between shm_open and shm_unlink) must not make the
/// allocation fail: `create_backing_fd` retries with the next sequence number.
#[test]
fn backing_fd_skips_names_left_by_a_previous_process() {
    let pid = std::process::id();
    let next = SHM_NAME_SEQ.load(Ordering::Relaxed);
    let stale: Vec<CString> = (next..next + 8)
        .map(|seq| CString::new(format!("/nam.{pid}.{}", seq % 1_000_000_000)).unwrap())
        .collect();
    let mut created = Vec::new();
    for name in &stale {
        // SAFETY: `name` is a valid NUL-terminated C string.
        let fd = unsafe {
            libc::shm_open(
                name.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                0o600 as libc::c_uint,
            )
        };
        if fd != -1 {
            // SAFETY: fd was just returned by shm_open.
            unsafe { libc::close(fd) };
            created.push(name);
        }
    }

    // SAFETY: the size is valid; the returned descriptor is closed below.
    let result = unsafe { create_backing_fd(4096, false) };

    for name in &created {
        // SAFETY: `name` is a valid NUL-terminated C string.
        unsafe { libc::shm_unlink(name.as_ptr()) };
    }
    let fd = result.expect("create_backing_fd must skip names that already exist");
    // SAFETY: `fd` is the descriptor returned above.
    unsafe { libc::close(fd) };
}
