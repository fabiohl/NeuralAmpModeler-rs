// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! macOS backing-store tests for `create_backing_fd`.

use super::create_backing_fd_with_seq;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};

/// A stale shared-memory object occupying the next names (a pid reused after
/// a process was killed between shm_open and shm_unlink) must not make the
/// allocation fail: `create_backing_fd` retries with the next sequence number.
///
/// Uses a private sequence counter far from the production counter's range, so
/// concurrent allocations in other tests cannot move it: the counter must end
/// exactly one past the 8 occupied names.
#[test]
fn backing_fd_skips_names_left_by_a_previous_process() {
    const START: u64 = 900_000_000;
    const STALE: u64 = 8;
    let pid = std::process::id();
    let stale: Vec<CString> = (START..START + STALE)
        .map(|n| CString::new(format!("/nam.{pid}.{n}")).unwrap())
        .collect();
    for name in &stale {
        // SAFETY: `name` is a valid NUL-terminated C string.
        let fd = unsafe {
            libc::shm_open(
                name.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                0o600 as libc::c_uint,
            )
        };
        assert_ne!(fd, -1, "could not pre-create {name:?}");
        // SAFETY: fd was just returned by shm_open.
        unsafe { libc::close(fd) };
    }

    let seq = AtomicU64::new(START);
    // SAFETY: the size is valid; the returned descriptor is closed below.
    let result = unsafe { create_backing_fd_with_seq(4096, &seq) };

    for name in &stale {
        // SAFETY: `name` is a valid NUL-terminated C string.
        unsafe { libc::shm_unlink(name.as_ptr()) };
    }
    let fd = result.expect("create_backing_fd must skip names that already exist");
    // SAFETY: `fd` is the descriptor returned above.
    unsafe { libc::close(fd) };
    assert_eq!(
        seq.load(Ordering::Relaxed),
        START + STALE + 1,
        "expected {STALE} collisions skipped, then one successful name"
    );
}
