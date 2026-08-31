// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use libc::sysconf;
use serial_test::serial;
use std::sync::atomic::Ordering;

#[test]
fn test_mirror_buf_page_alignment() -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
    let page_size = unsafe { sysconf(libc::_SC_PAGESIZE) } as usize;
    let element_size = std::mem::size_of::<f32>();

    // Request 1 element, should round to 1 page
    let buf = MirroredBuffer::<f32>::new(1)?;
    let expected_elements = page_size / element_size;

    assert_eq!(buf.size(), expected_elements);
    assert_eq!(buf.len(), expected_elements * 2);
    Ok(())
}

#[test]
fn test_mirror_buf_mirroring() -> Result<(), Box<dyn std::error::Error>> {
    // Mirroring Test: Verifies that the virtual memory "trick" works.
    // Any value written to the first half should instantly appear in the
    // second half (the mirror), since both windows point to the same physical location.
    // Creates a small buffer (will be rounded to 1 page)
    let mut buf = MirroredBuffer::<u32>::new(1)?;
    let size = buf.size();

    // 1. Write at the start of the first half
    buf[0] = 0x12345678;
    // Should be visible at the start of the second half (mirror)
    assert_eq!(buf[size], 0x12345678);

    // 2. Write at the end of the first half
    buf[size - 1] = 0xDEADBEEF;
    // Should be visible at the end of the second half
    assert_eq!(buf[2 * size - 1], 0xDEADBEEF);

    // 3. Contiguous access crossing the boundary (the "Ace in the Hole")
    // Let's write a sequence of values that crosses exactly the buffer midpoint.
    // In a regular buffer this would require two loops or an 'if', but here it is linear.
    let middle = size;
    let start = middle - 8;
    for i in 0..16 {
        buf[start + i] = i as u32;
    }

    // Verifies that the first half (original) reflects the changes
    // The first 8 values were written at the end of the first half
    for i in 0..8 {
        assert_eq!(buf[size - 8 + i], i as u32);
    }
    // The next 8 values were written at the start of the second half,
    // which should have modified the start of the physical FIRST half.
    for i in 8..16 {
        assert_eq!(buf[i - 8], i as u32);
    }
    Ok(())
}

#[test]
fn test_mirror_buf_clone() -> Result<(), Box<dyn std::error::Error>> {
    let mut buf = MirroredBuffer::<i32>::new(100)?;
    buf[0] = 42;

    let buf2 = buf.clone();
    assert_eq!(buf2[0], 42);
    assert_eq!(buf2.size(), buf.size());

    // Modifies the original, the clone should remain unchanged
    buf[0] = 99;
    assert_eq!(buf2[0], 42, "MirroredBuffer clones must be independent");
    Ok(())
}

#[test]
fn test_mirror_buf_zst_error() {
    assert!(MirroredBuffer::<()>::new(1024).is_err());
}

#[test]
fn test_mirror_buf_zero_size_error() {
    assert!(MirroredBuffer::<f32>::new(0).is_err());
}

#[test]
fn test_mirror_buf_large_allocation() -> Result<(), Box<dyn std::error::Error>> {
    // Tests allocation of ~1MB
    let size = 1024 * 1024 / 4;
    let buf = MirroredBuffer::<f32>::new(size)?;
    assert!(buf.size() >= size);
    // Just ensures no panic occurred and mmap succeeded
    Ok(())
}

#[test]
fn test_mirror_buf_debug() -> Result<(), Box<dyn std::error::Error>> {
    let buf = MirroredBuffer::<f32>::new(1024)?;
    let debug_str = format!("{:?}", buf);
    assert!(debug_str.contains("MirroredBuffer"));
    assert!(debug_str.contains("ptr"));
    assert!(debug_str.contains("size_elements"));
    Ok(())
}

#[test]
fn test_mirror_buf_channel_alignment() -> Result<(), Box<dyn std::error::Error>> {
    // MirroredBuffer::new_aligned guarantees size_elements % channels == 0
    // for all channel counts, including non-power-of-two channels (6, 12 — Lite).
    let min_buffer_frames = 3648usize;

    for &ch in &[4usize, 6, 8, 12, 16] {
        let min_elems = min_buffer_frames * ch;
        let buf = MirroredBuffer::<f32>::new_aligned(min_elems, ch)?;
        assert_eq!(
            buf.size() % ch,
            0,
            "MirroredBuffer size_elements={} must be divisible by channels={ch}",
            buf.size()
        );
    }

    Ok(())
}

#[test]
#[serial]
fn test_hugepage_state_is_monotonic() {
    let original = MIRROR_BUF_HUGEPAGE_STATE.load(Ordering::Relaxed);

    // Simulates successful >= 2MB allocation with HugeTLB (max state = 2).
    MIRROR_BUF_HUGEPAGE_STATE.fetch_max(HUGEPAGE_STATE_HUGETLB, Ordering::Relaxed);
    assert_eq!(
        MIRROR_BUF_HUGEPAGE_STATE.load(Ordering::Relaxed),
        HUGEPAGE_STATE_HUGETLB
    );

    // Small THP allocation (state 1) must not downgrade the global state (2 -> 1).
    MIRROR_BUF_HUGEPAGE_STATE.fetch_max(HUGEPAGE_STATE_THP, Ordering::Relaxed);
    assert_eq!(
        MIRROR_BUF_HUGEPAGE_STATE.load(Ordering::Relaxed),
        HUGEPAGE_STATE_HUGETLB,
        "global Huge Pages state was downgraded by a THP allocation"
    );

    // Standard allocation (state 0) also must not downgrade.
    MIRROR_BUF_HUGEPAGE_STATE.fetch_max(HUGEPAGE_STATE_STANDARD, Ordering::Relaxed);
    assert_eq!(
        MIRROR_BUF_HUGEPAGE_STATE.load(Ordering::Relaxed),
        HUGEPAGE_STATE_HUGETLB,
        "global Huge Pages state was downgraded by a standard allocation"
    );

    MIRROR_BUF_HUGEPAGE_STATE.store(original, Ordering::Relaxed);
}
