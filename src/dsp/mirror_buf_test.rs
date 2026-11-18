// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use serial_test::serial;
use std::sync::atomic::Ordering;

#[cfg(target_os = "linux")]
thread_local! {
    static TEST_GRANULARITY_OVERRIDE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_ALLOC_RECORD: std::cell::RefCell<Option<Vec<(usize, usize)>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(target_os = "linux")]
struct GranularityOverrideGuard {
    prev: usize,
}

#[cfg(target_os = "linux")]
impl GranularityOverrideGuard {
    fn new(granularity: usize) -> Self {
        let prev = TEST_GRANULARITY_OVERRIDE.with(|c| {
            let old = c.get();
            c.set(granularity);
            old
        });
        Self { prev }
    }
}

#[cfg(target_os = "linux")]
impl Drop for GranularityOverrideGuard {
    fn drop(&mut self) {
        TEST_GRANULARITY_OVERRIDE.with(|c| c.set(self.prev));
    }
}

#[cfg(target_os = "linux")]
struct AllocRecordGuard;

#[cfg(target_os = "linux")]
impl AllocRecordGuard {
    fn start() -> Self {
        TEST_ALLOC_RECORD.with(|r| *r.borrow_mut() = Some(Vec::new()));
        Self
    }

    fn take_records(&self) -> Vec<(usize, usize)> {
        TEST_ALLOC_RECORD.with(|r| {
            r.borrow_mut()
                .as_mut()
                .map(std::mem::take)
                .unwrap_or_default()
        })
    }
}

#[cfg(target_os = "linux")]
impl Drop for AllocRecordGuard {
    fn drop(&mut self) {
        TEST_ALLOC_RECORD.with(|r| *r.borrow_mut() = None);
    }
}

#[cfg(target_os = "linux")]
pub(super) fn granularity_override() -> usize {
    TEST_GRANULARITY_OVERRIDE.with(|c| c.get())
}

#[cfg(target_os = "linux")]
pub(super) fn record_test_alloc(requested_bytes: usize, size_bytes: usize) {
    TEST_ALLOC_RECORD.with(|r| {
        if let Some(vec) = r.borrow_mut().as_mut() {
            vec.push((requested_bytes, size_bytes));
        }
    });
}

#[test]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires memfd_create/mmap (FFI), which Miri cannot interpret; the alias/provenance of the double mapping is validated by the native test suite"
)]
fn test_mirror_buf_page_alignment() -> Result<(), Box<dyn std::error::Error>> {
    let page_size = alloc::allocation_granularity()?;
    let element_size = std::mem::size_of::<f32>();

    // Request 1 element, should round to 1 allocation granularity unit
    let buf = MirroredBuffer::<f32>::new(1)?;
    let expected_elements = page_size / element_size;

    assert_eq!(buf.size(), expected_elements);
    assert_eq!(buf.len(), expected_elements * 2);
    Ok(())
}

#[test]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires memfd_create/mmap (FFI), which Miri cannot interpret; the alias/provenance of the double mapping is validated by the native test suite"
)]
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
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires memfd_create/mmap (FFI), which Miri cannot interpret; the alias/provenance of the double mapping is validated by the native test suite"
)]
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

// Note: ZST error is statically prevented at compile-time by the MirrorSafe trait bound
// (verified via compile_fail doctests on MirroredBuffer).

#[test]
fn test_mirror_buf_zero_size_error() {
    assert!(MirroredBuffer::<f32>::new(0).is_err());
}

#[test]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires memfd_create/mmap (FFI), which Miri cannot interpret; the alias/provenance of the double mapping is validated by the native test suite"
)]
fn test_mirror_buf_large_allocation() -> Result<(), Box<dyn std::error::Error>> {
    // Tests allocation of ~1MB
    let size = 1024 * 1024 / 4;
    let buf = MirroredBuffer::<f32>::new(size)?;
    assert!(buf.size() >= size);
    // Just ensures no panic occurred and mmap succeeded
    Ok(())
}

#[test]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires memfd_create/mmap (FFI), which Miri cannot interpret; the alias/provenance of the double mapping is validated by the native test suite"
)]
fn test_mirror_buf_debug() -> Result<(), Box<dyn std::error::Error>> {
    let buf = MirroredBuffer::<f32>::new(1024)?;
    let debug_str = format!("{:?}", buf);
    assert!(debug_str.contains("MirroredBuffer"));
    assert!(debug_str.contains("ptr"));
    assert!(debug_str.contains("size_elements"));
    Ok(())
}

#[test]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires memfd_create/mmap (FFI), which Miri cannot interpret; the alias/provenance of the double mapping is validated by the native test suite"
)]
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

#[test]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires memfd_create/mmap (FFI), which Miri cannot interpret; the alias/provenance of the double mapping is validated by the native test suite"
)]
fn test_mirror_buf_alignment_invariants_r8() -> Result<(), Box<dyn std::error::Error>> {
    #[repr(align(64))]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Align64([u8; 64]);

    impl private::Sealed for Align64 {}
    // SAFETY: Align64 is Plain Old Data (Copy, zero-valid, no pointers/drop).
    unsafe impl MirrorSafe for Align64 {}

    // Types with align <= 64 compile and instantiate cleanly
    let buf_u8 = MirroredBuffer::<u8>::new(1)?;
    assert!(!buf_u8.is_empty());

    let buf_f64 = MirroredBuffer::<f64>::new(1)?;
    assert!(!buf_f64.is_empty());

    let buf_a64 = MirroredBuffer::<Align64>::new(1)?;
    assert!(!buf_a64.is_empty());
    assert_eq!(buf_a64.as_ptr() as usize % 64, 0);

    let buf_a64_aligned = MirroredBuffer::<Align64>::new_aligned(2, 2)?;
    assert_eq!(buf_a64_aligned.size() % 2, 0);
    assert_eq!(buf_a64_aligned.as_ptr() as usize % 64, 0);

    Ok(())
}

#[test]
fn test_mirror_buf_overflow_handling() {
    let result = MirroredBuffer::<u8>::new(usize::MAX);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn test_mirror_buf_struct_layout_16_bytes() {
    assert_eq!(std::mem::size_of::<MirroredBuffer<f32>>(), 16);
    assert_eq!(std::mem::size_of::<MirroredBuffer<u8>>(), 16);
    assert_eq!(std::mem::align_of::<MirroredBuffer<f32>>(), 8);
}

#[test]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires OS virtual memory FFI, which Miri cannot interpret"
)]
fn test_mirror_buf_bidirectional_aliasing_and_seam_slice() -> Result<(), Box<dyn std::error::Error>>
{
    let mut buf = MirroredBuffer::<f32>::new(1024)?;
    let n = buf.size();
    let k = 32usize;
    assert!(n > k * 2);

    // 1. Write to 1st half -> immediately visible in 2nd half.
    buf[0] = 1.25;
    buf[n - 1] = -3.5;
    assert_eq!(buf[n].to_bits(), 1.25_f32.to_bits());
    assert_eq!(buf[2 * n - 1].to_bits(), (-3.5_f32).to_bits());

    // 2. Write to 2nd half -> immediately visible in 1st half.
    buf[n + 7] = 42.5;
    buf[2 * n - 3] = -99.75;
    assert_eq!(buf[7].to_bits(), 42.5_f32.to_bits());
    assert_eq!(buf[n - 3].to_bits(), (-99.75_f32).to_bits());

    // 3. Populate both sides of the N..2N seam and read one contiguous slice &buf[n - k .. n + k].
    for i in 0..k {
        buf[n - k + i] = (1000 + i) as f32;
        buf[i] = (2000 + i) as f32;
    }
    {
        let slice: &[f32] = &buf;
        let seam = &slice[n - k..n + k];
        assert_eq!(seam.len(), 2 * k);
        for i in 0..k {
            assert_eq!(seam[i].to_bits(), ((1000 + i) as f32).to_bits());
            assert_eq!(seam[k + i].to_bits(), ((2000 + i) as f32).to_bits());
        }
    }

    // 4. Single bounded &mut [T] window crossing the N seam wraps cleanly into [0..k].
    {
        let seam_mut: &mut [f32] = buf.window_mut(n - k, 2 * k);
        for (idx, slot) in seam_mut.iter_mut().enumerate() {
            *slot = (idx as f32) + 0.5;
        }
    }
    for i in 0..k {
        assert_eq!(buf[n - k + i].to_bits(), ((i as f32) + 0.5).to_bits());
        assert_eq!(buf[i].to_bits(), (((k + i) as f32) + 0.5).to_bits());
    }

    Ok(())
}

#[test]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires OS virtual memory FFI, which Miri cannot interpret"
)]
fn test_mirror_buf_repeated_create_write_drop_stress() -> Result<(), Box<dyn std::error::Error>> {
    for iter in 0..300usize {
        let mut buf = if iter.is_multiple_of(10) {
            MirroredBuffer::<f32>::new_aligned(3648 * 12, 12)?
        } else {
            MirroredBuffer::<f32>::new(1)?
        };
        let n = buf.size();
        let sentinel_a = (iter as f32) + 0.25;
        let sentinel_b = -((iter as f32) + 0.75);

        buf[0] = sentinel_a;
        buf[n - 1] = sentinel_b;
        assert_eq!(buf[n].to_bits(), sentinel_a.to_bits());
        assert_eq!(buf[2 * n - 1].to_bits(), sentinel_b.to_bits());

        let seam_mut = buf.window_mut(n - 4, 8);
        for (j, slot) in seam_mut.iter_mut().enumerate() {
            *slot = (iter * 8 + j) as f32;
        }
        assert_eq!(buf[0].to_bits(), ((iter * 8 + 4) as f32).to_bits());
        assert_eq!(buf[n - 1].to_bits(), ((iter * 8 + 3) as f32).to_bits());
    }
    Ok(())
}

#[test]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires OS virtual memory FFI, which Miri cannot interpret"
)]
fn test_mirror_buf_bounded_window_mut_limits_and_wrap() -> Result<(), Box<dyn std::error::Error>> {
    let mut buf = MirroredBuffer::<u32>::new(1)?;
    let n = buf.size();
    assert!(n >= 4);

    // 1. len == 0 at 0, N, 2N (valid) and 2N + 1 (rejected).
    assert_eq!(buf.try_window_mut(0, 0).map(|s| s.len()), Some(0));
    assert_eq!(buf.try_window_mut(n, 0).map(|s| s.len()), Some(0));
    assert_eq!(buf.try_window_mut(n * 2, 0).map(|s| s.len()), Some(0));
    assert!(buf.try_window_mut(n * 2 + 1, 0).is_none());

    // 2. len == N - 1 wrapping across N, plus split_at_mut on the bounded slice (physically disjoint).
    let half = n / 2;
    {
        let w = buf.window_mut(half, n - 1);
        assert_eq!(w.len(), n - 1);
        let (left, right) = w.split_at_mut(n - half);
        for (i, slot) in left.iter_mut().enumerate() {
            *slot = 10_000 + i as u32;
        }
        for (i, slot) in right.iter_mut().enumerate() {
            *slot = 20_000 + i as u32;
        }
    }
    for i in 0..(n - half) {
        assert_eq!(buf[half + i], 10_000 + i as u32);
        assert_eq!(buf[n + half + i], 10_000 + i as u32);
    }
    for i in 0..(half - 1) {
        assert_eq!(buf[i], 20_000 + i as u32);
        assert_eq!(buf[n + i], 20_000 + i as u32);
    }

    // 3. len == N wrapping across N touches every physical slot exactly once.
    buf.fill(0);
    {
        let w = buf.window_mut(half, n);
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = 0xA000_0000 + i as u32;
        }
    }
    for i in 0..n {
        let virt = half + i;
        let expected = 0xA000_0000 + i as u32;
        assert_eq!(buf[virt], expected);
        assert_eq!(buf[virt % n], expected);
    }

    // 4. len == N + 1, out-of-bounds, and overflow are rejected.
    assert!(buf.try_window_mut(0, n + 1).is_none());
    assert!(buf.try_window_mut(n - 1, n + 1).is_none());
    assert!(buf.try_window_mut(n + 1, n).is_none());
    assert!(buf.try_window_mut(usize::MAX, 1).is_none());
    assert!(buf.try_window_mut(1, usize::MAX).is_none());

    // 5. write_slice_at across the N seam.
    buf.fill(0);
    let payload = [11_u32, 22, 33, 44, 55, 66];
    buf.write_slice_at(n - 3, &payload);
    assert_eq!(buf[n - 3], 11);
    assert_eq!(buf[n - 1], 33);
    assert_eq!(buf[0], 44);
    assert_eq!(buf[2], 66);

    Ok(())
}

#[test]
#[should_panic(expected = "exceeds physical size")]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires OS virtual memory FFI, which Miri cannot interpret"
)]
fn test_mirror_buf_window_mut_rejects_len_gt_n() {
    let mut buf = MirroredBuffer::<u32>::new(1).expect("alloc");
    let n = buf.size();
    let _ = buf.window_mut(0, n + 1);
}

#[test]
#[should_panic(expected = "out of bounds")]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires OS virtual memory FFI, which Miri cannot interpret"
)]
fn test_mirror_buf_index_mut_rejects_2n() {
    let mut buf = MirroredBuffer::<u32>::new(1).expect("alloc");
    let n = buf.size();
    buf[n * 2] = 1;
}

#[test]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires OS virtual memory FFI, which Miri cannot interpret"
)]
fn test_mirror_buf_copy_within_ring_overlap_cases() -> Result<(), Box<dyn std::error::Error>> {
    let mut buf = MirroredBuffer::<u32>::new(1)?;
    let n = buf.size();
    assert!(n >= 16);

    let reset_pattern = |b: &mut MirroredBuffer<u32>| {
        let w = b.window_mut(0, n);
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = i as u32;
        }
    };

    // Case A: Physical seam forward/backward traps (`src` and `dest` in opposite virtual halves).
    for &(src_start, dest, len) in &[
        (0, n + 1, 8),
        (n, 1, 8),
        (1, n, 8),
        (n + 1, 0, 8),
        (n - 4, n - 2, 8),
        (n - 2, n - 4, 8),
    ] {
        reset_pattern(&mut buf);
        let expected: Vec<u32> = (0..len).map(|i| ((src_start + i) % n) as u32).collect();
        buf.copy_within(src_start..src_start + len, dest);
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                buf[dest + i],
                exp,
                "mismatch for src={src_start}..{}, dest={dest}, i={i}",
                src_start + len
            );
        }
    }

    // Case B: `len > n / 2` where both forward and backward directions wrap-overlap.
    // Includes `len = n - 2` (sinks + possible pure coset cycles) and `len = n` (pure cyclic shift).
    for &(src_start, dest, len) in &[
        (0, 2, n - 2),
        (3, 1, n - 2),
        (n - 3, 2, n - 1),
        (2, n - 3, n - 1),
        (0, 4, n),
        (n / 2, n / 2 + 6, n),
    ] {
        reset_pattern(&mut buf);
        let expected: Vec<u32> = (0..len).map(|i| ((src_start + i) % n) as u32).collect();
        buf.copy_within(src_start..src_start + len, dest);
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                buf[(dest + i) % n],
                exp,
                "wrap-overlap mismatch for src={src_start}..{}, dest={dest}, len={len}, i={i}",
                src_start + len
            );
        }
    }

    // Case C: `src_p == dst_p` (same physical span across halves) is a no-op.
    reset_pattern(&mut buf);
    buf.copy_within(0..16, n);
    for i in 0..n {
        assert_eq!(buf[i], i as u32);
    }

    Ok(())
}

#[test]
fn test_win32_tx_fault_injection_s0_to_s4() {
    use windows::{Win32Stage, Win32VmOps, create_win32_mirrored_with};

    struct MockWin32VmOps {
        fail_at: Option<Win32Stage>,
        next_handle: usize,
        active_placeholders: Vec<(usize, usize)>,
        active_sections: Vec<(usize, usize)>,
        active_views: Vec<(usize, usize)>,
        error_clobber_count: usize,
    }

    impl MockWin32VmOps {
        fn new(fail_at: Option<Win32Stage>) -> Self {
            Self {
                fail_at,
                next_handle: 100,
                active_placeholders: Vec::new(),
                active_sections: Vec::new(),
                active_views: Vec::new(),
                error_clobber_count: 0,
            }
        }

        fn stage_error(stage: Win32Stage) -> std::io::Error {
            let code = match stage {
                Win32Stage::S0ReservePlaceholder => 1000,
                Win32Stage::S1SplitPlaceholder => 1001,
                Win32Stage::S2CreateMapping => 1002,
                Win32Stage::S3MapFirstView => 1003,
                Win32Stage::S4MapSecondView => 1004,
            };
            std::io::Error::from_raw_os_error(code)
        }

        fn assert_no_leaks(&self) {
            assert!(
                self.active_placeholders.is_empty(),
                "leaked placeholders: {:?}",
                self.active_placeholders
            );
            assert!(
                self.active_sections.is_empty(),
                "leaked section handles: {:?}",
                self.active_sections
            );
            assert!(
                self.active_views.is_empty(),
                "leaked mapped views: {:?}",
                self.active_views
            );
        }
    }

    impl Win32VmOps for MockWin32VmOps {
        fn alloc_placeholder_2n(&mut self, total_size: usize) -> std::io::Result<*mut u8> {
            if self.fail_at == Some(Win32Stage::S0ReservePlaceholder) {
                return Err(Self::stage_error(Win32Stage::S0ReservePlaceholder));
            }
            let base = 0x1000_0000usize;
            self.active_placeholders.push((base, total_size));
            Ok(base as *mut u8)
        }

        fn split_placeholder(&mut self, base: *mut u8, half_size: usize) -> std::io::Result<()> {
            if self.fail_at == Some(Win32Stage::S1SplitPlaceholder) {
                return Err(Self::stage_error(Win32Stage::S1SplitPlaceholder));
            }
            let base_addr = base as usize;
            let pos = self
                .active_placeholders
                .iter()
                .position(|&(addr, sz)| addr == base_addr && sz == half_size * 2)
                .expect("S1 requires an active 2N placeholder");
            self.active_placeholders.remove(pos);
            self.active_placeholders.push((base_addr, half_size));
            self.active_placeholders
                .push((base_addr + half_size, half_size));
            Ok(())
        }

        fn free_placeholder(&mut self, addr: *mut u8) {
            self.error_clobber_count += 1;
            let addr_u = addr as usize;
            let pos = self
                .active_placeholders
                .iter()
                .position(|&(p, _)| p == addr_u)
                .expect("free_placeholder called on unknown/already-freed placeholder");
            self.active_placeholders.remove(pos);
        }

        fn create_section(&mut self, size_bytes: usize) -> std::io::Result<usize> {
            if self.fail_at == Some(Win32Stage::S2CreateMapping) {
                return Err(Self::stage_error(Win32Stage::S2CreateMapping));
            }
            let h = self.next_handle;
            self.next_handle += 1;
            self.active_sections.push((h, size_bytes));
            Ok(h)
        }

        fn close_section(&mut self, handle: usize) {
            self.error_clobber_count += 1;
            let pos = self
                .active_sections
                .iter()
                .position(|&(h, _)| h == handle)
                .expect("close_section called on unknown/already-closed handle");
            self.active_sections.remove(pos);
        }

        fn map_view(
            &mut self,
            handle: usize,
            base: *mut u8,
            size_bytes: usize,
        ) -> std::io::Result<*mut u8> {
            let is_first = self.active_views.is_empty();
            if is_first && self.fail_at == Some(Win32Stage::S3MapFirstView) {
                return Err(Self::stage_error(Win32Stage::S3MapFirstView));
            }
            if !is_first && self.fail_at == Some(Win32Stage::S4MapSecondView) {
                return Err(Self::stage_error(Win32Stage::S4MapSecondView));
            }
            assert!(
                self.active_sections
                    .iter()
                    .any(|&(h, sz)| h == handle && sz == size_bytes),
                "map_view called with invalid section handle/size"
            );
            let base_addr = base as usize;
            let pos = self
                .active_placeholders
                .iter()
                .position(|&(addr, sz)| addr == base_addr && sz == size_bytes)
                .expect("map_view requires matching placeholder");
            self.active_placeholders.remove(pos);
            self.active_views.push((base_addr, size_bytes));
            Ok(base)
        }

        fn unmap_view(&mut self, base: *mut u8) {
            self.error_clobber_count += 1;
            let base_addr = base as usize;
            let pos = self
                .active_views
                .iter()
                .position(|&(addr, _)| addr == base_addr)
                .expect("unmap_view called on unknown/already-unmapped view");
            self.active_views.remove(pos);
        }
    }

    let stages = [
        (Win32Stage::S0ReservePlaceholder, 1000),
        (Win32Stage::S1SplitPlaceholder, 1001),
        (Win32Stage::S2CreateMapping, 1002),
        (Win32Stage::S3MapFirstView, 1003),
        (Win32Stage::S4MapSecondView, 1004),
    ];

    for (stage, expected_code) in stages {
        let mut ops = MockWin32VmOps::new(Some(stage));
        let err = create_win32_mirrored_with(&mut ops, 65_536)
            .expect_err("injected fault must fail allocation");
        assert_eq!(
            err.raw_os_error(),
            Some(expected_code),
            "original OS error code must survive rollback at stage {stage:?}"
        );
        ops.assert_no_leaks();
    }

    // S5 success path + Drop unmap simulation.
    let mut ops = MockWin32VmOps::new(None);
    let base = create_win32_mirrored_with(&mut ops, 65_536).expect("S5 success path");
    assert!(ops.active_placeholders.is_empty());
    assert!(
        ops.active_sections.is_empty(),
        "section handle must be closed immediately in S5"
    );
    assert_eq!(ops.active_views.len(), 2);
    ops.unmap_view(base);
    ops.unmap_view(base.wrapping_add(65_536));
    ops.assert_no_leaks();
}

#[cfg(all(not(miri), windows))]
#[test]
#[serial]
fn test_win32_native_fault_injection_s0_to_s4() -> Result<(), Box<dyn std::error::Error>> {
    use windows::{Win32Stage, set_win32_fault_stage};

    struct StageResetGuard;
    impl Drop for StageResetGuard {
        fn drop(&mut self) {
            set_win32_fault_stage(None);
        }
    }

    let _reset = StageResetGuard;
    let stages = [
        (Win32Stage::S0ReservePlaceholder, 8),
        (Win32Stage::S1SplitPlaceholder, 87),
        (Win32Stage::S2CreateMapping, 1455),
        (Win32Stage::S3MapFirstView, 487),
        (Win32Stage::S4MapSecondView, 14),
    ];

    for (stage, expected_code) in stages {
        set_win32_fault_stage(Some(stage));
        for _ in 0..32 {
            let err = MirroredBuffer::<f32>::new(1024)
                .expect_err("native Win32 stage injection must fail");
            assert_eq!(
                err.raw_os_error(),
                Some(expected_code),
                "stage {stage:?} must propagate original OS error"
            );
        }
    }

    set_win32_fault_stage(None);
    let mut buf = MirroredBuffer::<f32>::new(1024)?;
    let n = buf.size();
    buf[0] = 7.5;
    assert_eq!(buf[n].to_bits(), 7.5_f32.to_bits());
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
#[serial]
#[cfg_attr(
    miri,
    ignore = "MirroredBuffer allocation requires memfd_create/mmap (FFI), which Miri cannot interpret; footprint invariance is validated by the native test suite"
)]
fn test_mirror_buf_64kib_footprint_and_model_invariance() -> Result<(), Box<dyn std::error::Error>>
{
    use crate::loader::dispatcher::build_model;
    use crate::loader::nam_json::parse_nam_json;
    use crate::models::NamModel;
    use std::path::Path;

    const WIN32_GRANULARITY: usize = 65_536;
    const NUM_FRAMES: usize = 24_576; // 384 blocks x 64 frames; wraps 64 KiB and 192 KiB rings.
    const BLOCK_FRAMES: usize = 64;

    let fixtures = [
        (
            "WaveNet A1 Standard",
            "tests/fixtures/models/wavenet_a1_standard.nam",
        ),
        (
            "WaveNet A1 Lite CH=12",
            "tests/fixtures/models/BossWN-lite.nam",
        ),
        (
            "WaveNet A2 Lite",
            "tests/fixtures/models/wavenet_a2_lite.nam",
        ),
        (
            "WaveNet A2 Dyn CH=3",
            "tests/fixtures/models/a2_dynamic_blended_ch3.nam",
        ),
        ("Linear Direct", "tests/fixtures/models/linear_test.nam"),
        (
            "Linear Multichannel 1x2",
            "tests/fixtures/models/linear_1x2.nam",
        ),
    ];

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    for (label, rel_path) in fixtures {
        let json = std::fs::read_to_string(manifest_dir.join(rel_path))?;
        let data = parse_nam_json(&json)?;

        let (mut model_4k, rings_4k) = {
            let rec = AllocRecordGuard::start();
            let model = build_model(&data)?;
            let rings = rec.take_records();
            (model, rings)
        };

        let (mut model_64k, rings_64k) = {
            let _gran = GranularityOverrideGuard::new(WIN32_GRANULARITY);
            let rec = AllocRecordGuard::start();
            let model = build_model(&data)?;
            let rings = rec.take_records();
            (model, rings)
        };

        assert!(
            !rings_4k.is_empty(),
            "{label} must allocate MirroredBuffer rings"
        );
        assert_eq!(
            rings_4k.len(),
            rings_64k.len(),
            "{label} ring count mismatch"
        );

        if label == "WaveNet A1 Lite CH=12" {
            // CH=12 (48B stride) and CH=6 (24B stride) both have lcm(65536, stride) = 196_608 (192 KiB).
            for &(_req, size_bytes) in &rings_64k {
                assert_eq!(
                    size_bytes % (192 * 1024),
                    0,
                    "BossWN-lite ring {size_bytes} must be multiple of 192 KiB under 64 KiB granularity"
                );
            }
        }

        let phys_4k: usize = rings_4k.iter().map(|&(_, s)| s).sum();
        let virt_4k: usize = phys_4k * 2;
        let phys_64k: usize = rings_64k.iter().map(|&(_, s)| s).sum();
        let virt_64k: usize = phys_64k * 2;

        eprintln!(
            "[mirror_buf 64KiB footprint] {label}: rings={}, 4KiB=(phys={}B, virt={}B), 64KiB=(phys={}B, virt={}B), delta_phys=+{}B ({:.2}x), per_ring_4k={:?}, per_ring_64k={:?}",
            rings_4k.len(),
            phys_4k,
            virt_4k,
            phys_64k,
            virt_64k,
            phys_64k.saturating_sub(phys_4k),
            phys_64k as f64 / phys_4k.max(1) as f64,
            rings_4k,
            rings_64k,
        );

        let in_ch = model_4k.in_channels();
        let out_ch = model_4k.channels();
        assert_eq!(in_ch, model_64k.in_channels());
        assert_eq!(out_ch, model_64k.channels());

        model_4k.prewarm(2048);
        model_64k.prewarm(2048);

        let mut input = vec![0.0_f32; NUM_FRAMES * in_ch];
        for f in 0..NUM_FRAMES {
            let t = (f as f32) / 48_000.0;
            let sample = 0.45 * (std::f32::consts::TAU * 110.0 * t).sin()
                + 0.25 * (std::f32::consts::TAU * 440.0 * t).sin()
                + if f % 512 == 0 { 0.3 } else { 0.0 };
            for c in 0..in_ch {
                input[f * in_ch + c] = sample * (1.0 - 0.1 * (c as f32));
            }
        }

        let mut out_4k = vec![0.0_f32; NUM_FRAMES * out_ch];
        let mut out_64k = vec![0.0_f32; NUM_FRAMES * out_ch];

        for frame_start in (0..NUM_FRAMES).step_by(BLOCK_FRAMES) {
            let in_slice = &input[frame_start * in_ch..(frame_start + BLOCK_FRAMES) * in_ch];
            let out_slice_4k =
                &mut out_4k[frame_start * out_ch..(frame_start + BLOCK_FRAMES) * out_ch];
            let out_slice_64k =
                &mut out_64k[frame_start * out_ch..(frame_start + BLOCK_FRAMES) * out_ch];
            model_4k.process(in_slice, out_slice_4k);
            model_64k.process(in_slice, out_slice_64k);
        }

        for (idx, (&a, &b)) in out_4k.iter().zip(out_64k.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: 0-ULP mismatch at sample {idx}: 4KiB={a} ({:#010x}) vs 64KiB={b} ({:#010x})",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    Ok(())
}
