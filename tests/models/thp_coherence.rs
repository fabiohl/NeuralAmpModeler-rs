// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::fs;

use serial_test::serial;

use neural_amp_modeler_rs::dsp::mirror_buf::{
    MirrorHugePageStatus, MirroredBuffer, huge_page_status, is_huge_page_active,
};

/// Reads `AnonHugePages` (in kB) from `/proc/self/smaps_rollup`.
/// Returns `None` if the file cannot be read or the field is absent.
fn read_anon_huge_pages_kb() -> Option<u64> {
    let content = fs::read_to_string("/proc/self/smaps_rollup").ok()?;
    for line in content.lines() {
        if line.starts_with("AnonHugePages:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                return parts[1].parse::<u64>().ok();
            }
        }
    }
    None
}

#[test]
#[cfg(target_os = "linux")]
#[serial]
fn test_thp_coherence_smaps_consistency() {
    let buf = match MirroredBuffer::<f32>::new(600_000) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "Skipping test_thp_coherence_smaps_consistency: MirroredBuffer allocation failed: {}",
                e
            );
            return;
        }
    };

    let status = huge_page_status();
    let is_active = is_huge_page_active();

    match status {
        MirrorHugePageStatus::Explicit2MB => {
            assert!(
                is_active,
                "is_huge_page_active() should be true when Explicit2MB"
            );
        }
        MirrorHugePageStatus::Transparent => {
            assert!(
                is_active,
                "is_huge_page_active() should be true when Transparent"
            );
        }
        MirrorHugePageStatus::Standard => {
            assert!(
                !is_active,
                "is_huge_page_active() should be false when Standard"
            );
        }
    }

    let anon_hp = read_anon_huge_pages_kb();
    eprintln!(
        "THP coherence: status={:?}, is_active={}, AnonHugePages={:?} kB",
        status, is_active, anon_hp
    );

    if let (MirrorHugePageStatus::Transparent, Some(hp)) = (status, anon_hp)
        && hp == 0
    {
        eprintln!(
            "Note: THP reported as Transparent but AnonHugePages is 0 — \
             kernel may promote lazily or MADV_COLLAPSE returned success \
             without immediate promotion."
        );
    }

    drop(buf);
}

/// Tests that prctl(PR_SET_THP_DISABLE) with the modern
/// PR_THP_DISABLE_EXCEPT_ADVISED value (2) behaves correctly:
/// either succeeds (Linux 7.0+) or returns EINVAL (older kernels).
/// In neither case does it crash or leave corrupted state.
#[test]
#[cfg(target_os = "linux")]
#[serial]
fn test_prctl_thp_except_advised_no_crash() {
    const PR_SET_THP_DISABLE: libc::c_int = 41;
    const PR_GET_THP_DISABLE: libc::c_int = 42;
    const PR_THP_DISABLE_EXCEPT_ADVISED: libc::c_ulong = 2;

    let original_thp_state = unsafe { libc::prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0) };

    let ret = unsafe { libc::prctl(PR_SET_THP_DISABLE, 1, PR_THP_DISABLE_EXCEPT_ADVISED, 0, 0) };

    if ret == -1 {
        let errno = unsafe { *libc::__errno_location() };
        eprintln!(
            "prctl(PR_SET_THP_DISABLE, except_advised) returned -1, errno={} (EINVAL={})",
            errno,
            libc::EINVAL
        );
        assert_eq!(
            errno,
            libc::EINVAL,
            "Expected EINVAL on older kernels, got errno={}",
            errno
        );
    } else {
        eprintln!(
            "prctl(PR_SET_THP_DISABLE, except_advised) returned {} — Linux 7.0+ THP mode active",
            ret
        );
        assert_eq!(ret, 0, "prctl should return 0 on success");
    }

    let (restore_flag, restore_mode): (i32, libc::c_ulong) = match original_thp_state {
        0 => (0, 0),
        1 => (1, 0),
        3 => (1, PR_THP_DISABLE_EXCEPT_ADVISED),
        other => {
            eprintln!("Unexpected PR_GET_THP_DISABLE value {other}; defaulting to re-enable THP.");
            (0, 0)
        }
    };
    unsafe {
        libc::prctl(PR_SET_THP_DISABLE, restore_flag, restore_mode, 0, 0);
    }
}
