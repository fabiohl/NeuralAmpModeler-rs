// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//
// Union of the engine and downstream host thread-configuration
// test suites: scheduler-policy honesty, affinity mask construction, typed
// pinning errors and the out-of-range affinity sentinel.

use super::*;
use crate::common::spsc::RtStatusFlags;
use std::ffi::CString;
use std::sync::atomic::Ordering;

struct MockConfigurator {
    policy: i32,
    priority: i32,
    getsched_err: Option<i32>,
    setsched_ret: i32,
    affinity_ret: i32,
}

impl ThreadConfigurator for MockConfigurator {
    fn set_daz_ftz(&self) {}
    fn current_thread_id(&self) -> libc::pthread_t {
        0x1234 as libc::pthread_t
    }
    fn set_thread_affinity(&self, _thread_id: libc::pthread_t, _cpuset: &libc::cpu_set_t) -> i32 {
        self.affinity_ret
    }
    fn get_current_sched_param(&self) -> Result<(i32, libc::sched_param), i32> {
        match self.getsched_err {
            Some(e) => Err(e),
            None => Ok((
                self.policy,
                libc::sched_param {
                    sched_priority: self.priority,
                },
            )),
        }
    }
    fn set_current_sched_param(&self, _policy: i32, _param: &libc::sched_param) -> i32 {
        self.setsched_ret
    }
    fn get_current_cpu(&self) -> i32 {
        3
    }
}

impl MockConfigurator {
    fn other() -> Self {
        Self {
            policy: libc::SCHED_OTHER,
            priority: 0,
            getsched_err: None,
            setsched_ret: 0,
            affinity_ret: 0,
        }
    }
}

#[test]
fn test_build_mask_rejects_out_of_range() {
    assert!(build_cpu_affinity_mask(libc::CPU_SETSIZE as usize).is_none());
    assert!(build_cpu_affinity_mask(usize::MAX).is_none());
    let _ = CString::new("rt").unwrap();
    let Some(mask) = build_cpu_affinity_mask(0) else {
        panic!("CPU 0 must be representable");
    };
    // SAFETY: mask is fully initialized; CPU_ISSET only reads it.
    assert!(unsafe { libc::CPU_ISSET(0, &mask) });
}

#[test]
fn test_build_affinity_mask_cpu_zero() {
    let Some(cpuset) = build_cpu_affinity_mask(0) else {
        panic!("CPU 0 must be representable in the affinity mask");
    };
    // SAFETY: mask is fully initialized; CPU_ISSET only reads it.
    assert!(unsafe { libc::CPU_ISSET(0, &cpuset) });
    // SAFETY: mask is fully initialized; CPU_COUNT only reads it.
    assert_eq!(unsafe { libc::CPU_COUNT(&cpuset) }, 1);
}

#[test]
fn test_build_affinity_mask_max_index() {
    let max_cpu = libc::CPU_SETSIZE as usize - 1;
    let Some(cpuset) = build_cpu_affinity_mask(max_cpu) else {
        panic!("CPU_SETSIZE - 1 must be representable in the affinity mask");
    };
    // SAFETY: mask is fully initialized; CPU_ISSET only reads it.
    assert!(unsafe { libc::CPU_ISSET(max_cpu, &cpuset) });
    // SAFETY: mask is fully initialized; CPU_COUNT only reads it.
    assert_eq!(unsafe { libc::CPU_COUNT(&cpuset) }, 1);
}

#[test]
fn test_pin_current_thread_bounds_check() {
    assert!(pin_current_thread(None).is_ok());
    let max_cpu = libc::CPU_SETSIZE as usize;
    assert_eq!(
        pin_current_thread(Some(max_cpu)),
        Err(PinThreadError::CpuOutOfBounds {
            cpu: max_cpu,
            max: max_cpu,
        })
    );
    assert_eq!(
        pin_current_thread(Some(99999)),
        Err(PinThreadError::CpuOutOfBounds {
            cpu: 99999,
            max: max_cpu,
        })
    );
    assert_eq!(
        pin_current_thread(Some(usize::MAX)),
        Err(PinThreadError::CpuOutOfBounds {
            cpu: usize::MAX,
            max: max_cpu,
        })
    );
}

#[test]
fn test_pin_thread_error_display() {
    let err_oob = PinThreadError::CpuOutOfBounds {
        cpu: 99999,
        max: 1024,
    };
    assert_eq!(
        err_oob.to_string(),
        "CPU index 99999 is outside supported range [0, 1024)"
    );
    let err_sys = PinThreadError::SyscallFailed { cpu: 2, errno: 22 };
    assert_eq!(
        err_sys.to_string(),
        "could not pin thread to CPU 2 (errno 22)"
    );
}

#[test]
fn test_build_housekeeping_mask_empty_is_none() {
    assert!(build_housekeeping_affinity_mask(&[]).is_none());
    let Some(mask) = build_housekeeping_affinity_mask(&[0, 2]) else {
        panic!("non-empty housekeeping set must build a mask");
    };
    // SAFETY: mask is fully initialized; CPU_ISSET only reads it.
    assert!(unsafe { libc::CPU_ISSET(0, &mask) });
    // SAFETY: mask is fully initialized; CPU_ISSET only reads it.
    assert!(unsafe { libc::CPU_ISSET(2, &mask) });
    // SAFETY: mask is fully initialized; CPU_COUNT only reads it.
    assert_eq!(unsafe { libc::CPU_COUNT(&mask) }, 2);
    assert!(build_housekeeping_affinity_mask(&[libc::CPU_SETSIZE as usize]).is_none());
}

#[test]
fn test_affinity_out_of_range_records_sentinel() {
    let flags = RtStatusFlags::new();
    let cfg = MockConfigurator {
        policy: libc::SCHED_OTHER,
        priority: 0,
        getsched_err: None,
        setsched_ret: 0,
        affinity_ret: 0,
    };
    let res = set_cpu_affinity_with(libc::CPU_SETSIZE as usize, &flags, &cfg);
    assert!(res.is_err());
    assert_eq!(flags.rt_affinity_err.load(Ordering::Relaxed), -1);
}

#[test]
fn test_promote_keeps_fifo_without_elevation() {
    let flags = RtStatusFlags::new();
    let cfg = MockConfigurator {
        policy: libc::SCHED_FIFO,
        priority: 80,
        getsched_err: None,
        setsched_ret: 99,
        affinity_ret: 0,
    };
    let res = promote_sched_fifo_with(88, &flags, &cfg);
    assert!(res.is_ok());
    assert!(flags.check_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO));
    assert_eq!(flags.rt_policy.load(Ordering::Relaxed), libc::SCHED_FIFO);
    assert_eq!(flags.confirmed_priority.load(Ordering::Relaxed), 80);
}

#[test]
fn test_promote_keeps_rr_without_elevation() {
    // SCHED_RR is legitimate (some audio hosts / realtime kits): keep it, report honestly, and
    // never force an elevation to FIFO (setsched_ret=99 would be observable if
    // it were attempted, since the recorded policy would become FIFO).
    let flags = RtStatusFlags::new();
    let cfg = MockConfigurator {
        policy: libc::SCHED_RR,
        priority: 50,
        getsched_err: None,
        setsched_ret: 99,
        affinity_ret: 0,
    };
    let res = promote_sched_fifo_with(88, &flags, &cfg);
    assert!(res.is_ok());
    assert!(!flags.check_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO));
    assert_eq!(flags.rt_policy.load(Ordering::Relaxed), libc::SCHED_RR);
    assert_eq!(flags.rt_priority.load(Ordering::Relaxed), 50);
    assert_eq!(flags.confirmed_priority.load(Ordering::Relaxed), 50);
    assert_eq!(flags.rt_sched_err.load(Ordering::Relaxed), 0);
}

#[test]
fn test_promote_other_elevates_and_records_failure_gracefully() {
    let flags = RtStatusFlags::new();
    let ok_cfg = MockConfigurator::other();
    assert!(promote_sched_fifo_with(88, &flags, &ok_cfg).is_ok());
    assert!(flags.check_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO));

    let flags = RtStatusFlags::new();
    let denied_cfg = MockConfigurator {
        policy: libc::SCHED_OTHER,
        priority: 0,
        getsched_err: None,
        setsched_ret: libc::EPERM,
        affinity_ret: 0,
    };
    let res = promote_sched_fifo_with(88, &flags, &denied_cfg);
    assert!(res.is_err());
    assert_eq!(res.unwrap_err(), libc::EPERM);
    assert_eq!(flags.rt_sched_err.load(Ordering::Relaxed), libc::EPERM);
    assert!(!flags.check_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO));
}

#[test]
fn test_promote_getsched_failure_never_returns_err_zero() {
    let flags = RtStatusFlags::new();
    let cfg = MockConfigurator {
        policy: libc::SCHED_OTHER,
        priority: 0,
        getsched_err: Some(libc::EINVAL),
        setsched_ret: 0,
        affinity_ret: 0,
    };
    let res = promote_sched_fifo_with(88, &flags, &cfg);
    assert!(res.is_err());
    assert_ne!(res.unwrap_err(), 0);
    assert_eq!(res.unwrap_err(), libc::EINVAL);
    assert_eq!(flags.rt_getsched_err.load(Ordering::Relaxed), libc::EINVAL);
}
