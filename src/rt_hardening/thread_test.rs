// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

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

#[test]
fn test_build_mask_rejects_out_of_range() {
    assert!(build_cpu_affinity_mask(libc::CPU_SETSIZE as usize).is_none());
    let _ = CString::new("rt").unwrap();
    let Some(mask) = build_cpu_affinity_mask(0) else {
        panic!("CPU 0 must be representable");
    };
    // SAFETY: mask is fully initialized; CPU_ISSET only reads it.
    assert!(unsafe { libc::CPU_ISSET(0, &mask) });
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
    let res = set_cpu_affinity_one(libc::CPU_SETSIZE as usize, &flags, &cfg);
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
fn test_promote_other_elevates_and_records_failure_gracefully() {
    let flags = RtStatusFlags::new();
    let ok_cfg = MockConfigurator {
        policy: libc::SCHED_OTHER,
        priority: 0,
        getsched_err: None,
        setsched_ret: 0,
        affinity_ret: 0,
    };
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
