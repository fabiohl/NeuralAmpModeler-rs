// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Opt-in real-time host hardening for macOS audio hosts.
//!
//! These functions are the macOS counterpart of the Linux bring-up surface:
//! interactive QoS, a Mach time-constraint policy, memory locking and optional
//! Audio Workgroups. They must be called outside the hot loop (thread setup or
//! process initialization). Errors are returned as `Result`, never as panic.
//!
//! Affinity, `SCHED_FIFO`, PM-QoS and THP do not exist in the XNU contract and
//! are intentionally absent.

#![cfg(all(feature = "rt-hardening", target_os = "macos"))]

use std::io;
use std::sync::atomic::Ordering;

pub use super::timebase::{ThreadTiming, compute_thread_timing};

/// `QOS_CLASS_USER_INTERACTIVE` from `<pthread/qos.h>`.
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

/// Opaque `os_workgroup_t` supplied by the host.
pub type OsWorkgroup = *mut core::ffi::c_void;

/// Opaque token populated by `os_workgroup_join` and consumed by
/// `os_workgroup_leave` (LP64 layout: `sig` + 36 opaque bytes).
#[derive(Debug)]
#[repr(C)]
struct OsWorkgroupJoinToken {
    sig: u32,
    opaque: [u8; 36],
}

/// Mach absolute-time base.
#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

unsafe extern "C" {
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> libc::kern_return_t;
    fn mach_port_deallocate(
        task: libc::mach_port_t,
        name: libc::mach_port_t,
    ) -> libc::kern_return_t;
    fn pthread_set_qos_class_self_np(class: u32, priority: i32) -> i32;
    fn os_workgroup_join(
        workgroup: OsWorkgroup,
        token: *mut OsWorkgroupJoinToken,
    ) -> core::ffi::c_int;
    fn os_workgroup_leave(workgroup: OsWorkgroup, token: *mut OsWorkgroupJoinToken);
}

/// Enables DAZ/FTZ-equivalent denormal flushing on the calling thread.
///
/// On aarch64 this sets `FPCR.FZ`; x86_64 macOS uses the existing MXCSR
/// implementation. Cheap enough to call during RT-thread setup.
#[inline]
pub fn set_ftz_daz() {
    // SAFETY: `set_daz_ftz` only changes the calling thread's FP control
    // register (FPCR on aarch64, MXCSR on x86_64); no memory is touched.
    unsafe {
        crate::math::common::set_daz_ftz();
    }
}

/// Locks a host-owned model or scratch region in RAM (off-RT only).
///
/// # Safety
///
/// `addr` must point to a live mapping of at least `len` bytes for the
/// duration of the call. The engine never retains or mutates the region.
#[cold]
pub unsafe fn mlock_region(addr: *const u8, len: usize) -> io::Result<()> {
    if addr.is_null() || len == 0 {
        return Err(io::Error::from_raw_os_error(libc::EINVAL));
    }
    // SAFETY: the caller guarantees `addr..addr+len` is a live mapping, as
    // documented in the `# Safety` contract above.
    let ret = unsafe { libc::mlock(addr.cast(), len) };
    if ret == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Applies `QOS_CLASS_USER_INTERACTIVE` to the calling thread (off-RT setup).
#[cold]
pub fn set_user_interactive_qos() -> io::Result<()> {
    // SAFETY: `pthread_set_qos_class_self_np` only changes the calling
    // thread's QoS class; `QOS_CLASS_USER_INTERACTIVE` is a valid class and
    // the relative priority is in the documented range.
    let ret = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
    if ret == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(ret))
    }
}

/// Reads the live Mach timebase (off-RT setup only).
fn mach_timebase() -> io::Result<(u32, u32)> {
    let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
    // SAFETY: `info` is a valid out-pointer to a properly aligned
    // `mach_timebase_info` struct.
    let ret = unsafe { mach_timebase_info(&mut info) };
    if ret != 0 {
        return Err(io::Error::other(format!(
            "mach_timebase_info failed (kern_return={ret})"
        )));
    }
    if info.numer == 0 || info.denom == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "mach_timebase_info returned a zero component",
        ));
    }
    Ok((info.numer, info.denom))
}

/// Applies `THREAD_TIME_CONSTRAINT_POLICY` to the calling thread.
// libc marks its Mach helpers as deprecated in favor of the `mach2` crate; the
// engine deliberately keeps `libc` as its only Mach dependency.
#[expect(
    deprecated,
    reason = "libc forwards to mach2; the engine keeps libc only"
)]
fn apply_time_constraint(timing: ThreadTiming) -> Result<(), libc::kern_return_t> {
    let mut policy = libc::thread_time_constraint_policy {
        period: timing.period_ticks,
        computation: timing.computation_ticks,
        constraint: timing.constraint_ticks,
        preemptible: 0,
    };

    // SAFETY: `mach_thread_self` returns a send right for the calling thread.
    let thread = unsafe { libc::mach_thread_self() };
    // SAFETY: `policy` is a valid, properly aligned
    // `thread_time_constraint_policy`; `count` matches its layout.
    let ret = unsafe {
        libc::thread_policy_set(
            thread,
            libc::THREAD_TIME_CONSTRAINT_POLICY as libc::thread_policy_flavor_t,
            (&raw mut policy).cast::<libc::integer_t>(),
            libc::THREAD_TIME_CONSTRAINT_POLICY_COUNT,
        )
    };
    // SAFETY: `thread` is the send right returned above; `mach_task_self()`
    // is the task owning it. Releasing it on every path avoids a port leak.
    unsafe {
        mach_port_deallocate(libc::mach_task_self(), thread);
    }

    if ret == 0 { Ok(()) } else { Err(ret) }
}

/// Applies DAZ/FTZ, interactive QoS and a Mach time-constraint policy.
///
/// `buffer_frames` and `sample_rate` describe the current host quantum. The
/// returned [`ThreadTiming`] is the policy actually requested. Telemetry is
/// published through `rt_status`: `rt_policy` receives
/// `THREAD_TIME_CONSTRAINT_POLICY` on success, `rt_priority` /
/// `confirmed_priority` receive the QoS class, and `rt_sched_err` receives the
/// first failing code. `RT_STATUS_RT_IS_FIFO` is never set on macOS.
#[cold]
pub fn promote_time_constraint(
    buffer_frames: u32,
    sample_rate: u32,
    rt_status: &crate::common::spsc::RtStatusFlags,
) -> io::Result<ThreadTiming> {
    set_ftz_daz();

    // SAFETY: `pthread_self` has no preconditions and returns a valid handle.
    let thread_id = unsafe { libc::pthread_self() };
    rt_status.rt_tid.store(thread_id as i64, Ordering::Relaxed);
    rt_status.rt_cpu.store(-1, Ordering::Relaxed);

    let (numer, denom) = mach_timebase()?;
    let timing = compute_thread_timing(buffer_frames, sample_rate, numer, denom)?;

    let qos = set_user_interactive_qos();
    let constraint = apply_time_constraint(timing);

    match (qos, constraint) {
        (Ok(()), Ok(())) => {
            rt_status
                .rt_policy
                .store(libc::THREAD_TIME_CONSTRAINT_POLICY, Ordering::Relaxed);
            rt_status
                .rt_priority
                .store(QOS_CLASS_USER_INTERACTIVE as i32, Ordering::Relaxed);
            rt_status
                .confirmed_priority
                .store(QOS_CLASS_USER_INTERACTIVE as i32, Ordering::Relaxed);
            rt_status.rt_sched_err.store(0, Ordering::Relaxed);
            Ok(timing)
        }
        (Err(err), _) => {
            rt_status.rt_sched_err.store(
                err.raw_os_error().unwrap_or(libc::EINVAL),
                Ordering::Relaxed,
            );
            rt_status.rt_policy.store(-1, Ordering::Relaxed);
            Err(err)
        }
        (Ok(()), Err(kr)) => {
            rt_status.rt_sched_err.store(kr, Ordering::Relaxed);
            rt_status.rt_policy.store(-1, Ordering::Relaxed);
            Err(io::Error::other(format!(
                "thread_policy_set(THREAD_TIME_CONSTRAINT_POLICY) failed (kern_return={kr})"
            )))
        }
    }
}

/// Guard for a successful `os_workgroup_join`.
///
/// The token belongs to the joining thread; the guard is intentionally not
/// `Send`/`Sync` because it stores the raw workgroup pointer.
#[derive(Debug)]
pub struct AudioWorkgroupJoin {
    workgroup: OsWorkgroup,
    token: OsWorkgroupJoinToken,
    joined: bool,
}

impl AudioWorkgroupJoin {
    /// Leaves the workgroup explicitly, consuming the guard.
    pub fn leave(mut self) {
        self.leave_inner();
    }

    fn leave_inner(&mut self) {
        if self.joined {
            // SAFETY: the token was populated by a successful
            // `os_workgroup_join` and is passed back to the same workgroup on
            // the same thread, exactly once.
            unsafe {
                os_workgroup_leave(self.workgroup, &raw mut self.token);
            }
            self.joined = false;
        }
    }
}

impl Drop for AudioWorkgroupJoin {
    fn drop(&mut self) {
        self.leave_inner();
    }
}

/// Joins the calling thread to a host-supplied Audio Workgroup.
///
/// The workgroup is obtained by the host (for example from its audio device)
/// and remains owned by the host; the engine never retains or releases it.
/// Intended for thread setup, not for per-callback calls.
///
/// # Safety
///
/// `workgroup` must be a live `os_workgroup_t` for the lifetime of the
/// returned guard, and this function must be called on the thread that will
/// process the audio. The host must not release the workgroup while the guard
/// is alive.
#[cold]
pub unsafe fn join_audio_workgroup(workgroup: OsWorkgroup) -> io::Result<AudioWorkgroupJoin> {
    if workgroup.is_null() {
        return Err(io::Error::from_raw_os_error(libc::EINVAL));
    }

    let mut token = OsWorkgroupJoinToken {
        sig: 0,
        opaque: [0; 36],
    };
    // SAFETY: the caller guarantees a live workgroup and correct thread; the
    // token is a valid, properly aligned out-pointer.
    let ret = unsafe { os_workgroup_join(workgroup, &raw mut token) };
    if ret != 0 {
        return Err(io::Error::from_raw_os_error(ret));
    }

    Ok(AudioWorkgroupJoin {
        workgroup,
        token,
        joined: true,
    })
}
