// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Process-wide and per-thread real-time setup (off-RT only).
//!
//! These functions must be called exclusively outside the RT thread, during
//! audio host initialization. Errors are returned as `Result` — never as panic.

#![cfg(all(feature = "rt-hardening", target_os = "linux"))]

use std::io;

/// `PR_THP_DISABLE_EXCEPT_ADVISED` (value 2) — Linux 7.0+; absent from older
/// libc, so defined locally for forward compatibility.
const PR_THP_DISABLE_EXCEPT_ADVISED: libc::c_ulong = 2;

/// Disables Transparent Huge Pages for the calling process (off-RT only).
///
/// Tries the modern `PR_THP_DISABLE_EXCEPT_ADVISED` mode (Linux 7.0+, keeps
/// pages explicitly marked `MADV_HUGEPAGE` eligible) first, falling back to
/// classic `PR_SET_THP_DISABLE` on older kernels. Logs the outcome via
/// `log::*` (off-RT: never call on the audio thread).
#[cold]
pub fn disable_thp() -> io::Result<()> {
    // SAFETY: `prctl(PR_SET_THP_DISABLE, ...)` takes only integer arguments;
    // no pointers are dereferenced. Return value / errno contract per man 2 prctl.
    let ret = unsafe {
        libc::prctl(
            libc::PR_SET_THP_DISABLE,
            1,
            PR_THP_DISABLE_EXCEPT_ADVISED,
            0,
            0,
        )
    };
    if ret == 0 {
        log::info!("Transparent Huge Pages disabled (except MADV_HUGEPAGE regions).");
        return Ok(());
    }
    // SAFETY: `__errno_location` returns a valid thread-local errno pointer.
    let errno = unsafe { *libc::__errno_location() };
    if errno == libc::EINVAL {
        let err = io::Error::last_os_error();
        log::info!(
            "Kernel lacks PR_THP_DISABLE_EXCEPT_ADVISED (errno={errno}: {err}) — \
             falling back to classic PR_SET_THP_DISABLE."
        );
        // SAFETY: same integer-only `prctl` contract as above.
        let classic = unsafe { libc::prctl(libc::PR_SET_THP_DISABLE, 1, 0, 0, 0) };
        if classic == 0 {
            log::info!(
                "Transparent Huge Pages globally disabled (classic fallback). \
                 Only MADV_HUGEPAGE regions may use THP."
            );
            return Ok(());
        }
        let fallback_err = io::Error::last_os_error();
        log::warn!(
            "Classic PR_SET_THP_DISABLE also failed (errno={}: {fallback_err}). \
             THP may remain active — background compaction latencies possible.",
            fallback_err.raw_os_error().unwrap_or(-1),
        );
        return Err(fallback_err);
    }
    let err = io::Error::from_raw_os_error(errno);
    log::warn!(
        "prctl(PR_SET_THP_DISABLE) failed with unexpected errno={errno}: {err}. \
         THP state unknown — background compaction latencies possible."
    );
    Err(err)
}

/// Locks current and future process memory in RAM (off-RT only).
///
/// Prevents page faults on the audio thread. Fails gracefully when the
/// `memlock` rlimit is too low — the caller logs and continues degraded.
/// Logs the outcome via `log::*` (off-RT: never call on the audio thread).
#[cold]
pub fn mlockall_current() -> io::Result<()> {
    // SAFETY: `mlockall(MCL_CURRENT | MCL_FUTURE)` takes only flags; no
    // pointers. Return / errno contract per man 2 mlockall.
    let ret = unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) };
    if ret == 0 {
        log::info!("Memory locked in physical RAM (mlockall).");
        Ok(())
    } else {
        let err = io::Error::last_os_error();
        log::warn!(
            "mlockall() failed ({err}). Audio may experience dropouts if the \
             system swaps. Hint: verify the 'memlock' limit in ulimits."
        );
        Err(err)
    }
}

/// Enables DAZ (Denormals-Are-Zero) and FTZ (Flush-To-Zero) on the calling
/// thread. Cheap enough to call during RT-thread setup, before the hot loop.
#[inline]
pub fn set_ftz_daz() {
    // SAFETY: `set_daz_ftz` only manipulates the calling thread's MXCSR
    // (SSE2 is guaranteed by the x86-64-v3 baseline).
    unsafe {
        crate::math::common::set_daz_ftz();
    }
}

/// Injectable abstraction over the thread syscalls, for deterministic tests.
pub trait ThreadConfigurator {
    /// Enables Denormals-Are-Zero and Flush-To-Zero on the calling thread.
    fn set_daz_ftz(&self);

    /// Returns the current thread ID (`libc::pthread_t`).
    fn current_thread_id(&self) -> libc::pthread_t;

    /// Sets CPU affinity for `thread_id`.
    fn set_thread_affinity(&self, thread_id: libc::pthread_t, cpuset: &libc::cpu_set_t) -> i32;

    /// Reads the scheduling policy and parameters.
    fn get_sched_param(&self, thread_id: libc::pthread_t) -> Result<(i32, libc::sched_param), i32>;

    /// Requests a scheduling policy and parameters (`sched_setscheduler`).
    fn set_sched_param(
        &self,
        thread_id: libc::pthread_t,
        policy: i32,
        param: &libc::sched_param,
    ) -> i32;

    /// Returns the CPU the calling thread currently runs on.
    fn get_current_cpu(&self) -> i32;
}

/// Default system-backed [`ThreadConfigurator`] using libc.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemThreadConfigurator;

impl ThreadConfigurator for SystemThreadConfigurator {
    fn set_daz_ftz(&self) {
        set_ftz_daz();
    }

    fn current_thread_id(&self) -> libc::pthread_t {
        // SAFETY: `pthread_self` always succeeds and returns a valid handle.
        unsafe { libc::pthread_self() }
    }

    fn set_thread_affinity(&self, thread_id: libc::pthread_t, cpuset: &libc::cpu_set_t) -> i32 {
        // SAFETY: `cpuset` is a valid initialized mask; size matches
        // `cpu_set_t`. Return contract per man 3 pthread_setaffinity_np.
        unsafe {
            libc::pthread_setaffinity_np(thread_id, std::mem::size_of::<libc::cpu_set_t>(), cpuset)
        }
    }

    fn get_sched_param(&self, thread_id: libc::pthread_t) -> Result<(i32, libc::sched_param), i32> {
        let mut policy = 0i32;
        let mut param = libc::sched_param { sched_priority: 0 };
        // SAFETY: `&mut policy`/`&mut param` are valid out-pointers.
        let ret = unsafe { libc::pthread_getschedparam(thread_id, &mut policy, &mut param) };
        if ret == 0 {
            Ok((policy, param))
        } else {
            Err(ret)
        }
    }

    fn set_sched_param(
        &self,
        _thread_id: libc::pthread_t,
        policy: i32,
        param: &libc::sched_param,
    ) -> i32 {
        // SAFETY: `param` is a valid pointer for the syscall duration.
        let ret = unsafe { libc::sched_setscheduler(0, policy, param) };
        if ret == -1 {
            // SAFETY: `__errno_location` returns a valid thread-local pointer.
            unsafe { *libc::__errno_location() }
        } else {
            0
        }
    }

    fn get_current_cpu(&self) -> i32 {
        // SAFETY: `sched_getcpu` takes no pointers; returns the CPU id or -1.
        unsafe { libc::sched_getcpu() }
    }
}

/// Builds the `cpu_set_t` mask pinning a thread to `target_cpu`.
///
/// Returns `None` when `target_cpu` is outside `[0, CPU_SETSIZE)`.
pub(crate) fn build_cpu_affinity_mask(target_cpu: usize) -> Option<libc::cpu_set_t> {
    if target_cpu >= libc::CPU_SETSIZE as usize {
        return None;
    }
    // SAFETY: on the supported Linux targets `cpu_set_t` is a C bitmask whose
    // all-zero pattern is the empty set — a valid value with no reference
    // formed over uninitialized storage.
    let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: `CPU_ZERO`/`CPU_SET` mutate the initialized mask in place; the
    // bounds check above keeps libc's word index inside `cpu_set_t` storage.
    unsafe {
        libc::CPU_ZERO(&mut cpuset);
        libc::CPU_SET(target_cpu, &mut cpuset);
    }
    Some(cpuset)
}

/// Pins the calling thread to `cpus` (off-RT setup, before the hot loop).
///
/// Records the outcome in `rt_status` (`rt_affinity_err` / `rt_target_cpu`)
/// with `Relaxed` stores — RT-safe, no logging or allocation on this path.
/// Out-of-range CPUs are rejected before any syscall (`rt_affinity_err = -1`).
/// Returns `Ok(())` on success, `Err(errno)` otherwise.
pub fn set_cpu_affinity(
    cpus: &[usize],
    rt_status: &crate::common::spsc::RtStatusFlags,
) -> Result<(), i32> {
    use std::sync::atomic::Ordering;

    let Some(&target) = cpus.first() else {
        rt_status.rt_affinity_err.store(-1, Ordering::Relaxed);
        rt_status.rt_target_cpu.store(-1, Ordering::Relaxed);
        return Err(-1);
    };
    set_cpu_affinity_one(target, rt_status, &SystemThreadConfigurator)
}

/// Pins `thread_id` to `target_cpu` via `cfg`, recording the outcome in
/// `rt_status` (RT-safe: no logging or allocation on this path).
pub(crate) fn set_cpu_affinity_one<C: ThreadConfigurator>(
    target_cpu: usize,
    rt_status: &crate::common::spsc::RtStatusFlags,
    cfg: &C,
) -> Result<(), i32> {
    use std::sync::atomic::Ordering;

    let thread_id = cfg.current_thread_id();
    let Some(cpuset) = build_cpu_affinity_mask(target_cpu) else {
        rt_status.rt_affinity_err.store(-1, Ordering::Relaxed);
        rt_status
            .rt_target_cpu
            .store(target_cpu as i32, Ordering::Relaxed);
        return Err(-1);
    };
    let ret = cfg.set_thread_affinity(thread_id, &cpuset);
    if ret != 0 {
        rt_status.rt_affinity_err.store(ret, Ordering::Relaxed);
        rt_status
            .rt_target_cpu
            .store(target_cpu as i32, Ordering::Relaxed);
        return Err(ret);
    }
    Ok(())
}

/// Promotes the calling thread to `SCHED_FIFO` at `priority` (off-RT setup).
///
/// Honest-policy semantics: when the thread already runs under `SCHED_FIFO`
/// or `SCHED_RR` the existing policy is kept (records confirmed priority and
/// sets `RT_STATUS_RT_IS_FIFO` only for FIFO); only `SCHED_OTHER` (or other
/// non-RT) triggers an elevation attempt to `SCHED_FIFO`. Failures record
/// errno in `rt_sched_err` / `rt_getsched_err` without panicking.
/// Publishes the result via `rt_status` atomics for main-loop telemetry.
#[cold]
pub fn promote_sched_fifo(
    priority: i32,
    rt_status: &crate::common::spsc::RtStatusFlags,
) -> Result<(), i32> {
    promote_sched_fifo_with(priority, rt_status, &SystemThreadConfigurator)
}

/// [`promote_sched_fifo`] with an injectable [`ThreadConfigurator`].
#[cold]
pub fn promote_sched_fifo_with<C: ThreadConfigurator>(
    priority: i32,
    rt_status: &crate::common::spsc::RtStatusFlags,
    cfg: &C,
) -> Result<(), i32> {
    use std::sync::atomic::Ordering;

    cfg.set_daz_ftz();

    let thread_id = cfg.current_thread_id();
    let actual_cpu = cfg.get_current_cpu();
    rt_status.rt_cpu.store(actual_cpu, Ordering::Relaxed);
    // pthread_t is not the kernel TID, but it is the stable thread identity
    // the host logs alongside `rt_cpu`; keep the field populated.
    rt_status.rt_tid.store(thread_id as i64, Ordering::Relaxed);

    let (actual_policy, actual_param) = match cfg.get_sched_param(thread_id) {
        Ok((p, param)) => {
            let base_policy = p & !0x40000000i32;
            if base_policy == libc::SCHED_FIFO || base_policy == libc::SCHED_RR {
                (base_policy, param)
            } else {
                let target_param = libc::sched_param {
                    sched_priority: priority,
                };
                let ret_set = cfg.set_sched_param(thread_id, libc::SCHED_FIFO, &target_param);
                if ret_set == 0 {
                    (libc::SCHED_FIFO, target_param)
                } else {
                    rt_status.rt_sched_err.store(ret_set, Ordering::Relaxed);
                    (base_policy, param)
                }
            }
        }
        Err(ret_getsched) => {
            rt_status
                .rt_getsched_err
                .store(ret_getsched, Ordering::Relaxed);
            (-1, libc::sched_param { sched_priority: -1 })
        }
    };

    if actual_policy == libc::SCHED_FIFO {
        rt_status.set_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);
    } else {
        rt_status.clear_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);
    }

    rt_status.rt_priority.store(
        if actual_policy == -1 {
            0
        } else {
            actual_param.sched_priority
        },
        Ordering::Relaxed,
    );
    rt_status
        .confirmed_priority
        .store(actual_param.sched_priority, Ordering::Relaxed);
    rt_status.rt_policy.store(actual_policy, Ordering::Relaxed);
    if actual_policy == libc::SCHED_FIFO || actual_policy == libc::SCHED_RR {
        Ok(())
    } else {
        Err(rt_status
            .rt_sched_err
            .load(std::sync::atomic::Ordering::Relaxed))
    }
}

#[cfg(test)]
#[path = "thread_test.rs"]
mod thread_test;
