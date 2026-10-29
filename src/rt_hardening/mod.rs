// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Opt-in real-time host hardening for audio hosts.
//!
//! Linux: Transparent Huge Page control, `mlockall`, `SCHED_FIFO` promotion,
//! DAZ/FTZ, IRQ-aware CPU affinity and `/dev/cpu_dma_latency` (PM QoS).
//! macOS: interactive QoS, `THREAD_TIME_CONSTRAINT_POLICY`, `mlock_region`,
//! DAZ/FTZ and optional Audio Workgroups. XNU has no CPU affinity, so none is
//! exposed.
//!
//! These functions must be called exclusively outside the RT thread, during
//! audio host initialization. Errors are returned as `Result` — never as panic.
//!
//! The module is gated behind the `rt-hardening` Cargo feature and the
//! supported platforms, and is never enabled by default. Hosts that adopt it
//! gain immediate observability through the existing
//! [`RtStatusFlags`](crate::common::spsc::RtStatusFlags) telemetry
//! (`rt_affinity_err`, `rt_sched_err`, `rt_getsched_err`, `rt_target_cpu`,
//! `rt_cpu`, `rt_tid`, `rt_priority`, `confirmed_priority`, `rt_policy`), which
//! the host's main loop already polls.

#![cfg(all(
    feature = "rt-hardening",
    any(target_os = "linux", target_os = "macos")
))]

/// IRQ-aware CPU topology inspection and optimal core selection (Linux).
#[cfg(target_os = "linux")]
pub mod affinity;
/// PM QoS (`/dev/cpu_dma_latency`) latency guard (Linux).
#[cfg(target_os = "linux")]
pub mod pm_qos;
/// Process-wide setup and RT thread promotion (Linux).
#[cfg(target_os = "linux")]
pub mod thread;

/// macOS real-time setup (QoS, time constraint, memory locking, workgroups).
#[cfg(target_os = "macos")]
pub mod macos;

/// Pure Mach timebase conversion (tested on every host, used by macOS).
#[cfg(any(target_os = "macos", test))]
mod timebase;

#[cfg(target_os = "linux")]
pub use affinity::{
    CpuCoreTopology, CpuSelectionReason, CpuSelectionReceipt, SysfsTopologySource,
    SystemSysfsSource, get_allowed_cpus, parse_cpu_list, parse_interrupts_per_cpu,
    parse_proc_interrupts, select_cpu_with_source, select_optimal_cpu,
    select_optimal_cpu_with_receipt,
};
#[cfg(target_os = "macos")]
pub use macos::{
    AudioWorkgroupJoin, OsWorkgroup, ThreadTiming, compute_thread_timing, join_audio_workgroup,
    mlock_region, promote_time_constraint, set_ftz_daz, set_user_interactive_qos,
};
#[cfg(target_os = "linux")]
pub use pm_qos::{PmQosGuard, request_cpu_dma_latency};
#[cfg(target_os = "linux")]
pub use thread::{
    PinThreadError, SystemThreadConfigurator, ThreadConfigurator, apply_housekeeping_affinity,
    build_cpu_affinity_mask, build_housekeeping_affinity_mask, disable_thp, mlockall_current,
    pin_current_thread, promote_sched_fifo, promote_sched_fifo_with, set_cpu_affinity,
    set_cpu_affinity_with, set_ftz_daz,
};
