// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Opt-in real-time host hardening for audio hosts.
//!
//! Generic low-latency process setup extracted for any audio host on Linux:
//! Transparent Huge Page control, `mlockall`, `SCHED_FIFO` promotion,
//! DAZ/FTZ, IRQ-aware CPU affinity and `/dev/cpu_dma_latency` (PM QoS).
//!
//! These functions must be called exclusively outside the RT thread, during
//! audio host initialization. Errors are returned as `Result` — never as panic.
//!
//! The module is gated behind the `rt-hardening` Cargo feature and Linux, and
//! is never enabled by default. Hosts that adopt it gain immediate
//! observability through the existing [`RtStatusFlags`](crate::common::spsc::RtStatusFlags)
//! telemetry (`rt_affinity_err`, `rt_sched_err`, `rt_getsched_err`,
//! `rt_target_cpu`, `rt_cpu`, `rt_tid`, `rt_priority`, `confirmed_priority`,
//! `rt_policy`), which the host's main loop already polls.

#![cfg(all(feature = "rt-hardening", target_os = "linux"))]

/// IRQ-aware CPU topology inspection and optimal core selection.
pub mod affinity;
/// PM QoS (`/dev/cpu_dma_latency`) latency guard.
pub mod pm_qos;
/// Process-wide setup (THP, `mlockall`) and RT thread promotion.
pub mod thread;

pub use affinity::{
    CpuCoreTopology, CpuSelectionReason, CpuSelectionReceipt, SysfsTopologySource,
    SystemSysfsSource, get_allowed_cpus, parse_cpu_list, parse_interrupts_per_cpu,
    parse_proc_interrupts, select_cpu_with_source, select_optimal_cpu,
    select_optimal_cpu_with_receipt,
};
pub use pm_qos::{PmQosGuard, request_cpu_dma_latency};
pub use thread::{
    ThreadConfigurator, disable_thp, mlockall_current, promote_sched_fifo, promote_sched_fifo_with,
    set_cpu_affinity, set_ftz_daz,
};
