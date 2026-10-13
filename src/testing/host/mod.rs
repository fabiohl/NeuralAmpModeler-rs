// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Host/real-time environment probes shared by downstream test harnesses.
//!
//! Single source of truth for the neutral, generic host introspection that
//! standalone clients and plugin test suites used to duplicate:
//! `/proc` process telemetry ([`proc`]), real-time environment calibration
//! ([`rt_env`]) and per-thread PMU counters ([`pmu`]).
//!
//! Every probe is **best-effort and panic-free**: a missing `/proc`, a sysfs
//! without the requested node, or a `perf_event_open` denied by the kernel's
//! `perf_event_paranoid` level yields a typed `None`/fallback value (never a
//! panic), so harnesses can degrade honestly. Off-RT only — never call these
//! from the audio thread.

pub mod pmu;
pub mod proc;
pub mod rt_env;

#[cfg(test)]
#[path = "host_test.rs"]
mod host_test;
