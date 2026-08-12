// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Host-agnostic infrastructure shared by engine integrations.

/// Compile-time heap-allocation auditing infrastructure for RT safety verification.
pub mod alloc_audit;
/// Diagnostic engine: error codes, snapshots, system info, runtime log formatting.
pub mod diagnostics;
/// Panic hook: writes structured crash reports to disk with system state and log trace.
pub mod panic_hook;
/// Global processing parameters and configuration enums shared across DSP and models.
pub mod params;
/// Lock-free single-producer single-consumer protocol for real-time context sharing.
pub mod spsc;
#[cfg(target_arch = "x86_64")]
/// Time Stamp Counter: x86_64 monotonic nanosecond clock via RDTSC instruction.
pub mod tsc;

// These submodules are exposed via their qualified module paths (common::diagnostics, common::spsc, etc.).
// The crate root (lib.rs) selectively re-exports curated items from here — see
// the "API Surface Policy" comment in lib.rs.
