// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Host-agnostic infrastructure shared by engine integrations.

pub mod alloc_audit;
pub mod diagnostics;
pub mod panic_hook;
pub mod params;
pub mod spsc;
#[cfg(target_arch = "x86_64")]
pub mod tsc;

// These submodules are exposed via their qualified module paths (common::diagnostics, common::spsc, etc.).
// The crate root (lib.rs) selectively re-exports curated items from here — see
// the "API Surface Policy" comment in lib.rs.
