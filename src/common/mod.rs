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

pub use diagnostics::*;
pub use panic_hook::*;
pub use params::*;
pub use spsc::*;
