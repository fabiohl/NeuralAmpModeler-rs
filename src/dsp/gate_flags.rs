// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Canonical mapping from [`GateState`](crate::dsp::gate::GateState) to [`RtStatusFlags`](crate::common::spsc::RtStatusFlags) bits
//! (`RT_STATUS_IS_SILENT`, `RT_STATUS_IS_FADING`).
//!
//! This module is the single source of truth; processing pipelines and host
//! adapters use [`report_gate_flags`](crate::dsp::gate_flags::report_gate_flags) instead of duplicating the match arms.

use crate::common::spsc::RtStatusFlags;
use crate::dsp::gate::GateState;

/// Reports the current gate state to the real-time status atomics.
///
/// - `Closed` → `IS_SILENT` set, `IS_FADING` cleared.
/// - `FadingIn` / `FadingOut` → `IS_SILENT` cleared, `IS_FADING` set.
/// - `Open` → both flags cleared.
///
/// Uses `Relaxed` ordering; safe for the real-time hot-path.
#[inline(always)]
pub fn report_gate_flags(rt_status: &RtStatusFlags, gate_state: GateState) {
    match gate_state {
        GateState::Closed => {
            rt_status.set_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_IS_FADING);
        }
        GateState::FadingIn | GateState::FadingOut => {
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
            rt_status.set_flag(crate::common::spsc::RT_STATUS_IS_FADING);
        }
        GateState::Open => {
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_IS_FADING);
        }
    }
}
