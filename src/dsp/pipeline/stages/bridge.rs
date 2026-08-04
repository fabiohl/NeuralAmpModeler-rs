// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 4: Write to DspBridge.

use super::super::bridge::DspBridgeWriter;

/// Stage 4: Write to DspBridge.
#[inline(always)]
pub fn write_bridge(
    resamp_out_l: &[f32],
    resamp_out_r: &[f32],
    n_pw: usize,
    bridge: Option<DspBridgeWriter>,
    process_mono: bool,
) {
    if let Some(writer) = bridge {
        writer.write_block(resamp_out_l, resamp_out_r, n_pw, process_mono);
    }
}
