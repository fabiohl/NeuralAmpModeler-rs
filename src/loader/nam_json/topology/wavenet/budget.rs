// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! State budget validation for WaveNet geometries.

use crate::loader::nam_json::validation::MAX_TOTAL_STATE_FRAMES;

/// Validates that total state buffer frames do not exceed `MAX_TOTAL_STATE_FRAMES`.
///
/// `kernel_sizes[i][j]` is the kernel of layer `j` in layer-array `i`, matching
/// the C++ `kernel_sizes` array contract (`NAM/wavenet/model.cpp:1003-1032`).
pub fn compute_state_budget_per_layer(
    dilations: &[Vec<usize>],
    kernel_sizes: &[Vec<usize>],
    channels: &[usize],
    head_kernel_sizes: &[usize],
    head_dilations: &[usize],
) -> Result<(), String> {
    let total_state_frames: usize = dilations
        .iter()
        .zip(kernel_sizes.iter())
        .zip(channels.iter())
        .zip(head_kernel_sizes.iter())
        .zip(head_dilations.iter())
        .try_fold(0usize, |acc, ((((dils, ks), &ch), &head_k), &head_dil)| {
            let head_rf = head_k.saturating_sub(1).saturating_mul(head_dil);
            let head_state = head_rf.saturating_mul(ch);
            let array_state = dils.iter().zip(ks.iter()).try_fold(0usize, |a, (&d, &k)| {
                a.checked_add(k.saturating_sub(1).saturating_mul(d).saturating_mul(ch))
            })?;
            acc.checked_add(array_state)?.checked_add(head_state)
        })
        .unwrap_or(usize::MAX);
    if total_state_frames > MAX_TOTAL_STATE_FRAMES {
        Err(format!(
            "Aggregate state budget exceeded: {} total state frames vs max {} — \
             OOM/DoS prevention guardrail active.",
            total_state_frames, MAX_TOTAL_STATE_FRAMES
        ))
    } else {
        Ok(())
    }
}
