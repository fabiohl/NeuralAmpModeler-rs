// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Adaptive control — evaluates CPU load state and adjusts model layer count or slimmable size.

use crate::common::spsc::RtStatusFlags;
use crate::dsp::adaptive::{AdaptiveCompute, AdaptiveComputeMode, AdaptiveState, SlimOverride};
use crate::models::StaticModel;

/// Maximum number of WaveNet layer states to backup during double-pass crossfade.
/// Worst-case: stereo (×2) × 8 arrays × 64 dilations = 1024 state entries.
pub(crate) const WAVENET_CROSSFADE_MAX_STATES: usize = 1024;

/// Evaluates adaptive compute FSM state and updates model effective layer count or slimmable size.
///
/// Returns `true` if model processing should be skipped for this block (e.g., minimal mode on LSTM).
#[inline(always)]
pub(crate) fn configure_adaptive_model(
    model_l: &mut Option<Box<StaticModel>>,
    model_r: &mut Option<Box<StaticModel>>,
    adaptive: &AdaptiveCompute,
    rt_status: &RtStatusFlags,
) -> bool {
    if adaptive.mode() == AdaptiveComputeMode::Off && adaptive.slim_override() == SlimOverride::Auto
    {
        return false;
    }

    let hold_layers = adaptive.is_crossfading();

    match adaptive.state() {
        AdaptiveState::Full => {
            if let Some(m) = model_l {
                let layers = m.layer_count();
                if !hold_layers {
                    m.set_effective_layers(layers);
                }
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            if let Some(m) = model_r {
                let layers = m.layer_count();
                if !hold_layers {
                    m.set_effective_layers(layers);
                }
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            false
        }
        AdaptiveState::Reduced => {
            if let Some(m) = model_l.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                if !hold_layers {
                    m.set_effective_layers(effective);
                }
            }
            if let Some(m) = model_l {
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            if let Some(m) = model_r.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                if !hold_layers {
                    m.set_effective_layers(effective);
                }
            }
            if let Some(m) = model_r {
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            false
        }
        AdaptiveState::Minimal => {
            let lstm_skip = model_l.as_ref().is_some_and(|m| m.is_lstm());
            if lstm_skip {
                return true;
            }
            if let Some(m) = model_l.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                if !hold_layers {
                    m.set_effective_layers(effective);
                }
            }
            if let Some(m) = model_l {
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            if let Some(m) = model_r.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                if !hold_layers {
                    m.set_effective_layers(effective);
                }
            }
            if let Some(m) = model_r {
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            false
        }
    }
}
