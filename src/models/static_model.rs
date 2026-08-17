// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::slimmable::SlimmableModel;
use super::{NamModel, StaticModel};
use std::sync::Arc;

impl StaticModel {
    /// Injects `RtStatusFlags` into the model so it can signal its state
    /// to the UI via atomic flags.
    pub fn inject_rt_status(&mut self, rt_status: Arc<crate::common::spsc::RtStatusFlags>) {
        match self {
            Self::WavenetA2Full(m) => m.inject_rt_status(rt_status),
            Self::WavenetA2Lite(m) => m.inject_rt_status(rt_status),
            Self::WavenetA2Dyn(_) => {}
            Self::WavenetA2Cascade(_) => {}
            _ => {}
        }
    }

    /// Returns whether the model is capable of dynamic channel slimming (slimmable rebuilds).
    ///
    /// Conventional WaveNet models (fixed topology, no slimmable metadata) return `false`.
    /// `SlimmableContainer` and models explicitly marked as slimmable return `true`.
    pub fn is_slimmable_capable(&self) -> bool {
        match self {
            Self::Container(_) => true,
            Self::WavenetDyn(m) => m.is_slimmable_capable(),
            _ => false,
        }
    }

    /// Scalar processing path for LSTM models (exact tanh/sigmoid via libm).
    ///
    /// Only available under `#[cfg(test)]` or `#[cfg(feature = "testing")]`.
    /// For non-LSTM models, delegates to `process()`.
    #[cfg(any(test, feature = "testing"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
    pub fn process_scalar(&mut self, input: &[f32], output: &mut [f32]) {
        match self {
            Self::Lstm1x3(m) => m.process_scalar(input, output),
            Self::Lstm1x8(m) => m.process_scalar(input, output),
            Self::Lstm1x12(m) => m.process_scalar(input, output),
            Self::Lstm1x16(m) => m.process_scalar(input, output),
            Self::Lstm1x24(m) => m.process_scalar(input, output),
            Self::Lstm2x8(m) => m.process_scalar(input, output),
            Self::Lstm2x12(m) => m.process_scalar(input, output),
            Self::Lstm2x16(m) => m.process_scalar(input, output),
            Self::Lstm1x40(m) => m.process_scalar(input, output),
            Self::Lstm2x24(m) => m.process_scalar(input, output),
            Self::LstmDyn(m) => m.process_scalar(input, output),
            Self::ConvNet(m) => m.process(input, output),
            Self::WavenetA2Cascade(m) => m.process(input, output),
            other => other.process(input, output),
        }
    }

    /// Sets the effective number of layers for soft-degrade.
    /// Only applies to WaveNet variants. LSTM handles reduction at the pipeline level.
    #[inline(always)]
    pub fn set_effective_layers(&mut self, n: usize) {
        match self {
            Self::WavenetStandard(m) => m.set_effective_layers(n),
            Self::WavenetLite(m) => m.set_effective_layers(n),
            Self::WavenetFeather(m) => m.set_effective_layers(n),
            Self::WavenetNano(m) => m.set_effective_layers(n),
            Self::WavenetDyn(m) => m.set_effective_layers(n),
            // LSTM, A2, Container, and Linear: no-op — reduction handled at pipeline level
            Self::WavenetA2Cascade(_)
            | Self::WavenetA2Full(_)
            | Self::WavenetA2Lite(_)
            | Self::WavenetA2Dyn(_)
            | Self::Container(_)
            | Self::Lstm1x3(_)
            | Self::Lstm1x8(_)
            | Self::Lstm1x12(_)
            | Self::Lstm1x16(_)
            | Self::Lstm1x24(_)
            | Self::Lstm2x8(_)
            | Self::Lstm2x12(_)
            | Self::Lstm2x16(_)
            | Self::Lstm1x40(_)
            | Self::Lstm2x24(_)
            | Self::LstmDyn(_) => {}
            Self::Linear(_) => {}
            Self::ConvNet(_) => {}
        }
    }

    /// Backs up WaveNet buffer starts. No-op for non-WaveNet models.
    #[inline(always)]
    pub fn backup_buffer_starts(&self, starts: &mut [usize], offset: &mut usize) {
        match self {
            Self::WavenetStandard(m) => m.backup_buffer_starts(starts, offset),
            Self::WavenetLite(m) => m.backup_buffer_starts(starts, offset),
            Self::WavenetFeather(m) => m.backup_buffer_starts(starts, offset),
            Self::WavenetNano(m) => m.backup_buffer_starts(starts, offset),
            Self::WavenetDyn(m) => m.backup_buffer_starts(starts, offset),
            Self::WavenetA2Cascade(_) => {}
            _ => {}
        }
    }

    /// Restores WaveNet buffer starts. No-op for non-WaveNet models.
    #[inline(always)]
    pub fn restore_buffer_starts(&mut self, starts: &[usize], offset: &mut usize) {
        match self {
            Self::WavenetStandard(m) => m.restore_buffer_starts(starts, offset),
            Self::WavenetLite(m) => m.restore_buffer_starts(starts, offset),
            Self::WavenetFeather(m) => m.restore_buffer_starts(starts, offset),
            Self::WavenetNano(m) => m.restore_buffer_starts(starts, offset),
            Self::WavenetDyn(m) => m.restore_buffer_starts(starts, offset),
            Self::WavenetA2Cascade(_) => {}
            _ => {}
        }
    }

    /// Sets the slimmable quality level for `SlimmableModel` variants.
    ///
    /// Only applies to `Container`. Other variants are a no-op.
    ///
    /// `rt_status` — when on the RT hot-path, pass `Some(&rt_status)` so errors
    /// can be signaled atomically. Pass `None` in tests or off-RT contexts.
    #[inline(always)]
    pub fn set_slimmable_size(
        &mut self,
        val: f32,
        rt_status: Option<&crate::common::spsc::RtStatusFlags>,
    ) {
        match self {
            Self::Container(c) => c.set_slimmable_size(val, rt_status),
            Self::WavenetDyn(m) => m.set_slimmable_size(val, rt_status),
            _ => {}
        }
    }

    /// Returns the slimmable quality breakpoints for this model.
    ///
    /// Delegates to `ContainerModel::slimmable_breakpoints()` when this is a
    /// `Container` variant. Returns an empty vector for all other variants.
    pub fn slimmable_breakpoints(&self) -> Box<[f64]> {
        match self {
            Self::Container(c) => SlimmableModel::slimmable_breakpoints(c.as_ref()),
            Self::WavenetDyn(m) => SlimmableModel::slimmable_breakpoints(m.as_ref()),
            _ => Box::new([]),
        }
    }

    /// Returns the total number of layers for the model (0 for non-WaveNet).
    /// Used by the adaptive FSM to compute how many to keep.
    #[inline(always)]
    pub fn layer_count(&self) -> usize {
        match self {
            Self::WavenetStandard(m) => m.array1.layers.len(),
            Self::WavenetLite(m) => m.array1.layers.len(),
            Self::WavenetFeather(m) => m.array1.layers.len(),
            Self::WavenetNano(m) => m.array1.layers.len(),
            Self::WavenetDyn(m) => m.arrays[0].layers.len(),
            Self::WavenetA2Full(_) | Self::WavenetA2Lite(_) | Self::WavenetA2Dyn(_) => {
                crate::models::a2::A2_NUM_LAYERS
            }
            Self::WavenetA2Cascade(m) => m.arrays.iter().map(|a| a.num_layers).sum(),
            Self::Container(c) => c.active().layer_count(),
            Self::Lstm2x8(_) | Self::Lstm2x12(_) | Self::Lstm2x16(_) | Self::Lstm2x24(_) => 2,
            Self::LstmDyn(m) => m.layers.len(),
            Self::Lstm1x3(_)
            | Self::Lstm1x8(_)
            | Self::Lstm1x12(_)
            | Self::Lstm1x16(_)
            | Self::Lstm1x24(_)
            | Self::Lstm1x40(_) => 1,
            Self::Linear(_) => 0,
            Self::ConvNet(m) => m.blocks.len(),
        }
    }

    /// Returns a human-readable classification label for this model variant.
    ///
    /// Used by `nondist_validation.rs` to cross-check against the manifest's
    /// `expected_class` field and detect dispatcher routing errors.
    #[cold]
    pub fn class_label(&self) -> String {
        match self {
            Self::WavenetStandard(_) => "WaveNet A1 Standard (CH=16)".into(),
            Self::WavenetLite(_) => "WaveNet A1 Lite (CH=12)".into(),
            Self::WavenetFeather(_) => "WaveNet A1 Feather (CH=8)".into(),
            Self::WavenetNano(_) => "WaveNet A1 Nano (CH=4)".into(),
            Self::WavenetA2Full(_) => "WaveNet A2 (CH=8)".into(),
            Self::WavenetA2Lite(_) => "WaveNet A2 Lite (CH=3)".into(),
            Self::WavenetA2Dyn(m) => format!("WaveNet A2 (CH={})", m.channels),
            Self::WavenetA2Cascade(m) => format!("WaveNet A2 Cascade ({} arrays)", m.arrays.len()),
            Self::WavenetDyn(m) => {
                if m.arrays.len() == 2 {
                    "WaveNet A1 (Custom)".into()
                } else {
                    "WaveNet (Custom Layers)".into()
                }
            }
            Self::Container(c) => {
                let active_label = c.active().class_label();
                if active_label.starts_with("WaveNet") {
                    active_label
                } else {
                    "SlimmableContainer".into()
                }
            }
            Self::Lstm1x3(_) => "LSTM 1x3".into(),
            Self::Lstm1x8(_) => "LSTM 1x8".into(),
            Self::Lstm1x12(_) => "LSTM 1x12".into(),
            Self::Lstm1x16(_) => "LSTM 1x16".into(),
            Self::Lstm1x24(_) => "LSTM 1x24".into(),
            Self::Lstm1x40(_) => "LSTM 1x40".into(),
            Self::Lstm2x8(_) => "LSTM 2x8".into(),
            Self::Lstm2x12(_) => "LSTM 2x12".into(),
            Self::Lstm2x16(_) => "LSTM 2x16".into(),
            Self::Lstm2x24(_) => "LSTM 2x24".into(),
            Self::LstmDyn(m) => format!("LSTM {}x{}", m.layers.len(), m.head_weights.len()),
            Self::Linear(_) => "Linear".into(),
            Self::ConvNet(m) => format!("ConvNet (CH={})", m.in_channels()),
        }
    }

    /// Returns `true` if this is an LSTM model.
    #[inline(always)]
    pub fn is_lstm(&self) -> bool {
        matches!(
            self,
            Self::Lstm1x3(_)
                | Self::Lstm1x8(_)
                | Self::Lstm1x12(_)
                | Self::Lstm1x16(_)
                | Self::Lstm1x24(_)
                | Self::Lstm2x8(_)
                | Self::Lstm2x12(_)
                | Self::Lstm2x16(_)
                | Self::Lstm1x40(_)
                | Self::Lstm2x24(_)
                | Self::LstmDyn(_)
        )
    }

    /// Returns `true` if this is a WaveNet model.
    #[inline(always)]
    pub fn is_wavenet(&self) -> bool {
        matches!(
            self,
            Self::WavenetStandard(_)
                | Self::WavenetLite(_)
                | Self::WavenetFeather(_)
                | Self::WavenetNano(_)
                | Self::WavenetA2Full(_)
                | Self::WavenetA2Lite(_)
                | Self::WavenetA2Dyn(_)
                | Self::WavenetA2Cascade(_)
                | Self::WavenetDyn(_)
        )
    }

    /// Returns `true` if this model supports layer skipping (WaveNet A1 variants only).
    /// A2 architectures use `MirroredBuffer` + `layer_buffer_starts`/`head_write_pos`
    /// and cannot safely participate in the double-pass crossfade mechanism.
    #[inline(always)]
    pub fn supports_layer_skip(&self) -> bool {
        matches!(
            self,
            Self::WavenetStandard(_)
                | Self::WavenetLite(_)
                | Self::WavenetFeather(_)
                | Self::WavenetNano(_)
                | Self::WavenetDyn(_)
        )
    }

    /// Returns `true` if this is a SlimmableContainer model.
    #[inline(always)]
    pub fn is_container(&self) -> bool {
        matches!(self, Self::Container(_))
    }

    /// Returns the number of channels inside the model.
    pub fn channels(&self) -> usize {
        match self {
            Self::WavenetStandard(_) => 16,
            Self::WavenetLite(_) => 12,
            Self::WavenetFeather(_) => 8,
            Self::WavenetNano(_) => 4,
            Self::WavenetA2Full(_) => 8,
            Self::WavenetA2Lite(_) => 3,
            Self::WavenetA2Dyn(m) => m.channels,
            Self::WavenetA2Cascade(m) => m.channels(),
            Self::WavenetDyn(m) => m.ch,
            Self::Container(c) => c.active().channels(),
            Self::Lstm1x3(_) => 3,
            Self::Lstm1x8(_) | Self::Lstm2x8(_) => 8,
            Self::Lstm1x12(_) | Self::Lstm2x12(_) => 12,
            Self::Lstm1x16(_) | Self::Lstm2x16(_) => 16,
            Self::Lstm1x24(_) | Self::Lstm2x24(_) => 24,
            Self::Lstm1x40(_) => 40,
            Self::LstmDyn(m) => m.head_weights.len(),
            Self::Linear(_) => 1,
            Self::ConvNet(m) => m.in_channels(),
        }
    }

    /// Returns the number of input channels consumed by the model's `process()`.
    ///
    /// - WaveNet variants and LSTM: always 1 (single sample per frame).
    /// - ConvNet: `in_channels` from configuration (defaults to 1).
    /// - Container: delegates to active sub-model.
    pub fn in_channels(&self) -> usize {
        match self {
            Self::WavenetStandard(_)
            | Self::WavenetLite(_)
            | Self::WavenetFeather(_)
            | Self::WavenetNano(_) => 1,
            Self::WavenetA2Full(_) => 1,
            Self::WavenetA2Lite(_) => 1,
            Self::WavenetA2Dyn(_) => 1,
            Self::WavenetA2Cascade(_) => 1,
            Self::WavenetDyn(_) => 1,
            Self::Container(c) => c.active().in_channels(),
            Self::Lstm1x3(_)
            | Self::Lstm1x8(_)
            | Self::Lstm1x12(_)
            | Self::Lstm1x16(_)
            | Self::Lstm1x24(_)
            | Self::Lstm2x8(_)
            | Self::Lstm2x12(_)
            | Self::Lstm2x16(_)
            | Self::Lstm1x40(_)
            | Self::Lstm2x24(_)
            | Self::LstmDyn(_) => 1,
            Self::Linear(_) => 1,
            Self::ConvNet(m) => m.in_channels(),
        }
    }

    /// Returns the receptive field size of the model (or 0 for LSTM/Container).
    pub fn receptive_field(&self) -> usize {
        match self {
            Self::Container(_) => 0,
            Self::Linear(m) => m.receptive_field,
            _ if self.is_lstm() => 0,
            _ => self.prewarm_samples(),
        }
    }

    /// Returns the number of output channels produced by the model's `process()`.
    ///
    /// Mirrors C++ `DSP::NumOutputChannels()`:
    /// - WaveNet A1 without post-stack head: last array's `head_size` (`wave_net_output_channels`)
    /// - LSTM: `hidden_size`
    /// - Linear: 1
    /// - Container: delegates to active sub-model
    pub fn num_output_channels(&self) -> usize {
        match self {
            Self::WavenetStandard(_)
            | Self::WavenetLite(_)
            | Self::WavenetFeather(_)
            | Self::WavenetNano(_) => 1,
            Self::WavenetA2Full(_) => 1,
            Self::WavenetA2Lite(_) => 1,
            Self::WavenetA2Dyn(_) => 1,
            Self::WavenetA2Cascade(m) => m.arrays.last().map(|a| a.head_size).unwrap_or(1),
            Self::WavenetDyn(m) => m.arrays.last().map(|a| a.head).unwrap_or(0),
            Self::Container(c) => c.active().num_output_channels(),
            Self::Lstm1x3(_)
            | Self::Lstm1x8(_)
            | Self::Lstm2x8(_)
            | Self::Lstm1x12(_)
            | Self::Lstm2x12(_)
            | Self::Lstm1x16(_)
            | Self::Lstm2x16(_)
            | Self::Lstm1x24(_)
            | Self::Lstm2x24(_)
            | Self::Lstm1x40(_)
            | Self::LstmDyn(_) => 1,
            Self::Linear(_) => 1,
            Self::ConvNet(m) => m.out_channels(),
        }
    }
}

/// Clones a `StaticModel` for use in `condition_dsp` preservation during
/// channel slimming (`slice_wavenet_model`).
///
/// Only `WavenetDyn` is currently supported (the only variant that can appear
/// as a recursive `condition_dsp` sub-model in the current dispatcher).
/// All other variants return `None`.
pub(crate) fn clone_condition_dsp(model: &Option<Box<StaticModel>>) -> Option<Box<StaticModel>> {
    model.as_ref().and_then(|m| match m.as_ref() {
        StaticModel::WavenetDyn(w) => Some(Box::new(StaticModel::WavenetDyn(w.clone()))),
        StaticModel::WavenetA2Cascade(_) => None,
        _ => None,
    })
}
