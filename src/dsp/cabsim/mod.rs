// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Cab Sim — IR convolution engine.
//!
//! Loads `.wav` impulse responses (mono, PCM16/24/float32),
//! resamples them to the active sample rate, and delivers them
//! to the DSP thread via lock-free SPSC transfer.

/// SPSC adapter: bridges the off-RT loader to the RT convolution engine.
pub mod adapter;
/// Frequency-domain block convolution kernel: partitioned overlap-save FFT.
pub mod conv;
pub(crate) mod ir_parse;
pub(crate) mod ir_resample;
/// IR loader: parses WAV files, resamples to target rate, builds convolution buffers.
pub mod loader;

use crate::models::StaticModel;

/// Checks recursively that `model` and all container submodels strictly satisfy `1 in -> 1 out`.
///
/// Returns `Err((in_channels, out_channels))` for the first stage or submodel that
/// violates single-channel I/O.
pub fn validate_single_channel_model(model: &StaticModel) -> Result<(), (usize, usize)> {
    let in_ch = model.in_channels();
    let out_ch = model.num_output_channels();
    if in_ch != 1 || out_ch != 1 {
        return Err((in_ch, out_ch));
    }
    if let StaticModel::Container(container) = model {
        for (_threshold, sub) in container.submodels() {
            validate_single_channel_model(sub)?;
        }
    }
    Ok(())
}

/// Computes the host-rate tail drain budget in samples for a neural cabinet model.
///
/// Scales the model's receptive field from `model_rate` to `host_rate` (ceiling
/// division), enforcing a 128-sample minimum floor for FIR/WaveNet topologies and
/// a bounded ~100 ms IIR ring-out (`[2048, 24_000]` samples) for LSTM models.
pub fn compute_nam_tail_budget(model: &StaticModel, model_rate: u32, host_rate: u32) -> usize {
    let scaled_rf = if model_rate > 0 && host_rate > 0 {
        ((model.receptive_field() as u64 * u64::from(host_rate)).div_ceil(u64::from(model_rate)))
            as usize
    } else {
        model.receptive_field()
    };
    if model.is_lstm() {
        let iir_tail = (host_rate as usize / 10).clamp(2048, 24_000);
        scaled_rf.max(iir_tail)
    } else {
        scaled_rf.max(128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::nam_json::{LinearImplementation, LinearTopology};
    use crate::models::linear::LinearModel;

    #[test]
    fn validate_single_channel_model_accepts_mono_and_rejects_multi() {
        let mono = StaticModel::Linear(Box::new(
            LinearModel::new(vec![0.0; 8], 0.0, LinearImplementation::Auto).unwrap(),
        ));
        assert_eq!(validate_single_channel_model(&mono), Ok(()));

        let stereo_topo = LinearTopology {
            in_channels: 2,
            out_channels: 2,
            receptive_field: 4,
            has_bias: false,
            implementation: LinearImplementation::Auto,
        };
        let stereo = StaticModel::Linear(Box::new(
            LinearModel::new_with_topology(stereo_topo, vec![0.0; 16], vec![]).unwrap(),
        ));
        assert_eq!(validate_single_channel_model(&stereo), Err((2, 2)));
    }

    #[test]
    fn compute_nam_tail_budget_scales_rf_and_enforces_floor() {
        let small = StaticModel::Linear(Box::new(
            LinearModel::new(vec![0.0; 8], 0.0, LinearImplementation::Auto).unwrap(),
        ));
        assert_eq!(compute_nam_tail_budget(&small, 48_000, 48_000), 128);

        let large = StaticModel::Linear(Box::new(
            LinearModel::new(vec![0.0; 1024], 0.0, LinearImplementation::Auto).unwrap(),
        ));
        assert_eq!(compute_nam_tail_budget(&large, 48_000, 96_000), 2048);
    }
}
