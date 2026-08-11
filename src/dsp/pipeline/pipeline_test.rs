// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use super::super::test_util::infra::{TrackingGuard, get_alloc_count};
    use super::super::*;
    use crate::common::params::AdaptiveComputeMode;
    use crate::common::spsc::RtStatusFlags;
    use crate::dsp::adaptive::AdaptiveCompute;
    use crate::dsp::gate::{DynamicHysteresis, GateParams};
    use crate::dsp::oversample::{OversampleEngine, OversampleFactor};
    use crate::dsp::resampler::NamResampler;

    use std::sync::atomic::Ordering;

    /// Helper function that simulates a lab for testing the audio engine (pipeline).
    /// It sets up everything needed to check if sound enters and exits correctly.
    pub(super) fn run_pipeline_test(
        host_rate: u32,
        nam_rate: u32,
        input_l: &[f32],
        input_r: &[f32],
        force_hold_zero: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = input_l.len();
        let mut resampler = NamResampler::new(host_rate, nam_rate, n).unwrap();
        let rt_status = RtStatusFlags::default();

        let mut bridge = Box::new(DspBridge {
            buffers: [
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
            ],
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        let mut gate_params = GateParams::default();
        if force_hold_zero {
            gate_params.hold_frames = 0;
            gate_params.mono_epsilon = 1.0;
        }
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut samples_l = input_l.to_vec();
        let mut samples_r = input_r.to_vec();

        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

        let mut os_engine_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
        let mut os_engine_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();

        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            os_l: &mut os_engine_l,
            os_r: &mut os_engine_r,
            active_model_l: &mut None,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.0,
            threshold_close_sq: 0.0,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            adaptive: &mut adaptive,
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
            conv: None,
        };

        let mut os_buf: [f32; MAX_RESAMP_BUF * 6] = [0.0f32; MAX_RESAMP_BUF * 6];
        let (os_in_l_slice, rest) = os_buf.split_at_mut(MAX_RESAMP_BUF);
        let (os_in_r_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
        let (os_model_l_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
        let (os_model_r_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
        let (crossfade_scratch_l, crossfade_scratch_r) = rest.split_at_mut(MAX_RESAMP_BUF);

        let bufs = DspBuffers {
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
            os_in_l: os_in_l_slice,
            os_in_r: os_in_r_slice,
            os_model_l: os_model_l_slice,
            os_model_r: os_model_r_slice,
            crossfade_scratch_l,
            crossfade_scratch_r,
        };

        let _guard = TrackingGuard::new();
        let n_processed = capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);
        let allocs = get_alloc_count();
        drop(_guard);

        assert_eq!(
            allocs, 0,
            "Allocation detected on hot-path! The system cannot allocate memory while processing audio."
        );

        let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
        let out_buf = &bridge.buffers[read_idx];
        let n_out = out_buf.n_samples as usize;

        assert_eq!(
            n_processed, n_out,
            "capture_dsp_pipeline return value n_processed must equal bridge n_out"
        );

        (
            out_buf.buf_l[..n_out].to_vec(),
            out_buf.buf_r[..n_out].to_vec(),
        )
    }
}

#[cfg(test)]
#[path = "pipeline_bypass_test.rs"]
mod bypass_test;

#[cfg(test)]
#[path = "pipeline_gate_test.rs"]
mod gate_test;

#[cfg(test)]
#[path = "pipeline_dither_test.rs"]
mod dither_test;
