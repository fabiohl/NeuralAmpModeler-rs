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
        let rt_status = RtStatusFlags::default();
        run_pipeline_test_with_status(
            host_rate,
            nam_rate,
            input_l,
            input_r,
            n,
            force_hold_zero,
            &rt_status,
        )
    }

    /// Same as [`run_pipeline_test`], but with an explicit `n_samples` and a
    /// caller-provided [`RtStatusFlags`] handle — used by the host-contract
    /// fault-injection tests (F-12 / T2.4).
    pub(super) fn run_pipeline_test_with_status(
        host_rate: u32,
        nam_rate: u32,
        input_l: &[f32],
        input_r: &[f32],
        n_samples: usize,
        force_hold_zero: bool,
        rt_status: &RtStatusFlags,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = n_samples;
        let mut resampler = NamResampler::new(host_rate, nam_rate, n).unwrap();

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
            rt_status,
            adaptive: &mut adaptive,
            // SAFETY: `bridge` is a heap-allocated `Box` kept alive for the whole test
            // function, outliving `ctx` and the `capture_dsp_pipeline` call, so the raw
            // pointer passed to `DspBridgeWriter::new` stays valid and non-null.
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

    /// F-12 / T2.4: a host passing `n_samples` beyond the slice lengths must
    /// be clamped defensively — no slice OOB panic on the pipeline entry — and
    /// the `RT_STATUS_HOST_CONTRACT_VIOLATION` flag must be raised.
    #[test]
    fn divergent_n_samples_clamps_and_raises_flag() {
        use crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION;

        let input_l = vec![0.01f32; 64];
        let input_r = vec![0.01f32; 64];
        let rt_status = RtStatusFlags::default();

        let (out_l, out_r) = run_pipeline_test_with_status(
            48000, 48000, &input_l, &input_r, 128, // 2× the slice length
            false, &rt_status,
        );

        assert!(
            rt_status.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION),
            "divergent n_samples must raise HOST_CONTRACT_VIOLATION"
        );
        assert_eq!(out_l.len(), 64, "pipeline clamps to the slice length");
        assert_eq!(out_r.len(), 64, "pipeline clamps to the slice length");
    }

    /// F-12 / T2.4: `n_samples` beyond `MAX_RESAMP_BUF` (with longer slices)
    /// must be clamped and raise `RT_STATUS_HOST_CONTRACT_VIOLATION`.
    #[test]
    fn over_max_resamp_buf_clamps_and_raises_flag() {
        use crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION;

        let input_l = vec![0.01f32; MAX_RESAMP_BUF + 512];
        let input_r = vec![0.01f32; MAX_RESAMP_BUF + 512];
        let rt_status = RtStatusFlags::default();

        let (out_l, _out_r) = run_pipeline_test_with_status(
            48000,
            48000,
            &input_l,
            &input_r,
            MAX_RESAMP_BUF + 512,
            false,
            &rt_status,
        );

        assert!(
            rt_status.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION),
            "n_samples > MAX_RESAMP_BUF must raise HOST_CONTRACT_VIOLATION"
        );
        assert!(
            out_l.len() <= MAX_RESAMP_BUF,
            "pipeline clamps to MAX_RESAMP_BUF, got {}",
            out_l.len()
        );
    }

    /// F-12 / T2.4: compliant `n_samples` must not raise the flag.
    #[test]
    fn contract_compliant_n_samples_does_not_raise_flag() {
        use crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION;

        let input_l = vec![0.01f32; 256];
        let input_r = vec![0.01f32; 256];
        let rt_status = RtStatusFlags::default();

        let (_out_l, _out_r) =
            run_pipeline_test_with_status(48000, 48000, &input_l, &input_r, 256, false, &rt_status);

        assert!(!rt_status.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));
    }

    /// F-04 / T3.2: the pipeline entry point must reassert FTZ (MXCSR bit 15)
    /// and DAZ (MXCSR bit 6) on the audio thread. The test clears both bits
    /// first, runs a full `capture_dsp_pipeline` call, then asserts the bits
    /// are active again — proving the entry point configures the per-thread
    /// MXCSR register, not just the helper function in isolation.
    #[test]
    fn pipeline_entry_reasserts_ftz_daz() {
        const DAZ_FTZ_MASK: u32 = 0x8040;

        // SAFETY: stmxcsr/ldmxcsr manipulate only the MXCSR register of the
        // current test thread; the operands are properly aligned `u32` locals
        // and `DAZ_FTZ_MASK` contains valid MXCSR control-flag bits.
        unsafe {
            let mut original: u32 = 0;
            core::arch::asm!("stmxcsr [{0}]", in(reg) &mut original);
            core::arch::asm!(
                "ldmxcsr [{0}]",
                in(reg) &(original & !DAZ_FTZ_MASK)
            );
        }

        let input_l = vec![0.01f32; 64];
        let input_r = vec![0.01f32; 64];
        let _ = run_pipeline_test(48000, 48000, &input_l, &input_r, false);

        // SAFETY: stmxcsr reads the current thread's MXCSR into a properly
        // aligned `u32` local; no memory is accessed.
        let after = unsafe {
            let mut mxcsr: u32 = 0;
            core::arch::asm!("stmxcsr [{0}]", in(reg) &mut mxcsr);
            mxcsr
        };

        assert_eq!(
            after & DAZ_FTZ_MASK,
            DAZ_FTZ_MASK,
            "pipeline entry must reassert FTZ+DAZ: MXCSR=0x{:08X}",
            after
        );
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
