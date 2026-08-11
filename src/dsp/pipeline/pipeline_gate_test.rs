// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::*;
use crate::common::params::AdaptiveComputeMode;
use crate::common::spsc::RtStatusFlags;
use crate::dsp::adaptive::AdaptiveCompute;
use crate::dsp::gate::{DynamicHysteresis, GateParams};
use crate::dsp::oversample::{OversampleEngine, OversampleFactor};
use crate::dsp::pipeline::test_util::infra::{TrackingGuard, get_alloc_count};
use crate::dsp::resampler::NamResampler;
use std::sync::atomic::Ordering;

/// TEST: Processing Economy (Gate Closed and Silence).
/// If there is no sound and the gate is closed, the system should not waste time processing anything.
#[test]
fn test_hotpath_gate_closed_and_silence() {
    let n = 64; // Block size of samples (64 "little pieces" of sound).
    let input_l = vec![0.0; n]; // Silent input on left channel.
    let input_r = vec![0.0; n]; // Silent input on right channel.

    // We prepare the audio tools (resampler and data bridge).
    let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
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

    // Temporary working buffers for DSP calculations.
    let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
    let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
    let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
    let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
    let mut model_out_l = [0.0; MAX_RESAMP_BUF];
    let mut model_out_r = [0.0; MAX_RESAMP_BUF];

    // Noise gate configuration.
    let gate_params = GateParams::new(-70.0, -80.0, 0, 0, 1e-4);
    let mut silence_hysteresis = DynamicHysteresis::new();
    // Forces the gate to close manually for the test.
    // We simulate that the sound is very low (0.0) so that the gate closes.
    silence_hysteresis.update(0.0, 0.1, 0.01, &gate_params, 1000);
    silence_hysteresis.update(0.0, 0.1, 0.01, &gate_params, 1000);
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = false;

    let mut samples_l = input_l.clone();
    let mut samples_r = input_r.clone();

    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

    let mut os_engine_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut os_engine_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();

    // We group everything into the "Context" for processing.
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
        threshold_open_sq: 0.1,
        threshold_close_sq: 0.01,
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

    // Memory allocation watchdog.
    let _guard = TrackingGuard::new();
    // We run the audio orchestra (Pipeline).
    capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);
    let allocs = get_alloc_count();
    drop(_guard);

    // Verification: Allocating memory in the middle of audio is forbidden (causes pops).
    assert_eq!(allocs, 0, "Allocation on the critical path!");

    // Verification: The system must mark that the current state is silence.
    assert!(
        rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_SILENT),
        "The gate should be closed (silence)"
    );
    assert!(!rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_FADING));

    // Final verification: Since the gate is closed, no sample should have been sent to the bridge.
    let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
    let out_buf = &bridge.buffers[1 - read_idx];
    assert_eq!(
        out_buf.n_samples, 0,
        "There should be no processed samples when the gate is in absolute silence"
    );
}

/// TEST: Smooth Transition (FadeOut).
/// Verifies that the system correctly detects when the volume is gradually decreasing (fading).
#[test]
fn test_hotpath_gate_fading() {
    let n = 64;
    let mut input_l = vec![0.0; n];
    let mut input_r = vec![0.0; n];
    // We simulate a very weak signal, which should trigger smooth sound closing.
    for i in 0..n {
        input_l[i] = 0.05; // This value is between the "open" and "close" thresholds.
        input_r[i] = 0.05;
    }

    let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
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

    // We configure FadeOut to last 100 sound frames.
    let gate_params = GateParams::new(-70.0, -80.0, 0, 100, 1e-4);
    let mut silence_hysteresis = DynamicHysteresis::new();
    // First we open the sound (1.0) and then inject silence to start FadeOut.
    silence_hysteresis.update(1.0, 0.1, 0.0001, &gate_params, 100);

    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = false;

    let mut samples_l = vec![0.0; n];
    let mut samples_r = vec![0.0; n];

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
        threshold_open_sq: 0.1,
        threshold_close_sq: 0.01,
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
    capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);
    let allocs = get_alloc_count();
    drop(_guard);

    // Checks: No allocation and must indicate the "FADING" state.
    assert_eq!(allocs, 0);
    assert!(
        rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_FADING),
        "The system should indicate it is in the middle of a smooth close (FadeOut)"
    );
    assert!(!rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_SILENT));
}

/// TEST: Distortion Detection (Clipping).
/// Verifies that the system warns when volume exceeds the digital limit (1.0), causing unwanted noise.
#[test]
fn test_hotpath_clipping_detection() {
    let n = 64;
    let mut input_l = vec![0.0; n];
    let input_r = vec![0.0; n];
    // We force an impossible volume (1.5) on a specific sample to cause distortion.
    input_l[10] = 1.5;

    let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
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

    // Temporary working buffers for DSP calculations.
    let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
    let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
    let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
    let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
    let mut model_out_l = [0.0; MAX_RESAMP_BUF];
    let mut model_out_r = [0.0; MAX_RESAMP_BUF];

    let gate_params = GateParams::default();
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = false;

    let mut samples_l = input_l.clone();
    let mut samples_r = input_r.clone();

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
    capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);
    let allocs = get_alloc_count();
    drop(_guard);

    // Verification: Must detect the Clipping flag (RT_STATUS_HAS_CLIPPED).
    assert_eq!(allocs, 0);
    assert!(
        rt_status.check_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED),
        "The system should have detected that the sound exceeded the limit (clipping)"
    );
}

/// TEST: Dropped Frames Detection.
/// Verifies that the system detects when the computer is too slow and starts dropping audio packets
/// because whoever should process the sound can't keep up with reading in time.
#[test]
fn test_hotpath_dropped_frames() {
    let n = 64;
    let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
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
        // Simulates that whoever should "listen" to the sound (consumer) hasn't read anything yet.
        consumed_gen: std::sync::atomic::AtomicU64::new(0),
        dropped_frames: std::sync::atomic::AtomicU32::new(0),
    });

    let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
    let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
    let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
    let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
    let mut model_out_l = [0.0; MAX_RESAMP_BUF];
    let mut model_out_r = [0.0; MAX_RESAMP_BUF];
    let gate_params = GateParams::default();
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();

    // First pass: The pipeline processes the sound and stores it in the "bridge".
    let mut process_mono = false;
    let mut samples_l = vec![1.0; n];
    let mut samples_r = vec![1.0; n];

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
    capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);

    // Second pass: The system tries to process more sound, but sees that the bridge is still occupied
    // with the sound from the previous pass that nobody read.
    let mut process_mono2 = false;
    let mut samples_l2 = vec![1.0; n];
    let mut samples_r2 = vec![1.0; n];
    let mut os_engine_l2 = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut os_engine_r2 = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let ctx2 = DspPipelineContext {
        resampler: &mut resampler,
        os_l: &mut os_engine_l2,
        os_r: &mut os_engine_r2,
        active_model_l: &mut None,
        active_model_r: &mut None,
        input_gain_mult: 1.0,
        output_gain_mult: 1.0,
        gate_params: &gate_params,
        silence_hysteresis: &mut silence_hysteresis,
        mono_hysteresis: &mut mono_hysteresis,
        threshold_open_sq: 0.0,
        threshold_close_sq: 0.0,
        process_mono: &mut process_mono2,
        rt_status: &rt_status,
        adaptive: &mut adaptive,
        bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
        conv: None,
    };

    let mut os_buf2: [f32; MAX_RESAMP_BUF * 6] = [0.0f32; MAX_RESAMP_BUF * 6];
    let (os_in_l_slice2, rest2) = os_buf2.split_at_mut(MAX_RESAMP_BUF);
    let (os_in_r_slice2, rest2) = rest2.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_l_slice2, rest2) = rest2.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_r_slice2, rest2) = rest2.split_at_mut(MAX_RESAMP_BUF);
    let (crossfade_scratch_l2, crossfade_scratch_r2) = rest2.split_at_mut(MAX_RESAMP_BUF);

    let bufs2 = DspBuffers {
        resamp_mid_l: &mut resamp_mid_l,
        resamp_mid_r: &mut resamp_mid_r,
        resamp_out_l: &mut resamp_out_l,
        resamp_out_r: &mut resamp_out_r,
        model_out_l: &mut model_out_l,
        model_out_r: &mut model_out_r,
        os_in_l: os_in_l_slice2,
        os_in_r: os_in_r_slice2,
        os_model_l: os_model_l_slice2,
        os_model_r: os_model_r_slice2,
        crossfade_scratch_l: crossfade_scratch_l2,
        crossfade_scratch_r: crossfade_scratch_r2,
    };

    let _guard = TrackingGuard::new();
    // Here the system should be forced to discard this new sound packet.
    capture_dsp_pipeline(&mut samples_l2, &mut samples_r2, n, ctx2, bufs2, 48000);
    let allocs = get_alloc_count();
    drop(_guard);

    assert_eq!(allocs, 0);

    // The system must have incremented the dropped packets counter.
    let dropped = bridge.dropped_frames.load(Ordering::Relaxed);
    assert_eq!(dropped, 1, "Should have detected 1 dropped audio packet");
}
