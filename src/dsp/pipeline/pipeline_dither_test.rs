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

#[test]
fn test_denormal_dither_mono_symmetry() {
    use super::super::stages::{apply_input_stage, apply_output_stage};

    let n = 64;
    let mut samples_l = vec![0.0_f32; n];
    let mut samples_r = vec![0.0_f32; n];
    let rt_status = RtStatusFlags::default();
    let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
    let gate_params = GateParams::new(-70.0, -80.0, 0, 0, 1e-4);
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = true;

    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

    let mut os_engine_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
    let mut os_engine_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();

    let mut ctx = DspPipelineContext {
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
        bridge_writer: None,
        conv: None,
    };

    // 1. Run input stage (under mono mode, R shouldn't get dither)
    apply_input_stage(&mut samples_l, &mut samples_r, n, &mut ctx);

    // Sanity check: verify process_mono is indeed true
    assert!(*ctx.process_mono);

    // Under mono mode, samples_l should have DENORMAL_DITHER_OFFSET, samples_r should not.
    for &val in &samples_l {
        assert!((val - 1.0e-11_f32).abs() < 1e-15_f32);
    }
    for &val in &samples_r {
        assert!(val.abs() < 1e-15_f32);
    }

    // 2. Run output stage (with process_mono = true, R shouldn't have dither subtracted)
    apply_output_stage(
        &mut samples_l,
        &mut samples_r,
        n,
        1.0,
        &mut silence_hysteresis,
        &rt_status,
        true,
        &mut adaptive,
        48000,
    );

    // After output stage, both should be back to exactly 0.0 (or within float epsilon).
    for &val in &samples_l {
        assert!(
            val.abs() < 1e-15_f32,
            "L channel DC offset is too high: {}",
            val
        );
    }
    for &val in &samples_r {
        assert!(
            val.abs() < 1e-15_f32,
            "R channel DC offset is too high: {}",
            val
        );
    }
}

/// TEST: Dither addition SIMD vs Scalar parity.
/// Ensures that the SIMD-optimized dither implementation is bit-exact with the
/// scalar reference implementation, across various buffer sizes and offsets,
/// and performs zero heap allocations (RT-safe).
#[test]
fn test_dither_simd_vs_scalar_bit_exact() {
    use crate::math::common::scalar_ref::apply_dither_add_fallback;
    use crate::math::dsp::gain::apply_dither_add_simd;

    let lengths = [
        0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 256, 512, 1024,
    ];
    let offsets = [1.0e-11_f32, -1.0e-11_f32, 0.5_f32, -0.5_f32];

    for &len in &lengths {
        for &offset in &offsets {
            let mut buf_simd: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin()).collect();
            let mut buf_scalar = buf_simd.clone();

            // Track allocations to ensure RT-safety of dither operations.
            let _guard = TrackingGuard::new();
            let start_allocs = get_alloc_count();

            // Apply SIMD-dispatched dither.
            apply_dither_add_simd(&mut buf_simd, offset);

            // Apply scalar reference dither.
            // SAFETY: buffer is valid for its lifetime, size is matching.
            unsafe {
                apply_dither_add_fallback(&mut buf_scalar, offset);
            }

            let end_allocs = get_alloc_count();
            drop(_guard);

            assert_eq!(
                end_allocs - start_allocs,
                0,
                "Allocation detected in dither hot-path for len {}",
                len
            );

            assert_eq!(
                buf_simd, buf_scalar,
                "Dither SIMD output is not bit-exact with scalar reference for len {} and offset {}",
                len, offset
            );
        }
    }
}
