// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 3: Output Gain, Fading, Clipping Detection, and Degrade Crossfade.

use crate::common::spsc::RtStatusFlags;
use crate::dsp::adaptive::AdaptiveCompute;
use crate::dsp::gate::DynamicHysteresis;
use crate::math::common::SimdMath;

use super::input::DENORMAL_DITHER_OFFSET;

/// Stage 3 with opt-in non-finite output sanitization.
///
/// Applies output gain, gate fading, clipping detection, and additionally sanitizes
/// any non-finite values (NaN / Inf) in the output buffers in-place, preventing
/// corrupted floats from escaping to the host DAC or downstream mixer.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "FFI design or complex DSP kernel signature required by construction"
)]
pub fn apply_output_stage_sanitized(
    resamp_out_l: &mut [f32],
    resamp_out_r: &mut [f32],
    n_pw: usize,
    output_gain_mult: f32,
    silence_hysteresis: &mut DynamicHysteresis,
    rt_status: &RtStatusFlags,
    process_mono: bool,
    adaptive: &mut AdaptiveCompute,
    sample_rate: u32,
) {
    apply_output_stage(
        resamp_out_l,
        resamp_out_r,
        n_pw,
        output_gain_mult,
        silence_hysteresis,
        rt_status,
        process_mono,
        adaptive,
        sample_rate,
    );
    crate::math::dsp::sanitize_nonfinite_f32(&mut resamp_out_l[..n_pw]);
    if !process_mono {
        crate::math::dsp::sanitize_nonfinite_f32(&mut resamp_out_r[..n_pw]);
    }
}

/// Stage 3: Output Gain, Fading, Clipping Detection, and Degrade Crossfade.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "FFI design or complex DSP kernel signature required by construction"
)]
pub fn apply_output_stage(
    resamp_out_l: &mut [f32],
    resamp_out_r: &mut [f32],
    n_pw: usize,
    output_gain_mult: f32,
    silence_hysteresis: &mut DynamicHysteresis,
    rt_status: &RtStatusFlags,
    process_mono: bool,
    adaptive: &mut AdaptiveCompute,
    sample_rate: u32,
) {
    #[cfg(feature = "avx512")]
    {
        use crate::math::common::Avx512Math;
        use crate::math::common::{Avx2Math, InstructionSet, effective_instruction_set};
        #[expect(deprecated)]
        match effective_instruction_set() {
            InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
                // SAFETY: inner invariants upheld by caller.
                unsafe {
                    apply_output_stage_inner::<Avx512Math>(
                        resamp_out_l,
                        resamp_out_r,
                        n_pw,
                        output_gain_mult,
                        silence_hysteresis,
                        rt_status,
                        process_mono,
                        adaptive,
                        sample_rate,
                    )
                }
            }
            InstructionSet::Avx2 => {
                // SAFETY: inner invariants upheld by caller.
                unsafe {
                    apply_output_stage_inner::<Avx2Math>(
                        resamp_out_l,
                        resamp_out_r,
                        n_pw,
                        output_gain_mult,
                        silence_hysteresis,
                        rt_status,
                        process_mono,
                        adaptive,
                        sample_rate,
                    )
                }
            }
        }
    }
    #[cfg(not(feature = "avx512"))]
    {
        use crate::math::common::Avx2Math;
        // SAFETY: inner invariants upheld by caller.
        unsafe {
            apply_output_stage_inner::<Avx2Math>(
                resamp_out_l,
                resamp_out_r,
                n_pw,
                output_gain_mult,
                silence_hysteresis,
                rt_status,
                process_mono,
                adaptive,
                sample_rate,
            )
        }
    }
}

/// Inner generic implementation of the output stage, monomorphized over SIMD backend.
///
/// # Safety
/// Caller must ensure valid buffer references and that the SIMD backend is
/// supported by the CPU.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "FFI design or complex DSP kernel signature required by construction"
)]
pub(crate) unsafe fn apply_output_stage_inner<M: SimdMath>(
    resamp_out_l: &mut [f32],
    resamp_out_r: &mut [f32],
    n_pw: usize,
    output_gain_mult: f32,
    silence_hysteresis: &mut DynamicHysteresis,
    rt_status: &RtStatusFlags,
    process_mono: bool,
    adaptive: &mut AdaptiveCompute,
    sample_rate: u32,
) {
    // 1. FINAL VOLUME ADJUSTMENT, CLIPPING PROTECTION, NOISE GATE, AND DENORMAL DITHER COMPENSATION
    // Fused into a single SIMD pass per channel to eliminate separate dither subtraction memory passes.
    if silence_hysteresis.is_steady() {
        let gate_mult = silence_hysteresis.multiplier();
        if gate_mult == 0.0 {
            resamp_out_l[..n_pw].fill(0.0);
            if !process_mono {
                resamp_out_r[..n_pw].fill(0.0);
            }
        } else if process_mono {
            let fused_gain = output_gain_mult * gate_mult;
            let has_clipped = {
                // SAFETY: slice is valid, gain and dither offset are finite.
                unsafe {
                    M::apply_gain_with_dither_and_detect_clipping_mono(
                        &mut resamp_out_l[..n_pw],
                        fused_gain,
                        DENORMAL_DITHER_OFFSET,
                    )
                }
            };
            if has_clipped {
                rt_status.set_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED);
            }
        } else {
            let fused_gain = output_gain_mult * gate_mult;
            let has_clipped = {
                // SAFETY: slices are valid, gain and dither offset are finite.
                unsafe {
                    M::apply_gain_with_dither_and_detect_clipping_stereo(
                        &mut resamp_out_l[..n_pw],
                        &mut resamp_out_r[..n_pw],
                        fused_gain,
                        DENORMAL_DITHER_OFFSET,
                    )
                }
            };
            if has_clipped {
                rt_status.set_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED);
            }
        }
    } else if process_mono {
        let has_clipped = {
            // SAFETY: slice is valid, gain and dither offset are finite.
            unsafe {
                M::apply_gain_with_dither_and_detect_clipping_mono(
                    &mut resamp_out_l[..n_pw],
                    output_gain_mult,
                    DENORMAL_DITHER_OFFSET,
                )
            }
        };
        silence_hysteresis.apply_gain_rt(&mut resamp_out_l[..n_pw], n_pw);
        if has_clipped {
            rt_status.set_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED);
        }
    } else {
        let has_clipped = {
            // SAFETY: slices are valid, gain and dither offset are finite.
            unsafe {
                M::apply_gain_with_dither_and_detect_clipping_stereo(
                    &mut resamp_out_l[..n_pw],
                    &mut resamp_out_r[..n_pw],
                    output_gain_mult,
                    DENORMAL_DITHER_OFFSET,
                )
            }
        };

        silence_hysteresis.apply_gain_rt_stereo::<M>(
            &mut resamp_out_l[..n_pw],
            &mut resamp_out_r[..n_pw],
            n_pw,
        );

        if has_clipped {
            rt_status.set_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED);
        }
    }

    // Advance the crossfade clock. The return value (multiplier) is ignored here
    // because the actual crossfade blending is performed in the inference stage
    // to avoid redundant resampling passes.
    let _ = adaptive.crossfade_multiplier(sample_rate, n_pw);

    // If non-finite input was detected earlier in the pipeline, guarantee that
    // the output buffers are sanitized as well before delivering to the host.
    if rt_status.check_flag(crate::common::spsc::RT_STATUS_NON_FINITE_INPUT_DETECTED) {
        crate::math::dsp::sanitize_nonfinite_f32(&mut resamp_out_l[..n_pw]);
        if !process_mono {
            crate::math::dsp::sanitize_nonfinite_f32(&mut resamp_out_r[..n_pw]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_stage_sanitized_cleans_non_finite() {
        let mut out_l = vec![0.5f32; 16];
        let mut out_r = vec![0.5f32; 16];
        out_l[3] = f32::NAN;
        out_r[7] = f32::INFINITY;

        let mut hysteresis = DynamicHysteresis::new();
        let rt_status = RtStatusFlags::default();
        let mut adaptive = AdaptiveCompute::new(crate::common::params::AdaptiveComputeMode::Off);

        apply_output_stage_sanitized(
            &mut out_l,
            &mut out_r,
            16,
            1.0,
            &mut hysteresis,
            &rt_status,
            false,
            &mut adaptive,
            48000,
        );

        assert!(out_l[3].is_finite());
        assert_eq!(out_l[3], 0.0);
        assert!(out_r[7].is_finite());
        assert_eq!(out_r[7], 0.0);
    }
}
