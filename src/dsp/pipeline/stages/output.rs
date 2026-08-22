// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 3: Output Gain, Fading, Clipping Detection, and Degrade Crossfade.

use crate::common::spsc::RtStatusFlags;
use crate::dsp::adaptive::AdaptiveCompute;
use crate::dsp::gate::DynamicHysteresis;
use crate::math::common::SimdMath;

use super::input::DENORMAL_DITHER_OFFSET;

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
    use crate::math::common::Avx512Math;
    use crate::math::common::{Avx2Math, InstructionSet, effective_instruction_set};
    #[expect(deprecated)]
    match effective_instruction_set() {
        #[cfg(feature = "avx512")]
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
        #[cfg(not(feature = "avx512"))]
        InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
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
    // 0. DENORMAL SUPPRESSION COMPENSATION: Subtract the injected DC offset
    // SAFETY: slice is valid and offset is a finite f32.
    unsafe { M::apply_dither_add(&mut resamp_out_l[..n_pw], -DENORMAL_DITHER_OFFSET) };
    if !process_mono {
        // SAFETY: slice is valid and offset is a finite f32.
        unsafe { M::apply_dither_add(&mut resamp_out_r[..n_pw], -DENORMAL_DITHER_OFFSET) };
    }

    // 1. FINAL VOLUME ADJUSTMENT, CLIPPING PROTECTION, AND NOISE GATE (FUSED WHEN POSSIBLE)
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
                // SAFETY: slice is valid and gain is finite.
                unsafe {
                    M::apply_gain_and_detect_clipping_mono(&mut resamp_out_l[..n_pw], fused_gain)
                }
            };
            if has_clipped {
                rt_status.set_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED);
            }
        } else {
            let fused_gain = output_gain_mult * gate_mult;
            let has_clipped = {
                // SAFETY: slices are valid and gain is finite.
                unsafe {
                    M::apply_gain_and_detect_clipping_stereo(
                        &mut resamp_out_l[..n_pw],
                        &mut resamp_out_r[..n_pw],
                        fused_gain,
                    )
                }
            };
            if has_clipped {
                rt_status.set_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED);
            }
        }
    } else if process_mono {
        let has_clipped = {
            // SAFETY: slice is valid and gain is finite.
            unsafe {
                M::apply_gain_and_detect_clipping_mono(&mut resamp_out_l[..n_pw], output_gain_mult)
            }
        };
        silence_hysteresis.apply_gain_rt(&mut resamp_out_l[..n_pw], n_pw);
        if has_clipped {
            rt_status.set_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED);
        }
    } else {
        let has_clipped = {
            // SAFETY: slices are valid and gain is finite.
            unsafe {
                M::apply_gain_and_detect_clipping_stereo(
                    &mut resamp_out_l[..n_pw],
                    &mut resamp_out_r[..n_pw],
                    output_gain_mult,
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
}
