// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Half-Band Oversampling Engine for the neural stage.
//!
//! Implements optional 2×/4× oversampling around the neural model to reduce
//! aliasing from non-linear activations, following the half-band filter design
//! principles of Kahles, Esqueda & Välimäki (JAES 2019).
//!
//! ## Architecture
//!
//! Each 2× stage uses a Kaiser-windowed half-band FIR filter (25 taps, β=12,
//! \>100 dB stop-band). The half-band property h\[2n\]=0 (n≠D/2) halves the
//! effective MAC count per sample.
//!
//! - **Upsampler**: inserts zeros → filters. Even outputs = x[n-D/2]*0.5;
//!   odd outputs = convolution with non-zero odd taps.
//! - **Downsampler**: FIR at full rate → decimates by 2. Uses contiguous
//!   double-buffer delay line (same pattern as `NamResampler`).
//!
//! ## RT-Safety
//!
//! All allocation in `OversampleEngine::new()`. During `process()`,
//! only pre-allocated buffers — zero alloc, zero heap-drop.
//!
//! Factor change requires rebuild (off-RT), same path as model hot-swap.

use super::stage::X2Stage;
use crate::common::diagnostics::NamErrorCode;
use crate::common::spsc::{RT_STATUS_HOST_CONTRACT_VIOLATION, RtStatusFlags};
use crate::math::common::AlignedVec;

/// Half-band FIR filter length (≡ 1 mod 4 so D=HB_TAPS/2 is even).
/// 25 taps, D=12. Kaiser β=12 → >100 dB stop-band rejection.
pub(crate) const HB_TAPS: usize = 25;
/// Filter delay (group delay = HB_TAPS/2 = 12 samples at native rate).
pub(crate) const HB_DELAY: usize = HB_TAPS / 2;

/// Oversampling factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum OversampleFactor {
    /// No oversampling — pass-through with zero overhead.
    #[default]
    Off,
    /// 2× oversampling (one half-band stage).
    X2,
    /// 4× oversampling (two cascaded half-band stages).
    X4,
}

impl OversampleFactor {
    /// Creates an `OversampleFactor` from a host parameter value (0.0, 1.0, 2.0).
    pub fn from_f32(val: f32) -> Self {
        match val.round() as i32 {
            1 => Self::X2,
            2 => Self::X4,
            _ => Self::Off,
        }
    }

    /// Converts to its host parameter value (0.0, 1.0, 2.0).
    pub fn to_f32(self) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::X2 => 1.0,
            Self::X4 => 2.0,
        }
    }

    /// Returns the sample count multiplier (1, 2, or 4).
    #[inline]
    pub const fn multiplier(self) -> usize {
        match self {
            Self::Off => 1,
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }

    /// Returns the number of cascaded 2× stages (0, 1, or 2).
    #[inline]
    pub const fn stage_count(self) -> usize {
        match self {
            Self::Off => 0,
            Self::X2 => 1,
            Self::X4 => 2,
        }
    }
}

/// Typed bundle of oversampling stages, eliminating `Option::unwrap`
/// on the RT hot-path. Variant discriminant is a zero-cost compile-time
/// guarantee that the stages (when present) are always valid.
enum OsStages {
    Off,
    X2 {
        stage1: X2Stage,
    },
    X4 {
        stage1: X2Stage,
        stage2: Box<X2Stage>,
    },
}

/// RT-safe half-band oversampling engine.
///
/// Wraps 1–2 cascaded 2× stages. Off mode is zero-cost pass-through
/// (no state, infallible `copy_nonoverlapping`).
///
/// ## Usage (per stereo channel)
///
/// ```ignore
/// // 1. Upsample model-rate input to oversampled rate
/// let n_os = engine.upsample(&input[..n_native], os_up_buf);
/// // 2. Model processes at oversampled rate
/// model.process(&os_up_buf[..n_os], &mut os_model_buf[..n_os]);
/// // 3. Downsample back to native rate
/// let n_out = engine.downsample(&os_model_buf[..n_os], output);
/// ```
pub struct OversampleEngine {
    factor: OversampleFactor,
    stages: OsStages,
    /// Scratch for X4 cascaded inter-stage (2× intermediate).
    /// Sized for max_input × 2 (the intermediate upsampled rate between stages).
    inter_buf: AlignedVec<f32>,
    max_samples: usize,
}

impl OversampleEngine {
    /// Creates a new engine with pre-allocated buffers.
    ///
    /// `max_input_samples`: max block size at native model rate
    /// (e.g., `MAX_RESAMP_BUF = 8192`).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use neural_amp_modeler_rs::dsp::oversample::{OversampleEngine, OversampleFactor};
    ///
    /// const BLOCK_SIZE: usize = 128;
    /// let mut engine = OversampleEngine::new(OversampleFactor::X4, BLOCK_SIZE)
    ///     .expect("Failed to create oversampling engine");
    ///
    /// // Upsample → model processes at 4× rate → downsample back
    /// let multiplier = OversampleFactor::X4.multiplier();
    /// let input = [0.1_f32; BLOCK_SIZE];
    /// let mut os_buf = [0.0_f32; BLOCK_SIZE * 4];
    /// let mut output = [0.0_f32; BLOCK_SIZE];
    /// let n_os = engine.upsample(&input, &mut os_buf, None);
    /// engine.downsample(&os_buf[..n_os], &mut output, None);
    /// ```
    pub fn new(factor: OversampleFactor, max_input_samples: usize) -> Result<Self, NamErrorCode> {
        let inter_size = if factor.stage_count() >= 2 {
            max_input_samples * 2
        } else {
            1
        };

        let stages = match factor {
            OversampleFactor::Off => OsStages::Off,
            OversampleFactor::X2 => OsStages::X2 {
                stage1: X2Stage::new()?,
            },
            OversampleFactor::X4 => OsStages::X4 {
                stage1: X2Stage::new()?,
                stage2: Box::new(X2Stage::new()?),
            },
        };
        Ok(Self {
            factor,
            stages,
            inter_buf: AlignedVec::new(inter_size, 0.0f32)?,
            max_samples: max_input_samples,
        })
    }

    /// Returns the current oversampling factor.
    #[inline]
    pub fn factor(&self) -> OversampleFactor {
        self.factor
    }

    /// Returns `true` when oversampling is bypassed (Off).
    #[inline]
    pub fn is_bypass(&self) -> bool {
        matches!(self.stages, OsStages::Off)
    }

    /// Returns the group delay in samples at the native (model) rate.
    ///
    /// Each 2× half-band stage introduces HB_DELAY (= 12) samples.
    /// Off → 0, X2 → 12, X4 → 24.
    #[inline]
    pub fn latency_samples(&self) -> usize {
        match self.factor {
            OversampleFactor::Off => 0,
            OversampleFactor::X2 => HB_DELAY,
            OversampleFactor::X4 => 2 * HB_DELAY,
        }
    }

    /// Upsamples mono input from native rate to oversampled rate.
    ///
    /// `output` must have room for `input.len() * factor.multiplier()` samples.
    /// Returns number of oversampled samples written.
    ///
    /// Fail-closed (F-05): the output capacity contract is enforced in release
    /// builds, not just by `debug_assert!`. When `output` is smaller than
    /// required, the processed input is clamped so the stages never write past
    /// `output`, and `RT_STATUS_HOST_CONTRACT_VIOLATION` is raised when
    /// `rt_status` is provided.
    ///
    /// Input blocks larger than `max_samples` are truncated defensively (F-12 /
    /// T2.4): when `rt_status` is provided the truncation raises
    /// `RT_STATUS_HOST_CONTRACT_VIOLATION` — no more silent truncation.
    pub fn upsample(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        rt_status: Option<&RtStatusFlags>,
    ) -> usize {
        let multiplier = self.factor.multiplier();
        let mut n_in = input.len().min(self.max_samples);
        // F-05: clamp to the output capacity so the half-band stages always
        // write within `output` (previously release builds relied on a
        // `debug_assert!` + `core::hint::assert_unchecked`, which was UB).
        n_in = n_in.min(output.len() / multiplier);
        if n_in != input.len()
            && let Some(rt) = rt_status
        {
            rt.set_flag(RT_STATUS_HOST_CONTRACT_VIOLATION);
        }
        let input = &input[..n_in];
        debug_assert!(input.len() <= self.max_samples);
        debug_assert!(output.len() >= input.len() * self.factor.multiplier());

        match &mut self.stages {
            OsStages::Off => {
                let n = input.len().min(output.len());
                // SAFETY: both input and output are valid for `n` elements
                // (`n` is the minimum of both lengths, computed from the
                // caller-provided slices which are already in scope). The
                // regions do not overlap because `output` is a distinct
                // mutable buffer.
                unsafe {
                    core::ptr::copy_nonoverlapping(input.as_ptr(), output.as_mut_ptr(), n);
                }
                n
            }
            OsStages::X2 { stage1 } => stage1.upsample(input, output),
            OsStages::X4 { stage1, stage2 } => {
                let n_x2 = stage1.upsample(input, &mut self.inter_buf[..input.len() * 2]);
                stage2.upsample(&self.inter_buf[..n_x2], output)
            }
        }
    }

    /// Downsamples mono input from oversampled rate back to native rate.
    ///
    /// `output` must have room for `input.len() / factor.multiplier()` samples.
    /// Returns number of native-rate samples written.
    ///
    /// Fail-closed (F-05): the output capacity contract is enforced in release
    /// builds, not just by `debug_assert!`. When `output` is smaller than
    /// required, the processed input is clamped and
    /// `RT_STATUS_HOST_CONTRACT_VIOLATION` is raised when `rt_status` is
    /// provided.
    ///
    /// Input blocks larger than `max_samples × multiplier` are truncated
    /// defensively (F-12 / T2.4): when `rt_status` is provided the truncation
    /// raises `RT_STATUS_HOST_CONTRACT_VIOLATION` — no more silent truncation.
    pub fn downsample(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        rt_status: Option<&RtStatusFlags>,
    ) -> usize {
        let multiplier = self.factor.multiplier();
        let max_os = self.max_samples * multiplier;
        let mut n_in = input.len().min(max_os);
        // F-05: `output` must hold `n_in / multiplier` samples; clamp the
        // processed input so the decimation stages never write past `output`.
        n_in = n_in.min(output.len().saturating_mul(multiplier));
        if n_in != input.len()
            && let Some(rt) = rt_status
        {
            rt.set_flag(RT_STATUS_HOST_CONTRACT_VIOLATION);
        }
        let input = &input[..n_in];
        debug_assert!(
            output.len() >= input.len() / self.factor.multiplier(),
            "oversample: output buffer too small for downsampling factor"
        );
        match &mut self.stages {
            OsStages::Off => {
                let n = input.len().min(output.len());
                // SAFETY: both input and output are valid for `n` elements
                // (minimum of both lengths). The regions do not overlap
                // because `output` is a distinct mutable buffer.
                unsafe {
                    core::ptr::copy_nonoverlapping(input.as_ptr(), output.as_mut_ptr(), n);
                }
                n
            }
            OsStages::X2 { stage1 } => stage1.downsample(input, output),
            OsStages::X4 { stage1, stage2 } => {
                let n_x2 = stage2.downsample(input, &mut self.inter_buf[..input.len() / 2]);
                stage1.downsample(&self.inter_buf[..n_x2], output)
            }
        }
    }
}

/// Atomic bundle of stereo oversampling engines delivered via SPSC.
///
/// L and R engines are built together on the main thread and consumed
/// together on the RT thread, ensuring they always share the same factor.
pub struct OsEnginePair {
    /// Left-channel oversampling engine.
    pub l: Box<OversampleEngine>,
    /// Right-channel oversampling engine.
    pub r: Box<OversampleEngine>,
}

#[cfg(test)]
#[path = "oversample_test.rs"]
mod oversample_test;
