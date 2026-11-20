// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Routing helpers — stereo/mono dispatch, oversampled model processing, and passthrough.

use crate::common::spsc::RtStatusFlags;
use crate::dsp::oversample::OversampleEngine;
use crate::models::{NamModel, StaticModel};

/// Processes stereo/mono through neural models with optional half-band oversampling.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "FFI design or complex DSP kernel signature required by construction"
)]
pub(crate) fn model_process_stereo_with_os(
    os_l: &mut OversampleEngine,
    os_r: &mut OversampleEngine,
    active_model_l: &mut Option<Box<StaticModel>>,
    active_model_r: &mut Option<Box<StaticModel>>,
    model_in_l: &[f32],
    model_in_r: &[f32],
    os_buf_l: &mut [f32],
    os_buf_r: &mut [f32],
    os_model_l: &mut [f32],
    os_model_r: &mut [f32],
    native_out_l: &mut [f32],
    native_out_r: &mut [f32],
    process_mono: bool,
    rt_status: Option<&RtStatusFlags>,
) {
    // L channel
    if os_l.is_bypass() {
        if let Some(m) = active_model_l {
            m.process(model_in_l, native_out_l);
        } else {
            passthru(model_in_l, native_out_l);
        }
    } else {
        let n_os = os_l.upsample(model_in_l, os_buf_l, rt_status);
        if let Some(m) = active_model_l {
            m.process(&os_buf_l[..n_os], &mut os_model_l[..n_os]);
        } else {
            passthru(&os_buf_l[..n_os], &mut os_model_l[..n_os]);
        }
        let written = os_l.downsample(&os_model_l[..n_os], native_out_l, rt_status);
        finish_downsample(written, model_in_l.len(), native_out_l, rt_status);
    }

    // R channel (or mono copy)
    if process_mono {
        if native_out_r.len() >= native_out_l.len() {
            passthru(native_out_l, native_out_r);
        }
    } else if os_r.is_bypass() {
        if let Some(m) = active_model_r {
            m.process(model_in_r, native_out_r);
        } else {
            passthru(model_in_r, native_out_r);
        }
    } else {
        let n_os = os_r.upsample(model_in_r, os_buf_r, rt_status);
        if let Some(m) = active_model_r {
            m.process(&os_buf_r[..n_os], &mut os_model_r[..n_os]);
        } else {
            passthru(&os_buf_r[..n_os], &mut os_model_r[..n_os]);
        }
        let written = os_r.downsample(&os_model_r[..n_os], native_out_r, rt_status);
        finish_downsample(written, model_in_r.len(), native_out_r, rt_status);
    }
}

/// Enforces the downsample cardinality contract at the native boundary.
///
/// With the half-band warm-up gate removed (T3.4), an X2/X4 stage emits exactly
/// `input.len() / factor` samples, so a well-formed host block yields
/// `written == n_native`. If a stage ever under-produces, the tail is
/// zero-filled (never stale samples from a previous block) and
/// `RT_STATUS_HOST_CONTRACT_VIOLATION` is latched — fail-closed and RT-safe.
#[inline(always)]
pub(crate) fn finish_downsample(
    written: usize,
    n_native: usize,
    native_out: &mut [f32],
    rt_status: Option<&RtStatusFlags>,
) {
    debug_assert_eq!(
        written, n_native,
        "downsample cardinality contract violated: wrote {written}, expected {n_native}"
    );
    if written < n_native {
        let end = n_native.min(native_out.len());
        if written < end {
            native_out[written..end].fill(0.0);
        }
        if let Some(s) = rt_status {
            s.set_flag(crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION);
        }
    }
}

/// Copies `model_in` through to `model_out` when no model is active.
#[inline(always)]
pub(crate) fn passthru(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len().min(out_buf.len());
    // SAFETY: `n` is the minimum of the two lengths, so the copy of `n` initialized
    // `f32` values stays within both ranges; `in_buf`/`out_buf` are distinct buffers.
    unsafe {
        core::ptr::copy_nonoverlapping(in_buf.as_ptr(), out_buf.as_mut_ptr(), n);
    }
}
