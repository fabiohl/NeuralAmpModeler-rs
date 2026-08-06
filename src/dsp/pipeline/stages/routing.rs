// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Routing helpers — stereo/mono dispatch, oversampled model processing, and passthrough.

use crate::dsp::oversample::OversampleEngine;
use crate::models::{NamModel, StaticModel};

/// Unified helper for mono/stereo processing of neural models.
///
/// Processes the L channel model (_always_) and decides whether the R channel is a mono copy
/// or independent processing via the active R model.
#[inline(always)]
pub(crate) fn run_stereo_or_mono(
    active_model_l: &mut Option<Box<StaticModel>>,
    active_model_r: &mut Option<Box<StaticModel>>,
    model_in_l: &[f32],
    model_in_r: &[f32],
    m_out_l: &mut [f32],
    m_out_r: &mut [f32],
    process_mono: bool,
) {
    if let Some(model_l) = active_model_l {
        model_l.process(model_in_l, m_out_l);
    } else {
        unsafe {
            core::ptr::copy_nonoverlapping(
                model_in_l.as_ptr(),
                m_out_l.as_mut_ptr(),
                model_in_l.len().min(m_out_l.len()),
            );
        }
    }

    if process_mono {
        unsafe {
            core::ptr::copy_nonoverlapping(
                m_out_l.as_ptr(),
                m_out_r.as_mut_ptr(),
                m_out_l.len().min(m_out_r.len()),
            );
        }
    } else if let Some(model_r) = active_model_r {
        model_r.process(model_in_r, m_out_r);
    } else {
        unsafe {
            core::ptr::copy_nonoverlapping(
                model_in_r.as_ptr(),
                m_out_r.as_mut_ptr(),
                model_in_r.len().min(m_out_r.len()),
            );
        }
    }
}

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
) {
    // L channel
    if os_l.is_bypass() {
        if let Some(m) = active_model_l {
            m.process(model_in_l, native_out_l);
        } else {
            passthru(model_in_l, native_out_l);
        }
    } else {
        let n_os = os_l.upsample(model_in_l, os_buf_l);
        if let Some(m) = active_model_l {
            m.process(&os_buf_l[..n_os], &mut os_model_l[..n_os]);
        } else {
            passthru(&os_buf_l[..n_os], &mut os_model_l[..n_os]);
        }
        os_l.downsample(&os_model_l[..n_os], native_out_l);
    }

    // R channel (or mono copy)
    if process_mono {
        passthru(native_out_l, native_out_r);
    } else if os_r.is_bypass() {
        if let Some(m) = active_model_r {
            m.process(model_in_r, native_out_r);
        } else {
            passthru(model_in_r, native_out_r);
        }
    } else {
        let n_os = os_r.upsample(model_in_r, os_buf_r);
        if let Some(m) = active_model_r {
            m.process(&os_buf_r[..n_os], &mut os_model_r[..n_os]);
        } else {
            passthru(&os_buf_r[..n_os], &mut os_model_r[..n_os]);
        }
        os_r.downsample(&os_model_r[..n_os], native_out_r);
    }
}

/// Copies `model_in` through to `model_out` when no model is active.
#[inline(always)]
pub(crate) fn passthru(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len().min(out_buf.len());
    unsafe {
        core::ptr::copy_nonoverlapping(in_buf.as_ptr(), out_buf.as_mut_ptr(), n);
    }
}
