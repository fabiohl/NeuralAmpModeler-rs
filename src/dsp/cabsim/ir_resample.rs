// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! IR resampling: polyphase sample-rate conversion for impulse responses.
//!
//! Extracted from `loader.rs` (E-RF4).

use crate::dsp::pipeline::MAX_RESAMP_BUF;
use crate::dsp::resampler::NamResampler;
use crate::dsp::sinc_kernel::TAPS_PER_PHASE;

use std::io;

/// Resamples `input` from `input_rate` to `output_rate` using the polyphase resampler.
///
/// This is a batch/offline operation: feeds the entire IR through `NamResampler`
/// and pads with zeros to flush the filter’s delay line.
pub(crate) fn resample(input: &[f32], input_rate: u32, output_rate: u32) -> io::Result<Vec<f32>> {
    let mut resampler =
        NamResampler::new(input_rate, output_rate, MAX_RESAMP_BUF).map_err(|e| {
            io::Error::other(format!(
                "IR resample failed ({} Hz → {} Hz): {:#}",
                input_rate, output_rate, e
            ))
        })?;

    let max_input_chunk = NamResampler::max_input_samples(MAX_RESAMP_BUF, input_rate, output_rate);

    let est_len =
        ((input.len() as f64 * output_rate as f64 / input_rate as f64).ceil() as usize) + 256;
    let mut output = Vec::with_capacity(est_len);

    let mut in_buf = vec![0.0f32; MAX_RESAMP_BUF];
    let mut out_l = vec![0.0f32; MAX_RESAMP_BUF];
    let mut out_r = vec![0.0f32; MAX_RESAMP_BUF];

    let mut pos = 0usize;
    while pos < input.len() {
        let chunk = (input.len() - pos).min(max_input_chunk);
        in_buf[..chunk].copy_from_slice(&input[pos..pos + chunk]);
        let written = resampler
            .process_input_mono(&in_buf[..chunk], &mut out_l, &mut out_r)
            .samples_written;
        output.extend_from_slice(&out_l[..written]);
        pos += chunk;
    }

    in_buf.fill(0.0);
    let flush_iters = TAPS_PER_PHASE * 2;
    for _ in 0..flush_iters {
        let written = resampler
            .process_input_mono(&in_buf, &mut out_l, &mut out_r)
            .samples_written;
        if written == 0 {
            break;
        }
        output.extend_from_slice(&out_l[..written]);
    }

    if output.is_empty() {
        return Err(io::Error::other("IR resample produced no output samples"));
    }

    Ok(output)
}
