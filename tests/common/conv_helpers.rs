// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared convolution helpers for integration tests.
//!
//! Provides direct convolution reference and block-by-block UPOLS processing
//! used by golden tests and parity validation across test files.

use neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine;

/// Naive direct convolution: y[n] = Σ_m h[m] * x[n-m].
///
/// Used as the ground-truth reference for ESR validation.
pub fn direct_convolve(ir: &[f32], input: &[f32]) -> Vec<f32> {
    let out_len = input.len() + ir.len() - 1;
    let mut output = vec![0.0f32; out_len];
    for (n, out) in output.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (m, &ir_val) in ir.iter().enumerate() {
            let x_idx = n as isize - m as isize;
            if x_idx >= 0 && x_idx < input.len() as isize {
                acc += ir_val * input[x_idx as usize];
            }
        }
        *out = acc;
    }
    output
}

/// Feeds a full signal through the UPOLS engine block-by-block and collects output.
///
/// Includes flush blocks to drain the FDL of remaining energy after the signal ends.
pub fn process_full_signal(engine: &mut ConvEngine, signal: &[f32]) -> Vec<f32> {
    let b = engine.partition_size();
    let mut output = Vec::with_capacity(signal.len());
    let mut buf_in = vec![0.0f32; b];
    let mut buf_out = vec![0.0f32; b];

    let mut pos = 0;
    while pos < signal.len() {
        let chunk = (signal.len() - pos).min(b);
        buf_in[..chunk].copy_from_slice(&signal[pos..pos + chunk]);
        if chunk < b {
            buf_in[chunk..].fill(0.0);
        }
        engine.process(&buf_in, &mut buf_out);
        output.extend_from_slice(&buf_out[..chunk.min(b)]);
        pos += chunk;
    }

    let flush_blocks = engine.num_partitions();
    for _ in 0..flush_blocks {
        buf_in.fill(0.0);
        engine.process(&buf_in, &mut buf_out);
        output.extend_from_slice(&buf_out[..b]);
    }

    output
}
