// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! # Streaming Partial-Progress Example (`streaming_partial`)
//!
//! Demonstrates the correct handling of [`StreamingResampleBuffer::process`]
//! partial input acceptance and of a callback that makes no progress:
//!
//! 1. A block larger than `max_block()` is offered on purpose; only the
//!    accepted prefix is consumed (`consumed < n`).
//! 2. The unaccepted suffix is presented again in order. Input already accepted
//!    into the FIFO is never re-sent.
//! 3. A zero-progress callback stops the call immediately and is surfaced via
//!    `underflow`; the caller must stop instead of spinning.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example streaming_partial
//! ```

use neural_amp_modeler_rs::dsp::resampling::StreamingResampleBuffer;

/// Host sample rate (e.g. a 44.1 kHz DAW session).
const HOST_RATE: u32 = 44_100;
/// Model sample rate (e.g. a 48 kHz NAM model).
const MODEL_RATE: u32 = 48_000;
/// Maximum host block size the adapter accepts per call.
const MAX_BLOCK: usize = 64;

/// Identity model at model rate: writes every input sample to the output.
fn passthrough(in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) -> usize {
    let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
    out_l[..n].copy_from_slice(&in_l[..n]);
    out_r[..n].copy_from_slice(&in_r[..n]);
    n
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = StreamingResampleBuffer::new(HOST_RATE, MODEL_RATE, MAX_BLOCK)?;
    println!("host={HOST_RATE} Hz model={MODEL_RATE} Hz max_block={MAX_BLOCK}");
    println!(
        "declared latency = {} host samples",
        stream.latency_samples()
    );

    // A deterministic resampler warm-up (priming) is part of the first output.
    let mut warmup_l = vec![0.0f32; MAX_BLOCK];
    let mut warmup_r = vec![0.0f32; MAX_BLOCK];
    let warm = stream.process(
        &[0.0f32; MAX_BLOCK],
        &[0.0f32; MAX_BLOCK],
        &mut warmup_l,
        &mut warmup_r,
        passthrough,
    );
    println!(
        "warm-up: consumed={} written={} real={} padded={}",
        warm.consumed, warm.written, warm.real, warm.padded
    );

    // A block deliberately twice as large as `max_block`.
    let input: Vec<f32> = (0..(MAX_BLOCK * 2)).map(|i| i as f32 * 0.001).collect();
    let mut out_l = vec![0.0f32; input.len()];
    let mut out_r = vec![0.0f32; input.len()];

    // Phase 1: offer the oversized block. `written` is always the full block
    // length, but `consumed` is bounded by the input FIFO capacity.
    let first = stream.process(&input, &input, &mut out_l, &mut out_r, passthrough);
    println!(
        "oversized block: offered={} consumed={} written={} real={} padded={} underflow={}",
        input.len(),
        first.consumed,
        first.written,
        first.real,
        first.padded,
        first.underflow
    );
    assert_eq!(first.written, input.len());
    assert!(first.consumed < input.len());
    assert!(first.consumed <= first.written);

    let mut accepted = first.consumed;

    // Phase 2: only the unaccepted suffix is presented again, in order.
    while accepted < input.len() {
        let suffix = &input[accepted..];
        let n = suffix.len();
        let mut suffix_out_l = vec![0.0f32; n];
        let mut suffix_out_r = vec![0.0f32; n];
        let result = stream.process(
            suffix,
            suffix,
            &mut suffix_out_l,
            &mut suffix_out_r,
            passthrough,
        );
        println!(
            "suffix: offered={} consumed={} written={} real={} padded={}",
            n, result.consumed, result.written, result.real, result.padded
        );
        if result.consumed == 0 {
            // Zero progress: stop and keep the suffix for the caller. The
            // accepted prefix is never re-sent.
            println!("zero progress: stopping with {n} input samples still pending");
            break;
        }
        accepted += result.consumed;
    }
    println!("accepted={} of {} input samples", accepted, input.len());

    // Flush retained output in fixed-size pulls.
    let mut chunk_l = [0.0f32; 16];
    let mut chunk_r = [0.0f32; 16];
    let mut flushed = 0usize;
    while stream.output_pending() > 0 || stream.model_pending() > 0 {
        let pull = stream.pull_output(&mut chunk_l, &mut chunk_r, 16);
        flushed += pull.written;
        if pull.written == 0 {
            break;
        }
    }
    println!(
        "flushed={flushed} underflow_total={}",
        stream.underflow_total()
    );

    // Phase 3: a callback that never produces output. The adapter stops on the
    // first zero-progress call and flags underflow instead of looping.
    let mut stalled_l = [0.0f32; 8];
    let mut stalled_r = [0.0f32; 8];
    let stalled = stream.process(
        &[0.25f32; 8],
        &[0.25f32; 8],
        &mut stalled_l,
        &mut stalled_r,
        |_in_l, _in_r, _out_l, _out_r| 0,
    );
    println!(
        "zero-progress callback: consumed={} written={} real={} padded={} underflow={}",
        stalled.consumed, stalled.written, stalled.real, stalled.padded, stalled.underflow
    );
    assert!(stalled.underflow);
    assert_eq!(stalled.written, 8);

    println!(
        "done: input_total={} real_total={} padded_total={}",
        stream.input_total(),
        stream.output_real_total(),
        stream.output_padded_total()
    );
    Ok(())
}
