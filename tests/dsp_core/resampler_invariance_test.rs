// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Resampler Block Invariance Test.
//!
//! Validates that a polyphase FIR sinc resampler produces identical output
//! regardless of how the input signal is fragmented into blocks.
//!
//! Rationale: the resampler is a time-invariant, deterministic FIR filter with
//! stateful phase accumulator.  As long as sample values are fed in temporal
//! order, the block boundaries must not alter the output — this is a
//! mathematical invariant of the polyphase architecture.

use neural_amp_modeler_rs::dsp::resampler::NamResampler;

const SIGNAL_LEN: usize = 100_000;
const FIXED_BLOCK: usize = 1024;
const FRAG_MIN: usize = 7;
const FRAG_MAX: usize = 513;
const TAPS_PER_PHASE: usize = 64;

// ── PRNG (self-contained) ──────────────────────────────────────────────────

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((old >> 18) ^ old) >> 27
    }

    fn next_usize_bounded(&mut self, lo: usize, hi: usize) -> usize {
        let range = hi - lo + 1;
        lo + (self.next() as usize % range)
    }
}

// ── Signal generation ──────────────────────────────────────────────────────

fn generate_signal(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..len)
        .map(|_| {
            let u = rng.next() as f32 / u64::MAX as f32;
            u * 2.0 - 1.0
        })
        .collect()
}

// ── Random block size partitioning ─────────────────────────────────────────

fn random_block_sizes(total: usize, seed: u64) -> Vec<usize> {
    let mut rng = Lcg::new(seed);
    let mut remaining = total;
    let mut sizes = Vec::new();
    while remaining > 0 {
        let max_allowed = remaining.min(FRAG_MAX);
        let size = if max_allowed <= FRAG_MIN {
            remaining
        } else {
            rng.next_usize_bounded(FRAG_MIN, max_allowed)
        };
        sizes.push(size);
        remaining -= size;
    }
    sizes
}

// ── Processing helpers ─────────────────────────────────────────────────────

fn process_monolithic(rs: &mut NamResampler, signal: &[f32], in_sr: u32, out_sr: u32) -> Vec<f32> {
    let out_cap = NamResampler::min_output_samples(FIXED_BLOCK, in_sr, out_sr) + 1;
    let mut out_l = vec![0.0f32; out_cap];
    let mut out_r = vec![0.0f32; out_cap];
    let mut output = Vec::with_capacity(NamResampler::min_output_samples(
        signal.len(),
        in_sr,
        out_sr,
    ));
    let mut total_read: usize = 0;

    for chunk in signal.chunks(FIXED_BLOCK) {
        let in_r = vec![0.0f32; chunk.len()];
        let progress = rs.process_input(chunk, &in_r, &mut out_l, &mut out_r);
        output.extend_from_slice(&out_l[..progress.samples_written]);
        total_read += progress.samples_read;
    }

    assert_eq!(
        total_read,
        signal.len(),
        "monolithic: consumed {} instead of {} — block truncated",
        total_read,
        signal.len()
    );
    output
}

fn process_fragmented(rs: &mut NamResampler, signal: &[f32], in_sr: u32, out_sr: u32) -> Vec<f32> {
    let max_block = FRAG_MAX;
    let out_cap = NamResampler::min_output_samples(max_block, in_sr, out_sr) + 1;
    let mut out_l = vec![0.0f32; out_cap];
    let mut out_r = vec![0.0f32; out_cap];
    let mut output = Vec::with_capacity(
        NamResampler::min_output_samples(signal.len(), in_sr, out_sr) + TAPS_PER_PHASE,
    );

    let sizes = random_block_sizes(signal.len(), 0xDEAD_BEEF_CAFE_BABE);
    let mut offset = 0;
    let mut total_consumed: usize = 0;
    for &size in &sizes {
        let end = (offset + size).min(signal.len());
        let chunk = &signal[offset..end];
        let in_r = vec![0.0f32; chunk.len()];
        let progress = rs.process_input(chunk, &in_r, &mut out_l, &mut out_r);
        output.extend_from_slice(&out_l[..progress.samples_written]);
        offset += progress.samples_read;
        total_consumed += progress.samples_read;
    }
    assert_eq!(
        total_consumed,
        signal.len(),
        "fragmented mode consumed {} samples but signal has {} — truncated block detected",
        total_consumed,
        signal.len()
    );

    output
}

// ── Test ───────────────────────────────────────────────────────────────────

#[test]
fn test_resampler_block_invariance() {
    let signal = generate_signal(SIGNAL_LEN, 12345);

    let scenarios: &[(u32, u32, &str)] = &[
        (44100, 48000, "44.1k→48k"),
        (48000, 44100, "48k→44.1k"),
        (96000, 48000, "96k→48k"),
    ];

    for &(in_sr, out_sr, label) in scenarios {
        let mut rs_a = NamResampler::new(in_sr, out_sr, 64).unwrap();
        let out_a = process_monolithic(&mut rs_a, &signal, in_sr, out_sr);

        let mut rs_b = NamResampler::new(in_sr, out_sr, 64).unwrap();
        let out_b = process_fragmented(&mut rs_b, &signal, in_sr, out_sr);

        // ── Count identity ──────────────────────────────────────────────
        assert_eq!(
            out_a.len(),
            out_b.len(),
            "{}: output count mismatch — monolithic={}, fragmented={}",
            label,
            out_a.len(),
            out_b.len()
        );

        // ── Sample-level comparison ──────────────────────────────────────
        let first_divergence = out_a
            .iter()
            .zip(out_b.iter())
            .position(|(a, b)| (a - b).abs() > f32::EPSILON);

        let max_diff = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("--- Resampler Block Invariance ({label}) ---");
        println!("Signal length:             {}", SIGNAL_LEN);
        println!("Output samples:            {} (mono)", out_a.len());
        println!("Max amplitude diff:        {:.6e}", max_diff);
        if let Some(pos) = first_divergence {
            let ctx = pos.saturating_sub(3);
            let end = (pos + 4).min(out_a.len());
            println!("First divergence at index {pos}:");
            for i in ctx..end {
                let da = out_a[i];
                let db = out_b[i];
                println!(
                    "  [{i}] A={da:.8e} B={db:.8e} diff={:.2e} {marker}",
                    (da - db).abs(),
                    marker = if (da - db).abs() > f32::EPSILON {
                        "***"
                    } else {
                        ""
                    }
                );
            }
        }
        println!(
            "Verdict:             {}",
            if max_diff < 1e-6 { "PASS" } else { "FAIL" }
        );

        assert!(
            max_diff < 1e-6,
            "{}: amplitude divergence — max_diff={:.6e} exceeds 1e-6 threshold",
            label,
            max_diff
        );

        assert!(
            max_diff <= f32::EPSILON,
            "{}: non-bit-identical output — max_diff={:.6e} > f32::EPSILON ({:.6e})",
            label,
            max_diff,
            f32::EPSILON
        );
    }
}
