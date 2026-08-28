// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};

/// Passthrough model: identity copy at model rate.
fn passthrough_model(in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) -> usize {
    let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
    out_l[..n].copy_from_slice(&in_l[..n]);
    out_r[..n].copy_from_slice(&in_r[..n]);
    n
}

/// Runs `iterations` one-shot `process` calls with a constant block size and
/// verifies the strict-cardinality contract.
fn check_process_invariants(host: u32, model: u32, n: usize, iterations: usize) {
    let mut stream = StreamingResampleBuffer::new(host, model, 8192).expect("new failed");
    let latency = stream.latency_samples() as u64;

    let in_l = vec![0.5f32; n];
    let in_r = vec![0.25f32; n];
    let mut out_l = vec![f32::NAN; n];
    let mut out_r = vec![f32::NAN; n];

    let mut real_total: u64 = 0;
    let mut padded_total: u64 = 0;

    for _ in 0..iterations {
        let result = stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
        assert_eq!(result.consumed, n, "consumed must be exactly {n}");
        assert_eq!(result.written, n, "written must be exactly {n}");
        assert!(
            !result.underflow,
            "underflow at host={host} model={model} n={n} iter={iterations}"
        );
        assert!(
            out_l[..n].iter().all(|x| x.is_finite()),
            "non-finite output at host={host} model={model} n={n}"
        );
        assert!(
            out_r[..n].iter().all(|x| x.is_finite()),
            "non-finite R output at host={host} model={model} n={n}"
        );
        real_total += result.real as u64;
        padded_total += result.padded as u64;
        assert!(
            stream.output_pending() <= stream.output_capacity_actual(),
            "out_fifo overflow: host={host} model={model} n={n}"
        );
        assert_eq!(
            stream.input_pending(),
            0,
            "input must be fully drained: host={host} model={model} n={n}"
        );
        assert_eq!(
            stream.model_pending(),
            0,
            "model FIFO must be drained: host={host} model={model} n={n}"
        );
    }

    let pushed = stream.input_total();
    assert_eq!(
        real_total,
        pushed.saturating_sub(latency),
        "conservation violated: host={host} model={model} n={n} iter={iterations} \
         real={real_total} pushed={pushed} latency={latency}"
    );
    assert_eq!(
        padded_total, latency,
        "priming must equal the declared latency: host={host} model={model} n={n} \
         padded={padded_total} latency={latency}"
    );
    assert_eq!(
        stream.underflow_total(),
        0,
        "zero underflow expected: host={host} model={model} n={n}"
    );
}

const TEST_RATES: &[u32] = &[44_100, 48_000, 96_000, 192_000];
const IRREGULAR_BLOCKS: &[usize] = &[1, 7, 31, 63, 64, 65, 127, 256];

#[test]
fn test_process_irregular_blocks_all_rates() {
    for &host in TEST_RATES {
        for &model in TEST_RATES {
            for &n in IRREGULAR_BLOCKS {
                check_process_invariants(host, model, n, 64);
            }
        }
    }
}

#[test]
fn test_process_max_block_worst_case() {
    for &host in TEST_RATES {
        for &model in TEST_RATES {
            check_process_invariants(host, model, 8192, 16);
        }
    }
}

#[test]
fn test_process_bypass_exact() {
    let mut stream = StreamingResampleBuffer::new(48_000, 48_000, 256).expect("new failed");
    assert!(stream.is_bypass());
    assert_eq!(stream.latency_samples(), 0);

    let in_l = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let in_r = [5.0f32, 4.0, 3.0, 2.0, 1.0];
    let mut out_l = [0.0f32; 5];
    let mut out_r = [0.0f32; 5];
    let result = stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
    assert_eq!(result.consumed, 5);
    assert_eq!(result.written, 5);
    assert_eq!(result.padded, 0);
    assert_eq!(result.real, 5);
    assert_eq!(out_l, in_l);
    assert_eq!(out_r, in_r);
    assert_eq!(stream.underflow_total(), 0);
}

#[test]
fn test_process_irregular_sequence() {
    // Deterministic pseudo-random sequence of irregular block sizes (LCG).
    let mut lcg: u64 = 0x1234_5678_9abc_def0;
    let mut next = move || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (lcg >> 33) as usize % 8191 + 1
    };

    for &(host, model) in &[(44_100u32, 48_000u32), (96_000, 48_000), (48_000, 44_100)] {
        let mut stream = StreamingResampleBuffer::new(host, model, 8192).expect("new failed");
        let latency = stream.latency_samples() as u64;
        let iterations = 200usize;

        for _ in 0..iterations {
            let n = next().min(8192);
            let in_l = vec![0.5f32; n];
            let in_r = vec![0.25f32; n];
            let mut out_l = vec![f32::NAN; n];
            let mut out_r = vec![f32::NAN; n];
            let result = stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
            assert_eq!(result.consumed, n, "consumed={} n={}", result.consumed, n);
            assert_eq!(result.written, n);
            assert!(!result.underflow, "sequence underflow at {host}->{model}");
            assert!(stream.output_pending() <= stream.output_capacity_actual());
        }

        let pushed = stream.input_total();
        let real = stream.output_real_total();
        assert_eq!(
            real,
            pushed.saturating_sub(latency),
            "sequence conservation violated at {host}->{model}: real={real} pushed={pushed} latency={latency}"
        );
        assert_eq!(stream.underflow_total(), 0);
    }
}

#[test]
fn test_lower_level_api_cardinality() {
    let host = 44_100u32;
    let model = 48_000u32;
    let n = 63usize;
    let mut stream = StreamingResampleBuffer::new(host, model, 256).expect("new failed");
    let latency = stream.latency_samples() as u64;

    let in_l = vec![0.5f32; n];
    let in_r = vec![0.25f32; n];
    let mut mid_l = vec![0.0f32; 4096];
    let mut mid_r = vec![0.0f32; 4096];
    let mut mid_out_l = vec![0.0f32; 4096];
    let mut mid_out_r = vec![0.0f32; 4096];
    let mut out_l = vec![0.0f32; n];
    let mut out_r = vec![0.0f32; n];

    let iterations = 64usize;
    for _ in 0..iterations {
        let accepted = stream.push_input(&in_l, &in_r);
        assert_eq!(accepted, n);

        // Drain + model + feed output until fully drained.
        loop {
            let drained = stream.drain_model_samples(&mut mid_l, &mut mid_r);
            if drained == 0 {
                break;
            }
            let produced = passthrough_model(
                &mid_l[..drained],
                &mid_r[..drained],
                &mut mid_out_l,
                &mut mid_out_r,
            );
            assert_eq!(produced, drained);
            stream.push_model_output(&mid_out_l[..produced], &mid_out_r[..produced]);
        }

        let pull = stream.pull_output(&mut out_l, &mut out_r, n);
        assert_eq!(pull.written, n, "exactly {n} samples must be delivered");
        assert!(!pull.underflow);
        assert!(stream.output_pending() <= stream.output_capacity_actual());
    }

    let pushed = stream.input_total();
    let real = stream.output_real_total();
    assert_eq!(
        real,
        pushed.saturating_sub(latency),
        "lower-level conservation: real={real} pushed={pushed} latency={latency}"
    );
    assert_eq!(stream.underflow_total(), 0);
}

#[test]
fn test_reset_restores_initial_state() {
    let host = 44_100u32;
    let model = 48_000u32;
    let n = 63usize;
    let mut stream = StreamingResampleBuffer::new(host, model, 256).expect("new failed");
    let latency = stream.latency_samples() as u64;

    let in_l = vec![0.5f32; n];
    let in_r = vec![0.25f32; n];
    let mut out_l = vec![0.0f32; n];
    let mut out_r = vec![0.0f32; n];

    for _ in 0..32 {
        stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
    }
    assert!(stream.input_total() > 0);

    stream.reset();
    assert_eq!(stream.input_total(), 0);
    assert_eq!(stream.output_real_total(), 0);
    assert_eq!(stream.output_padded_total(), 0);
    assert_eq!(stream.underflow_total(), 0);
    assert_eq!(stream.input_pending(), 0);
    assert_eq!(stream.output_pending(), 0);
    assert_eq!(stream.model_pending(), 0);

    // After reset the priming budget is re-armed: reprocessing reproduces the
    // same totals as a fresh adapter.
    let mut fresh = StreamingResampleBuffer::new(host, model, 256).expect("new failed");
    for _ in 0..32 {
        fresh.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
        stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
    }
    assert_eq!(stream.input_total(), fresh.input_total());
    assert_eq!(stream.output_real_total(), fresh.output_real_total());
    assert_eq!(stream.output_padded_total(), latency);
    assert_eq!(stream.underflow_total(), 0);
}

#[test]
fn test_linear_phase_variant() {
    let host = 96_000u32;
    let model = 48_000u32;
    let mut stream =
        StreamingResampleBuffer::new_linear(host, model, 1024).expect("new_linear failed");
    assert!(stream.is_linear_phase());
    let latency = stream.latency_samples() as u64;
    assert!(latency > 0);

    let n = 127usize;
    let in_l = vec![0.5f32; n];
    let in_r = vec![0.25f32; n];
    let mut out_l = vec![0.0f32; n];
    let mut out_r = vec![0.0f32; n];

    let iterations = 64usize;
    for _ in 0..iterations {
        let result = stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
        assert_eq!(result.written, n);
        assert!(!result.underflow);
    }
    assert_eq!(
        stream.output_real_total(),
        stream.input_total().saturating_sub(latency)
    );
    assert_eq!(stream.output_padded_total(), latency);
}

#[test]
fn test_mono_through_stereo_path() {
    let host = 44_100u32;
    let model = 48_000u32;
    let n = 256usize;
    let mut stream = StreamingResampleBuffer::new(host, model, 512).expect("new failed");

    let in_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();
    let mut out_l = vec![0.0f32; n];
    let mut out_r = vec![0.0f32; n];

    let iterations = 32usize;
    for _ in 0..iterations {
        let result = stream.process(&in_l, &in_l, &mut out_l, &mut out_r, passthrough_model);
        assert_eq!(result.written, n);
        assert!(!result.underflow);
        for i in 0..n {
            assert_eq!(out_l[i], out_r[i], "mono duplicated to both channels");
        }
    }
}

#[test]
fn test_process_zero_alloc_rt_path() {
    let host = 44_100u32;
    let model = 48_000u32;
    let n = 511usize;
    let mut stream = StreamingResampleBuffer::new(host, model, 1024).expect("new failed");

    let in_l = vec![0.5f32; n];
    let in_r = vec![0.25f32; n];
    let mut out_l = vec![0.0f32; n];
    let mut out_r = vec![0.0f32; n];

    let _guard = TrackingGuard::new();
    for _ in 0..64 {
        stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
    }
    assert_eq!(
        get_alloc_count(),
        0,
        "continuous processing must be zero-alloc"
    );
}

#[test]
fn test_lower_level_api_zero_alloc() {
    let host = 44_100u32;
    let model = 48_000u32;
    let n = 255usize;
    let mut stream = StreamingResampleBuffer::new(host, model, 512).expect("new failed");

    let in_l = vec![0.5f32; n];
    let in_r = vec![0.25f32; n];
    let mut mid_l = vec![0.0f32; 1024];
    let mut mid_r = vec![0.0f32; 1024];
    let mut mid_out_l = vec![0.0f32; 1024];
    let mut mid_out_r = vec![0.0f32; 1024];
    let mut out_l = vec![0.0f32; n];
    let mut out_r = vec![0.0f32; n];

    let _guard = TrackingGuard::new();
    for _ in 0..64 {
        let accepted = stream.push_input(&in_l, &in_r);
        assert_eq!(accepted, n);
        loop {
            let drained = stream.drain_model_samples(&mut mid_l, &mut mid_r);
            if drained == 0 {
                break;
            }
            let produced = passthrough_model(
                &mid_l[..drained],
                &mid_r[..drained],
                &mut mid_out_l,
                &mut mid_out_r,
            );
            stream.push_model_output(&mid_out_l[..produced], &mid_out_r[..produced]);
        }
        let pull = stream.pull_output(&mut out_l, &mut out_r, n);
        assert_eq!(pull.written, n);
    }
    assert_eq!(
        get_alloc_count(),
        0,
        "lower-level processing must be zero-alloc"
    );
}

#[test]
fn test_reset_zero_alloc() {
    let host = 44_100u32;
    let model = 48_000u32;
    let n = 63usize;
    let mut stream = StreamingResampleBuffer::new(host, model, 256).expect("new failed");

    let in_l = vec![0.5f32; n];
    let in_r = vec![0.25f32; n];
    let mut out_l = vec![0.0f32; n];
    let mut out_r = vec![0.0f32; n];

    let _guard = TrackingGuard::new();
    stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
    stream.reset();
    stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
    assert_eq!(get_alloc_count(), 0, "reset must be zero-alloc");
}

#[test]
fn test_constructor_accessors_and_deterministic_capacities() {
    let host = 44_100u32;
    let model = 48_000u32;
    let max_block = 1024usize;
    let stream = StreamingResampleBuffer::new(host, model, max_block).expect("new failed");

    assert_eq!(stream.host_rate(), host);
    assert_eq!(stream.model_rate(), model);
    assert_eq!(stream.max_block(), max_block);
    assert_eq!(stream.input_capacity(), max_block);
    assert_eq!(
        stream.model_capacity(),
        StreamingResampleBuffer::max_model_samples(max_block, host, model)
    );
    assert_eq!(
        stream.output_capacity_actual(),
        StreamingResampleBuffer::output_capacity(max_block, stream.latency_samples())
    );
    assert!(
        stream.model_capacity() >= max_block,
        "model FIFO must hold at least one host block's worth"
    );
}

#[test]
fn test_model_cap_extreme_ratio_rejected() {
    // Extreme host↔model ratio would scale model_cap far beyond the guard.
    let err = StreamingResampleBuffer::new(4_000, 384_000, 8192).err();
    assert!(err.is_some(), "extreme ratio must be rejected");
}

#[test]
fn test_bypass_fast_path_skips_fifo() {
    let n = 4096usize;
    let mut stream = StreamingResampleBuffer::new(48_000, 48_000, 8192).expect("new failed");
    assert!(stream.is_bypass());

    let in_l = vec![0.3f32; n];
    let in_r = vec![0.1f32; n];
    let mut out_l = vec![f32::NAN; n];
    let mut out_r = vec![f32::NAN; n];

    let _guard = TrackingGuard::new();
    for _ in 0..64 {
        let result = stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
        assert_eq!(result.consumed, n);
        assert_eq!(result.written, n);
        assert_eq!(result.real, n);
        assert_eq!(result.padded, 0);
        assert!(!result.underflow);
    }
    assert_eq!(get_alloc_count(), 0, "bypass fast path must be zero-alloc");

    assert_eq!(stream.input_total() as usize, n * 64);
    assert_eq!(stream.output_real_total() as usize, n * 64);
    assert_eq!(stream.underflow_total(), 0);
    assert_eq!(stream.output_pending(), 0);
}

#[test]
fn test_process_over_max_block_is_safe_and_detectable() {
    // A single call with n > max_block violates the block-size contract: the
    // adapter never panics or overflows, but input beyond max_block is dropped
    // and the output deficit is fabricated as zeros — surfaced via `underflow`.
    let max_block = 256usize;
    let n = 512usize;
    let mut stream = StreamingResampleBuffer::new(44_100, 48_000, max_block).expect("new failed");

    let in_l = vec![0.5f32; n];
    let in_r = vec![0.25f32; n];
    let mut out_l = vec![0.0f32; n];
    let mut out_r = vec![0.0f32; n];

    let result = stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
    assert_eq!(result.written, n, "exactly-n output is preserved");
    assert!(result.underflow, "oversized block must be flagged");
    assert_eq!(
        stream.input_total(),
        max_block as u64,
        "only max_block consumed"
    );
    assert!(stream.output_pending() <= stream.output_capacity_actual());
    assert_eq!(stream.input_pending(), 0, "all accepted input drained");
}

#[test]
fn test_waveform_alignment_after_latency() {
    // The adapter must deliver the raw resampler round-trip signal (`ro`)
    // contiguously starting at position `latency`, with the first `latency`
    // samples zeroed (declared latency priming). The warm-up transient is
    // discarded, so the delivered stream is NOT the 2×-delayed `ro` shifted by
    // the priming — it is `ro` itself with leading zeros.
    let host = 44_100u32;
    let model = 48_000u32;
    let n = 512usize;
    let iterations = 64usize;

    let freq = 440.0f32;
    let input: Vec<f32> = (0..n * iterations)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / host as f32).sin())
        .collect();

    let mut stream = StreamingResampleBuffer::new(host, model, 512).expect("new failed");
    let latency = stream.latency_samples() as usize;

    // Reference: raw NamResampler round trip (in → model-rate → host-rate)
    // driven with the exact same block structure.
    let mut raw = NamResampler::new(host, model, n).expect("new failed");
    let mut mid = vec![0.0f32; 4096];
    let mut mid_r = vec![0.0f32; 4096];
    let mut ro = Vec::with_capacity(n * iterations);

    let mut collected = Vec::with_capacity(n * iterations);
    for k in 0..iterations {
        let seg = &input[k * n..(k + 1) * n];
        let mut out_l = vec![0.0f32; n];
        let mut out_r = vec![0.0f32; n];
        stream.process(seg, seg, &mut out_l, &mut out_r, passthrough_model);
        collected.extend_from_slice(&out_l);

        let p = raw.process_input(seg, seg, &mut mid, &mut mid_r);
        let m = p.samples_written;
        let mut out = vec![0.0f32; 4096];
        let mut out_r2 = vec![0.0f32; 4096];
        let po = raw.process_output(&mid[..m], &mid_r[..m], &mut out, &mut out_r2);
        ro.extend_from_slice(&out[..po.samples_written]);
    }

    // 1. Leading `latency` samples are zeros (declared latency priming).
    for (i, &v) in collected[..latency].iter().enumerate() {
        assert_eq!(v, 0.0, "priming sample {i} must be zero");
    }

    // 2. From `latency` onward the delivered stream equals the raw round trip
    //    contiguously (no 2× shift, no mid-stream gap).
    let common = collected.len().min(ro.len() + latency) - latency;
    let rmse: f64 = (0..common)
        .map(|k| {
            let d = collected[latency + k] - ro[latency + k];
            (d * d) as f64
        })
        .sum::<f64>()
        .sqrt()
        / common as f64;
    let rms_ref: f64 = (ro[latency..latency + common]
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        / common as f64)
        .sqrt();
    assert!(
        rmse < rms_ref * 1e-4,
        "delivered stream diverges from raw round trip: rmse={rmse:.3e} ref_rms={rms_ref:.3e}"
    );
}
