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
#[ignore = "long: sweep / soak de reamostragem multirate"]
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
#[ignore = "long: sweep / soak de reamostragem multirate"]
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
#[ignore = "long: sweep / soak de reamostragem multirate"]
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
#[ignore = "long: sweep / soak de reamostragem multirate"]
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
    assert_eq!(
        StreamingResampleBuffer::new(4_000, 384_000, 8192).err(),
        Some(NamErrorCode::OutOfMemory),
        "extreme ratio must be rejected with OutOfMemory (min-phase)"
    );
    assert_eq!(
        StreamingResampleBuffer::new_linear(4_000, 384_000, 8192).err(),
        Some(NamErrorCode::OutOfMemory),
        "extreme ratio must be rejected with OutOfMemory (linear-phase)"
    );
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
fn test_process_exact_max_block_cadence_valid_input() {
    let max_block = 256usize;
    let iterations = 32usize;
    let mut stream = StreamingResampleBuffer::new(44_100, 48_000, max_block).expect("new failed");
    let latency = stream.latency_samples() as u64;

    let in_l = vec![0.5f32; max_block];
    let in_r = vec![0.25f32; max_block];
    let mut out_l = vec![f32::NAN; max_block];
    let mut out_r = vec![f32::NAN; max_block];

    let mut real_total: u64 = 0;
    let mut padded_total: u64 = 0;

    for _ in 0..iterations {
        let result = stream.process(&in_l, &in_r, &mut out_l, &mut out_r, passthrough_model);
        assert_eq!(result.consumed, max_block);
        assert_eq!(result.written, max_block);
        assert_eq!(result.real + result.padded, max_block);
        assert!(!result.underflow);
        assert!(out_l.iter().all(|x| x.is_finite()));
        assert!(out_r.iter().all(|x| x.is_finite()));
        assert_eq!(stream.input_pending(), 0);
        real_total += result.real as u64;
        padded_total += result.padded as u64;
    }

    let pushed = stream.input_total();
    assert_eq!(pushed, (iterations * max_block) as u64);
    assert_eq!(real_total, pushed - latency);
    assert_eq!(padded_total, latency);
    assert_eq!(stream.underflow_total(), 0);
}

#[test]
#[ignore = "long: sweep / soak de reamostragem multirate"]
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
    let mut raw = NamResampler::new_simple(host, model).expect("new failed");
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

#[test]
fn test_push_model_output_backpressure_and_saturation_no_loss() {
    // Regression for A2 (S0.2 / S2.1): with max_block = 1 at 48k -> 48k,
    // model_cap = 3 and out_cap = 5. Two pushes of 3 samples without an
    // intermediate pull must preserve all 6 samples without panic or drop.
    let mut stream = StreamingResampleBuffer::new(48_000, 48_000, 1).expect("new failed");
    assert_eq!(stream.model_capacity(), 3);
    assert_eq!(stream.output_capacity_actual(), 5);

    let a1 = stream.push_model_output(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]);
    assert_eq!(a1, 3);
    assert_eq!(stream.output_pending(), 3);
    assert_eq!(stream.model_pending(), 0);

    // Second push fills out_fifo (5) and leaves 1 sample buffered in model_fifo.
    let a2 = stream.push_model_output(&[4.0, 5.0, 6.0], &[40.0, 50.0, 60.0]);
    assert_eq!(a2, 3);
    assert_eq!(stream.output_pending(), 5);
    assert_eq!(stream.model_pending(), 1);

    // Third push encounters saturation: only 2 slots remain in model_fifo.
    let a3 = stream.push_model_output(&[7.0, 8.0, 9.0], &[70.0, 80.0, 90.0]);
    assert_eq!(a3, 2);
    assert_eq!(stream.output_pending(), 5);
    assert_eq!(stream.model_pending(), 3);

    // Fourth push when completely saturated accepts 0 and discards nothing.
    let a4 = stream.push_model_output(&[9.0], &[90.0]);
    assert_eq!(a4, 0);

    // Pulling 5 samples resumes draining model_fifo -> out_fifo automatically.
    let mut out_l = [0.0f32; 8];
    let mut out_r = [0.0f32; 8];
    let p1 = stream.pull_output(&mut out_l[..5], &mut out_r[..5], 5);
    assert_eq!(p1.written, 5);
    assert_eq!(p1.real, 5);
    assert_eq!(p1.padded, 0);
    assert!(!p1.underflow);
    assert_eq!(stream.model_pending(), 0);
    assert_eq!(stream.output_pending(), 3);

    let p2 = stream.pull_output(&mut out_l[5..], &mut out_r[5..], 3);
    assert_eq!(p2.written, 3);
    assert_eq!(p2.real, 3);
    assert!(!p2.underflow);
    assert_eq!(out_l, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(out_r, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    assert_eq!(stream.output_real_total(), 8);
    assert_eq!(stream.underflow_total(), 0);

    // Pulling across out_fifo + model_fifo in one shot delivers all pending
    // without artificial underflow.
    assert_eq!(
        stream.push_model_output(
            &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            &[21.0, 22.0, 23.0, 24.0, 25.0, 26.0]
        ),
        6
    );
    let mut bulk_l = [0.0f32; 6];
    let mut bulk_r = [0.0f32; 6];
    let p3 = stream.pull_output(&mut bulk_l, &mut bulk_r, 6);
    assert_eq!(p3.real, 6);
    assert!(!p3.underflow);
    assert_eq!(bulk_l, [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    assert_eq!(bulk_r, [21.0, 22.0, 23.0, 24.0, 25.0, 26.0]);
}

#[test]
fn test_bypass_process_respects_pending_granular_fifos_and_order() {
    // Regression for A2 (S0.2 / S2.1): push_input(A) followed by process(B)
    // in bypass mode must deliver A before B and preserve exact conservation.
    let mut stream = StreamingResampleBuffer::new(48_000, 48_000, 4).expect("new failed");
    assert!(stream.is_bypass());

    let acc = stream.push_input(&[1.0, 2.0], &[11.0, 12.0]);
    assert_eq!(acc, 2);
    assert_eq!(stream.input_pending(), 2);

    let mut out_l = [0.0f32; 2];
    let mut out_r = [0.0f32; 2];
    let res = stream.process(
        &[3.0, 4.0],
        &[13.0, 14.0],
        &mut out_l,
        &mut out_r,
        passthrough_model,
    );
    assert_eq!(res.consumed, 2);
    assert_eq!(res.written, 2);
    assert_eq!(res.real, 2);
    assert_eq!(res.padded, 0);
    assert!(!res.underflow);
    assert_eq!(out_l, [1.0, 2.0]);
    assert_eq!(out_r, [11.0, 12.0]);
    assert_eq!(stream.input_pending(), 0);
    assert_eq!(stream.model_pending(), 0);
    assert_eq!(stream.output_pending(), 2);
    assert_eq!(stream.input_total(), 4);
    assert_eq!(stream.output_real_total(), 2);

    // Next process still sees out_fifo pending, so [3.0, 4.0] comes out first.
    let res2 = stream.process(
        &[5.0, 6.0],
        &[15.0, 16.0],
        &mut out_l,
        &mut out_r,
        passthrough_model,
    );
    assert_eq!(res2.consumed, 2);
    assert_eq!(res2.real, 2);
    assert!(!res2.underflow);
    assert_eq!(out_l, [3.0, 4.0]);
    assert_eq!(out_r, [13.0, 14.0]);

    // Drain remaining buffered samples via pull_output; once empty, bypass fast
    // path resumes seamlessly.
    let p = stream.pull_output(&mut out_l, &mut out_r, 2);
    assert_eq!(p.real, 2);
    assert_eq!(out_l, [5.0, 6.0]);
    assert_eq!(out_r, [15.0, 16.0]);
    assert_eq!(stream.output_pending(), 0);

    let res3 = stream.process(
        &[7.0, 8.0],
        &[17.0, 18.0],
        &mut out_l,
        &mut out_r,
        passthrough_model,
    );
    assert_eq!(res3.consumed, 2);
    assert_eq!(res3.real, 2);
    assert_eq!(out_l, [7.0, 8.0]);
    assert_eq!(out_r, [17.0, 18.0]);
    assert_eq!(stream.input_total(), 8);
    assert_eq!(stream.output_real_total(), 8);
    assert_eq!(stream.underflow_total(), 0);
}

#[test]
fn test_bypass_process_when_model_and_out_fifos_saturated() {
    let mut stream = StreamingResampleBuffer::new(48_000, 48_000, 4).expect("new failed");
    assert_eq!(stream.model_capacity(), 6);
    assert_eq!(stream.output_capacity_actual(), 8);

    // Saturate out_fifo (8) and model_fifo (6) completely with distinct L/R.
    let out_seed_l = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let out_seed_r = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
    assert_eq!(stream.push_model_output(&out_seed_l, &out_seed_r), 8);
    let mod_seed_l = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
    let mod_seed_r = [-9.0, -10.0, -11.0, -12.0, -13.0, -14.0];
    assert_eq!(stream.push_model_output(&mod_seed_l, &mod_seed_r), 6);
    assert_eq!(stream.output_pending(), 8);
    assert_eq!(stream.model_pending(), 6);

    // process() pulls 4 samples from out_fifo first, freeing space so
    // model_fifo drains and the new block [15..18] is processed without loss.
    let mut out_l = [0.0f32; 4];
    let mut out_r = [0.0f32; 4];
    let res = stream.process(
        &[15.0, 16.0, 17.0, 18.0],
        &[-15.0, -16.0, -17.0, -18.0],
        &mut out_l,
        &mut out_r,
        passthrough_model,
    );
    assert_eq!(res.consumed, 4);
    assert_eq!(res.written, 4);
    assert_eq!(res.real, 4);
    assert!(!res.underflow);
    assert_eq!(out_l, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(out_r, [-1.0, -2.0, -3.0, -4.0]);

    // Pull remaining 14 samples sequentially (partial pulls + wrap-around).
    let mut collected_l = Vec::new();
    let mut collected_r = Vec::new();
    for &chunk in &[3usize, 5, 6] {
        let mut buf_l = vec![0.0f32; chunk];
        let mut buf_r = vec![0.0f32; chunk];
        let p = stream.pull_output(&mut buf_l, &mut buf_r, chunk);
        assert_eq!(p.real, chunk);
        assert!(!p.underflow);
        collected_l.extend_from_slice(&buf_l);
        collected_r.extend_from_slice(&buf_r);
    }
    let expected_l: Vec<f32> = (5..=18).map(|i| i as f32).collect();
    let expected_r: Vec<f32> = (5..=18).map(|i| -(i as f32)).collect();
    assert_eq!(collected_l, expected_l);
    assert_eq!(collected_r, expected_r);
    assert_eq!(stream.underflow_total(), 0);
}

#[test]
fn test_multirate_backpressure_asymmetric_and_distinct_lr() {
    // Exercise non-bypass multirate pairs with asymmetric producer pushes,
    // partial pulls, priming discard under backpressure, and distinct L/R.
    for &(host, model) in &[(44_100u32, 48_000u32), (48_000, 96_000), (48_000, 192_000)] {
        let max_block = 16usize;
        let mut stream = StreamingResampleBuffer::new(host, model, max_block).expect("new failed");
        let latency = stream.latency_samples() as u64;
        assert!(latency > 0);

        let mut mid_l = vec![0.0f32; stream.model_capacity() + 16];
        let mut mid_r = vec![0.0f32; stream.model_capacity() + 16];
        let mut real_pulled = 0u64;
        let mut padded_pulled = 0u64;

        // Asymmetric: push 3 blocks of host input before pulling at model/host,
        // and saturate push_model_output cleanly without losing accepted items.
        let mut pushed_model_accepted = 0usize;
        let mut pushed_model_attempted = 0usize;
        for blk in 0..24usize {
            let in_l: Vec<f32> = (0..8)
                .map(|i| ((blk * 8 + i) as f32 * 0.03).sin())
                .collect();
            let in_r: Vec<f32> = (0..8)
                .map(|i| -0.5 * ((blk * 8 + i) as f32 * 0.07).cos())
                .collect();
            let acc_in = stream.push_input(&in_l, &in_r);
            assert_eq!(acc_in, 8);

            let drained = stream.drain_model_samples(&mut mid_l, &mut mid_r);
            if drained > 0 {
                pushed_model_attempted += drained;
                let mut offset = 0usize;
                while offset < drained {
                    let acc =
                        stream.push_model_output(&mid_l[offset..drained], &mid_r[offset..drained]);
                    if acc == 0 {
                        // Backpressure active: pull a small chunk to relieve out_fifo.
                        let mut tmp_l = [0.0f32; 5];
                        let mut tmp_r = [0.0f32; 5];
                        let p = stream.pull_output(&mut tmp_l, &mut tmp_r, 5);
                        assert_eq!(p.written, 5);
                        real_pulled += p.real as u64;
                        padded_pulled += p.padded as u64;
                        assert!(!p.underflow);
                    } else {
                        offset += acc;
                        pushed_model_accepted += acc;
                    }
                }
            }
            assert!(stream.output_pending() <= stream.output_capacity_actual());
            assert!(stream.model_pending() <= stream.model_capacity());

            // Every 3rd block, do an irregular partial pull.
            if blk % 3 == 2 {
                let mut out_l = [0.0f32; 7];
                let mut out_r = [0.0f32; 7];
                let p = stream.pull_output(&mut out_l, &mut out_r, 7);
                assert_eq!(p.written, 7);
                assert!(!p.underflow);
                real_pulled += p.real as u64;
                padded_pulled += p.padded as u64;
            }
        }
        assert_eq!(pushed_model_accepted, pushed_model_attempted);

        // Drain whatever remains in model_fifo + out_fifo via small pulls of 1.
        let mut one_l = [0.0f32; 1];
        let mut one_r = [0.0f32; 1];
        loop {
            if stream.output_pending() == 0 && stream.model_pending() == 0 {
                break;
            }
            let p = stream.pull_output(&mut one_l, &mut one_r, 1);
            assert_eq!(p.written, 1);
            assert!(!p.underflow);
            real_pulled += p.real as u64;
            padded_pulled += p.padded as u64;
        }
        assert_eq!(stream.output_real_total(), real_pulled);
        assert_eq!(stream.output_padded_total(), padded_pulled);
        assert_eq!(stream.underflow_total(), 0);
        // Once all buffered outputs are drained, priming matches declared latency.
        assert_eq!(padded_pulled, latency);
    }
}

#[test]
fn test_process_model_underproduction_and_zero_progress_contract() {
    // Verify explicit contract when callback returns `produced < drained`
    // (including `produced == 0`) in both non-bypass and bypass modes:
    // - never fabricates non-zero model-rate samples
    // - flags `underflow: true` immediately on the underproducing call
    // - never enters an infinite loop on `produced == 0`
    // - preserves remaining `in_fifo` samples when stalling mid-drain.
    let mut stream = StreamingResampleBuffer::new(44_100, 48_000, 64).expect("new failed");
    let in_l = [0.5f32; 32];
    let in_r = [-0.5f32; 32];
    let mut out_l = [f32::NAN; 32];
    let mut out_r = [f32::NAN; 32];

    // 1. Partial model output (half of drained) during priming: must STILL
    //    flag `underflow: true` on StreamingResult immediately.
    let res_partial = stream.process(&in_l, &in_r, &mut out_l, &mut out_r, |ml, _mr, ol, or| {
        let k = ml.len() / 2;
        ol[..k].fill(0.25);
        or[..k].fill(-0.25);
        k
    });
    assert_eq!(res_partial.consumed, 32);
    assert_eq!(res_partial.written, 32);
    assert!(res_partial.underflow);
    assert!(out_l.iter().all(|x| x.is_finite()));
    assert!(out_r.iter().all(|x| x.is_finite()));

    // 2. Zero progress (`produced == 0`): terminates immediately, flags underflow.
    let res_zero = stream.process(&in_l, &in_r, &mut out_l, &mut out_r, |_ml, _mr, _ol, _or| 0);
    assert_eq!(res_zero.consumed, 32);
    assert_eq!(res_zero.written, 32);
    assert!(res_zero.underflow);

    // 3. Over-reporting callback (`produced > ml.len()`): clamped safely to ml.len().
    stream.reset();
    let res_over = stream.process(&in_l, &in_r, &mut out_l, &mut out_r, |ml, mr, ol, or| {
        let n = ml.len();
        ol[..n].copy_from_slice(ml);
        or[..n].copy_from_slice(mr);
        n + 999
    });
    assert_eq!(res_over.consumed, 32);
    assert_eq!(res_over.written, 32);
    assert!(!res_over.underflow);

    // 4. Bypass mode underproduction: reports exact real vs padded/underflow.
    let mut bypass = StreamingResampleBuffer::new(48_000, 48_000, 16).expect("new failed");
    let mut b_out_l = [f32::NAN; 8];
    let mut b_out_r = [f32::NAN; 8];
    let b_res = bypass.process(
        &[1.0; 8],
        &[2.0; 8],
        &mut b_out_l,
        &mut b_out_r,
        |ml, mr, ol, or| {
            ol[..3].copy_from_slice(&ml[..3]);
            or[..3].copy_from_slice(&mr[..3]);
            3
        },
    );
    assert_eq!(b_res.consumed, 8);
    assert_eq!(b_res.written, 8);
    assert_eq!(b_res.real, 3);
    assert_eq!(b_res.padded, 5);
    assert!(b_res.underflow);
    assert_eq!(b_out_l, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(b_out_r, [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(bypass.underflow_total(), 5);
}

#[test]
fn test_backpressure_and_mixed_paths_zero_alloc() {
    let mut stream = StreamingResampleBuffer::new(48_000, 48_000, 4).expect("new failed");
    let mut out_l = [0.0f32; 8];
    let mut out_r = [0.0f32; 8];

    let _guard = TrackingGuard::new();
    let _ = stream.push_model_output(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
    let _ = stream.push_model_output(&[5.0, 6.0, 7.0, 8.0], &[5.0, 6.0, 7.0, 8.0]);
    let _ = stream.push_model_output(&[9.0, 10.0, 11.0, 12.0], &[9.0, 10.0, 11.0, 12.0]);
    let _ = stream.push_input(&[13.0, 14.0], &[13.0, 14.0]);
    let _ = stream.process(
        &[15.0, 16.0],
        &[15.0, 16.0],
        &mut out_l[..4],
        &mut out_r[..4],
        passthrough_model,
    );
    let _ = stream.pull_output(&mut out_l, &mut out_r, 8);
    stream.reset();
    assert_eq!(
        get_alloc_count(),
        0,
        "backpressure, resume, mixed bypass, and reset must be zero-alloc"
    );
}

#[test]
fn test_reset_after_saturation_and_underflow_behavior() {
    let mut stream = StreamingResampleBuffer::new(48_000, 48_000, 2).expect("new failed");
    assert_eq!(
        stream.push_model_output(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1.0; 6]),
        6
    );
    assert_eq!(stream.push_input(&[7.0, 8.0], &[7.0, 8.0]), 2);
    assert!(stream.model_pending() > 0 || stream.output_pending() > 0);
    assert_eq!(stream.input_pending(), 2);

    stream.reset();
    assert_eq!(stream.input_pending(), 0);
    assert_eq!(stream.model_pending(), 0);
    assert_eq!(stream.output_pending(), 0);
    assert_eq!(stream.input_total(), 0);
    assert_eq!(stream.output_real_total(), 0);
    assert_eq!(stream.output_padded_total(), 0);
    assert_eq!(stream.underflow_total(), 0);

    // Pull when empty must deliver exact zeros and track underflow_total.
    let mut out_l = [f32::NAN; 4];
    let mut out_r = [f32::NAN; 4];
    let p = stream.pull_output(&mut out_l, &mut out_r, 4);
    assert_eq!(p.written, 4);
    assert_eq!(p.real, 0);
    assert_eq!(p.padded, 4);
    assert!(p.underflow);
    assert_eq!(out_l, [0.0; 4]);
    assert_eq!(out_r, [0.0; 4]);
    assert_eq!(stream.underflow_total(), 4);
}

#[test]
fn test_warmup_discard_under_full_out_fifo_and_partial_push_input() {
    // Verify warm-up discard advances cleanly even when `out_fifo` fills up
    // before any `pull_output`, and `push_input` enforces bounded L/R parity.
    let mut stream = StreamingResampleBuffer::new(44_100, 48_000, 16).expect("new failed");
    let latency = stream.latency_samples() as u64;
    assert!(latency > 0);

    // Partial push_input when input exceeds capacity or L/R lengths differ.
    let big_l = vec![0.5f32; 32];
    let big_r = vec![-0.25f32; 20];
    let acc1 = stream.push_input(&big_l, &big_r);
    assert_eq!(acc1, 16);
    assert_eq!(stream.input_pending(), 16);
    assert_eq!(stream.push_input(&[1.0], &[1.0]), 0);

    // Non-bypass push_input(A) followed by process(B) drains A then B in order.
    let mut out_l = [0.0f32; 16];
    let mut out_r = [0.0f32; 16];
    let b_l = [0.25f32; 16];
    let b_r = [-0.5f32; 16];
    let res = stream.process(&b_l, &b_r, &mut out_l, &mut out_r, passthrough_model);
    assert_eq!(res.consumed, 16);
    assert_eq!(res.written, 16);
    assert!(!res.underflow);
    assert_eq!(stream.input_pending(), 0);
    assert_eq!(stream.input_total(), 32);

    // Pull remaining samples from A+B in small chunks and verify exact
    // conservation against raw NamResampler output (allowing only integer
    // phase rounding, zero underflow, zero lost samples).
    let mut one_l = [0.0f32; 1];
    let mut one_r = [0.0f32; 1];
    let mut drained_l = Vec::new();
    let mut drained_r = Vec::new();
    while stream.output_pending() > 0 || stream.model_pending() > 0 {
        let p = stream.pull_output(&mut one_l, &mut one_r, 1);
        assert!(!p.underflow);
        drained_l.push(one_l[0]);
        drained_r.push(one_r[0]);
    }

    let mut raw = NamResampler::new_simple(44_100, 48_000).expect("raw failed");
    let mut mid_l = vec![0.0f32; 64];
    let mut mid_r = vec![0.0f32; 64];
    let p_in1 = raw.process_input(&big_l[..16], &big_r[..16], &mut mid_l, &mut mid_r);
    let mut raw_out_l = vec![0.0f32; 64];
    let mut raw_out_r = vec![0.0f32; 64];
    let p_out1 = raw.process_output(
        &mid_l[..p_in1.samples_written],
        &mid_r[..p_in1.samples_written],
        &mut raw_out_l,
        &mut raw_out_r,
    );
    let p_in2 = raw.process_input(&b_l, &b_r, &mut mid_l, &mut mid_r);
    let p_out2 = raw.process_output(
        &mid_l[..p_in2.samples_written],
        &mid_r[..p_in2.samples_written],
        &mut raw_out_l[p_out1.samples_written..],
        &mut raw_out_r[p_out1.samples_written..],
    );
    let raw_total = p_out1.samples_written + p_out2.samples_written;
    let expected_real = raw_total - latency as usize;

    assert_eq!(stream.output_real_total(), expected_real as u64);
    assert_eq!(stream.output_padded_total(), latency);
    assert_eq!(stream.underflow_total(), 0);
    assert_eq!(
        &out_l[..latency as usize],
        vec![0.0f32; latency as usize].as_slice()
    );
    let mut combined_l = out_l[latency as usize..].to_vec();
    combined_l.extend_from_slice(&drained_l);
    assert_eq!(combined_l, raw_out_l[latency as usize..raw_total]);
    let mut combined_r = out_r[latency as usize..].to_vec();
    combined_r.extend_from_slice(&drained_r);
    assert_eq!(combined_r, raw_out_r[latency as usize..raw_total]);
}

#[test]
fn test_process_zero_progress_mid_drain_preserves_remaining_input() {
    // Pre-fill out_fifo and leave only 2 slots in model_fifo so drain_to_mid
    // only drains 2 of 4 pending input samples; when model returns 0, the
    // remaining 2 input samples stay safely queued in in_fifo.
    let mut stream = StreamingResampleBuffer::new(48_000, 48_000, 4).expect("new failed");
    assert_eq!(stream.push_model_output(&[1.0; 8], &[1.0; 8]), 8);
    assert_eq!(stream.push_model_output(&[2.0; 4], &[2.0; 4]), 4);
    assert_eq!(stream.output_pending(), 8);
    assert_eq!(stream.model_pending(), 4);

    let mut out_l = [0.0f32; 2];
    let mut out_r = [0.0f32; 2];
    // pull_available_real pulls 2 from out_fifo -> drains 2 from model_fifo,
    // leaving 2 in model_fifo (4 free). Next push_input adds 2; wait: to hit
    // model_free == 2 when draining 4, push 2 more to model_fifo via process?
    // Even simpler: push 4 to in_fifo, saturate model_fifo to cap-2 by calling
    // pull_output(2) then push_model_output(2), leaving 2 free in model_fifo.
    let _ = stream.pull_output(&mut out_l, &mut out_r, 2);
    assert_eq!(stream.push_model_output(&[3.0; 2], &[3.0; 2]), 2);
    assert_eq!(stream.output_pending(), 8);
    assert_eq!(stream.model_pending(), 4);

    // Now process with n=2: step 1 pulls 2 -> drains 2 -> model_fifo has 4 free,
    // wait: request n=0 after push_input(4) so out_fifo stays full and
    // model_fifo keeps only 2 free slots!
    let mut empty_l = [];
    let mut empty_r = [];
    assert_eq!(stream.push_input(&[9.0, 10.0, 11.0, 12.0], &[9.0; 4]), 4);
    let res = stream.process(&[], &[], &mut empty_l, &mut empty_r, |_ml, _mr, _ol, _or| 0);
    assert_eq!(res.consumed, 0);
    assert_eq!(res.written, 0);
    assert!(res.underflow);
    assert_eq!(stream.input_pending(), 2);
}

#[test]
fn test_typed_constructor_invalid_rates_preserve_resampler_build_failed() {
    let invalid_rates = [0u32, 3_999, 384_001];
    for &bad_rate in &invalid_rates {
        assert_eq!(
            StreamingResampleBuffer::new(bad_rate, 48_000, 256).err(),
            Some(NamErrorCode::ResamplerBuildFailed),
            "new(host={bad_rate}, model=48000) must return ResamplerBuildFailed"
        );
        assert_eq!(
            StreamingResampleBuffer::new(48_000, bad_rate, 256).err(),
            Some(NamErrorCode::ResamplerBuildFailed),
            "new(host=48000, model={bad_rate}) must return ResamplerBuildFailed"
        );
        assert_eq!(
            StreamingResampleBuffer::new_linear(bad_rate, 48_000, 256).err(),
            Some(NamErrorCode::ResamplerBuildFailed),
            "new_linear(host={bad_rate}, model=48000) must return ResamplerBuildFailed"
        );
        assert_eq!(
            StreamingResampleBuffer::new_linear(48_000, bad_rate, 256).err(),
            Some(NamErrorCode::ResamplerBuildFailed),
            "new_linear(host=48000, model={bad_rate}) must return ResamplerBuildFailed"
        );
    }
    assert_eq!(
        StreamingResampleBuffer::new(0, 0, 256).err(),
        Some(NamErrorCode::ResamplerBuildFailed)
    );
    assert_eq!(
        StreamingResampleBuffer::new_linear(0, 0, 256).err(),
        Some(NamErrorCode::ResamplerBuildFailed)
    );
}

#[test]
fn test_typed_constructor_rate_boundaries_and_max_block_clamp() {
    for (host, model) in [(4_000u32, 4_000u32), (384_000, 384_000), (44_100, 192_000)] {
        let min_b = StreamingResampleBuffer::new(host, model, 0).expect("valid min boundary");
        assert_eq!(min_b.max_block(), 1, "max_block=0 must clamp to 1");

        let max_b =
            StreamingResampleBuffer::new(host, model, 8_193).expect("valid max_block clamp");
        assert_eq!(
            max_b.max_block(),
            8_192,
            "max_block=8193 must clamp to 8192"
        );

        let lin_min =
            StreamingResampleBuffer::new_linear(host, model, 0).expect("valid linear min boundary");
        assert_eq!(lin_min.max_block(), 1);

        let lin_max = StreamingResampleBuffer::new_linear(host, model, 8_193)
            .expect("valid linear max_block clamp");
        assert_eq!(lin_max.max_block(), 8_192);
    }
}

#[test]
fn test_typed_constructor_failures_record_in_log_buffer() {
    use crate::common::diagnostics::{LoggerConfig, NamLogger};
    let _ = NamLogger::init(LoggerConfig::default());
    let Some(buf) = NamLogger::log_buffer() else {
        return;
    };

    // Snapshot immediately after each failure: building the 4_000 -> 384_000 Hz
    // resampler below takes ~1s in debug, during which parallel tests can fill
    // and wrap the 256-slot global LogBuffer ring.
    let _ = StreamingResampleBuffer::new(3_999, 48_000, 256);
    let snap_build = buf.snapshot();
    assert!(
        snap_build
            .iter()
            .any(|r| r.message.contains("[StreamingResampleBuffer]") && r.message.contains("E2200")),
        "LogBuffer must record ResamplerBuildFailed (E2200)"
    );

    let _ = StreamingResampleBuffer::new(4_000, 384_000, 8_192);
    let snap_oom = buf.snapshot();
    assert!(
        snap_oom
            .iter()
            .any(|r| r.message.contains("[StreamingResampleBuffer]") && r.message.contains("E5000")),
        "LogBuffer must record OutOfMemory (E5000)"
    );
}
