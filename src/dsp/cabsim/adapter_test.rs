// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION;

fn direct_convolve(ir: &[f32], input: &[f32]) -> Vec<f32> {
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

fn compute_esr(reference: &[f32], computed: &[f32]) -> f64 {
    assert_eq!(reference.len(), computed.len());
    let mut ref_energy = 0.0f64;
    let mut err_energy = 0.0f64;
    for (r, c) in reference.iter().zip(computed.iter()) {
        let diff = *r as f64 - *c as f64;
        ref_energy += (*r as f64) * (*r as f64);
        err_energy += diff * diff;
    }
    if ref_energy < 1e-30 {
        return 0.0;
    }
    err_energy / ref_energy
}

fn synth_ir(len: usize, freq: f32, decay: f32, sample_rate: u32) -> Vec<f32> {
    (0..len)
        .map(|n| {
            let t = n as f32 / sample_rate as f32;
            (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
        })
        .collect()
}

fn adapter_from_ir(ir: &[f32], partition_size: usize) -> CabSimAdapter {
    let engine =
        Box::new(ConvEngine::new(ir, partition_size).expect("construction should succeed"));
    CabSimAdapter::new(engine).expect("adapter construction should succeed")
}

fn process_full_signal_fixed(engine: &mut ConvEngine, signal: &[f32]) -> Vec<f32> {
    let b = engine.partition_size();
    let mut output = Vec::with_capacity(signal.len() + b);
    let mut buf_in = vec![0.0f32; b];
    let mut buf_out = vec![0.0f32; b];

    let mut pos = 0;
    while pos < signal.len() {
        let chunk = (signal.len() - pos).min(b);
        buf_in[..chunk].copy_from_slice(&signal[pos..pos + chunk]);
        if chunk < b {
            buf_in[chunk..].fill(0.0);
        }
        engine.process(&buf_in, &mut buf_out, None);
        output.extend_from_slice(&buf_out[..chunk.min(b)]);
        pos += chunk;
    }

    let flush_blocks = engine.num_partitions();
    for _ in 0..flush_blocks {
        buf_in.fill(0.0);
        engine.process(&buf_in, &mut buf_out, None);
        output.extend_from_slice(&buf_out[..b]);
    }

    output
}

fn process_full_with_prefix(
    adapter: &mut CabSimAdapter,
    signal: &[f32],
    sub_sizes: &[usize],
) -> (Vec<f32>, usize) {
    let p = adapter.partition_size();
    let mut output = Vec::with_capacity(signal.len() + p * 4);
    let mut pos = 0;
    let mut sub_idx = 0;
    let mut zero_prefix = 0usize;
    let mut first_partition_done = false;

    while pos < signal.len() {
        let sub = if sub_idx < sub_sizes.len() {
            sub_sizes[sub_idx].min(signal.len() - pos)
        } else {
            signal.len() - pos
        };
        let sub = sub.min(p);
        if sub == 0 {
            break;
        }
        let mut buf_out = vec![0.0f32; sub];
        adapter.process_variable(&signal[pos..pos + sub], &mut buf_out, None);
        if !first_partition_done {
            if buf_out.iter().any(|&s| s.abs() > 1e-8) {
                first_partition_done = true;
            } else {
                zero_prefix += sub;
            }
        }
        output.extend_from_slice(&buf_out);
        pos += sub;
        sub_idx += 1;
    }

    let z = vec![0.0f32; p];
    let max_flush = adapter.num_partitions() + 3;
    for _ in 0..max_flush {
        let mut buf_out = vec![0.0f32; p];
        adapter.process_variable(&z[..], &mut buf_out, None);
        output.extend_from_slice(&buf_out);
    }

    let expected_output = (signal.len().div_ceil(p) + adapter.num_partitions()) * p;
    if output.len() > expected_output {
        output.truncate(expected_output);
    }

    (output, zero_prefix)
}

#[test]
fn passthrough_on_empty_ir() {
    let engine = Box::new(ConvEngine::new(&[], 64).expect("construction should succeed"));
    let adapter = CabSimAdapter::new(engine).expect("adapter construction should succeed");
    assert!(adapter.is_passthrough());
    assert_eq!(adapter.num_partitions(), 0);
    assert_eq!(adapter.latency_samples(), 64);
    assert_eq!(adapter.partition_size(), 64);

    let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();

    for &sub_size in &[1, 17, 47, 63, 64, 128] {
        let mut adapter2 = adapter_from_ir(&[], 64);
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < signal.len() {
            let n = sub_size
                .min(adapter2.partition_size())
                .min(signal.len() - pos);
            let mut buf = vec![0.0f32; n];
            adapter2.process_variable(&signal[pos..pos + n], &mut buf, None);
            for (i, &s) in buf.iter().enumerate() {
                assert!(
                    (s - signal[pos + i]).abs() < 1e-10,
                    "passthrough mismatch at pos={pos}, sub={sub_size}",
                );
            }
            out.extend_from_slice(&buf);
            pos += n;
        }
        assert_eq!(out, signal[..out.len()], "passthrough output differs");
    }
}

#[test]
fn reset_clears_fdl_and_fifos_matches_fresh() {
    // Reset invariant: after `reset()`, processing must be bit-identical to
    // a freshly constructed adapter with the same IR — no tail or accumulated
    // sub-block may survive the reset.
    let partition = 64;
    let ir = synth_ir(partition * 3, 700.0, 10.0, 48000);

    let mut dirty = adapter_from_ir(&ir, partition);
    let mut fresh = adapter_from_ir(&ir, partition);

    // Pollute: partial sub-blocks (odd sizes) exercise the input accumulator
    // and output FIFO, then a full signal fills the engine FDL.
    let signal: Vec<f32> = (0..200).map(|i| (i as f32 * 0.05).sin()).collect();
    for &sub in &[13usize, 29, 61, 64] {
        let mut pos = 0;
        while pos < signal.len() {
            let n = sub.min(signal.len() - pos);
            let mut out = vec![0.0f32; n];
            dirty.process_variable(&signal[pos..pos + n], &mut out, None);
            pos += n;
        }
    }
    assert!(dirty.needs_flush(), "adapter must hold state before reset");

    dirty.reset();
    assert!(
        !dirty.needs_flush(),
        "reset must empty the accumulator/output queue"
    );

    // Draining right after the reset must be silent — no residual tail from the
    // pre-reset FDL or FIFOs may replay.
    let z = vec![0.0f32; partition];
    let mut buf = vec![0.0f32; partition];
    for _ in 0..=dirty.num_partitions() {
        dirty.process_variable(&z, &mut buf, None);
        assert!(
            buf.iter().all(|&s| s == 0.0),
            "drain after reset must not replay pre-reset tail"
        );
    }

    // Same post-reset signal through both adapters.
    let probe: Vec<f32> = (0..partition * 4).map(|i| (i % 97) as f32 * 0.01).collect();
    let run = |adapter: &mut CabSimAdapter| {
        let mut out = Vec::new();
        for chunk in probe.chunks(partition) {
            let mut buf = vec![0.0f32; chunk.len()];
            adapter.process_variable(chunk, &mut buf, None);
            out.extend_from_slice(&buf);
        }
        out
    };

    let dirty_out = run(&mut dirty);
    let fresh_out = run(&mut fresh);
    assert_eq!(
        dirty_out, fresh_out,
        "post-reset processing must be bit-identical to a fresh adapter"
    );
}

#[test]
fn regular_blocks_parity() {
    let ir = synth_ir(200, 500.0, 8.0, 48000);
    let signal: Vec<f32> = (0..384)
        .map(|i| {
            let t = i as f32 / 48000.0;
            (std::f32::consts::TAU * 220.0 * t).sin()
                + 0.5 * (std::f32::consts::TAU * 660.0 * t).sin()
        })
        .collect();

    let partition = 128;
    let mut engine = ConvEngine::new(&ir, partition).expect("construction should succeed");
    let fixed_out = process_full_signal_fixed(&mut engine, &signal);

    let mut adapter = adapter_from_ir(&ir, partition);
    let sub_sizes: Vec<usize> = (0..signal.len()).map(|_| partition).collect();
    let (var_out, prefix) = process_full_with_prefix(&mut adapter, &signal, &sub_sizes);

    assert_eq!(prefix, 0, "regular blocks should have no zero prefix");

    let min_len = fixed_out.len().min(var_out.len());
    let esr = compute_esr(&fixed_out[..min_len], &var_out[..min_len]);
    assert!(
        esr < 1e-5,
        "ESR = {:.2e} for regular P-sized sub-blocks",
        esr
    );
}

#[test]
fn variable_sub_blocks_parity() {
    let ir = synth_ir(200, 500.0, 8.0, 48000);
    let signal: Vec<f32> = (0..384)
        .map(|i| {
            let t = i as f32 / 48000.0;
            (std::f32::consts::TAU * 220.0 * t).sin()
                + 0.5 * (std::f32::consts::TAU * 660.0 * t).sin()
        })
        .collect();

    let partition = 128;

    let mut engine = ConvEngine::new(&ir, partition).expect("construction should succeed");
    let fixed_out = process_full_signal_fixed(&mut engine, &signal);

    let mut adapter = adapter_from_ir(&ir, partition);

    let mut sub_list = Vec::new();
    let pattern = [17usize, 63, 48];
    let mut covered = 0;
    while covered < signal.len() {
        for &s in &pattern {
            let take = s.min(partition);
            sub_list.push(take);
            covered += take;
            if covered >= signal.len() {
                break;
            }
        }
    }

    let (var_out, prefix) = process_full_with_prefix(&mut adapter, &signal, &sub_list);

    assert!(
        prefix > 0,
        "variable sub-blocks should have zero prefix before first partition"
    );

    let fixed_slice = &fixed_out[0..fixed_out.len().min(var_out.len() - prefix)];
    let var_slice = &var_out[prefix..prefix + fixed_slice.len()];

    let esr = compute_esr(fixed_slice, var_slice);
    assert!(
        esr < 1e-5,
        "ESR = {:.2e} for variable sub-blocks (17-63-48) vs fixed (prefix={prefix})",
        esr
    );
}

#[test]
fn parity_with_direct_convolution() {
    let ir = synth_ir(256, 400.0, 6.0, 48000);
    let signal: Vec<f32> = (0..512)
        .map(|i| {
            let t = i as f32 / 48000.0;
            (std::f32::consts::TAU * 180.0 * t).sin()
        })
        .collect();

    let partition = 128;
    let ref_full = direct_convolve(&ir, &signal);

    let mut adapter = adapter_from_ir(&ir, partition);

    let mut sub_list = Vec::new();
    let pattern = [11usize, 97, 32, 44, 128, 7, 85, 53, 55];
    let mut covered = 0;
    while covered < signal.len() {
        for &s in &pattern {
            let take = s.min(partition);
            sub_list.push(take);
            covered += take;
            if covered >= signal.len() {
                break;
            }
        }
    }

    let (var_out, prefix) = process_full_with_prefix(&mut adapter, &signal, &sub_list);

    let ref_slice = &ref_full[0..ref_full.len().min(var_out.len() - prefix)];
    let var_slice = &var_out[prefix..prefix + ref_slice.len()];

    let esr = compute_esr(ref_slice, var_slice);
    assert!(
        esr < 1e-5,
        "ESR = {:.2e} for variable sub-blocks vs direct convolution (prefix={prefix})",
        esr
    );
}

#[test]
fn first_partition_produces_silence_during_accumulation() {
    let ir = synth_ir(100, 500.0, 10.0, 48000);
    let partition = 64;

    let mut adapter = adapter_from_ir(&ir, partition);

    let signal: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();

    let mut output = Vec::new();
    let mut pos = 0;
    for &sub in &[17, 17, 17, 13] {
        let mut buf = vec![0.0f32; sub];
        adapter.process_variable(&signal[pos..pos + sub], &mut buf, None);
        output.extend_from_slice(&buf);
        pos += sub;
    }

    assert_eq!(output.len(), 64);
    for (i, &s) in output.iter().enumerate().take(51) {
        assert!(
            s.abs() < 1e-6,
            "expected silence during accumulation at offset {i}, got {s}"
        );
    }
}

#[test]
fn sub_block_of_exact_partition_size() {
    let ir = synth_ir(100, 500.0, 8.0, 48000);
    let partition = 128;
    let signal: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();

    let mut engine = ConvEngine::new(&ir, partition).expect("construction should succeed");
    let fixed_out = process_full_signal_fixed(&mut engine, &signal);

    let mut adapter = adapter_from_ir(&ir, partition);
    let sub_sizes: Vec<usize> = vec![partition; signal.len().div_ceil(partition)];
    let (var_out, _prefix) = process_full_with_prefix(&mut adapter, &signal, &sub_sizes);

    let min_len = fixed_out.len().min(var_out.len());
    let esr = compute_esr(&fixed_out[..min_len], &var_out[..min_len]);
    assert!(esr < 1e-10, "ESR = {:.2e} for P-sized sub-blocks", esr);
}

#[test]
fn single_sample_sub_blocks() {
    let ir = synth_ir(30, 800.0, 10.0, 48000);
    let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
    let partition = 64;

    let mut engine = ConvEngine::new(&ir, partition).expect("construction should succeed");
    let fixed_out = process_full_signal_fixed(&mut engine, &signal);

    let mut adapter = adapter_from_ir(&ir, partition);

    let sub_sizes: Vec<usize> = (0..signal.len()).map(|_| 1).collect();
    let (var_out, prefix) = process_full_with_prefix(&mut adapter, &signal, &sub_sizes);

    assert!(
        prefix > 0,
        "single-sample sub-blocks should have zero prefix"
    );

    let fixed_slice = &fixed_out[0..fixed_out.len().min(var_out.len() - prefix)];
    let var_slice = &var_out[prefix..prefix + fixed_slice.len()];

    let esr = compute_esr(fixed_slice, var_slice);
    assert!(esr < 5e-4, "ESR = {:.2e} for single-sample sub-blocks", esr);
}

#[test]
fn non_power_of_two_partition_size() {
    let ir = synth_ir(80, 600.0, 12.0, 48000);
    let signal: Vec<f32> = (0..300).map(|i| (i as f32 * 0.03).sin()).collect();
    let partition = 75;

    let mut engine = ConvEngine::new(&ir, partition).expect("construction should succeed");
    let fixed_out = process_full_signal_fixed(&mut engine, &signal);

    let mut adapter = adapter_from_ir(&ir, partition);

    let mut sub_list = Vec::new();
    let pattern = [13usize, 47, 23, 29, 37, 61, 17];
    let mut covered = 0;
    while covered < signal.len() {
        for &s in &pattern {
            let take = s.min(partition);
            sub_list.push(take);
            covered += take;
            if covered >= signal.len() {
                break;
            }
        }
    }

    let (var_out, prefix) = process_full_with_prefix(&mut adapter, &signal, &sub_list);

    let fixed_slice = &fixed_out[0..fixed_out.len().min(var_out.len() - prefix)];
    let var_slice = &var_out[prefix..prefix + fixed_slice.len()];

    let esr = compute_esr(fixed_slice, var_slice);
    assert!(
        esr < 5e-1,
        "ESR = {:.2e} for non-power-of-2 partition ({partition})",
        esr
    );
}

#[test]
fn process_zero_length_input_no_panic() {
    let ir = synth_ir(50, 440.0, 10.0, 48000);
    let mut adapter = adapter_from_ir(&ir, 64);

    let mut empty_out = vec![];
    adapter.process_variable(&[], &mut empty_out, None);
}

#[test]
fn single_sample_ir() {
    let ir = vec![0.75f32];
    let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02).sin()).collect();
    let partition = 64;

    let ref_full = direct_convolve(&ir, &signal);

    let mut adapter = adapter_from_ir(&ir, partition);

    let mut sub_list = Vec::new();
    let pattern = [17usize, 63, 48];
    let mut covered = 0;
    while covered < signal.len() {
        for &s in &pattern {
            let take = s.min(partition);
            sub_list.push(take);
            covered += take;
            if covered >= signal.len() {
                break;
            }
        }
    }

    let (var_out, prefix) = process_full_with_prefix(&mut adapter, &signal, &sub_list);

    let ref_slice = &ref_full[0..ref_full.len().min(var_out.len() - prefix)];
    let var_slice = &var_out[prefix..prefix + ref_slice.len()];

    let esr = compute_esr(ref_slice, var_slice);
    assert!(
        esr < 1e-5,
        "ESR = {:.2e} for single-sample IR with variable sub-blocks",
        esr
    );
}

#[test]
fn deterministic_output() {
    let ir = synth_ir(60, 350.0, 8.0, 48000);
    let signal: Vec<f32> = (0..300).map(|i| (i as f32 * 0.01).sin()).collect();
    let partition = 64;

    let mut adapter1 = adapter_from_ir(&ir, partition);
    let mut adapter2 = adapter_from_ir(&ir, partition);

    let sub_sizes: &[usize] = &[17, 63, 48, 17, 63, 48, 17, 63, 48];
    let (out1, _) = process_full_with_prefix(&mut adapter1, &signal, sub_sizes);
    let (out2, _) = process_full_with_prefix(&mut adapter2, &signal, sub_sizes);

    assert_eq!(out1.len(), out2.len());
    for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "non-deterministic output at index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn needs_flush_after_partial_input() {
    let ir = synth_ir(64, 440.0, 10.0, 48000);
    let mut adapter = adapter_from_ir(&ir, 64);

    assert!(!adapter.needs_flush());
    assert_eq!(adapter.tail_samples(), 128);

    let signal: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut out = vec![0.0f32; 32];
    adapter.process_variable(&signal, &mut out, None);
    assert!(adapter.needs_flush());
}

#[test]
fn needs_flush_cleared_when_drained() {
    let ir = synth_ir(32, 440.0, 10.0, 48000);
    let mut adapter = adapter_from_ir(&ir, 32);

    let signal: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut out = vec![0.0f32; 32];
    adapter.process_variable(&signal[..32], &mut out, None);
    assert!(!adapter.needs_flush());

    let mut out2 = vec![0.0f32; 32];
    adapter.process_variable(&signal[32..], &mut out2, None);
    assert!(!adapter.needs_flush());
}

#[test]
fn tail_samples_passthrough_returns_zero() {
    let engine = Box::new(ConvEngine::new(&[], 64).expect("construction failed"));
    let adapter = CabSimAdapter::new(engine).expect("adapter construction should succeed");
    assert!(adapter.is_passthrough());
    assert_eq!(adapter.tail_samples(), 0);
    assert!(!adapter.needs_flush());
}

#[test]
fn tail_samples_single_partition() {
    let ir = synth_ir(30, 500.0, 10.0, 48000);
    let partition = 64;
    let adapter = adapter_from_ir(&ir, partition);
    assert_eq!(adapter.tail_samples(), 128);
    assert_eq!(adapter.num_partitions(), 1);
}

#[test]
fn oversize_sub_block_clamps_and_raises_flag() {
    // Cabsim contract guard (F-03): a sub-block 2× the partition (host quantum renegotiation
    // window) must not panic — the adapter clamps defensively and raises the
    // CABSIM_CONTRACT_VIOLATION status flag.
    let ir = synth_ir(60, 500.0, 10.0, 48000);
    let partition = 64;
    let mut adapter = adapter_from_ir(&ir, partition);

    let rt = RtStatusFlags::new();
    let signal: Vec<f32> = (0..(2 * partition))
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let mut out = vec![0.0f32; 2 * partition];

    adapter.process_variable(&signal, &mut out, Some(&rt));

    assert!(rt.check_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION));
    assert!(!rt.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));
}

#[test]
fn mismatched_output_len_clamps_and_raises_flag() {
    // Cabsim contract guard (F-03): input/output length disagreement must never panic in
    // release — the adapter clamps to the shortest slice and raises the
    // contract flag. Debug builds keep `debug_assert_eq!` (loud during
    // development), so the two modes are asserted separately.
    let ir = synth_ir(60, 500.0, 10.0, 48000);
    let partition = 64;
    let mut adapter = adapter_from_ir(&ir, partition);

    let signal: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();

    let rt = RtStatusFlags::new();
    let mut short_out = vec![0.0f32; 16];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adapter.process_variable(&signal, &mut short_out, Some(&rt));
    }));

    #[cfg(debug_assertions)]
    {
        assert!(
            result.is_err(),
            "debug build must assert on input/output length mismatch"
        );
    }
    #[cfg(not(debug_assertions))]
    {
        assert!(
            result.is_ok(),
            "release build must never panic on input/output length mismatch"
        );
        assert!(rt.check_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION));
    }
}

#[test]
fn contract_compliant_sub_blocks_do_not_raise_flag() {
    let ir = synth_ir(60, 500.0, 10.0, 48000);
    let partition = 64;
    let mut adapter = adapter_from_ir(&ir, partition);

    let rt = RtStatusFlags::new();
    let signal: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut out = vec![0.0f32; 64];
    adapter.process_variable(&signal, &mut out, Some(&rt));
    assert!(!rt.check_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION));
}

#[test]
fn in_place_matches_separate_buffer() {
    // `process_in_place` must produce bit-identical output to the
    // separate-buffer `process_variable` for the same sub-block schedule
    // (the FIFO paths are shared, so the only difference is the destination).
    let ir = synth_ir(200, 500.0, 8.0, 48000);
    let signal: Vec<f32> = (0..384)
        .map(|i| {
            let t = i as f32 / 48000.0;
            (std::f32::consts::TAU * 220.0 * t).sin()
                + 0.5 * (std::f32::consts::TAU * 660.0 * t).sin()
        })
        .collect();

    let partition = 128;
    let pattern = [17usize, 63, 48];

    let mut adapter_ref = adapter_from_ir(&ir, partition);
    let mut ref_out = Vec::new();
    let mut pos = 0;
    let mut sub_idx = 0;
    while pos < signal.len() {
        let sub = pattern[sub_idx % pattern.len()]
            .min(partition)
            .min(signal.len() - pos);
        let mut buf = vec![0.0f32; sub];
        adapter_ref.process_variable(&signal[pos..pos + sub], &mut buf, None);
        ref_out.extend_from_slice(&buf);
        pos += sub;
        sub_idx += 1;
    }

    let mut adapter_ip = adapter_from_ir(&ir, partition);
    let mut ip_out = Vec::new();
    let mut pos = 0;
    let mut sub_idx = 0;
    while pos < signal.len() {
        let sub = pattern[sub_idx % pattern.len()]
            .min(partition)
            .min(signal.len() - pos);
        let mut buf = vec![0.0f32; sub];
        buf[..sub].copy_from_slice(&signal[pos..pos + sub]);
        adapter_ip.process_in_place(&mut buf, None);
        ip_out.extend_from_slice(&buf);
        pos += sub;
        sub_idx += 1;
    }

    assert_eq!(ref_out.len(), ip_out.len());
    for (i, (a, b)) in ref_out.iter().zip(ip_out.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "in-place vs separate-buffer mismatch at index {i}"
        );
    }
}

#[test]
fn in_place_passthrough_identity() {
    // Passthrough in-place must preserve the buffer unchanged (identity).
    let engine = Box::new(ConvEngine::new(&[], 64).expect("construction should succeed"));
    let mut adapter = CabSimAdapter::new(engine).expect("adapter construction should succeed");
    assert!(adapter.is_passthrough());

    let mut buf: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();
    let reference = buf.clone();
    adapter.process_in_place(&mut buf, None);
    assert_eq!(buf, reference, "passthrough in-place must preserve samples");
}

#[test]
fn in_place_oversize_clamps_and_raises_flag() {
    // In-place oversize sub-block must clamp and raise the contract flag
    // without panicking (same fail-closed contract as `process_variable`).
    let ir = synth_ir(60, 500.0, 10.0, 48000);
    let partition = 64;
    let mut adapter = adapter_from_ir(&ir, partition);

    let rt = RtStatusFlags::new();
    let mut buf = vec![0.0f32; 2 * partition];
    adapter.process_in_place(&mut buf, Some(&rt));
    assert!(rt.check_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION));
}

// ── CabSimPair: stereo decoupling ─────────────────────────

/// Unit delta on L with absolute silence on R must yield an R
/// output that is rigorously `0.0` (-inf dB crosstalk) for the entire IR tail.
#[test]
fn pair_crosstalk_delta_l_only_r_exactly_silent() {
    let ir = synth_ir(160, 300.0, 12.0, 48000);
    let partition = 32;
    let mut pair = CabSimPair {
        l: Box::new(adapter_from_ir(&ir, partition)),
        r: Box::new(adapter_from_ir(&ir, partition)),
        sample_rate: 48000,
    };
    assert_eq!(pair.partition_size(), partition);

    let zeros = vec![0.0f32; partition];
    let mut delta_block = vec![0.0f32; partition];
    delta_block[0] = 1.0;
    let mut out_l = vec![0.0f32; partition];
    let mut out_r = vec![0.0f32; partition];

    let n_blocks = pair.l.num_partitions() + 6;
    let mut l_energy = 0.0f32;
    for block in 0..n_blocks {
        let input_l: &[f32] = if block == 0 { &delta_block } else { &zeros };
        pair.l.process_variable(input_l, &mut out_l, None);
        pair.r.process_variable(&zeros, &mut out_r, None);
        l_energy += out_l.iter().map(|s| s * s).sum::<f32>();
        assert!(
            out_r.iter().all(|&s| s == 0.0),
            "R output must be rigorously 0.0 with silent input (block {block})"
        );
    }
    assert!(
        l_energy > 0.0,
        "L must convolve the delta (test would be vacuous otherwise)"
    );
}

/// Deterministic LCG noise — identical sequences must feed both channels so
/// the bit-exact comparison isolates state coupling from signal differences.
fn lcg_noise(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / 16_777_216.0) - 1.0
        })
        .collect()
}

/// `process_in_place_stereo` must be bit-identical to driving each channel's
/// adapter individually — the stereo convenience must never introduce shared
/// state or reordering between L and R, under variable sub-block sizes.
#[test]
fn pair_process_in_place_stereo_matches_individual_adapters() {
    let ir = synth_ir(220, 700.0, 8.0, 48000);
    let partition = 64;
    let mut pair = CabSimPair {
        l: Box::new(adapter_from_ir(&ir, partition)),
        r: Box::new(adapter_from_ir(&ir, partition)),
        sample_rate: 48000,
    };
    let mut mono_l = adapter_from_ir(&ir, partition);
    let mut mono_r = adapter_from_ir(&ir, partition);

    let signal_l = lcg_noise(0x5EED_0011, 512);
    let signal_r = lcg_noise(0x5EED_0022, 512);
    let sub_sizes = [64usize, 13, 37, 64, 1, 51, 64, 29];

    let mut pos = 0usize;
    let mut block = 0usize;
    let mut out_l = vec![0.0f32; partition];
    let mut out_r = vec![0.0f32; partition];
    let mut ref_out = vec![0.0f32; partition];

    while pos < signal_l.len() {
        let sub = sub_sizes[block % sub_sizes.len()].min(partition);
        let n = sub.min(signal_l.len() - pos);

        out_l[..n].copy_from_slice(&signal_l[pos..pos + n]);
        out_r[..n].copy_from_slice(&signal_r[pos..pos + n]);
        pair.process_in_place_stereo(&mut out_l[..n], &mut out_r[..n], None);

        mono_l.process_variable(&signal_l[pos..pos + n], &mut ref_out[..n], None);
        assert_eq!(
            &out_l[..n],
            &ref_out[..n],
            "pair.process_in_place_stereo L must be bit-exact vs an independent mono adapter (block {block})"
        );
        mono_r.process_variable(&signal_r[pos..pos + n], &mut ref_out[..n], None);
        assert_eq!(
            &out_r[..n],
            &ref_out[..n],
            "pair.process_in_place_stereo R must be bit-exact vs an independent mono adapter (block {block})"
        );

        pos += n;
        block += 1;
    }
}

// ── IR tail drain (gate-closed ring-out) ──────────────────────────────────

/// After the input goes silent, `drain_tail` must render the decaying IR
/// ring-out (non-zero output), consume the ring-out budget strictly, and
/// terminate with silence once the budget and the FDL are exhausted.
#[test]
fn drain_tail_rings_out_and_terminates() {
    let ir = synth_ir(256, 500.0, 4.0, 48000); // slow decay: tail stays audible
    let partition = 64;
    let mut adapter = adapter_from_ir(&ir, partition);

    // Feed enough active signal to fill the FDL.
    let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
    let mut out = vec![0.0f32; partition];
    for chunk in signal.chunks(partition) {
        adapter.process_variable(chunk, &mut out[..chunk.len()], None);
    }

    // Budget armed by the pipeline on the active path.
    adapter.rearm_tail();
    let budget = adapter.remaining_tail_samples();
    assert_eq!(
        budget,
        adapter.tail_samples(),
        "budget must equal tail_samples"
    );

    // Drain: non-zero ring-out while the budget lasts, strictly counting down.
    let mut drained_samples = 0usize;
    let mut tail_energy = 0.0f32;
    let mut buf = vec![0.0f32; partition];
    while adapter.remaining_tail_samples() > 0 {
        buf.fill(0.0);
        adapter.drain_tail(&mut buf, None);
        tail_energy += buf.iter().map(|s| s * s).sum::<f32>();
        drained_samples += partition;
    }
    assert!(tail_energy > 0.0, "drain must emit the audible IR ring-out");
    assert_eq!(
        drained_samples, budget,
        "drain must consume exactly the armed budget"
    );

    // Post-budget: silence forever (FDL fully flushed, counter stays 0).
    for _ in 0..(adapter.num_partitions() + 2) {
        buf.fill(0.0);
        adapter.drain_tail(&mut buf, None);
        assert!(
            buf.iter().all(|&s| s == 0.0),
            "drain after budget exhaustion must emit silence"
        );
        assert_eq!(adapter.remaining_tail_samples(), 0);
    }
}

/// Active audio after a partial drain must re-arm the budget, so every gate
/// close flushes the complete IR response (a consumed budget must not
/// truncate the ring-out of a later passage).
#[test]
fn rearm_tail_after_activity_resets_budget() {
    let ir = synth_ir(128, 600.0, 6.0, 48000);
    let partition = 64;
    let mut adapter = adapter_from_ir(&ir, partition);

    let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
    let mut out = vec![0.0f32; partition];
    adapter.process_variable(&signal[..64], &mut out, None);
    adapter.rearm_tail();
    assert_eq!(adapter.remaining_tail_samples(), adapter.tail_samples());

    // Partial drain consumes the budget.
    out.fill(0.0);
    adapter.drain_tail(&mut out, None);
    assert_eq!(
        adapter.remaining_tail_samples(),
        adapter.tail_samples() - partition
    );

    // New active audio re-arms the full budget.
    adapter.process_variable(&signal[64..], &mut out, None);
    adapter.rearm_tail();
    assert_eq!(adapter.remaining_tail_samples(), adapter.tail_samples());
}

/// Oversize drain sub-blocks follow the same fail-closed contract as
/// `process_in_place`: clamp + contract flag, never a panic. Only the clamped
/// (partition-sized) prefix drains per call, so the budget decreases by
/// exactly one partition.
#[test]
fn drain_tail_oversize_clamps_and_raises_flag() {
    let ir = synth_ir(60, 500.0, 10.0, 48000);
    let partition = 64;
    let mut adapter = adapter_from_ir(&ir, partition);
    adapter.rearm_tail();
    let budget = adapter.remaining_tail_samples();

    let rt = RtStatusFlags::new();
    let mut buf = vec![0.0f32; 2 * partition];
    adapter.drain_tail(&mut buf, Some(&rt));
    assert!(rt.check_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION));
    assert_eq!(adapter.remaining_tail_samples(), budget - partition);
}

/// Passthrough adapters have no IR tail: drain must emit silence and the
/// budget must stay at zero through re-arm.
#[test]
fn drain_tail_passthrough_is_silent_and_budget_free() {
    let engine = Box::new(ConvEngine::new(&[], 64).expect("construction should succeed"));
    let mut adapter = CabSimAdapter::new(engine).expect("adapter construction should succeed");
    adapter.rearm_tail();
    assert_eq!(adapter.remaining_tail_samples(), 0);

    let mut buf = vec![0.0f32; 64];
    adapter.drain_tail(&mut buf, None);
    assert!(buf.iter().all(|&s| s == 0.0));
}

/// `reset()` must clear the ring-out budget along with the FIFOs and the FDL.
#[test]
fn reset_clears_tail_budget() {
    let ir = synth_ir(64, 440.0, 10.0, 48000);
    let mut adapter = adapter_from_ir(&ir, 64);
    adapter.rearm_tail();
    assert!(adapter.remaining_tail_samples() > 0);

    adapter.reset();
    assert_eq!(adapter.remaining_tail_samples(), 0);
}

/// Each pair channel must be bit-identical to an independent
/// mono `CabSimAdapter` running the same IR and the same signal — no shared
/// FIFO/FDL state may leak between L and R, under variable sub-block sizes.
#[test]
fn pair_bit_exact_vs_two_mono_adapters() {
    let ir = synth_ir(220, 700.0, 8.0, 48000);
    let partition = 64;
    let mut pair = CabSimPair {
        l: Box::new(adapter_from_ir(&ir, partition)),
        r: Box::new(adapter_from_ir(&ir, partition)),
        sample_rate: 48000,
    };
    let mut mono_l = adapter_from_ir(&ir, partition);
    let mut mono_r = adapter_from_ir(&ir, partition);

    let signal_l = lcg_noise(0x5EED_0001, 1024);
    let signal_r = lcg_noise(0x5EED_0002, 1024);
    let sub_sizes = [64usize, 13, 37, 64, 1, 51, 64, 29];

    let mut pos_l = 0usize;
    let mut pos_r = 0usize;
    let mut block = 0usize;
    let mut out = vec![0.0f32; partition];
    let mut ref_out = vec![0.0f32; partition];

    while pos_l < signal_l.len() || pos_r < signal_r.len() {
        let sub = sub_sizes[block % sub_sizes.len()];

        if pos_l < signal_l.len() {
            let n = sub.min(partition).min(signal_l.len() - pos_l);
            pair.l
                .process_variable(&signal_l[pos_l..pos_l + n], &mut out[..n], None);
            mono_l.process_variable(&signal_l[pos_l..pos_l + n], &mut ref_out[..n], None);
            assert_eq!(
                &out[..n],
                &ref_out[..n],
                "pair.l must be bit-exact vs an independent mono adapter (block {block})"
            );
            pos_l += n;
        }

        if pos_r < signal_r.len() {
            let n = sub.min(partition).min(signal_r.len() - pos_r);
            pair.r
                .process_variable(&signal_r[pos_r..pos_r + n], &mut out[..n], None);
            mono_r.process_variable(&signal_r[pos_r..pos_r + n], &mut ref_out[..n], None);
            assert_eq!(
                &out[..n],
                &ref_out[..n],
                "pair.r must be bit-exact vs an independent mono adapter (block {block})"
            );
            pos_r += n;
        }

        block += 1;
    }
}

// ── process_block: driver de convolução particionada (bloco de tamanho arbitrário) ──

/// Reference stream for a given host block size: what a consumer runs today —
/// each host block is sliced into `partition_size`-capped windows and each
/// window goes through one `process_in_place` call — the exact blocking
/// `process_block` must reproduce bit-for-bit.
fn drive_reference_fixed_path(
    adapter: &mut CabSimAdapter,
    signal: &[f32],
    total: usize,
    block: usize,
) -> Vec<f32> {
    let p = adapter.partition_size();
    let mut out = Vec::with_capacity(total);
    let mut pos = 0;
    while pos < total {
        let n = block.min(total - pos);
        let mut host = vec![0.0f32; n];
        if pos < signal.len() {
            let take = n.min(signal.len() - pos);
            host[..take].copy_from_slice(&signal[pos..pos + take]);
        }
        let mut wpos = 0;
        while wpos < n {
            let w = p.min(n - wpos);
            let mut win = vec![0.0f32; w];
            win.copy_from_slice(&host[wpos..wpos + w]);
            adapter.process_in_place(&mut win, None);
            out.extend_from_slice(&win);
            wpos += w;
        }
        pos += n;
    }
    out
}

/// Same input stream through the driver with host blocks of `block` samples.
fn drive_process_block(
    adapter: &mut CabSimAdapter,
    signal: &[f32],
    total: usize,
    block: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(total);
    let mut pos = 0;
    while pos < total {
        let n = block.min(total - pos);
        let mut buf = vec![0.0f32; n];
        if pos < signal.len() {
            let take = n.min(signal.len() - pos);
            buf[..take].copy_from_slice(&signal[pos..pos + take]);
        }
        adapter.process_block(&mut buf, None);
        out.extend_from_slice(&buf);
        pos += n;
    }
    out
}

/// Causal silence prefix of a fresh adapter driven with host blocks of
/// `block` samples: the samples delivered before the first partition
/// completes (`block * (ceil(P / block) - 1)` for `block <= partition`).
fn accumulation_prefix(partition: usize, block: usize) -> usize {
    block * (partition.div_ceil(block) - 1)
}

/// Sweep: for every partition × host-block combination, `process_block` must
/// be bit-identical to driving `process_in_place` with the same host-block
/// windows (same accumulate order, same partition MAC order, same delivery
/// order), including the full zero-input ring-out.
#[test]
fn process_block_bit_exact_blocking_sweep() {
    let partitions = [32usize, 64, 128, 256];
    let blocks = [16usize, 32, 64, 256, 333];
    let ir = synth_ir(600, 350.0, 5.0, 48000);
    let signal = lcg_noise(0x5EED_00B1, 2331);

    for &p in &partitions {
        let probe = adapter_from_ir(&ir, p);
        let total = signal.len() + probe.tail_samples() + p;
        for &b in &blocks {
            let mut reference = adapter_from_ir(&ir, p);
            let ref_out = drive_reference_fixed_path(&mut reference, &signal, total, b);
            let mut driven = adapter_from_ir(&ir, p);
            let drv_out = drive_process_block(&mut driven, &signal, total, b);

            assert_eq!(drv_out.len(), ref_out.len());
            for (i, (r, d)) in ref_out.iter().zip(drv_out.iter()).enumerate() {
                assert_eq!(
                    r, d,
                    "bit-exact violation at sample {i} (partition {p}, block {b})"
                );
            }
        }
    }
}

/// The convolved body is blocking-invariant: for host blocks that divide the
/// partition (aligned FIFO windows, no causal underrun gaps), stripping each
/// stream's accumulation prefix yields the exact same sample sequence for
/// every host-block size.
#[test]
fn process_block_body_blocking_invariant() {
    let partition = 64usize;
    let ir = synth_ir(600, 350.0, 5.0, 48000);
    let signal = lcg_noise(0x5EED_00B2, 2331);

    let probe = adapter_from_ir(&ir, partition);
    let total = signal.len() + probe.tail_samples() + partition;
    let mut golden: Option<Vec<f32>> = None;
    for &b in &[16usize, 32, 64, 256] {
        let mut adapter = adapter_from_ir(&ir, partition);
        let out = drive_process_block(&mut adapter, &signal, total, b);
        let prefix = accumulation_prefix(partition, b);
        match &golden {
            None => golden = Some(out[prefix..].to_vec()),
            Some(g) => {
                for (i, (a, d)) in g.iter().zip(out[prefix..].iter()).enumerate() {
                    assert_eq!(a, d, "blocking-invariance violation at {i} (block {b})");
                }
            }
        }
    }
}

/// ESR acceptance gate: cab-sim stage with host blocks of 32 samples through
/// a partition-64 engine stays within the established ESR threshold of the
/// direct-convolution oracle.
#[test]
fn process_block_esr_gate_chunking_32_to_64() {
    let ir = synth_ir(256, 400.0, 6.0, 48000);
    let signal: Vec<f32> = (0..512)
        .map(|i| {
            let t = i as f32 / 48000.0;
            (std::f32::consts::TAU * 180.0 * t).sin()
        })
        .collect();
    let partition = 64usize;
    let block = 32usize;

    let mut adapter = adapter_from_ir(&ir, partition);
    let total = signal.len() + adapter.tail_samples() + partition;
    let out = drive_process_block(&mut adapter, &signal, total, block);

    let ref_full = direct_convolve(&ir, &signal);
    let prefix = accumulation_prefix(partition, block);
    let n = ref_full.len().min(out.len() - prefix);
    let esr = compute_esr(&ref_full[..n], &out[prefix..prefix + n]);
    assert!(esr < 1e-5, "ESR = {esr:.2e} for chunking 32→64");
}

/// ESR acceptance gate with 16-sample host blocks through a partition-64
/// engine (worst practical sub-block ratio).
#[test]
fn process_block_esr_gate_chunking_16_to_64() {
    let ir = synth_ir(256, 400.0, 6.0, 48000);
    let signal: Vec<f32> = (0..512)
        .map(|i| {
            let t = i as f32 / 48000.0;
            (std::f32::consts::TAU * 180.0 * t).sin()
        })
        .collect();
    let partition = 64usize;
    let block = 16usize;

    let mut adapter = adapter_from_ir(&ir, partition);
    let total = signal.len() + adapter.tail_samples() + partition;
    let out = drive_process_block(&mut adapter, &signal, total, block);

    let ref_full = direct_convolve(&ir, &signal);
    let prefix = accumulation_prefix(partition, block);
    let n = ref_full.len().min(out.len() - prefix);
    let esr = compute_esr(&ref_full[..n], &out[prefix..prefix + n]);
    assert!(esr < 1e-5, "ESR = {esr:.2e} for chunking 16→64");
}

/// Ring-out semantics are unchanged: `rearm_tail` stays per host block and
/// the `drain_tail` ring-out is bit-identical to the current path. Both
/// adapters consume the same input through their own path (bit-identical
/// delivered streams and therefore identical FIFO/FDL states), re-arm once,
/// and drain with the same sub-blocking.
#[test]
fn process_block_tail_ringout_matches_fixed_path() {
    let partition = 64usize;
    let ir = synth_ir(400, 300.0, 4.0, 48000);
    let signal = lcg_noise(0x5EED_00B3, 777);

    // Driver: arbitrary host blocks through process_block.
    let mut driver = adapter_from_ir(&ir, partition);
    let mut pos = 0;
    while pos < signal.len() {
        let n = 333.min(signal.len() - pos);
        let mut buf = vec![0.0f32; n];
        buf.copy_from_slice(&signal[pos..pos + n]);
        driver.process_block(&mut buf, None);
        pos += n;
    }

    // Fixed path: the same input through partition-capped process_in_place
    // windows (the current consumer path).
    let mut fixed = adapter_from_ir(&ir, partition);
    let _ = drive_reference_fixed_path(&mut fixed, &signal, signal.len(), 333);

    // Identical states: re-arm once per path and drain with the same
    // sub-blocking until both budgets are exhausted.
    driver.rearm_tail();
    fixed.rearm_tail();
    assert_eq!(
        driver.remaining_tail_samples(),
        fixed.remaining_tail_samples()
    );

    let mut drain_out = Vec::new();
    let mut slice = vec![0.0f32; partition];
    while driver.remaining_tail_samples() > 0 {
        driver.drain_tail(&mut slice, None);
        drain_out.extend_from_slice(&slice);
    }
    let mut fixed_out = Vec::new();
    while fixed.remaining_tail_samples() > 0 {
        fixed.drain_tail(&mut slice, None);
        fixed_out.extend_from_slice(&slice);
    }

    assert_eq!(drain_out.len(), fixed_out.len());
    assert_eq!(drain_out, fixed_out, "ring-out must be bit-identical");
    assert!(
        drain_out.iter().any(|&s| s != 0.0),
        "the ring-out must carry the IR tail (non-vacuous comparison)"
    );
    // Beyond the armed budget the FDL residue (if any) evolves identically on
    // both paths: post-exhaustion drains stay bit-identical.
    let mut post_driver = vec![1.0f32; 64];
    let mut post_fixed = vec![1.0f32; 64];
    driver.drain_tail(&mut post_driver, None);
    fixed.drain_tail(&mut post_fixed, None);
    assert_eq!(post_driver, post_fixed);
}

/// The driver chunks by construction: a host block larger than the partition
/// never raises `RT_STATUS_CABSIM_CONTRACT_VIOLATION`, while the direct
/// sub-block contract still does (preserved for direct adapter use).
#[test]
fn process_block_never_raises_contract_violation() {
    use crate::common::spsc::RT_STATUS_CABSIM_CONTRACT_VIOLATION;
    use crate::common::spsc::RtStatusFlags;

    let ir = synth_ir(128, 500.0, 8.0, 48000);
    let partition = 64;

    let rt_driver = RtStatusFlags::new();
    let mut driver = adapter_from_ir(&ir, partition);
    let mut oversized = vec![0.5f32; 333];
    driver.process_block(&mut oversized, Some(&rt_driver));
    assert!(
        !rt_driver.check_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION),
        "process_block must never flag a contract violation"
    );

    let rt_direct = RtStatusFlags::new();
    let mut direct = adapter_from_ir(&ir, partition);
    let mut oversized_direct = vec![0.5f32; 333];
    direct.process_in_place(&mut oversized_direct, Some(&rt_direct));
    assert!(
        rt_direct.check_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION),
        "the existing sub-block contract must keep flagging oversized slices"
    );
}

/// Heap-audit lane: the driver hot path performs zero allocations across the
/// full blocking matrix.
#[test]
fn process_block_zero_alloc_hot_path() {
    use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};

    let partition = 64usize;
    let ir = synth_ir(300, 400.0, 6.0, 48000);
    let signal = lcg_noise(0x5EED_00B4, 4096);
    let blocks = [1usize, 7, 16, 48, 64, 333, 1024];

    for &block in &blocks {
        let mut adapter = adapter_from_ir(&ir, partition);
        let mut buf = vec![0.0f32; block];
        // Warm the path so lazy state (if any) is outside the watchdog.
        buf.copy_from_slice(&signal[..block]);
        adapter.process_block(&mut buf, None);

        let guard = TrackingGuard::new();
        let mut pos = 0;
        while pos < signal.len() {
            let n = block.min(signal.len() - pos);
            buf[..n].copy_from_slice(&signal[pos..pos + n]);
            adapter.process_block(&mut buf[..n], None);
            pos += n;
        }
        let allocs = get_alloc_count();
        drop(guard);
        assert_eq!(
            allocs, 0,
            "allocation detected in process_block (block {block})"
        );
    }
}

/// `process_block_stereo` must be bit-identical to driving each channel's
/// adapter individually with the same host blocks.
#[test]
fn pair_process_block_stereo_matches_individual_adapters() {
    let ir = synth_ir(220, 700.0, 8.0, 48000);
    let partition = 64;
    let mut pair = CabSimPair {
        l: Box::new(adapter_from_ir(&ir, partition)),
        r: Box::new(adapter_from_ir(&ir, partition)),
        sample_rate: 48000,
    };
    let mut mono_l = adapter_from_ir(&ir, partition);
    let mut mono_r = adapter_from_ir(&ir, partition);

    let signal_l = lcg_noise(0x5EED_00C1, 2048);
    let signal_r = lcg_noise(0x5EED_00C2, 1024);
    let blocks = [64usize, 13, 37, 333, 1, 51];

    let mut pos_l = 0usize;
    let mut pos_r = 0usize;
    let mut block = 0usize;
    while pos_l < signal_l.len() || pos_r < signal_r.len() {
        let size = blocks[block % blocks.len()];

        let n_l = size.min(signal_l.len().saturating_sub(pos_l));
        let n_r = size.min(signal_r.len().saturating_sub(pos_r));
        let n = n_l.max(n_r);
        if n == 0 {
            break;
        }

        let mut buf_l = vec![0.0f32; n];
        let mut buf_r = vec![0.0f32; n];
        buf_l[..n_l].copy_from_slice(&signal_l[pos_l..pos_l + n_l]);
        buf_r[..n_r].copy_from_slice(&signal_r[pos_r..pos_r + n_r]);

        let mut ref_l = buf_l.clone();
        let mut ref_r = buf_r.clone();

        pair.process_block_stereo(&mut buf_l, &mut buf_r, None);
        mono_l.process_block(&mut ref_l, None);
        mono_r.process_block(&mut ref_r, None);

        assert_eq!(buf_l, ref_l, "pair L must be bit-exact (block {block})");
        assert_eq!(buf_r, ref_r, "pair R must be bit-exact (block {block})");
        pos_l += n_l;
        pos_r += n_r;
        block += 1;
    }
}
