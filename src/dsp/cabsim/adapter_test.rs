// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

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
        adapter.process_variable(&signal[pos..pos + sub], &mut buf_out);
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
        adapter.process_variable(&z[..], &mut buf_out);
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
            adapter2.process_variable(&signal[pos..pos + n], &mut buf);
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
        adapter.process_variable(&signal[pos..pos + sub], &mut buf);
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
    adapter.process_variable(&[], &mut empty_out);
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
    adapter.process_variable(&signal, &mut out);
    assert!(adapter.needs_flush());
}

#[test]
fn needs_flush_cleared_when_drained() {
    let ir = synth_ir(32, 440.0, 10.0, 48000);
    let mut adapter = adapter_from_ir(&ir, 32);

    let signal: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut out = vec![0.0f32; 32];
    adapter.process_variable(&signal[..32], &mut out);
    assert!(!adapter.needs_flush());

    let mut out2 = vec![0.0f32; 32];
    adapter.process_variable(&signal[32..], &mut out2);
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
