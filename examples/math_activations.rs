// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! # SIMD Neural Activation Kernels Example (`math_activations`)
//!
//! Demonstrates SIMD-accelerated activation kernels (`tanh`, `sigmoid`, `relu`, `prelu`, `softsign`),
//! comparing exact `ActivationPrecision::Standard` polynomial paths vs `ActivationPrecision::Fast` Padé/minimax
//! approximations, and measuring throughput and numerical error budgets.
//!
//! ## Overview
//!
//! - **`tanh_slice`**: Vectorized hyperbolic tangent. `Standard` (degree-6 Taylor polynomial, error ≤ 2.4e-7)
//!   vs `Fast` (Padé [5,4] rational approximant with hardware div, max error ~2.32e-3).
//! - **`sigmoid_slice`**: Vectorized logistic sigmoid. `Standard` (high-fidelity polynomial, error ≤ 2.1e-7)
//!   vs `Fast` (Lawson minimax degree-17 polynomial, error ~4.09e-4).
//! - **Zero-Branch SIMD Execution**: Operations rely strictly on SIMD register operations and masks.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example math_activations
//! ```

use neural_amp_modeler_rs::math::activations::{
    ActivationPrecision, prelu_slice, relu_slice, set_thread_local_activation_precision,
    sigmoid_slice, softsign_slice, tanh_slice,
};
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("  NeuralAmpModeler-rs — SIMD Math Activations & Fidelity   ");
    println!("============================================================");

    // Generate test sweep inputs in range [-4.0, 4.0]
    let num_samples = 128;
    let inputs: Vec<f32> = (0..num_samples)
        .map(|i| -4.0 + (i as f32 / (num_samples - 1) as f32) * 8.0)
        .collect();

    // 1. Tanh Precision Comparison (Standard vs Fast vs f32::tanh reference)
    println!("\n[1/4] Hyperbolic Tangent (Tanh) Fidelity Sweep");
    let mut tanh_std = inputs.clone();
    let mut tanh_fast = inputs.clone();

    // Standard mode (exact-grade, default)
    {
        let _guard = set_thread_local_activation_precision(Some(ActivationPrecision::Standard));
        tanh_slice(&mut tanh_std);
    }

    // Fast mode (Padé [5,4] approximation)
    {
        let _guard = set_thread_local_activation_precision(Some(ActivationPrecision::Fast));
        tanh_slice(&mut tanh_fast);
    }

    let mut max_err_std_tanh = 0.0_f32;
    let mut max_err_fast_tanh = 0.0_f32;

    for i in 0..num_samples {
        let ref_val = inputs[i].tanh();
        let err_std = (tanh_std[i] - ref_val).abs();
        let err_fast = (tanh_fast[i] - ref_val).abs();
        if err_std > max_err_std_tanh {
            max_err_std_tanh = err_std;
        }
        if err_fast > max_err_fast_tanh {
            max_err_fast_tanh = err_fast;
        }
    }

    println!(
        "  Standard Mode Max Absolute Error : {:.8e} (exact-grade, ≤ 2.4e-7 spec)",
        max_err_std_tanh
    );
    println!(
        "  Fast Mode Max Absolute Error     : {:.8e} (Padé [5,4], ≤ 2.4e-3 spec)",
        max_err_fast_tanh
    );

    // 2. Sigmoid Precision Comparison (Standard vs Fast vs exp reference)
    println!("\n[2/4] Logistic Sigmoid (Sigmoid) Fidelity Sweep");
    let mut sig_std = inputs.clone();
    let mut sig_fast = inputs.clone();

    {
        let _guard = set_thread_local_activation_precision(Some(ActivationPrecision::Standard));
        sigmoid_slice(&mut sig_std);
    }

    {
        let _guard = set_thread_local_activation_precision(Some(ActivationPrecision::Fast));
        sigmoid_slice(&mut sig_fast);
    }

    let ref_sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let mut max_err_std_sig = 0.0_f32;
    let mut max_err_fast_sig = 0.0_f32;

    for i in 0..num_samples {
        let ref_val = ref_sigmoid(inputs[i]);
        let err_std = (sig_std[i] - ref_val).abs();
        let err_fast = (sig_fast[i] - ref_val).abs();
        if err_std > max_err_std_sig {
            max_err_std_sig = err_std;
        }
        if err_fast > max_err_fast_sig {
            max_err_fast_sig = err_fast;
        }
    }

    println!(
        "  Standard Mode Max Absolute Error : {:.8e} (polynomial exp, ≤ 2.1e-7 spec)",
        max_err_std_sig
    );
    println!(
        "  Fast Mode Max Absolute Error     : {:.8e} (Minimax deg-17, ≤ 4.1e-4 spec)",
        max_err_fast_sig
    );

    // 3. Other SIMD Neural Activations
    println!("\n[3/4] Vectorized Neural Activations (ReLU, PReLU, Softsign)");
    let mut relu_buf = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    relu_slice(&mut relu_buf);
    println!("  ReLU([-2..2])       : {:?}", relu_buf);

    let mut prelu_buf = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let slopes = vec![0.1; 7];
    prelu_slice(&mut prelu_buf, &slopes);
    println!("  PReLU(slope=0.1)   : {:?}", prelu_buf);

    let mut softsign_buf = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    softsign_slice(&mut softsign_buf);
    println!("  Softsign([-2..2])   : {:?}", softsign_buf);

    // 4. SIMD Throughput Benchmark (100k samples x 1000 iterations)
    println!("\n[4/4] SIMD Throughput Benchmark");
    let bench_samples = 65_536; // 64k aligned buffer
    let iterations = 2_000;
    let mut bench_buf = vec![0.5_f32; bench_samples];

    let start = Instant::now();
    for _ in 0..iterations {
        tanh_slice(&mut bench_buf);
    }
    let elapsed = start.elapsed();
    let total_evals = bench_samples as f64 * iterations as f64;
    let throughput_mops = (total_evals / elapsed.as_secs_f64()) / 1_000_000.0;

    println!(
        "  Evaluated {} tanh samples in {:.2?}",
        total_evals as usize, elapsed
    );
    println!("  SIMD Throughput : {:.2} MSamples/sec", throughput_mops);

    println!("\n[Status] Math activations demonstration completed successfully.");
}
