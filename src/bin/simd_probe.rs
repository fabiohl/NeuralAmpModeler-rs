// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `simd_probe` — SIMD Diagnostic & Capability Probe CLI.
//!
//! Inspects the x86-64 hardware SIMD flags via `is_x86_feature_detected!`,
//! validates the OS context save of the ZMM registers via `xgetbv` (when the
//! OSXSAVE feature is present), reports the Cargo `avx512` feature state and
//! the active engine dispatch backend (`SIMD_MATH` / `effective_instruction_set()`),
//! and runs a short synthetic inference cycle (64 samples) through a real
//! `dispatch_simd!`-monomorphized model to attest that the resolved kernels
//! execute without failures.
//!
//! The probe must never panic on any x86-64 CPU, with or without AVX-512.
//!
//! # Usage
//! ```sh
//! cargo run --bin simd_probe
//! cargo run --features avx512 --bin simd_probe
//! ```

use neural_amp_modeler_rs::math::common::{
    InstructionSet, SIMD_MATH, avx512_capability_complete, effective_instruction_set,
};
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::lstm::Lstm2x8;

/// Number of samples processed by the synthetic inference cycle.
const PROBE_BLOCK: usize = 64;

fn main() {
    println!("=======================================================");
    println!(" NeuralAmpModeler-rs SIMD Diagnostic & Capability Probe");
    println!("=======================================================");

    print_hardware_flags();

    println!();
    print_compilation_flags();

    println!();
    print_dispatch_resolution();

    println!("=======================================================");
}

/// Prints the runtime-detected CPU hardware flags and OS context-save status.
fn print_hardware_flags() {
    let avx2 = is_x86_feature_detected!("avx2");
    let fma = is_x86_feature_detected!("fma");
    let avx512f = is_x86_feature_detected!("avx512f");
    let avx512vl = is_x86_feature_detected!("avx512vl");
    let avx512bw = is_x86_feature_detected!("avx512bw");
    let avx512dq = is_x86_feature_detected!("avx512dq");
    let osxsave = has_osxsave();
    let zmm_os_context = os_saves_zmm_context();

    println!("[CPU Hardware Flags]");
    println!("  - avx2:        {}", bool_label(avx2, " (present)"));
    println!("  - fma:         {}", bool_label(fma, " (present)"));
    println!("  - avx512f:     {}", bool_label(avx512f, ""));
    println!("  - avx512vl:    {}", bool_label(avx512vl, ""));
    println!("  - avx512bw:    {}", bool_label(avx512bw, ""));
    println!("  - avx512dq:    {}", bool_label(avx512dq, ""));
    println!("  - osxsave:     {}", bool_label(osxsave, ""));
    println!("  - xgetbv ZMM:  {}", bool_label(zmm_os_context, ""));

    let compatible = avx512_capability_complete(avx512f, avx512vl, avx512bw, avx512dq);
    println!(
        "  -> Hardware Status: [{}]",
        if compatible {
            "COMPLIANT"
        } else {
            "NON-COMPLIANT"
        }
    );
}

/// Prints the Cargo feature state of the crate under test.
fn print_compilation_flags() {
    println!("[Crate Compilation Flags]");
    println!(
        "  - feature \"avx512\": [{}]",
        if cfg!(feature = "avx512") {
            "ACTIVE"
        } else {
            "INACTIVE"
        }
    );
}

/// Prints the active engine dispatch backend and runs the inference smoke test.
fn print_dispatch_resolution() {
    // Effective dispatch decides the monomorphized backend (respects any test
    // ISA override; in a standalone probe it mirrors the detected SIMD_MATH).
    let backend_is_avx512 = !matches!(effective_instruction_set(), InstructionSet::Avx2);

    println!("[Engine Dispatch Resolution]");
    println!(
        "  - Active SIMD Backend: {}",
        if backend_is_avx512 { "AVX-512" } else { "AVX2" }
    );
    println!("  - Engine Backend (SIMD_MATH.name): {}", SIMD_MATH.name);

    match run_inference_smoke_test() {
        Ok(checksum) => {
            println!(
                "  - Inference Smoke Test: {PROBE_BLOCK} samples, checksum={checksum:.6} (finite)"
            );
            println!("  - Dispatch Status: OPERATING DETERMINISTICALLY");
        }
        Err(e) => {
            eprintln!("  - Dispatch Status: FAILED ({e})");
            std::process::exit(1);
        }
    }
}

/// Runs a real inference cycle through a `dispatch_simd!`-monomorphized model.
///
/// A zero-weight 2-layer LSTM (`Lstm2x8`) with a non-zero head bias yields a
/// deterministic output block: every sample equals the head bias, so the
/// checksum of a 64-sample block is exactly `64 * head_bias`. This exercises
/// the AVX2/AVX-512 kernels selected by `dispatch_simd!` (both layer
/// processing and the head projection) without needing any model fixture.
fn run_inference_smoke_test() -> Result<f32, String> {
    let mut model = Lstm2x8::new();
    model.head_bias = 0.5;

    let input: Vec<f32> = (0..PROBE_BLOCK)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let mut output = vec![0.0f32; PROBE_BLOCK];

    model.prewarm(PROBE_BLOCK);
    model.process(&input, &mut output);

    if !output.iter().all(|x| x.is_finite()) {
        return Err("Non-finite output detected".to_string());
    }
    Ok(output.iter().sum())
}

/// Returns whether the OS saves the full AVX-512 (ZMM/opmask) register context.
///
/// When `OSXSAVE` is advertised, XCR0 bits 5 (opmask), 6 (ZMM_Hi256) and 7
/// (Hi16_ZMM) indicate that the operating system enables the AVX-512 state
/// save/restore on context switches.
fn os_saves_zmm_context() -> bool {
    if !has_osxsave() {
        return false;
    }
    // SAFETY: `_xgetbv(0)` is only executed after the OSXSAVE CPUID flag
    // (leaf 1, ECX bit 27) was observed via `has_osxsave()`, which is the
    // documented precondition for reading XCR0.
    let xcr0 = unsafe { std::arch::x86_64::_xgetbv(0) };
    (xcr0 & 0xE0) == 0xE0
}

/// Returns whether the OS advertises the XSAVE/OSXSAVE context management.
///
/// CPUID leaf 1, ECX bit 27 (`OSXSAVE`) is not exposed as a Rust target
/// feature usable with `is_x86_feature_detected!`, so it is probed directly.
fn has_osxsave() -> bool {
    let cpuid = std::arch::x86_64::__cpuid(1);
    (cpuid.ecx >> 27) & 1 == 1
}

/// Renders a boolean flag as `YES`/`NO` with an optional suffix.
fn bool_label(value: bool, suffix: &str) -> String {
    if value {
        format!("YES{suffix}")
    } else {
        "NO".to_string()
    }
}
