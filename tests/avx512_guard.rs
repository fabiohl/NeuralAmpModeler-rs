// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Binary surface guard: verifies zero AVX-512 symbols and zero EVEX/ZMM
//! instructions in default (non-avx512) compiled artifacts.
//!
//! Enforces PO compliance: the default shipping artifact must target strictly
//! the x86-64-v3 baseline without emitting AVX-512 kernels or symbols.

#![cfg(not(feature = "avx512"))]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const FORBIDDEN_SYMBOLS: &[&str] = &[
    "gemv_4gate_avx512",
    "dot_product_4x_f32_avx512",
    "Avx512Math",
    "process_sample_avx512",
    "process_avx512",
    "hard_swish_slice_avx512",
    "leaky_hard_tanh_slice_avx512",
    "simd_relu_avx512",
    "relu_slice_avx512",
    "simd_silu_avx512",
    "silu_slice_avx512",
    "simd_silu_poly_avx512",
    "silu_poly_slice_avx512",
    "simd_tanh_sigmoid_dual_avx512",
];

const FORBIDDEN_DISASM_PATTERNS: &[&str] = &[
    "vpdpbusd", "%zmm", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8",
    "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18",
    "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28",
    "zmm29", "zmm30", "zmm31", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21", "xmm22",
    "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29", "xmm30", "xmm31", "ymm16",
    "ymm17", "ymm18", "ymm19", "ymm20", "ymm21", "ymm22", "ymm23", "ymm24", "ymm25", "ymm26",
    "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
];

fn resolve_rustc_tool(tool_name: &str) -> Option<PathBuf> {
    let sysroot_out = Command::new("rustc")
        .arg("--print")
        .arg("sysroot")
        .output()
        .ok()?;
    if !sysroot_out.status.success() {
        return None;
    }
    let sysroot = String::from_utf8_lossy(&sysroot_out.stdout)
        .trim()
        .to_string();
    let host_out = Command::new("rustc").arg("-vV").output().ok()?;
    let host_str = String::from_utf8_lossy(&host_out.stdout);
    for line in host_str.lines() {
        if let Some(host) = line.strip_prefix("host: ") {
            let candidate = PathBuf::from(&sysroot)
                .join("lib/rustlib")
                .join(host.trim())
                .join("bin")
                .join(tool_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn resolve_nm_tool() -> Option<PathBuf> {
    if let Some(tool) = resolve_rustc_tool("llvm-nm") {
        return Some(tool);
    }

    for tool in ["llvm-nm", "llvm-nm-21", "nm"] {
        let is_available = Command::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if is_available {
            return Some(PathBuf::from(tool));
        }
    }
    None
}

fn resolve_objdump_tool() -> Option<PathBuf> {
    if let Some(tool) = resolve_rustc_tool("llvm-objdump") {
        return Some(tool);
    }

    for tool in ["llvm-objdump", "llvm-objdump-21", "objdump"] {
        let is_available = Command::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if is_available {
            return Some(PathBuf::from(tool));
        }
    }
    None
}

fn check_symbols_clean(nm: &Path, target: &Path) -> Vec<String> {
    let output = Command::new(nm)
        .arg("--demangle")
        .arg(target)
        .output()
        .or_else(|_| Command::new(nm).arg("-C").arg(target).output());

    let mut violations = Vec::new();
    if let Ok(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines() {
            for forbidden in FORBIDDEN_SYMBOLS {
                if line.contains(forbidden) {
                    violations.push(line.to_string());
                    break;
                }
            }
        }
    }
    violations
}

fn check_disasm_clean(objdump: &Path, target: &Path) -> Vec<String> {
    let output = Command::new(objdump).arg("-d").arg(target).output();

    let mut violations = Vec::new();
    if let Ok(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines() {
            for forbidden in FORBIDDEN_DISASM_PATTERNS {
                if line.contains(forbidden) {
                    violations.push(line.to_string());
                    break;
                }
            }
        }
    }
    violations
}

#[test]
fn test_no_avx512_symbols_or_instructions_in_default_build() {
    #[cfg(feature = "avx512")]
    {
        // Explicit opt-in build: AVX-512 allowed
    }

    #[cfg(not(feature = "avx512"))]
    {
        let nm = resolve_nm_tool()
            .expect("avx512_guard: nm/llvm-nm tool required for binary certification");
        let current_exe = env::current_exe().expect("avx512_guard: current_exe unavailable");

        // 1. Scan current test executable
        let sym_violations = check_symbols_clean(&nm, &current_exe);
        assert!(
            sym_violations.is_empty(),
            "AVX-512 symbol leak in test binary {}: {sym_violations:?}",
            current_exe.display()
        );

        if let Some(objdump) = resolve_objdump_tool() {
            let disasm_violations = check_disasm_clean(&objdump, &current_exe);
            assert!(
                disasm_violations.is_empty(),
                "EVEX/ZMM instruction leak in test binary {}: {disasm_violations:?}",
                current_exe.display()
            );
        }

        // 2. If release rlib exists, scan it as well
        let rlib = Path::new("target/release/libneural_amp_modeler_rs.rlib");
        if rlib.exists() {
            let rlib_sym_violations = check_symbols_clean(&nm, rlib);
            assert!(
                rlib_sym_violations.is_empty(),
                "AVX-512 symbol leak in release rlib {}: {rlib_sym_violations:?}",
                rlib.display()
            );

            if let Some(objdump) = resolve_objdump_tool() {
                let rlib_disasm_violations = check_disasm_clean(&objdump, rlib);
                assert!(
                    rlib_disasm_violations.is_empty(),
                    "EVEX/ZMM instruction leak in release rlib {}: {rlib_disasm_violations:?}",
                    rlib.display()
                );
            }
        }
    }
}
