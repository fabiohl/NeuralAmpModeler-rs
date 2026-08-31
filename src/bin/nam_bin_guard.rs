// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CLI entry point of the fail-closed AVX-512 absence certificate.
//!
//! Single Rust scanner reused by the integration guard
//! (`tests/avx512_guard.rs`) and by the isolated-build release wrapper
//! (`utils/verify_no_avx512_release.sh`). It scans an ELF object/executable
//! or GNU `ar` archive (`.rlib`) for EVEX-prefixed instructions and forbidden
//! AVX-512 symbols, and fails closed on any tool or format failure.
//!
//! Usage:
//!   nam_bin_guard scan `<artifact>` [--objdump PATH] [--nm PATH]
//!
//! Exit codes:
//!   0 — certificate passed (zero EVEX, zero forbidden symbols).
//!   1 — EVEX or forbidden symbol found, or any fail-closed tool/format error.
//!   2 — usage error.

use neural_amp_modeler_rs::testing::bin_guard::{
    EvexScanReport, ToolKind, resolve_llvm_tool, scan_for_evex, scan_symbols,
};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// AVX-512 kernel/symbol names that must never appear in the default build.
/// Defense-in-depth only; the authoritative proof is the EVEX opcode scan.
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

/// SHA-256 of the artifact bytes, logged before inspection so a human can
/// audit that the certificate refers to the freshly-built artifact.
fn sha256_of(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn usage() -> ! {
    eprintln!("Usage: nam_bin_guard scan <artifact> [--objdump PATH] [--nm PATH]");
    std::process::exit(2);
}

fn print_report(report: &EvexScanReport, sha: &str, symbol_matches: &[String]) {
    println!("artifact: {}", report.artifact.display());
    println!("sha256: {sha}");
    println!("sections_scanned: {}", report.sections_scanned);
    println!("instructions_scanned: {}", report.instructions_scanned);
    println!("members_scanned: {}", report.members_scanned);
    println!("bitcode_members: {}", report.bitcode_members);
    println!("metadata_members: {}", report.metadata_members);
    println!("evex_violations: {}", report.evex_violations.len());
    println!("forbidden_symbols: {}", symbol_matches.len());
    for violation in &report.evex_violations {
        let where_ = violation
            .member
            .as_deref()
            .map(|m| format!("{m}:"))
            .unwrap_or_default();
        println!("EVEX {where_}{}: {}", violation.section, violation.line);
    }
    for line in symbol_matches {
        println!("SYMBOL {line}");
    }
    println!(
        "result: {}",
        if report.is_clean() && symbol_matches.is_empty() {
            "PASS"
        } else {
            "FAIL"
        }
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("scan") {
        usage();
    }

    let artifact = args.get(1).map(PathBuf::from).unwrap_or_else(|| usage());
    let mut objdump_override: Option<PathBuf> = None;
    let mut nm_override: Option<PathBuf> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--objdump" => {
                let value = args
                    .get(i + 1)
                    .map(PathBuf::from)
                    .unwrap_or_else(|| usage());
                objdump_override = Some(value);
                i += 2;
            }
            "--nm" => {
                let value = args
                    .get(i + 1)
                    .map(PathBuf::from)
                    .unwrap_or_else(|| usage());
                nm_override = Some(value);
                i += 2;
            }
            other => {
                eprintln!("nam_bin_guard: unexpected argument '{other}'");
                usage();
            }
        }
    }

    // Mandatory inspection tools: absence fails closed before any scan.
    let objdump = match objdump_override.or_else(|| resolve_llvm_tool(ToolKind::LlvmObjdump).ok()) {
        Some(tool) => tool,
        None => {
            eprintln!(
                "nam_bin_guard: llvm-objdump required for binary certification but not found \
                 (rustc sysroot or PATH)"
            );
            return ExitCode::FAILURE;
        }
    };
    let nm = match nm_override.or_else(|| resolve_llvm_tool(ToolKind::LlvmNm).ok()) {
        Some(tool) => tool,
        None => {
            eprintln!(
                "nam_bin_guard: llvm-nm required for binary certification but not found \
                 (rustc sysroot or PATH)"
            );
            return ExitCode::FAILURE;
        }
    };

    let sha = match sha256_of(&artifact) {
        Ok(sha) => sha,
        Err(message) => {
            eprintln!("nam_bin_guard: {message}");
            return ExitCode::FAILURE;
        }
    };

    println!("objdump: {}", objdump.display());
    println!("nm: {}", nm.display());

    let report = match scan_for_evex(&objdump, &artifact) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("nam_bin_guard: fail-closed scan error: {error}");
            print_report(
                &EvexScanReport {
                    artifact,
                    ..EvexScanReport::default()
                },
                &sha,
                &[],
            );
            return ExitCode::FAILURE;
        }
    };

    let symbol_matches = match scan_symbols(&nm, &artifact, FORBIDDEN_SYMBOLS) {
        Ok(matches) => matches,
        Err(error) => {
            eprintln!("nam_bin_guard: fail-closed symbol scan error: {error}");
            print_report(&report, &sha, &[]);
            return ExitCode::FAILURE;
        }
    };

    let clean = report.is_clean() && symbol_matches.is_empty();
    print_report(&report, &sha, &symbol_matches);
    if !clean {
        eprintln!("nam_bin_guard: AVX-512 violation(s) detected — certificate FAILED.");
        return ExitCode::FAILURE;
    }
    eprintln!(
        "nam_bin_guard: certificate PASSED — {} executable section(s), {} instruction(s), {} member(s) inspected.",
        report.sections_scanned, report.instructions_scanned, report.members_scanned
    );
    ExitCode::SUCCESS
}
