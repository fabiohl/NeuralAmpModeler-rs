// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Binary surface guard: fail-closed proof of zero EVEX/AVX-512 machine code
//! in the default (non-`avx512`) build.
//!
//! Sprint 1 / F-ROB-01: the guard must prove — without false greens — that no
//! EVEX-encoded instruction exists in any executable section of the machine
//! code produced by the default build. The EVEX `0x62` encoding prefix is
//! detected from the raw instruction bytes emitted by `llvm-objdump -d`,
//! never from register names or mnemonics (which can miss AVX-512VL code that
//! uses only low `ymm0..15` registers and opmasks).
//!
//! Fail-closed contract: any tool failure, empty tool output, unreadable
//! artifact, unsupported format, or undecodable archive member aborts the
//! scan with an error instead of reporting a clean result.

#![cfg(not(feature = "avx512"))]

mod common;

use common::bin_fixtures as fixtures;
use neural_amp_modeler_rs::testing::bin_guard::{
    BinGuardError, EvexScanReport, ToolKind, resolve_llvm_tool, scan_for_evex, scan_symbols,
};
use std::env;
use std::path::Path;

/// AVX-512 kernel/symbol names that must never appear in the default build.
/// This is a defense-in-depth layer on top of the authoritative EVEX opcode
/// scan — it is not a substitute for it.
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

fn scan_clean_or_panic(objdump: &Path, path: &Path, label: &str) -> EvexScanReport {
    let report = scan_for_evex(objdump, path).unwrap_or_else(|e| {
        panic!(
            "avx512_guard: fail-closed scan error on {label} ({}): {e}",
            path.display()
        )
    });
    assert!(
        report.is_clean(),
        "avx512_guard: EVEX instruction leak in {label} ({}): {:#?}",
        path.display(),
        report.evex_violations
    );
    assert!(
        report.sections_scanned > 0,
        "avx512_guard: {label} reported zero executable sections — scan is vacuous"
    );
    assert!(
        report.instructions_scanned > 0,
        "avx512_guard: {label} reported zero instructions — scan is vacuous"
    );
    report
}

/// Primary certification scan: the linked test binary is the real machine-code
/// surface of the build under test. The scanner must run against it cleanly
/// and report a non-vacuous number of inspected sections and instructions.
#[test]
fn test_no_avx512_in_linked_test_binary() {
    let objdump = resolve_llvm_tool(ToolKind::LlvmObjdump)
        .expect("avx512_guard: llvm-objdump required for binary certification");
    let nm = resolve_llvm_tool(ToolKind::LlvmNm)
        .expect("avx512_guard: llvm-nm required for binary certification");
    let current_exe = env::current_exe().expect("avx512_guard: current_exe unavailable");

    let report = scan_clean_or_panic(&objdump, &current_exe, "linked test binary");
    eprintln!(
        "avx512_guard: certified clean — {} ({} executable sections, {} instructions).",
        current_exe.display(),
        report.sections_scanned,
        report.instructions_scanned
    );

    let sym_violations = scan_symbols(&nm, &current_exe, FORBIDDEN_SYMBOLS).unwrap_or_else(|e| {
        panic!(
            "avx512_guard: nm scan failed on {}: {e}",
            current_exe.display()
        )
    });
    assert!(
        sym_violations.is_empty(),
        "AVX-512 symbol leak in linked test binary {}: {sym_violations:?}",
        current_exe.display()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// T1.3 — Synthetic fixture mutation battery
// ═══════════════════════════════════════════════════════════════════════════

/// Positive fixture: EVEX VL256 with a low `ymm0..15` register and zeroing
/// mask — the exact evasion family the old register-regex guard could miss.
#[test]
fn guard_rejects_evex_vl256_ymm_low_registers() {
    let path = fixtures::write_temp(
        &fixtures::mini_elf(fixtures::EVEX_VL256_MASK_ZERO),
        "evex-vl256.o",
    );
    let report = scan_for_evex(&fixtures::sysroot_objdump(), &path).expect("scan must run");
    assert!(
        !report.is_clean(),
        "guard must reject EVEX VL256 with low ymm register"
    );
    assert_eq!(report.evex_violations.len(), 1);
    assert_eq!(report.sections_scanned, 1);
    fixtures::cleanup(&path);
}

/// Positive fixture: EVEX VL256 with an opmask (no ZMM, no high registers).
#[test]
fn guard_rejects_evex_opmask() {
    let path = fixtures::write_temp(
        &fixtures::mini_elf(fixtures::EVEX_VL256_MASK),
        "evex-opmask.o",
    );
    let report = scan_for_evex(&fixtures::sysroot_objdump(), &path).expect("scan must run");
    assert!(!report.is_clean(), "guard must reject EVEX with opmask");
    assert_eq!(report.evex_violations.len(), 1);
    fixtures::cleanup(&path);
}

/// Positive fixture: EVEX ZMM0 (`zmm0..15`).
#[test]
fn guard_rejects_evex_zmm0() {
    let path = fixtures::write_temp(&fixtures::mini_elf(fixtures::EVEX_ZMM0), "evex-zmm0.o");
    let report = scan_for_evex(&fixtures::sysroot_objdump(), &path).expect("scan must run");
    assert!(!report.is_clean(), "guard must reject EVEX ZMM");
    assert_eq!(report.evex_violations.len(), 1);
    fixtures::cleanup(&path);
}

/// Positive fixture: EVEX with high `zmm16..31` registers.
#[test]
fn guard_rejects_evex_zmm_high_registers() {
    let path = fixtures::write_temp(&fixtures::mini_elf(fixtures::EVEX_ZMM16_31), "evex-zmm16.o");
    let report = scan_for_evex(&fixtures::sysroot_objdump(), &path).expect("scan must run");
    assert!(
        !report.is_clean(),
        "guard must reject EVEX with zmm16..31 registers"
    );
    assert_eq!(report.evex_violations.len(), 1);
    fixtures::cleanup(&path);
}

/// Negative fixture: pure x86-64-v3 (AVX2/FMA/SSE) code must pass cleanly.
#[test]
fn guard_accepts_clean_x86_64_v3_elf() {
    let mut text = Vec::new();
    text.extend_from_slice(fixtures::CLEAN_VEX);
    text.extend_from_slice(fixtures::CLEAN_SSE);
    let path = fixtures::write_temp(&fixtures::mini_elf(&text), "clean-avx2.o");
    let report = scan_clean_or_panic(
        &fixtures::sysroot_objdump(),
        &path,
        "clean x86-64-v3 fixture",
    );
    assert_eq!(report.instructions_scanned, 2);
    fixtures::cleanup(&path);
}

/// Positive archive fixture: an EVEX member hidden inside an `.rlib`-style
/// archive must be caught, proving every archive member is scanned.
#[test]
fn guard_rejects_evex_inside_archive() {
    let archive = fixtures::mini_archive(&[
        ("clean.o", &fixtures::mini_elf(fixtures::CLEAN_VEX)),
        ("evil.o", &fixtures::mini_elf(fixtures::EVEX_VL256_MASK)),
    ]);
    let path = fixtures::write_temp(&archive, "evil.rlib");
    let report = scan_for_evex(&fixtures::sysroot_objdump(), &path).expect("scan must run");
    assert!(!report.is_clean(), "guard must reject EVEX inside archive");
    assert_eq!(report.members_scanned, 2);
    assert_eq!(report.evex_violations.len(), 1);
    assert_eq!(
        report.evex_violations[0].member.as_deref(),
        Some("evil.o"),
        "violation must be attributed to the correct archive member"
    );
    fixtures::cleanup(&path);
}

/// Negative archive fixture: a clean `.rlib`-style archive must pass.
#[test]
fn guard_accepts_clean_archive() {
    let archive = fixtures::mini_archive(&[
        ("clean-a.o", &fixtures::mini_elf(fixtures::CLEAN_VEX)),
        ("clean-b.o", &fixtures::mini_elf(fixtures::CLEAN_SSE)),
    ]);
    let path = fixtures::write_temp(&archive, "clean.rlib");
    let report = scan_clean_or_panic(&fixtures::sysroot_objdump(), &path, "clean archive fixture");
    assert_eq!(report.members_scanned, 2);
    fixtures::cleanup(&path);
}

/// Failure injection: a corrupted (non-binary) file must fail closed.
#[test]
fn guard_fails_closed_on_corrupted_artifact() {
    let path = fixtures::write_temp(b"garbage bytes that are not a binary", "corrupt.bin");
    let err = scan_for_evex(&fixtures::sysroot_objdump(), &path).expect_err("must fail");
    assert!(matches!(err, BinGuardError::UnsupportedFormat { .. }));
    fixtures::cleanup(&path);
}

/// Failure injection: an empty artifact must fail closed.
#[test]
fn guard_fails_closed_on_empty_artifact() {
    let path = fixtures::write_temp(b"", "empty.bin");
    let err = scan_for_evex(&fixtures::sysroot_objdump(), &path).expect_err("must fail");
    assert!(matches!(err, BinGuardError::EmptyInput { .. }));
    fixtures::cleanup(&path);
}

/// Failure injection: a missing artifact must fail closed.
#[test]
fn guard_fails_closed_on_missing_artifact() {
    let missing = std::env::temp_dir().join("nam-avx512-guard-does-not-exist.o");
    let err = scan_for_evex(&fixtures::sysroot_objdump(), &missing).expect_err("must fail");
    assert!(matches!(err, BinGuardError::ReadFailed { .. }));
}

/// Failure injection: a missing inspection tool must fail closed.
#[test]
fn guard_fails_closed_on_missing_tool() {
    let path = fixtures::write_temp(&fixtures::mini_elf(fixtures::CLEAN_VEX), "no-tool.o");
    let err = scan_for_evex(Path::new("/definitely/not/an/objdump"), &path).expect_err("must fail");
    assert!(matches!(err, BinGuardError::ReadFailed { .. }));
    fixtures::cleanup(&path);
}

/// Failure injection: an inspection tool exiting non-zero must fail closed
/// instead of being treated as a clean scan.
#[test]
fn guard_fails_closed_on_tool_exit_nonzero() {
    let path = fixtures::write_temp(&fixtures::mini_elf(fixtures::CLEAN_VEX), "tool-fail.o");
    let fake = fixtures::write_fake_tool("#!/bin/sh\nexit 1\n", "exit1");
    let err = scan_for_evex(&fake, &path).expect_err("must fail");
    assert!(matches!(err, BinGuardError::ToolFailed { .. }));
    fixtures::cleanup(&path);
    fixtures::cleanup(&fake);
}

/// Failure injection: an inspection tool producing empty output must fail
/// closed instead of being treated as "zero instructions".
#[test]
fn guard_fails_closed_on_tool_empty_output() {
    let path = fixtures::write_temp(&fixtures::mini_elf(fixtures::CLEAN_VEX), "empty-out.o");
    let fake = fixtures::write_fake_tool("#!/bin/sh\nexit 0\n", "emptyout");
    let err = scan_for_evex(&fake, &path).expect_err("must fail");
    assert!(matches!(err, BinGuardError::EmptyOutput { .. }));
    fixtures::cleanup(&path);
    fixtures::cleanup(&fake);
}

/// Failure injection: a thin-LTO `.rlib` whose members are LLVM bitcode cannot
/// certify machine-code absence from IR — the scan must fail closed.
#[test]
fn guard_fails_closed_on_bitcode_only_archive() {
    let bitcode = b"BC\xc0\xde\x35\x14\x00\x00\x00\x00\x00\x00";
    let archive = fixtures::mini_archive(&[("lib.rmeta", b"meta"), ("code.rcgu.o", bitcode)]);
    let path = fixtures::write_temp(&archive, "thinlto.rlib");
    let err = scan_for_evex(&fixtures::sysroot_objdump(), &path).expect_err("must fail");
    assert!(matches!(err, BinGuardError::NoDisassemblableMembers { .. }));
    fixtures::cleanup(&path);
}

/// Failure injection: an archive with an unrecognized member must fail closed.
#[test]
fn guard_fails_closed_on_unknown_archive_member() {
    let archive = fixtures::mini_archive(&[("junk.bin", b"definitely not an object file")]);
    let path = fixtures::write_temp(&archive, "junk.rlib");
    let err = scan_for_evex(&fixtures::sysroot_objdump(), &path).expect_err("must fail");
    assert!(matches!(
        err,
        BinGuardError::UnsupportedArchiveMember { .. }
    ));
    fixtures::cleanup(&path);
}

/// Defensive symbol scan: the nm fallback must fail closed on tool errors.
#[test]
fn guard_symbol_scan_fails_closed_on_tool_error() {
    let path = fixtures::write_temp(&fixtures::mini_elf(fixtures::CLEAN_VEX), "symscan-fail.o");
    let fake = fixtures::write_fake_tool("#!/bin/sh\nexit 3\n", "nm-fail");
    let err = scan_symbols(&fake, &path, FORBIDDEN_SYMBOLS).expect_err("must fail");
    assert!(matches!(err, BinGuardError::ToolFailed { .. }));
    fixtures::cleanup(&path);
    fixtures::cleanup(&fake);
}
