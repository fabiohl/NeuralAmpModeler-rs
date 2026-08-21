// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests for the environment probe (R-09) — fixture-driven CPU classification
//! plus a live-probe smoke test mirroring `tests-long.sh:825-826`.

use super::*;
use crate::testing::fixtures::fixture_dir;

/// Reads a cpuinfo fixture from `tests/fixtures/qa/`.
fn cpuinfo_fixture(name: &str) -> String {
    std::fs::read_to_string(fixture_dir().join("qa").join(name))
        .expect("cpuinfo fixture must exist under tests/fixtures/qa/")
}

/// Acceptance (S2.T4): the Ryzen-like fixture classifies as the canonical
/// v3 string, with model and physical cores extracted.
#[test]
fn ryzen_like_fixture_classifies_x86_64_v3() {
    let cpu = parse_cpuinfo(&cpuinfo_fixture("cpuinfo_ryzen_v3.txt"));
    assert_eq!(cpu.effective_isa, ISA_X86_64_V3);
    assert_eq!(cpu.model_name, "AMD Ryzen 9 5900X 12-Core Processor");
    assert_eq!(cpu.physical_cores, 12);
}

#[test]
fn avx512_fixture_classifies_avx512() {
    let cpu = parse_cpuinfo(&cpuinfo_fixture("cpuinfo_avx512.txt"));
    assert_eq!(cpu.effective_isa, ISA_AVX512);
    assert_eq!(cpu.model_name, "AMD Ryzen 9 7950X 16-Core Processor");
    assert_eq!(cpu.physical_cores, 16);
}

#[test]
fn avx2_incomplete_fixture_classifies_incomplete() {
    let cpu = parse_cpuinfo(&cpuinfo_fixture("cpuinfo_avx2_incomplete.txt"));
    assert_eq!(cpu.effective_isa, ISA_AVX2_INCOMPLETE);
    assert_eq!(cpu.physical_cores, 4);
}

#[test]
fn base_fixture_classifies_base() {
    let cpu = parse_cpuinfo(&cpuinfo_fixture("cpuinfo_base.txt"));
    assert_eq!(cpu.effective_isa, ISA_X86_64_BASE);
    assert_eq!(
        cpu.model_name,
        "Intel(R) Core(TM)2 Duo CPU     E8400  @ 3.00GHz"
    );
    assert_eq!(cpu.physical_cores, 2);
}

/// `avx512f` and `avx512vl` together classify as AVX-512 over v3 set.
#[test]
fn avx512f_and_vl_takes_precedence_over_v3_set() {
    let flags = "flags : fpu sse avx avx2 bmi1 bmi2 f16c fma abm movbe avx512f avx512vl";
    assert_eq!(classify_isa(flags), ISA_AVX512);
}

/// `avx512f` without `avx512vl` does not classify as AVX-512; falls back to v3 / incomplete.
#[test]
fn avx512f_without_vl_falls_back_to_v3() {
    let flags_v3 = "flags : fpu sse avx avx2 bmi1 bmi2 f16c fma abm movbe avx512f";
    assert_eq!(classify_isa(flags_v3), ISA_X86_64_V3);

    let flags_incomplete = "flags : fpu sse avx avx2 bmi1 f16c fma movbe avx512f";
    assert_eq!(classify_isa(flags_incomplete), ISA_AVX2_INCOMPLETE);
}

/// Both spellings of LZCNT (`lzcnt` Intel / `abm` AMD) satisfy the v3 gate.
#[test]
fn v3_gate_accepts_lzcnt_or_abm() {
    let base = "avx avx2 bmi1 bmi2 f16c fma movbe";
    assert_eq!(
        classify_isa(&format!("flags : {base} lzcnt")),
        ISA_X86_64_V3
    );
    assert_eq!(classify_isa(&format!("flags : {base} abm")), ISA_X86_64_V3);
}

/// Missing any single v3 flag demotes below v3; without `avx2` at all the
/// result is base, otherwise the incomplete label (bash `elif avx2`).
#[test]
fn v3_gate_requires_the_full_feature_set() {
    for missing in ["avx", "bmi1", "bmi2", "f16c", "fma", "movbe", "abm"] {
        let flags: Vec<&str> = "avx avx2 bmi1 bmi2 f16c fma abm movbe"
            .split_whitespace()
            .filter(|flag| *flag != missing)
            .collect();
        let line = format!("flags : {}", flags.join(" "));
        assert_eq!(
            classify_isa(&line),
            ISA_AVX2_INCOMPLETE,
            "missing {missing} must demote to AVX2 incomplete"
        );
    }
    let no_avx2: Vec<&str> = "avx bmi1 bmi2 f16c fma abm movbe"
        .split_whitespace()
        .collect();
    assert_eq!(
        classify_isa(&format!("flags : {}", no_avx2.join(" "))),
        ISA_X86_64_BASE,
        "without avx2 the classifier must fall back to base"
    );
}

/// Matching is whole-word (`grep -w`): a substring like `noavx2` is not a
/// flag, and a line with no flags classifies as base.
#[test]
fn classification_is_whole_word_and_fail_safe() {
    assert_eq!(classify_isa("flags : noavx2 avx512f_sim"), ISA_X86_64_BASE);
    assert_eq!(classify_isa(""), ISA_X86_64_BASE);
    assert_eq!(classify_isa("garbage without flags"), ISA_X86_64_BASE);
}

/// First `model name` / `cpu cores` / `flags` lines win; absent fields stay
/// empty/0 so `EnvProbe` can apply its `unknown`/`nproc` fallbacks.
#[test]
fn parse_cpuinfo_uses_first_match_and_defaults() {
    let text = "model name      : First CPU\n\
                flags           : avx\n\
                cpu cores       : 8\n\
                model name      : Second CPU\n\
                flags           : avx2\n\
                cpu cores       : 16\n";
    let cpu = parse_cpuinfo(text);
    assert_eq!(cpu.model_name, "First CPU");
    assert_eq!(
        cpu.effective_isa, ISA_AVX2_INCOMPLETE,
        "first flags line wins"
    );
    assert_eq!(cpu.physical_cores, 8);

    let bare = parse_cpuinfo("processor       : 0\nwp              : yes\n");
    assert_eq!(bare.model_name, "");
    assert_eq!(bare.effective_isa, ISA_X86_64_BASE);
    assert_eq!(bare.physical_cores, 0);
}

/// `probe_from_cpuinfo` keeps the CPU portion fixture-driven and still fills
/// the live toolchain/environment fields — the acceptance shape of the
/// future fingerprint.
#[test]
fn probe_from_cpuinfo_combines_fixture_cpu_with_live_toolchain() {
    let probe = EnvProbe::probe_from_cpuinfo(&cpuinfo_fixture("cpuinfo_ryzen_v3.txt"));
    assert_eq!(probe.effective_isa, ISA_X86_64_V3);
    assert_eq!(probe.cpu_model, "AMD Ryzen 9 5900X 12-Core Processor");
    assert_eq!(probe.physical_cores, 12);
    assert!(!probe.rustc_version.is_empty(), "rustc must be available");
    assert!(
        !probe.host_triple.is_empty(),
        "host triple must be available"
    );
}

/// Live probe smoke test — the Rust equivalent of the bash
/// `detect_isa returns a known ISA string` assertions (`tests-long.sh:825-826`).
#[test]
fn live_probe_yields_canonical_isa_and_toolchain() {
    let probe = EnvProbe::probe();
    assert!(
        matches!(
            probe.effective_isa,
            ISA_AVX512 | ISA_X86_64_V3 | ISA_AVX2_INCOMPLETE | ISA_X86_64_BASE
        ),
        "effective_isa must be one of the canonical strings, got {}",
        probe.effective_isa
    );
    assert!(!probe.cpu_model.is_empty());
    assert!(!probe.rustc_version.is_empty());
    assert!(!probe.host_triple.is_empty());
    assert!(probe.physical_cores >= 1);
}
