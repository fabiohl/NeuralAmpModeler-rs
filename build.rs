// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//
// Fix + permanent regression guard for an ELF symbol-interposition hang —
// see docs/postmortem-libm-symbol-interposition.md and
// .cargo/hide-libm-shadow.map for the full, GDB-verified root-cause
// analysis. Some part of the dependency graph pulls in libm-shaped
// fallback symbols (`log10f`, `atan2f`, `acosf`, ...) that end up compiled
// into the final binary with GLOBAL (exported) visibility and the same C
// name as the real functions in the system's `libm.so.6`. Under standard
// ELF symbol interposition rules, `ld.so` then resolves calls to those
// names back to our own binary instead of the real dynamic library,
// forming a self-referential `trampoline -> PLT -> GOT -> trampoline`
// infinite loop (zero computation, zero syscalls — exactly the observed
// hang).
//
// The fix: force every standard libm C symbol name to `local` binding via
// a linker version script, applied only to this crate's own link targets
// (not to dependency build-script helper binaries, which is why this is
// done here via `cargo:rustc-link-arg` rather than as a blanket
// `[build] rustflags` entry in `.cargo/config.toml` — see the comment
// there for why that approach failed).
fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-changed=.cargo/hide-libm-shadow.map");
    // Keep `target/bench_constants.env` in sync with the canonical constant —
    // `utils/quality-dashboard.sh` sources this file instead of hard-coding it.
    println!("cargo:rerun-if-changed=benches/constants.rs");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    let is_linux_elf = target_os == "linux" && target_env == "gnu";

    let target_feature = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let features: Vec<&str> = target_feature.split(',').collect();

    let required_v3_features = [
        "avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "lzcnt", "movbe",
    ];

    let missing_features: Vec<&str> = required_v3_features
        .iter()
        .copied()
        .filter(|&req| !features.iter().any(|f| f.trim() == req))
        .collect();

    if !missing_features.is_empty() {
        let missing_str = missing_features.join(", ");
        let msg = format!(
            "NeuralAmpModeler-rs requires full x86-64-v3 target support. \
             Missing required feature(s): {missing_str}. \
             Set RUSTFLAGS=\"-Ctarget-cpu=x86-64-v3\" to compile. \
             Detected features: {target_feature}"
        );
        println!("cargo:warning={msg}");
        std::process::exit(1);
    }

    if is_linux_elf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        println!("cargo:rustc-link-arg=-Wl,--undefined-version");
        println!(
            "cargo:rustc-link-arg=-Wl,--version-script={manifest_dir}/.cargo/hide-libm-shadow.map"
        );
    }

    export_bench_constants_env();
}

/// Extracts `DSP_MICRO_BATCH` from the canonical `benches/constants.rs` and
/// writes it to `target/bench_constants.env` for `utils/quality-dashboard.sh`
/// (`parse_benchmarks` sources it to divide micro-bench medians by the batch
/// factor). Fails the build loudly if the constant cannot be resolved — a
/// silently stale env file would let the dashboard compare wrong units.
fn export_bench_constants_env() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let constants_path = format!("{manifest_dir}/benches/constants.rs");
    let source = std::fs::read_to_string(&constants_path)
        .unwrap_or_else(|e| panic!("build.rs: cannot read {constants_path}: {e}"));

    let needle = "pub const DSP_MICRO_BATCH: usize";
    let decl = source
        .lines()
        .find(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("build.rs: `{needle}` not found in benches/constants.rs"));
    let value = decl
        .split('=')
        .next_back()
        .and_then(|rhs| {
            rhs.trim()
                .trim_end_matches(';')
                .trim()
                .parse::<usize>()
                .ok()
        })
        .unwrap_or_else(|| panic!("build.rs: cannot parse `usize` value from `{decl}`"));

    let env_path = format!("{manifest_dir}/target/bench_constants.env");
    if let Err(e) = std::fs::write(&env_path, format!("DSP_MICRO_BATCH={value}\n")) {
        panic!("build.rs: cannot write {env_path}: {e}");
    }
}
