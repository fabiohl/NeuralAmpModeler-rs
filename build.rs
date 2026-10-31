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
    emit_avx512_cfg();

    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=CARGO_PRIMARY_PACKAGE");
    println!("cargo:rerun-if-changed=.cargo/hide-libm-shadow.map");
    // Keep `target/bench_constants.env` in sync with the canonical constant —
    // `utils/quality-dashboard.sh` sources this file instead of hard-coding it.
    println!("cargo:rerun-if-changed=benches/constants.rs");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let is_linux_elf = target_os == "linux" && target_env == "gnu";

    if target_arch == "x86_64" {
        check_x86_64_v3();
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

/// Emits `nam_avx512` only when the `avx512` feature is enabled for an x86_64
/// target. The AVX-512 modules gate on this cfg, so aarch64 documentation builds
/// (docs.rs `all-features`) compile without them. Runs before the `DOCS_RS`
/// early return so docs.rs builds see the same gate.
fn emit_avx512_cfg() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_AVX512");
    let x86_64 = std::env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|arch| arch == "x86_64");
    if x86_64 && std::env::var_os("CARGO_FEATURE_AVX512").is_some() {
        println!("cargo:rustc-cfg=nam_avx512");
    }
}

/// Fails the build when an x86_64 target is compiled without the full
/// x86-64-v3 feature set (the AVX2/FMA kernels are selected at compile time).
fn check_x86_64_v3() {
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
        // Cargo renders each `cargo:warning=` as a separate one-line warning:
        // `\n` escapes inside a value are not expanded and real newlines are
        // dropped by the line-based stdout parser, so the multi-line fix is
        // emitted as one warning per line. The `.cargo/config.toml` snippet
        // mirrors README.md verbatim so consumers can copy it as-is.
        let role = if is_primary_build() {
            "NeuralAmpModeler-rs requires full x86-64-v3 target support."
        } else {
            "NeuralAmpModeler-rs is compiled as a dependency and requires full x86-64-v3 target support."
        };
        let role_id = if is_primary_build() {
            "primary"
        } else {
            "dependency"
        };
        let mut msg_lines = vec![
            format!("NAM_CONTRACT code=x86-64-v3-missing role={role_id}"),
            role.to_string(),
            format!("Missing required feature(s): {missing_str} (detected: {target_feature})."),
            "Set RUSTFLAGS=\"-Ctarget-cpu=x86-64-v3\" to compile, or add the \
             following snippet to the .cargo/config.toml of your APPLICATION \
             (Cargo never propagates a dependency's RUSTFLAGS downstream):"
                .to_string(),
            "[build]".to_string(),
            "rustflags = [\"-Ctarget-cpu=x86-64-v3\"]".to_string(),
        ];
        if !is_primary_build() {
            msg_lines
                .push("See README.md, section \"SIMD Baseline per Architecture\".".to_string());
        }
        for line in msg_lines {
            println!("cargo:warning={line}");
        }
        std::process::exit(1);
    }
}

/// Returns true when this package is the build root (the repository itself),
/// rather than a dependency compiled on behalf of a consumer.
///
/// `CARGO_PRIMARY_PACKAGE` is the canonical signal, but Cargo 1.99.1 does not
/// expose it to the build-script process (verified: the variable is present for
/// `rustc` invocations but absent at runtime), so fall back to layout: an
/// in-repo build places `OUT_DIR` under `<manifest>/target`, while a dependency
/// places it under the consumer's target directory. `starts_with` is
/// component-wise and does not resolve symlinks.
fn is_primary_build() -> bool {
    if std::env::var_os("CARGO_PRIMARY_PACKAGE").is_some() {
        return true;
    }
    let (Some(manifest_dir), Some(out_dir)) = (
        std::env::var_os("CARGO_MANIFEST_DIR"),
        std::env::var_os("OUT_DIR"),
    ) else {
        return false;
    };
    std::path::Path::new(&out_dir).starts_with(std::path::Path::new(&manifest_dir).join("target"))
}

/// Extracts `DSP_MICRO_BATCH` from the canonical `benches/constants.rs` and
/// writes it to `target/bench_constants.env` for `utils/quality-dashboard.sh`
/// (`parse_benchmarks` sources it to divide micro-bench medians by the batch
/// factor). Fails the build loudly if the constant cannot be resolved — a
/// silently stale env file would let the dashboard compare wrong units.
///
/// Runs only when this package is the build root. As a dependency, a build
/// script must not write outside `OUT_DIR` (the registry checkout is usually
/// read-only) nor fail a consumer build because `benches/constants.rs` is not
/// part of the published package.
fn export_bench_constants_env() {
    if !is_primary_build() {
        return;
    }

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

    // Dashboard-only artifact: `utils/quality-dashboard.sh` sources this file to
    // divide micro-bench medians by the batch factor. Only the primary package
    // reaches this point, and the unpacked `cargo package --verify` copy has no
    // `target/` directory — so export stays best-effort: missing parents are
    // created, and any residual I/O failure degrades to a warning instead of
    // failing the build. A wrong constant would still compare wrong units, so
    // unparsable `benches/constants.rs` keeps panicking above.
    let env_path = format!("{manifest_dir}/target/bench_constants.env");
    if let Some(parent) = std::path::Path::new(&env_path).parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        println!(
            "cargo:warning=build.rs: cannot create {}: {e}; skipping bench_constants.env export (dashboard-only artifact)",
            parent.display()
        );
        return;
    }
    if let Err(e) = std::fs::write(&env_path, format!("DSP_MICRO_BATCH={value}\n")) {
        println!(
            "cargo:warning=build.rs: cannot write {env_path}: {e}; skipping bench_constants.env export (dashboard-only artifact)"
        );
    }
}
