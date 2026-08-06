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

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    let is_linux_elf = target_os == "linux" && target_env == "gnu";

    let target_feature = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let features: Vec<&str> = target_feature.split(',').collect();

    let has_avx2 = features.iter().any(|f| f.trim() == "avx2");
    let has_fma = features.iter().any(|f| f.trim() == "fma");

    if !has_avx2 || !has_fma {
        let msg = format!(
            "NeuralAmpModeler-rs requires x86-64-v3 (avx2+fma). \
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
        println!("cargo:rerun-if-changed=.cargo/hide-libm-shadow.map");
    }
}
