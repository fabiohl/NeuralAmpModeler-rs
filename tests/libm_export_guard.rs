// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! ELF surface guard: the *linked* binary must not export libm symbols.
//!
//! The hang in `docs/postmortem-libm-symbol-interposition.md` is interposition
//! of GLOBAL symbols in the final ELF. `build.rs` applies
//! `.cargo/hide-libm-shadow.map` at link time — scanning a `.rlib` is the
//! wrong surface (object archives still carry `T` before the version script).
//!
//! Canonical gate: wired into `tests-quick.sh` Phase 1 and `tests-long.sh`
//! Defense phase. The former standalone wrapper
//! `utils/debug/verify_no_libm_exports.sh` was removed (S4-T03).

use std::env;
use std::path::PathBuf;
use std::process::Command;

const LIBM_SYMBOLS: &[&str] = &["log10f", "atan2f", "acosf", "cbrt", "cbrtf", "fma", "fmod"];

fn nm_tool() -> &'static str {
    for tool in ["nm", "llvm-nm"] {
        if Command::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return tool;
        }
    }
    panic!(
        "libm_export_guard: neither nm nor llvm-nm found in PATH. \
         DIAGNOSTIC: cannot certify the ELF surface. Install binutils or llvm."
    );
}

fn is_global_export(line: &str, symbol: &str) -> bool {
    let mut parts = line.split_whitespace();
    let addr = parts.next().unwrap_or("");
    let kind = parts.next().unwrap_or("");
    let name = parts.next().unwrap_or("");
    if name != symbol || parts.next().is_some() {
        return false;
    }
    if addr != "U" && (!addr.chars().all(|c| c.is_ascii_hexdigit()) || addr.is_empty()) {
        return false;
    }
    matches!(kind, "T" | "W")
}

fn leaked_in(nm: &str, artifact: &PathBuf, extra_args: &[&str]) -> Vec<&'static str> {
    let output = Command::new(nm)
        .args(extra_args)
        .arg(artifact)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "libm_export_guard: failed to run {nm} on {}: {e}",
                artifact.display()
            )
        });
    let stdout = String::from_utf8_lossy(&output.stdout);
    LIBM_SYMBOLS
        .iter()
        .copied()
        .filter(|symbol| stdout.lines().any(|line| is_global_export(line, symbol)))
        .collect()
}

#[test]
fn no_libm_symbols_exported_from_linked_binary() {
    let exe = env::current_exe().expect("libm_export_guard: current_exe unavailable");
    let nm = nm_tool();
    let mut leaked = leaked_in(nm, &exe, &[]);
    leaked.extend(leaked_in(nm, &exe, &["-D"]));
    leaked.sort_unstable();
    leaked.dedup();
    assert!(
        leaked.is_empty(),
        "LIBM SYMBOL LEAK in linked ELF {}: {leaked:?}. \
         DIAGNOSTIC: GLOBAL/WEAK export of libm names can interpose libc and hang. \
         build.rs must apply .cargo/hide-libm-shadow.map. \
         See docs/postmortem-libm-symbol-interposition.md.",
        exe.display()
    );
}
