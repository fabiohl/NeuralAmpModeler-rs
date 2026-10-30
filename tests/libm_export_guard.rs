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
//! `utils/debug/verify_no_libm_exports.sh` was consolidated into this test.
//!
//! Linux GNU only: ELF symbol interposition (and the version script that
//! prevents it) does not apply to Mach-O or other targets.
#![cfg(all(target_os = "linux", target_env = "gnu"))]

use std::env;
use std::path::Path;
use std::process::Command;

const LIBM_SYMBOLS: &[&str] = &["log10f", "atan2f", "acosf", "cbrt", "cbrtf", "fma", "fmod"];

/// `__libc_start_main` is imported by any glibc-linked executable and
/// remains visible in `.dynsym` even when the binary is stripped. It is the
/// witness that `nm -D` actually read the dynamic symbol table.
const DYNAMIC_WITNESS: &str = "__libc_start_main";

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

fn symbol_name(field: &str) -> &str {
    field.split('@').next().unwrap_or(field)
}

fn is_global_export(line: &str, symbol: &str) -> bool {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let (addr, kind, name) = match parts.as_slice() {
        [addr, kind, name] => (*addr, *kind, *name),
        _ => return false,
    };
    if symbol_name(name) != symbol {
        return false;
    }
    if addr != "U" && (!addr.chars().all(|c| c.is_ascii_hexdigit()) || addr.is_empty()) {
        return false;
    }
    matches!(kind, "T" | "W")
}

fn has_undefined(stdout: &str, symbol: &str) -> bool {
    stdout.lines().any(|line| {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["U", name] => symbol_name(name) == symbol,
            [addr, "U", name] if !addr.is_empty() => symbol_name(name) == symbol,
            _ => false,
        }
    })
}

/// Runs `nm` with a fixed C locale so the verdict never depends on the
/// process locale, and fails closed on any tool error.
fn nm_stdout(nm: &str, artifact: &Path, extra_args: &[&str]) -> String {
    let output = Command::new(nm)
        .env("LC_ALL", "C")
        .env("LANG", "C")
        .args(extra_args)
        .arg(artifact)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "libm_export_guard: failed to run {nm} on {}: {e}",
                artifact.display()
            )
        });
    // Fail-closed: a failed `nm` yields empty stdout, which would otherwise
    // look like "no leaks" and certify an unchecked surface.
    assert!(
        output.status.success(),
        "libm_export_guard: {} exited {:?} on {}; cannot certify ELF surface",
        nm,
        output.status.code(),
        artifact.display()
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn leaked_in(stdout: &str) -> Vec<&'static str> {
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

    // The dynamic table is the canonical surface for interposition.
    let dynamic = nm_stdout(nm, &exe, &["-D"]);
    assert!(
        has_undefined(&dynamic, DYNAMIC_WITNESS),
        "libm_export_guard: {} -D reported no {DYNAMIC_WITNESS} import on {}; \
         cannot certify the dynamic ELF surface (stdout was empty or invalid)",
        nm,
        exe.display()
    );

    let mut leaked = leaked_in(&dynamic);

    // The static table is a complementary surface. A stripped release binary
    // legitimately has none; in that case the dynamic witness above is the
    // certification. Never decide by localized `nm` stderr text.
    let static_surface = nm_stdout(nm, &exe, &[]);
    if !static_surface.trim().is_empty() {
        leaked.extend(leaked_in(&static_surface));
    }

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
