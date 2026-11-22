// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Build script for `nam-qa`: captures authentic compiler provenance
//! (`NAM_QA_RUSTC_VERSION`, `NAM_QA_TARGET`, `NAM_QA_COMPILE_FLAGS`) at
//! compile time so parity vector headers never fall back to stale defaults.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=OPT_LEVEL");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=TARGET");

    let rustc_bin = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let rustc_version = Command::new(&rustc_bin)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .map(|l| l.trim().to_string())
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    let opt_level = std::env::var("OPT_LEVEL").unwrap_or_else(|_| "0".to_string());
    let encoded_flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let mut compile_flags = format!("-C opt-level={opt_level}");
    if !encoded_flags.is_empty() {
        let extra = encoded_flags.split('\x1f').collect::<Vec<_>>().join(" ");
        if !extra.trim().is_empty() {
            compile_flags.push(' ');
            compile_flags.push_str(extra.trim());
        }
    }

    println!("cargo:rustc-env=NAM_QA_RUSTC_VERSION={rustc_version}");
    println!("cargo:rustc-env=NAM_QA_TARGET={target}");
    println!("cargo:rustc-env=NAM_QA_COMPILE_FLAGS={compile_flags}");
}
