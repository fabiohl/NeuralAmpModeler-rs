// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Binary CLI utility that emits the golden generation catalog from Rust.
//!
//! The golden registry lives in `src/testing/catalog.rs::GOLDEN_GEN_CATALOG`
//! (single source of truth). This binary serializes it in
//! the shell line format consumed by `tests/fixtures/golden_gen_build.sh`,
//! which previously carried the same data as a static bash `CATALOG=(...)`
//! array. Model lists are therefore never duplicated in shell scripts.
//!
//! Output modes:
//! - `emit-catalog`: prints one catalog line per registry entry
//!   (`nam_file:golden_name:label:v2_scope[:skip_srs[:skip_reason]]`) to stdout.
//!
//! Exit codes:
//! - 0: catalog emitted successfully
//! - 2: unknown or missing subcommand

use std::process::exit;

use neural_amp_modeler_rs::testing::catalog::golden_gen_catalog_lines;

fn main() {
    let command = std::env::args().nth(1).unwrap_or_default();

    match command.as_str() {
        "emit-catalog" => print!("{}", golden_gen_catalog_lines()),
        "--help" | "-h" => {
            println!("NAM Golden Catalog Emitter");
            println!("Usage: nam_golden_catalog emit-catalog");
            exit(0);
        }
        _ => {
            eprintln!("usage: nam_golden_catalog emit-catalog");
            exit(2);
        }
    }
}
