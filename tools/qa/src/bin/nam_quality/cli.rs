// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Strict fail-closed CLI flag parsing and error exit helpers.

use std::collections::HashMap;
use std::process::exit;

/// Sentinel used internally to forward a `--help` request from `parse_flags`.
pub(crate) const HELP_REQUEST: &str = "__help__";

pub(crate) struct Flags {
    map: HashMap<String, String>,
}

impl Flags {
    pub(crate) fn get(&self, name: &str) -> Option<&str> {
        self.map.get(name).map(String::as_str)
    }
}

/// Parses `--flag value` pairs; only `allowed` flags are accepted. Flags in
/// `boolean` are valueless switches (stored as `"true"`). A `-h`/`--help`
/// request surfaces as [`HELP_REQUEST`].
pub(crate) fn parse_flags(
    args: &[String],
    allowed: &[&str],
    boolean: &[&str],
) -> Result<Flags, String> {
    let mut map = HashMap::new();
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        if flag == "-h" || flag == "--help" {
            return Err(HELP_REQUEST.to_string());
        }
        let Some(name) = flag.strip_prefix("--") else {
            return Err(format!("{flag}: expected a --flag"));
        };
        if name.is_empty() {
            return Err(format!("{flag}: expected a flag name"));
        }
        if !allowed.contains(&name) {
            return Err(format!("unknown flag: --{name}"));
        }
        if boolean.contains(&name) {
            map.insert(name.to_string(), "true".to_string());
            continue;
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("--{name} requires a value"))?;
        map.insert(name.to_string(), value.clone());
    }
    Ok(Flags { map })
}

/// Exits 2 with a usage message (fail-closed CLI contract).
pub(crate) fn usage_error(context: &str, message: &str) -> ! {
    eprintln!("{context}: {message}");
    eprintln!("Run `nam_quality --help` for usage.");
    exit(2);
}

/// Exits 1 with an `ERROR:` line (fail-closed run-time contract).
pub(crate) fn run_error(message: String) -> ! {
    eprintln!("ERROR: {message}");
    exit(1);
}

pub(crate) fn required<'a>(flags: &'a Flags, name: &str) -> Result<&'a str, String> {
    flags
        .get(name)
        .ok_or_else(|| format!("missing required flag --{name}"))
}

/// Parses a non-negative integer flag value.
///
/// Values produced by shell `grep -c` pipelines may carry multiple
/// lines. The FIRST non-empty, trimmed line must be the plain count — the
/// embedded newline is tolerated instead of failing the whole receipt. A
/// non-numeric value (including the `file:count` form of a multi-file grep,
/// which is rejected fail-closed) still fails with a usage error.
pub(crate) fn parse_u32(flags: &Flags, name: &str, context: &str) -> u32 {
    match flags.get(name) {
        None => 0,
        Some(v) => {
            let first_line = v
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or("");
            match first_line.parse() {
                Ok(n) => n,
                Err(_) => usage_error(
                    context,
                    &format!("--{name} must be a non-negative integer (got '{first_line}')"),
                ),
            }
        }
    }
}
