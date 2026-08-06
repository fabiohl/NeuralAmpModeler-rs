// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! # NAM Model Inspector (`inspect_model`)
//!
//! Comprehensive inspection tool for Neural Amp Modeler files (`.nam` and `.namb`).
//! This is the **canonical replacement** for `check-model.py`, implemented entirely
//! via the official loader API — ensuring correctness, completeness, and automatic
//! compatibility with future format updates.
//!
//! ## Usage
//!
//! ```bash
//! # Human-readable inspection (default):
//! cargo run --example inspect_model -- model.nam
//! cargo run --example inspect_model -- model.namb
//!
//! # JSON output (machine-readable, for scripting):
//! cargo run --example inspect_model -- model.nam --json
//!
//! # Batch / manifest mode (multiple files → JSON array):
//! cargo run --example inspect_model -- *.nam *.namb --manifest
//!
//! # Via the convenience shell script:
//! utils/check-model.sh model.namb
//! utils/check-model.sh --json model.nam
//! utils/check-model.sh --manifest models/*.nam
//! ```
//!
//! ## Output Sections (human-readable mode)
//!
//! - **File** — path, size, SHA-256, format, load time
//! - **Architecture** — arch type, topology class, weights layout, weight count
//! - **DSP Profile** — sample rate, receptive field
//! - **Gain Staging** — loudness, input/output dBu levels, multiplier adjustments
//! - **Metadata** — name, author, gear make/model/type, tone type, training date
//! - **Compatibility** — engine recognition status, stereo support

use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use neural_amp_modeler_rs::common::diagnostics::SystemSnapshot;
use neural_amp_modeler_rs::loader::{LoadOptions, LoadedModelPair, load_and_build_model};

// ─── SHA-256 ──────────────────────────────────────────────────────────────────

fn sha256_hex(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    match std::fs::read(path) {
        Ok(bytes) => format!("{:x}", Sha256::digest(&bytes)),
        Err(_) => "(unreadable)".to_string(),
    }
}

// ─── ANSI colour helpers ──────────────────────────────────────────────────────

fn use_color() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
        && std::env::var("NO_COLOR").is_err()
        && std::env::var("TERM").as_deref() != Ok("dumb")
}

macro_rules! ansi {
    ($code:expr, $s:expr) => {
        if use_color() {
            format!("\x1b[{}m{}\x1b[0m", $code, $s)
        } else {
            $s.to_string()
        }
    };
}

fn bold(s: &str) -> String {
    ansi!("1", s)
}
fn green(s: &str) -> String {
    ansi!("92", s)
}
fn yellow(s: &str) -> String {
    ansi!("93", s)
}
fn cyan(s: &str) -> String {
    ansi!("96", s)
}
fn red(s: &str) -> String {
    ansi!("91", s)
}
fn dim(s: &str) -> String {
    ansi!("2", s)
}

// ─── Topology helpers ─────────────────────────────────────────────────────────

/// Returns `true` for the four canonical topologies with pre-built SIMD kernels.
fn is_standard_topology(pair: &LoadedModelPair) -> bool {
    matches!(
        pair.topology.as_str(),
        "Standard" | "Lite" | "Feather" | "Nano"
    )
}

fn compat_label(pair: &LoadedModelPair) -> String {
    if is_standard_topology(pair) {
        green("STANDARD / SUPPORTED")
    } else {
        yellow("CUSTOM / TARGET-RESEARCH")
    }
}

// ─── Human-readable report ────────────────────────────────────────────────────

fn section(title: &str) {
    println!();
    println!("{}", bold(&format!("── {} ", title)));
}

fn field(label: &str, value: &str) {
    println!("   {:<28} {}", dim(&format!("{}:", label)), value);
}

fn field_opt(label: &str, value: Option<&str>) {
    field(label, value.unwrap_or(&dim("—")));
}

fn human_report(path: &Path, pair: &LoadedModelPair, file_size: u64, load_ms: f64) {
    let info = pair.model_info(path);
    let sha = sha256_hex(path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_uppercase();

    println!();
    println!(
        "{}",
        bold("══════════════════════════════════════════════════════════════")
    );
    println!("  {}  NeuralAmpModeler-rs — Model Inspector", cyan("◈"));
    println!(
        "{}",
        bold("══════════════════════════════════════════════════════════════")
    );

    // ── File ─────────────────────────────────────────────────────────────────
    section("FILE");
    field("Path", &path.display().to_string());
    field(
        "Format",
        &format!(
            ".{} ({})",
            ext.to_lowercase(),
            if ext == "NAMB" { "binary" } else { "JSON" }
        ),
    );
    field(
        "Size",
        &format!(
            "{} bytes  ({:.2} KiB)",
            file_size,
            file_size as f64 / 1024.0
        ),
    );
    field("Load Time", &format!("{:.2} ms", load_ms));

    // ── Architecture ─────────────────────────────────────────────────────────
    section("ARCHITECTURE");
    field("Type", &cyan(&pair.architecture));
    field("Topology", &cyan(&pair.topology));
    field("Weights Layout", &pair.weights_layout);
    field(
        "Channels",
        &if info.channels > 0 {
            info.channels.to_string()
        } else {
            dim("n/a")
        },
    );

    // ── DSP Profile ───────────────────────────────────────────────────────────
    section("DSP PROFILE");
    field("Sample Rate", &format!("{} Hz", pair.sample_rate));
    field(
        "Receptive Field",
        &if info.receptive_field > 0 {
            format!(
                "{} samples  ({:.1} ms @ {} Hz)",
                info.receptive_field,
                info.receptive_field as f64 / pair.sample_rate as f64 * 1000.0,
                pair.sample_rate
            )
        } else {
            dim("n/a")
        },
    );

    // ── Gain Staging ──────────────────────────────────────────────────────────
    section("GAIN STAGING");
    field(
        "Loudness",
        &match pair.loudness() {
            Some(l) => format!("{:.2} dB LUFS", l),
            None => dim("not specified  (default −18.0 dB LUFS)"),
        },
    );
    field(
        "Input Level",
        &match pair.input_level_dbu() {
            Some(v) => format!("{:.2} dBu", v),
            None => dim("not specified  (default 12.0 dBu)"),
        },
    );
    field(
        "Output Level",
        &match pair.output_level_dbu() {
            Some(v) => format!("{:.2} dBu", v),
            None => dim("not specified"),
        },
    );
    field(
        "Input Adj",
        &format!(
            "×{:.6}  ({:+.3} dB)",
            pair.input_mult_adj,
            20.0 * pair.input_mult_adj.max(1e-9_f32).log10()
        ),
    );
    field(
        "Output Adj",
        &format!(
            "×{:.6}  ({:+.3} dB)",
            pair.output_mult_adj,
            20.0 * pair.output_mult_adj.max(1e-9_f32).log10()
        ),
    );

    // ── Metadata ──────────────────────────────────────────────────────────────
    section("METADATA");
    if let Some(meta) = &pair.metadata {
        field_opt("Model Name", meta.name.as_deref());
        field_opt("Author", meta.modeled_by.as_deref());
        field_opt("Gear Make", meta.gear_make.as_deref());
        field_opt("Gear Model", meta.gear_model.as_deref());
        field_opt("Gear Type", meta.gear_type.as_deref());
        field_opt("Tone Type", meta.tone_type.as_deref());

        if let Some(date) = &meta.date {
            let y = date.year.map_or("????".into(), |v| v.to_string());
            let mo = date.month.map_or("??".into(), |v| format!("{:02}", v));
            let d = date.day.map_or("??".into(), |v| format!("{:02}", v));
            let h = date.hour.map_or("??".into(), |v| format!("{:02}", v));
            let mi = date.minute.map_or("??".into(), |v| format!("{:02}", v));
            field(
                "Training Date",
                &format!("{}-{}-{}  {}:{}", y, mo, d, h, mi),
            );
        } else {
            field("Training Date", &dim("—"));
        }

        if meta.training.is_some() {
            field("Training Config", &dim("[present — use --json to inspect]"));
        }
    } else {
        println!("   {}", dim("(no metadata block present in this file)"));
    }

    // ── Integrity ─────────────────────────────────────────────────────────────
    section("INTEGRITY");
    field("SHA-256", &sha);

    // ── Compatibility ─────────────────────────────────────────────────────────
    section("COMPATIBILITY");
    field("Engine Status", &compat_label(pair));
    let model_l_status = if pair.model_l.is_some() {
        green("ready")
    } else {
        red("build failed")
    };
    field("Model (L)", &model_l_status);
    let model_r_status = if pair.model_r.is_some() {
        green("ready (stereo)")
    } else {
        dim("none (mono)")
    };
    field("Model (R)", &model_r_status);

    println!();
    println!(
        "{}",
        bold("══════════════════════════════════════════════════════════════")
    );
    println!();
}

// ─── JSON report ──────────────────────────────────────────────────────────────

fn json_report(path: &Path, pair: &LoadedModelPair, file_size: u64, load_ms: f64) {
    let info = pair.model_info(path);
    let sha = sha256_hex(path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let meta_json = pair.metadata.as_ref().map(|meta| {
        let date_str = meta.date.as_ref().map(|d| {
            format!(
                "{}-{:02}-{:02}T{:02}:{:02}",
                d.year.unwrap_or(0),
                d.month.unwrap_or(0),
                d.day.unwrap_or(0),
                d.hour.unwrap_or(0),
                d.minute.unwrap_or(0),
            )
        });
        serde_json::json!({
            "name":               meta.name,
            "modeled_by":         meta.modeled_by,
            "gear_make":          meta.gear_make,
            "gear_model":         meta.gear_model,
            "gear_type":          meta.gear_type,
            "tone_type":          meta.tone_type,
            "training_date":      date_str,
            "has_training_config": meta.training.is_some(),
            "loudness_db":        meta.loudness,
            "input_level_dbu":    meta.input_level_dbu,
            "output_level_dbu":   meta.output_level_dbu,
        })
    });

    let output = serde_json::json!({
        "file": {
            "path":       path.display().to_string(),
            "filename":   path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
            "format":     ext,
            "size_bytes": file_size,
            "sha256":     sha,
        },
        "load_time_ms": load_ms,
        "architecture": {
            "type":           pair.architecture,
            "topology":       pair.topology,
            "weights_layout": pair.weights_layout,
            "channels":       if info.channels > 0 { Some(info.channels) } else { None },
        },
        "dsp_profile": {
            "sample_rate_hz":           pair.sample_rate,
            "receptive_field_samples":  if info.receptive_field > 0 { Some(info.receptive_field) } else { None },
        },
        "gain_staging": {
            "loudness_db":      pair.loudness(),
            "input_level_dbu":  pair.input_level_dbu(),
            "output_level_dbu": pair.output_level_dbu(),
            "input_mult_adj":   pair.input_mult_adj,
            "output_mult_adj":  pair.output_mult_adj,
        },
        "metadata": meta_json,
        "compatibility": {
            "is_standard_topology": is_standard_topology(pair),
            "is_supported":         pair.model_l.is_some(),
            "has_stereo_model":     pair.model_r.is_some(),
        },
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
}

// ─── Manifest helpers ─────────────────────────────────────────────────────────

fn manifest_entry(
    path: &Path,
    pair: &LoadedModelPair,
    file_size: u64,
    sha: &str,
) -> serde_json::Value {
    let meta = &pair.metadata;
    serde_json::json!({
        "filename":              path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
        "path":                  path.display().to_string(),
        "sha256":                sha,
        "architecture":          pair.architecture,
        "topology":              pair.topology,
        "weights_layout":        pair.weights_layout,
        "sample_rate_hz":        pair.sample_rate,
        "size_bytes":            file_size,
        "is_standard_topology":  is_standard_topology(pair),
        "is_supported":          pair.model_l.is_some(),
        "name":                  meta.as_ref().and_then(|m| m.name.as_deref()),
        "modeled_by":            meta.as_ref().and_then(|m| m.modeled_by.as_deref()),
        "gear_make":             meta.as_ref().and_then(|m| m.gear_make.as_deref()),
        "gear_model":            meta.as_ref().and_then(|m| m.gear_model.as_deref()),
        "gear_type":             meta.as_ref().and_then(|m| m.gear_type.as_deref()),
        "tone_type":             meta.as_ref().and_then(|m| m.tone_type.as_deref()),
        "loudness_db":           meta.as_ref().and_then(|m| m.loudness),
        "input_level_dbu":       meta.as_ref().and_then(|m| m.input_level_dbu),
        "output_level_dbu":      meta.as_ref().and_then(|m| m.output_level_dbu),
    })
}

fn manifest_error(path: &Path, error: &str) -> serde_json::Value {
    serde_json::json!({
        "filename":    path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
        "path":        path.display().to_string(),
        "error":       error,
        "is_supported": false,
    })
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        process::exit(if args.is_empty() { 1 } else { 0 });
    }

    let manifest_mode = args.iter().any(|a| a == "--manifest");
    let json_mode = args.iter().any(|a| a == "--json") || manifest_mode;

    let file_args: Vec<&str> = args
        .iter()
        .filter(|a| !a.starts_with('-'))
        .map(String::as_str)
        .collect();

    if file_args.is_empty() {
        eprintln!("{} No model file(s) specified.", red("Error:"));
        print_usage();
        process::exit(1);
    }

    // One SystemSnapshot for the entire run (cheap — reads CPUID once)
    let sys = SystemSnapshot::capture();

    if manifest_mode {
        // ── Batch mode ─────────────────────────────────────────────────────
        let mut entries: Vec<serde_json::Value> = Vec::new();
        for filepath in &file_args {
            let path = PathBuf::from(filepath);
            if !path.exists() {
                entries.push(manifest_error(&path, "file not found"));
                continue;
            }
            let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let sha = sha256_hex(&path);
            match load_and_build_model(&path, &sys, false, LoadOptions::default()) {
                Ok(pair) => entries.push(manifest_entry(&path, &pair, file_size, &sha)),
                Err(e) => entries.push(manifest_error(&path, &e.to_string())),
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&entries).unwrap_or_default()
        );
        return;
    }

    // ── Single / multi file mode ──────────────────────────────────────────────
    let mut had_error = false;
    for filepath in &file_args {
        let path = PathBuf::from(filepath);
        if !path.exists() {
            eprintln!("{} File not found: {}", red("Error:"), filepath);
            had_error = true;
            continue;
        }
        let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let t0 = Instant::now();
        match load_and_build_model(&path, &sys, false, LoadOptions::default()) {
            Ok(pair) => {
                let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
                if json_mode {
                    json_report(&path, &pair, file_size, load_ms);
                } else {
                    human_report(&path, &pair, file_size, load_ms);
                }
            }
            Err(e) => {
                eprintln!("{} Failed to load \"{}\": {}", red("Error:"), filepath, e);
                had_error = true;
            }
        }
    }

    if had_error {
        process::exit(1);
    }
}

fn print_usage() {
    eprintln!("NAM Model Inspector — powered by the official NeuralAmpModeler-rs loader");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  cargo run --example inspect_model -- [OPTIONS] <file.nam|.namb> [...]");
    eprintln!("  utils/check-model.sh [OPTIONS] <file.nam|.namb> [...]");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --json       Emit machine-readable JSON instead of human-readable text");
    eprintln!("  --manifest   Batch mode: emit a JSON array with one entry per file");
    eprintln!("  --help, -h   Show this help message");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("  cargo run --example inspect_model -- model.nam");
    eprintln!("  cargo run --example inspect_model -- model.namb --json");
    eprintln!("  cargo run --example inspect_model -- *.nam *.namb --manifest");
}
