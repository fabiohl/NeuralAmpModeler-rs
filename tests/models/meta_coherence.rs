// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Meta-tests that enforce catalog↔test coherence.
//
//  Ensures every `.nam` model referenced by an `#[ignore]` test
//  (golden-missing reason) is registered in the canonical golden registry
//  `src/testing/catalog.rs::GOLDEN_GEN_CATALOG`. Prevents silent drift where
//  a model is added to tests but forgotten in the golden generator.
//
//  The golden registry is Rust-only (single source of
//  truth): `tests/fixtures/golden_gen_build.sh` sources its catalog from the
//  `nam_golden_catalog` binary instead of defining a bash array. These tests
//  therefore validate against the Rust registry, and `test_generator_sources_
//  catalog_from_rust` guards the script against reintroducing a static
//  CATALOG array.
//
//  Part of the bidirectional coherence check suite.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use neural_amp_modeler_rs::testing::catalog::golden_gen_entries;
use neural_amp_modeler_rs::testing::qa::QualityContract;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn golden_gen_build_path() -> PathBuf {
    fixture_dir().join("golden_gen_build.sh")
}

/// Returns the set of `.nam` filenames registered in the canonical golden
/// registry (`src/testing/catalog.rs::GOLDEN_GEN_CATALOG`).
fn rust_catalog_models() -> HashSet<String> {
    let nam_files: HashSet<String> = golden_gen_entries()
        .iter()
        .map(|e| e.nam_file.to_string())
        .collect();

    assert!(
        !nam_files.is_empty(),
        "GOLDEN_GEN_CATALOG in src/testing/catalog.rs is empty. \
         Check the registry and ensure it contains valid entries."
    );
    assert!(
        nam_files.len() >= 15,
        "Expected ≥ 15 models in GOLDEN_GEN_CATALOG, found {}. \
         Regression — fewer models than documented 20 entries.",
        nam_files.len(),
    );
    nam_files
}

/// Returns the set of models that have a non-empty `skip_reason` field in the
/// golden registry. These models are intentionally excluded from golden
/// generation and freshness gates.
fn rust_skip_reason_models() -> HashSet<String> {
    golden_gen_entries()
        .iter()
        .filter(|e| e.skip_reason.is_some())
        .map(|e| e.nam_file.to_string())
        .collect()
}

/// Extracts `.nam` string literals from a block of text.
///
/// Only returns strings that pass `is_valid_model_filename` — i.e.,
/// actual model references, not error/log strings that happen to
/// end in `.nam`.
fn extract_nam_strings(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut in_string = false;
    let mut current = String::new();

    for ch in text.chars() {
        match ch {
            '"' => {
                if in_string {
                    if current.ends_with(".nam") && current.len() > 4 {
                        let candidate = std::mem::take(&mut current);
                        if is_valid_model_filename(&candidate) {
                            result.push(candidate);
                        }
                    }
                    current.clear();
                    in_string = false;
                } else {
                    in_string = true;
                }
            }
            _ if in_string => current.push(ch),
            _ => {}
        }
    }
    result
}

fn is_valid_model_filename(s: &str) -> bool {
    // Reject strings that are clearly error/log messages
    let lower = s.to_lowercase();
    if lower.contains("failed") || lower.contains("not found") || s.len() > 120 {
        return false;
    }
    // Model filenames use a limited character set (allow spaces)
    s.chars().all(|c| c.is_ascii_graphic() || c == ' ')
        && s.matches('.').count() <= 2
        && s.ends_with(".nam")
}

/// Scans a test source file for `#[ignore]` blocks and extracts
/// `.nam` model references found within each ignored test function.
///
/// Returns `(nam_file, source_file_name, test_fn_signature)` tuples.
fn scan_ignored_test_models(
    file_path: &PathBuf,
    source_name: &str,
) -> Vec<(String, String, String)> {
    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut result = Vec::new();

    let blocks: Vec<&str> = content.split("#[ignore").collect();
    for block in &blocks[1..] {
        let fn_sig = block
            .lines()
            .find(|l| l.trim().starts_with("fn test_"))
            .map(|l| l.trim().to_string())
            .unwrap_or_else(|| String::from("(unknown test fn)"));

        let window: String = block.lines().take(50).collect::<Vec<_>>().join("\n");

        for nam_file in extract_nam_strings(&window) {
            result.push((nam_file, source_name.to_string(), fn_sig.clone()));
        }
    }

    result
}

/// Scans a file for all `.nam` string literals, regardless of context.
fn scan_all_nam_strings_in_file(file_path: &PathBuf) -> HashSet<String> {
    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => return HashSet::new(),
    };
    extract_nam_strings(&content).into_iter().collect()
}

fn tests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

/// Every model in the golden registry (except those with `skip_reason`) MUST
/// have at least one test consumer that references it by `.nam` filename.
///
/// This is the **inverse** direction of `test_ignored_models_are_in_catalog`:
/// that test ensures models in test files are registered in the registry;
/// THIS test ensures models in the registry are exercised by test files.
///
/// Together they form a bidirectional coherence gate (Registry ↔ Testes).
///
/// ## Implementation
///
/// Scans all `.rs` files under `tests/` for `.nam` filename string literals,
/// then checks that every registry model (minus `skip_reason` models) appears
/// in the resulting set.
///
/// ## Known exemptions
///
/// Models with `skip_reason` in the registry are exempt — they are shown to
/// have known incompatibilities (separately documented in
/// `src/testing/catalog.rs`).
#[test]
fn test_catalog_models_have_consumers() {
    let catalog = rust_catalog_models();
    let skip_models = rust_skip_reason_models();

    // Collect all .nam filenames referenced across every test source file
    let mut consumers: HashSet<String> = HashSet::new();
    let tests_root = tests_dir();

    fn walk_dir(dir: &std::path::Path, consumers: &mut HashSet<String>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk_dir(&path, consumers);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let found = scan_all_nam_strings_in_file(&path);
                    consumers.extend(found);
                }
            }
        }
    }
    walk_dir(&tests_root, &mut consumers);

    // Build the set of test-known .nam filenames
    let consumers: HashSet<String> = consumers
        .into_iter()
        .filter(|s| is_valid_model_filename(s))
        .collect();

    assert!(
        consumers.len() >= 15,
        "Expected ≥ 15 .nam consumers across test files, found {}. \
         Scanner may be broken — check is_valid_model_filename filters.",
        consumers.len(),
    );

    let mut missing = Vec::new();
    for model in catalog.iter() {
        if skip_models.contains(model.as_str()) {
            continue;
        }
        if !consumers.contains(model.as_str()) {
            missing.push(model.clone());
        }
    }

    if !missing.is_empty() {
        panic!(
            "{} model(s) in GOLDEN_GEN_CATALOG have NO test consumer:\n  {}\n\
             \n\
             Add at least one test that references each model by `.nam` filename \
             (e.g., model_path(\"{}\") or run_v1(\"{}\", ...)).\n\
             If a model is intentionally untested, add its skip_reason field \
             to the registry entry in src/testing/catalog.rs.",
            missing.len(),
            missing.join("\n  "),
            missing.first().unwrap(),
            missing.first().unwrap(),
        );
    }

    let skipped_count = skip_models.len();
    let total_catalog = catalog.len();
    let active = total_catalog - skipped_count;
    eprintln!(
        "  ✓ catalog→ tests coherence: {active}/{total_catalog} active models all have consumers \
         ({skipped_count} skipped via skip_reason).",
    );
}

/// Structural guard: `golden_gen_build.sh` must source its
/// catalog from the Rust registry (`nam_golden_catalog emit-catalog`) and must
/// NOT define a static `CATALOG=(...)` bash array. This prevents the catalog
/// duplication from silently coming back.
#[test]
fn test_generator_sources_catalog_from_rust() {
    let script = golden_gen_build_path();
    let content = fs::read_to_string(&script).expect("Failed to read golden_gen_build.sh");

    assert!(
        content.contains("nam_golden_catalog"),
        "golden_gen_build.sh must source its catalog from the Rust registry \
         via `nam_golden_catalog emit-catalog`.\n\
         The single source of truth is src/testing/catalog.rs::GOLDEN_GEN_CATALOG."
    );
    assert!(
        !content.contains("CATALOG=("),
        "golden_gen_build.sh must NOT define a static CATALOG=(...) array.\n\
         Model lists live exclusively in src/testing/catalog.rs::GOLDEN_GEN_CATALOG \
         — the generator consumes them through `nam_golden_catalog emit-catalog`."
    );
}

/// Structural guard: `utils/tests-long.sh` must emit the
/// long-audit receipt exclusively through the Rust emitter
/// (`nam_long_receipt append|summary` → `target/logs/long-audit-receipt.jsonl`)
/// and must NOT hand-serialize the receipt as inline bash JSON — the fragile
/// quoting this eliminated.
#[test]
fn test_long_suite_receipt_emitted_from_rust_bin() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("utils/tests-long.sh");
    let content = fs::read_to_string(&script).expect("Failed to read utils/tests-long.sh");

    assert!(
        content.contains("nam_long_receipt"),
        "utils/tests-long.sh must emit the long-audit receipt through the Rust \
         emitter `nam_long_receipt append` / `nam_long_receipt summary`.\n\
         The single generator is src/testing/receipt.rs + src/bin/nam_long_receipt.rs."
    );
    assert!(
        !content.contains("\"phase_id\""),
        "utils/tests-long.sh must NOT hand-serialize long-audit receipt JSON \
         via bash printf — the structured emitter (src/testing/receipt.rs + \
         src/bin/nam_long_receipt.rs) is the single generator."
    );
}

/// Ensures every `.nam` model referenced by an `#[ignore]` test
/// (golden-missing reason) is registered in the canonical golden registry
/// `src/testing/catalog.rs::GOLDEN_GEN_CATALOG`. This prevents silent drift
/// where a model is added to test files but forgotten in the golden generator.
///
/// ## Scope
///
/// Currently scans `tests/golden_vectors.rs` and `tests/linear_fft_test.rs`.
/// `tests/cpp_parity.rs` is excluded because its `#[ignore]` is due to
/// C++ toolchain dependency, not missing goldens.
#[test]
fn test_ignored_models_are_in_catalog() {
    let catalog = rust_catalog_models();

    let test_files: &[(&str, &str)] = &[
        ("tests/models/golden_vectors.rs", "golden_vectors.rs"),
        ("tests/models/linear_fft_test.rs", "linear_fft_test.rs"),
        (
            "tests/parity/reference_oracle_f64.rs",
            "reference_oracle_f64.rs",
        ),
    ];

    let mut checked = 0usize;

    for (rel_path, source_name) in test_files {
        let full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel_path);
        let models = scan_ignored_test_models(&full_path, source_name);

        for (nam_file, _, fn_sig) in &models {
            assert!(
                catalog.contains(nam_file.as_str()),
                "Model '{nam_file}' is referenced by an #[ignore] test in {source_name} \
                 (fn: {fn_sig}) but is NOT in GOLDEN_GEN_CATALOG.\n\
                 \n\
                 Add '{nam_file}' to src/testing/catalog.rs::GOLDEN_GEN_CATALOG.\n\
                 \n\
                 This is the guard against coherence bugs: models that silently enter \
                 the test suite but never have their goldens generated because the \
                 generator registry was never updated.",
            );
            checked += 1;
        }
    }

    assert!(
        checked >= 10,
        "Only {checked} models were checked for catalog coherence. \
         Expected ≥ 10 (the v2 golden_vectors.rs subset plus golden-missing exceptions). \
         Possible regression: scanner is missing valid .nam references or the test \
         structure has changed.",
    );
}

/// Loads `docs/quality-contract.json` (the canonical machine contract, R-01)
/// and ensures every fidelity entry has a unique `id` and a unique `label`,
/// with no label being a prefix of another — preventing collisions like
/// "Quick A2-Full" vs "Quick A2-Full v2". The future `--check` matches by
/// `id` (exact); the label prefix defense stays as residual protection.
#[test]
fn test_quality_contract_uniqueness() {
    let contract_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("docs")
        .join("quality-contract.json");
    let content =
        fs::read_to_string(&contract_path).expect("Failed to read docs/quality-contract.json");
    let contract = QualityContract::from_json_str(&content)
        .expect("docs/quality-contract.json must validate against the QA schema");

    assert!(
        contract.fidelity.len() >= 15,
        "Expected ≥ 15 fidelity entries, found {}. Contract may be incomplete.",
        contract.fidelity.len()
    );

    let ids: Vec<&str> = contract.fidelity.iter().map(|f| f.id.as_str()).collect();
    let labels: Vec<&str> = contract.fidelity.iter().map(|f| f.label.as_str()).collect();

    for (i, a) in ids.iter().enumerate() {
        for (j, b) in ids.iter().enumerate() {
            if i == j {
                continue;
            }
            assert!(
                a != b,
                "Duplicate id: '{}' appears twice. The `--check` matches by id.",
                b,
            );
        }
    }

    for (i, a) in labels.iter().enumerate() {
        for (j, b) in labels.iter().enumerate() {
            if i == j {
                continue;
            }
            assert!(
                !a.starts_with(*b),
                "Label collision by prefix: '{}' is a prefix of '{}'.\n\
                 This causes false-green contract verification.\n\
                 Regenerate docs/quality-contract.json via the S1.T3 transcription test.",
                b,
                a,
            );
        }
    }

    eprintln!(
        "  ✓ quality-contract uniqueness: {} ids / {} labels, no prefix collisions.",
        ids.len(),
        labels.len()
    );
}

/// Structural test: ensures the DSP hot-path (`src/dsp/`, `src/models/`, `src/math/`)
/// contains no blocking I/O patterns (`log::`, `println!`, `eprintln!`, `format!`)
/// outside of test modules, `#[cold]`-annotated functions, or constructor paths.
///
/// This prevents regression of blocking I/O bugs where I/O sneaks into the
/// real-time audio callback — the #1 rule of the project.
///
/// ## Exemptions
///
/// Known false-positives or intentional cold-path uses can be listed in
/// `RT_LOG_SAFETY_EXEMPTIONS` with a `file:line` key and documented reason.
#[test]
fn test_rt_logging_safety() {
    use std::collections::BTreeSet;

    const HOT_PATH_DIRS: &[&str] = &["src/dsp", "src/models", "src/math"];
    const BANNED: &[&str] = &["log::", "println!", "eprintln!", "format!"];

    let known_exempt: BTreeSet<String> = BTreeSet::new();

    let cold_path_files: &[&str] = &[
        "src/dsp/cabsim/loader.rs",
        "src/dsp/cabsim/ir_parse.rs",
        "src/dsp/cabsim/ir_resample.rs",
        "src/models/slicing.rs",
        "src/models/a2/model/set_weights.rs",
        "src/models/a2/model/dynamic/build.rs",
        "src/models/convnet/block.rs",
        "src/models/wavenet/post_stack_head.rs",
        "src/models/wavenet/common.rs",
    ];

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut violations: Vec<(String, usize, String)> = Vec::new();

    for dir in HOT_PATH_DIRS {
        let full_dir = manifest_dir.join(dir);
        if !full_dir.is_dir() {
            continue;
        }

        let mut files: Vec<PathBuf> = Vec::new();
        walk_rs_files(&full_dir, &mut files);

        for file_path in &files {
            let content = match fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let rel_path = file_path
                .strip_prefix(&manifest_dir)
                .unwrap_or(file_path)
                .display()
                .to_string();

            // Skip test-only files — they do not run on the RT thread
            if rel_path.ends_with("_test.rs") {
                continue;
            }

            // Skip known cold-path files (loaders, builders, weight-setters)
            if cold_path_files.contains(&rel_path.as_str()) {
                continue;
            }

            let violations_in_file =
                scan_rt_logging_violations(&rel_path, &content, BANNED, &known_exempt);
            violations.extend(violations_in_file);
        }
    }

    if !violations.is_empty() {
        let mut msg = String::from("Banned I/O patterns found in DSP hot-path sources:\n\n");
        for (file, line, detail) in &violations {
            msg.push_str(&format!("  {file}:{line} — {detail}\n"));
        }
        msg.push_str(
            "\nThese patterns cause blocking I/O on the real-time audio thread. \
             Replace with atomic flag setting (RtStatusFlags) and deferred logging \
             in the main thread (housekeeping.rs / telemetry.rs).\n",
        );
        panic!("{msg}");
    }

    eprintln!("  ✓ RT logging safety: no banned patterns in hot-path sources.");
}

fn walk_rs_files(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk_rs_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
}

fn scan_rt_logging_violations(
    rel_path: &str,
    content: &str,
    banned: &[&str],
    known_exempt: &std::collections::BTreeSet<String>,
) -> Vec<(String, usize, String)> {
    let mut violations = Vec::new();

    let mut in_cfg_test = false;
    let mut in_test_mod = false;
    let mut test_mod_brace_depth: u32 = 0;
    let mut in_cold_fn = false;
    let mut cold_fn_brace_depth: u32 = 0;
    let mut block_comment = false;
    let mut saw_cold_attr = false;
    let mut cold_attr_line: usize = 0;

    for (line_idx, raw_line) in content.lines().enumerate() {
        let lineno = line_idx + 1;
        let trimmed = raw_line.trim();

        if block_comment {
            if trimmed.contains("*/") {
                block_comment = false;
            }
            continue;
        }
        if trimmed.starts_with("/*") {
            if !trimmed.contains("*/") {
                block_comment = true;
            }
            continue;
        }
        if trimmed.starts_with("//") || trimmed.starts_with("///") {
            continue;
        }

        // Track #[cfg(test)] gates
        if trimmed.starts_with("#[cfg(test)]") {
            in_cfg_test = true;
            continue;
        }
        if in_cfg_test {
            if trimmed.starts_with("#[cfg(") && trimmed.contains("test") {
                continue;
            }
            if trimmed.starts_with("mod ") || trimmed.starts_with("fn ") {
                in_cfg_test = false;
            } else if trimmed.starts_with('#') || trimmed.is_empty() {
                continue;
            } else {
                in_cfg_test = false;
            }
        }

        // Track test modules
        if trimmed.starts_with("mod tests") || trimmed.starts_with("pub mod tests") {
            in_test_mod = true;
            test_mod_brace_depth = 0;
            continue;
        }
        if in_test_mod {
            test_mod_brace_depth += trimmed.matches('{').count() as u32;
            test_mod_brace_depth =
                test_mod_brace_depth.saturating_sub(trimmed.matches('}').count() as u32);
            if test_mod_brace_depth == 0 && trimmed.contains('}') {
                in_test_mod = false;
            }
            continue;
        }

        // Track #[cold] annotated functions
        if trimmed == "#[cold]" {
            saw_cold_attr = true;
            cold_attr_line = lineno;
            continue;
        }
        if saw_cold_attr
            && (trimmed.starts_with("fn ")
                || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("pub(crate) fn ")
                || trimmed.starts_with("pub(super) fn ")
                || trimmed.starts_with("unsafe fn ")
                || trimmed.starts_with("extern ")
                || trimmed.starts_with("const fn "))
            && trimmed.contains('{')
        {
            saw_cold_attr = false;
            in_cold_fn = true;
            cold_fn_brace_depth = trimmed.matches('{').count() as u32;
            cold_fn_brace_depth =
                cold_fn_brace_depth.saturating_sub(trimmed.matches('}').count() as u32);
            if cold_fn_brace_depth == 0 {
                in_cold_fn = false;
            }
            continue;
        }
        if saw_cold_attr && (lineno - cold_attr_line > 3) {
            saw_cold_attr = false;
        }

        if in_cold_fn {
            cold_fn_brace_depth += trimmed.matches('{').count() as u32;
            cold_fn_brace_depth =
                cold_fn_brace_depth.saturating_sub(trimmed.matches('}').count() as u32);
            if cold_fn_brace_depth == 0 {
                in_cold_fn = false;
                cold_fn_brace_depth = 0;
            }
            continue;
        }

        // `use` imports of `log::` macros are cold-path metadata, not calls
        if trimmed.starts_with("use ") && trimmed.contains("log::") {
            continue;
        }

        // `.map_err(|e| ... format!(...))` are error construction, not hot-path I/O
        if trimmed.contains(".map_err") && trimmed.contains("format!") {
            continue;
        }

        let exempt_key = format!("{rel_path}:{lineno}");
        if known_exempt.contains(&exempt_key) {
            continue;
        }

        for pattern in banned {
            if trimmed.contains(pattern) {
                violations.push((
                    rel_path.to_string(),
                    lineno,
                    format!("found `{pattern}` in `{trimmed}`"),
                ));
                break;
            }
        }
    }

    violations
}
