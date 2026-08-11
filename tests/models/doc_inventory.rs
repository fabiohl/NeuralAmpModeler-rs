// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! S8-E8-T03 — Meta-Testes de Inventário de Documentação e Scripts
//!
//! Garante que a documentação (`docs/`) e os scripts (`utils/`) estejam
//! mutuamente sincronizados e que comandos, features e paths referenciados
//! nos documentos reflitam a implementação real.
//!
//! Qualquer divergência entre comandos documentados e a implementação
//! resulta em falha do meta-teste.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn docs_dir() -> PathBuf {
    manifest_dir().join("docs")
}

fn utils_dir() -> PathBuf {
    manifest_dir().join("utils")
}

fn collect_md_files(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_md_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "md") {
                out.push(path);
            }
        }
    }
}

fn collect_all_files(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    if !dir.is_dir() {
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_all_files(&path, out);
            } else if path.is_file() {
                out.push(path);
            }
        }
    }
}

fn relative_path(path: &std::path::Path, base: &std::path::Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// Normalizes text for path extraction: replaces markdown link delimiters
/// `[` `]` `(` `)` with spaces to separate anchor text and link target.
fn strip_markdown_links(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '[' | ']' | '(' | ')' => ' ',
            other => other,
        })
        .collect()
}

/// Extracts `utils/<name>.sh` and `utils/<name>.py` references.
fn extract_script_refs(text: &str) -> HashSet<String> {
    let clean = strip_markdown_links(text);
    let mut refs = HashSet::new();

    let mut i = 0;
    while i + 6 <= clean.len() {
        if &clean.as_bytes()[i..i + 6] == b"utils/" {
            let rest = &clean[i + 6..];
            let end = rest
                .find(|c: char| c.is_whitespace() || c == ')' || c == '`' || c == '"' || c == '<')
                .unwrap_or(rest.len());
            let candidate = &rest[..end];
            if (candidate.ends_with(".sh") || candidate.ends_with(".py"))
                && !candidate.contains("../")
                && candidate.len() <= 120
            {
                refs.insert(format!("utils/{candidate}"));
            }
            i += 6;
        } else {
            i += 1;
        }
    }
    refs
}

/// Extracts feature flag names from text.
fn extract_feature_refs(text: &str) -> HashSet<String> {
    const KNOWN: &[&str] = &[
        "stereo",
        "testing",
        "heap-audit",
        "long_bench",
        "pgo",
        "dynamic-engine",
    ];
    let lower = text.to_lowercase();
    let mut refs = HashSet::new();
    for &feat in KNOWN {
        if lower.contains(&feat.to_lowercase()) {
            refs.insert(feat.to_string());
        }
    }
    refs
}

/// Extracts `src/<path>.rs` references.
fn extract_src_path_refs(text: &str) -> HashSet<String> {
    let clean = strip_markdown_links(text);
    let mut refs = HashSet::new();

    let mut i = 0;
    while i + 4 <= clean.len() {
        if &clean.as_bytes()[i..i + 4] == b"src/" {
            let rest = &clean[i + 4..];
            let end = rest
                .find(|c: char| {
                    c.is_whitespace() || c == ')' || c == '`' || c == '"' || c == '<' || c == '>'
                })
                .unwrap_or(rest.len());
            let candidate = &rest[..end];
            // Strip trailing 'f' if followed by "ile://" (file:// URL contamination)
            let candidate = if let Some(idx) = candidate.find("file://") {
                &candidate[..idx]
            } else {
                candidate
            };
            // Strip URL fragments (e.g. #L10-L20)
            let candidate = if let Some(idx) = candidate.find('#') {
                &candidate[..idx]
            } else {
                candidate
            };
            // Also strip trailing '.' or ',' that may be punctuation
            let candidate = candidate.trim_end_matches(['.', ',']);
            if candidate.ends_with(".rs")
                && candidate.len() > 6
                && !candidate.contains("../")
                && !candidate.contains('{')
                && !candidate.contains('}')
            {
                refs.insert(format!("src/{candidate}"));
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    refs
}

fn read_all_docs() -> String {
    let mut all = String::new();
    let mut md_files = Vec::new();
    collect_md_files(&docs_dir(), &mut md_files);
    let readme = manifest_dir().join("README.md");
    if readme.exists() {
        md_files.push(readme);
    }
    for md in &md_files {
        if let Ok(content) = fs::read_to_string(md) {
            all.push_str(&content);
            all.push('\n');
        }
    }
    all
}

fn parse_cargo_features() -> Vec<String> {
    let cargo_path = manifest_dir().join("Cargo.toml");
    let content = fs::read_to_string(&cargo_path).expect("Failed to read Cargo.toml");
    let mut features = Vec::new();
    let mut in_features = false;
    for line in content.lines() {
        let trimmed = line.trim();
        // Skip comments
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if trimmed == "[features]" {
            in_features = true;
            continue;
        }
        if in_features {
            if trimmed.starts_with('[') {
                break;
            }
            if let Some(name) = trimmed.split('=').next().map(|n| n.trim())
                && !name.is_empty()
                && name != "default"
            {
                features.push(name.to_string());
            }
        }
    }
    features
}

fn collect_scripts() -> Vec<String> {
    let mut all = Vec::new();
    collect_all_files(&utils_dir(), &mut all);
    all.into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "sh" || e == "py"))
        .map(|p| relative_path(&p, &manifest_dir()))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

/// Scripts that may be referenced in historical/postmortem docs but
/// no longer exist (e.g., one-shot debug scripts that were archived).
// Scripts allowed as doc references even when not treated as standalone
// user-facing entry points (shared libs / supply-chain helpers).
const SCRIPT_DOC_EXEMPT: &[&str] = &["utils/_lib.sh", "utils/mod-update.sh"];

/// Known stale source paths in documentation — to be fixed in the
/// docs-rewrite epic (.agents/TODO-findings_NeuralAmpModeler-rs.md, EP-1).
const STALE_SRC_PATH_EXEMPT: &[&str] = &[];

#[test]
fn test_script_refs_in_docs_exist() {
    let all_docs = read_all_docs();
    let refs = extract_script_refs(&all_docs);

    let manifest = manifest_dir();
    let mut missing = Vec::new();

    for script_ref in &refs {
        if SCRIPT_DOC_EXEMPT.contains(&script_ref.as_str()) {
            continue;
        }
        // Self-containment: references must resolve inside this repository.
        let full_path = manifest.join(script_ref);
        if !full_path.exists() {
            missing.push(script_ref.clone());
        }
    }

    if !missing.is_empty() {
        panic!(
            "{} script(s) referenced in docs do NOT exist:\n  {}\n\
             \n\
             Update the documentation to reference existing scripts.",
            missing.len(),
            missing.join("\n  "),
        );
    }

    eprintln!("  ✓ script refs: {} referenced, all exist.", refs.len());
}

#[test]
fn test_all_utils_scripts_are_documented() {
    let scripts = collect_scripts();
    let all_docs = read_all_docs();
    let refs = extract_script_refs(&all_docs);

    let exempt: &[&str] = &["utils/_lib.sh"];

    let mut undocumented = Vec::new();
    for script in &scripts {
        if exempt.contains(&script.as_str()) {
            continue;
        }
        if !refs.contains(script) {
            undocumented.push(script.clone());
        }
    }

    if !undocumented.is_empty() {
        panic!(
            "{} script(s) exist in utils/ but have NO documentation reference:\n  {}\n\
             \n\
             Add a reference to each script in docs/ or README.md.",
            undocumented.len(),
            undocumented.join("\n  "),
        );
    }

    eprintln!(
        "  ✓ script inventory: {}/{} documented ({} exempt).",
        scripts.len() - undocumented.len() - exempt.len(),
        scripts.len() - exempt.len(),
        exempt.len(),
    );
}

#[test]
fn test_feature_flags_in_docs_match_cargo_toml() {
    let declared = parse_cargo_features();
    let declared_set: HashSet<String> = declared.iter().cloned().collect();

    let all_docs = read_all_docs();
    let doc_features = extract_feature_refs(&all_docs);

    let mut stale = Vec::new();
    for feat in &doc_features {
        if !declared_set.contains(feat) {
            stale.push(feat.clone());
        }
    }
    if !stale.is_empty() {
        panic!(
            "{} feature(s) in docs but NOT in Cargo.toml [features]:\n  {}\n\
             Declared: {declared:#?}",
            stale.len(),
            stale.join("\n  "),
        );
    }

    let mut undocumented = Vec::new();
    for feat in &declared {
        if !doc_features.contains(feat) {
            undocumented.push(feat.clone());
        }
    }
    if !undocumented.is_empty() {
        panic!(
            "{} feature(s) in Cargo.toml but NOT in docs:\n  {}\n\
             \n\
             Document these features in docs/ or README.md.",
            undocumented.len(),
            undocumented.join("\n  "),
        );
    }

    eprintln!(
        "  ✓ feature flags: {} declared, {} documented — full match.",
        declared.len(),
        doc_features.len(),
    );
}

#[test]
fn test_src_paths_in_docs_exist() {
    let all_docs = read_all_docs();
    let refs = extract_src_path_refs(&all_docs);

    let manifest = manifest_dir();
    let mut missing = Vec::new();

    for path_ref in &refs {
        if STALE_SRC_PATH_EXEMPT.contains(&path_ref.as_str()) {
            continue;
        }
        // Self-containment: references must resolve inside this repository.
        let full_path = manifest.join(path_ref);
        if !full_path.exists() {
            missing.push(path_ref.clone());
        }
    }

    if !missing.is_empty() {
        panic!(
            "{} source path(s) in docs do NOT exist:\n  {}\n\
             \n\
             Update documentation to reference current file paths.",
            missing.len(),
            missing.join("\n  "),
        );
    }

    eprintln!("  ✓ source paths: {} references, all exist.", refs.len());
}
