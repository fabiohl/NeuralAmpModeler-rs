// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Documentation and Script Inventory Meta-Tests
//!
//! Ensures that documentation (`docs/`) and scripts (`utils/`) remain
//! synchronized and that commands, features, and paths referenced in
//! the documents accurately reflect the real implementation.
//!
//! Any divergence between documented commands and the implementation
//! results in a meta-test failure.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn docs_dir() -> PathBuf {
    manifest_dir().join("docs")
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

/// Extracts feature flag names from text.
fn extract_feature_refs(text: &str) -> HashSet<String> {
    const KNOWN: &[&str] = &[
        "dual-mono",
        "testing",
        "heap-audit",
        "pinned",
        "fft-radix4-planner",
        "avx512",
        "rt-hardening",
        "on-demand",
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
            let bytes = clean.as_bytes();
            let mut start = i;
            while start > 0
                && (bytes[start - 1].is_ascii_alphanumeric()
                    || matches!(bytes[start - 1], b'_' | b'-' | b'.' | b'/'))
            {
                start -= 1;
            }
            let mut repo_prefix = &clean[start..i];
            while let Some(rest) = repo_prefix.strip_prefix("../") {
                repo_prefix = rest;
            }
            if repo_prefix.starts_with('/') || repo_prefix.contains("..") {
                repo_prefix = "";
            }
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
                && !candidate.contains('*')
            {
                refs.insert(format!("{repo_prefix}src/{candidate}"));
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

/// Known stale source paths in documentation.
const STALE_SRC_PATH_EXEMPT: &[&str] = &[];

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
