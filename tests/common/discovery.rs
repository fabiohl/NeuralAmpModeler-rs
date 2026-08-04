// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::fs;
use std::path::{Path, PathBuf};

/// Finds all `.nam` and `.json` model files in the given directory recursively.
pub fn find_models_in_dir(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_models_in_dir(&path));
            } else if path
                .extension()
                .is_some_and(|ext| ext == "nam" || ext == "json")
            {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}
