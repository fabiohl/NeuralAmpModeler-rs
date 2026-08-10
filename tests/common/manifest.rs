// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use serde::Deserialize;

/// Deserialized entry from the fixture manifest (manifest.json).
#[derive(Deserialize, Debug, Clone)]
pub struct ManifestEntry {
    pub filename: String,
    #[serde(default)]
    pub expected_class: Option<String>,
}
