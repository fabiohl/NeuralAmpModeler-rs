// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Date, path, and timestamp formatting utilities for diagnostics.

use std::time::SystemTime;

/// Converts days since 1970-01-01 to Gregorian (year, month, day).
///
/// Minimal implementation of Howard Hinnant's civil algorithm,
/// without external dependencies.
pub(crate) fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Howard Hinnant's civil date algorithm: https://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

/// Generates the ISO 8601 timestamp for the current moment.
pub(crate) fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();

    let secs = now.as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    let (year, month, day) = days_to_date(days);

    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Helper function to redact paths by replacing `$HOME` with `~` and `$XDG_RUNTIME_DIR` with `$XDG_RUNTIME_DIR`.
/// Returns the raw (unredacted) path if `full` is true.
pub(crate) fn redact_path(p: &std::path::Path, full: bool) -> String {
    if full {
        return p.to_string_lossy().into_owned();
    }

    if let Some(xdg_runtime_dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        let xdg_path = std::path::Path::new(&xdg_runtime_dir);
        if let Ok(stripped) = p.strip_prefix(xdg_path) {
            return format!("$XDG_RUNTIME_DIR/{}", stripped.to_string_lossy());
        }
    }

    if let Some(home_dir) = std::env::var_os("HOME") {
        let home_path = std::path::Path::new(&home_dir);
        if let Ok(stripped) = p.strip_prefix(home_path) {
            return format!("~/{}", stripped.to_string_lossy());
        }
    }

    p.to_string_lossy().into_owned()
}

/// Redacts absolute paths embedded in free-form text (log traces, etc.).
///
/// When `full` is false, scans the string for token sequences that start with `/`
/// and look like absolute paths, and replaces `$HOME` with `~` /
/// `$XDG_RUNTIME_DIR` with `$XDG_RUNTIME_DIR`. When `full` is true, returns
/// the text unchanged.
pub(crate) fn redact_text(text: &str, full: bool) -> String {
    if full {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'/' && (i == 0 || !is_path_byte(bytes[i - 1])) {
            if i > last_end {
                result.push_str(&text[last_end..i]);
            }
            let start = i;
            i += 1;
            while i < bytes.len() && is_path_byte(bytes[i]) {
                i += 1;
            }
            let candidate = &text[start..i];
            if candidate.len() > 1 {
                let path = std::path::Path::new(candidate);
                let redacted = redact_path(path, false);
                result.push_str(&redacted);
            } else {
                result.push_str(candidate);
            }
            last_end = i;
        } else {
            i += 1;
        }
    }
    if last_end < bytes.len() {
        result.push_str(&text[last_end..]);
    }
    result
}

fn is_path_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'/' || b == b'.' || b == b'_' || b == b'-'
}

/// Helper function to format model paths.
/// Default (not full): only prints file basename.
/// Full: prints the raw absolute path.
pub(crate) fn format_model_path(path_str: &str, full: bool) -> String {
    if path_str.is_empty() {
        return String::new();
    }
    if full {
        path_str.to_string()
    } else {
        let path = std::path::Path::new(path_str);
        if let Some(file_name) = path.file_name() {
            file_name.to_string_lossy().into_owned()
        } else {
            path_str.to_string()
        }
    }
}
