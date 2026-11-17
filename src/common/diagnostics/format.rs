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
#[doc(hidden)]
pub fn timestamp() -> String {
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

/// Checks whether `path_str` is an absolute POSIX path, Windows drive path (`C:\...` / `C:/...`),
/// or Windows UNC path (`\\server\share...`).
pub(crate) fn is_portable_absolute_path(path_str: &str) -> bool {
    let bytes = path_str.as_bytes();
    if bytes.starts_with(b"/") || bytes.starts_with(b"\\\\") {
        return true;
    }
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
}

fn looks_like_windows_path(s: &str) -> bool {
    let bytes = s.as_bytes();
    s.contains('\\')
        || bytes.starts_with(b"\\\\")
        || (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
}

fn normalize_slashes(s: &str) -> String {
    s.replace('\\', "/")
}

fn normalize_portable_path(s: &str) -> String {
    let norm = normalize_slashes(s);
    let bytes = norm.as_bytes();
    if bytes.len() >= 8 && &bytes[..4] == b"//?/" && bytes[4..8].eq_ignore_ascii_case(b"unc/") {
        return format!("//{}", &norm[8..]);
    }
    if bytes.len() >= 7
        && &bytes[..4] == b"//?/"
        && bytes[4].is_ascii_alphabetic()
        && bytes[5] == b':'
        && bytes[6] == b'/'
    {
        return norm[4..].to_string();
    }
    norm
}

fn strip_dir_prefix_portable(path_str: &str, prefix_str: &str) -> Option<String> {
    if prefix_str.is_empty() {
        return None;
    }
    let windows_mode = looks_like_windows_path(path_str) || looks_like_windows_path(prefix_str);
    let norm_path = normalize_portable_path(path_str);
    let mut norm_prefix = normalize_portable_path(prefix_str);
    while norm_prefix.len() > 1 && norm_prefix.ends_with('/') {
        norm_prefix.pop();
    }
    if norm_prefix.is_empty() || norm_path.len() < norm_prefix.len() {
        return None;
    }

    let prefix_bytes = norm_prefix.as_bytes();
    let path_head = &norm_path.as_bytes()[..prefix_bytes.len()];
    let matches = if windows_mode {
        path_head.eq_ignore_ascii_case(prefix_bytes)
    } else {
        path_head == prefix_bytes
    };
    if !matches {
        return None;
    }

    let rest = &norm_path[norm_prefix.len()..];
    if norm_prefix != "/" && !rest.is_empty() && !rest.starts_with('/') {
        return None;
    }
    Some(rest.trim_start_matches('/').to_string())
}

fn format_redacted_prefix(token: &str, rel: &str) -> String {
    if rel.is_empty() {
        token.to_string()
    } else {
        format!("{token}/{rel}")
    }
}

/// Pure helper to redact a path against explicit environment directory prefixes.
///
/// Supports `$XDG_RUNTIME_DIR`, `%LOCALAPPDATA%`, `%USERPROFILE%`, and `$HOME`
/// across POSIX and Windows (`C:\...`, `D:/...`, UNC `\\...`) paths.
pub(crate) fn redact_path_from_env(
    path_str: &str,
    full: bool,
    xdg_runtime_dir: Option<&str>,
    local_app_data: Option<&str>,
    user_profile: Option<&str>,
    home_dir: Option<&str>,
) -> String {
    if full {
        return path_str.to_string();
    }

    // Check specific subdirectories before parent home/profile roots.
    for (prefix, token) in [
        (xdg_runtime_dir, "$XDG_RUNTIME_DIR"),
        (local_app_data, "<LOCALAPPDATA>"),
        (home_dir, "~"),
        (user_profile, "<USERPROFILE>"),
    ] {
        if let Some(dir) = prefix
            && let Some(rel) = strip_dir_prefix_portable(path_str, dir)
        {
            return format_redacted_prefix(token, &rel);
        }
    }

    path_str.to_string()
}

/// Helper function to redact paths by replacing `$HOME` with `~`, `$XDG_RUNTIME_DIR`
/// with `$XDG_RUNTIME_DIR`, `%LOCALAPPDATA%` with `<LOCALAPPDATA>`, and `%USERPROFILE%`
/// with `<USERPROFILE>`. Returns the raw (unredacted) path if `full` is true.
pub(crate) fn redact_path(p: &std::path::Path, full: bool) -> String {
    let raw = p.to_string_lossy();
    if full {
        return raw.into_owned();
    }
    let xdg = std::env::var_os("XDG_RUNTIME_DIR").map(|v| v.to_string_lossy().into_owned());
    let local_app_data = std::env::var_os("LOCALAPPDATA").map(|v| v.to_string_lossy().into_owned());
    let user_profile = std::env::var_os("USERPROFILE").map(|v| v.to_string_lossy().into_owned());
    let home = std::env::var_os("HOME").map(|v| v.to_string_lossy().into_owned());
    redact_path_from_env(
        &raw,
        false,
        xdg.as_deref(),
        local_app_data.as_deref(),
        user_profile.as_deref(),
        home.as_deref(),
    )
}

/// Pure helper to redact embedded POSIX (`/…`), Windows drive (`C:\…`, `D:/…`),
/// and Windows UNC (`\\…`) paths in free-form text against explicit environment prefixes.
pub(crate) fn redact_text_from_env(
    text: &str,
    full: bool,
    xdg_runtime_dir: Option<&str>,
    local_app_data: Option<&str>,
    user_profile: Option<&str>,
    home_dir: Option<&str>,
) -> String {
    if full {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // 1. Windows UNC path: \\server\share\... (or \\?\C:\...)
        if i + 2 < bytes.len()
            && bytes[i] == b'\\'
            && bytes[i + 1] == b'\\'
            && (i == 0 || !is_win_path_byte(bytes[i - 1]))
            && (bytes[i + 2].is_ascii_alphanumeric()
                || bytes[i + 2] == b'?'
                || bytes[i + 2] == b'.')
        {
            if i > last_end {
                result.push_str(&text[last_end..i]);
            }
            let start = i;
            let prefix_jump = matched_env_prefix_len(
                &text[start..],
                [xdg_runtime_dir, local_app_data, home_dir, user_profile],
            );
            i += 2;
            if prefix_jump > 0 {
                i = i.max(start + prefix_jump);
            }
            while i < bytes.len() {
                let is_unc_drive_colon = bytes[i] == b':'
                    && i == start + 5
                    && bytes[start + 2] == b'?'
                    && bytes[start + 3] == b'\\'
                    && bytes[start + 4].is_ascii_alphabetic();
                if is_win_path_byte(bytes[i]) || is_unc_drive_colon {
                    i += 1;
                } else {
                    break;
                }
            }
            let candidate = &text[start..i];
            result.push_str(&redact_path_from_env(
                candidate,
                false,
                xdg_runtime_dir,
                local_app_data,
                user_profile,
                home_dir,
            ));
            last_end = i;
        } else if i + 2 < bytes.len()
            // 2. Windows drive-letter path: C:\... or D:/...
            && bytes[i].is_ascii_alphabetic()
            && bytes[i + 1] == b':'
            && (bytes[i + 2] == b'\\' || bytes[i + 2] == b'/')
            && (i == 0 || !is_win_path_byte(bytes[i - 1]))
        {
            if i > last_end {
                result.push_str(&text[last_end..i]);
            }
            let start = i;
            let prefix_jump = matched_env_prefix_len(
                &text[start..],
                [xdg_runtime_dir, local_app_data, home_dir, user_profile],
            );
            i += 3;
            if prefix_jump > 0 {
                i = i.max(start + prefix_jump);
            }
            while i < bytes.len() && is_win_path_byte(bytes[i]) {
                i += 1;
            }
            let candidate = &text[start..i];
            result.push_str(&redact_path_from_env(
                candidate,
                false,
                xdg_runtime_dir,
                local_app_data,
                user_profile,
                home_dir,
            ));
            last_end = i;
        } else if bytes[i] == b'/'
            // 3. POSIX absolute path: /... (skip already-redacted ~/, <USERPROFILE>/, etc.)
            && (i == 0
                || (!is_path_byte(bytes[i - 1])
                    && bytes[i - 1] != b'~'
                    && bytes[i - 1] != b'>'))
        {
            if i > last_end {
                result.push_str(&text[last_end..i]);
            }
            let start = i;
            let prefix_jump = matched_env_prefix_len(
                &text[start..],
                [xdg_runtime_dir, local_app_data, home_dir, user_profile],
            );
            i += 1;
            if prefix_jump > 0 {
                i = i.max(start + prefix_jump);
            }
            while i < bytes.len() && is_path_byte(bytes[i]) {
                i += 1;
            }
            let candidate = &text[start..i];
            if candidate.len() > 1 {
                result.push_str(&redact_path_from_env(
                    candidate,
                    false,
                    xdg_runtime_dir,
                    local_app_data,
                    user_profile,
                    home_dir,
                ));
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

/// Redacts absolute paths embedded in free-form text (log traces, etc.).
///
/// When `full` is false, scans the string for POSIX paths (`/...`), Windows
/// drive paths (`C:\...`, `D:/...`), and UNC paths (`\\...`), replacing
/// configured user/runtime directory prefixes. When `full` is true, returns
/// the text unchanged.
pub(crate) fn redact_text(text: &str, full: bool) -> String {
    if full {
        return text.to_string();
    }
    let xdg = std::env::var_os("XDG_RUNTIME_DIR").map(|v| v.to_string_lossy().into_owned());
    let local_app_data = std::env::var_os("LOCALAPPDATA").map(|v| v.to_string_lossy().into_owned());
    let user_profile = std::env::var_os("USERPROFILE").map(|v| v.to_string_lossy().into_owned());
    let home = std::env::var_os("HOME").map(|v| v.to_string_lossy().into_owned());
    redact_text_from_env(
        text,
        false,
        xdg.as_deref(),
        local_app_data.as_deref(),
        user_profile.as_deref(),
        home.as_deref(),
    )
}

fn matched_env_prefix_len(tail: &str, prefixes: [Option<&str>; 4]) -> usize {
    let tail_bytes = tail.as_bytes();
    let (inner_offset, is_unc_ext) = if tail_bytes.len() >= 8
        && &tail_bytes[..4] == br"\\?\"
        && tail_bytes[4..8].eq_ignore_ascii_case(br"unc\")
    {
        (6, true)
    } else if tail_bytes.len() >= 7
        && &tail_bytes[..4] == br"\\?\"
        && tail_bytes[4].is_ascii_alphabetic()
        && tail_bytes[5] == b':'
        && (tail_bytes[6] == b'\\' || tail_bytes[6] == b'/')
    {
        (4, false)
    } else {
        (0, false)
    };
    let mut best = 0;
    for prefix in prefixes.into_iter().flatten() {
        let p = prefix.trim_end_matches(['/', '\\']);
        if p.is_empty() {
            continue;
        }
        let p_bytes = p.as_bytes();
        let (cmp_tail, total_len) = if is_unc_ext && p_bytes.starts_with(br"\\") {
            (&tail[8..], 8 + (p.len() - 2))
        } else {
            (&tail[inner_offset..], inner_offset + p.len())
        };
        let p_cmp = if is_unc_ext && p_bytes.starts_with(br"\\") {
            &p[2..]
        } else {
            p
        };
        if cmp_tail.len() >= p_cmp.len()
            && cmp_tail.is_char_boundary(p_cmp.len())
            && cmp_tail[..p_cmp.len()]
                .bytes()
                .zip(p_cmp.bytes())
                .all(|(a, b)| {
                    (a == b)
                        || ((a == b'\\' || a == b'/') && (b == b'\\' || b == b'/'))
                        || (looks_like_windows_path(p) && a.eq_ignore_ascii_case(&b))
                })
        {
            let after = &cmp_tail.as_bytes()[p_cmp.len()..];
            if after.is_empty()
                || after[0] == b'\\'
                || after[0] == b'/'
                || after[0].is_ascii_whitespace()
            {
                best = best.max(total_len);
            }
        }
    }
    best
}

fn is_path_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'/' || b == b'.' || b == b'_' || b == b'-' || !b.is_ascii()
}

fn is_win_path_byte(b: u8) -> bool {
    is_path_byte(b) || b == b'\\'
}

/// Extracts only the final file name component across both POSIX (`/`) and
/// Windows (`\`, drive prefixes, UNC) path separators.
pub(crate) fn sanitize_model_basename(path_str: &str) -> String {
    if path_str.is_empty() {
        return String::new();
    }
    let trimmed = path_str.trim_end_matches(['/', '\\']);
    if trimmed.is_empty() {
        return String::new();
    }
    let candidate = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed);
    let bytes = candidate.as_bytes();
    if bytes.len() == 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return String::new();
    }
    candidate.to_string()
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
        let base = sanitize_model_basename(path_str);
        if base.is_empty() {
            path_str.to_string()
        } else {
            base
        }
    }
}
