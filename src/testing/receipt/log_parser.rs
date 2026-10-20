// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Log parsing engine for extracting test counts and detecting gap markers.

use std::fs;
use std::path::Path;

/// Canonical typed gap marker signatures emitted by long-suite phase scripts.
///
/// Order matters: more specific markers (`[STATUS] SKIP_CAPABILITY`) are
/// checked before broader prefixes. A phase log containing any of these
/// needles is considered to have declared a gap.
///
/// The legacy free-form `SKIP:` prefix is still recognized during the
/// transition (fallback safety net) as `legacy_skip:<detail>` so
/// an unconverted print is never silently promoted to a clean `PASSED`.
pub const LONG_GAP_MARKERS: &[(&str, &str)] = &[
    ("INCONCLUSIVE_ENVIRONMENT", "inconclusive_environment"),
    ("[STATUS] SKIP_CAPABILITY", "skip_capability"),
    ("[STATUS] SKIP_OPTIONAL", "skip_optional"),
    ("[STATUS] KNOWN_GAP", "known_gap"),
    ("[STATUS] INCONCLUSIVE", "inconclusive"),
    ("MISSING-REQUIRED:", "missing_required"),
    // The default local runner compiles the long suite without
    // `--features avx512`, so the cross-ISA AVX-512 subphase can legitimately
    // execute zero cases. The explicit declaration below prevents that from
    // being silently promoted to a clean PASSED verdict.
    ("AVX512_OPT_IN: NOT_RUN", "avx512_opt_in_not_run"),
    // Legacy free-form `SKIP:` prints still found in suites
    // outside the converted scope (tests/models, tests/perf_soak, ...) must
    // never be masked — they surface as a typed transitional gap.
    ("SKIP:", "legacy_skip"),
];

/// Extracts the value of a double-quoted attribute (`name="value"`) from a
/// marker line, if present.
pub fn extract_attr(line: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extracts the trailing text after a literal prefix (`MISSING-REQUIRED: ...`),
/// trimming surrounding whitespace, if the prefix is present.
pub fn detail_after(line: &str, prefix: &str) -> Option<String> {
    let pos = line.find(prefix)?;
    let detail = line[pos + prefix.len()..].trim();
    (!detail.is_empty()).then(|| detail.to_string())
}

/// Extracts the canonical colon-form detail
/// (`[STATUS] SKIP_<MOTIVO>: <detalhes>`) immediately after `needle`, if the
/// marker line uses that dialect. Returns `None` for the attribute form
/// (`reason="..."`) and for lines without a `: ` suffix.
///
/// The match is located with `find` (like `detail_after`/`extract_attr`) so a
/// marker occurring mid-line — `detect_gap_markers` matches on `.contains` —
/// still yields the correct detail instead of a byte-offset slice.
pub fn colon_detail(line: &str, needle: &str) -> Option<String> {
    let pos = line.find(needle)?;
    let rest = line[pos + needle.len()..].trim_start();
    let detail = rest.strip_prefix(':')?.trim_start();
    if detail.is_empty() || detail.starts_with('"') {
        return None;
    }
    Some(detail.to_string())
}

/// Computes the gap entry string for a matched marker line, attaching the
/// typed detail when the grammar carries one (`reason=`/`id=`/colon detail).
pub fn marker_gap_entry(needle: &str, id: &str, line: &str) -> String {
    let detail = match needle {
        "[STATUS] KNOWN_GAP" => extract_attr(line, "id")
            .or_else(|| extract_attr(line, "reason"))
            .or_else(|| colon_detail(line, needle)),
        "[STATUS] SKIP_CAPABILITY" | "[STATUS] SKIP_OPTIONAL" | "[STATUS] INCONCLUSIVE" => {
            extract_attr(line, "reason").or_else(|| colon_detail(line, needle))
        }
        "MISSING-REQUIRED:" => detail_after(line, "MISSING-REQUIRED:"),
        "SKIP:" => detail_after(line, "SKIP:"),
        _ => None,
    };
    match detail {
        Some(detail) => format!("{id}:{detail}"),
        None => id.to_string(),
    }
}

/// Scans a phase log for canonical typed gap markers, returning the stable
/// gap identifiers (with detail suffixes, e.g. `skip_optional:model_not_found`)
/// in canonical order (deduplicated).
///
/// **all** occurrences are accumulated per phase — a phase log with
/// several distinct skips of the same family yields one gap entry per unique
/// skip, never just the first line of each marker family. Deduplication keeps
/// repeated emissions of the same reason from inflating the array while a
/// second distinct reason is never swallowed.
///
/// Fail-closed: an unreadable log is itself a deviation and yields
/// `log_unreadable` — a missing/corrupt phase log can never be promoted to a
/// clean `PASSED` with `gaps: []`.
pub fn detect_gap_markers(path: &Path) -> Vec<String> {
    let Ok(content) = fs::read_to_string(path) else {
        return vec!["log_unreadable".to_string()];
    };
    let mut gaps: Vec<String> = Vec::new();
    for (needle, id) in LONG_GAP_MARKERS {
        for line in content.lines().filter(|l| l.contains(needle)) {
            let entry = marker_gap_entry(needle, id, line);
            if !gaps.contains(&entry) {
                gaps.push(entry);
            }
        }
    }
    gaps
}

/// Extracts the integer immediately preceding `needle` in `line` (the libtest
/// `test result: ok. 123 passed; 0 failed; ...` counter format).
pub fn count_before(line: &str, needle: &str) -> u64 {
    let Some(pos) = line.find(needle) else {
        return 0;
    };
    let bytes = line.as_bytes();
    let mut start = pos;
    while start > 0 && bytes[start - 1].is_ascii_digit() {
        start -= 1;
    }
    if start == pos {
        return 0;
    }
    line[start..pos].parse::<u64>().unwrap_or(0)
}

/// `true` when `line` matches the criterion-style benchmark signature that
/// `_lib.sh::assert_ran_tests` greps as `^\S.*time:\s+\[` (e.g.
/// `soak_test/bench1  time:   [1.1 ms 1.2 ms 1.3 ms]`).
pub fn is_benchmark_time_line(line: &str) -> bool {
    let Some(pos) = line.find("time:") else {
        return false;
    };
    let rest = &line[pos + "time:".len()..];
    let trimmed = rest.trim_start();
    trimmed.starts_with('[')
}

/// Counts tests/benchmarks actually executed by a phase from its log.
///
/// This is the single counter behind `_lib.sh::assert_ran_tests` (the
/// shell delegates to `nam_long_receipt count-log`). It mirrors the bash
/// semantics: `passed`/`failed` are summed from every `test result:` line (a
/// FAILED result line still proves execution; the bash port only read the
/// `ok.` lines for `passed`), `measured` is counted on ANY line (the bash
/// `grep -oP '\K\d+(?=\s+measured)'`), and when no counter was found it falls
/// back to counting criterion-style `time: [...]` benchmark lines, mirroring
/// the bash `^\S.*time:\s+\[` fallback. `ignored`/`filtered out` tests did
/// NOT execute and are never counted.
pub fn count_tests_executed_from_log(path: &Path) -> u64 {
    let Ok(content) = fs::read_to_string(path) else {
        return 0;
    };
    let mut total = 0u64;
    for line in content.lines() {
        if line.contains("test result:") {
            total += count_before(line, " passed") + count_before(line, " failed");
        }
        total += count_before(line, " measured");
    }
    if total == 0 {
        total = content
            .lines()
            .filter(|l| is_benchmark_time_line(l))
            .count() as u64;
    }
    total
}
