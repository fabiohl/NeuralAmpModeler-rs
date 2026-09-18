// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Source code executable guard: Dual-Frame kernel prohibition in WaveNet main loop.
//!
//! Enforces architectural invariant R6: `process_dual_frame_with_mixin` must NEVER
//! be called inside `WaveNetLayer::process_block_internal`.
//!
//! On x86-64-v3 (AVX2/FMA), Temporal Tiling (Dual-Frame) causes a ~19% performance
//! regression due to YMM register pressure (8 accumulators per channel, port 5
//! blend/shuffle contention). Single-Frame processing (`process_single_frame_with_mixin`)
//! is mandatory in the audio hot-path.
//!
//! See `docs/benchmarks.md §"Temporal Tiling (Dual-Frame) on Conv1D"`.

const LAYER_RS_SOURCE: &str = include_str!("../../src/models/wavenet/layer.rs");

/// Scans source code and extracts code lines belonging to `fn_name`,
/// filtering out comments (`//`, `///`, `/* ... */`, `*`).
///
/// Returns a list of `(line_number, violating_code_line)` where `forbidden_symbol`
/// is referenced or called in active code within the specified function.
fn find_forbidden_calls_in_fn(
    source: &str,
    fn_name: &str,
    forbidden_symbol: &str,
) -> Vec<(usize, String)> {
    let mut in_target_fn = false;
    let mut brace_depth = 0;
    let mut in_block_comment = false;
    let mut violations = Vec::new();

    for (idx, raw_line) in source.lines().enumerate() {
        let line_num = idx + 1;
        let trimmed = raw_line.trim();

        // 1. Function boundary detection
        if !in_target_fn {
            if raw_line.contains("fn ") && raw_line.contains(fn_name) {
                in_target_fn = true;
                // Count opening and closing braces on the signature line itself
                let opens = raw_line.matches('{').count();
                let closes = raw_line.matches('}').count();
                brace_depth = opens.saturating_sub(closes);
                continue;
            }
            continue;
        }

        // 2. Track block comment state across lines
        let mut filtered_code = String::new();
        let chars: Vec<char> = raw_line.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if in_block_comment {
                if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '/' {
                    in_block_comment = false;
                    i += 2;
                } else {
                    i += 1;
                }
            } else if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '*' {
                in_block_comment = true;
                i += 2;
            } else if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '/' {
                // Line comment rest of line
                break;
            } else {
                filtered_code.push(chars[i]);
                i += 1;
            }
        }

        // Also ignore pure comment continuation lines starting with '*'
        if trimmed.starts_with('*') || trimmed.starts_with("///") || trimmed.starts_with("//") {
            // Count braces if any (unlikely in pure comments, but keep depth accurate)
            continue;
        }

        // 3. Update brace depth for the active function
        let opens = filtered_code.matches('{').count();
        let closes = filtered_code.matches('}').count();
        brace_depth += opens;

        // 4. Check for forbidden symbol in active code portion of this line
        if filtered_code.contains(forbidden_symbol) {
            violations.push((line_num, raw_line.trim().to_string()));
        }

        brace_depth = brace_depth.saturating_sub(closes);
        if brace_depth == 0 {
            // Exited target function
            break;
        }
    }

    violations
}

#[test]
fn test_wavenet_layer_does_not_call_dual_frame_in_main_loop() {
    let violations = find_forbidden_calls_in_fn(
        LAYER_RS_SOURCE,
        "process_block_internal",
        "process_dual_frame_with_mixin",
    );

    assert!(
        violations.is_empty(),
        "Executable guard failed! Forbidden call to `process_dual_frame_with_mixin` \
         detected in `WaveNetLayer::process_block_internal`:\n\
         {:?}\n\n\
         Architectural Invariant Violation (R6):\n\
         Single-Frame processing (`process_single_frame_with_mixin`) is mandatory on \
         x86-64-v3 (AVX2/FMA). Temporal Tiling (Dual-Frame) introduces a ~19% performance \
         regression due to YMM register pressure (8 accumulators per channel and port 5 \
         blend/shuffle contention).\n\
         See `docs/benchmarks.md §\"Temporal Tiling (Dual-Frame) on Conv1D\"`.",
        violations
    );

    // Also assert that the mandatory single-frame kernel IS present and called in the body
    assert!(
        LAYER_RS_SOURCE.contains("process_single_frame_with_mixin"),
        "Sanity check failed: `process_single_frame_with_mixin` was not found in layer.rs"
    );
}

#[test]
fn test_wavenet_layer_runtime_invariant_chunks_remainder_empty() {
    // S5-T5 requirement 2: Confirm debug_assert!(chunks.into_remainder().is_empty()) is present
    assert!(
        LAYER_RS_SOURCE.contains("debug_assert!(chunks.into_remainder().is_empty())"),
        "Runtime invariant missing: `debug_assert!(chunks.into_remainder().is_empty())` \
         must be present in `src/models/wavenet/layer.rs`."
    );
}

#[test]
fn test_guard_detects_forbidden_call_in_synthetic_snippet() {
    let synthetic_bad = r#"
    pub unsafe fn process_block_internal<M: SimdMath>(&mut self, ctx: WavenetProcessContext<'_>) {
        // Safe comment mentioning process_dual_frame_with_mixin
        for (i, frame) in chunks.by_ref().enumerate() {
            self.conv1d.process_dual_frame_with_mixin::<M>(layer_buffer, frame);
        }
    }
    "#;

    let violations = find_forbidden_calls_in_fn(
        synthetic_bad,
        "process_block_internal",
        "process_dual_frame_with_mixin",
    );

    assert_eq!(
        violations.len(),
        1,
        "Guard must detect forbidden dual-frame call in code"
    );
    assert!(
        violations[0]
            .1
            .contains("self.conv1d.process_dual_frame_with_mixin::<M>")
    );
}

#[test]
fn test_guard_ignores_comments_referencing_forbidden_symbol() {
    let synthetic_clean_with_comments = r#"
    pub unsafe fn process_block_internal<M: SimdMath>(&mut self, ctx: WavenetProcessContext<'_>) {
        // NUNCA usar process_dual_frame_with_mixin como laço principal
        /// Doc comment process_dual_frame_with_mixin
        /* Block comment process_dual_frame_with_mixin */
        * continuation process_dual_frame_with_mixin
        self.conv1d.process_single_frame_with_mixin::<M>(); // inline process_dual_frame_with_mixin
    }
    "#;

    let violations = find_forbidden_calls_in_fn(
        synthetic_clean_with_comments,
        "process_block_internal",
        "process_dual_frame_with_mixin",
    );

    assert!(
        violations.is_empty(),
        "Guard must ignore comments mentioning the forbidden symbol: {:?}",
        violations
    );
}
