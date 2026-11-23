// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Human-facing dashboard renderer.
//!
//! Pure `QualityReport → String`: no I/O, no cargo, no environment access —
//! the renderer only formats a typed report. The dashboard bash wrapper and
//! the `nam_quality render` binary both pipe this output; the long/perf
//! suites never call it (they stay WARNING/ERROR + forensic JSONL only).
//!
//! The `QualityReport` is the typed superset of the verify report: on top of
//! the `verify.rs` phase/fidelity/latency streams it carries the extra
//! oracle/ISA/activation/coverage kinds that the verification sink writes
//! (`f64_table`, `f64_decomp`, `activation`, `isa`, `coverage_matrix`,
//! `test_counts`) plus the `build_metadata` provenance record.
//!
//! Design invariants:
//! - Section order matches the current dashboard: header → quick summary →
//!   fidelity details → performance → ISA parity → activation precision →
//!   f64 decomposition → spectral summary → coverage matrix → footer.
//! - English-only (unified to international English).
//! - `NOT_VERIFIED` performance is **never** rendered green.
//! - ESR / CPU-budget colors follow the qualitative thresholds of the old
//!   bash render (green < 1e-5, yellow < 1e-1, red ≥ 1e-1; CPU headroom
//!   green > 50 %, yellow > 25 %, red otherwise).
//! - The f64-decomposition cold-start note is emitted in English.

mod parse;
mod sections;
mod types;

pub use parse::{parse_quality_report, parse_quality_report_file};
pub use types::{
    ActivationRow, CoverageMatrix, F64Decomp, F64TableRow, IsaRow, QualityReport, RenderStyle,
    ReportError, ReportHeader, TestCounts,
};

use sections::{
    Palette, render_activation_precision, render_coverage_matrix, render_f64_decomposition,
    render_fidelity_details, render_footer, render_header, render_isa_parity, render_performance,
    render_quick_summary, render_spectral_summary,
};

// ── Top-level render ────────────────────────────────────────────────────────

/// Renders the typed report in dashboard section order.
pub fn render_quality_report(report: &QualityReport, style: RenderStyle) -> String {
    let p = Palette::new(style);
    let mut out = String::new();
    out.push_str(&render_header(report, &p));
    out.push_str(&render_quick_summary(report, &p));
    out.push_str(&render_fidelity_details(report, &p));
    out.push_str(&render_performance(report, &p));
    out.push_str(&render_isa_parity(report, &p));
    out.push_str(&render_activation_precision(report, &p));
    out.push_str(&render_f64_decomposition(report, &p));
    out.push_str(&render_spectral_summary(report, &p));
    out.push_str(&render_coverage_matrix(report, &p));
    out.push_str(&render_footer(report, &p));
    out
}

#[cfg(test)]
#[path = "../render_test.rs"]
mod tests;
