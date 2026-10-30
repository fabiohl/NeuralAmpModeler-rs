// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Machine-readable test execution receipts, skip classification, and audit reporting.

pub mod capability;
pub mod log_parser;
pub mod long_audit;
pub mod long_phase;

pub use capability::{
    CapabilityReceipt, ReceiptModelEntry, ReceiptSummary, StageDimensions, StageStatus,
    generate_capability_receipt,
};
pub use log_parser::{
    LONG_GAP_MARKERS, colon_detail, count_before, count_tests_executed_from_log, detail_after,
    detect_gap_markers, extract_attr, is_benchmark_time_line, marker_gap_entry, strip_ansi,
    tests_passed_from_output,
};
pub use long_audit::{LongAuditReceipt, LongReceiptError, now_iso8601};
pub use long_phase::{
    CANONICAL_ROSTER, FIDELITY_PHASE_IDS, LongPhaseReceipt, LongPhaseStatus, LongReason,
    PERFORMANCE_PHASE_IDS, PREFLIGHT_PHASE_IDS, is_preflight_id,
};
