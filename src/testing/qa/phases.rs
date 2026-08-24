// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Canonical QA phase-identifier constants shared by the report renderer, the
//! verify engine docs and the dashboard receipt emitter.
//!
//! T3.2 (G-02) introduced two distinct ISA phase records so the QA report can
//! never mistake internal consistency for inter-ISA parity:
//!
//! - `ISA_SELF_CONSISTENCY_PHASE` — the local dashboard subphase (AVX2 vs
//!   AVX2, MSE=0).
//! - `ISA_PARITY_CROSS_ISA_PHASE` — the full AVX2 vs AVX-512 matrix, restricted
//!   to the remote/multi-target gate. The local runner declares it as
//!   `SKIP_CAPABILITY` with `CROSS_ISA_GAP_REASON`.
//!
//! The shell emitter (`utils/quality-dashboard.sh`) keeps mirror variables of
//! these three values; `qa_test::dashboard_phase_ids_match_rust_constants`
//! fails if they drift.

/// Local dashboard ISA subphase: AVX2 vs AVX2 internal consistency (MSE=0).
pub const ISA_SELF_CONSISTENCY_PHASE: &str = "isa_self_consistency";

/// Full cross-ISA parity matrix phase (AVX2 vs AVX-512), remote-gate only.
pub const ISA_PARITY_CROSS_ISA_PHASE: &str = "isa_parity_cross_isa";

/// Typed `SKIP_CAPABILITY` reason declaring that the local runner did not
/// (and cannot) execute the cross-ISA matrix.
pub const CROSS_ISA_GAP_REASON: &str = "cross_isa_matrix_requires_avx512_remote";
