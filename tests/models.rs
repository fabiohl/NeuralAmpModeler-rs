// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Model-level integration and validation test entry point.
//!
//! Aggregates integration submodules for loader verification, golden vector regression,
//! dynamic activation precision, proptest parsers/math, and spectral fidelity.

mod common;

use common::alloc_audit::CountingAllocator;

#[cfg_attr(not(feature = "heap-audit"), global_allocator)]
#[allow(dead_code, clippy::allow_attributes)]
static GLOBAL: CountingAllocator = CountingAllocator;

// ── Model Loading & Parsing Submodules ────────────────────────────────────────
#[path = "models/a2_loader.rs"]
mod a2_loader;
#[path = "models/block_invariance_test.rs"]
mod block_invariance_test;
#[path = "models/catalog_test.rs"]
mod catalog_test;
#[path = "models/container_slimmable.rs"]
mod container_slimmable;
#[path = "models/deterministic_energy_test.rs"]
mod deterministic_energy_test;
#[path = "models/diagnostic_bundle.rs"]
mod diagnostic_bundle;
#[path = "models/doc_inventory.rs"]
mod doc_inventory;
#[path = "models/ebu_lufs_compliance.rs"]
mod ebu_lufs_compliance;
#[path = "models/receipt_test.rs"]
mod receipt_test;
#[path = "models/wavenet_clone_exact_test.rs"]
mod wavenet_clone_exact_test;

// ── Activation Precision & FSM Submodules ────────────────────────────────────
#[path = "models/activation_precision.rs"]
mod activation_precision;
#[path = "models/adaptive_fsm_proptest.rs"]
mod adaptive_fsm_proptest;
#[path = "models/gate_fsm_proptest.rs"]
mod gate_fsm_proptest;

// ── Golden Vector & Parity Submodules ────────────────────────────────────────
#[path = "models/cabsim_golden.rs"]
mod cabsim_golden;
#[path = "models/fixture_b1_2_smoke.rs"]
mod fixture_b1_2_smoke;
#[path = "models/golden_vectors.rs"]
mod golden_vectors;
#[path = "models/linear_fft_test.rs"]
mod linear_fft_test;
#[path = "models/linear_golden.rs"]
mod linear_golden;
#[path = "models/lstm_activation_precision.rs"]
mod lstm_activation_precision;
#[path = "models/lstm_model_dyn_validation.rs"]
mod lstm_model_dyn_validation;
#[path = "models/meta_coherence.rs"]
mod meta_coherence;
#[path = "models/metrics_jsonl_test.rs"]
mod metrics_jsonl_test;
#[path = "models/mirror_buf_fault_injection.rs"]
mod mirror_buf_fault_injection;
#[path = "models/nam_infer_test.rs"]
mod nam_infer_test;
#[path = "models/namb_v2_roundtrip.rs"]
mod namb_v2_roundtrip;
#[path = "models/namb_v2_validation.rs"]
mod namb_v2_validation;
#[path = "models/nondist_validation.rs"]
mod nondist_validation;
#[path = "models/oversampling_characterization.rs"]
mod oversampling_characterization;
#[path = "models/prewarm_test.rs"]
mod prewarm_test;
#[path = "models/process_contract_test.rs"]
mod process_contract_test;
#[path = "models/proptest_math.rs"]
mod proptest_math;
#[path = "models/proptest_parsers.rs"]
mod proptest_parsers;
#[path = "models/self_consistency.rs"]
mod self_consistency;
#[path = "models/spectral_fidelity.rs"]
mod spectral_fidelity;
#[path = "models/thp_coherence.rs"]
mod thp_coherence;
#[path = "models/threshold_calibration.rs"]
mod threshold_calibration;
#[path = "models/wavenet_lite_block_invariance.rs"]
mod wavenet_lite_block_invariance;
#[path = "models/wavenet_prewarm_edge.rs"]
mod wavenet_prewarm_edge;
#[path = "models/zero_alloc_infer.rs"]
mod zero_alloc_infer;
