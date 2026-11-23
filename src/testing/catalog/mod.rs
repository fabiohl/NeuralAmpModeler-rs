// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Structured catalog and SHA-256 registry for test fixtures and community models.
//!
//! Provides a canonical, deduplicated registry of all 61 catalog paths mapped to
//! 51 unique SHA-256 identities. Classifies support status (47 supported, 2 intentional
//! negative fixtures, 2 known gaps) and tracks all 10 redundant file aliases.
//!
//! Also hosts the **V2 golden catalog** — the single source of truth for which
//! models participate in the V2 multi-SR golden vector matrix, which sample
//! rates each model requires, and which fixture files are expected on disk.
//! `validate_v2_catalog` powers the `catalog_preflight` gate; the shell
//! generator (`tests/fixtures/golden_gen_build.sh`) consumes the same registry
//! through `golden_gen_catalog_lines` (via the `nam_golden_catalog` binary), so
//! no bash array defines the catalog anymore.
//!
//! Also hosts the **V1 golden catalog** (`V1_GOLDEN_CATALOG`) — the single
//! source of truth for the 48 kHz v1 golden vectors (DistributedCore model
//! goldens, the LocalNonDistributable WaveNet Lite golden, and the CabSim
//! convolution goldens). `validate_v1_goldens` powers the same
//! `catalog_preflight` gate; the former bash lists `REQUIRED_GOLDEN_MODELS`,
//! `NONDIST_GOLDEN_MODELS` and `REQUIRED_CABSIM_GOLDENS` in
//! `utils/tests-long.sh` were removed.
//!
//! Also hosts the **Canonical Reference Architecture Manifesto**
//! ([`reference_architectures`], [`ArchitectureFixtureSpec`], [`ArchitectureFamily`])
//! — the host-agnostic, versioned registry enumerating representative model
//! fixtures across all five supported neural network architecture families (WaveNet A1,
//! WaveNet A2, LSTM, ConvNet, Linear FIR/FFT) for profiling, benchmarking,
//! and validation workloads.

mod architectures;
mod golden;
mod models;
mod validation;

pub use architectures::*;
pub use golden::*;
pub use models::*;
pub use validation::*;

#[cfg(test)]
#[path = "../catalog_test.rs"]
mod tests;
