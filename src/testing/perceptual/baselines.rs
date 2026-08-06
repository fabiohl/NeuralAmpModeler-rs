// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Published baselines from t3k-mushra / A2ESR test framework.
#![allow(dead_code)]

/// A1-Standard median ESR baseline from t3k-mushra/A2Esr.tsx
pub const A2ESR_A1_STANDARD_MEDIAN: f64 = 0.00623;

/// A2-Full median ESR baseline from t3k-mushra/A2Esr.tsx
pub const A2ESR_A2_FULL_MEDIAN: f64 = 0.00334;

/// A2-Lite median ESR baseline (preliminary — pending t3k-mushra publication).
/// A2-Lite shares the A2-Full architecture with rank-reduced weights; ESR is
/// expected to be in the same order of magnitude as A2-Full (≤ 0.005).
pub const A2ESR_A2_LITE_MEDIAN: f64 = 0.005;
/// A2-Lite Q1 ESR baseline (preliminary)
pub const A2ESR_A2_LITE_Q1: f64 = 0.0015;
/// A2-Lite Q3 ESR baseline (preliminary)
pub const A2ESR_A2_LITE_Q3: f64 = 0.012;
