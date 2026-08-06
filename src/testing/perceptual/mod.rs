// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Perceptual metrics for audio quality evaluation.
//!
//! Includes ESR (Error-to-Signal Ratio), LUFS (ITU-R BS.1770-4 full 2-pass gating),
//! LRA (EBU Tech 3342), true-peak (BS.1770-4 Annex 2), MR-STFT, and baseline
//! constants from published A2/Tone3000 data.

pub mod baselines;
pub mod esr;
pub mod lra;
pub mod lufs;
pub mod true_peak;

pub use baselines::*;
pub use esr::*;
pub use lra::*;
pub use lufs::*;
pub use true_peak::*;

#[cfg(test)]
#[path = "../perceptual_test.rs"]
mod perceptual_test;
