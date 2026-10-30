// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Modularized Fused Residual Batch GEMM kernels across ISAs.

mod avx2;
#[cfg(nam_avx512)]
mod avx512;
mod scalar;

pub use avx2::*;
#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub use avx512::*;
pub use scalar::*;

#[cfg(test)]
#[path = "fused_residual_batch_test.rs"]
mod tests;
