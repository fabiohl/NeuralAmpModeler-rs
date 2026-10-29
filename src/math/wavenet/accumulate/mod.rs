// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Accumulation and activation kernels for WaveNet — AVX2, AVX-512 and scalar fallback.

#[cfg(all(test, nam_avx512))]
mod accumulate_test;
mod avx2;
#[cfg(nam_avx512)]
mod avx512;
#[cfg(nam_avx512)]
mod avx512vl;
#[cfg(test)]
mod gated_test;
mod kernel_macro;
#[cfg(test)]
pub mod scalar;

pub use avx2::*;
#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub use avx512::*;
#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub use avx512vl::*;
#[cfg(test)]
pub use scalar::*;
