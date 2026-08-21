// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! GEMV 4-Gate kernels for LSTM — AVX2 and AVX-512 (including BF16).

mod avx2; // gemv_4gate_avx2
#[cfg(any(test, feature = "testing"))]
mod avx512; // gemv_4gate_avx512 (ZMM 512-bit — benchmark/parity test only)
#[cfg(any(test, feature = "testing"))]
mod avx512_bf16; // gemv_4gate_bf16_avx512 (parity test only)
mod avx512vl;
mod kernel_macro;

pub use avx2::gemv_4gate_avx2;
#[cfg(any(test, feature = "testing"))]
pub use avx512::gemv_4gate_avx512;
#[cfg(any(test, feature = "testing"))]
pub use avx512_bf16::gemv_4gate_bf16_avx512;
pub use avx512vl::gemv_4gate_avx512vl;
