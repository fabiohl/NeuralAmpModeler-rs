// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernels de Dot Product 16x (16 canais de saída simultâneos) — AVX2 e AVX-512.
//!
//! Processa 16 output channels por chamada. No baseline x86-64-v3 (AVX2/FMA),
//! duas leituras `__m256` (lo/hi) cobrem os 16 pesos por linha. Em AVX-512F,
//! um único `__m512` cobre todos os 16 pesos.

pub mod dot_f32_avx2;
pub mod dot_f32_avx512;
pub mod scalar;

pub use dot_f32_avx2::*;
pub use dot_f32_avx512::*;
pub use scalar::*;

#[cfg(test)]
#[path = "dot_16x_test.rs"]
mod dot_16x_test;
