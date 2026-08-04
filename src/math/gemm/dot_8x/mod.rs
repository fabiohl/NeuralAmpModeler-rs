// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernels de Dot Product 8x (8 canais de saída simultâneos) — AVX2/FMA.
//!
//! Processa 8 output channels por chamada, reduzindo o número de iterações
//! do loop externo e melhorando o reuso de registradores.

pub mod dot_f32_avx2;
pub mod scalar;

pub use dot_f32_avx2::*;
pub use scalar::*;

#[cfg(test)]
#[path = "dot_8x_test.rs"]
mod dot_8x_test;
