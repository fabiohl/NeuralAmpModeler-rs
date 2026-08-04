// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[macro_use]
mod base;
#[macro_use]
mod vnni_bf16;

pub(super) use impl_avx512_gemv;
pub(super) use impl_avx512vnni_bf16_gemv;
