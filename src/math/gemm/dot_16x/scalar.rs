// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Scalar oracle for Dot Product 16x — exclusive to test/parity validation.
//!
//! This module re-exports the scalar reference implementation from
//! `crate::math::common::scalar_ref` for use as a correctness oracle
//! in unit tests and benchmarks. It MUST NOT be used in any production
//! code path (trait dispatch, model inference, etc.).

pub use crate::math::common::scalar_ref::dot_product_16x_f32_scalar;
