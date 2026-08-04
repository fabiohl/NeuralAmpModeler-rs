// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Reference scalar implementations for Dot Product 4x.
//!
//! Re-exports the scalar implementations centralized in `crate::math::common::scalar_ref`.

pub use crate::math::common::scalar_ref::{
    dot_product_4x_f32_dual_scalar, dot_product_4x_f32_scalar,
    dot_product_4x_interleaved_dual_frame_fallback, dot_product_4x_interleaved_fallback,
    dot_product_8x_f32_scalar, dot_product_16x_f32_scalar,
};
