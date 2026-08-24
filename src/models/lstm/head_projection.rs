// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Head projection logic for LSTM models.
//!
//! After quantization removal (SQ3), all head weights are f32 native.
//! Head computation uses the 4-lane Kahan accumulator
//! `dot_product_f32_native_kahan4` (O(ε) error class with four independent
//! compensation chains for ILP), inlined in each model's processing kernel.
