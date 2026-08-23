// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Root module for mathematical operations and neural inference kernels.
//!
//! NeuralAmpModeler-rs organizes its mathematical infrastructure in a modular way to ensure
//! extreme performance (SIMD) and maintainability. This module coordinates the
//! linear algebra kernels, activation functions, and DSP utilities.
//!
//! # Structure
//! - `common`: Traits, dynamic dispatch, and base implementations (AVX2/AVX-512).
//! - `activations`: Optimized activation functions (tanh, sigmoid, etc.).
//! - `gemm`: High-throughput matrix-vector and dot product operations.
//! - `dsp`: Audio processing (gain, stereo, conversion).
//! - `lstm` & `wavenet`: Specialized kernels for each model architecture.

pub mod activations;
pub mod common;
pub mod constants;
pub mod dsp;
pub mod gemm;
pub mod lstm;
pub mod wavenet;
