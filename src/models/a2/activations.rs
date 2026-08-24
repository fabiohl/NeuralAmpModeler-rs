// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Activation functions module for the NAM A2 architecture.
//!
//! This module provides implementations of various non-linear activation functions
//! used in Neural Amp Modeler models, ensuring parity with the
//! original C++ implementation.
//!
//! ## Current scope (fast-path A2-Full/Lite)
//!
//! Only `LeakyReLU { negative_slope: 0.01 }` is exercised by the fast-path
//! (`a2_fast.cpp`), applied uniformly across all 23 layers.
//!
//! ## Reserved for general A2 engine (future)
//!
//! All other `ActivationType` variants are preserved for future support
//! of the complete A2 engine (per-layer heterogeneous activations, FiLM, gating).

/// Activation types supported by NAM A2.
///
/// Only `LeakyReLU` is exercised by the fast-path A2-Full/Lite.
/// Other variants: reserved for general A2 engine (future).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type")]
pub enum ActivationType {
    /// Hyperbolic Tangent (standard).
    /// Reserves general A2 full engine support (future).
    Tanh,
    /// Hyperbolic Tangent with hard saturation within [-1, 1].
    /// Reserves general A2 full engine support (future).
    HardTanh,
    /// Fast rational approximation for Tanh.
    /// Reserves general A2 full engine support (future).
    FastTanh,
    /// Rectified Linear Unit: max(0, x).
    /// Reserves general A2 full engine support (future).
    ReLU,
    /// Leaky ReLU with fixed negative slope (fast-path A2-Full/Lite: slope = 0.01).
    LeakyReLU {
        /// Slope applied when x < 0.
        negative_slope: f32,
    },
    /// Parametric ReLU with per-channel or global negative slope.
    /// Reserves general A2 full engine support (future).
    PReLU {
        /// Vector of slopes for negative values.
        negative_slopes: Vec<f32>,
    },
    /// Logistic Sigmoid function: 1 / (1 + exp(-x)).
    /// Reserves general A2 full engine support (future).
    Sigmoid,
    /// Sigmoid Linear Unit: x * sigmoid(x).
    /// Reserves general A2 full engine support (future).
    SiLU,
    /// Efficient version of the Swish function.
    /// Reserves general A2 full engine support (future).
    HardSwish,
    /// HardTanh with configurable slopes for saturation regions.
    /// Reserves general A2 full engine support (future).
    LeakyHardTanh {
        /// Minimum value for linear saturation.
        min_val: f32,
        /// Maximum value for linear saturation.
        max_val: f32,
        /// Slope applied below `min_val`.
        min_slope: f32,
        /// Slope applied above `max_val`.
        max_slope: f32,
    },
    /// Softsign: x / (1 + |x|).
    /// Reserves general A2 full engine support (future).
    Softsign,
}

use crate::math::common::SimdMath;

/// Trait for applying activation functions on audio buffers.
pub trait ActivationFn {
    /// Applies the activation function in-place on the provided buffer.
    fn apply(&self, data: &mut [f32]);
}

impl ActivationType {
    /// Applies the activation function using the monomorphized SIMD backend `M`.
    ///
    /// Calls `M`'s associated functions directly, bypassing the dynamic
    /// `dispatch_simd!` macro used by the free functions in `crate::math::activations`.
    /// When `M` is known at compile time (e.g., inside a `process_internal<M: SimdMath>`
    /// method), this eliminates runtime redispatch on every activation call.
    ///
    /// # Safety
    /// `data` must be a valid mutable slice. The caller must ensure `M` matches
    /// the CPU ISA capabilities (guaranteed by the top-level `dispatch_simd!`).
    #[inline(always)]
    pub unsafe fn apply_simd<M: SimdMath>(&self, data: &mut [f32]) {
        match self {
            // SAFETY: `data` is a valid mutable slice from the caller of this
            // `unsafe fn` and `M` matches the CPU ISA (top-level `dispatch_simd!`).
            Self::Tanh => unsafe {
                M::tanh_slice(data);
            },
            // SAFETY: `data` is a valid mutable slice and `M` matches the CPU ISA
            // (top-level `dispatch_simd!`).
            Self::HardTanh => unsafe {
                M::hard_tanh_slice(data);
            },
            // SAFETY: `data` is a valid mutable slice and `M` matches the CPU ISA
            // (top-level `dispatch_simd!`).
            Self::FastTanh => unsafe {
                M::fast_tanh_slice(data);
            },
            // SAFETY: `data` is a valid mutable slice and `M` matches the CPU ISA
            // (top-level `dispatch_simd!`).
            Self::ReLU => unsafe {
                M::relu_slice(data);
            },
            // SAFETY: `data` is a valid mutable slice and `M` matches the CPU ISA
            // (top-level `dispatch_simd!`).
            Self::LeakyReLU { negative_slope } => unsafe {
                let slopes = [*negative_slope];
                M::prelu_slice(data, &slopes);
            },
            Self::PReLU { negative_slopes } => {
                if negative_slopes.is_empty() {
                    return;
                }
                // SAFETY: `data` and `negative_slopes` are valid slices provided by
                // the caller of this `unsafe fn`, and `M` matches the CPU ISA
                // (guaranteed by the top-level `dispatch_simd!`).
                unsafe {
                    M::prelu_slice(data, negative_slopes);
                }
            }
            // SAFETY: `data` is a valid mutable slice and `M` matches the CPU ISA
            // (top-level `dispatch_simd!`).
            Self::Sigmoid => unsafe {
                M::sigmoid_slice(data);
            },
            // SAFETY: `data` is a valid mutable slice and `M` matches the CPU ISA
            // (top-level `dispatch_simd!`).
            Self::SiLU => unsafe {
                if crate::math::activations::activation_precision()
                    == crate::math::activations::ActivationPrecision::Standard
                {
                    M::silu_slice_hf(data);
                } else {
                    M::silu_slice(data);
                }
            },
            // SAFETY: `data` is a valid mutable slice and `M` matches the CPU ISA
            // (top-level `dispatch_simd!`).
            Self::HardSwish => unsafe {
                M::hard_swish_slice(data);
            },
            Self::LeakyHardTanh {
                min_val,
                max_val,
                min_slope,
                max_slope,
                // SAFETY: `data` is a valid mutable slice from the caller and `M`
                // matches the CPU ISA (guaranteed by the top-level `dispatch_simd!`).
            } => unsafe {
                M::leaky_hard_tanh_slice(data, *min_val, *max_val, *min_slope, *max_slope);
            },
            // SAFETY: `data` is a valid mutable slice and `M` matches the CPU ISA
            // (top-level `dispatch_simd!`).
            Self::Softsign => unsafe {
                M::softsign_slice(data);
            },
        }
    }
}

impl ActivationFn for ActivationType {
    fn apply(&self, data: &mut [f32]) {
        match self {
            // Tanh (Hyperbolic Tangent): Smooth S-shaped curve that compresses
            // any input value to the range [-1.0, 1.0].
            Self::Tanh => {
                crate::math::activations::tanh_slice(data);
            }
            // HardTanh (Rigid Hyperbolic Tangent): Abrupt saturation (hard clipping/culling).
            // Limits values strictly to a minimum of -1.0 and a maximum of 1.0.
            Self::HardTanh => {
                crate::math::activations::hard_tanh_slice(data);
            }
            // FastTanh: A fast mathematical approximation of the processor's native Tanh function.
            // Uses a rational polynomial to avoid computing slow exponentials.
            Self::FastTanh => {
                crate::math::activations::fast_tanh_slice(data);
            }
            // ReLU (Rectified Linear Unit): Zeros all negative values, letting
            // positive values pass through without any change.
            Self::ReLU => {
                crate::math::activations::relu_slice(data);
            }
            // LeakyReLU: Similar to ReLU, but instead of completely zeroing negative values,
            // applies a small multiplier (fixed negative slope) retaining a fraction of the signal.
            Self::LeakyReLU { negative_slope } => {
                let slopes = [*negative_slope];
                crate::math::activations::prelu_slice(data, &slopes);
            }
            // PReLU (Parametric ReLU): Allows adjustable negative slopes learned by the model.
            // The multiplicative factors can vary per channel or element.
            Self::PReLU { negative_slopes } => {
                if negative_slopes.is_empty() {
                    return;
                }
                crate::math::activations::prelu_slice(data, negative_slopes);
            }
            // Sigmoid: Logistic function that smoothly maps the input between 0.0 and 1.0,
            // widely used to compute gating factors (gates) for switching signals on/off.
            Self::Sigmoid => {
                crate::math::activations::sigmoid_slice(data);
            }
            // SiLU (Sigmoid Linear Unit): Multiplies the input value by its own sigmoid
            // activation. Also known as Swish.
            Self::SiLU => {
                crate::math::activations::silu_slice(data);
            }
            // HardSwish: A linear approximation of the Swish/SiLU function designed to be
            // computed efficiently without calculating complex exponential functions.
            Self::HardSwish => {
                crate::math::activations::hard_swish_slice(data);
            }
            // LeakyHardTanh: A hybrid version that acts like HardTanh, but in the saturation
            // zones (outside min_val/max_val) the signal continues to grow with attenuated gain.
            Self::LeakyHardTanh {
                min_val,
                max_val,
                min_slope,
                max_slope,
            } => {
                crate::math::activations::leaky_hard_tanh_slice(
                    data, *min_val, *max_val, *min_slope, *max_slope,
                );
            }
            // Softsign: Smooth symmetric curve similar to Tanh, given by the formula x / (1 + |x|),
            // being smoother at the edges and cheaper to compute on the processor.
            Self::Softsign => {
                crate::math::activations::softsign_slice(data);
            }
        }
    }
}

#[cfg(test)]
#[path = "a2_activations_test.rs"]
mod tests;
