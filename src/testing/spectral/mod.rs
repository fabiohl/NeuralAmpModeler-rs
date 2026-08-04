// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Spectral fidelity measurement suite.
//!
//! Implements:
//! - **Farina exponential sine sweep** (AES Convention 108, 2000):
//!   Simultaneous measurement of impulse response (FR magnitude/phase)
//!   and THD per harmonic order via deconvolution.
//! - **THD+N per AES17**: 997 Hz pure tone, notch-filtered fundamental,
//!   THD+N = RMS(rest) / RMS(total).
//! - **IMD SMPTE/DIN**: two-tone 60 Hz + 7 kHz (4:1 amplitude ratio),
//!   sidebands around the 7 kHz carrier.
//!
//! ```text
//! Farina method:
//!   x(t) = sin[φ(t)],  φ(t) = ω₁·T / ln(ω₂/ω₁) · (exp(t·ln(ω₂/ω₁)/T) − 1)
//!
//!   inverse filter:  f(t) = x(T−t) · β(t)   where β(t) compensates the
//!   −3 dB/octave spectral envelope of the exponential sweep.
//!
//!   deconvolution:   h(t) = y(t) ∗ f(t)     (via FFT multiplication)
//!
//!   After deconvolution, the linear IR appears at delay ≈ 0,
//!   and the k-th harmonic distortion kernel appears at:
//!       Δtₖ = −T · ln(k) / ln(f₂/f₁)   (expressed as positive time lag)
//! ```
//!
//! All routines are pure analytics (no heap allocation in hot loops
//! aside from FFT planner construction) — safe for off-RT QA use.

pub mod farina;
pub mod thd;

pub use farina::*;
pub use thd::*;

/// Returns the next power-of-two ≥ `n`.
pub(crate) fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    if n.is_power_of_two() {
        n
    } else {
        1 << (usize::BITS - (n - 1).leading_zeros())
    }
}

#[cfg(test)]
#[path = "../spectral_test.rs"]
mod tests;
