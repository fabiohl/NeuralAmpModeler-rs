// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::half::f16_bits_to_f32;
use crate::math::common::kahan::Kahan4F32;
use crate::math::common::kahan::KahanF32;

/// 4-lane interleaved dot product implementations (scalar reference and fallback).
#[path = "dot_4x.rs"]
pub mod dot_4x;
/// 8-lane and 16-lane interleaved dot product implementations (scalar reference).
#[path = "dot_8x16x.rs"]
pub mod dot_8x16x;
pub use dot_4x::*;
pub use dot_8x16x::*;

// ── Base & Fallback Functions ──────────────────────────────────────────

/// Computes the "Dot Product" between two sets of numbers.
/// Imagine multiplying each item from one list by the corresponding item of another list
/// and, in the end, summing everything up.
///
/// - `a`: List of decimal numbers (f32).
/// - `b`: List of "weights" stored in compact form (u16/f16).
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn dot_product_fallback(a: &[f32], b: &[u16]) -> f32 {
    // We pick the size of the smaller list to ensure we won't "run over" memory.
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32; // Start the sum at zero.

    for i in 0..len {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe {
            // The weight 'b' is "shrunken" (f16). Here we transform it back into
            // a normal decimal number (f32) so we can do the math.
            let fb = f16_bits_to_f32(*b.get_unchecked(i));

            // We multiply the input value 'a' by the weight 'fb' and add to the total.
            // `get_unchecked` is like telling the computer: "Go straight to this address,
            // I guarantee it exists", which saves a bit of speed.
            sum += *a.get_unchecked(i) * fb;
        }
    }
    sum // Returns the final result of the sum.
}

/// Version of the Dot Product for the "BF16" (Brain Floating Point) format.
/// This is a decimal number format widely used in AI because
/// it takes half the space but preserves the "scale" of large numbers.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn dot_product_bf16_fallback(a: &[u16], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32;

    for i in 0..len {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe {
            // Here we do some bit "magic": we shift the number 16 places to the left.
            // This transforms the compact BF16 format back into standard f32 decimal.
            let fa = f32::from_bits((*a.get_unchecked(i) as u32) << 16);
            let fb = f32::from_bits((*b.get_unchecked(i) as u32) << 16);

            // Multiply and accumulate into the sum.
            sum += fa * fb;
        }
    }
    sum
}

/// Native f32 dot_product for mixed-precision head projection.
///
/// Used for the final head weights (WaveNet head_rechannel, LSTM head_weights)
/// when running in full FP32 precision, while the backbone uses quantized
/// (BF16/F16) weights for performance.
#[inline(always)]
pub fn dot_product_f32_native(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let a_sub = &a[..len];
    let b_sub = &b[..len];
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a_sub[i] * b_sub[i];
    }
    sum
}

/// Kahan-compensated native f32 dot product for mixed-precision head projection.
///
/// Uses [`KahanF32`] to bound the accumulated floating-point error at O(ε) instead
/// of O(N·ε) in the naive [`dot_product_f32_native`]. Recommended for LSTM heads
/// with long channel counts (> 32 accumulators) where error drift can degrade
/// audio fidelity.
#[inline(always)]
pub fn dot_product_f32_native_kahan(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let a_sub = &a[..len];
    let b_sub = &b[..len];
    let mut acc = KahanF32::new(0.0);
    for i in 0..len {
        acc.add(a_sub[i] * b_sub[i]);
    }
    acc.value()
}

/// Kahan-compensated native f32 dot product with 4 independent Kahan lanes.
///
/// Accumulates the dot product across 4 independent [`Kahan4F32`] channels
/// (four parallel compensation chains) and sums the lanes at the end. This
/// preserves the O(ε) error class of [`dot_product_f32_native_kahan`] while
/// breaking the single sequential compensation dependency chain — the four
/// lanes are independent and can be executed in parallel by the CPU (ILP) or
/// vectorized by the compiler.
///
/// Used for the LSTM final head projection (P-02) where the head weight
/// vectors exceed 32 channels and Kahan compensation is required.
#[inline(always)]
pub fn dot_product_f32_native_kahan4(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let a_sub = &a[..len];
    let b_sub = &b[..len];
    let mut acc = Kahan4F32::new([0.0; 4]);
    let mut i = 0;
    while i + 4 <= len {
        let x = [
            a_sub[i] * b_sub[i],
            a_sub[i + 1] * b_sub[i + 1],
            a_sub[i + 2] * b_sub[i + 2],
            a_sub[i + 3] * b_sub[i + 3],
        ];
        acc.add(x);
        i += 4;
    }
    while i < len {
        acc.add_channel(0, a_sub[i] * b_sub[i]);
        i += 1;
    }
    let s = acc.value();
    s[0] + s[1] + s[2] + s[3]
}

/// Computes 4 dot products at once for BF16.
/// This is a shortcut to call the single-line function 4 times.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn dot_product_bf16_4x_fallback(
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    in_frame: &[u16],
) -> [f32; 4] {
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    unsafe {
        [
            dot_product_bf16_fallback(in_frame, w0),
            dot_product_bf16_fallback(in_frame, w1),
            dot_product_bf16_fallback(in_frame, w2),
            dot_product_bf16_fallback(in_frame, w3),
        ]
    }
}

#[cfg(test)]
mod kahan_dot_tests {
    use super::*;

    #[test]
    fn test_kahan_naive_agree_small() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let naive = dot_product_f32_native(&a, &b);
        let kahan = dot_product_f32_native_kahan(&a, &b);
        assert!((naive - kahan).abs() < 1e-6, "naive={naive}, kahan={kahan}");
    }

    #[test]
    fn test_kahan_advantage_high_dynamic_range() {
        let n = 10_000;
        let big = 1.0f32;
        let tiny = 1e-7f32;
        let a: Vec<f32> = std::iter::repeat_n(1.0, n).collect();
        let mut b: Vec<f32> = vec![big];
        b.extend(std::iter::repeat_n(tiny, n - 1));

        let naive = dot_product_f32_native(&a, &b);
        let kahan = dot_product_f32_native_kahan(&a, &b);
        // f64 reference
        let ref_result = big as f64 + (n - 1) as f64 * tiny as f64;

        let naive_err = (naive as f64 - ref_result).abs() / ref_result.abs();
        let kahan_err = (kahan as f64 - ref_result).abs() / ref_result.abs();

        assert!(
            kahan_err < naive_err,
            "Kahan error ({kahan_err:.3e}) should be smaller than naive ({naive_err:.3e})"
        );
    }

    #[test]
    fn test_kahan_empty_slices() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        assert_eq!(dot_product_f32_native_kahan(&a, &b), 0.0);
    }

    #[test]
    fn test_kahan_single_element() {
        assert_eq!(dot_product_f32_native_kahan(&[3.0], &[4.0]), 12.0);
    }

    #[test]
    fn test_kahan_different_lengths() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];
        let result = dot_product_f32_native_kahan(&a, &b);
        assert_eq!(result, 1.0 * 4.0 + 2.0 * 5.0);
    }

    #[test]
    fn test_kahan_deep_accumulation_drift() {
        let total_terms: usize = 15 * 3 * 256;
        let mut rng_state: u32 = 0xCAFE_BABE;
        let mut next_f32 = move || -> f32 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let bits = (rng_state >> 9) | 0x3F80_0000;
            f32::from_bits(bits) - 1.0
        };

        let mut a = Vec::with_capacity(total_terms);
        let mut b = Vec::with_capacity(total_terms);
        let mut f64_ref = 0.0f64;
        for i in 0..total_terms {
            let large = (next_f32() * 1e4).abs() + 1e3;
            let tiny = next_f32() * 1e-7;
            let a_val = if i % 128 == 0 {
                large
            } else if i % 64 == 0 {
                large * 0.1
            } else {
                tiny
            };
            let b_val = 1.0f32;
            a.push(a_val);
            b.push(b_val);
            f64_ref += a_val as f64 * b_val as f64;
        }

        let naive = dot_product_f32_native(&a, &b);
        let kahan = dot_product_f32_native_kahan(&a, &b);

        let naive_err = (naive as f64 - f64_ref).abs() / f64_ref.abs().max(1e-30);
        let kahan_err = (kahan as f64 - f64_ref).abs() / f64_ref.abs().max(1e-30);
        let naive_drift_db = 20.0 * naive_err.max(1e-30).log10();
        let kahan_drift_db = 20.0 * kahan_err.max(1e-30).log10();
        let improvement_db = naive_drift_db - kahan_drift_db;

        assert!(
            improvement_db >= 2.0,
            "Kahan improvement ({improvement_db:.2} dB) < 2 dB"
        );
    }
}
