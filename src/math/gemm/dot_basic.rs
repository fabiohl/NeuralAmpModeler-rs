// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Dot Product kernels — AVX2 and AVX-512.

use core::arch::x86_64::*;

/// Computes the Dot Product in a blazing-fast manner using hardware acceleration (AVX2).
///
/// This function is the "heart" of many neural network models. Instead of multiplying and adding
/// one number at a time, it processes blocks of data simultaneously (up to 32 numbers at once),
/// making the most of modern processor power.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;

    unsafe {
        // Prepare 4 "accumulators" (sum buckets) to work in parallel.
        // This allows the processor to do multiple sums at the same time without waiting for each to finish.
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        // Main Loop: Processes 32 numbers at a time (4x8 unrolling).
        while i + 32 <= len {
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(va0, vb0, sum0); // Multiply and Add in a single step (FMA)

            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

            let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
            let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
            sum2 = _mm256_fmadd_ps(va2, vb2, sum2);

            let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
            let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
            sum3 = _mm256_fmadd_ps(va3, vb3, sum3);

            i += 32;
        }

        // Handle remaining groups of 16 items.
        while i + 16 <= len {
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

            i += 16;
        }

        // Handle the last groups of 8 items.
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(va, vb, sum0);
            i += 8;
        }

        // Combine the results of the 4 parallel accumulators into one.
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        let sum = _mm256_add_ps(sum0, sum2);

        // Horizontal Sum: Merges the 8 partial values from the SIMD register into a single final number.
        let mut scalar_sum = crate::math::common::utility::hsum_avx2(sum);
        let mut compensation = 0.0f32;

        // Final Cleanup: Processes the very few leftover items (fewer than 8).
        while i < len {
            let term = a[i] * b[i];
            (scalar_sum, compensation) =
                crate::math::common::kahan_add(scalar_sum, compensation, term);
            i += 1;
        }

        scalar_sum
    }
}

/// Dot product f32 with u16 weights using AVX-512.
/// Basically: (number_a1 * weight_b1) + (number_a2 * weight_b2) + ...
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();
    // Process 16 at a time.
    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        sum_v = _mm512_fmadd_ps(va, vb, sum_v); // Multiply and accumulate.
        i += 16;
    }
    // Sum the results within the register and add the remainder.
    let mut sum = crate::math::common::utility::hsum_avx512(sum_v);
    let mut compensation = 0.0f32;
    while i < len {
        let term = *a.get_unchecked(i) * *b.get_unchecked(i);
        (sum, compensation) = crate::math::common::kahan_add(sum, compensation, term);
        i += 1;
    }
    sum
}

/// Dot product BF16 using AVX-512 BF16.
/// BF16 is a "brain" floating-point format that focuses on what matters for AI.
/// Here the processor handles 32 numbers at once with a single instruction (dpbf16_ps).
#[target_feature(enable = "avx512bf16,avx512vl")]
pub unsafe fn dot_product_bf16_avx512(a: &[u16], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();
    while i + 32 <= len {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
        // This instruction is magic: it handles the computation for 32 BF16 pairs at once.
        sum_v = _mm512_dpbf16_ps(
            sum_v,
            core::mem::transmute::<__m512i, __m512bh>(va),
            core::mem::transmute::<__m512i, __m512bh>(vb),
        );
        i += 32;
    }
    let mut sum = crate::math::common::utility::hsum_avx512(sum_v);
    let mut compensation = 0.0f32;
    // Handle leftovers manually.
    while i < len {
        let fa = f32::from_bits((*a.get_unchecked(i) as u32) << 16);
        let fb = f32::from_bits((*b.get_unchecked(i) as u32) << 16);
        (sum, compensation) = crate::math::common::kahan_add(sum, compensation, fa * fb);
        i += 1;
    }
    sum
}
