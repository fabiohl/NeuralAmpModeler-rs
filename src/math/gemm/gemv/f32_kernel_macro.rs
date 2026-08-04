// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! f32 batched GEMV kernel macros — parameterized SIMD loop skeletons.
//!
//! Provides the 8-accumulator inner loop for batched (multi-frame) f32-weight
//! GEMV operations. Shared by AVX2 (8-wide) and AVX-512 (16-wide).

/// 8-accumulator inner loop (step 8) for batched f32 GEMV with AVX2 — caller owns `$in_c`.
///
/// The input values `vs0..vs7` and the weight-load/FMA chain for all 8 accumulators
/// are inlined into the loop. The caller provides a `$vs` expression that computes
/// the broadcast state value at a given index.
#[macro_export]
macro_rules! gemv_f32_inner_loop_avx2 {
    (
        $in_c:ident, $in_len:expr, $out_len:expr, $out_c:expr,
        $in_frames:expr, $weights:expr,
        $acc0:ident, $acc1:ident, $acc2:ident, $acc3:ident,
        $acc4:ident, $acc5:ident, $acc6:ident, $acc7:ident
    ) => {
        while $in_c + 8 <= $in_len {
            let vs0 = _mm256_set1_ps(*$in_frames.get_unchecked($in_c));
            let vs1 = _mm256_set1_ps(*$in_frames.get_unchecked($in_c + 1));
            let vs2 = _mm256_set1_ps(*$in_frames.get_unchecked($in_c + 2));
            let vs3 = _mm256_set1_ps(*$in_frames.get_unchecked($in_c + 3));
            let vs4 = _mm256_set1_ps(*$in_frames.get_unchecked($in_c + 4));
            let vs5 = _mm256_set1_ps(*$in_frames.get_unchecked($in_c + 5));
            let vs6 = _mm256_set1_ps(*$in_frames.get_unchecked($in_c + 6));
            let vs7 = _mm256_set1_ps(*$in_frames.get_unchecked($in_c + 7));

            let w_ptr = $weights.as_ptr().add($in_c * $out_len + $out_c);
            let w0 = _mm256_loadu_ps(w_ptr);
            $acc0 = _mm256_fmadd_ps(vs0, w0, $acc0);
            let w1 = _mm256_loadu_ps(w_ptr.add($out_len));
            $acc1 = _mm256_fmadd_ps(vs1, w1, $acc1);
            let w2 = _mm256_loadu_ps(w_ptr.add(2 * $out_len));
            $acc2 = _mm256_fmadd_ps(vs2, w2, $acc2);
            let w3 = _mm256_loadu_ps(w_ptr.add(3 * $out_len));
            $acc3 = _mm256_fmadd_ps(vs3, w3, $acc3);
            let w4 = _mm256_loadu_ps(w_ptr.add(4 * $out_len));
            $acc4 = _mm256_fmadd_ps(vs4, w4, $acc4);
            let w5 = _mm256_loadu_ps(w_ptr.add(5 * $out_len));
            $acc5 = _mm256_fmadd_ps(vs5, w5, $acc5);
            let w6 = _mm256_loadu_ps(w_ptr.add(6 * $out_len));
            $acc6 = _mm256_fmadd_ps(vs6, w6, $acc6);
            let w7 = _mm256_loadu_ps(w_ptr.add(7 * $out_len));
            $acc7 = _mm256_fmadd_ps(vs7, w7, $acc7);

            $in_c += 8;
        }
    };
}
