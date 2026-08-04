// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// GEMV kernel macro — generates platform-specific FMADD accumulate loops.
///
/// Parameterized by SIMD width (4 = AVX2/256-bit, 8 = AVX-512/512-bit)
/// and all relevant SIMD operations as inline closures.
/// Both variants use 8 independent accumulators to maximize FMA throughput.
///
/// # Safety
///
/// Caller must ensure valid pointer arithmetic and slice bounds.
#[macro_export]
macro_rules! gemv_kernel {
    (
        4,
        $is_fused:expr,
        $out_c:expr,
        $out_len:expr,
        $in_frame:expr,
        $weights:expr,
        $bias:expr,
        $out_frame:expr,
        $do_bias:expr,
        $setzero:expr,
        $load_out:expr,
        $load_bias:expr,
        $add_ps:expr,
        $load_weight:expr,
        $fmadd_ps:expr,
        $store_ps:expr
    ) => {
        let mut acc0 = if $is_fused {
            let mut acc = $load_out($out_c);
            if $do_bias {
                acc = $add_ps(acc, $load_bias($out_c));
            }
            acc
        } else {
            if $do_bias {
                $load_bias($out_c)
            } else {
                $setzero()
            }
        };
        let mut acc1 = $setzero();
        let mut acc2 = $setzero();
        let mut acc3 = $setzero();
        let mut acc4 = $setzero();
        let mut acc5 = $setzero();
        let mut acc6 = $setzero();
        let mut acc7 = $setzero();

        let mut in_c = 0;
        let in_len = $in_frame.len();
        while in_c + 8 <= in_len {
            _mm_prefetch::<_MM_HINT_T0>($in_frame.as_ptr().wrapping_add(in_c + 64) as *const i8);

            let vs0 = _mm256_set1_ps(*$in_frame.get_unchecked(in_c));
            let vs1 = _mm256_set1_ps(*$in_frame.get_unchecked(in_c + 1));
            let vs2 = _mm256_set1_ps(*$in_frame.get_unchecked(in_c + 2));
            let vs3 = _mm256_set1_ps(*$in_frame.get_unchecked(in_c + 3));
            let vs4 = _mm256_set1_ps(*$in_frame.get_unchecked(in_c + 4));
            let vs5 = _mm256_set1_ps(*$in_frame.get_unchecked(in_c + 5));
            let vs6 = _mm256_set1_ps(*$in_frame.get_unchecked(in_c + 6));
            let vs7 = _mm256_set1_ps(*$in_frame.get_unchecked(in_c + 7));

            let w_ptr = $weights.as_ptr().add(in_c * $out_len + $out_c);
            let w0 = $load_weight(w_ptr);
            acc0 = $fmadd_ps(vs0, w0, acc0);

            let w1 = $load_weight(w_ptr.add($out_len));
            acc1 = $fmadd_ps(vs1, w1, acc1);

            let w2 = $load_weight(w_ptr.add(2 * $out_len));
            acc2 = $fmadd_ps(vs2, w2, acc2);

            let w3 = $load_weight(w_ptr.add(3 * $out_len));
            acc3 = $fmadd_ps(vs3, w3, acc3);

            let w4 = $load_weight(w_ptr.add(4 * $out_len));
            acc4 = $fmadd_ps(vs4, w4, acc4);

            let w5 = $load_weight(w_ptr.add(5 * $out_len));
            acc5 = $fmadd_ps(vs5, w5, acc5);

            let w6 = $load_weight(w_ptr.add(6 * $out_len));
            acc6 = $fmadd_ps(vs6, w6, acc6);

            let w7 = $load_weight(w_ptr.add(7 * $out_len));
            acc7 = $fmadd_ps(vs7, w7, acc7);

            in_c += 8;
        }

        acc0 = $add_ps(acc0, acc1);
        acc2 = $add_ps(acc2, acc3);
        acc4 = $add_ps(acc4, acc5);
        acc6 = $add_ps(acc6, acc7);
        acc0 = $add_ps(acc0, acc2);
        acc4 = $add_ps(acc4, acc6);
        acc0 = $add_ps(acc0, acc4);

        while in_c < in_len {
            let vs = _mm256_set1_ps(*$in_frame.get_unchecked(in_c));
            let weight_ptr = $weights.as_ptr().add(in_c * $out_len + $out_c);
            let vw = $load_weight(weight_ptr);
            acc0 = $fmadd_ps(vs, vw, acc0);
            in_c += 1;
        }

        $store_ps($out_c, acc0);
    };

    (
        8,
        $is_fused:expr,
        $out_c:expr,
        $out_len:expr,
        $in_frame:expr,
        $weights:expr,
        $bias:expr,
        $out_frame:expr,
        $do_bias:expr,
        $setzero:expr,
        $load_out:expr,
        $load_bias:expr,
        $add_ps:expr,
        $load_weight:expr,
        $fmadd_ps:expr,
        $store_ps:expr
    ) => {
        let mut acc0 = if $is_fused {
            let mut acc = $load_out($out_c);
            if $do_bias {
                acc = $add_ps(acc, $load_bias($out_c));
            }
            acc
        } else {
            if $do_bias {
                $load_bias($out_c)
            } else {
                $setzero()
            }
        };
        let mut acc1 = $setzero();
        let mut acc2 = $setzero();
        let mut acc3 = $setzero();
        let mut acc4 = $setzero();
        let mut acc5 = $setzero();
        let mut acc6 = $setzero();
        let mut acc7 = $setzero();

        let mut in_c = 0;
        let in_len = $in_frame.len();
        while in_c + 8 <= in_len {
            _mm_prefetch::<_MM_HINT_T0>($in_frame.as_ptr().wrapping_add(in_c + 64) as *const i8);

            let vs0 = _mm512_set1_ps(*$in_frame.get_unchecked(in_c));
            let vs1 = _mm512_set1_ps(*$in_frame.get_unchecked(in_c + 1));
            let vs2 = _mm512_set1_ps(*$in_frame.get_unchecked(in_c + 2));
            let vs3 = _mm512_set1_ps(*$in_frame.get_unchecked(in_c + 3));
            let vs4 = _mm512_set1_ps(*$in_frame.get_unchecked(in_c + 4));
            let vs5 = _mm512_set1_ps(*$in_frame.get_unchecked(in_c + 5));
            let vs6 = _mm512_set1_ps(*$in_frame.get_unchecked(in_c + 6));
            let vs7 = _mm512_set1_ps(*$in_frame.get_unchecked(in_c + 7));

            let w_ptr = $weights.as_ptr().add(in_c * $out_len + $out_c);
            let w0 = $load_weight(w_ptr);
            acc0 = $fmadd_ps(vs0, w0, acc0);

            let w1 = $load_weight(w_ptr.add($out_len));
            acc1 = $fmadd_ps(vs1, w1, acc1);

            let w2 = $load_weight(w_ptr.add(2 * $out_len));
            acc2 = $fmadd_ps(vs2, w2, acc2);

            let w3 = $load_weight(w_ptr.add(3 * $out_len));
            acc3 = $fmadd_ps(vs3, w3, acc3);

            let w4 = $load_weight(w_ptr.add(4 * $out_len));
            acc4 = $fmadd_ps(vs4, w4, acc4);

            let w5 = $load_weight(w_ptr.add(5 * $out_len));
            acc5 = $fmadd_ps(vs5, w5, acc5);

            let w6 = $load_weight(w_ptr.add(6 * $out_len));
            acc6 = $fmadd_ps(vs6, w6, acc6);

            let w7 = $load_weight(w_ptr.add(7 * $out_len));
            acc7 = $fmadd_ps(vs7, w7, acc7);

            in_c += 8;
        }

        acc0 = $add_ps(acc0, acc1);
        acc2 = $add_ps(acc2, acc3);
        acc4 = $add_ps(acc4, acc5);
        acc6 = $add_ps(acc6, acc7);
        acc0 = $add_ps(acc0, acc2);
        acc4 = $add_ps(acc4, acc6);
        acc0 = $add_ps(acc0, acc4);

        while in_c < in_len {
            let vs = _mm512_set1_ps(*$in_frame.get_unchecked(in_c));
            let weight_ptr = $weights.as_ptr().add(in_c * $out_len + $out_c);
            let vw = $load_weight(weight_ptr);
            acc0 = $fmadd_ps(vs, vw, acc0);
            in_c += 1;
        }

        $store_ps($out_c, acc0);
    };
}
