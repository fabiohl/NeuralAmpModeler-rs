// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Fused kernels for LSTM gates (AVX2 and AVX-512).
//!
//! Extracted from `activations/fused.rs` and `simd/avx2.rs`/`simd/avx512.rs`.

use crate::math::activations::ActivationPrecision;
use crate::math::activations::activation_precision;
use crate::math::activations::sigmoid::high_fidelity::simd_sigmoid_poly_avx2;
#[cfg(feature = "avx512")]
use crate::math::activations::sigmoid::high_fidelity::simd_sigmoid_poly_avx512;
#[cfg(feature = "avx512")]
use crate::math::activations::sigmoid::simd_sigmoid_avx512;
use crate::math::activations::sigmoid::{simd_sigmoid_avx2, simd_sigmoid_dual_avx2};
use crate::math::activations::tanh::high_fidelity::simd_tanh_poly_avx2;
#[cfg(feature = "avx512")]
use crate::math::activations::tanh::high_fidelity::simd_tanh_poly_avx512;
use crate::math::activations::tanh::simd_tanh_avx2;
#[cfg(feature = "avx512")]
use crate::math::activations::tanh::simd_tanh_avx512;
use core::arch::x86_64::*;

/// Fused kernel for LSTM gates (AVX2) — Standard (exact-grade) accuracy path.
/// Uses polynomial exp-based tanh/sigmoid approximations with Kahan
/// compensated summation for the cell state.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_lstm_gates_avx2_hf(
    gf: __m256,
    gi: __m256,
    gg: __m256,
    go: __m256,
    cs: __m256,
    cs_err: __m256,
) -> (__m256, __m256, __m256) {
    let sig_f = unsafe { simd_sigmoid_poly_avx2(gf) };
    let sig_i = unsafe { simd_sigmoid_poly_avx2(gi) };
    let sig_o = unsafe { simd_sigmoid_poly_avx2(go) };
    let tanh_g = unsafe { simd_tanh_poly_avx2(gg) };

    let f_cs = _mm256_mul_ps(sig_f, cs);
    let f_err = _mm256_mul_ps(sig_f, cs_err);
    let i_g = _mm256_mul_ps(sig_i, tanh_g);
    let y = _mm256_sub_ps(i_g, f_err);
    let new_cs = _mm256_add_ps(f_cs, y);
    let new_cs_err = _mm256_sub_ps(_mm256_sub_ps(new_cs, f_cs), y);

    let hidden = _mm256_mul_ps(sig_o, unsafe { simd_tanh_poly_avx2(new_cs) });

    (new_cs, new_cs_err, hidden)
}

/// Fused kernel for LSTM gates (AVX2) — Fast (production) accuracy path.
/// Uses Padé tanh + minimax sigmoid with interleaved dual-sigmoid and
/// Kahan compensated summation for the cell state.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_lstm_gates_avx2_std(
    gf: __m256,
    gi: __m256,
    gg: __m256,
    go: __m256,
    cs: __m256,
    cs_err: __m256,
) -> (__m256, __m256, __m256) {
    let (sig_f, sig_i) = unsafe { simd_sigmoid_dual_avx2(gf, gi) };
    let sig_o = unsafe { simd_sigmoid_avx2(go) };
    let tanh_g = unsafe { simd_tanh_avx2(gg) };

    let f_cs = _mm256_mul_ps(sig_f, cs);
    let f_err = _mm256_mul_ps(sig_f, cs_err);
    let i_g = _mm256_mul_ps(sig_i, tanh_g);
    let y = _mm256_sub_ps(i_g, f_err);
    let new_cs = _mm256_add_ps(f_cs, y);
    let new_cs_err = _mm256_sub_ps(_mm256_sub_ps(new_cs, f_cs), y);

    let hidden = _mm256_mul_ps(sig_o, unsafe { simd_tanh_avx2(new_cs) });

    (new_cs, new_cs_err, hidden)
}

/// Fused kernel for LSTM gates (AVX2).
/// Thin dispatch wrapper — prefer `_hf` / `_std` variants in hot paths.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_lstm_gates_avx2(
    gf: __m256,
    gi: __m256,
    gg: __m256,
    go: __m256,
    cs: __m256,
    cs_err: __m256,
) -> (__m256, __m256, __m256) {
    if activation_precision() == ActivationPrecision::Standard {
        unsafe { fused_lstm_gates_avx2_hf(gf, gi, gg, go, cs, cs_err) }
    } else {
        unsafe { fused_lstm_gates_avx2_std(gf, gi, gg, go, cs, cs_err) }
    }
}

/// Fused kernel for LSTM gates (AVX-512) — Standard (exact-grade) accuracy path.
/// Uses polynomial exp-based tanh/sigmoid approximations with Kahan
/// compensated summation for the cell state.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_lstm_gates_avx512_hf(
    gf: __m512,
    gi: __m512,
    gg: __m512,
    go: __m512,
    cs: __m512,
    cs_err: __m512,
) -> (__m512, __m512, __m512) {
    let sig_f = unsafe { simd_sigmoid_poly_avx512(gf) };
    let sig_i = unsafe { simd_sigmoid_poly_avx512(gi) };
    let sig_o = unsafe { simd_sigmoid_poly_avx512(go) };
    let tanh_g = unsafe { simd_tanh_poly_avx512(gg) };

    let f_cs = _mm512_mul_ps(sig_f, cs);
    let f_err = _mm512_mul_ps(sig_f, cs_err);
    let i_g = _mm512_mul_ps(sig_i, tanh_g);
    let y = _mm512_sub_ps(i_g, f_err);
    let new_cs = _mm512_add_ps(f_cs, y);
    let new_cs_err = _mm512_sub_ps(_mm512_sub_ps(new_cs, f_cs), y);

    let hidden = _mm512_mul_ps(sig_o, unsafe { simd_tanh_poly_avx512(new_cs) });

    (new_cs, new_cs_err, hidden)
}

/// Fused kernel for LSTM gates (AVX-512, research opt-in) — Fast accuracy path.
/// Uses Padé tanh + minimax sigmoid with Kahan compensated summation
/// for the cell state.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_lstm_gates_avx512_std(
    gf: __m512,
    gi: __m512,
    gg: __m512,
    go: __m512,
    cs: __m512,
    cs_err: __m512,
) -> (__m512, __m512, __m512) {
    let sig_f = unsafe { simd_sigmoid_avx512(gf) };
    let sig_i = unsafe { simd_sigmoid_avx512(gi) };
    let sig_o = unsafe { simd_sigmoid_avx512(go) };
    let tanh_g = unsafe { simd_tanh_avx512(gg) };

    let f_cs = _mm512_mul_ps(sig_f, cs);
    let f_err = _mm512_mul_ps(sig_f, cs_err);
    let i_g = _mm512_mul_ps(sig_i, tanh_g);
    let y = _mm512_sub_ps(i_g, f_err);
    let new_cs = _mm512_add_ps(f_cs, y);
    let new_cs_err = _mm512_sub_ps(_mm512_sub_ps(new_cs, f_cs), y);

    let hidden = _mm512_mul_ps(sig_o, unsafe { simd_tanh_avx512(new_cs) });

    (new_cs, new_cs_err, hidden)
}

/// Fused kernel for LSTM gates (AVX-512).
/// Thin dispatch wrapper — prefer `_hf` / `_std` variants in hot paths.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_lstm_gates_avx512(
    gf: __m512,
    gi: __m512,
    gg: __m512,
    go: __m512,
    cs: __m512,
    cs_err: __m512,
) -> (__m512, __m512, __m512) {
    if activation_precision() == ActivationPrecision::Standard {
        unsafe { fused_lstm_gates_avx512_hf(gf, gi, gg, go, cs, cs_err) }
    } else {
        unsafe { fused_lstm_gates_avx512_std(gf, gi, gg, go, cs, cs_err) }
    }
}

/// Fused kernel for dynamic LSTM gate processing via AVX2.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_lstm_gates_dyn_avx2(
    gates: &mut [f32],
    cell_state: &mut [f32],
    cell_error: &mut [f32],
    hidden_state: &mut [f32],
    hidden_size: usize,
) {
    let is_hf = activation_precision() == ActivationPrecision::Standard;
    let mut j = 0;
    while j + 8 <= hidden_size {
        let gi = _mm256_loadu_ps(gates.as_ptr().add(j));
        let gf = _mm256_loadu_ps(gates.as_ptr().add(j + hidden_size));
        let gg = _mm256_loadu_ps(gates.as_ptr().add(j + 2 * hidden_size));
        let go = _mm256_loadu_ps(gates.as_ptr().add(j + 3 * hidden_size));
        let cs = _mm256_loadu_ps(cell_state.as_ptr().add(j));
        let cs_err = _mm256_loadu_ps(cell_error.as_ptr().add(j));

        let (new_cs, new_cs_err, hidden) = if is_hf {
            fused_lstm_gates_avx2_hf(gf, gi, gg, go, cs, cs_err)
        } else {
            fused_lstm_gates_avx2_std(gf, gi, gg, go, cs, cs_err)
        };

        _mm256_storeu_ps(cell_state.as_mut_ptr().add(j), new_cs);
        _mm256_storeu_ps(cell_error.as_mut_ptr().add(j), new_cs_err);
        _mm256_storeu_ps(hidden_state.as_mut_ptr().add(j), hidden);

        j += 8;
    }
    if j < hidden_size {
        // SAFETY: caller guarantees AVX2+FMA; tail shares the same feature set.
        unsafe {
            fused_lstm_gates_dyn_tail(
                gates,
                cell_state,
                cell_error,
                hidden_state,
                hidden_size,
                is_hf,
                j,
            );
        }
    }
}

/// AVX2 tail for dynamic LSTM gates — processes remainders (<8 elements) using
/// a stack-buffer pattern that loads into `__m256` with neutral padding lanes,
/// matching the approach in `src/models/lstm/layer_kernels.rs:89-130`.
///
/// Also re-used by the AVX-512 path for remainders <16 (processes up to one
/// 8-wide chunk via AVX2 before falling to the stack-buffer tail).
///
/// Replacement for the previous scalar tail — the `Standard` and `Fast` paths
/// share the same buffer-and-dispatch structure.
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_lstm_gates_dyn_tail(
    gates: &mut [f32],
    cell_state: &mut [f32],
    cell_error: &mut [f32],
    hidden_state: &mut [f32],
    hidden_size: usize,
    is_hf: bool,
    mut j: usize,
) {
    let gp = gates.as_mut_ptr();
    let cp = cell_state.as_ptr();
    let ep = cell_error.as_ptr();

    while j + 8 <= hidden_size {
        let gi = _mm256_loadu_ps(gp.add(j));
        let gf = _mm256_loadu_ps(gp.add(j + hidden_size));
        let gg = _mm256_loadu_ps(gp.add(j + 2 * hidden_size));
        let go = _mm256_loadu_ps(gp.add(j + 3 * hidden_size));
        let cs = _mm256_loadu_ps(cp.add(j));
        let cs_err = _mm256_loadu_ps(ep.add(j));

        let (new_cs, new_cs_err, hidden) = if is_hf {
            fused_lstm_gates_avx2_hf(gf, gi, gg, go, cs, cs_err)
        } else {
            fused_lstm_gates_avx2_std(gf, gi, gg, go, cs, cs_err)
        };

        _mm256_storeu_ps(cell_state.as_mut_ptr().add(j), new_cs);
        _mm256_storeu_ps(cell_error.as_mut_ptr().add(j), new_cs_err);
        _mm256_storeu_ps(hidden_state.as_mut_ptr().add(j), hidden);

        j += 8;
    }

    if j < hidden_size {
        let tail_len = hidden_size - j;
        let mut temp_gf = [0.0f32; 8];
        let mut temp_gi = [0.0f32; 8];
        let mut temp_gg = [0.0f32; 8];
        let mut temp_go = [0.0f32; 8];
        let mut temp_cs = [0.0f32; 8];
        let mut temp_ce = [0.0f32; 8];

        for k in 0..tail_len {
            temp_gf[k] = *gp.add(j + k + hidden_size);
            temp_gi[k] = *gp.add(j + k);
            temp_gg[k] = *gp.add(j + k + 2 * hidden_size);
            temp_go[k] = *gp.add(j + k + 3 * hidden_size);
            temp_cs[k] = *cp.add(j + k);
            temp_ce[k] = *ep.add(j + k);
        }

        let gf_v = _mm256_loadu_ps(temp_gf.as_ptr());
        let gi_v = _mm256_loadu_ps(temp_gi.as_ptr());
        let gg_v = _mm256_loadu_ps(temp_gg.as_ptr());
        let go_v = _mm256_loadu_ps(temp_go.as_ptr());
        let cs_v = _mm256_loadu_ps(temp_cs.as_ptr());
        let ce_v = _mm256_loadu_ps(temp_ce.as_ptr());

        let (new_cs_v, new_ce_v, hidden_v) = if is_hf {
            fused_lstm_gates_avx2_hf(gf_v, gi_v, gg_v, go_v, cs_v, ce_v)
        } else {
            fused_lstm_gates_avx2_std(gf_v, gi_v, gg_v, go_v, cs_v, ce_v)
        };

        let mut out_cs = [0.0f32; 8];
        let mut out_ce = [0.0f32; 8];
        let mut out_hs = [0.0f32; 8];
        _mm256_storeu_ps(out_cs.as_mut_ptr(), new_cs_v);
        _mm256_storeu_ps(out_ce.as_mut_ptr(), new_ce_v);
        _mm256_storeu_ps(out_hs.as_mut_ptr(), hidden_v);

        let cp_mut = cell_state.as_mut_ptr();
        let ep_mut = cell_error.as_mut_ptr();
        let hp_mut = hidden_state.as_mut_ptr();
        for k in 0..tail_len {
            *cp_mut.add(j + k) = out_cs[k];
            *ep_mut.add(j + k) = out_ce[k];
            *hp_mut.add(j + k) = out_hs[k];
        }
    }
}

/// Fused kernel to update the memory (state) of an LSTM network.
/// This function decides what the network should "forget" from the past and what to "learn" from the present,
/// updating the values all at once for 16 memory cells.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_lstm_gates_dyn_avx512(
    gates: &mut [f32],
    cell_state: &mut [f32],
    cell_error: &mut [f32],
    hidden_state: &mut [f32],
    hidden_size: usize,
) {
    let is_hf = activation_precision() == ActivationPrecision::Standard;
    let mut j = 0;
    while j + 16 <= hidden_size {
        // Load the 4 decisions (forget, learn, etc.) for 16 cells.
        let gi = _mm512_loadu_ps(gates.as_ptr().add(j));
        let gf = _mm512_loadu_ps(gates.as_ptr().add(j + hidden_size));
        let gg = _mm512_loadu_ps(gates.as_ptr().add(j + 2 * hidden_size));
        let go = _mm512_loadu_ps(gates.as_ptr().add(j + 3 * hidden_size));
        let cs = _mm512_loadu_ps(cell_state.as_ptr().add(j));
        let cs_err = _mm512_loadu_ps(cell_error.as_ptr().add(j));

        // Perform the memory computation in a fused manner.
        let (new_cs, new_cs_err, hidden) = if is_hf {
            fused_lstm_gates_avx512_hf(gf, gi, gg, go, cs, cs_err)
        } else {
            fused_lstm_gates_avx512_std(gf, gi, gg, go, cs, cs_err)
        };

        _mm512_storeu_ps(cell_state.as_mut_ptr().add(j), new_cs);
        _mm512_storeu_ps(cell_error.as_mut_ptr().add(j), new_cs_err);
        _mm512_storeu_ps(hidden_state.as_mut_ptr().add(j), hidden);

        j += 16;
    }
    if j < hidden_size {
        // SAFETY: AVX-512 implies AVX2+FMA; tail requires only AVX2+FMA.
        unsafe {
            fused_lstm_gates_dyn_tail(
                gates,
                cell_state,
                cell_error,
                hidden_state,
                hidden_size,
                is_hf,
                j,
            );
        }
    }
}
