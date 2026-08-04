// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// BF16 version for the 4 LSTM gates.
/// Weights are f32; input is u16 BF16 (converted in-place from state).
#[expect(
    clippy::too_many_arguments,
    reason = "Scalar reference LSTM implementation requiring many matrix dimension parameters for correct reference computation"
)]
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn gemv_4gate_bf16_fallback(
    in_frame: &[u16],
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    gemv_overwrite_bf16_f32w(
        in_frame,
        w0,
        &bias[0..out_len],
        &mut out[0..out_len],
        do_bias,
    );
    gemv_overwrite_bf16_f32w(
        in_frame,
        w1,
        &bias[out_len..2 * out_len],
        &mut out[out_len..2 * out_len],
        do_bias,
    );
    gemv_overwrite_bf16_f32w(
        in_frame,
        w2,
        &bias[2 * out_len..3 * out_len],
        &mut out[2 * out_len..3 * out_len],
        do_bias,
    );
    gemv_overwrite_bf16_f32w(
        in_frame,
        w3,
        &bias[3 * out_len..4 * out_len],
        &mut out[3 * out_len..4 * out_len],
        do_bias,
    );
}

/// BF16 input × f32 weights GEMV — scalar fallback.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller.
#[inline]
unsafe fn gemv_overwrite_bf16_f32w(
    in_frame: &[u16],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        let mut sum = if do_bias { b } else { 0.0 };
        for in_c in 0..in_len {
            let s = f32::from_bits((*in_frame.get_unchecked(in_c) as u32) << 16);
            sum += s * weights.get_unchecked(in_c * out_len + out_c);
        }
        *out_frame.get_unchecked_mut(out_c) = sum;
    }
}
