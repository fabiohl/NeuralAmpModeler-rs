// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::A2Conv1dCh8;
use super::MAX_KERNEL_FRAMES;
use crate::math::common::SimdMath;
use crate::models::a2::film::FilmBlock;
use crate::models::a2::params::A2_LEAKY_SLOPE;
use core::arch::x86_64::*;

/// T=8 frame-tiled tap-major dilated conv for CH=8, AVX2+FMA.
///
/// Processes frames in groups of 8, accumulating all K taps into `T*C`
/// register-allocated accumulators. For each (tap, input_channel) pair,
/// loads the 8 output-channel weights once and broadcasts the history
/// value for each of the 8 frames via `_mm256_set1_ps`.
///
/// Maintaining 8 independent accumulator chains (a0..a7) saturates the
/// 2 FMA ports (~4-5 cycle latency → 8 in-flight chains needed for full
/// throughput). Register pressure: 8 accumulators + wcol + temp ≈ 10 YMM
/// (fits within 16 YMM of x86-64-v3).
///
/// # Safety
/// - `weights` must have at least `kernel * 64` valid f32 elements.
/// - `layer_buffer` must be large enough for all frame lookbacks.
/// - `z_out` must have at least `num_frames * 8` elements.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn conv1d_ch8_t8_avx2(
    weights: &[f32],
    bias: &[f32],
    dilation: usize,
    kernel: usize,
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    z_out: &mut [f32],
) {
    debug_assert!(z_out.len() >= num_frames * 8);
    debug_assert!(weights.len() >= kernel * 64);
    debug_assert!(bias.len() >= 8);

    let ch: usize = 8;
    let d = dilation as isize;
    let k_i = kernel as isize;
    let buf = layer_buffer.as_ptr();
    let w_ptr = weights.as_ptr();

    let bias_v = _mm256_loadu_ps(bias.as_ptr());

    const T: usize = 8;
    let n_tiled = (num_frames / T) * T;

    for f in (0..n_tiled).step_by(T) {
        let mut a0 = bias_v;
        let mut a1 = bias_v;
        let mut a2 = bias_v;
        let mut a3 = bias_v;
        let mut a4 = bias_v;
        let mut a5 = bias_v;
        let mut a6 = bias_v;
        let mut a7 = bias_v;

        let frame0 = (frame_start + f) as isize;

        for k in 0..kernel {
            let wk_base = (k * 64) as isize;
            let taps_back = k_i - 1 - k as isize;
            let tap0 = frame0 - d * taps_back;
            let hb = buf.offset(tap0 * ch as isize);
            for cp in 0..ch {
                let wcol = _mm256_loadu_ps(w_ptr.offset(wk_base + (cp * 8) as isize));
                let h0 = *hb.add(cp);
                let h1 = *hb.add(ch + cp);
                let h2 = *hb.add(2 * ch + cp);
                let h3 = *hb.add(3 * ch + cp);
                let h4 = *hb.add(4 * ch + cp);
                let h5 = *hb.add(5 * ch + cp);
                let h6 = *hb.add(6 * ch + cp);
                let h7 = *hb.add(7 * ch + cp);
                a0 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h0), a0);
                a1 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h1), a1);
                a2 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h2), a2);
                a3 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h3), a3);
                a4 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h4), a4);
                a5 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h5), a5);
                a6 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h6), a6);
                a7 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h7), a7);
            }
        }

        _mm256_storeu_ps(z_out.as_mut_ptr().add(f * ch), a0);
        _mm256_storeu_ps(z_out.as_mut_ptr().add((f + 1) * ch), a1);
        _mm256_storeu_ps(z_out.as_mut_ptr().add((f + 2) * ch), a2);
        _mm256_storeu_ps(z_out.as_mut_ptr().add((f + 3) * ch), a3);
        _mm256_storeu_ps(z_out.as_mut_ptr().add((f + 4) * ch), a4);
        _mm256_storeu_ps(z_out.as_mut_ptr().add((f + 5) * ch), a5);
        _mm256_storeu_ps(z_out.as_mut_ptr().add((f + 6) * ch), a6);
        _mm256_storeu_ps(z_out.as_mut_ptr().add((f + 7) * ch), a7);
    }

    for f in n_tiled..num_frames {
        let frame_idx = (frame_start + f) as isize;
        let mut acc = bias_v;
        for k in 0..kernel {
            let wk_base = (k * 64) as isize;
            let taps_back = k_i - 1 - k as isize;
            let tap_base = frame_idx - d * taps_back;
            let hb = buf.offset(tap_base * ch as isize);
            for cp in 0..ch {
                let wcol = _mm256_loadu_ps(w_ptr.offset(wk_base + (cp * 8) as isize));
                let hv = *hb.add(cp);
                acc = _mm256_fmadd_ps(wcol, _mm256_set1_ps(hv), acc);
            }
        }
        _mm256_storeu_ps(z_out.as_mut_ptr().add(f * ch), acc);
    }
}

/// Full layer forward pass for CH=8 using T=8 tiled tap-major conv.
///
/// Processes `num_frames` through: dilated conv → [FiLM post-conv] → bias → mixin →
/// [FiLM post-mixin] → LeakyReLU → [FiLM post-activation] → head accumulate →
/// l1x1 residual → [FiLM post-l1x1]. All operations use SIMD block processing
/// on `__m256` vectors.
///
/// FiLM insertion points are conditionally executed when `film.*_film.is_some()`.
///
/// # Safety
/// Buffers must be sized appropriately. Caller ensures linear ring history
/// includes lookback + block frames.
#[expect(
    clippy::too_many_arguments,
    reason = "A2 CH=8 SIMD convolution kernel requiring many shape/stride parameters for optimized audio processing"
)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn layer_forward_ch8_block(
    conv: &A2Conv1dCh8,
    mixin_w: &[f32],
    l1x1_w: &[f32],
    l1x1_b: &[f32],
    film: &mut FilmBlock<'_>,
    use_blending: bool,
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    input_cond: &[f32],
    head_accum: &mut [f32],
    head_col: usize,
    layer_in: &mut [f32],
    is_first: bool,
    is_last: bool,
) {
    let ch: usize = 8;
    debug_assert!(mixin_w.len() >= ch);
    debug_assert!(l1x1_w.len() >= ch * ch);
    debug_assert!(l1x1_b.len() >= ch);
    debug_assert!(layer_in.len() >= num_frames * ch);
    debug_assert!(input_cond.len() >= num_frames);

    // `process()` guarantees ≤ MAX_KERNEL_FRAMES via internal chunking.
    debug_assert!(num_frames <= MAX_KERNEL_FRAMES);
    let mut z_buf = [0.0f32; MAX_KERNEL_FRAMES * 8];

    conv1d_ch8_t8_avx2(
        &conv.weights,
        &conv.bias,
        conv.dilation,
        conv.kernel,
        layer_buffer,
        frame_start,
        num_frames,
        &mut z_buf[..num_frames * ch],
    );

    // 1b. FiLM: conv_post_film (post-conv, pre-mixin).
    for f in 0..num_frames {
        let cond = &input_cond[f..f + 1];
        let z_slice = &mut z_buf[f * ch..(f + 1) * ch];
        if let Some(ref mut film) = film.conv_post_film {
            film.process(z_slice, cond);
        }
    }

    // 2. Post-conv: mixin (isolated scratch buffer).
    {
        let z = z_buf.as_mut_ptr();
        let mixin_v = _mm256_loadu_ps(mixin_w.as_ptr());
        for (f, cond_val) in input_cond.iter().take(num_frames).enumerate() {
            let off = f * ch;
            // 2a. Apply input_mixin_pre_film to condition (self-modulation,
            // C++ model.cpp:188-197). For cond_size == 1: cond = scale * cond + shift.
            let mut cond_mod = *cond_val;
            if let Some(ref mut film) = film.input_mixin_pre_film {
                let orig = cond_mod;
                unsafe {
                    film.process(
                        core::slice::from_mut(&mut cond_mod),
                        core::slice::from_ref(&orig),
                    );
                }
            }
            let cond_v = _mm256_set1_ps(cond_mod);
            let mix_v = _mm256_mul_ps(mixin_v, cond_v);

            let mut mixin_scratch = [0.0f32; 8];
            _mm256_storeu_ps(mixin_scratch.as_mut_ptr(), mix_v);

            let cond = &input_cond[f..f + 1];
            if let Some(ref mut film) = film.input_mixin_post_film {
                film.process(&mut mixin_scratch, cond);
            }

            let mix_v_modulated = _mm256_loadu_ps(mixin_scratch.as_ptr());
            let mut zv = _mm256_loadu_ps(z.add(off));
            zv = _mm256_add_ps(zv, mix_v_modulated);
            _mm256_storeu_ps(z.add(off), zv);

            let z_slice = &mut z_buf[off..off + ch];
            if let Some(ref mut film) = film.activation_pre_film {
                film.process(z_slice, cond);
            }
        }
    }

    // 3. LeakyReLU (in-place on z_buf).
    {
        let z = z_buf.as_mut_ptr();
        let slope_v = _mm256_set1_ps(A2_LEAKY_SLOPE);
        let zero_v = _mm256_setzero_ps();
        for f in 0..num_frames {
            let off = f * ch;
            let zv = _mm256_loadu_ps(z.add(off));
            let mask = _mm256_cmp_ps(zv, zero_v, _CMP_LT_OS);
            let zv_leaky = _mm256_mul_ps(zv, slope_v);
            _mm256_storeu_ps(z.add(off), _mm256_blendv_ps(zv, zv_leaky, mask));
        }
    }

    // 3b. FiLM: activation_post_film (post-activation).
    for f in 0..num_frames {
        let cond = &input_cond[f..f + 1];
        let z_slice = &mut z_buf[f * ch..(f + 1) * ch];
        if let Some(ref mut film) = film.activation_post_film {
            film.process(z_slice, cond);
        }
    }

    // 4. Head accumulate.
    {
        let head = head_accum.as_mut_ptr();
        for f in 0..num_frames {
            let head_off = (head_col + f) * ch;
            let zv = _mm256_loadu_ps(z_buf.as_ptr().add(f * ch));
            if is_first {
                _mm256_storeu_ps(head.add(head_off), zv);
            } else {
                let hv = _mm256_loadu_ps(head.add(head_off));
                _mm256_storeu_ps(head.add(head_off), _mm256_add_ps(hv, zv));
            }
        }
    }

    // 5. Layer1x1 residual (skipped on last layer) — isolated scratch buffer.
    if !is_last {
        let lin = layer_in.as_mut_ptr();
        let l1x1_b_v = _mm256_loadu_ps(l1x1_b.as_ptr());
        let l1x1_w_ptr = l1x1_w.as_ptr();
        for f in 0..num_frames {
            let off = f * ch;
            let mut acc = l1x1_b_v;
            for u in 0..ch {
                let zu = *z_buf.get_unchecked(off + u);
                let zu_v = _mm256_set1_ps(zu);
                let w_col = _mm256_loadu_ps(l1x1_w_ptr.add(u * ch));
                acc = _mm256_fmadd_ps(zu_v, w_col, acc);
            }

            let mut l1x1_scratch = [0.0f32; 8];
            _mm256_storeu_ps(l1x1_scratch.as_mut_ptr(), acc);

            let cond = &input_cond[f..f + 1];
            if let Some(film) = film
                .layer1x1_post_film
                .as_deref_mut()
                .filter(|_| use_blending)
            {
                film.process(&mut l1x1_scratch, cond);
            }

            let l1x1_v_modulated = _mm256_loadu_ps(l1x1_scratch.as_ptr());
            let lv = _mm256_loadu_ps(lin.add(off));
            _mm256_storeu_ps(lin.add(off), _mm256_add_ps(lv, l1x1_v_modulated));
        }
    }
}

/// SimdMath-dispatched full layer forward pass for CH=8.
///
/// Same semantics as `layer_forward_ch8_block` but uses `M::dot_product_8x_f32`
/// for the convolution step via monomorphized SimdMath dispatch, enabling ISA-optimal
/// kernel selection at compile time. Post-conv operations (mixin, LeakyReLU, head,
/// l1x1) remain on raw `__m256` intrinsics.
///
/// # Safety
/// Buffers must be sized appropriately. Caller ensures linear ring history
/// includes lookback + block frames.
#[expect(
    clippy::too_many_arguments,
    reason = "A2 CH=8 SIMD convolution kernel requiring many shape/stride parameters for optimized audio processing"
)]
#[inline(always)]
pub unsafe fn layer_forward_ch8_block_simdmath<M: SimdMath>(
    conv: &A2Conv1dCh8,
    mixin_w: &[f32],
    l1x1_w: &[f32],
    l1x1_b: &[f32],
    film: &mut FilmBlock<'_>,
    use_blending: bool,
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    input_cond: &[f32],
    head_accum: &mut [f32],
    head_col: usize,
    layer_in: &mut [f32],
    is_first: bool,
    is_last: bool,
) {
    let ch: usize = 8;
    debug_assert!(mixin_w.len() >= ch);
    debug_assert!(l1x1_w.len() >= ch * ch);
    debug_assert!(l1x1_b.len() >= ch);
    debug_assert!(layer_in.len() >= num_frames * ch);
    debug_assert!(input_cond.len() >= num_frames);
    debug_assert!(num_frames <= MAX_KERNEL_FRAMES);

    let mut z_buf = [0.0f32; MAX_KERNEL_FRAMES * 8];
    let ch_pad = 8usize;
    let stride = ch_pad * ch_pad;
    let d = conv.dilation as isize;
    let k_i = conv.kernel as isize;
    let buf = layer_buffer.as_ptr();
    let w_ptr = conv.weights.as_ptr();

    let bias = &conv.bias;

    for f in 0..num_frames {
        let frame_idx = (frame_start + f) as isize;
        let mut acc = [
            bias[0], bias[1], bias[2], bias[3], bias[4], bias[5], bias[6], bias[7],
        ];

        for k in 0..conv.kernel {
            let taps_back = k_i - 1 - k as isize;
            let tap_base = frame_idx - d * taps_back;
            let hb = buf.offset(tap_base * ch as isize);
            let in_slice = core::slice::from_raw_parts(hb, ch);

            let w_slice: &[[f32; 8]] = {
                let ptr = w_ptr.add(k * stride) as *const [f32; 8];
                core::slice::from_raw_parts(ptr, ch)
            };

            let t = M::dot_product_8x_f32(w_slice, in_slice);
            for c in 0..8 {
                acc[c] += t[c];
            }
        }

        let off = f * ch;
        z_buf[off..off + ch].copy_from_slice(&acc);
    }

    // 1b. FiLM: conv_post_film (post-conv, pre-mixin).
    for f in 0..num_frames {
        let cond = &input_cond[f..f + 1];
        let z_slice = &mut z_buf[f * ch..(f + 1) * ch];
        if let Some(ref mut film) = film.conv_post_film {
            film.process(z_slice, cond);
        }
    }

    // 2. Post-conv: mixin (isolated scratch buffer).
    {
        let z = z_buf.as_mut_ptr();
        let mixin_v = _mm256_loadu_ps(mixin_w.as_ptr());
        for (f, cond_val) in input_cond.iter().take(num_frames).enumerate() {
            let off = f * ch;
            // 2a. Apply input_mixin_pre_film to condition (self-modulation,
            // C++ model.cpp:188-197). For cond_size == 1: cond = scale * cond + shift.
            let mut cond_mod = *cond_val;
            if let Some(ref mut film) = film.input_mixin_pre_film {
                let orig = cond_mod;
                unsafe {
                    film.process(
                        core::slice::from_mut(&mut cond_mod),
                        core::slice::from_ref(&orig),
                    );
                }
            }
            let cond_v = _mm256_set1_ps(cond_mod);
            let mix_v = _mm256_mul_ps(mixin_v, cond_v);

            let mut mixin_scratch = [0.0f32; 8];
            _mm256_storeu_ps(mixin_scratch.as_mut_ptr(), mix_v);

            let cond = &input_cond[f..f + 1];
            if let Some(ref mut film) = film.input_mixin_post_film {
                film.process(&mut mixin_scratch, cond);
            }

            let mix_v_modulated = _mm256_loadu_ps(mixin_scratch.as_ptr());
            let mut zv = _mm256_loadu_ps(z.add(off));
            zv = _mm256_add_ps(zv, mix_v_modulated);
            _mm256_storeu_ps(z.add(off), zv);

            let z_slice = &mut z_buf[off..off + ch];
            if let Some(ref mut film) = film.activation_pre_film {
                film.process(z_slice, cond);
            }
        }
    }

    // 3. LeakyReLU (in-place on z_buf).
    {
        let z = z_buf.as_mut_ptr();
        let slope_v = _mm256_set1_ps(A2_LEAKY_SLOPE);
        let zero_v = _mm256_setzero_ps();
        for f in 0..num_frames {
            let off = f * ch;
            let zv = _mm256_loadu_ps(z.add(off));
            let mask = _mm256_cmp_ps(zv, zero_v, _CMP_LT_OS);
            let zv_leaky = _mm256_mul_ps(zv, slope_v);
            _mm256_storeu_ps(z.add(off), _mm256_blendv_ps(zv, zv_leaky, mask));
        }
    }

    // 3b. FiLM: activation_post_film (post-activation).
    for f in 0..num_frames {
        let cond = &input_cond[f..f + 1];
        let z_slice = &mut z_buf[f * ch..(f + 1) * ch];
        if let Some(ref mut film) = film.activation_post_film {
            film.process(z_slice, cond);
        }
    }

    // 4. Head accumulate.
    {
        let head = head_accum.as_mut_ptr();
        for f in 0..num_frames {
            let head_off = (head_col + f) * ch;
            let zv = _mm256_loadu_ps(z_buf.as_ptr().add(f * ch));
            if is_first {
                _mm256_storeu_ps(head.add(head_off), zv);
            } else {
                let hv = _mm256_loadu_ps(head.add(head_off));
                _mm256_storeu_ps(head.add(head_off), _mm256_add_ps(hv, zv));
            }
        }
    }

    // 5. Layer1x1 residual (skipped on last layer) — isolated scratch buffer.
    if !is_last {
        let lin = layer_in.as_mut_ptr();
        let l1x1_b_v = _mm256_loadu_ps(l1x1_b.as_ptr());
        let l1x1_w_ptr = l1x1_w.as_ptr();
        for f in 0..num_frames {
            let off = f * ch;
            let mut acc = l1x1_b_v;
            for u in 0..ch {
                let zu = *z_buf.get_unchecked(off + u);
                let zu_v = _mm256_set1_ps(zu);
                let w_col = _mm256_loadu_ps(l1x1_w_ptr.add(u * ch));
                acc = _mm256_fmadd_ps(zu_v, w_col, acc);
            }

            let mut l1x1_scratch = [0.0f32; 8];
            _mm256_storeu_ps(l1x1_scratch.as_mut_ptr(), acc);

            let cond = &input_cond[f..f + 1];
            if let Some(film) = film
                .layer1x1_post_film
                .as_deref_mut()
                .filter(|_| use_blending)
            {
                film.process(&mut l1x1_scratch, cond);
            }

            let l1x1_v_modulated = _mm256_loadu_ps(l1x1_scratch.as_ptr());
            let lv = _mm256_loadu_ps(lin.add(off));
            _mm256_storeu_ps(lin.add(off), _mm256_add_ps(lv, l1x1_v_modulated));
        }
    }
}
