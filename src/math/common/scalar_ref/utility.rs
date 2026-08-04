// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// Converts high-precision decimal numbers (f32) to the compact format (BF16).
/// This saves a lot of memory and is used to store neural network "weights".
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn f32_to_bf16_fallback(src: &[f32], dest: &mut [u16]) {
    for (s, d) in src.iter().zip(dest.iter_mut()) {
        // Gets the 16 most important bits of the number and discards the rest.
        *d = (s.to_bits() >> 16) as u16;
    }
}

/// Applies the Tanh "squashing" function across an entire list of numbers.
///
/// Uses `f32::tanh()` as an **independent** reference oracle — not the Padé
/// approximant used by SIMD kernels.  This provides genuine cross-validation
/// in parity tests: the scalar fallback and SIMD paths now compute tanh via
/// different algorithms, so agreement is a meaningful correctness signal.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn tanh_slice_fallback(slice: &mut [f32]) {
    for v in slice.iter_mut() {
        *v = (*v).tanh();
    }
}

/// Applies the "Sigmoid" function across an entire list of numbers.
/// Sigmoid squashes numbers to stay between 0.0 and 1.0.
/// It's great for creating "gates" or automatic volume controls.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn sigmoid_slice_fallback(slice: &mut [f32]) {
    for v in slice.iter_mut() {
        *v = crate::math::activations::scalar_minimax_sigmoid(*v);
    }
}

/// Sums all numbers in a list and returns a single final value.
/// Uses Kahan compensated summation for bounded error O(ε) instead of O(N·ε).
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn horizontal_sum_fallback(ptr: *const f32, len: usize) -> f32 {
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let slice = unsafe { core::slice::from_raw_parts(ptr, len) };
    let mut sum = 0.0f32;
    let mut compensation = 0.0f32;
    for &x in slice {
        let y = x - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Adds a broadcast constant to every element of a mono buffer (fallback).
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn apply_dither_add_fallback(data: &mut [f32], offset: f32) {
    for x in data.iter_mut() {
        *x += offset;
    }
}

/// Fused gain + dither: `data[i] = data[i] * gain + offset` (scalar fallback).
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn apply_gain_then_dither_fallback(data: &mut [f32], gain: f32, offset: f32) {
    for x in data.iter_mut() {
        *x = f32::mul_add(*x, gain, offset);
    }
}

/// Applies a volume (gain) by multiplying each sample by the desired value.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn apply_gain_fallback(data: &mut [f32], gain: f32) {
    for x in data.iter_mut() {
        *x *= gain;
    }
}

/// Computes the energy (Mean Square) of a block via scalar.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn compute_energy_fallback(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum / (len as f32)
}

/// Computes the maximum energy between two channels via scalar.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn compute_energy_stereo_fallback(l: &[f32], r: &[f32]) -> f32 {
    let len = core::cmp::min(l.len(), r.len());
    if len == 0 {
        return 0.0;
    }
    let mut sum_l = 0.0f32;
    let mut sum_r = 0.0f32;
    for i in 0..len {
        let xl = *l.get_unchecked(i);
        let xr = *r.get_unchecked(i);
        sum_l += xl * xl;
        sum_r += xr * xr;
    }
    let energy_l = sum_l / (len as f32);
    let energy_r = sum_r / (len as f32);
    energy_l.max(energy_r)
}

/// Computes the maximum absolute difference between two blocks via scalar.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn compute_max_diff_fallback(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    if len == 0 {
        return 0.0;
    }
    let mut max_diff = 0.0f32;
    for i in 0..len {
        let d = (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    max_diff
}

/// Computes the peak absolute value of both channels (fallback).
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn compute_peak_abs_stereo_fallback(left: &[f32], right: &[f32]) -> (f32, f32) {
    let mut peak_l = 0.0f32;
    let mut peak_r = 0.0f32;
    let len = core::cmp::min(left.len(), right.len());
    for i in 0..len {
        let al = (*left.get_unchecked(i)).abs();
        let ar = (*right.get_unchecked(i)).abs();
        if al > peak_l {
            peak_l = al;
        }
        if ar > peak_r {
            peak_r = ar;
        }
    }
    (peak_l, peak_r)
}

/// Computes the peak absolute value of a single channel (fallback).
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn compute_peak_abs_mono_fallback(data: &[f32]) -> f32 {
    let mut peak = 0.0f32;
    for i in 0..data.len() {
        let a = (*data.get_unchecked(i)).abs();
        if a > peak {
            peak = a;
        }
    }
    peak
}

/// Crossfade blend (scalar fallback): `out[i] = out[i] * (1-t) + pending[i] * t`.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn crossfade_blend_mono_fallback(out: &mut [f32], pending: &[f32], t: f32) {
    let one_minus_t = 1.0 - t;
    let n = core::cmp::min(out.len(), pending.len());
    for i in 0..n {
        *out.get_unchecked_mut(i) =
            *out.get_unchecked(i) * one_minus_t + *pending.get_unchecked(i) * t;
    }
}

/// Complex multiply-accumulate (overwrite): scalar reference.
///
/// `out_re[i] = h_re[i] * x_re[i] - h_im[i] * x_im[i]`
/// `out_im[i] = h_re[i] * x_im[i] + h_im[i] * x_re[i]`
#[inline]
pub fn complex_mac_overwrite_scalar(
    h_re: &[f32],
    h_im: &[f32],
    x_re: &[f32],
    x_im: &[f32],
    out_re: &mut [f32],
    out_im: &mut [f32],
) {
    let n = h_re.len();
    for i in 0..n {
        let hr = h_re[i];
        let hi = h_im[i];
        let xr = x_re[i];
        let xi = x_im[i];
        out_re[i] = f32::mul_add(hr, xr, -hi * xi);
        out_im[i] = f32::mul_add(hr, xi, hi * xr);
    }
}

/// Complex multiply-accumulate (accumulate): scalar reference.
///
/// `acc_re[i] += h_re[i] * x_re[i] - h_im[i] * x_im[i]`
/// `acc_im[i] += h_re[i] * x_im[i] + h_im[i] * x_re[i]`
#[inline]
pub fn complex_mac_accumulate_scalar(
    h_re: &[f32],
    h_im: &[f32],
    x_re: &[f32],
    x_im: &[f32],
    acc_re: &mut [f32],
    acc_im: &mut [f32],
) {
    let n = h_re.len();
    for i in 0..n {
        let hr = h_re[i];
        let hi = h_im[i];
        let xr = x_re[i];
        let xi = x_im[i];
        acc_re[i] += f32::mul_add(hr, xr, -hi * xi);
        acc_im[i] += f32::mul_add(hr, xi, hi * xr);
    }
}

/// Frame-major scalar reference for batch normalization affine transform.
///
/// Computes `data[f*n_ch + c] = data[f*n_ch + c] * scale[c] + offset[c]`
/// traversing frames in the outer loop for contiguous channel access.
///
/// # Safety
/// `data.len() == num_frames * n_ch`. `scale` and `offset` must each have at least `n_ch` elements.
#[inline]
pub unsafe fn batch_norm_process_fallback(
    data: &mut [f32],
    scale: &[f32],
    offset: &[f32],
    n_ch: usize,
    num_frames: usize,
) {
    for f in 0..num_frames {
        let frame_start = f * n_ch;
        for c in 0..n_ch {
            let idx = frame_start + c;
            *data.get_unchecked_mut(idx) = (*data.get_unchecked(idx))
                .mul_add(*scale.get_unchecked(c), *offset.get_unchecked(c));
        }
    }
}

/// Scalar Radix-2 DIT FFT butterfly stage for one group.
///
/// See `SimdMath::fft_butterfly_stage` for semantics.
///
/// # Safety
/// See `SimdMath::fft_butterfly_stage`.
#[inline]
pub unsafe fn fft_butterfly_stage_scalar(
    re: *mut f32,
    im: *mut f32,
    half: usize,
    tw_re: *const f32,
    tw_im: *const f32,
    group_start: usize,
    inverse: bool,
) {
    let top = group_start;
    let bot = group_start + half;
    for j in 0..half {
        let w_re = *tw_re.add(j);
        let w_im = if inverse {
            -(*tw_im.add(j))
        } else {
            *tw_im.add(j)
        };

        let re_top = *re.add(top + j);
        let im_top = *im.add(top + j);
        let re_bot = *re.add(bot + j);
        let im_bot = *im.add(bot + j);

        let t_re = f32::mul_add(w_re, re_bot, -w_im * im_bot);
        let t_im = f32::mul_add(w_re, im_bot, w_im * re_bot);

        *re.add(bot + j) = re_top - t_re;
        *im.add(bot + j) = im_top - t_im;
        *re.add(top + j) = re_top + t_re;
        *im.add(top + j) = im_top + t_im;
    }
}
