// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Single 2× oversampling stage (up + down half-band delay-line state).
//!
//! Uses pre-allocated double-buffer delay lines for contiguous SIMD access,
//! eliminating per-sample modulo indexing from the hot-path.

use super::oversample::HB_DELAY;
use super::oversample::HB_TAPS;
use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;
use crate::math::common::hsum_avx2;
use core::arch::x86_64::*;

const HB_ODD_COUNT: usize = HB_TAPS / 2;

const UP_DELAY_LINE_LEN: usize = HB_DELAY * 2;
const _: () = const { assert!(UP_DELAY_LINE_LEN > (HB_DELAY - 1) + (HB_ODD_COUNT - 1)) };
const DOWN_EVEN_LEN: usize = HB_TAPS.div_ceil(2);
const DOWN_ODD_LEN: usize = HB_TAPS / 2;
const DOWN_EVEN_DELAY_LINE_LEN: usize = DOWN_EVEN_LEN * 2;
const DOWN_ODD_DELAY_LINE_LEN: usize = DOWN_ODD_LEN * 2;

fn bessel_i0(x: f64) -> f64 {
    // Duplicated from `sinc_kernel.rs` to keep the half-band filter design
    // self-contained within this module. Both copies implement the same
    // series expansion (1 + Σ(x²/4)ᵏ/(k!)² for k=1..20).
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = x / 2.0;
    for k in 1..=20 {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term < 1e-15 * sum {
            break;
        }
    }
    sum
}

/// Half-band filter kernel with Kaiser window.
pub(crate) struct HalfBandFilter {
    pub(crate) coeffs: [f32; HB_ODD_COUNT],
}

impl HalfBandFilter {
    /// Designs a half-band FIR filter using a Kaiser window.
    ///
    /// The center tap (h[HB_DELAY]) is determined separately via `dc_gain` normalization.
    /// Odd-indexed coefficients are zero for half-band filters; only odd taps
    /// are stored.
    pub(crate) fn design(beta: f64, dc_gain: f64) -> Self {
        let i0_beta = bessel_i0(beta);
        let half = HB_DELAY as f64;
        let mut coeffs = [0.0f32; HB_ODD_COUNT];

        for i in 0..HB_TAPS {
            let offset = i as f64 - half;
            if offset.abs() < 1e-10 || (offset.abs() as i64) % 2 == 0 {
                continue;
            }

            let x = std::f64::consts::PI * offset;
            let sinc = (x * 0.5).sin() / x;
            let ratio = offset / half;
            let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
            let window = bessel_i0(arg) / i0_beta;

            if i % 2 == 1 {
                coeffs[i / 2] = (sinc * window) as f32;
            }
        }

        let target_h_center = dc_gain / 2.0;
        let target_odd_sum = dc_gain - target_h_center;
        let odd_sum: f32 = coeffs.iter().sum();
        if odd_sum.abs() > 1e-10 {
            let scale = target_odd_sum as f32 / odd_sum;
            for c in coeffs.iter_mut() {
                *c *= scale;
            }
        }

        coeffs.reverse();

        HalfBandFilter { coeffs }
    }
}

/// Single 2× oversampling stage (up + down delay-line state).
///
/// Uses pre-allocated double-buffer delay lines for contiguous SIMD access,
/// eliminating per-sample modulo indexing from the hot-path.
pub(crate) struct X2Stage {
    pub(crate) up_filter: HalfBandFilter,
    pub(crate) down_filter: HalfBandFilter,
    pub(crate) up_center: f32,
    pub(crate) down_center: f32,
    up_ring: AlignedVec<f32>,
    up_pos: usize,
    down_ring_even: AlignedVec<f32>,
    down_ring_odd: AlignedVec<f32>,
    down_pos_even: usize,
    down_pos_odd: usize,
    down_total: u64,
}

impl X2Stage {
    /// Creates a new 2× oversampling stage with default Kaiser beta=12.0 filters.
    ///
    /// Allocates mirrored delay-line buffers for contiguous SIMD access.
    /// Returns `Err(NamErrorCode)` on aligned allocation failure.
    pub(crate) fn new() -> Result<Self, NamErrorCode> {
        let dc_up = 2.0;
        let dc_down = 1.0;
        Ok(Self {
            up_filter: HalfBandFilter::design(12.0, dc_up),
            down_filter: HalfBandFilter::design(12.0, dc_down),
            up_center: (dc_up / 2.0) as f32,
            down_center: (dc_down / 2.0) as f32,
            up_ring: AlignedVec::new(UP_DELAY_LINE_LEN, 0.0f32)?,
            up_pos: 0,
            down_ring_even: AlignedVec::new(DOWN_EVEN_DELAY_LINE_LEN, 0.0f32)?,
            down_ring_odd: AlignedVec::new(DOWN_ODD_DELAY_LINE_LEN, 0.0f32)?,
            down_pos_even: 0,
            down_pos_odd: 0,
            down_total: 0,
        })
    }

    /// Upsamples `input` by 2× using the half-band FIR kernel.
    ///
    /// Produces interleaved even/odd output samples. Even samples use the
    /// center tap; odd samples are convolved against the odd-coefficient
    /// filter bank using AVX2 FMA.
    ///
    /// # Returns
    ///
    /// Number of output samples written (always `2 * input.len()`).
    #[inline(always)]
    pub(crate) fn upsample(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let coeffs = &self.up_filter.coeffs;
        let center = self.up_center;
        let n = HB_DELAY;
        let n_in = input.len();

        for (i, &x) in input.iter().enumerate() {
            // ── Step 1: Write sample into double-buffer delay line ──
            let p = self.up_pos;
            // SAFETY: p < HB_DELAY by invariant; p+n < UP_DELAY_LINE_LEN.
            // The mirror buffer is double-mapped, so both writes target the
            // same physical location for the primary and mirror halves.
            unsafe {
                core::hint::assert_unchecked(p < self.up_ring.len());
                core::hint::assert_unchecked(p + n < self.up_ring.len());
            }
            self.up_ring[p] = x;
            self.up_ring[p + n] = x;
            self.up_pos = (p + 1) % n;

            // ── Step 2: Read from delay line at write-head position ──
            // SAFETY: up_pos < HB_DELAY and up_ring has UP_DELAY_LINE_LEN elements
            // (verified by the compile-time assertion above). Reads at offsets 0..11
            // stay within bounds: up_pos + 11 < UP_DELAY_LINE_LEN.
            let wptr = unsafe { self.up_ring.as_ptr().add(self.up_pos) };

            // ── Step 3: Even output = center tap × delay[5] ──
            // SAFETY: wptr.add(5) is in bounds — see invariant above.
            let even_out = unsafe { *wptr.add(5) * center };

            // ── Step 4: Odd output = AVX2 8-wise dot product + scalar tail ──
            let odd_out = unsafe {
                let c8 = _mm256_loadu_ps(coeffs.as_ptr());
                let s8 = _mm256_loadu_ps(wptr);
                let acc8 = _mm256_fmadd_ps(c8, s8, _mm256_setzero_ps());
                let mut sum = hsum_avx2(acc8);
                // SAFETY: coeffs has HB_ODD_COUNT elements; wptr+offset in bounds.
                core::hint::assert_unchecked(8 < coeffs.len());
                core::hint::assert_unchecked(9 < coeffs.len());
                core::hint::assert_unchecked(10 < coeffs.len());
                core::hint::assert_unchecked(11 < coeffs.len());
                sum += coeffs[8] * *wptr.add(8);
                sum += coeffs[9] * *wptr.add(9);
                sum += coeffs[10] * *wptr.add(10);
                sum += coeffs[11] * *wptr.add(11);
                sum
            };

            // ── Step 5: Write interleaved output pair ──
            // SAFETY: output has 2*n_in elements (output.len() == 2 * input.len()).
            // 2*i+1 < 2*n_in for all i < n_in.
            unsafe {
                core::hint::assert_unchecked(2 * i < output.len());
                core::hint::assert_unchecked(2 * i + 1 < output.len());
            }
            output[2 * i] = even_out;
            output[2 * i + 1] = odd_out;
        }

        n_in * 2
    }

    /// Downsamples `input` by 2× using the half-band FIR kernel.
    ///
    /// Splits arrivals into even/odd sample queues, then processes each
    /// complete even/odd pair using a 12-tap half-band convolution.
    ///
    /// # Returns
    ///
    /// Number of output samples written (`input.len() / 2`, truncated).
    #[inline(always)]
    pub(crate) fn downsample(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let coeffs = &self.down_filter.coeffs;
        let center = self.down_center;
        let mut out_idx = 0;

        for &x in input.iter() {
            // ── Step 1: Demux input samples into even/odd delay lines ──
            let is_even = (self.down_total & 1) == 0;
            if is_even {
                let p = self.down_pos_even;
                // SAFETY: p < DOWN_EVEN_LEN; p+DOWN_EVEN_LEN < DOWN_EVEN_DELAY_LINE_LEN.
                // Both writes hit the same physical page in the mirrored buffer.
                unsafe {
                    core::hint::assert_unchecked(p < self.down_ring_even.len());
                    core::hint::assert_unchecked(p + DOWN_EVEN_LEN < self.down_ring_even.len());
                }
                self.down_ring_even[p] = x;
                self.down_ring_even[p + DOWN_EVEN_LEN] = x;
                self.down_pos_even = (p + 1) % DOWN_EVEN_LEN;
            } else {
                let p = self.down_pos_odd;
                // SAFETY: p < DOWN_ODD_LEN; p+DOWN_ODD_LEN < DOWN_ODD_DELAY_LINE_LEN.
                unsafe {
                    core::hint::assert_unchecked(p < self.down_ring_odd.len());
                    core::hint::assert_unchecked(p + DOWN_ODD_LEN < self.down_ring_odd.len());
                }
                self.down_ring_odd[p] = x;
                self.down_ring_odd[p + DOWN_ODD_LEN] = x;
                self.down_pos_odd = (p + 1) % DOWN_ODD_LEN;
            }
            self.down_total += 1;

            // ── Step 2: Every odd count after HB_TAPS samples, produce one output ──
            if self.down_total >= HB_TAPS as u64 && (self.down_total & 1) == 1 {
                // ── Step 2a: Read center tap from even delay line at offset 6 ──
                // SAFETY: down_pos_even < DOWN_EVEN_LEN; the mirror buffer doubles
                // the capacity, so reads up to offset 6 stay within bounds.
                let ev_ptr = unsafe { self.down_ring_even.as_ptr().add(self.down_pos_even) };
                let center_sample = unsafe { *ev_ptr.add(6) };
                let mut sum = center_sample * center;

                // ── Step 2b: Accumulate AVX2 8-wise dot product from odd delay line ──
                // SAFETY: down_pos_odd < DOWN_ODD_LEN; max access at offset 11
                // fits within DOWN_ODD_DELAY_LINE_LEN (2*DOWN_ODD_LEN).
                let od_ptr = unsafe { self.down_ring_odd.as_ptr().add(self.down_pos_odd) };
                unsafe {
                    let c8 = _mm256_loadu_ps(coeffs.as_ptr());
                    let s8 = _mm256_loadu_ps(od_ptr);
                    let acc8 = _mm256_fmadd_ps(c8, s8, _mm256_setzero_ps());
                    sum += hsum_avx2(acc8);
                    // SAFETY: coeffs[8..11] valid (HB_ODD_COUNT=12); od_ptr+offset in bounds.
                    core::hint::assert_unchecked(8 < coeffs.len());
                    core::hint::assert_unchecked(9 < coeffs.len());
                    core::hint::assert_unchecked(10 < coeffs.len());
                    core::hint::assert_unchecked(11 < coeffs.len());
                    sum += coeffs[8] * *od_ptr.add(8);
                    sum += coeffs[9] * *od_ptr.add(9);
                    sum += coeffs[10] * *od_ptr.add(10);
                    sum += coeffs[11] * *od_ptr.add(11);
                }

                // ── Step 2c: Emit completed downsampled sample ──
                if out_idx < output.len() {
                    output[out_idx] = sum;
                    out_idx += 1;
                }
            }
        }

        out_idx
    }
}
