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
const DOWN_EVEN_LEN: usize = HB_TAPS.div_ceil(2);
const DOWN_ODD_LEN: usize = HB_TAPS / 2;
const DOWN_EVEN_DELAY_LINE_LEN: usize = DOWN_EVEN_LEN * 2;
const DOWN_ODD_DELAY_LINE_LEN: usize = DOWN_ODD_LEN * 2;

// ── Compile-time invariant proofs (F-09 / T4.1) ────────────────────────────
//
// The 16 `assert_unchecked` calls below eliminate runtime bounds checks on the
// AVX2/FMA hot path. Each one is *only* sound because of a specific arithmetic
// relation between the filter constants above. Those relations are proven here
// with `const { assert!(..) }`, so any future edit to `HB_TAPS`/`HB_DELAY`
// that breaks a bound fails the build instead of turning into release-only UB.
// Every `// SAFETY:` comment inside `upsample`/`downsample` cites the exact
// const assertion that sustains it.

/// The unrolled scalar tail of the odd-tap dot product reads `coeffs[8]`,
/// `coeffs[9]`, `coeffs[10]`, `coeffs[11]`. These are in-bounds iff
/// `HB_ODD_COUNT > 11`, i.e. `HB_ODD_COUNT >= 12` (holds today: 25 / 2 = 12).
const _: () = const {
    assert!(
        HB_ODD_COUNT >= 12,
        "HB_ODD_COUNT (= HB_TAPS/2) must be >= 12 so the unrolled tail coeffs[8..12] is in-bounds"
    )
};

/// The upsample delay line is a mirrored double-buffer of `2 * HB_DELAY`
/// elements. The mirrored write `up_ring[p + HB_DELAY]` is in-bounds for every
/// `p < HB_DELAY` iff `2*HB_DELAY - 1 < UP_DELAY_LINE_LEN`.
const _: () = const {
    assert!(
        UP_DELAY_LINE_LEN >= 2 * HB_DELAY,
        "UP_DELAY_LINE_LEN must be >= 2*HB_DELAY for the mirrored write up_ring[p + HB_DELAY]"
    )
};

/// The furthest upsample read is `wptr.add(HB_ODD_COUNT - 1)` from a write
/// head `up_pos < HB_DELAY`. It is in-bounds for every `up_pos` iff
/// `(HB_DELAY - 1) + (HB_ODD_COUNT - 1) < UP_DELAY_LINE_LEN`.
const _: () = const {
    assert!(
        UP_DELAY_LINE_LEN > (HB_DELAY - 1) + (HB_ODD_COUNT - 1),
        "UP_DELAY_LINE_LEN must exceed (HB_DELAY-1)+(HB_ODD_COUNT-1) for the furthest upsample read"
    )
};

/// The downsample center-tap read `ev_ptr.add(6)` from a write head
/// `down_pos_even < DOWN_EVEN_LEN` is in-bounds iff
/// `(DOWN_EVEN_LEN - 1) + 6 < DOWN_EVEN_DELAY_LINE_LEN`, i.e. `DOWN_EVEN_LEN > 5`.
const _: () = const {
    assert!(
        DOWN_EVEN_LEN >= 6,
        "DOWN_EVEN_LEN must be >= 6 for the center-tap read down_ring_even[down_pos_even + 6]"
    )
};

/// The furthest downsample read is `od_ptr.add(11)` from a write head
/// `down_pos_odd < DOWN_ODD_LEN`. It is in-bounds iff
/// `(DOWN_ODD_LEN - 1) + 11 < DOWN_ODD_DELAY_LINE_LEN`, i.e. `DOWN_ODD_LEN > 10`.
const _: () = const {
    assert!(
        DOWN_ODD_LEN >= 11,
        "DOWN_ODD_LEN must be >= 11 for the furthest read down_ring_odd[down_pos_odd + 11]"
    )
};

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
            // SAFETY: `up_pos` is maintained in `0..HB_DELAY` (it is only ever
            // assigned `(p + 1) % n` with `n = HB_DELAY`), so `p < HB_DELAY`.
            // Hence `p < UP_DELAY_LINE_LEN` (const assert
            // `UP_DELAY_LINE_LEN >= 2*HB_DELAY`) and `p + n <= 2*HB_DELAY - 1
            // < UP_DELAY_LINE_LEN` (same const assert). Both writes target the
            // mirrored double-buffer, so they stay within the allocation.
            unsafe {
                core::hint::assert_unchecked(p < self.up_ring.len());
                core::hint::assert_unchecked(p + n < self.up_ring.len());
            }
            self.up_ring[p] = x;
            self.up_ring[p + n] = x;
            self.up_pos = (p + 1) % n;

            // ── Step 2: Read from delay line at write-head position ──
            // SAFETY: `up_pos < HB_DELAY` (above) and `up_ring` has
            // `UP_DELAY_LINE_LEN` elements, so `wptr` points at a valid element.
            // The subsequent reads `wptr.add(offset)` for `offset in 0..HB_ODD_COUNT`
            // are bounded by `up_pos + (HB_ODD_COUNT - 1) < UP_DELAY_LINE_LEN`,
            // proven by the const assert
            // `UP_DELAY_LINE_LEN > (HB_DELAY-1)+(HB_ODD_COUNT-1)`.
            let wptr = unsafe { self.up_ring.as_ptr().add(self.up_pos) };

            // ── Step 3: Even output = center tap × delay[5] ──
            // SAFETY: `wptr.add(5)` is in-bounds because `5 <= HB_ODD_COUNT - 1`
            // and the furthest read (offset `HB_ODD_COUNT - 1`) is proven
            // in-bounds by the const assert cited in Step 2.
            let even_out = unsafe { *wptr.add(5) * center };

            // ── Step 4: Odd output = AVX2 8-wise dot product + scalar tail ──
            // SAFETY: `coeffs` has `HB_ODD_COUNT` elements and `HB_ODD_COUNT >= 12`
            // is proven by the const assert `HB_ODD_COUNT >= 12`, so the tail
            // indices 8..12 are in-bounds. The 8-wide AVX2 load reads
            // `coeffs[0..8]` and `wptr[0..8]`, which are also within the bounds
            // proven in Step 2 (`8 <= HB_ODD_COUNT - 1`).
            let odd_out = unsafe {
                let c8 = _mm256_loadu_ps(coeffs.as_ptr());
                let s8 = _mm256_loadu_ps(wptr);
                let acc8 = _mm256_fmadd_ps(c8, s8, _mm256_setzero_ps());
                let mut sum = hsum_avx2(acc8);
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
            // SAFETY: the caller contract is `output.len() == 2 * input.len()`
            // (enforced by `OversampleEngine::upsample`'s `debug_assert!`). With
            // `i < n_in`, both `2*i` and `2*i + 1` are `< 2*n_in == output.len()`.
            // This is a runtime caller invariant (not a compile-time constant),
            // so no `const` assert applies to these two.
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
                // SAFETY: `down_pos_even` is maintained in `0..DOWN_EVEN_LEN`
                // (it is only ever assigned `(p + 1) % DOWN_EVEN_LEN`), so
                // `p < DOWN_EVEN_LEN`. Hence `p < DOWN_EVEN_DELAY_LINE_LEN` and
                // `p + DOWN_EVEN_LEN <= 2*DOWN_EVEN_LEN - 1`; both follow
                // directly from the const definition
                // `DOWN_EVEN_DELAY_LINE_LEN = 2 * DOWN_EVEN_LEN`, so no extra
                // const assert is required. Both writes hit the mirrored buffer.
                unsafe {
                    core::hint::assert_unchecked(p < self.down_ring_even.len());
                    core::hint::assert_unchecked(p + DOWN_EVEN_LEN < self.down_ring_even.len());
                }
                self.down_ring_even[p] = x;
                self.down_ring_even[p + DOWN_EVEN_LEN] = x;
                self.down_pos_even = (p + 1) % DOWN_EVEN_LEN;
            } else {
                let p = self.down_pos_odd;
                // SAFETY: `down_pos_odd` is maintained in `0..DOWN_ODD_LEN`
                // (`down_pos_odd = (p + 1) % DOWN_ODD_LEN`), so `p < DOWN_ODD_LEN`.
                // Hence `p < DOWN_ODD_DELAY_LINE_LEN` and
                // `p + DOWN_ODD_LEN <= 2*DOWN_ODD_LEN - 1`; both follow from the
                // const definition `DOWN_ODD_DELAY_LINE_LEN = 2 * DOWN_ODD_LEN`.
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
                // SAFETY: `down_pos_even < DOWN_EVEN_LEN` (above) and the read
                // at offset 6 is in-bounds because
                // `down_pos_even + 6 <= (DOWN_EVEN_LEN - 1) + 6 < 2*DOWN_EVEN_LEN`,
                // i.e. `DOWN_EVEN_LEN >= 6` — proven by the const assert
                // `DOWN_EVEN_LEN >= 6`.
                let ev_ptr = unsafe { self.down_ring_even.as_ptr().add(self.down_pos_even) };
                let center_sample = unsafe { *ev_ptr.add(6) };
                let mut sum = center_sample * center;

                // ── Step 2b: Accumulate AVX2 8-wise dot product from odd delay line ──
                // SAFETY: `down_pos_odd < DOWN_ODD_LEN` (above). The furthest
                // read `od_ptr.add(11)` is in-bounds because
                // `down_pos_odd + 11 <= (DOWN_ODD_LEN - 1) + 11 < 2*DOWN_ODD_LEN`,
                // i.e. `DOWN_ODD_LEN >= 11` — proven by the const assert
                // `DOWN_ODD_LEN >= 11`. `coeffs[8..12]` are in-bounds by the
                // const assert `HB_ODD_COUNT >= 12`.
                let od_ptr = unsafe { self.down_ring_odd.as_ptr().add(self.down_pos_odd) };
                unsafe {
                    let c8 = _mm256_loadu_ps(coeffs.as_ptr());
                    let s8 = _mm256_loadu_ps(od_ptr);
                    let acc8 = _mm256_fmadd_ps(c8, s8, _mm256_setzero_ps());
                    sum += hsum_avx2(acc8);
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
