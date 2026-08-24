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

/// The fused AVX2 dot product reads the first 8 coefficients with a 256-bit
/// load and the remaining 4 (`coeffs[8]`, `coeffs[9]`, `coeffs[10]`,
/// `coeffs[11]`) with a 128-bit load. These are in-bounds iff
/// `HB_ODD_COUNT > 11`, i.e. `HB_ODD_COUNT >= 12` (holds today: 25 / 2 = 12).
const _: () = const {
    assert!(
        HB_ODD_COUNT >= 12,
        "HB_ODD_COUNT (= HB_TAPS/2) must be >= 12 so the fused 8+4 tail (coeffs[8..12]) is in-bounds"
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

/// The furthest upsample read is `wptr.add(HB_ODD_COUNT - 1)` from the read
/// head at `p + 1` (the write position `p < HB_DELAY` plus one). It is
/// in-bounds for every `p` iff
/// `(HB_DELAY - 1) + 1 + (HB_ODD_COUNT - 1) < UP_DELAY_LINE_LEN`, i.e.
/// `HB_ODD_COUNT <= HB_DELAY`.
const _: () = const {
    assert!(
        UP_DELAY_LINE_LEN > (HB_DELAY - 1) + HB_ODD_COUNT,
        "UP_DELAY_LINE_LEN must exceed (HB_DELAY-1)+HB_ODD_COUNT for the furthest upsample read (head at p+1, span HB_ODD_COUNT)"
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

/// Returns the next phase index wrapping at `len`.
///
/// Cheaper than `% len` on the hot path (single compare + select instead of the
/// constant-division multiply-shift sequence).
#[inline(always)]
const fn next_phase(p: usize, len: usize) -> usize {
    let np = p + 1;
    if np == len { 0 } else { np }
}

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
    /// filter bank using fused AVX2 FMA (8 lanes + 4 lanes, single reduction).
    ///
    /// Fail-closed (F-05): the caller contract `output.len() >= 2*input.len()`
    /// is enforced here by clamping the processed sample count to
    /// `output.len() / 2`, so the `assert_unchecked` bounds below are
    /// established by this in-function check — not by an external
    /// `debug_assert!` that disappears in release builds.
    ///
    /// # Returns
    ///
    /// Number of output samples written (≤ `2 * input.len()`, clamped to
    /// `output.len()`).
    #[inline(always)]
    pub(crate) fn upsample(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let coeffs = &self.up_filter.coeffs;
        let center = self.up_center;
        let n = HB_DELAY;
        let n_in = input.len().min(output.len() / 2);

        // Phase batching (P-08 / T5.3): process samples in contiguous runs that
        // never wrap the write head, amortizing the `% n` modulo to once per run
        // instead of once per sample. Within a run the head advances
        // monotonically (`p = up_pos + j`), keeping the mirrored double-buffer
        // and the forward read window valid.
        let mut i = 0;
        while i < n_in {
            let run = (n - self.up_pos).min(n_in - i);
            for j in 0..run {
                let x = input[i + j];
                let p = self.up_pos + j;

                // ── Step 1: Write sample into double-buffer delay line ──
                // SAFETY: `up_pos` is maintained in `0..HB_DELAY` and
                // `j < run <= n - up_pos`, so `p = up_pos + j < n`. Hence
                // `p < UP_DELAY_LINE_LEN` and `p + n < 2*n = UP_DELAY_LINE_LEN`
                // (const assert `UP_DELAY_LINE_LEN >= 2*HB_DELAY`). Both writes
                // target the mirrored double-buffer within the allocation.
                unsafe {
                    core::hint::assert_unchecked(p < self.up_ring.len());
                    core::hint::assert_unchecked(p + n < self.up_ring.len());
                }
                self.up_ring[p] = x;
                self.up_ring[p + n] = x;

                // ── Step 2: Read from delay line at the next head position ──
                // SAFETY: the read window starts at `p + 1` and spans
                // `HB_ODD_COUNT` samples; the furthest read is at
                // `(p + 1) + (HB_ODD_COUNT - 1) <= (n - 1) + 1 + 11 <
                // UP_DELAY_LINE_LEN` (const assert
                // `UP_DELAY_LINE_LEN > (HB_DELAY-1)+HB_ODD_COUNT`).
                let wptr = unsafe { self.up_ring.as_ptr().add(p + 1) };

                // ── Step 3: Even output = center tap × delay[5] ──
                // SAFETY: `5 <= HB_ODD_COUNT - 1` and the furthest read
                // (offset `HB_ODD_COUNT - 1`) is proven in-bounds in Step 2.
                let even_out = unsafe { *wptr.add(5) * center };

                // ── Step 4: Odd output = fused AVX2 8-lane + 4-lane FMA ──
                // SAFETY: `coeffs` has `HB_ODD_COUNT >= 12` elements (const
                // assert `HB_ODD_COUNT >= 12`), so the 128-bit tail loads at
                // `coeffs[8..12]`/`wptr[8..12]` are in-bounds; the 256-bit head
                // loads at `[0..8]` are within the window proven in Step 2. The
                // tail lanes 4..8 are zero (via `_mm256_set_m128`), so a single
                // horizontal reduction covers all 12 coefficients.
                let odd_out = unsafe {
                    let acc8 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(coeffs.as_ptr()),
                        _mm256_loadu_ps(wptr),
                        _mm256_setzero_ps(),
                    );
                    let fused = _mm256_fmadd_ps(
                        _mm256_set_m128(_mm_setzero_ps(), _mm_loadu_ps(coeffs.as_ptr().add(8))),
                        _mm256_set_m128(_mm_setzero_ps(), _mm_loadu_ps(wptr.add(8))),
                        acc8,
                    );
                    hsum_avx2(fused)
                };

                // ── Step 5: Write interleaved output pair ──
                // SAFETY: the in-function clamp `n_in = input.len().min(output.len()/2)`
                // guarantees `i + j < n_in` implies
                // `2*(i+j)+1 <= 2*n_in-1 < output.len()` (F-05 — no reliance on
                // an external caller `debug_assert!`).
                let out = 2 * (i + j);
                // SAFETY: the in-function clamp `n_in = input.len().min(output.len()/2)`
                // guarantees `i + j < n_in` implies `out + 1 <= 2*n_in - 1 < output.len()`
                // (F-05), so both `out < output.len()` and `out + 1 < output.len()` hold.
                unsafe {
                    core::hint::assert_unchecked(out < output.len());
                    core::hint::assert_unchecked(out + 1 < output.len());
                }
                output[out] = even_out;
                output[out + 1] = odd_out;
            }
            self.up_pos = (self.up_pos + run) % n;
            i += run;
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
        let coeffs = self.down_filter.coeffs;
        let center = self.down_center;
        let mut out_idx = 0;

        // Phase batching (P-08 / T5.3): consume input in even/odd phase pairs so
        // the per-sample demux parity branch is eliminated. The phase at the
        // start of each call is derived from the persistent `down_total`, so
        // odd-length calls never desynchronize the batch loop (an odd total
        // consumes one leading odd-phase sample before pairing resumes). The
        // emit condition keeps the `& 1` test — it is evaluated once per pair
        // instead of once per sample.
        let mut i = 0;
        if (self.down_total & 1) == 1 && i < input.len() {
            self.write_odd_sample(input[i]);
            i += 1;
        }
        while i + 1 < input.len() {
            self.write_even_sample(input[i]);
            if self.down_total >= HB_TAPS as u64 && (self.down_total & 1) == 1 {
                self.emit_downsample_output(&coeffs, center, output, &mut out_idx);
            }
            self.write_odd_sample(input[i + 1]);
            i += 2;
        }
        if i < input.len() {
            self.write_even_sample(input[i]);
            if self.down_total >= HB_TAPS as u64 && (self.down_total & 1) == 1 {
                self.emit_downsample_output(&coeffs, center, output, &mut out_idx);
            }
        }

        out_idx
    }

    /// Writes one even-phase sample into the even mirrored delay line.
    #[inline(always)]
    fn write_even_sample(&mut self, x: f32) {
        let p = self.down_pos_even;
        // SAFETY: `down_pos_even` is maintained in `0..DOWN_EVEN_LEN`
        // (it is only ever assigned `next_phase(p, DOWN_EVEN_LEN)`), so
        // `p < DOWN_EVEN_LEN`. Hence `p < DOWN_EVEN_DELAY_LINE_LEN` and
        // `p + DOWN_EVEN_LEN <= 2*DOWN_EVEN_LEN - 1`; both follow directly
        // from `DOWN_EVEN_DELAY_LINE_LEN = 2 * DOWN_EVEN_LEN`. Both writes
        // hit the mirrored buffer.
        unsafe {
            core::hint::assert_unchecked(p < self.down_ring_even.len());
            core::hint::assert_unchecked(p + DOWN_EVEN_LEN < self.down_ring_even.len());
        }
        self.down_ring_even[p] = x;
        self.down_ring_even[p + DOWN_EVEN_LEN] = x;
        self.down_pos_even = next_phase(p, DOWN_EVEN_LEN);
        self.down_total += 1;
    }

    /// Writes one odd-phase sample into the odd mirrored delay line.
    #[inline(always)]
    fn write_odd_sample(&mut self, x: f32) {
        let p = self.down_pos_odd;
        // SAFETY: `down_pos_odd` is maintained in `0..DOWN_ODD_LEN`
        // (`next_phase(p, DOWN_ODD_LEN)`), so `p < DOWN_ODD_LEN`. Hence
        // `p < DOWN_ODD_DELAY_LINE_LEN` and
        // `p + DOWN_ODD_LEN <= 2*DOWN_ODD_LEN - 1`; both follow from
        // `DOWN_ODD_DELAY_LINE_LEN = 2 * DOWN_ODD_LEN`.
        unsafe {
            core::hint::assert_unchecked(p < self.down_ring_odd.len());
            core::hint::assert_unchecked(p + DOWN_ODD_LEN < self.down_ring_odd.len());
        }
        self.down_ring_odd[p] = x;
        self.down_ring_odd[p + DOWN_ODD_LEN] = x;
        self.down_pos_odd = next_phase(p, DOWN_ODD_LEN);
        self.down_total += 1;
    }

    /// Computes one downsampled sample from the current even/odd delay-line
    /// heads using a fused AVX2 8-lane + 4-lane FMA dot (single reduction,
    /// P-08 / T5.3), and appends it to `output`.
    #[inline(always)]
    fn emit_downsample_output(
        &mut self,
        coeffs: &[f32],
        center: f32,
        output: &mut [f32],
        out_idx: &mut usize,
    ) {
        // ── Step 2a: Read center tap from even delay line at offset 6 ──
        // SAFETY: `down_pos_even < DOWN_EVEN_LEN` (maintained) and the read
        // at offset 6 is in-bounds because
        // `down_pos_even + 6 <= (DOWN_EVEN_LEN - 1) + 6 < 2*DOWN_EVEN_LEN`,
        // i.e. `DOWN_EVEN_LEN >= 6` — proven by the const assert
        // `DOWN_EVEN_LEN >= 6`.
        let ev_ptr = unsafe { self.down_ring_even.as_ptr().add(self.down_pos_even) };
        // SAFETY: `ev_ptr` points to `down_ring_even[down_pos_even]` with
        // `down_pos_even < DOWN_EVEN_LEN`; the read at offset 6 is in-bounds
        // because `down_pos_even + 6 <= (DOWN_EVEN_LEN - 1) + 6 < 2*DOWN_EVEN_LEN`
        // (const assert `DOWN_EVEN_LEN >= 6`).
        let center_sample = unsafe { *ev_ptr.add(6) };
        let mut sum = center_sample * center;

        // ── Step 2b: Accumulate fused AVX2 8+4 dot from odd delay line ──
        // SAFETY: `down_pos_odd < DOWN_ODD_LEN` (maintained). The furthest
        // read `od_ptr.add(11)` is in-bounds because
        // `down_pos_odd + 11 <= (DOWN_ODD_LEN - 1) + 11 < 2*DOWN_ODD_LEN`,
        // i.e. `DOWN_ODD_LEN >= 11` — proven by the const assert
        // `DOWN_ODD_LEN >= 11`. `coeffs[8..12]` are in-bounds by the const
        // assert `HB_ODD_COUNT >= 12`.
        let od_ptr = unsafe { self.down_ring_odd.as_ptr().add(self.down_pos_odd) };
        // SAFETY: `od_ptr` points to `down_ring_odd[down_pos_odd]` with
        // `down_pos_odd < DOWN_ODD_LEN`; the furthest load `od_ptr.add(11)` is
        // in-bounds by the const assert `DOWN_ODD_LEN >= 11`, and the `coeffs`
        // tail loads `coeffs[8..12]` are in-bounds by `HB_ODD_COUNT >= 12`.
        unsafe {
            let acc8 = _mm256_fmadd_ps(
                _mm256_loadu_ps(coeffs.as_ptr()),
                _mm256_loadu_ps(od_ptr),
                _mm256_setzero_ps(),
            );
            let fused = _mm256_fmadd_ps(
                _mm256_set_m128(_mm_setzero_ps(), _mm_loadu_ps(coeffs.as_ptr().add(8))),
                _mm256_set_m128(_mm_setzero_ps(), _mm_loadu_ps(od_ptr.add(8))),
                acc8,
            );
            sum += hsum_avx2(fused);
        }

        // ── Step 2c: Emit completed downsampled sample ──
        if *out_idx < output.len() {
            output[*out_idx] = sum;
            *out_idx += 1;
        }
    }
}
