// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Radix-4 Decimation-in-Time (DIT) FFT — Research Prototype
//!
//! **Status: Complete — Do not use in production.**
//!
//! This module is a research artifact preserved for future reference.
//! The prototype implements an in-place iterative Radix-4 DIT FFT over
//! SoA buffers (`&mut [T]`), with an API compatible with the production
//! `FftPlanner` (Radix-2).
//!
//! # Engineering Decision
//!
//! Criterion benchmarks (N=256 and N=1024, f32) demonstrate that the scalar
//! Radix-4 is **7–19% slower** than the scalar Radix-2, despite having half
//! the stages (`log₄N` vs `log₂N`). Root causes:
//!
//! 1. **3× more twiddle accesses** per butterfly (W¹, W², W³), stressing L1
//!    data cache.
//! 2. **Strided access pattern** (L, 2L, 3L) that degrades hardware prefetch.
//! 3. **Heavier butterfly**: 30 operations for 4 elements vs 8 operations for
//!    2 elements in Radix-2 — in practice worse than the theoretical 3.75:4
//!    ops/element due to register pressure and branching.
//! 4. **Conditional branch** (`if inverse`) in the inner loop, breaking
//!    compiler pipelining.
//!
//! Stockham (auto-sort, eliminates bit-reversal) and Split-Radix were also
//! analyzed and discarded. Bit-reversal represents <2% of total time for
//! N≤1024; Split-Radix has an irregular access pattern that prevents
//! efficient SIMD vectorization.
//!
//! The canonical project algorithm remains the Radix-2 DIT with SIMD
//! acceleration, implemented in the canonical Radix-2 SIMD FFT planner.
//!
//! # Research History
//!
//! * **Theory**: Radix-4 DIT would have a theoretical advantage (~6% fewer
//!   operations), half the stages, and SIMD potential with register reuse.
//! * **Prototype**: Functional implementation with 14 tests (forward parity
//!   vs Radix-2 for N=4,16,64,256,1024; roundtrip; impulse; f64). Applied
//!   corrections: base-4 bit-reversal (instead of base-2) and swapping the
//!   X₁/X₃ formulas in the inverse butterfly.
//! * **Benchmarks**: `cargo bench --bench fft_radix4_bench`
//! * **Conclusion**: Radix-4, Stockham, and Split-Radix do not justify the
//!   added complexity compared to Radix-2 SIMD.
//!
//! # Technical Limitations (for reference)
//!
//! * N must be a **power of 4** (4, 16, 64, 256, 1024, …). For mixed sizes
//!   (512, 2048), a hybrid Radix-2+4 would be required.
//! * Prototype is **scalar**; SIMD would require a new method in the
//!   `SimdMath` trait (analogous to `fft_butterfly_stage`) with
//!   shuffle/permute to recombine the 4 butterfly outputs from the 3
//!   twiddled inputs.

#[cfg(any(test, feature = "long_bench"))]
use super::fft::FftFloat;

/// Pre-computed Radix-4 DIT FFT plan.
#[cfg(any(test, feature = "long_bench"))]
#[cfg_attr(docsrs, doc(cfg(feature = "long_bench")))]
pub struct FftPlannerRadix4<T: FftFloat> {
    n: usize,
    bit_reverse: Vec<usize>,
    stage_twiddle_re1: Vec<T>,
    stage_twiddle_im1: Vec<T>,
    stage_twiddle_re2: Vec<T>,
    stage_twiddle_im2: Vec<T>,
    stage_twiddle_re3: Vec<T>,
    stage_twiddle_im3: Vec<T>,
    stage_l: Vec<usize>,
}

#[cfg(any(test, feature = "long_bench"))]
#[cfg_attr(docsrs, doc(cfg(feature = "long_bench")))]
impl<T: FftFloat> FftPlannerRadix4<T> {
    /// Creates a new Radix-4 FFT plan for size `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is not a power of two, is less than 4.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "FFT size must be positive");
        assert!(
            n.is_power_of_two(),
            "FFT size must be a power of two (Radix-4 requires power of four), got {n}"
        );
        assert!(n >= 4, "Radix-4 FFT requires N ≥ 4, got {n}");

        let n_half = n / 2;
        let num_stages_radix4 = (n.ilog2() / 2) as usize;
        let base4_digits = num_stages_radix4;

        let mut bit_reverse = vec![0usize; n];
        #[expect(
            clippy::needless_range_loop,
            reason = "Range loop required for explicit SIMD lane indexing not expressible via iterator"
        )]
        for i in 0..n {
            let mut rev = 0usize;
            let mut x = i;
            for _ in 0..base4_digits {
                rev = (rev << 2) | (x & 0x3);
                x >>= 2;
            }
            bit_reverse[i] = rev;
        }

        let tau = T::tau();
        let n_t = T::from_usize(n);
        let twiddle_re: Vec<T> = (0..n_half)
            .map(|k| {
                let angle = tau * T::from_usize(k) * n_t.recip();
                angle.cos()
            })
            .collect();
        let twiddle_im: Vec<T> = (0..n_half)
            .map(|k| {
                let angle = tau * T::from_usize(k) * n_t.recip();
                -angle.sin()
            })
            .collect();

        let total = n.saturating_sub(1);
        let mut stage_twiddle_re1 = Vec::with_capacity(total);
        let mut stage_twiddle_im1 = Vec::with_capacity(total);
        let mut stage_twiddle_re2 = Vec::with_capacity(total);
        let mut stage_twiddle_im2 = Vec::with_capacity(total);
        let mut stage_twiddle_re3 = Vec::with_capacity(total);
        let mut stage_twiddle_im3 = Vec::with_capacity(total);
        let mut stage_l = Vec::with_capacity(num_stages_radix4);

        let mut len = 4;
        while len <= n {
            let l = len / 4;
            let step = n / len;
            stage_l.push(l);
            for j in 0..l {
                let w1 = j * step;
                let w2 = (2 * j) * step;
                let w3 = (3 * j) * step;
                stage_twiddle_re1.push(twiddle_re[w1]);
                stage_twiddle_im1.push(twiddle_im[w1]);
                stage_twiddle_re2.push(twiddle_re[w2]);
                stage_twiddle_im2.push(twiddle_im[w2]);
                if w3 < n_half {
                    stage_twiddle_re3.push(twiddle_re[w3]);
                    stage_twiddle_im3.push(twiddle_im[w3]);
                } else {
                    stage_twiddle_re3.push(-twiddle_re[w3 - n_half]);
                    stage_twiddle_im3.push(-twiddle_im[w3 - n_half]);
                }
            }
            len <<= 2;
        }

        Self {
            n,
            bit_reverse,
            stage_twiddle_re1,
            stage_twiddle_im1,
            stage_twiddle_re2,
            stage_twiddle_im2,
            stage_twiddle_re3,
            stage_twiddle_im3,
            stage_l,
        }
    }

    /// Returns the FFT size.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the size is zero (never, guarded at construction).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Forward complex FFT, in-place on SoA buffers.
    ///
    /// Validates buffer lengths with `assert_eq!` in all profiles
    /// before delegating to the unchecked Radix-4 kernel.
    pub fn process(&self, re: &mut [T], im: &mut [T]) {
        assert_eq!(re.len(), self.n, "re length mismatch");
        assert_eq!(im.len(), self.n, "im length mismatch");

        // Bit-reversal
        for (i, &j) in self.bit_reverse.iter().enumerate() {
            if i < j {
                // SAFETY: `re.len() == im.len() == self.n` (asserted above) and
                // `bit_reverse` holds a permutation of `0..self.n`, so both `i`
                // and `j` index in bounds.
                unsafe {
                    std::ptr::swap(re.get_unchecked_mut(i), re.get_unchecked_mut(j));
                    std::ptr::swap(im.get_unchecked_mut(i), im.get_unchecked_mut(j));
                }
            }
        }

        // Radix-4 butterflies
        let mut tw_offset = 0usize;
        for &l in &self.stage_l {
            // SAFETY: `re.len() == im.len() == self.n` (asserted above); `l`
            // and `tw_offset` come from the construction-time stage tables, so
            // `radix4_stage`'s documented preconditions hold.
            unsafe { self.radix4_stage(re, im, l, tw_offset, false) };
            tw_offset += l;
        }
    }

    /// Inverse complex FFT, in-place on SoA buffers.
    ///
    /// Validates buffer lengths with `assert_eq!` in all profiles
    /// before delegating to the unchecked Radix-4 kernel. Applies
    /// `1/n` scaling at the end.
    pub fn process_inverse(&self, re: &mut [T], im: &mut [T]) {
        assert_eq!(re.len(), self.n, "re length mismatch");
        assert_eq!(im.len(), self.n, "im length mismatch");

        for (i, &j) in self.bit_reverse.iter().enumerate() {
            if i < j {
                // SAFETY: `re.len() == im.len() == self.n` (asserted above) and
                // `bit_reverse` holds a permutation of `0..self.n`, so both `i`
                // and `j` index in bounds.
                unsafe {
                    std::ptr::swap(re.get_unchecked_mut(i), re.get_unchecked_mut(j));
                    std::ptr::swap(im.get_unchecked_mut(i), im.get_unchecked_mut(j));
                }
            }
        }

        let mut tw_offset = 0usize;
        for &l in &self.stage_l {
            // SAFETY: `re.len() == im.len() == self.n` (asserted above); `l`
            // and `tw_offset` come from the construction-time stage tables, so
            // `radix4_stage`'s documented preconditions hold.
            unsafe { self.radix4_stage(re, im, l, tw_offset, true) };
            tw_offset += l;
        }

        let scale = T::from_usize(self.n).recip();
        for s in re.iter_mut() {
            *s = *s * scale;
        }
        for s in im.iter_mut() {
            *s = *s * scale;
        }
    }

    /// Processes one Radix-4 butterfly stage.
    ///
    /// # Safety
    ///
    /// The caller must guarantee:
    /// - `re.len() == im.len() == self.n`
    /// - `l` is a valid stage sub-block size where `4 * l ≤ n`
    /// - `tw_offset + l` does not exceed the capacity of stage twiddle
    ///   vectors, which are pre-computed at construction time.
    ///
    /// For a stage of length `4*L`:
    /// - Processes `N / (4*L)` groups of 4 elements each.
    /// - Each group at positions [k, k+L, k+2L, k+3L].
    /// - Twiddles for offset j (0..L) are at tw_offset + j.
    unsafe fn radix4_stage(
        &self,
        re: &mut [T],
        im: &mut [T],
        l: usize,
        tw_offset: usize,
        inverse: bool,
    ) {
        let len = 4 * l;
        for k in (0..self.n).step_by(len) {
            for j in 0..l {
                let idx0 = k + j;
                let idx1 = k + j + l;
                let idx2 = k + j + 2 * l;
                let idx3 = k + j + 3 * l;

                let tw_idx = tw_offset + j;
                // SAFETY: `tw_idx = tw_offset + j < tw_offset + l`, and the
                // caller's `tw_offset`/`l` pair satisfies `tw_offset + l <=
                // stage_twiddle_re1.len()` (`radix4_stage` precondition).
                let w1_re = unsafe { *self.stage_twiddle_re1.get_unchecked(tw_idx) };
                let w1_im = if inverse {
                    // SAFETY: `tw_idx < tw_offset + l <= stage_twiddle_im1.len()`.
                    unsafe { -*self.stage_twiddle_im1.get_unchecked(tw_idx) }
                } else {
                    // SAFETY: `tw_idx < tw_offset + l <= stage_twiddle_im1.len()`.
                    unsafe { *self.stage_twiddle_im1.get_unchecked(tw_idx) }
                };
                // SAFETY: `tw_idx < tw_offset + l <= stage_twiddle_re2.len()`.
                let w2_re = unsafe { *self.stage_twiddle_re2.get_unchecked(tw_idx) };
                let w2_im = if inverse {
                    // SAFETY: `tw_idx < tw_offset + l <= stage_twiddle_im2.len()`.
                    unsafe { -*self.stage_twiddle_im2.get_unchecked(tw_idx) }
                } else {
                    // SAFETY: `tw_idx < tw_offset + l <= stage_twiddle_im2.len()`.
                    unsafe { *self.stage_twiddle_im2.get_unchecked(tw_idx) }
                };
                // SAFETY: `tw_idx < tw_offset + l <= stage_twiddle_re3.len()`.
                let w3_re = unsafe { *self.stage_twiddle_re3.get_unchecked(tw_idx) };
                let w3_im = if inverse {
                    // SAFETY: `tw_idx < tw_offset + l <= stage_twiddle_im3.len()`.
                    unsafe { -*self.stage_twiddle_im3.get_unchecked(tw_idx) }
                } else {
                    // SAFETY: `tw_idx < tw_offset + l <= stage_twiddle_im3.len()`.
                    unsafe { *self.stage_twiddle_im3.get_unchecked(tw_idx) }
                };

                // SAFETY: `idx3 = k + j + 3*l <= self.n - 1` because `k` steps
                // by `4*l` (so `k <= n - 4*l`) and `j < l`; `re`/`im` have
                // length `self.n` (`radix4_stage` precondition).
                let (r0, i0, r1, i1, r2, i2, r3, i3) = unsafe {
                    (
                        *re.get_unchecked(idx0),
                        *im.get_unchecked(idx0),
                        *re.get_unchecked(idx1),
                        *im.get_unchecked(idx1),
                        *re.get_unchecked(idx2),
                        *im.get_unchecked(idx2),
                        *re.get_unchecked(idx3),
                        *im.get_unchecked(idx3),
                    )
                };

                let y1_re = w1_re.mul_add(r1, -w1_im * i1);
                let y1_im = w1_re.mul_add(i1, w1_im * r1);
                let y2_re = w2_re.mul_add(r2, -w2_im * i2);
                let y2_im = w2_re.mul_add(i2, w2_im * r2);
                let y3_re = w3_re.mul_add(r3, -w3_im * i3);
                let y3_im = w3_re.mul_add(i3, w3_im * r3);

                // SAFETY: all `idx0..idx3` are `< self.n` (same bounds as the
                // reads above) and `re`/`im` have length `self.n`, so the
                // unchecked writes stay in bounds.
                unsafe {
                    *re.get_unchecked_mut(idx0) = (r0 + y1_re) + (y2_re + y3_re);
                    *im.get_unchecked_mut(idx0) = (i0 + y1_im) + (y2_im + y3_im);
                    if inverse {
                        *re.get_unchecked_mut(idx3) = (r0 + y1_im) - (y2_re + y3_im);
                        *im.get_unchecked_mut(idx3) = (i0 - y1_re) - (y2_im - y3_re);
                        *re.get_unchecked_mut(idx2) = (r0 - y1_re) + (y2_re - y3_re);
                        *im.get_unchecked_mut(idx2) = (i0 - y1_im) + (y2_im - y3_im);
                        *re.get_unchecked_mut(idx1) = (r0 - y1_im) - (y2_re - y3_im);
                        *im.get_unchecked_mut(idx1) = (i0 + y1_re) - (y2_im + y3_re);
                    } else {
                        *re.get_unchecked_mut(idx1) = (r0 + y1_im) - (y2_re + y3_im);
                        *im.get_unchecked_mut(idx1) = (i0 - y1_re) - (y2_im - y3_re);
                        *re.get_unchecked_mut(idx2) = (r0 - y1_re) + (y2_re - y3_re);
                        *im.get_unchecked_mut(idx2) = (i0 - y1_im) + (y2_im - y3_im);
                        *re.get_unchecked_mut(idx3) = (r0 - y1_im) - (y2_re - y3_im);
                        *im.get_unchecked_mut(idx3) = (i0 + y1_re) - (y2_im + y3_re);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "fft_radix4_test.rs"]
mod tests;
