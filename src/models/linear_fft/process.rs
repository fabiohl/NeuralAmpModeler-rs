// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Hot-path processing methods for the FFT convolution state.
//!
//! Separated from the state definition to keep the core struct and
//! constructors in `linear_fft.rs` while isolating the RT-critical
//! reset and tail block processing logic.

use crate::math::common::Avx2Math;
use crate::math::common::Avx512Math;
use crate::math::common::Avx512VnniBf16Math;
use crate::math::common::dispatch::InstructionSet;
use crate::math::common::traits::SimdMath;

impl super::LinearFftState {
    /// Resets all runtime buffers to zero and re-initializes counters.
    ///
    /// This operation is allocation-free: it only zero-fills the existing
    /// pre-allocated buffers. The pre-computed `h_fdl_*` spectra are
    /// **not** modified (they depend only on the IR, which is static).
    #[cold]
    pub fn reset(&mut self) {
        self.fdl_re.fill(0.0f32);
        self.fdl_im.fill(0.0f32);
        self.input_buf.fill(0.0f32);
        self.fft_re.fill(0.0f32);
        self.fft_im.fill(0.0f32);
        self.acc_re.fill(0.0f32);
        self.acc_im.fill(0.0f32);
        self.output_buf.fill(0.0f32);
        self.tail_output_buf.fill(0.0f32);
        self.fdl_write_idx = self.num_partitions.saturating_sub(1);
        self.sample_counter = 0;
    }

    /// Processes one block of tail convolution using overlap-save FFT.
    ///
    /// `input_window` must be a contiguous slice of the last `2P` input
    /// samples (oldest to newest), typically obtained from the
    /// `MirroredBuffer` via `history[write_pos - 2*P .. write_pos]`.
    ///
    /// After this call completes, `tail_output_buf` contains `P` valid
    /// tail output samples ready for sequential per-sample consumption
    /// via `sample_counter`.
    ///
    /// # Algorithm (overlap-save, zero-latency hybrid)
    ///
    /// 1. Compute forward RFFT of the `2P`-sample input window.
    /// 2. Read past input spectra from the circular FDL (delays `P`,
    ///    `2P`, …, `K×P`), multiply by the corresponding pre-computed
    ///    tail IR spectra in the frequency domain using SIMD complex MAC.
    /// 3. Store the new input spectrum in the FDL and advance the write
    ///    index (overwrites the oldest entry, now `K+1` blocks ago).
    /// 4. Inverse RFFT the accumulated spectrum back to time domain.
    /// 5. Extract the valid `P` output samples (overlap-save: indices
    ///    `P..2P-1`) into `tail_output_buf`.
    ///
    /// # RT-Safety
    ///
    /// Zero heap allocation, zero locks, zero panics in production
    /// (debug assertions only). All buffers were pre-allocated at
    /// construction time. The ISA for SIMD dispatch was captured once
    /// at construction time — no runtime CPU feature checks on the hot
    /// path.
    pub fn process_tail_block(&mut self, input_window: &[f32]) {
        let p = self.p;
        let block_size = 2 * p;
        let num_bins = self.num_bins;
        let num_partitions = self.num_partitions;

        debug_assert_eq!(input_window.len(), block_size);

        if num_partitions == 0 {
            return;
        }

        // ── Step 1: Copy input window and compute forward RFFT ──
        self.input_buf[..block_size].copy_from_slice(input_window);
        self.rfft
            .process_forward(&self.input_buf, &mut self.fft_re, &mut self.fft_im);

        // ── Step 2: Frequency-domain MAC ──
        // The tail output for the NEXT block needs the current input spectrum
        // (block B) for partition 0 (delay P) and FDL entries (blocks B-1,
        // B-2, ...) for partitions 1..K-1 (delays 2P, 3P, ..., K×P).
        // SAFETY: all slices have length num_bins, guaranteed by construction.
        // ISA was captured at construction time.
        self.acc_re[..num_bins].fill(0.0);
        self.acc_im[..num_bins].fill(0.0);

        // Partition 0 (delays P..2P−1): uses the current block's input spectrum.
        unsafe {
            match self.isa {
                InstructionSet::Avx512VnniBf16 => Avx512VnniBf16Math::complex_mac_accumulate(
                    &self.h_fdl_re[..num_bins],
                    &self.h_fdl_im[..num_bins],
                    &self.fft_re[..num_bins],
                    &self.fft_im[..num_bins],
                    &mut self.acc_re[..num_bins],
                    &mut self.acc_im[..num_bins],
                ),
                InstructionSet::Avx512 => Avx512Math::complex_mac_accumulate(
                    &self.h_fdl_re[..num_bins],
                    &self.h_fdl_im[..num_bins],
                    &self.fft_re[..num_bins],
                    &self.fft_im[..num_bins],
                    &mut self.acc_re[..num_bins],
                    &mut self.acc_im[..num_bins],
                ),
                InstructionSet::Avx2 => Avx2Math::complex_mac_accumulate(
                    &self.h_fdl_re[..num_bins],
                    &self.h_fdl_im[..num_bins],
                    &self.fft_re[..num_bins],
                    &self.fft_im[..num_bins],
                    &mut self.acc_re[..num_bins],
                    &mut self.acc_im[..num_bins],
                ),
            }
        }

        // Partitions 1..K−1 (delays 2P..K×P): use past input spectra from FDL.
        for k in 1..num_partitions {
            let input_idx = (self.fdl_write_idx + num_partitions - k) % num_partitions;
            let fdl_start = input_idx * num_bins;
            let h_start = k * num_bins;

            unsafe {
                match self.isa {
                    InstructionSet::Avx512VnniBf16 => Avx512VnniBf16Math::complex_mac_accumulate(
                        &self.h_fdl_re[h_start..h_start + num_bins],
                        &self.h_fdl_im[h_start..h_start + num_bins],
                        &self.fdl_re[fdl_start..fdl_start + num_bins],
                        &self.fdl_im[fdl_start..fdl_start + num_bins],
                        &mut self.acc_re[..num_bins],
                        &mut self.acc_im[..num_bins],
                    ),
                    InstructionSet::Avx512 => Avx512Math::complex_mac_accumulate(
                        &self.h_fdl_re[h_start..h_start + num_bins],
                        &self.h_fdl_im[h_start..h_start + num_bins],
                        &self.fdl_re[fdl_start..fdl_start + num_bins],
                        &self.fdl_im[fdl_start..fdl_start + num_bins],
                        &mut self.acc_re[..num_bins],
                        &mut self.acc_im[..num_bins],
                    ),
                    InstructionSet::Avx2 => Avx2Math::complex_mac_accumulate(
                        &self.h_fdl_re[h_start..h_start + num_bins],
                        &self.h_fdl_im[h_start..h_start + num_bins],
                        &self.fdl_re[fdl_start..fdl_start + num_bins],
                        &self.fdl_im[fdl_start..fdl_start + num_bins],
                        &mut self.acc_re[..num_bins],
                        &mut self.acc_im[..num_bins],
                    ),
                }
            }
        }

        // ── Step 3: Store new input spectrum in FDL ──
        {
            let fdl_base = self.fdl_write_idx * num_bins;
            self.fdl_re[fdl_base..fdl_base + num_bins].copy_from_slice(&self.fft_re);
            self.fdl_im[fdl_base..fdl_base + num_bins].copy_from_slice(&self.fft_im);
        }

        // ── Step 4: Advance FDL write index ──
        self.fdl_write_idx += 1;
        if self.fdl_write_idx >= num_partitions {
            self.fdl_write_idx = 0;
        }

        // ── Step 5: Inverse RFFT (complex → real, 2P samples) ──
        self.rfft
            .process_inverse(&mut self.acc_re, &mut self.acc_im, &mut self.output_buf);

        // ── Step 6: Extract valid output (overlap-save: samples P..2P-1) ──
        self.tail_output_buf[..p].copy_from_slice(&self.output_buf[p..block_size]);
    }
}
