// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Uniform-Partitioned Overlap-Save (UPOLS) convolution engine.
//!
//! Implements real-time convolution of an audio stream with an impulse response
//! using the Uniform-Partitioned Overlap-Save method in the frequency domain.
//!
//! ## Design
//!
//! - **Partition size** equals the audio block size (typically 64–2048 samples).
//!   Latency is exactly `partition_size` samples.
//! - **FFT size** is `2 × partition_size` (rounded up to next power of two).
//! - **Kernel pre-FFT**: all IR partitions are transformed to the frequency domain
//!   at construction time via native `RfftPlanner`, outside the audio thread.
//! - **FDL (Frequency Delay Line)** is pre-allocated as contiguous SoA buffers
//!   of real/imaginary spectra (size `fft_size/2 + 1` bins per partition).
//! - **Zero-alloc hot-path**: `process()` only mutates pre-allocated buffers.
//!   It never allocates, never blocks, and never panics.
//!
//! ## Reference
//!
//! Gardner, W. G. "Efficient Convolution without Input-Output Delay"
//! JAES Vol. 43, No. 3, 1995 March.

use crate::common::diagnostics::NamErrorCode;
use crate::common::spsc::{RT_STATUS_CABSIM_CONTRACT_VIOLATION, RtStatusFlags};
use crate::common::unlikely;
use crate::math::common::AlignedVec;
use crate::math::common::Avx2Math;
use crate::math::common::traits::SimdMath;
use crate::math::dsp::fft::RfftPlanner;
use log::info;

/// Uniform-Partitioned Overlap-Save convolution engine.
///
/// All memory is allocated at construction time (`ConvEngine::new()`).
/// The [`process`](ConvEngine::process) method is zero-alloc and safe for real-time
/// audio threads.
///
/// ## Latency
///
/// UPOLS introduces exactly `partition_size` samples of latency
/// (one full audio block).
pub struct ConvEngine {
    /// FFT size (2 × partition_size rounded up to next power of two).
    fft_size: usize,
    /// Number of frequency bins per partition (= fft_size / 2 + 1).
    n_bins: usize,
    /// Number of samples per input/output block.
    partition_size: usize,
    /// Number of IR partitions.
    num_partitions: usize,
    /// Pre-FFT'd kernel partitions (real part).
    /// Flat storage: `num_partitions × n_bins` f32 values.
    h_fdl_re: AlignedVec<f32>,
    /// Pre-FFT'd kernel partitions (imaginary part).
    h_fdl_im: AlignedVec<f32>,
    /// Frequency Delay Line (FDL): circular buffer of input spectra (real part).
    /// Flat storage: `num_partitions × n_bins` f32 values.
    fdl_re: AlignedVec<f32>,
    /// FDL imaginary part.
    fdl_im: AlignedVec<f32>,
    /// Write index into the FDL circular buffer.
    fdl_idx: usize,
    /// Input overlap buffer for overlap-save (length = `fft_size`).
    /// Layout: the most recent `partition_size` samples are loaded at
    /// offset `fft_size - partition_size`. After FFT and IFFT,
    /// the valid output starts at offset `fft_size - partition_size`.
    input_buf: AlignedVec<f32>,
    /// Native RFFT planner (handles both forward RFFT and inverse IRFFT).
    rfft: RfftPlanner<f32>,
    /// Accumulation buffer in frequency domain, real part (length = `n_bins`).
    acc_re: AlignedVec<f32>,
    /// Accumulation buffer imaginary part (length = `n_bins`).
    acc_im: AlignedVec<f32>,
    /// Time-domain output buffer for IRFFT (length = `fft_size`).
    output_buf: AlignedVec<f32>,
    /// Cached output start index (= fft_size - partition_size).
    output_start: usize,
}

impl ConvEngine {
    /// Builds a UPOLS convolution engine for the given impulse response.
    ///
    /// The IR is partitioned into blocks of `partition_size` samples.
    /// All RFFTs of the kernel partitions are computed here — outside the
    /// audio thread — so that [`process`](ConvEngine::process) is zero-alloc.
    ///
    /// # Parameters
    /// - `ir`: impulse response samples (mono, f32).
    /// - `partition_size`: size of each partition / audio block size.
    ///
    /// # Returns
    /// A fully initialized `ConvEngine`. If `ir` is empty, the engine
    /// acts as a passthrough (output = input).
    pub fn new(ir: &[f32], partition_size: usize) -> Result<Self, NamErrorCode> {
        assert!(partition_size > 0, "partition_size must be positive");

        let fft_size = (2 * partition_size).next_power_of_two();
        let n_bins = fft_size / 2 + 1;
        let output_start = fft_size - partition_size;

        // Partition IR: P = ceil(N / B)
        let num_partitions = if ir.is_empty() {
            0
        } else {
            ir.len().div_ceil(partition_size)
        };

        // Build native RFFT plan (handles both forward RFFT and inverse IRFFT)
        let mut rfft = RfftPlanner::<f32>::new(fft_size);

        // Pre-FFT each kernel partition
        let h_fdl_part_len = num_partitions * n_bins;
        let mut h_fdl_re = AlignedVec::new(h_fdl_part_len, 0.0_f32)?;
        let mut h_fdl_im = AlignedVec::new(h_fdl_part_len, 0.0_f32)?;

        let mut ir_buf = vec![0.0f32; fft_size];
        let mut tmp_re = vec![0.0f32; n_bins];
        let mut tmp_im = vec![0.0f32; n_bins];

        for p in 0..num_partitions {
            let ir_start = p * partition_size;
            let ir_end = (ir_start + partition_size).min(ir.len());

            ir_buf.fill(0.0);
            for (i, &sample) in ir[ir_start..ir_end].iter().enumerate() {
                ir_buf[i] = sample;
            }

            rfft.process_forward(&ir_buf, &mut tmp_re, &mut tmp_im);

            let base = p * n_bins;
            for k in 0..n_bins {
                h_fdl_re[base + k] = tmp_re[k];
                h_fdl_im[base + k] = tmp_im[k];
            }
        }

        // Pre-allocate FDL (all zeros initially)
        let fdl_part_len = num_partitions * n_bins;
        let fdl_re = AlignedVec::new(fdl_part_len, 0.0_f32)?;
        let fdl_im = AlignedVec::new(fdl_part_len, 0.0_f32)?;

        // Pre-allocate runtime buffers
        let input_buf = AlignedVec::new(fft_size, 0.0_f32)?;
        let acc_re = AlignedVec::new(n_bins, 0.0_f32)?;
        let acc_im = AlignedVec::new(n_bins, 0.0_f32)?;
        let output_buf = AlignedVec::new(fft_size, 0.0_f32)?;

        if num_partitions == 0 {
            info!(
                "[Conv] Engine built: passthrough (empty IR), partition={}, fft={}",
                partition_size, fft_size
            );
        } else {
            info!(
                "[Conv] Engine built: {} IR samples, partition={}, fft={}, {} partitions",
                ir.len(),
                partition_size,
                fft_size,
                num_partitions
            );
        }

        Ok(Self {
            fft_size,
            n_bins,
            partition_size,
            num_partitions,
            h_fdl_re,
            h_fdl_im,
            fdl_re,
            fdl_im,
            fdl_idx: 0,
            input_buf,
            rfft,
            acc_re,
            acc_im,
            output_buf,
            output_start,
        })
    }

    /// Returns the partition size (== audio block size) in samples.
    #[inline(always)]
    pub fn partition_size(&self) -> usize {
        self.partition_size
    }

    /// Returns the FFT size used for frequency-domain processing.
    #[inline(always)]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Returns the number of IR partitions.
    #[inline(always)]
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Returns the algorithmic latency in samples (= `partition_size`).
    #[inline(always)]
    pub fn latency_samples(&self) -> usize {
        self.partition_size
    }

    /// Returns `true` if no IR is loaded (passthrough mode).
    #[inline(always)]
    pub fn is_passthrough(&self) -> bool {
        self.num_partitions == 0
    }

    /// Resets the engine to its post-construction state: zeroes the frequency
    /// delay line, the overlap-save input buffer and every scratch buffer, and
    /// rewinds the FDL write index.
    ///
    /// In-place and zero-alloc (RT-safe). Preserves the pre-FFT'd kernel
    /// partitions (`h_fdl_re`/`h_fdl_im`) and the RFFT plan; a subsequent
    /// [`process`](ConvEngine::process) on a delta impulse is bit-identical to
    /// a freshly constructed engine with the same IR — no tail from previous
    /// audio survives the reset.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.fdl_re.fill(0.0);
        self.fdl_im.fill(0.0);
        self.fdl_idx = 0;
        self.input_buf.fill(0.0);
        self.acc_re.fill(0.0);
        self.acc_im.fill(0.0);
        self.output_buf.fill(0.0);
    }

    /// Processes one block of mono audio through the convolution engine.
    ///
    /// ## RT-Safety
    ///
    /// This function is **zero-alloc**, **lock-free**, and never panics.
    /// It only mutates pre-allocated internal buffers.
    ///
    /// ## Parameters
    /// - `input`: slice of at least `partition_size` samples. Only the
    ///   first `partition_size` samples are read.
    /// - `output`: slice of at least `partition_size` samples where the
    ///   convolved result is written.
    /// - `rt_status`: optional lock-free status flags. On a contract
    ///   violation (a slice shorter than `partition_size`), the engine
    ///   raises [`RT_STATUS_CABSIM_CONTRACT_VIOLATION`].
    ///
    /// ## Buffer Contract (release-safe)
    ///
    /// UPOLS transforms whole blocks, so a partial block cannot be
    /// convolved. If `input.len() < partition_size` or
    /// `output.len() < partition_size`, this function **zeros the entire
    /// output**, raises the contract-violation flag (when `rt_status` is
    /// provided), skips the transform, and returns — it never reads or
    /// writes beyond the caller's slices and never panics. Copies are
    /// limited strictly to `min(input.len(), output.len(), partition_size)`.
    #[inline]
    pub fn process(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        rt_status: Option<&RtStatusFlags>,
    ) {
        let n = input.len().min(output.len()).min(self.partition_size);
        if unlikely(n < self.partition_size) {
            output.fill(0.0);
            if let Some(rt) = rt_status {
                rt.set_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION);
            }
            return;
        }

        if self.num_partitions == 0 {
            // SAFETY: the buffer contract guard above guarantees `input`
            // and `output` each have at least `self.partition_size`
            // elements. The regions are distinct (output is a
            // caller-provided mutable buffer, input is caller-provided
            // immutable data).
            unsafe {
                core::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    self.partition_size,
                );
            }
            return;
        }

        let in_len = self.fft_size;
        let out_start = self.output_start;
        // SAFETY: Overlap-save shift-left: copies the trailing
        // `fft_size - partition_size` samples from the old input
        // block (starting at offset `partition_size`) to the front
        // of the same buffer. Both source and destination are within
        // `self.input_buf` (length `fft_size`), so `add` and
        // `copy` are in-bounds. Regions may overlap (source >
        // destination) but `copy` handles that correctly.
        unsafe {
            core::ptr::copy(
                self.input_buf.as_ptr().add(self.partition_size),
                self.input_buf.as_mut_ptr(),
                in_len - self.partition_size,
            );
        }

        // SAFETY: the buffer contract guard at the top guarantees `input`
        // has at least `self.partition_size` samples. `out_start +
        // partition_size == fft_size`, so the destination range is within
        // `input_buf`. Source and destination do not overlap (input is
        // caller data).
        unsafe {
            core::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input_buf.as_mut_ptr().add(out_start),
                self.partition_size,
            );
        }

        // ── Step 2: Forward RFFT of input segment, written directly into the
        //            FDL slot (P-05 / T5.2) ──
        // The FDL write index advances after the MAC step, so the slot
        // `[fdl_base, fdl_base + n_bins)` is the destination of this block's
        // spectrum: writing the RFFT output straight into it eliminates the
        // intermediate `fft_buf_re/im` copy from the hot path.
        let fdl_base = self.fdl_idx * self.n_bins;
        self.rfft.process_forward(
            &self.input_buf,
            &mut self.fdl_re[fdl_base..fdl_base + self.n_bins],
            &mut self.fdl_im[fdl_base..fdl_base + self.n_bins],
        );

        // ── Step 3: Frequency-domain MAC over all partitions ──
        let p_count = self.num_partitions;
        let n_bins = self.n_bins;

        if p_count == 1 {
            let fdl_start = self.fdl_idx * self.n_bins;
            // SAFETY: all slices have length n_bins, guaranteed by construction.
            // Baseline x86-64-v3 (Avx2Math) used directly without runtime branching.
            unsafe {
                Avx2Math::complex_mac_overwrite(
                    &self.h_fdl_re[..n_bins],
                    &self.h_fdl_im[..n_bins],
                    &self.fdl_re[fdl_start..fdl_start + n_bins],
                    &self.fdl_im[fdl_start..fdl_start + n_bins],
                    &mut self.acc_re[..n_bins],
                    &mut self.acc_im[..n_bins],
                );
            }
        } else {
            self.acc_re[..n_bins].fill(0.0);
            self.acc_im[..n_bins].fill(0.0);

            for p in 0..p_count {
                let fdl_p = (self.fdl_idx + p_count - p) % p_count;
                let fdl_start = fdl_p * self.n_bins;
                let h_start = p * self.n_bins;

                // SAFETY: all slices have length n_bins, guaranteed by construction.
                unsafe {
                    Avx2Math::complex_mac_accumulate(
                        &self.h_fdl_re[h_start..h_start + n_bins],
                        &self.h_fdl_im[h_start..h_start + n_bins],
                        &self.fdl_re[fdl_start..fdl_start + n_bins],
                        &self.fdl_im[fdl_start..fdl_start + n_bins],
                        &mut self.acc_re[..n_bins],
                        &mut self.acc_im[..n_bins],
                    );
                }
            }
        }

        // ── Step 4: Inverse RFFT (complex → real) ──
        // process_inverse takes in_re/in_im of length N/2+1 (n_bins) and
        // produces real output of length N (fft_size). The inverse scaling
        // is handled internally by the IRFFT algorithm.
        self.rfft
            .process_inverse(&mut self.acc_re, &mut self.acc_im, &mut self.output_buf);

        // ── Step 5: Extract valid output (overlap-save discard) ──
        // SAFETY: the buffer contract guard at the top guarantees `output`
        // has at least `self.partition_size` elements. `out_start +
        // partition_size == fft_size`, so the source range is within
        // `output_buf`. Source and destination do not overlap (output is a
        // caller-provided mutable buffer, output_buf is internal).
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.output_buf.as_ptr().add(out_start),
                output.as_mut_ptr(),
                self.partition_size,
            );
        }

        // ── Step 6: Advance FDL write index ──
        self.fdl_idx += 1;
        if self.fdl_idx >= p_count {
            self.fdl_idx = 0;
        }
    }
}

#[cfg(test)]
#[path = "conv_test.rs"]
mod conv_test;
