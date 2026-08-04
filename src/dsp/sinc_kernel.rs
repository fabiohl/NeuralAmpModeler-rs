// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Offline generation of minimum-phase Sinc FIR kernels for the native resampler.
//!
//! This module is invoked **exclusively** in `NamResampler::new()` — outside the RT thread.
//! All allocation and heavy computation (FFT, cepstrum) occurs here during initialization.
//!
//! ## Generation Pipeline
//!
//! 1. **Ideal Sinc + Kaiser Windowing (β=12)** — generates the linear-phase FIR lowpass prototype
//!    with cutoff at 0.95×min(f₁,f₂) / (f₁×NUM_PHASES), normalized to the prototype Nyquist.
//! 2. **Minimum-Phase Transform (Real Cepstrum)** — eliminates pre-ringing by concentrating
//!    energy into the shortest possible delay, using f64 FFT for numerical precision.
//!    Magnitude-preserving; measured ripple added < 1e-3 dB.
//! 3. **Polyphase Partition** — decomposes the prototype into 256 sub-filters,
//!    each with TAPS_PER_PHASE taps aligned to 64 bytes for AVX2/AVX-512 convolution.
//!
//! ## Linear-Phase Option
//!
//! `generate_polyphase_bank_linear()` skips the cepstrum transform, producing a
//! linear-phase filter bank suitable for offline/mixdown use where zero pre-ringing
//! is not critical and perfect phase linearity is preferred.
//!
//! ## Measured Performance (TAPS_PER_PHASE=64)
//!
//! Stopband attenuation is a *filter-design* property (alias-image rejection); the end-to-end
//! multitone SNR vs. soxr is *bank-dependent* and far lower for the production minimum-phase
//! bank (per-phase gain dispersion + ~0.06 dB cepstrum ripple). The two must not be conflated.
//!
//! | Rate Pair    | Passband Ripple | Stopband (design) | Multitone SNR (min-phase) |
//! |------------- |---------------- |------------------ |-------------------------- |
//! | 44.1→48 kHz  | < 0.05 dB       | ≥ 105 dB          | ~31 dB (gate ≥ 25 dB)     |
//! | 48→44.1 kHz  | < 0.02 dB       | ≥ 105 dB          | ~31 dB (gate ≥ 25 dB)     |
//! | 48→96 kHz    | < 0.02 dB       | ≥ 110 dB          | ~31 dB (gate ≥ 25 dB)     |
//! | 96→48 kHz    | < 0.02 dB       | ≥ 115 dB          | ~31 dB (gate ≥ 25 dB)     |
//!
//! The linear-phase variant (`new_linear()`, offline) scores significantly higher on multitone
//! SNR but reintroduces pre-ringing, so it is not the production default.

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;
use crate::math::dsp::fft::FftPlanner;

/// Phase type of the polyphase bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PhaseType {
    /// Linear-phase FIR: symmetric impulse, constant group delay = TAPS_PER_PHASE/2.
    Linear,
    /// Minimum-phase FIR via real cepstrum: asymmetric impulse, minimal group delay.
    Minimum,
}

/// Number of phases in the overabundant polyphase bank.
///
/// Controls the fractional resolution of the resampler. With 256 phases,
/// the maximum phase error between adjacent sub-filters is < 0.4%, so the
/// adjacent-phase linear interpolation contributes negligible error (its own
/// SNR contribution exceeds 140 dB) and is never the end-to-end bottleneck.
/// See `TAPS_PER_PHASE` and the module-level table for the *measured*
/// end-to-end multitone SNR (~31 dB, min-phase, dispersion-limited).
pub const NUM_PHASES: usize = 256;

/// Number of taps per phase in the polyphase bank.
///
/// With 64 taps per phase and a Kaiser window (β=12), the filter achieves
/// passband ripple < 0.01 dB up to 0.45×Nyquist and stopband attenuation
/// ≥ 110 dB measured across all rate pairs. The value is a multiple of 8
/// for AVX2/AVX-512 alignment (8 SIMD loads of 8 floats for AVX2, 4 of 16
/// for AVX-512).
///
/// Previously this was 32 taps (~24 dB end-to-end multitone SNR).
/// Doubling to 64 taps raised the stopband attenuation to ≥ 110 dB (filter
/// design) and the end-to-end multitone SNR vs. soxr to ~31 dB (min-phase,
/// per-phase-dispersion limited). The earlier ">120 dB" figure was an
/// aspiration for the *filter design* (stopband rejection), not the achieved
/// end-to-end SNR — see the module-level table and `resampler_test.rs`.
pub const TAPS_PER_PHASE: usize = 64;

/// Total prototype FIR length = NUM_PHASES × TAPS_PER_PHASE.
const PROTO_LEN: usize = NUM_PHASES * TAPS_PER_PHASE;

/// Polyphase filter bank with coefficients aligned for SIMD.
///
/// Memory layout: `coeffs[phase * TAPS_PER_PHASE .. (phase+1) * TAPS_PER_PHASE]`
/// Each phase is contiguous and aligned to 64 bytes for `_mm512_load_ps` / `_mm256_load_ps`.
pub struct PolyphaseBank {
    /// Aligned f32 coefficients. Size = `NUM_PHASES * TAPS_PER_PHASE`.
    /// The 64-byte alignment of the buffer start ensures that each 128-byte phase
    /// also begins on a 64-byte boundary.
    coeffs: AlignedVec<f32>,
    /// Taps per phase (always TAPS_PER_PHASE, already a multiple of 8).
    pub taps_per_phase: usize,
    /// Empirical group delay (centroid of energy) in output-rate samples.
    /// For linear-phase: always `TAPS_PER_PHASE / 2`.
    /// For minimum-phase: `centroid_of_prototype / NUM_PHASES`.
    pub group_delay: f64,
    /// Phase type of this bank (Linear or Minimum).
    pub phase_type: PhaseType,
}

impl PolyphaseBank {
    /// Returns the pointer to the start of phase `phase` coefficients.
    ///
    /// # Safety
    /// `phase` deve ser < `NUM_PHASES`.
    #[inline]
    pub fn phase_ptr(&self, phase: usize) -> *const f32 {
        debug_assert!(phase < NUM_PHASES);
        unsafe { self.coeffs.as_ptr().add(phase * self.taps_per_phase) }
    }

    /// Returns the coefficient slice for phase `phase`.
    #[inline]
    pub fn phase_coeffs(&self, phase: usize) -> &[f32] {
        debug_assert!(phase < NUM_PHASES);
        let start = phase * self.taps_per_phase;
        &self.coeffs[start..start + self.taps_per_phase]
    }
}

/// Generates the complete polyphase bank for conversion `from_rate → to_rate`.
///
/// Pipeline: Sinc+Kaiser → Minimum Phase (Cepstrum) → Polyphase Partition.
///
/// # Parameters
/// - `from_rate`: source sample rate (Hz).
/// - `to_rate`: destination sample rate (Hz).
///
/// # Returns
/// Polyphase bank ready for SIMD convolution.
pub fn generate_polyphase_bank(
    from_rate: u32,
    to_rate: u32,
) -> Result<PolyphaseBank, NamErrorCode> {
    // 1. Generate Sinc + Kaiser prototype in f64.
    //
    // The prototype operates at the conceptual proto rate = from_rate × NUM_PHASES.
    // Its passband must span only the baseband of the signal being resampled:
    //   - interpolation (from_rate < to_rate): passband = from_rate / 2  (input Nyquist)
    //   - decimation   (from_rate > to_rate): passband = to_rate   / 2  (output Nyquist)
    // The common expression is min_rate / 2, normalized to the proto Nyquist.
    let passband_rate = from_rate.min(to_rate) as f64;
    let proto_nyq = from_rate as f64 * NUM_PHASES as f64 / 2.0;
    let cutoff = 0.95 * passband_rate / 2.0 / proto_nyq;
    let proto_f64 = generate_sinc_kaiser(PROTO_LEN, cutoff, 12.0);

    // 2. Transform to minimum phase via real cepstrum
    let min_phase = to_minimum_phase(&proto_f64);

    // 2a. Compute empirical group delay (energy centroid) of the minimum-phase kernel
    let centroid = calculate_centroid(&min_phase);
    let group_delay = centroid / NUM_PHASES as f64;

    // 3. Normalize energy (DC gain = 1.0 per phase)
    let proto_f32: Vec<f32> = min_phase.iter().map(|&x| x as f32).collect();

    // 4. Partition into NUM_PHASES sub-filters
    let mut bank = partition_polyphase(&proto_f32)?;
    bank.group_delay = group_delay;
    bank.phase_type = PhaseType::Minimum;
    Ok(bank)
}

/// Generates a Sinc FIR kernel with Kaiser windowing.
///
/// # Parameters
/// - `length`: total filter length (samples).
/// - `cutoff`: normalized cutoff frequency (0..1, relative to Nyquist).
/// - `beta`: Kaiser window β parameter (controls stop-band attenuation).
///   β=12 → ~120 dB of rejection.
fn generate_sinc_kaiser(length: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let half = (length - 1) as f64 / 2.0;
    let i0_beta = bessel_i0(beta);

    let mut kernel = Vec::with_capacity(length);
    for i in 0..length {
        let n = i as f64 - half;

        // Normalized Sinc
        let sinc = if n.abs() < 1e-10 {
            cutoff
        } else {
            let x = std::f64::consts::PI * n * cutoff;
            x.sin() / (std::f64::consts::PI * n)
        };

        // Kaiser window: I0(β × sqrt(1 - (2n/N-1)²)) / I0(β)
        let ratio = n / half;
        let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
        let window = bessel_i0(arg) / i0_beta;

        kernel.push(sinc * window);
    }

    // Normalize for unit DC gain
    let dc_sum: f64 = kernel.iter().sum();
    if dc_sum.abs() > 1e-15 {
        for k in &mut kernel {
            *k /= dc_sum;
        }
    }

    kernel
}

/// Modified Bessel function of the first kind, order zero — I₀(x).
///
/// Taylor series expansion with 20 terms (precision > 1e-12 for β ≤ 25).
fn bessel_i0(x: f64) -> f64 {
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

/// Computes the energy-weighted centroid (center of mass) of an impulse response.
///
/// Returns the average time index (in samples) where the energy is concentrated.
/// For minimum-phase filters, this centroid is significantly smaller than
/// `len/2` (the linear-phase theoretical group delay).
fn calculate_centroid(h: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (n, &val) in h.iter().enumerate() {
        let energy = val * val;
        num += n as f64 * energy;
        den += energy;
    }
    if den > 1e-30 { num / den } else { 0.0 }
}

/// Transforms a linear-phase FIR kernel to minimum phase via Real Cepstrum.
///
/// ## Algorithm (Oppenheim & Schafer, Discrete-Time Signal Processing)
///
/// 1. Zero-pad kernel to `N_fft` (power of 2, ≥ 4× original length).
/// 2. FFT → complex spectrum `H(k)`.
/// 3. Log-magnitude: `L(k) = ln(|H(k)| + ε)`.
/// 4. IFFT of `L` → real cepstrum `c[n]`.
/// 5. Causal truncation: `c[0]` unchanged, `c[1..N/2-1] × 2`, `c[N/2+1..] = 0`.
/// 6. FFT of causal cepstrum → `Ĉ(k)`.
/// 7. Complex exponential: `H_min(k) = exp(Ĉ(k))`.
/// 8. IFFT → `h_min[n]` (real part), truncate to original length.
///
/// All computation is in f64 for numerical stability in the logarithmic domain,
/// as recommended by r8brain-free-src (Vaneev).
fn to_minimum_phase(kernel: &[f64]) -> Vec<f64> {
    let n_proto = kernel.len();
    let n_fft = (4 * n_proto).next_power_of_two();

    let planner = FftPlanner::<f64>::new(n_fft);

    // Step 1-2: Zero-pad kernel into SoA buffers
    let mut buf_re = vec![0.0_f64; n_fft];
    let mut buf_im = vec![0.0_f64; n_fft];
    buf_re[..n_proto].copy_from_slice(kernel);
    planner.process(&mut buf_re, &mut buf_im);

    // Step 3: Log-magnitude (real-only complex)
    let eps = 1e-10_f64;
    for i in 0..n_fft {
        buf_re[i] = (buf_re[i].hypot(buf_im[i]) + eps).ln();
        buf_im[i] = 0.0;
    }

    // Step 4: IFFT -> real cepstrum (process_inverse already scales by 1/n)
    planner.process_inverse(&mut buf_re, &mut buf_im);

    // Step 5: Causal truncation
    // c[0] unchanged, c[1..N/2-1] × 2, c[N/2] unchanged, c[N/2+1..] = 0
    let half = n_fft / 2;
    for item in buf_re.iter_mut().take(half).skip(1) {
        *item *= 2.0;
    }
    // c[half] remains unchanged
    for item in buf_re.iter_mut().skip(half + 1) {
        *item = 0.0;
    }
    // Cepstrum is purely real — zero entire imaginary part.
    // process_inverse may leave numerical noise in buf_im.
    buf_im.fill(0.0);

    // Step 6: FFT of causal cepstrum
    planner.process(&mut buf_re, &mut buf_im);

    // Step 7: Complex exponential
    for i in 0..n_fft {
        let exp_a = buf_re[i].exp();
        let (sin_b, cos_b) = buf_im[i].sin_cos();
        buf_re[i] = exp_a * cos_b;
        buf_im[i] = exp_a * sin_b;
    }

    // Step 8: IFFT -> minimum-phase impulse (process_inverse already scales by 1/n)
    planner.process_inverse(&mut buf_re, &mut buf_im);

    buf_re.truncate(n_proto);
    buf_re
}

/// Partitions the FIR prototype into `NUM_PHASES` polyphase sub-filters.
///
/// Coefficient `proto[n]` goes to phase `n % NUM_PHASES`, tap `n / NUM_PHASES`.
/// Each phase is zero-padded to `TAPS_PER_PHASE` (multiple of 8).
fn partition_polyphase(proto: &[f32]) -> Result<PolyphaseBank, NamErrorCode> {
    let taps = TAPS_PER_PHASE;
    let total = NUM_PHASES * taps;
    let mut coeffs = AlignedVec::new(total, 0.0f32)?;

    // Scale by NUM_PHASES to compensate for the polyphase decomposition.
    // In conceptual upsampling (insertion of L-1 zeros between samples),
    // the prototype filter is applied at L×fs rate. The polyphase partition
    // divides the total gain by L, requiring gain compensation.
    let gain = NUM_PHASES as f32;

    for (n, &coeff) in proto.iter().enumerate() {
        let phase = n % NUM_PHASES;
        let tap = n / NUM_PHASES;
        if tap < taps {
            coeffs[phase * taps + (taps - 1 - tap)] = coeff * gain;
        }
    }

    // Normalize each phase sub-filter individually to ensure flat DC gain.
    // For min-phase kernels, phase DC gains can vary significantly because
    // energy is concentrated at the impulse start. Per-phase normalization
    // is essential despite introducing small ripple (~0.01-0.1 dB) because
    // the alternative (no normalization) creates massive gain variation
    // (>40 dB) across phases, which linear interpolation cannot smooth.
    // See QA report for detailed analysis.
    for phase in 0..NUM_PHASES {
        let start = phase * taps;
        let mut sum = 0.0f32;
        for tap in 0..taps {
            sum += coeffs[start + tap];
        }
        if sum.abs() > 1e-9 {
            for tap in 0..taps {
                coeffs[start + tap] /= sum;
            }
        }
    }

    Ok(PolyphaseBank {
        coeffs,
        taps_per_phase: taps,
        group_delay: taps as f64 / 2.0,
        phase_type: PhaseType::Linear,
    })
}

/// Generates a linear-phase polyphase bank (without cepstrum transform).
///
/// Produces traditional linear-phase FIR filters with symmetric coefficients,
/// suitable for offline/mixdown use where pre-ringing is acceptable and
/// exact phase linearity is preferred.
pub fn generate_polyphase_bank_linear(
    from_rate: u32,
    to_rate: u32,
) -> Result<PolyphaseBank, NamErrorCode> {
    let passband_rate = from_rate.min(to_rate) as f64;
    let proto_nyq = from_rate as f64 * NUM_PHASES as f64 / 2.0;
    let cutoff = 0.95 * passband_rate / 2.0 / proto_nyq;
    let proto_f64 = generate_sinc_kaiser(PROTO_LEN, cutoff, 12.0);
    let proto_f32: Vec<f32> = proto_f64.iter().map(|&x| x as f32).collect();
    partition_polyphase(&proto_f32)
}

/// Compares magnitude responses of linear-phase vs minimum-phase kernels.
///
/// Returns (max_ripple_db, rms_ripple_db) where ripple is the per-bin
/// magnitude difference introduced by the cepstrum minimum-phase transform.
/// Validates that the cepstrum is magnitude-preserving as expected.
///
/// # Parameters
/// - `from_rate`, `to_rate`: rate pair for kernel generation.
///
/// # Panics
/// If FFT planner creation fails.
#[doc(hidden)]
pub fn measure_cepstrum_ripple(from_rate: u32, to_rate: u32) -> (f64, f64) {
    let passband_rate = from_rate.min(to_rate) as f64;
    let proto_nyq = from_rate as f64 * NUM_PHASES as f64 / 2.0;
    let cutoff = 0.95 * passband_rate / 2.0 / proto_nyq;
    let proto_f64 = generate_sinc_kaiser(PROTO_LEN, cutoff, 12.0);
    let min_phase = to_minimum_phase(&proto_f64);

    let n_fft = (4usize * PROTO_LEN.max(min_phase.len())).next_power_of_two();
    let planner = FftPlanner::<f64>::new(n_fft);

    let mut lp_re = vec![0.0f64; n_fft];
    let mut lp_im = vec![0.0f64; n_fft];
    lp_re[..proto_f64.len()].copy_from_slice(&proto_f64);
    planner.process(&mut lp_re, &mut lp_im);

    let mut mp_re = vec![0.0f64; n_fft];
    let mut mp_im = vec![0.0f64; n_fft];
    mp_re[..min_phase.len()].copy_from_slice(&min_phase);
    planner.process(&mut mp_re, &mut mp_im);

    let mut max_db = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut passband_count = 0usize;

    let passband_bins = ((cutoff * 0.98) * (n_fft as f64 / 2.0)).ceil() as usize;
    let nyq_idx = passband_bins.min(n_fft / 2);

    let floor_db: f64 = -60.0;

    for i in 0..=nyq_idx {
        let lp_mag = lp_re[i].hypot(lp_im[i]).max(1e-30);
        let mp_mag = mp_re[i].hypot(mp_im[i]).max(1e-30);
        let lp_db = 20.0 * lp_mag.log10();
        if lp_db < floor_db {
            continue;
        }
        let diff_db = (lp_db - 20.0 * mp_mag.log10()).abs();
        max_db = max_db.max(diff_db);
        sum_sq += diff_db * diff_db;
        passband_count += 1;
    }
    let rms_db = (sum_sq / passband_count.max(1) as f64).sqrt();
    (max_db, rms_db)
}

#[cfg(test)]
#[path = "sinc_kernel_test.rs"]
mod sinc_kernel_test;
