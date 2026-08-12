// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! True-peak (dBTP) — ITU-R BS.1770-4 Annex 2 — 4× oversampling FIR.
//!
//! RT-SAFETY DECISION:
//!   - RT hot-path (src/dsp/pipeline/stages/output.rs): keeps sample-peak
//!     detection for `RT_STATUS_HAS_CLIPPED`. True-peak with 48-tap FIR × 4×
//!     oversampling adds ~48 MAC/sample (12 per phase × 4) — prohibitive in the
//!     DSP callback where every μs matters.
//!   - Off-RT QA/telemetry: functions below expose full BS.1770-4 dBTP via
//!     `compute_true_peak_db()` and `find_true_peak_overs()`. The main-thread
//!     telemetry loop (src/standalone/rt_setup/telemetry.rs:81) can optionally
//!     run these on buffered audio for comprehensive inter-sample over detection.
//!   - Bench number: pending hardware-validation measurements.

/// Oversampling factor for BS.1770-4 true-peak measurement.
const TP_OVERSAMPLE: usize = 4;

/// Number of taps in the BS.1770-4 Annex 2 FIR filter (full, 48 taps).
const TP_FIR_LEN: usize = 48;

/// Number of taps per polyphase sub-filter (48 / 4 = 12).
const TP_TAPS: usize = TP_FIR_LEN / TP_OVERSAMPLE;

/// BS.1770-4 Annex 2 polyphase sub-filter coefficients (4 phases × 12 taps each).
///
/// These are the polyphase sub-filters H_p(z) for 4× oversampling, given directly
/// by ITU-R BS.1770-4 Annex 2 Table (p. 17). Each phase sums to ~1.0 (unity DC gain).
///
/// Phase ordering follows the standard: H_0 through H_3 are convolution filters
/// that produce outputs y\[4n\], y\[4n+1\], y\[4n+2\], y\[4n+3\] respectively.
///
/// Symmetry properties: phase 3 = reversed(phase 0), phase 2 = reversed(phase 1).
#[rustfmt::skip]
pub const BS1770_PHASES: [[f64; TP_TAPS]; TP_OVERSAMPLE] = [
    // Phase 0: produces y[4n+0]
    [
         0.0017089843750,  0.0109863281250, -0.0196533203125,  0.0332031250000,
        -0.0594482421875,  0.1373291015625,  0.9721679687500, -0.1022949218750,
         0.0476074218750, -0.0266113281250,  0.0148925781250, -0.0083007812500,
    ],
    // Phase 1: produces y[4n+1]
    [
        -0.0291748046875,  0.0292968750000, -0.0517578125000,  0.0891113281250,
        -0.1665039062500,  0.4650878906250,  0.7797851562500, -0.2003173828125,
         0.1015625000000, -0.0582275390625,  0.0330810546875, -0.0189208984375,
    ],
    // Phase 2: produces y[4n+2]
    [
        -0.0189208984375,  0.0330810546875, -0.0582275390625,  0.1015625000000,
        -0.2003173828125,  0.7797851562500,  0.4650878906250, -0.1665039062500,
         0.0891113281250, -0.0517578125000,  0.0292968750000, -0.0291748046875,
    ],
    // Phase 3: produces y[4n+3]
    [
        -0.0083007812500,  0.0148925781250, -0.0266113281250,  0.0476074218750,
        -0.1022949218750,  0.9721679687500,  0.1373291015625, -0.0594482421875,
         0.0332031250000, -0.0196533203125,  0.0109863281250,  0.0017089843750,
    ],
];

/// 4× oversampling via BS.1770-4 Annex 2 polyphase FIR.
///
/// For input sample `x[n]`, produces 4 output samples:
/// ```text
/// y[4n+p] = x[n]*h_p[0] + x[n-1]*h_p[1] + ... + x[n-11]*h_p[11]   (p = 0,1,2,3)
/// ```
/// Uses a sliding window over the input (off-RT — allocates).
fn oversample_4x_bs1770(samples: &[f32]) -> Vec<f64> {
    let in_len = samples.len();
    if in_len == 0 {
        return Vec::new();
    }
    let out_len = in_len * TP_OVERSAMPLE;
    let mut out = vec![0.0f64; out_len];

    for n in 0..in_len {
        let base = n * TP_OVERSAMPLE;
        for p in 0..TP_OVERSAMPLE {
            let phase = &BS1770_PHASES[p];
            let mut acc = 0.0f64;
            for k in 0..TP_TAPS {
                if k > n {
                    break;
                }
                acc += (samples[n - k] as f64) * phase[k];
            }
            out[base + p] = acc;
        }
    }
    out
}

/// Computes the true-peak level in dBTP per ITU-R BS.1770-4 Annex 2.
///
/// Applies 4× oversampling via the standard 48-tap FIR polyphase filter,
/// then measures the absolute peak of the upsampled signal.
///
/// Returns `f64::NEG_INFINITY` for an empty or all-zero input.
pub fn compute_true_peak_db(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }

    let upsampled = oversample_4x_bs1770(samples);
    let peak_abs = upsampled.iter().fold(0.0f64, |max, &x| max.max(x.abs()));

    if peak_abs <= 1e-15 {
        f64::NEG_INFINITY
    } else {
        20.0 * peak_abs.log10()
    }
}

/// A detected inter-sample over.
#[derive(Debug, Clone, PartialEq)]
pub struct TruePeakOver {
    /// Sample index in the original (un-upsampled) signal.
    pub position: usize,
    /// True-peak level in dBTP at this position.
    pub dbtp: f64,
}

/// Finds all inter-sample overs (> 0 dBFS) via BS.1770-4 Annex 2 4× oversampling.
///
/// Scans the 4× upsampled signal and reports each region where `|y[m]| > 1.0`.
/// Consecutive overs within the same original-sample window are merged into a
/// single event with the maximum dBTP of that window.
pub fn find_true_peak_overs(samples: &[f32]) -> Vec<TruePeakOver> {
    let upsampled = oversample_4x_bs1770(samples);
    let mut overs = Vec::new();
    let len = upsampled.len();
    let mut i = 0;

    while i < len {
        if upsampled[i].abs() > 1.0 {
            let start_sample = i / TP_OVERSAMPLE;
            let mut peak = upsampled[i].abs();
            i += 1;
            while i < len && i / TP_OVERSAMPLE == start_sample {
                if upsampled[i].abs() > 1.0 {
                    peak = peak.max(upsampled[i].abs());
                }
                i += 1;
            }
            overs.push(TruePeakOver {
                position: start_sample,
                dbtp: 20.0 * peak.log10(),
            });
        } else {
            i += 1;
        }
    }

    overs
}

/// Returns the full BS.1770-4 Annex 2 4× oversampled signal.
///
/// Output length = `samples.len() * 4`. Useful for detailed analysis and plotting.
pub fn oversample_4x(samples: &[f32]) -> Vec<f64> {
    oversample_4x_bs1770(samples)
}
