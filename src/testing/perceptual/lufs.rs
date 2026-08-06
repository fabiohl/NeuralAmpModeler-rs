// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! LUFS — ITU-R BS.1770-4 full 2-pass gating (integrated).

// =============================================================================
// LUFS — ITU-R BS.1770-4 full 2-pass gating (integrated)
// =============================================================================

/// K-weighting pre-filter coefficients (2nd-order high-pass, ~38 Hz).
/// H(z) = (1 - 2z⁻¹ + z⁻²) / (1 - 1.99004745483398z⁻¹ + 0.99007225036621z⁻²)
const KW_PRE_B: (f64, f64, f64) = (1.0, -2.0, 1.0);
const KW_PRE_A: (f64, f64) = (1.99004745483398, -0.99007225036621);

/// K-weighting RLB high-shelf coefficients (+4 dB above ~1-2 kHz).
/// H(z) = (1.53512485958697 - 2.69169618940638z⁻¹ + 1.19839281085285z⁻²)
///      / (1.0 - 1.69065929318241z⁻¹ + 0.73248077421585z⁻²)
const KW_SHELF_B: (f64, f64, f64) = (1.53512485958697, -2.69169618940638, 1.19839281085285);
const KW_SHELF_A: (f64, f64) = (1.69065929318241, -0.73248077421585);

/// LUFS block duration (400 ms per ITU-R BS.1770-4 §2.2).
const LUFS_BLOCK_MS: u64 = 400;

/// LUFS block overlap fraction (75 % = 3/4 per ITU-R BS.1770-4 §2.2).
const LUFS_OVERLAP_NUM: u64 = 3;
const LUFS_OVERLAP_DEN: u64 = 4;

/// Absolute gate threshold in LUFS (ITU-R BS.1770-4 §2.2 Table 1).
pub(crate) const LUFS_ABS_GATE: f64 = -70.0;

/// Relative gate threshold in LU (ITU-R BS.1770-4 §2.2).
const LUFS_REL_GATE: f64 = -10.0;

/// BS.1770-4 loudness constant (converts mean-square to LKFS/LUFS).
const LUFS_OFFSET: f64 = -0.691;

/// Applies a biquad IIR filter.
/// Direct Form II Transposed biquad filter.
///
/// Implements `H(z) = (b0 + b1*z⁻¹ + b2*z⁻²) / (1 - a1*z⁻¹ - a2*z⁻²)`.
fn apply_biquad(samples: &[f32], b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    let mut s1: f64 = 0.0;
    let mut s2: f64 = 0.0;
    for &x in samples {
        let xf = x as f64;
        let y = b0 * xf + s1;
        s1 = b1 * xf + a1 * y + s2;
        s2 = b2 * xf + a2 * y;
        out.push(y as f32);
    }
    out
}

/// Applies the full K-weighting filter chain (pre-filter + RLB high-shelf).
///
/// ITU-R BS.1770-4 §2.1: K-weighting shapes the spectrum to approximate
/// the frequency response of human hearing at typical listening levels.
pub(crate) fn apply_k_weighting(samples: &[f32]) -> Vec<f32> {
    let pre_filtered = apply_biquad(
        samples, KW_PRE_B.0, KW_PRE_B.1, KW_PRE_B.2, KW_PRE_A.0, KW_PRE_A.1,
    );
    apply_biquad(
        &pre_filtered,
        KW_SHELF_B.0,
        KW_SHELF_B.1,
        KW_SHELF_B.2,
        KW_SHELF_A.0,
        KW_SHELF_A.1,
    )
}

/// Computes loudness level in LKFS from mean-square power.
#[inline]
pub(crate) fn power_to_lkfs(power: f64) -> f64 {
    if power <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        LUFS_OFFSET + 10.0 * power.log10()
    }
}

/// Computes mean-square powers for overlapping blocks.
///
/// Returns `(block_powers, hop)` where `block_powers[i]` is the mean square
/// of the K-weighted signal over block `i`, and `hop` is the stride in samples.
fn compute_block_powers(
    k_weighted: &[f32],
    sample_rate: u32,
    block_ms: u64,
    overlap_num: u64,
    overlap_den: u64,
) -> (Vec<f64>, usize) {
    let block_samples = (sample_rate as usize * block_ms as usize) / 1000;
    let hop = block_samples * (overlap_den - overlap_num) as usize / overlap_den as usize;
    if block_samples > k_weighted.len() || hop == 0 {
        return (Vec::new(), hop);
    }
    let num_blocks = (k_weighted.len() - block_samples) / hop + 1;
    let mut powers = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let start = b * hop;
        let sum_sq: f64 = k_weighted[start..start + block_samples]
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum();
        powers.push(sum_sq / block_samples as f64);
    }
    (powers, hop)
}

/// Full ITU-R BS.1770-4 integrated loudness with 2-pass gating.
///
/// Algorithm:
/// 1. Apply K-weighting (pre-filter + RLB high-shelf)
/// 2. Divide into 400 ms blocks with 75 % overlap
/// 3. Compute mean-square power per block
/// 4. **Pass 1 — absolute gate:** discard blocks below −70 LUFS
/// 5. Compute ungated integrated loudness from surviving blocks
/// 6. **Pass 2 — relative gate:** discard blocks below (ungated − 10 LU)
/// 7. Integrated LUFS = loudness of finally surviving blocks
///
/// Returns `f64::NEG_INFINITY` if `samples` is empty or all blocks are gated out.
///
/// Validated against EBU reference vectors within ±0.1 LU (see tests).
pub fn compute_integrated_lufs(samples: &[f32], sample_rate: u32) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }
    let k_weighted = apply_k_weighting(samples);

    let (block_powers, _hop) = compute_block_powers(
        &k_weighted,
        sample_rate,
        LUFS_BLOCK_MS,
        LUFS_OVERLAP_NUM,
        LUFS_OVERLAP_DEN,
    );
    if block_powers.is_empty() {
        return f64::NEG_INFINITY;
    }

    let block_lkfs: Vec<f64> = block_powers.iter().map(|&p| power_to_lkfs(p)).collect();

    // Pass 1: absolute gate at -70 LUFS
    let gated_1: Vec<f64> = (0..block_powers.len())
        .filter(|&i| block_lkfs[i] > LUFS_ABS_GATE)
        .map(|i| block_powers[i])
        .collect();
    if gated_1.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mean_power_1: f64 = gated_1.iter().sum::<f64>() / gated_1.len() as f64;
    let ungated_lkfs = power_to_lkfs(mean_power_1);

    // Pass 2: relative gate at (ungated - 10 LU)
    let rel_threshold_lk = ungated_lkfs + LUFS_REL_GATE;

    let gated_2: Vec<f64> = (0..block_powers.len())
        .filter(|&i| block_lkfs[i] > LUFS_ABS_GATE && block_lkfs[i] > rel_threshold_lk)
        .map(|i| block_powers[i])
        .collect();
    if gated_2.is_empty() {
        return ungated_lkfs;
    }
    let mean_power_2: f64 = gated_2.iter().sum::<f64>() / gated_2.len() as f64;
    power_to_lkfs(mean_power_2)
}

/// Simplified compatibility wrapper — calls `compute_integrated_lufs`.
///
/// For diagnostic use in test reports. Prefer `compute_integrated_lufs` for
/// new code and `measure_loudness` for combined LUFS/LRA/dBTP reporting.
#[inline]
pub fn compute_lufs(samples: &[f32], sample_rate: u32) -> f64 {
    compute_integrated_lufs(samples, sample_rate)
}
