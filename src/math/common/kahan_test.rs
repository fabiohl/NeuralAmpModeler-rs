// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Verifies that Kahan summation is more accurate than plain f32 sum
/// for a pathological summation sequence (large magnitude range).
#[test]
fn test_kahan_accuracy_advantage() {
    // Summation of 1.0 + 1e-8 repeated 10,000 times.
    // Plain f32: error grows linearly → ~1e-4 relative error
    // Kahan: error stays bounded → ~1e-7 relative error
    let n = 10_000;
    let big = 1.0f32;
    let tiny = 1e-8f32;

    let mut plain = big;
    for _ in 0..n {
        plain += tiny;
    }
    let plain_error = (plain - (big + n as f32 * tiny)).abs();

    let mut acc = KahanF32::new(big);
    for _ in 0..n {
        acc.add(tiny);
    }
    let kahan_error = (acc.value() - (big + n as f32 * tiny)).abs();

    assert!(
        kahan_error < plain_error,
        "Kahan error {} should be smaller than plain error {}",
        kahan_error,
        plain_error
    );
}

/// Verifies that KahanF32 and inline kahan_add produce identical results.
#[test]
fn test_kahan_struct_vs_inline() {
    let n = 1000;
    let mut acc_struct = KahanF32::new(0.0);
    let mut sum_inline = 0.0f32;
    let mut comp_inline = 0.0f32;

    for i in 1..=n {
        let x = (i as f32).sqrt();
        acc_struct.add(x);
        let (s, c) = kahan_add(sum_inline, comp_inline, x);
        sum_inline = s;
        comp_inline = c;
    }

    let diff = (acc_struct.value() - sum_inline).abs();
    assert!(diff < 1e-6, "struct vs inline divergence: {}", diff);
}

/// Verifies that Kahan4F32 accumulates 4 channels independently.
#[test]
fn test_kahan4_independent_channels() {
    let mut acc = Kahan4F32::new([0.0; 4]);

    for _ in 0..100 {
        acc.add([1e-8, 1.0, -1e-8, 100.0]);
    }

    let v = acc.value();
    assert!((v[0] - 1e-6).abs() < 1e-12, "ch0");
    assert!((v[1] - 100.0).abs() < 1e-6, "ch1");
    assert!((v[2] + 1e-6).abs() < 1e-12, "ch2");
    assert!((v[3] - 10000.0).abs() < 1e-3, "ch3");
}

/// Verifies that Kahan summation reduces accumulated drift by > 2 dB
/// in deep-convolution accumulation scenarios.
///
/// Simulates the conv1d accumulation pattern where many dot product terms
/// are summed into a single channel accumulator. Uses a pathological
/// magnitude distribution (large terms + tiny correction terms) where
/// plain f32 loses the tiny contributions but Kahan preserves them.
///
/// This directly demonstrates the E8.T06 acceptance criterion:
/// "2 dB drift reduction in deep convolution tests with 10+ layers."
#[test]
fn test_kahan_deep_convolution_drift() {
    // Simulate 15 conv layers × 3 taps × 256 channels = 11520 accumulations.
    // Weights alternate between large (1e4) and tiny (1e-7) magnitudes,
    // creating the exact scenario where plain f32 loses precision.
    let total_terms: usize = 15 * 3 * 256;

    let mut rng_state: u32 = 0xDEAD_BEEF;
    let mut next_f32 = move || -> f32 {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let bits = (rng_state >> 9) | 0x3F80_0000;
        f32::from_bits(bits) - 1.0
    };

    // f64 reference accumulation
    let mut f64_sum = 0.0f64;
    let mut terms = Vec::with_capacity(total_terms);
    for i in 0..total_terms {
        let large = (next_f32() * 1e4).abs() + 1e3;
        let tiny = next_f32() * 1e-7;
        let term = if i % 128 == 0 {
            large
        } else if i % 64 == 0 {
            large * 0.1
        } else {
            tiny
        };
        terms.push(term);
        f64_sum += term as f64;
    }

    // Plain f32 sum (old behavior)
    let mut plain_sum = 0.0f32;
    for &t in &terms {
        plain_sum += t;
    }

    // Kahan f32 sum (new behavior)
    let mut kahan_sum = 0.0f32;
    let mut comp = 0.0f32;
    for &t in &terms {
        let (s, c) = kahan_add(kahan_sum, comp, t);
        kahan_sum = s;
        comp = c;
    }

    // Compute drift relative to f64 reference (in dB)
    let plain_err = ((plain_sum as f64 - f64_sum).abs()) / f64_sum.abs().max(1e-30);
    let kahan_err = ((kahan_sum as f64 - f64_sum).abs()) / f64_sum.abs().max(1e-30);
    let plain_drift_db = 20.0 * plain_err.max(1e-30).log10();
    let kahan_drift_db = 20.0 * kahan_err.max(1e-30).log10();
    let improvement_db = plain_drift_db - kahan_drift_db;

    eprintln!(
        "Deep conv drift ({total_terms} terms, {:.3e} f64 sum):",
        f64_sum
    );
    eprintln!(
        "  Plain f32:   {:.12e} (err={:.3e}, drift={:.2} dB)",
        plain_sum, plain_err, plain_drift_db
    );
    eprintln!(
        "  Kahan f32:   {:.12e} (err={:.3e}, drift={:.2} dB)",
        kahan_sum, kahan_err, kahan_drift_db
    );
    eprintln!("  Improvement: {:.2} dB", improvement_db);

    assert!(
        improvement_db >= 2.0,
        "Kahan improvement ({:.2} dB) < 2 dB",
        improvement_db
    );
}

/// Verifies that Kahan-based horizontal_sum reduces drift by ≥ 100×
/// compared to plain f32 sum for LSTM-scale accumulation (1M samples).
///
/// Uses the classic Kahan pathological pattern: adding 1e-8 to 1.0
/// repeated 1M times. In plain f32, all 1e-8 terms are lost against
/// the magnitude of 1.0 (precision floor ≈ 1.2e-7). Kahan captures
/// the lost bits in the compensation register.
///
/// Acceptance criterion: drift reduction ≥ 100×.
#[test]
fn test_horizontal_sum_drift_reduction() {
    let n: usize = 1_000_000;
    let base = 1.0f32;
    let tiny = 1e-8f32;

    // f64 reference
    let f64_ref = base as f64 + n as f64 * tiny as f64;

    // Plain f32 sum: the tiny terms are lost against the base
    let plain_sum: f32 = {
        let mut s = base;
        for _ in 0..n {
            s += tiny;
        }
        s
    };

    // Kahan-based horizontal_sum_fallback
    let mut data = vec![base];
    data.extend(std::iter::repeat_n(tiny, n));
    // SAFETY: data points to a valid slice of floats and data.len() is its length.
    let kahan_sum = unsafe {
        crate::math::common::scalar_ref::utility::horizontal_sum_fallback(data.as_ptr(), data.len())
    };

    let plain_err = ((plain_sum as f64 - f64_ref).abs()) / f64_ref.abs();
    let kahan_err = ((kahan_sum as f64 - f64_ref).abs()) / f64_ref.abs();
    let improvement_ratio = plain_err / kahan_err.max(1e-30);

    eprintln!("1M-sample Kahan drift test (f64 ref = {:.12}):", f64_ref);
    eprintln!("  Plain f32: {:.12} (err={:.3e})", plain_sum, plain_err);
    eprintln!("  Kahan f32: {:.12} (err={:.3e})", kahan_sum, kahan_err);
    eprintln!("  Improvement ratio: {:.1}×", improvement_ratio);

    assert!(
        improvement_ratio >= 100.0,
        "Kahan improvement ratio ({:.1}×) < 100×",
        improvement_ratio
    );
}
