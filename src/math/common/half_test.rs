// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Round-trip: f32 → f16_bits → f32 must be idempotent (within f16 precision).
#[test]
fn roundtrip_f32_f16_f32() {
    // Edge cases.
    let edge = [
        0.0f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        1.0,
        -1.0,
        65504.0, // max f16 normal
        -65504.0,
        6.1035156e-5, // min f16 normal (2^-14)
        5.9604645e-8, // min f16 subnormal (2^-24)
    ];
    for &v in &edge {
        let bits = f32_to_f16_bits(v);
        let back = f16_bits_to_f32(bits);
        let bits2 = f32_to_f16_bits(back);
        assert_eq!(bits, bits2, "idempotent failed for {:e}", v);
    }

    // Full positive normal range sampled at powers of two.
    for e in (-14i32..=15i32).rev() {
        for m in 0..1024 {
            let v = (1.0 + m as f32 / 1024.0) * 2.0f32.powi(e);
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);
            let bits2 = f32_to_f16_bits(back);
            assert_eq!(bits, bits2, "roundtrip failed for v={:e}", v);
        }
    }
}

/// ±0 must round-trip correctly.
#[test]
fn zero_roundtrip() {
    assert_eq!(f32_to_f16_bits(0.0), 0);
    assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    assert_eq!(f16_bits_to_f32(0), 0.0);
    assert_eq!(f16_bits_to_f32(0x8000), -0.0);
}

/// Infinity must round-trip correctly.
#[test]
fn infinity_roundtrip() {
    assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7C00);
    assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xFC00);
    assert!(f16_bits_to_f32(0x7C00).is_infinite());
    assert!(f16_bits_to_f32(0xFC00).is_infinite());
}

/// NaN must preserve the sign and produce a NaN.
#[test]
fn nan_roundtrip() {
    assert!(f16_bits_to_f32(0x7C01).is_nan());
    assert!(f16_bits_to_f32(0xFC01).is_nan());
    // Encoding f32::NAN → f16 → f32 must still be NaN.
    let bits = f32_to_f16_bits(f32::NAN);
    assert!(f16_bits_to_f32(bits).is_nan());
}

/// Exhaustive decode: F16C variant must be bit-exact vs software for all
/// 65.536 binary16 patterns.
#[test]
fn exhaustive_decode_f16c_vs_software() {
    for bits in 0u16..=0xFFFFu16 {
        let expected = f16_bits_to_f32(bits);
        // SAFETY: exhaustive test on x86_64 with F16C (x86-64-v3 guarantees it).
        let got = unsafe { f16_bits_to_f32_f16c(bits) };
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "F16C decode mismatch for bits=0x{:04X}",
            bits
        );
    }
}

/// Exhaustive encode: F16C variant must be bit-exact vs software for all
/// f32 values derived from the 65.536 binary16 patterns.
#[test]
fn exhaustive_encode_f16c_vs_software() {
    for bits in 0u16..=0xFFFFu16 {
        let f = f16_bits_to_f32(bits);
        let expected = f32_to_f16_bits(f);
        // SAFETY: exhaustive test on x86_64 with F16C (x86-64-v3 guarantees it).
        let got = unsafe { f32_to_f16_bits_f16c(f) };
        assert_eq!(
            got, expected,
            "F16C encode mismatch for f={:e} (f16 bits 0x{:04X})",
            f, bits
        );
    }
}

/// Regression for the mantissa-overflow carry bug (half.rs:117): values that
/// round UP into a mantissa overflow must carry into the exponent. The
/// previous code used `e | 0x0400`, which silently failed to increment the
/// exponent when the exponent field was odd (e.g. 7.9999 → 4.0 instead of
/// 8.0). `exhaustive_encode_f16c_vs_software` missed it because it only feeds
/// exact f16 values, which never trigger the rounding path. Here we sweep
/// the non-exact rounding windows just below each power-of-two boundary and
/// compare against the hardware F16C encoder (ground truth).
#[test]
fn encode_rounding_overflow_carry_vs_f16c() {
    // Sweep the upper rounding window of every binary16 exponent octave,
    // including the odd-exponent boundaries (±2, ±8, ±32, ±0.5, ...).
    for exp in -14i32..=15 {
        let boundary = 2.0f32.powi(exp + 1);
        // Walk just below the boundary through the last few representable
        // f16 steps, where rounding may overflow the mantissa.
        for k in 1..=4096u32 {
            let f = boundary * (1.0 - (k as f32) * 1e-7);
            for &v in &[f, -f] {
                // SAFETY: exhaustive test on x86_64 with F16C (x86-64-v3 guarantees it).
                let expected = unsafe { f32_to_f16_bits_f16c(v) };
                let got = f32_to_f16_bits(v);
                assert_eq!(
                    got, expected,
                    "encoder carry mismatch for v={:e}: software=0x{:04X} hardware=0x{:04X}",
                    v, got, expected
                );
            }
        }
    }
}
