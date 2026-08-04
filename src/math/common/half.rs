// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Software IEEE-754 binary16 (half-precision) ↔ f32 conversion, bit-exact.
//!
//! ## Binary16 layout (16 bits)
//! | Sign (1) | Exponent (5) | Mantissa (10) |
//!
//! - bias = 15
//! - Subnormal exponent = 0, implicit leading 0; value = (−1)^s × 2^(−14) × 0.m
//! - Normal   exponent ∈ [1, 30], implicit leading 1
//! - exp = 31: Infinity (m=0) or NaN (m≠0)

/// Decodes a binary16 bit-pattern to f32.
///
/// Handles normals, subnormals, ±0, ±Inf, and NaN.
#[inline]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let u = bits as u32;
    let sign = (u >> 15) << 31;
    let exp = (u >> 10) & 0x1F;
    let mant = u & 0x3FF;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign) // ±0
        } else {
            // Subnormal → normalize mantissa into f32 range.
            let z = mant.leading_zeros().wrapping_sub(22); // leading zeros in 10-bit field
            let e = (112u32).wrapping_sub(z);
            let frac = mant.wrapping_shl(z + 14) & 0x7F_FFFF;
            f32::from_bits(sign | (e << 23) | frac)
        }
    } else if exp == 0x1F {
        if mant == 0 {
            f32::from_bits(sign | 0x7F80_0000) // Infinity
        } else {
            f32::from_bits(sign | 0x7FC0_0000 | (mant << 13)) // NaN (quiet)
        }
    } else {
        let e = ((exp as i32) - 15 + 127).wrapping_shl(23) as u32;
        f32::from_bits(sign | e | (mant << 13))
    }
}

/// Encodes an f32 value to binary16 bits (u16) with round-to-nearest-even.
///
/// Overflow → ±Inf. Subnormals are produced when the value is too small for
/// the normal range.
#[inline]
pub fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) as u16) & 0x8000;
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_mant = bits & 0x7F_FFFF;

    // Special: NaN / Infinity.
    if f32_exp == 0xFF {
        return if f32_mant == 0 {
            sign | 0x7C00 // Infinity
        } else {
            sign | 0x7E00 | ((f32_mant >> 13) as u16 & 0x1FF) // NaN (quiet)
        };
    }

    // Normalize the f32 value into (1.fraction) × 2^exp_adj.
    let (exp_adj, fraction) = if f32_exp == 0 {
        if f32_mant == 0 {
            return sign; // ±0
        }
        // Subnormal f32 → find the leading 1.
        let lz = f32_mant.leading_zeros().wrapping_sub(9); // lz in 23-bit mantissa
        (-127 - lz as i32, f32_mant.wrapping_shl(lz + 1) & 0x7F_FFFF)
    } else {
        (f32_exp - 127, f32_mant)
    };

    // Overflow → Infinity.
    if exp_adj > 15 {
        return sign | 0x7C00;
    }

    // Subnormal f16 range: exp_adj ∈ [-25, -15).
    if exp_adj < -14 {
        if exp_adj < -25 {
            return sign; // Underflow → ±0
        }
        let shift = (-1 - exp_adj) as u32;
        let mant_full = 0x80_0000u32 | fraction;
        let mut m = (mant_full >> shift) as u16;

        // Round-to-nearest-even on the dropped bits.
        let dropped = mant_full & ((1u32 << shift).wrapping_sub(1));
        let guard = 1u32 << (shift - 1);
        if dropped > guard || (dropped == guard && (m & 1) != 0) {
            m += 1;
            if m >= 0x400 {
                return sign | 0x0400; // Smallest normal
            }
        }
        return sign | (m & 0x3FF);
    }

    // Normal range: exp_adj ∈ [-14, 15].
    let e = ((exp_adj + 15) as u16) << 10;
    let mut m = ((fraction >> 13) as u16) & 0x3FF;

    // Round-to-nearest-even on the 13 lower bits of the f32 mantissa.
    let dropped = fraction & 0x1FFF;
    if dropped > 0x1000 || (dropped == 0x1000 && (m & 1) != 0) {
        m += 1;
        if m >= 0x400 {
            // Mantissa overflow → carry into the exponent (mantissa wraps to 0).
            // Must ADD the carry (0x0400) to the exponent field, not OR it:
            // when the exponent field is odd, bit 10 is already set and an OR
            // would silently fail to increment the exponent (halving the value).
            if e + 0x0400 >= 0x7C00 {
                return sign | 0x7C00; // Overflow to Inf
            }
            return sign | (e + 0x0400);
        }
    }
    sign | e | m
}

// ---------------------------------------------------------------------------
// F16C hardware-accelerated variants (scalar, single f16 ↔ f32)
// ---------------------------------------------------------------------------

/// Decodes a binary16 bit-pattern to f32 using the F16C `VCVTPH2PS` instruction.
///
/// `x86-64-v3` guarantees F16C availability (`.cargo/config.toml`).
///
/// # Safety
///
/// The caller must ensure the CPU supports the `f16c` target feature.
/// This is guaranteed when compiling with `x86-64-v3` or higher.
#[inline]
#[target_feature(enable = "f16c")]
pub unsafe fn f16_bits_to_f32_f16c(bits: u16) -> f32 {
    use std::arch::x86_64::_mm_cvtph_ps;
    use std::arch::x86_64::_mm_cvtsi32_si128;
    use std::arch::x86_64::_mm_cvtss_f32;
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
}

/// Encodes an f32 value to binary16 bits (u16) using the F16C `VCVTPS2PH`
/// instruction with round-to-nearest-even.
///
/// # Safety
///
/// The caller must ensure the CPU supports the `f16c` target feature.
/// This is guaranteed when compiling with `x86-64-v3` or higher.
#[inline]
#[target_feature(enable = "f16c")]
pub unsafe fn f32_to_f16_bits_f16c(f: f32) -> u16 {
    use std::arch::x86_64::_MM_FROUND_TO_NEAREST_INT;
    use std::arch::x86_64::_mm_cvtps_ph;
    use std::arch::x86_64::_mm_cvtsi128_si32;
    use std::arch::x86_64::_mm_set_ss;
    (_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT)) as u32 & 0xFFFF)
        as u16
}

#[cfg(test)]
mod tests {
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
}
