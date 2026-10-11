// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Exact-semantics tests for the NEON implementation of the x86 intrinsic
//! surface. Each test compares against a scalar model of the x86 instruction.

use super::{
    __m128, __m128i, __m256, __m256i, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT,
    _MM_FROUND_TO_NEG_INF, _MM_FROUND_TO_POS_INF, _MM_FROUND_TO_ZERO, _MM_HINT_T0, _MM_HINT_T1,
    _mm_add_ss, _mm_cvtph_ps, _mm_cvtps_ph, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_cvtss_f32,
    _mm_hadd_ps, _mm_loadu_ps, _mm_loadu_si64, _mm_movehdup_ps, _mm_movehl_ps, _mm_prefetch,
    _mm_set_ss, _mm_storeu_si128, _mm256_add_epi32, _mm256_add_ps, _mm256_blend_ps,
    _mm256_blendv_ps, _mm256_castps256_ps128, _mm256_cmp_ps, _mm256_cvtph_ps, _mm256_cvtps_epi32,
    _mm256_div_ps, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_fmsub_ps, _mm256_fnmadd_ps,
    _mm256_i32gather_ps, _mm256_insertf128_ps, _mm256_loadu_ps, _mm256_loadu_si256,
    _mm256_maskload_ps, _mm256_maskstore_ps, _mm256_max_ps, _mm256_min_ps, _mm256_movemask_ps,
    _mm256_mul_ps, _mm256_packus_epi32, _mm256_permute2f128_ps, _mm256_permute4x64_epi64,
    _mm256_rcp_ps, _mm256_round_ps, _mm256_set_m128, _mm256_set_ps, _mm256_set1_epi32,
    _mm256_set1_ps, _mm256_setr_ps, _mm256_shuffle_ps, _mm256_slli_epi32, _mm256_srli_epi32,
    _mm256_sub_ps, _mm256_unpackhi_ps, _mm256_unpacklo_ps,
};
use proptest::prelude::{Just, Strategy, prop_assert, prop_assert_eq, prop_oneof, proptest};

fn a8(v: __m256) -> [f32; 8] {
    unsafe { core::mem::transmute(v) }
}

fn v8(a: [f32; 8]) -> __m256 {
    unsafe { _mm256_loadu_ps(a.as_ptr()) }
}

fn a4(v: __m128) -> [f32; 4] {
    unsafe { core::mem::transmute(v) }
}

fn v4(a: [f32; 4]) -> __m128 {
    unsafe { _mm_loadu_ps(a.as_ptr()) }
}

fn i8x(v: __m256i) -> [i32; 8] {
    unsafe { core::mem::transmute(v) }
}

fn vi8(a: [i32; 8]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) }
}

fn bits8(a: [f32; 8]) -> [u32; 8] {
    a.map(f32::to_bits)
}

/// Floats including the edge cases that separate x86 and NEON semantics.
fn edge_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        4 => -1.0e3f32..1.0e3,
        1 => Just(0.0f32),
        1 => Just(-0.0f32),
        1 => Just(f32::NAN),
        1 => Just(f32::INFINITY),
        1 => Just(f32::NEG_INFINITY),
        1 => Just(1.0e-40f32),
    ]
}

fn arr8() -> impl Strategy<Value = [f32; 8]> {
    proptest::array::uniform8(edge_f32())
}

/// Scalar model of x86 `cmpps` for all 32 predicates.
fn cmp_model(a: f32, b: f32, p: i32) -> bool {
    let unord = a.is_nan() || b.is_nan();
    match p & 0xf {
        0x0 => !unord && a == b,
        0x1 => !unord && a < b,
        0x2 => !unord && a <= b,
        0x3 => unord,
        0x4 => unord || a != b,
        0x5 => unord || a >= b,
        0x6 => unord || a > b,
        0x7 => !unord,
        0x8 => unord || a == b,
        0x9 => unord || a < b,
        0xa => unord || a <= b,
        0xb => false,
        0xc => !unord && a != b,
        0xd => !unord && a >= b,
        0xe => !unord && a > b,
        _ => true,
    }
}

proptest! {
    #[test]
    fn arithmetic_matches_scalar_bitwise(a in arr8(), b in arr8(), c in arr8()) {
        let (va, vb, vc) = (v8(a), v8(b), v8(c));
        let add = a8(_mm256_add_ps(va, vb));
        let sub = a8(_mm256_sub_ps(va, vb));
        let mul = a8(_mm256_mul_ps(va, vb));
        let div = a8(_mm256_div_ps(va, vb));
        let fma = a8(_mm256_fmadd_ps(va, vb, vc));
        let fnma = a8(_mm256_fnmadd_ps(va, vb, vc));
        let fms = a8(_mm256_fmsub_ps(va, vb, vc));
        for i in 0..8 {
            let same = |x: f32, y: f32| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
            prop_assert!(same(add[i], a[i] + b[i]));
            prop_assert!(same(sub[i], a[i] - b[i]));
            prop_assert!(same(mul[i], a[i] * b[i]));
            prop_assert!(same(div[i], a[i] / b[i]));
            prop_assert!(same(fma[i], a[i].mul_add(b[i], c[i])));
            prop_assert!(same(fnma[i], (-a[i]).mul_add(b[i], c[i])));
            prop_assert!(same(fms[i], a[i].mul_add(b[i], -c[i])));
        }
    }

    #[test]
    fn max_min_select_like_x86(a in arr8(), b in arr8()) {
        let max = bits8(a8(_mm256_max_ps(v8(a), v8(b))));
        let min = bits8(a8(_mm256_min_ps(v8(a), v8(b))));
        for i in 0..8 {
            let x86_max = if a[i] > b[i] { a[i] } else { b[i] };
            let x86_min = if a[i] < b[i] { a[i] } else { b[i] };
            prop_assert_eq!(max[i], x86_max.to_bits());
            prop_assert_eq!(min[i], x86_min.to_bits());
        }
    }

    #[test]
    fn cmp_all_predicates(a in arr8(), b in arr8()) {
        for p in 0..32 {
            let m = bits8(a8(_mm256_cmp_ps(v8(a), v8(b), p)));
            for i in 0..8 {
                let expect = if cmp_model(a[i], b[i], p) { u32::MAX } else { 0 };
                prop_assert_eq!(m[i], expect, "predicate {:#x} lane {}", p, i);
            }
        }
    }

    #[test]
    fn movemask_and_blendv(a in arr8(), b in arr8(), m in arr8()) {
        let mask = _mm256_movemask_ps(v8(m));
        let blend = bits8(a8(_mm256_blendv_ps(v8(a), v8(b), v8(m))));
        for i in 0..8 {
            let sign = m[i].to_bits() >> 31 == 1;
            prop_assert_eq!((mask >> i) & 1 == 1, sign);
            prop_assert_eq!(blend[i], if sign { b[i] } else { a[i] }.to_bits());
        }
    }

    #[test]
    fn shuffles_and_blend_imm(a in arr8(), b in arr8(), imm in 0i32..256) {
        let s = bits8(a8(_mm256_shuffle_ps(v8(a), v8(b), imm)));
        let bl = bits8(a8(_mm256_blend_ps(v8(a), v8(b), imm)));
        let p = bits8(a8(_mm256_permute2f128_ps(v8(a), v8(b), imm)));
        let idx = |sh: i32| ((imm >> sh) & 3) as usize;
        for half in 0..2 {
            let o = half * 4;
            let expect = [a[o + idx(0)], a[o + idx(2)], b[o + idx(4)], b[o + idx(6)]];
            for j in 0..4 {
                prop_assert_eq!(s[o + j], expect[j].to_bits());
            }
            let ctl = imm >> (4 * half);
            let src = [&a[0..4], &a[4..8], &b[0..4], &b[4..8]][(ctl & 3) as usize];
            for j in 0..4 {
                let e = if ctl & 8 != 0 { 0.0f32 } else { src[j] };
                prop_assert_eq!(p[o + j], e.to_bits());
            }
        }
        for i in 0..8 {
            prop_assert_eq!(bl[i], if (imm >> i) & 1 == 1 { b[i] } else { a[i] }.to_bits());
        }
    }

    #[test]
    fn f16_roundtrip_matches_half_crate_semantics(x in proptest::array::uniform4(-7.0e4f32..7.0e4)) {
        let h = _mm_cvtps_ph(v4(x), _MM_FROUND_TO_NEAREST_INT);
        let back = a4(_mm_cvtph_ps(h));
        let wide = a8(_mm256_cvtph_ps(h));
        for i in 0..4 {
            let expect = crate::math::common::half::f16_bits_to_f32(crate::math::common::half::f32_to_f16_bits(x[i]));
            prop_assert_eq!(back[i].to_bits(), expect.to_bits());
            prop_assert_eq!(wide[i].to_bits(), expect.to_bits());
            prop_assert_eq!(wide[i + 4], 0.0);
        }
    }

    #[test]
    fn rcp_is_close(x in 1.0e-30f32..1.0e30) {
        let r = a8(_mm256_rcp_ps(_mm256_set1_ps(x)))[0];
        prop_assert!(((r * x) - 1.0).abs() < 1.0 / 4096.0);
    }
}

#[test]
fn construction_lane_order() {
    let r = a8(_mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0));
    let s = a8(_mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0));
    assert_eq!(r, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    assert_eq!(s, r);
    let m = a8(_mm256_set_m128(
        v4([4.0, 5.0, 6.0, 7.0]),
        v4([0.0, 1.0, 2.0, 3.0]),
    ));
    assert_eq!(m, r);
    assert_eq!(a4(_mm256_extractf128_ps(v8(r), 1)), [4.0, 5.0, 6.0, 7.0]);
    assert_eq!(a4(_mm256_castps256_ps128(v8(r))), [0.0, 1.0, 2.0, 3.0]);
    let ins = a8(_mm256_insertf128_ps(v8(r), v4([9.0; 4]), 1));
    assert_eq!(ins, [0.0, 1.0, 2.0, 3.0, 9.0, 9.0, 9.0, 9.0]);
    assert_eq!(a4(_mm_set_ss(3.0)), [3.0, 0.0, 0.0, 0.0]);
}

#[test]
fn lane_movement_128() {
    let a = v4([0.0, 1.0, 2.0, 3.0]);
    let b = v4([10.0, 11.0, 12.0, 13.0]);
    assert_eq!(a4(_mm_movehl_ps(a, b)), [12.0, 13.0, 2.0, 3.0]);
    assert_eq!(a4(_mm_movehdup_ps(a)), [1.0, 1.0, 3.0, 3.0]);
    assert_eq!(a4(_mm_hadd_ps(a, b)), [1.0, 5.0, 21.0, 25.0]);
    assert_eq!(a4(_mm_add_ss(a, b)), [10.0, 1.0, 2.0, 3.0]);
    assert_eq!(_mm_cvtss_f32(b), 10.0);
    let a8v = v8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let b8v = v8([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    assert_eq!(
        a8(_mm256_unpacklo_ps(a8v, b8v)),
        [0.0, 10.0, 1.0, 11.0, 4.0, 14.0, 5.0, 15.0]
    );
    assert_eq!(
        a8(_mm256_unpackhi_ps(a8v, b8v)),
        [2.0, 12.0, 3.0, 13.0, 6.0, 16.0, 7.0, 17.0]
    );
}

#[test]
fn rounding_modes() {
    let x = v8([0.5, 1.5, 2.5, -0.5, -1.5, 2.7, -2.7, 1.2]);
    let near = a8(_mm256_round_ps(
        x,
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    ));
    assert_eq!(near, [0.0, 2.0, 2.0, -0.0, -2.0, 3.0, -3.0, 1.0]);
    assert_eq!(a8(_mm256_round_ps(x, _MM_FROUND_TO_NEG_INF))[5], 2.0);
    assert_eq!(a8(_mm256_round_ps(x, _MM_FROUND_TO_POS_INF))[7], 2.0);
    assert_eq!(a8(_mm256_round_ps(x, _MM_FROUND_TO_ZERO))[6], -2.0);
    assert_eq!(i8x(_mm256_cvtps_epi32(x)), [0, 2, 2, 0, -2, 3, -3, 1]);
}

#[test]
fn integer_ops() {
    let a = vi8([1, -2, 3, -4, 70000, -70000, 65535, 0]);
    assert_eq!(i8x(_mm256_slli_epi32(a, 23))[0], 1 << 23);
    assert_eq!(i8x(_mm256_slli_epi32(a, 32)), [0; 8]);
    assert_eq!(
        i8x(_mm256_srli_epi32(a, 16))[1],
        (-2i32 as u32 >> 16) as i32
    );
    assert_eq!(i8x(_mm256_add_epi32(a, _mm256_set1_epi32(1)))[3], -3);
    // packus: per half [sat(a0..a3), sat(b0..b3)] as u16.
    let p: [u16; 16] = unsafe { core::mem::transmute(_mm256_packus_epi32(a, a)) };
    assert_eq!(p[..8], [1, 0, 3, 0, 1, 0, 3, 0]);
    assert_eq!(p[8..], [65535, 0, 65535, 0, 65535, 0, 65535, 0]);
    let q: [u64; 4] = unsafe { core::mem::transmute(_mm256_permute4x64_epi64(a, 0b00_01_10_11)) };
    let src: [u64; 4] = unsafe { core::mem::transmute(a) };
    assert_eq!(q, [src[3], src[2], src[1], src[0]]);
    assert_eq!(_mm_cvtsi128_si32(_mm_cvtsi32_si128(-7)), -7);
}

#[test]
fn masked_access_and_gather() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mask = vi8([-1, 0, -1, 0, -1, 0, 0, -1]);
    // Only the first three lanes are enabled here, so reading past `short`
    // would be out of bounds; masked-off lanes must not be read.
    let short = [9.0f32, 9.5, 10.0];
    let tail = vi8([-1, -1, -1, 0, 0, 0, 0, 0]);
    unsafe {
        assert_eq!(
            a8(_mm256_maskload_ps(data.as_ptr(), mask)),
            [1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 0.0, 8.0]
        );
        assert_eq!(a8(_mm256_maskload_ps(short.as_ptr(), tail))[..3], short);
        let mut out = [0.0f32; 8];
        _mm256_maskstore_ps(out.as_mut_ptr(), mask, v8(data));
        assert_eq!(out, [1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 0.0, 8.0]);
        let idx = vi8([7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(
            a8(_mm256_i32gather_ps(data.as_ptr(), idx, 4)),
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        );
        let mut half = [0i32; 4];
        _mm_storeu_si128(
            half.as_mut_ptr() as *mut __m128i,
            _mm_loadu_si64([1u8, 0, 0, 0, 2, 0, 0, 0].as_ptr()),
        );
        assert_eq!(half, [1, 2, 0, 0]);
        _mm_prefetch::<_MM_HINT_T0>(data.as_ptr() as *const i8);
        _mm_prefetch::<_MM_HINT_T1>(data.as_ptr() as *const i8);
    }
}
