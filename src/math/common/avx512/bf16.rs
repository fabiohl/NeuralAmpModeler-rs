// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 BF16 dispatch macro (`impl_avx512_bf16!()`).
//!
//! Generates BF16 conversion, GEMV, and activation wrappers when
//! AVX-512 VNNI+BF16 ISA is available. Invoked by [`dispatch_simd!`].

macro_rules! impl_avx512_bf16 {
    () => {
        #[inline(always)]
        // SAFETY: src and dest are valid slices with dest.len() >= src.len();
        // CPU supports AVX-512F+VL (verified by dispatch). Kernel uses unaligned 512-bit loads
        // and 256-bit stores.
        unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
            // SAFETY: src and dest satisfy the function's documented invariants.
            unsafe { crate::math::common::ops::f32_to_bf16_avx512(src, dest) }
        }

        #[inline(always)]
        // SAFETY: ptr is a valid mutable pointer to 16 u16 elements (256-bit aligned or
        // unaligned acceptable); v is a valid __m512 register holding f32 values in the
        // low 16 bits of each lane; CPU supports AVX-512F+VL (verified by dispatch).
        unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
            // SAFETY: ptr points to at least 16 writable u16 elements; the _mm256_storeu_si256
            // intrinsic performs an unaligned store.
            unsafe {
                let v_i = _mm512_castps_si512(v);
                let v_shifted = _mm512_srli_epi32(v_i, 16);
                let packed = _mm512_cvtepi32_epi16(v_shifted);
                _mm256_storeu_si256(ptr as *mut __m256i, packed);
            }
        }
    };
}

macro_rules! impl_avx512vnni_bf16_bf16 {
    () => {
        #[inline(always)]
        // SAFETY: src and dest are valid slices; CPU supports AVX-512 VNNI+BF16.
        unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
            Avx512Math::f32_to_bf16(src, dest)
        }

        #[inline(always)]
        // SAFETY: ptr is a valid mutable pointer to 16 u16 elements; CPU supports AVX-512 VNNI+BF16.
        unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
            Avx512Math::store_bf16(ptr, v)
        }
    };
}

pub(super) use impl_avx512_bf16;
pub(super) use impl_avx512vnni_bf16_bf16;
