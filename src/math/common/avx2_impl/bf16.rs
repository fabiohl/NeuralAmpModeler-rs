// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX2 BF16 dispatch stubs.
//!
//! AVX2 does not support BF16 instructions — all methods are `unreachable!()`.
//! These stubs exist so the `IS_BF16=false` compile-time path compiles cleanly.
//! Invoked by [`dispatch_simd!`] via the Avx2Math implementation.

macro_rules! impl_avx2_bf16 {
    () => {
        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_bf16(_a: &[u16], _b: &[u16]) -> f32 {
            unreachable!("AVX2 IS_BF16=false; BF16 paths are never reached at runtime")
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn dot_product_bf16_4x(
            _w0: &[u16],
            _w1: &[u16],
            _w2: &[u16],
            _w3: &[u16],
            _in_frame: &[u16],
        ) -> [f32; 4] {
            unreachable!("AVX2 IS_BF16=false; BF16 paths are never reached at runtime")
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn f32_to_bf16(_src: &[f32], _dest: &mut [u16]) {
            unreachable!("AVX2 IS_BF16=false; BF16 paths are never reached at runtime")
        }

        /// Converts a register containing 8 32-bit floats (Self::V) to the compact BF16 (16-bit)
        /// format and stores the results in memory.
        ///
        /// ## Bitwise SIMD Magic Details:
        /// To convert f32 to BF16 without spending many CPU cycles on slow mathematical conversions,
        /// the technique leverages the structural similarity between IEEE 754 single-precision float and BF16:
        /// Both share the same dynamic range (8 exponent bits), but BF16 discards the
        /// 16 least significant mantissa bits (truncation/rounding).
        #[inline(always)]
        // SAFETY: ptr is a valid mutable pointer to 8 u16 elements; v is a valid __m256 register;
        // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch). Intrinsic transformation
        // uses bitwise repacking; the _mm_storeu_si128 performs an unaligned 128-bit store.
        unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
            // SAFETY: ptr points to at least 8 writable u16 elements (128-bit store).
            unsafe {
                // 1. Reinterpret the f32 register (__m256) as 32-bit integers (__m256i). Cost: 0 cycles.
                let v_i = _mm256_castps_si256(v);
                // 2. Shift the integers 16 bits to the right. The upper half (BF16 mantissa)
                // moves to the lower half of each 32-bit element.
                let v_shifted = _mm256_srli_epi32(v_i, 16);
                // 3. Pack with unsigned saturation (Packus) 32-bit elements into 16-bit elements.
                // This consolidates the useful 16-bit data.
                let packed = _mm256_packus_epi32(v_shifted, v_shifted);
                // 4. Permute 64-bit lanes using pattern (8/0x08) to regroup the valid results
                // that got mixed up due to the inherent behavior of the _mm256_packus_epi32 instruction.
                let permuted = _mm256_permute4x64_epi64(packed, 8);
                // 5. Extract the lower half (128 bits of a 256-bit register), containing the 8 BF16 values.
                let v_low = _mm256_castsi256_si128(permuted);
                // 6. Write the 8 compacted BF16 values directly to the destination in RAM.
                _mm_storeu_si128(ptr as *mut __m128i, v_low);
            }
        }
    };
}
pub(crate) use impl_avx2_bf16;
