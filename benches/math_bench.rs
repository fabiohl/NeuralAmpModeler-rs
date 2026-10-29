// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Micro-benchmarks of FastMath activation kernels (Tanh, Sigmoid) and dot-product
//! primitives.
//!
//! Covers AVX2 and AVX-512 SIMD paths, plus multiple Padé/polynomial approximation
//! schemes used across WaveNet, LSTM, and A2 inference.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench math_bench
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::math::common::AlignedVec;

/// Measures the performance of the AVX2-optimized `tanh` activation kernel.
/// Uses piecewise minimax odd polynomials (degree 5) with branchless blending
/// for maximum throughput at the expense of sub-sample precision.
fn bench_tanh_slice_256(c: &mut Criterion) {
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();

    c.bench_function("FastMath_tanh_AVX2_256elem", |b| {
        let mut buf = base.clone();
        b.iter(|| {
            buf.copy_from_slice(&base);
            unsafe { neural_amp_modeler_rs::math::activations::tanh_slice_avx2(&mut buf) };
        });
    });
}

/// Padé [5,4] tanh with single Newton-Raphson on reciprocal (AVX2).
/// Evaluates rcp_ps + 1×NR throughput vs div_ps and NR2.
fn bench_tanh_pade_nr1_256(c: &mut Criterion) {
    use neural_amp_modeler_rs::arch::simd::*;
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
    c.bench_function("FastMath_tanh_PadeNR1_AVX2_256elem", |b| {
        b.iter(|| {
            let mut buf = base.clone();
            unsafe {
                for chunk in buf.chunks_exact_mut(8) {
                    let x = _mm256_loadu_ps(chunk.as_ptr());
                    let y = neural_amp_modeler_rs::math::activations::simd_tanh_pade_nr1_avx2(x);
                    _mm256_storeu_ps(chunk.as_mut_ptr(), y);
                }
            }
        });
    });
}

/// Padé [5,4] tanh with single Newton-Raphson — dual path (AVX2).
/// Measures throughput when processing two independent registers.
fn bench_tanh_pade_nr1_dual_256(c: &mut Criterion) {
    use neural_amp_modeler_rs::arch::simd::*;
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
    c.bench_function("FastMath_tanh_PadeNR1_Dual_AVX2_256elem", |b| {
        b.iter(|| {
            let mut buf = base.clone();
            unsafe {
                for pair in buf.chunks_exact_mut(16) {
                    let x1 = _mm256_loadu_ps(pair[0..8].as_ptr());
                    let x2 = _mm256_loadu_ps(pair[8..16].as_ptr());
                    let (y1, y2) =
                        neural_amp_modeler_rs::math::activations::simd_tanh_pade_nr1_dual_avx2(
                            x1, x2,
                        );
                    _mm256_storeu_ps(pair[0..8].as_mut_ptr(), y1);
                    _mm256_storeu_ps(pair[8..16].as_mut_ptr(), y2);
                }
            }
        });
    });
}

/// Padé [5,4] tanh with double Newton-Raphson on reciprocal (AVX2).
/// Evaluates rcp_ps + 2×NR iteration throughput vs piecewise minimax.
fn bench_tanh_pade_nr2_256(c: &mut Criterion) {
    use neural_amp_modeler_rs::arch::simd::*;
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
    c.bench_function("FastMath_tanh_PadeNR2_AVX2_256elem", |b| {
        b.iter(|| {
            let mut buf = base.clone();
            unsafe {
                for chunk in buf.chunks_exact_mut(8) {
                    let x = _mm256_loadu_ps(chunk.as_ptr());
                    let y = neural_amp_modeler_rs::math::activations::simd_tanh_pade_nr2_avx2(x);
                    _mm256_storeu_ps(chunk.as_mut_ptr(), y);
                }
            }
        });
    });
}

/// Padé [5,4] tanh with hardware division oracle (AVX2).
/// IEEE 754 full-precision reference — maximum fidelity, minimum throughput.
fn bench_tanh_pade_div_256(c: &mut Criterion) {
    use neural_amp_modeler_rs::arch::simd::*;
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
    c.bench_function("FastMath_tanh_PadeDiv_AVX2_256elem", |b| {
        b.iter(|| {
            let mut buf = base.clone();
            unsafe {
                for chunk in buf.chunks_exact_mut(8) {
                    let x = _mm256_loadu_ps(chunk.as_ptr());
                    let y = neural_amp_modeler_rs::math::activations::simd_tanh_avx2(x);
                    _mm256_storeu_ps(chunk.as_mut_ptr(), y);
                }
            }
        });
    });
}

/// Measures the performance of the AVX2-optimized `sigmoid` activation kernel.
/// Essential for LSTM models, this kernel converts the approximate tanh into a
/// logistic function (0 to 1) to control memory gates.
fn bench_sigmoid_slice_256(c: &mut Criterion) {
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();

    c.bench_function("FastMath_sigmoid_AVX2_256elem", |b| {
        let mut buf = base.clone();
        b.iter(|| {
            buf.copy_from_slice(&base);
            unsafe { neural_amp_modeler_rs::math::activations::sigmoid_slice_avx2(&mut buf) };
        });
    });
}

/// Polynomial tanh with hardware division (AVX2).
/// IEEE 754 full-precision division — baseline for NR evaluation.
fn bench_tanh_poly_div_256(c: &mut Criterion) {
    use neural_amp_modeler_rs::arch::simd::*;
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
    c.bench_function("FastMath_tanh_PolyDiv_AVX2_256elem", |b| {
        b.iter(|| {
            let mut buf = base.clone();
            unsafe {
                for chunk in buf.chunks_exact_mut(8) {
                    let x = _mm256_loadu_ps(chunk.as_ptr());
                    let y = neural_amp_modeler_rs::math::activations::simd_tanh_poly_avx2(x);
                    _mm256_storeu_ps(chunk.as_mut_ptr(), y);
                }
            }
        });
    });
}

/// Polynomial tanh with single Newton-Raphson on reciprocal (AVX2).
/// Evaluates rcp_ps + 1×NR throughput vs hardware division.
fn bench_tanh_poly_nr1_256(c: &mut Criterion) {
    use neural_amp_modeler_rs::arch::simd::*;
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
    c.bench_function("FastMath_tanh_PolyNR1_AVX2_256elem", |b| {
        b.iter(|| {
            let mut buf = base.clone();
            unsafe {
                for chunk in buf.chunks_exact_mut(8) {
                    let x = _mm256_loadu_ps(chunk.as_ptr());
                    let y = neural_amp_modeler_rs::math::activations::simd_tanh_poly_nr1_avx2(x);
                    _mm256_storeu_ps(chunk.as_mut_ptr(), y);
                }
            }
        });
    });
}

/// Polynomial tanh with double Newton-Raphson on reciprocal (AVX2).
/// Evaluates rcp_ps + 2×NR throughput vs NR1 and hardware division (oracle).
fn bench_tanh_poly_nr2_256(c: &mut Criterion) {
    use neural_amp_modeler_rs::arch::simd::*;
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
    c.bench_function("FastMath_tanh_PolyNR2_AVX2_256elem", |b| {
        b.iter(|| {
            let mut buf = base.clone();
            unsafe {
                for chunk in buf.chunks_exact_mut(8) {
                    let x = _mm256_loadu_ps(chunk.as_ptr());
                    let y = neural_amp_modeler_rs::math::activations::simd_tanh_poly_nr2_avx2(x);
                    _mm256_storeu_ps(chunk.as_mut_ptr(), y);
                }
            }
        });
    });
}

/// Measures the throughput of the AVX2 dot product with f16 (Half Precision) weights.
/// This technique reduces memory bandwidth usage by 50% and improves
/// L1 cache locality, crucial for WaveNet dense layers.
fn bench_dot_product_avx2_256(c: &mut Criterion) {
    let vec_a = AlignedVec::from_vec((0..256).map(|i| (i as f32) * 0.1).collect())
        .expect("bench allocation failed");
    let vec_b = AlignedVec::from_vec((0..256).map(|i| (i as f32) * -0.1).collect())
        .expect("bench allocation failed");

    c.bench_function("DotProduct_AVX2_256elem", |b| {
        b.iter(|| unsafe {
            neural_amp_modeler_rs::math::gemm::dot_basic::dot_product_avx2(
                std::hint::black_box(&vec_a),
                std::hint::black_box(&vec_b),
            )
        });
    });
}

/// Dot product version for small vectors (64 elements).
/// Represents the typical intermediate layer size in lightweight models.
fn bench_dot_product_avx2_64(c: &mut Criterion) {
    let vec_a = AlignedVec::from_vec((0..64).map(|i| (i as f32) * 0.1).collect())
        .expect("bench allocation failed");
    let vec_b = AlignedVec::from_vec((0..64).map(|i| (i as f32) * -0.1).collect())
        .expect("bench allocation failed");
    c.bench_function("DotProduct_AVX2_64elem", |b| {
        b.iter(|| unsafe {
            neural_amp_modeler_rs::math::gemm::dot_basic::dot_product_avx2(
                std::hint::black_box(&vec_a),
                std::hint::black_box(&vec_b),
            )
        });
    });
}

/// Measures the performance of `dot_product_f32_native_kahan4` (LSTM per-sample head kernel)
/// for 64 elements (typical LSTM hidden state size).
fn bench_dot_product_f32_native_kahan4_64(c: &mut Criterion) {
    let vec_a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    let vec_b: Vec<f32> = (0..64).map(|i| (i as f32) * -0.1).collect();

    c.bench_function("DotProduct_Kahan4_64elem", |b| {
        b.iter(|| {
            neural_amp_modeler_rs::math::common::scalar_ref::dot_product_f32_native_kahan4(
                std::hint::black_box(&vec_a),
                std::hint::black_box(&vec_b),
            )
        });
    });
}

/// Measures the performance of `dot_product_f32_native_kahan4` (LSTM per-sample head kernel)
/// for 128 elements (larger LSTM hidden state size).
fn bench_dot_product_f32_native_kahan4_128(c: &mut Criterion) {
    let vec_a: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
    let vec_b: Vec<f32> = (0..128).map(|i| (i as f32) * -0.1).collect();

    c.bench_function("DotProduct_Kahan4_128elem", |b| {
        b.iter(|| {
            neural_amp_modeler_rs::math::common::scalar_ref::dot_product_f32_native_kahan4(
                std::hint::black_box(&vec_a),
                std::hint::black_box(&vec_b),
            )
        });
    });
}

/// Benchmarks for processors that support AVX-512 (e.g. AMD Zen 4, Intel Ice Lake+).
/// AVX-512 allows processing 16 floats simultaneously (512 bits), theoretically
/// doubling throughput compared to AVX2.
#[cfg(feature = "avx512")]
fn bench_tanh_avx512_256elem(c: &mut Criterion) {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
        let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
        c.bench_function("FastMath_tanh_AVX512_256elem", |b| {
            let mut buf = base.clone();
            b.iter(|| {
                buf.copy_from_slice(&base);
                unsafe { neural_amp_modeler_rs::math::activations::tanh_slice_avx512(&mut buf) };
            });
        });
    }
}
#[cfg(not(feature = "avx512"))]
fn bench_tanh_avx512_256elem(_c: &mut Criterion) {}

#[cfg(feature = "avx512")]
fn bench_sigmoid_avx512_256elem(c: &mut Criterion) {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
        let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
        c.bench_function("FastMath_sigmoid_AVX512_256elem", |b| {
            let mut buf = base.clone();
            b.iter(|| {
                buf.copy_from_slice(&base);
                unsafe { neural_amp_modeler_rs::math::activations::sigmoid_slice_avx512(&mut buf) };
            });
        });
    }
}
#[cfg(not(feature = "avx512"))]
fn bench_sigmoid_avx512_256elem(_c: &mut Criterion) {}

#[cfg(feature = "avx512")]
macro_rules! bench_avx512 {
    ($func_name:ident, $bench_name:literal, |$x:ident| $simd_call:expr) => {
        fn $func_name(c: &mut Criterion) {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl")
            {
                use neural_amp_modeler_rs::arch::simd::*;
                let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
                c.bench_function($bench_name, |b| {
                    b.iter(|| {
                        let mut buf = base.clone();
                        unsafe {
                            for chunk in buf.chunks_exact_mut(16) {
                                let $x = _mm512_loadu_ps(chunk.as_ptr());
                                let y = $simd_call;
                                _mm512_storeu_ps(chunk.as_mut_ptr(), y);
                            }
                        }
                    });
                });
            }
        }
    };
}
#[cfg(not(feature = "avx512"))]
macro_rules! bench_avx512 {
    ($func_name:ident, $bench_name:literal, |$x:ident| $simd_call:expr) => {
        fn $func_name(_c: &mut Criterion) {}
    };
}

bench_avx512!(
    bench_tanh_pade_nr1_avx512_256elem,
    "FastMath_tanh_PadeNR1_AVX512_256elem",
    |x| neural_amp_modeler_rs::math::activations::simd_tanh_pade_nr1_avx512(x)
);

bench_avx512!(
    bench_tanh_pade_nr2_avx512_256elem,
    "FastMath_tanh_PadeNR2_AVX512_256elem",
    |x| neural_amp_modeler_rs::math::activations::simd_tanh_pade_nr2_avx512(x)
);

bench_avx512!(
    bench_tanh_pade_div_avx512_256elem,
    "FastMath_tanh_PadeDiv_AVX512_256elem",
    |x| neural_amp_modeler_rs::math::activations::simd_tanh_avx512(x)
);

/// Block sizes that expose the 1–3-frame tail ("latency cliff") plus the
/// 64/128/256 regression guards.
const FUSED_RESIDUAL_BLOCK_SIZES: &[usize] = &[1, 2, 3, 5, 7, 63, 64, 65, 128, 256];

/// Fused residual const-generic batch GEMM — Nano CH=4, Feather CH=8 and
/// Standard CH=16 production shapes.
fn bench_fused_residual_const<const IN: usize, const OUT: usize>(c: &mut Criterion, label: &str) {
    use neural_amp_modeler_rs::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_const;
    for &frames in FUSED_RESIDUAL_BLOCK_SIZES {
        let in_frames: Vec<f32> = (0..frames * IN).map(|i| (i as f32 * 0.07).sin()).collect();
        let weights: Vec<f32> = (0..IN * OUT).map(|i| (i as f32 * 0.13).cos()).collect();
        let bias: Vec<f32> = (0..OUT).map(|i| i as f32 * 0.01).collect();
        let residual: Vec<f32> = (0..frames * OUT).map(|i| i as f32 * 0.02).collect();
        let mut out = vec![0.0f32; frames * OUT];
        c.bench_function(&format!("FusedResidual_{label}_{frames}frames"), |b| {
            b.iter(|| {
                // SAFETY: every buffer matches the declared IN×OUT batch dimensions.
                unsafe {
                    fused_gemm_residual_batch_f32_const::<IN, OUT>(
                        &in_frames, &weights, &bias, &residual, &mut out, frames, true,
                    );
                }
            });
        });
    }
}

fn bench_fused_residual_4x4(c: &mut Criterion) {
    bench_fused_residual_const::<4, 4>(c, "4x4");
}

fn bench_fused_residual_8x8(c: &mut Criterion) {
    bench_fused_residual_const::<8, 8>(c, "8x8");
}

fn bench_fused_residual_16x16(c: &mut Criterion) {
    bench_fused_residual_const::<16, 16>(c, "16x16");
}

/// Fused residual 12x12 batch GEMM — CH=12 production shape.
fn bench_fused_residual_12x12(c: &mut Criterion) {
    use neural_amp_modeler_rs::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_12x12;
    for &frames in FUSED_RESIDUAL_BLOCK_SIZES {
        let in_frames: Vec<f32> = (0..frames * 12).map(|i| (i as f32 * 0.07).sin()).collect();
        let weights: Vec<f32> = (0..144).map(|i| (i as f32 * 0.13).cos()).collect();
        let bias: Vec<f32> = (0..12).map(|i| i as f32 * 0.01).collect();
        let residual: Vec<f32> = (0..frames * 12).map(|i| i as f32 * 0.02).collect();
        let mut out = vec![0.0f32; frames * 12];
        c.bench_function(&format!("FusedResidual_12x12_{frames}frames"), |b| {
            b.iter(|| {
                // SAFETY: every buffer matches the declared 12x12 batch dimensions.
                unsafe {
                    fused_gemm_residual_batch_f32_12x12(
                        &in_frames, &weights, &bias, &residual, &mut out, frames, true,
                    );
                }
            });
        });
    }
}

/// Gated activation CH=4/CH=12 — the paired-frame remainder path.
fn bench_gated_activation_pairing(c: &mut Criterion) {
    use neural_amp_modeler_rs::math::wavenet::accumulate::gated_activation_and_accumulate_block_avx2;
    for &ch in &[4usize, 12] {
        let frames = 64;
        let head: Vec<f32> = (0..frames * ch).map(|i| (i as f32 * 0.03).sin()).collect();
        let block: Vec<f32> = (0..frames * 2 * ch)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let mut head_mut = head.clone();
        let mut block_mut = block.clone();
        c.bench_function(&format!("GatedActivation_CH{ch}_64frames"), |b| {
            b.iter(|| {
                head_mut.copy_from_slice(&head);
                block_mut.copy_from_slice(&block);
                // SAFETY: buffers hold `frames * ch` and `frames * 2 * ch` elements.
                unsafe {
                    gated_activation_and_accumulate_block_avx2(&mut head_mut, &mut block_mut, ch);
                }
            });
        });
    }
}

criterion_group! {
    name = math_benches;
    config = criterion::Criterion::default().sample_size(50).noise_threshold(0.05);
    targets = bench_tanh_slice_256,
    bench_tanh_pade_nr1_256,
    bench_tanh_pade_nr1_dual_256,
    bench_tanh_pade_nr2_256,
    bench_tanh_pade_div_256,
    bench_sigmoid_slice_256,
    bench_tanh_poly_div_256,
    bench_tanh_poly_nr1_256,
    bench_tanh_poly_nr2_256,
    bench_dot_product_avx2_256,
    bench_dot_product_avx2_64,
    bench_dot_product_f32_native_kahan4_64,
    bench_dot_product_f32_native_kahan4_128,
    bench_fused_residual_4x4,
    bench_fused_residual_8x8,
    bench_fused_residual_16x16,
    bench_fused_residual_12x12,
    bench_gated_activation_pairing,
    bench_tanh_avx512_256elem,
    bench_sigmoid_avx512_256elem,
    bench_tanh_pade_nr1_avx512_256elem,
    bench_tanh_pade_nr2_avx512_256elem,
    bench_tanh_pade_div_avx512_256elem
}

criterion_main!(math_benches);
