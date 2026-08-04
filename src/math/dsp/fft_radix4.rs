// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Radix-4 Decimation-in-Time (DIT) FFT — Research Prototype (Task A9, Spr.4).
//!
//! **Status: CONCLUÍDO — NÃO USAR EM PRODUÇÃO.**
//!
//! Este módulo é um artefato de pesquisa preservado para referência futura.
//! O protótipo implementa um FFT Radix-4 DIT iterativo in-place sobre buffers
//! SoA (`&mut [T]`), with API compatible with the `FftPlanner` (Radix-2) from
//! produção.
//!
//! # Decisão de engenharia
//!
//! Os benchmarks criterion (N=256 e N=1024, f32) demonstram que o Radix-4
//! escalar é **7–19% mais lento** que o Radix-2 escalar, apesar de ter metade
//! dos estágios (`log₄N` vs `log₂N`). As causas identificadas:
//!
//! 1. **3× mais acessos a twiddles** por butterfly (W¹, W², W³), sobrecarregando
//!    cache L1 de dados.
//! 2. **Padrão de acesso strided** (L, 2L, 3L) que prejudica prefetch de hardware.
//! 3. **Butterfly mais pesado**: 30 operações para 4 elementos vs 8 operações
//!    para 2 elementos no Radix-2 — na prática pior que a razão teórica de
//!    3.75:4 ops/elemento devido a pressão de registrador e branching.
//! 4. **Branch condicional** (`if inverse`) no laço interno, quebrando
//!    pipelining do compilador.
//!
//! Stockham (auto-sort, elimina bit-reversal) e Split-Radix também foram
//! analisados e descartados. O bit-reversal representa <2% do tempo total
//! para N≤1024; Split-Radix tem padrão de acesso irregular que impede
//! vetorização SIMD eficiente.
//!
//! O algoritmo canônico do projeto permanece sendo o Radix-2 DIT com SIMD
//! (Tarefa A8), implementado em [`super::fft`].
//!
//! # Histórico da pesquisa
//!
//! * **Teoria**: Radix-4 DIT teria vantagem teórica (~6% menos operações),
//!   metade dos estágios e potencial de SIMD com reuse de registrador.
//! * **Protótipo**: Implementação funcional com 14 testes (paridade forward
//!   vs Radix-2 para N=4,16,64,256,1024; roundtrip; impulso; f64).
//!   Correções aplicadas: bit-reversal em base-4 (ao invés de base-2) e
//!   swap das fórmulas de X₁/X₃ no butterfly inverso.
//! * **Benchmarks**: `cargo bench --bench fft_radix4_bench`
//! * **Conclusão**: 2026-06-25. Radix-4, Stockham e Split-Radix não justificam
//!   a complexidade adicional frente ao Radix-2 SIMD.
//!
//! # Limitações técnicas (para referência)
//!
//! * N deve ser **potência de 4** (4, 16, 64, 256, 1024, …). Para tamanhos
//!   mistos (512, 2048), seria necessário um híbrido Radix-2+4.
//! * Protótipo **escalar**; SIMD exigiria um novo método no trait `SimdMath`
//!   (análogo a `fft_butterfly_stage`) com shuffle/permute para recombinar
//!   os 4 outputs do butterfly a partir dos 3 inputs twiddlados.

#[cfg(any(test, feature = "long_bench"))]
use super::fft::FftFloat;

/// Pre-computed Radix-4 DIT FFT plan.
#[cfg(any(test, feature = "long_bench"))]
pub struct FftPlannerRadix4<T: FftFloat> {
    n: usize,
    bit_reverse: Vec<usize>,
    stage_twiddle_re1: Vec<T>,
    stage_twiddle_im1: Vec<T>,
    stage_twiddle_re2: Vec<T>,
    stage_twiddle_im2: Vec<T>,
    stage_twiddle_re3: Vec<T>,
    stage_twiddle_im3: Vec<T>,
    stage_l: Vec<usize>,
}

#[cfg(any(test, feature = "long_bench"))]
impl<T: FftFloat> FftPlannerRadix4<T> {
    /// Creates a new Radix-4 FFT plan for size `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is not a power of two, is less than 4.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "FFT size must be positive");
        assert!(
            n.is_power_of_two(),
            "FFT size must be a power of two (Radix-4 requires power of four), got {n}"
        );
        assert!(n >= 4, "Radix-4 FFT requires N ≥ 4, got {n}");

        let n_half = n / 2;
        let num_stages_radix4 = (n.ilog2() / 2) as usize;
        let base4_digits = num_stages_radix4;

        let mut bit_reverse = vec![0usize; n];
        #[expect(
            clippy::needless_range_loop,
            reason = "Range loop required for explicit SIMD lane indexing not expressible via iterator"
        )]
        for i in 0..n {
            let mut rev = 0usize;
            let mut x = i;
            for _ in 0..base4_digits {
                rev = (rev << 2) | (x & 0x3);
                x >>= 2;
            }
            bit_reverse[i] = rev;
        }

        let tau = T::tau();
        let n_t = T::from_usize(n);
        let twiddle_re: Vec<T> = (0..n_half)
            .map(|k| {
                let angle = tau * T::from_usize(k) * n_t.recip();
                angle.cos()
            })
            .collect();
        let twiddle_im: Vec<T> = (0..n_half)
            .map(|k| {
                let angle = tau * T::from_usize(k) * n_t.recip();
                -angle.sin()
            })
            .collect();

        let total = n.saturating_sub(1);
        let mut stage_twiddle_re1 = Vec::with_capacity(total);
        let mut stage_twiddle_im1 = Vec::with_capacity(total);
        let mut stage_twiddle_re2 = Vec::with_capacity(total);
        let mut stage_twiddle_im2 = Vec::with_capacity(total);
        let mut stage_twiddle_re3 = Vec::with_capacity(total);
        let mut stage_twiddle_im3 = Vec::with_capacity(total);
        let mut stage_l = Vec::with_capacity(num_stages_radix4);

        let mut len = 4;
        while len <= n {
            let l = len / 4;
            let step = n / len;
            stage_l.push(l);
            for j in 0..l {
                let w1 = j * step;
                let w2 = (2 * j) * step;
                let w3 = (3 * j) * step;
                stage_twiddle_re1.push(twiddle_re[w1]);
                stage_twiddle_im1.push(twiddle_im[w1]);
                stage_twiddle_re2.push(twiddle_re[w2]);
                stage_twiddle_im2.push(twiddle_im[w2]);
                if w3 < n_half {
                    stage_twiddle_re3.push(twiddle_re[w3]);
                    stage_twiddle_im3.push(twiddle_im[w3]);
                } else {
                    stage_twiddle_re3.push(-twiddle_re[w3 - n_half]);
                    stage_twiddle_im3.push(-twiddle_im[w3 - n_half]);
                }
            }
            len <<= 2;
        }

        Self {
            n,
            bit_reverse,
            stage_twiddle_re1,
            stage_twiddle_im1,
            stage_twiddle_re2,
            stage_twiddle_im2,
            stage_twiddle_re3,
            stage_twiddle_im3,
            stage_l,
        }
    }

    /// Returns the FFT size.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the size is zero (never, guarded at construction).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Forward complex FFT, in-place on SoA buffers.
    ///
    /// Validates buffer lengths with `assert_eq!` in all profiles
    /// before delegating to the unchecked Radix-4 kernel.
    pub fn process(&self, re: &mut [T], im: &mut [T]) {
        assert_eq!(re.len(), self.n, "re length mismatch");
        assert_eq!(im.len(), self.n, "im length mismatch");

        // Bit-reversal
        for (i, &j) in self.bit_reverse.iter().enumerate() {
            if i < j {
                unsafe {
                    std::ptr::swap(re.get_unchecked_mut(i), re.get_unchecked_mut(j));
                    std::ptr::swap(im.get_unchecked_mut(i), im.get_unchecked_mut(j));
                }
            }
        }

        // Radix-4 butterflies
        let mut tw_offset = 0usize;
        for &l in &self.stage_l {
            unsafe { self.radix4_stage(re, im, l, tw_offset, false) };
            tw_offset += l;
        }
    }

    /// Inverse complex FFT, in-place on SoA buffers.
    ///
    /// Validates buffer lengths with `assert_eq!` in all profiles
    /// before delegating to the unchecked Radix-4 kernel. Applies
    /// `1/n` scaling at the end.
    pub fn process_inverse(&self, re: &mut [T], im: &mut [T]) {
        assert_eq!(re.len(), self.n, "re length mismatch");
        assert_eq!(im.len(), self.n, "im length mismatch");

        for (i, &j) in self.bit_reverse.iter().enumerate() {
            if i < j {
                unsafe {
                    std::ptr::swap(re.get_unchecked_mut(i), re.get_unchecked_mut(j));
                    std::ptr::swap(im.get_unchecked_mut(i), im.get_unchecked_mut(j));
                }
            }
        }

        let mut tw_offset = 0usize;
        for &l in &self.stage_l {
            unsafe { self.radix4_stage(re, im, l, tw_offset, true) };
            tw_offset += l;
        }

        let scale = T::from_usize(self.n).recip();
        for s in re.iter_mut() {
            *s = *s * scale;
        }
        for s in im.iter_mut() {
            *s = *s * scale;
        }
    }

    /// Processes one Radix-4 butterfly stage.
    ///
    /// # Safety
    ///
    /// The caller must guarantee:
    /// - `re.len() == im.len() == self.n`
    /// - `l` is a valid stage sub-block size where `4 * l ≤ n`
    /// - `tw_offset + l` does not exceed the capacity of stage twiddle
    ///   vectors, which are pre-computed at construction time.
    ///
    /// For a stage of length `4*L`:
    /// - Processes `N / (4*L)` groups of 4 elements each.
    /// - Each group at positions [k, k+L, k+2L, k+3L].
    /// - Twiddles for offset j (0..L) are at tw_offset + j.
    unsafe fn radix4_stage(
        &self,
        re: &mut [T],
        im: &mut [T],
        l: usize,
        tw_offset: usize,
        inverse: bool,
    ) {
        let len = 4 * l;
        for k in (0..self.n).step_by(len) {
            for j in 0..l {
                let idx0 = k + j;
                let idx1 = k + j + l;
                let idx2 = k + j + 2 * l;
                let idx3 = k + j + 3 * l;

                let tw_idx = tw_offset + j;
                let w1_re = unsafe { *self.stage_twiddle_re1.get_unchecked(tw_idx) };
                let w1_im = if inverse {
                    unsafe { -*self.stage_twiddle_im1.get_unchecked(tw_idx) }
                } else {
                    unsafe { *self.stage_twiddle_im1.get_unchecked(tw_idx) }
                };
                let w2_re = unsafe { *self.stage_twiddle_re2.get_unchecked(tw_idx) };
                let w2_im = if inverse {
                    unsafe { -*self.stage_twiddle_im2.get_unchecked(tw_idx) }
                } else {
                    unsafe { *self.stage_twiddle_im2.get_unchecked(tw_idx) }
                };
                let w3_re = unsafe { *self.stage_twiddle_re3.get_unchecked(tw_idx) };
                let w3_im = if inverse {
                    unsafe { -*self.stage_twiddle_im3.get_unchecked(tw_idx) }
                } else {
                    unsafe { *self.stage_twiddle_im3.get_unchecked(tw_idx) }
                };

                let (r0, i0, r1, i1, r2, i2, r3, i3) = unsafe {
                    (
                        *re.get_unchecked(idx0),
                        *im.get_unchecked(idx0),
                        *re.get_unchecked(idx1),
                        *im.get_unchecked(idx1),
                        *re.get_unchecked(idx2),
                        *im.get_unchecked(idx2),
                        *re.get_unchecked(idx3),
                        *im.get_unchecked(idx3),
                    )
                };

                let y1_re = w1_re.mul_add(r1, -w1_im * i1);
                let y1_im = w1_re.mul_add(i1, w1_im * r1);
                let y2_re = w2_re.mul_add(r2, -w2_im * i2);
                let y2_im = w2_re.mul_add(i2, w2_im * r2);
                let y3_re = w3_re.mul_add(r3, -w3_im * i3);
                let y3_im = w3_re.mul_add(i3, w3_im * r3);

                unsafe {
                    *re.get_unchecked_mut(idx0) = (r0 + y1_re) + (y2_re + y3_re);
                    *im.get_unchecked_mut(idx0) = (i0 + y1_im) + (y2_im + y3_im);
                    if inverse {
                        *re.get_unchecked_mut(idx3) = (r0 + y1_im) - (y2_re + y3_im);
                        *im.get_unchecked_mut(idx3) = (i0 - y1_re) - (y2_im - y3_re);
                        *re.get_unchecked_mut(idx2) = (r0 - y1_re) + (y2_re - y3_re);
                        *im.get_unchecked_mut(idx2) = (i0 - y1_im) + (y2_im - y3_im);
                        *re.get_unchecked_mut(idx1) = (r0 - y1_im) - (y2_re - y3_im);
                        *im.get_unchecked_mut(idx1) = (i0 + y1_re) - (y2_im + y3_re);
                    } else {
                        *re.get_unchecked_mut(idx1) = (r0 + y1_im) - (y2_re + y3_im);
                        *im.get_unchecked_mut(idx1) = (i0 - y1_re) - (y2_im - y3_re);
                        *re.get_unchecked_mut(idx2) = (r0 - y1_re) + (y2_re - y3_re);
                        *im.get_unchecked_mut(idx2) = (i0 - y1_im) + (y2_im - y3_im);
                        *re.get_unchecked_mut(idx3) = (r0 - y1_im) - (y2_re - y3_im);
                        *im.get_unchecked_mut(idx3) = (i0 + y1_re) - (y2_im + y3_re);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "fft_radix4_test.rs"]
mod tests;
