// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Feature-wise Linear Modulation (FiLM) module for the NAM A2 architecture.
//!
//! FiLM enables the model to adapt its behavior based on external
//! conditioning signals, applying per-channel scale and shift:
//!
//! `output[c] = input[c] * scale[c] + shift[c]`
//!
//! The `_cond_to_scale_shift` operation (C++ `NAM/film.h:28-85`) maps
//! the condition vector to per-channel scale+shift via a Conv1x1
//! implemented as a dense GEMV (matrix-vector multiply with AVX2 FMA).
//!
//! # RT-Safety
//! - The `scale_shift_buf` is pre-allocated in `load()` — zero allocation
//!   on the hot-path.
//! - The `apply_modulation` inner loop uses `chunks_exact_mut` with no
//!   internal branches.
//! - All SIMD operates on 64-byte aligned `AlignedVec` buffers.
//!
//! # Source of truth
//! - `NAM/film.h:28-85` (`_cond_to_scale_shift` via Conv1x1 / Dense)
//! - `NeuralAmpModelerCore/NAM/film.cpp` (per-channel modulation)

pub(crate) mod modulation;
pub use modulation::FiLMModulationError;

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;

use crate::arch::simd::*;

/// Configuration for a FiLM layer or operation.
///
/// Corresponds to the `_FiLMParams` struct in C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiLMConfig {
    /// Whether FiLM is active at this location.
    pub active: bool,
    /// Whether to apply both scale and shift (true) or only scale (false).
    pub shift: bool,
    /// Number of groups for grouped convolution in the conditioning submodule.
    pub groups: u32,
}

impl Default for FiLMConfig {
    fn default() -> Self {
        Self {
            active: false,
            shift: true,
            groups: 1,
        }
    }
}

/// Concrete FiLM layer — `_cond_to_scale_shift` + per-channel modulation.
///
/// # Weight layout
/// Row-major `[out_rows][cond_size]` where:
/// - When `groups == 1`: `out_rows = channels * (shift ? 2 : 1)`.
///   First `channels` rows produce `scale`; next `channels` rows produce `shift`.
/// - When `groups > 1`: weights are partitioned into `groups` independent blocks,
///   each of size `out_per_group * (cond_size / groups)`.
///
/// The `scale_shift_buf` always has `channels * 2` elements — when `shift == false`
/// the second half remains zero so the modulation step stays branch-free.
#[derive(Clone)]
#[repr(align(64))]
pub struct FiLMLayer {
    /// FiLM configuration (active, shift, groups).
    pub config: FiLMConfig,
    /// Dimension of the conditioning vector.
    pub cond_size: usize,
    /// Number of feature channels to modulate.
    pub channels: usize,
    /// Row-major weight matrix for `_cond_to_scale_shift`.
    pub weights: AlignedVec<f32>,
    /// Bias vector, indexed by global output channel.
    pub bias: AlignedVec<f32>,
    /// Pre-allocated scratch buffer: `[scale[..channels], shift[..channels]]`.
    scale_shift_buf: AlignedVec<f32>,
    // Set only after a complete, validated host modulation update.
    precomputed_modulation: bool,
}

impl FiLMLayer {
    /// Allocates a FiLM layer with the given topology and copies in the weights.
    ///
    /// `weights` must hold at least `groups * out_per_group * cond_per_group`
    /// elements — the exact extent walked by [`FiLMLayer::process`] via
    /// `get_unchecked` (see `cond_to_scale_shift`). The constructor owns this
    /// invariant so the hot-path reader never depends on caller discipline
    /// (pattern: `A2HeadConv::new_with_kernel`).
    ///
    /// # Buffers
    /// `scale_shift_buf` is pre-allocated here — no heap traffic on the DSP
    /// hot-path.
    ///
    /// # Errors
    /// Returns [`NamErrorCode::InvalidModelTopology`] for `groups == 0`
    /// (division by zero in the grouped reader; defense-in-depth — the JSON
    /// parser already rejects it, but `FiLMConfig` is a plain public struct)
    /// and [`NamErrorCode::WeightCountMismatch`] when `weights` cannot cover
    /// the row layout above (count overflow included).
    pub fn load(
        config: FiLMConfig,
        cond_size: usize,
        channels: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, NamErrorCode> {
        // Mirror of the `cond_to_scale_shift` row layout (truncated per-group
        // divisions included), so the check proves exactly what the hot-path
        // reads — no more, no less.
        let groups = config.groups as usize;
        if groups == 0 {
            return Err(NamErrorCode::InvalidModelTopology);
        }
        let cond_per_group = cond_size / groups;
        let out_rows = if config.shift {
            channels.checked_mul(2)
        } else {
            Some(channels)
        };
        let needed_rows = out_rows
            .map(|rows| rows / groups)
            .and_then(|rows| rows.checked_mul(cond_per_group))
            .and_then(|rows| rows.checked_mul(groups));
        let Some(needed_rows) = needed_rows else {
            return Err(NamErrorCode::WeightCountMismatch);
        };
        if weights.len() < needed_rows {
            return Err(NamErrorCode::WeightCountMismatch);
        }
        let expected_bias = if config.shift { channels * 2 } else { channels };
        let mut bias_padded = bias;
        if bias_padded.len() < expected_bias {
            bias_padded.resize(expected_bias, 0.0);
        }
        Ok(Self {
            config,
            cond_size,
            channels,
            weights: AlignedVec::from_vec(weights)?,
            bias: AlignedVec::from_vec(bias_padded)?,
            scale_shift_buf: AlignedVec::new(channels * 2, 0.0f32)?,
            precomputed_modulation: false,
        })
    }

    /// Processes FiLM modulation over the input buffer.
    ///
    /// 1. `_cond_to_scale_shift`: maps `condition` → `scale[..channels]` +
    ///    `shift[..channels]` via a dense GEMV with AVX2 FMA.
    /// 2. Per-channel modulation: `input[c] = input[c] * scale[c] + shift[c]`.
    ///
    /// With [`Self::set_precomputed_modulation`] active, step 1 is skipped and
    /// the installed scale/shift vectors are reused across samples and blocks.
    ///
    /// # Safety
    /// `condition.len()` must be ≥ `self.cond_size`.
    /// `input.len()` may be ≤ `self.channels` (e.g. activation_post_film on
    /// bottleneck-sized slice when bottleneck < channels).
    /// The CPU must support the crate's architecture baseline: x86-64-v3
    /// (AVX2+FMA) or aarch64 NEON. `.cargo/config.toml` targets x86-64-v3 on
    /// x86, but this method does not runtime-check the host ISA.
    ///
    /// `#[inline]` (relaxed, LLVM-decided) rather than `#[inline(always)]`:
    /// `process_frame_dyn` calls this up to 8× per frame for the grouped-FiLM
    /// A2 topologies, and force-inlining the full `cond_to_scale_shift` +
    /// `apply_modulation` body at every site bloats the per-frame kernel past
    /// the µop-cache, regressing the end-to-end A2-Dynamic FiLM benches by
    /// ~30-45% (measured). The relaxed hint lets LLVM inline where hot and
    /// keep a single out-of-line body where the aggregate call count dominates.
    #[inline]
    pub unsafe fn process(&mut self, input: &mut [f32], condition: &[f32]) {
        debug_assert!(
            condition.len() >= self.cond_size,
            "FiLM process: condition slice length ({}) must be >= cond_size ({})",
            condition.len(),
            self.cond_size
        );
        if !self.precomputed_modulation {
            // SAFETY: `condition.len() >= self.cond_size` is the caller contract
            // above; `cond_to_scale_shift` only reads `cond_size` lanes of
            // `condition` and stays within the `weights`/`bias` buffers whose
            // extents were validated by `load()`. ISA: the x86 build requires a
            // CPU compatible with x86-64-v3 (AVX2+FMA); aarch64 uses baseline
            // NEON. The call does not check runtime CPU support.
            unsafe { self.cond_to_scale_shift(condition) };
        }
        // SAFETY: `apply_modulation` restricts its work to
        // `input.len().min(channels)` and reads only `scale_shift_buf`, which is
        // sized to `channels * 2` at `load()`; it is sound for any `input` slice.
        // ISA: its `_mm256_*` intrinsics are covered by the same x86-64-v3
        // baseline / aarch64 NEON lowering described above.
        unsafe { self.apply_modulation(input) };
    }

    // ── _cond_to_scale_shift ──────────────────────────────────────────

    #[inline(always)]
    unsafe fn cond_to_scale_shift(&mut self, condition: &[f32]) {
        debug_assert!(
            condition.len() >= self.cond_size,
            "FiLM cond_to_scale_shift: condition slice length ({}) must be >= cond_size ({})",
            condition.len(),
            self.cond_size
        );
        let g = self.config.groups as usize;
        let cond_per_group = self.cond_size / g;
        let out_per_group = if self.config.shift {
            self.channels * 2 / g
        } else {
            self.channels / g
        };

        for grp in 0..g {
            // SAFETY: `process` requires `condition.len() >= cond_size` and
            // `grp < g`, so `grp * cond_per_group .. (grp + 1) * cond_per_group`
            // lies within `cond_size` (truncated `cond_per_group` keeps the
            // upper bound ≤ `cond_size`).
            let cond_slice = unsafe {
                condition.get_unchecked(grp * cond_per_group..(grp + 1) * cond_per_group)
            };
            let w_offset = grp * out_per_group * cond_per_group;

            for row in 0..out_per_group {
                // Conv1x1 groups partition all output rows; scale/shift are
                // split only after projection, not within each group.
                let global_out = grp * out_per_group + row;

                // SAFETY: `load()` sizes `bias` to at least `channels` (or
                // `2 * channels` when shifting) and `global_out < out_rows`, so
                // this read is in bounds.
                let mut sum = unsafe { *self.bias.get_unchecked(global_out) };
                let w_start = w_offset + row * cond_per_group;
                // SAFETY: `load()` guarantees `weights.len() >= groups *
                // out_per_group * cond_per_group` (same truncated-division
                // layout), so `w_start + cond_per_group` ≤ that extent for
                // every `grp < g`, `row < out_per_group`.
                let w_row = unsafe {
                    self.weights
                        .get_unchecked(w_start..w_start + cond_per_group)
                };
                // SAFETY: `dot_product_avx2` requires both slices to have the
                // same length; `cond_slice` and `w_row` are both `cond_per_group`
                // elements and live in initialized f32 buffers. Its AVX2+FMA
                // intrinsics are valid under the x86-64-v3 build baseline (or the
                // aarch64 NEON lowering) noted in `process`.
                sum += unsafe { dot_product_avx2(w_row, cond_slice) };
                // SAFETY: `scale_shift_buf` always has `channels * 2` elements
                // and `global_out < out_rows <= 2 * channels`.
                unsafe { *self.scale_shift_buf.get_unchecked_mut(global_out) = sum };
            }
        }

        if !self.config.shift {
            for c in self.channels..self.channels * 2 {
                // SAFETY: `scale_shift_buf` has `channels * 2` elements, so
                // `c < channels * 2` is in bounds.
                unsafe { *self.scale_shift_buf.get_unchecked_mut(c) = 0.0 };
            }
        }
    }

    // ── Per-channel modulation ────────────────────────────────────────

    /// `input[c] *= scale[c]; input[c] += shift[c]` — AVX2, 8-wide.
    ///
    /// Uses [`as_chunks_mut`](core::slice::slice::as_chunks_mut) for the SIMD
    /// block, yielding `&mut [[f32; 8]]` with compile-time guaranteed width.
    /// The remainder tail uses safe indexing — bounds are guaranteed by the
    /// chunk decomposition.
    #[inline(always)]
    unsafe fn apply_modulation(&mut self, input: &mut [f32]) {
        let scale = &self.scale_shift_buf[..self.channels];
        let shift = &self.scale_shift_buf[self.channels..self.channels * 2];
        let limit = input.len().min(self.channels);

        let (in_chunks, in_rem) = input[..limit].as_chunks_mut::<8>();
        let (scale_chunks, _) = scale[..limit].as_chunks::<8>();
        let (shift_chunks, _) = shift[..limit].as_chunks::<8>();

        for i in 0..in_chunks.len() {
            // SAFETY: `in_chunks`, `scale_chunks`, and `shift_chunks` all have
            // `limit / 8` elements of exactly 8 f32 lanes, so each load reads 8
            // initialized values and the store targets the same 8-lane mutable
            // chunk. `loadu`/`storeu` accept unaligned addresses. ISA: callers must
            // use an x86-64-v3-compatible CPU (AVX2+FMA); aarch64 maps to baseline
            // NEON. This function does not runtime-check CPU support.
            unsafe {
                let v_in = _mm256_loadu_ps(in_chunks[i].as_ptr());
                let v_scale = _mm256_loadu_ps(scale_chunks[i].as_ptr());
                let v_shift = _mm256_loadu_ps(shift_chunks[i].as_ptr());
                _mm256_storeu_ps(
                    in_chunks[i].as_mut_ptr(),
                    _mm256_fmadd_ps(v_in, v_scale, v_shift),
                );
            }
        }

        let off = in_chunks.len() * 8;
        for c in 0..in_rem.len() {
            input[off + c] = input[off + c] * scale[off + c] + shift[off + c];
        }
    }
}

// ── FilmBlock: mutable FiLM references for block-level dispatch ──────────

/// Bundle of mutable references to the 8 FiLM insertion points in an A2 layer.
///
/// Used by `layer_forward_ch{3,8}_block` to conditionally apply FiLM
/// at the correct positions in the signal chain without per-point parameter
/// explosion.
///
/// Callers must hoist presence tests out of per-frame loops: each `Option`
/// is fixed at load time, so `film.is_active(mask)` is evaluated once per
/// block and the per-frame path becomes a single predictable `u8` test
/// (see `layer_forward_ch8_block` / `layer_forward_ch3_block`). This keeps
/// the hot loop linear in the I-cache and avoids re-testing 8 pointers
/// per frame.
pub struct FilmBlock<'a> {
    /// FiLM before dilated convolution.
    pub conv_pre_film: Option<&'a mut FiLMLayer>,
    /// FiLM after dilated convolution.
    pub conv_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM before input mixin (same insertion point as conv_post_film).
    pub input_mixin_pre_film: Option<&'a mut FiLMLayer>,
    /// FiLM after input mixin, before activation.
    pub input_mixin_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM before activation (same insertion point as input_mixin_post_film).
    pub activation_pre_film: Option<&'a mut FiLMLayer>,
    /// FiLM after activation.
    pub activation_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM after layer 1x1 residual.
    pub layer1x1_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM after head 1x1 (reserved for future general A2 engine).
    pub head1x1_post_film: Option<&'a mut FiLMLayer>,
}

impl<'a> FilmBlock<'a> {
    /// Creates an empty `FilmBlock` with all fields set to `None`.
    /// Useful for tests and for the fast-path when no FiLM is active.
    pub fn empty() -> Self {
        Self {
            conv_pre_film: None,
            conv_post_film: None,
            input_mixin_pre_film: None,
            input_mixin_post_film: None,
            activation_pre_film: None,
            activation_post_film: None,
            layer1x1_post_film: None,
            head1x1_post_film: None,
        }
    }

    /// Pre-computed per-insertion-point presence mask for one block dispatch.
    ///
    /// Bit assignment (LSB-first): 0 = `conv_post`, 1 = `mixin_pre`,
    /// 2 = `mixin_post`, 3 = `act_pre`, 4 = `act_post`, 5 = `l1x1_post`.
    /// `conv_pre` is intentionally excluded: it runs once per layer in
    /// `layer_forward_dispatch`, before the block kernel is entered.
    ///
    /// The mask is a plain `u8` snapshot evaluated once per block, so the
    /// per-frame loops below test a single predictable integer instead of
    /// 5-6 `Option` discriminants. Bit-exact: presence is load-time fixed,
    /// no allocation, no behaviour change.
    #[inline(always)]
    pub fn active_mask(&self) -> u8 {
        (self.conv_post_film.is_some() as u8)
            | ((self.input_mixin_pre_film.is_some() as u8) << 1)
            | ((self.input_mixin_post_film.is_some() as u8) << 2)
            | ((self.activation_pre_film.is_some() as u8) << 3)
            | ((self.activation_post_film.is_some() as u8) << 4)
            | ((self.layer1x1_post_film.is_some() as u8) << 5)
    }
}

impl super::layer::A2Layer {
    /// Returns a [`FilmBlock`] with mutable references to all 8 FiLM insertion points.
    #[inline]
    pub fn film_block(&mut self) -> FilmBlock<'_> {
        FilmBlock {
            conv_pre_film: self.conv_pre_film.as_mut(),
            conv_post_film: self.conv_post_film.as_mut(),
            input_mixin_pre_film: self.input_mixin_pre_film.as_mut(),
            input_mixin_post_film: self.input_mixin_post_film.as_mut(),
            activation_pre_film: self.activation_pre_film.as_mut(),
            activation_post_film: self.activation_post_film.as_mut(),
            layer1x1_post_film: self.layer1x1_post_film.as_mut(),
            head1x1_post_film: self.head1x1_post_film.as_mut(),
        }
    }
}

// ── AVX2 dot product ───────────────────────────────────────────────────

/// Dot product `sum(a[i] * b[i])` — AVX2+FMA, 8-wide.
///
/// Uses dual accumulator and horizontal reduction matching the
/// `convolve_mono_avx2` pattern.
// KEEP IN SYNC WITH: `src/math/gemm/dot_basic.rs::dot_product_avx2`
//
// Architectural Rationale for Duplication (F-PERF-34):
// - FiLM operates on condition vectors where `cond_per_group` is small (typically 1..8).
// - This specialized kernel uses 2 accumulators (16-wide unroll), naive scalar tail,
//   and `#[inline(always)]` to inline directly into `FiLMLayer::process` without register
//   spilling or call overhead.
// - In contrast, `gemm::dot_basic::dot_product_avx2` targets large GEMM matrices,
//   employing 4 accumulators (32-wide unroll) and Kahan compensated summation in the tail.
// - Parity is verified in `film_test.rs::test_dot_product_avx2_identity_with_gemm`.
/// # Safety
/// `a` and `b` must have equal lengths (the routine reads the full slices),
/// and the CPU must support the crate's x86-64-v3 baseline (AVX2+FMA) or the
/// aarch64 NEON baseline. `.cargo/config.toml` selects x86-64-v3; this function
/// performs no runtime CPU check.
#[inline(always)]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();

    // Tiny-vector fast path: for `len < 8` the AVX2 loops never execute and the
    // horizontal reduction below would sum two zero vectors (a pure-waste fixed
    // cost paid on every grouped-FiLM row). A direct left-to-right scalar
    // accumulation is bit-identical: the reduction of the zero accumulators is
    // exactly `+0.0`, and `0.0 + x == x` for every finite `x` (including `-0.0`
    // products, which normalize to `+0.0` — the same value the reduction yields).
    if len < 8 {
        let mut out = 0.0f32;
        let mut i = 0;
        while i < len {
            // SAFETY: `i < len == a.len()` and callers pass `b` with
            // `b.len() == a.len()`, so both unchecked reads are in bounds.
            out += unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) };
            i += 1;
        }
        return out;
    }

    // SAFETY: the caller's documented x86-64-v3 (AVX2+FMA) / aarch64 NEON
    // precondition guarantees the ISA; this initializes a register only.
    let mut sum0 = unsafe { _mm256_setzero_ps() };
    // SAFETY: same documented ISA precondition; this initializes a register only.
    let mut sum1 = unsafe { _mm256_setzero_ps() };
    let mut i = 0;

    while i + 16 <= len {
        // SAFETY: `i + 16 <= len`, so `i` and `i + 8` are each valid for an
        // 8-f32 unaligned load from both `a` and `b` (equal lengths). ISA:
        // caller must provide x86-64-v3 (AVX2+FMA) or aarch64 NEON; no runtime
        // ISA check is performed here.
        unsafe {
            let ha0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let hb0 = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(ha0, hb0, sum0);

            let ha1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let hb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            sum1 = _mm256_fmadd_ps(ha1, hb1, sum1);
        }

        i += 16;
    }

    while i + 8 <= len {
        // SAFETY: `i + 8 <= len`, so an 8-f32 unaligned load from both `a` and
        // `b` (equal lengths) stays in bounds. ISA: caller must provide
        // x86-64-v3 (AVX2+FMA) or aarch64 NEON; no runtime check is performed.
        unsafe {
            let ha = _mm256_loadu_ps(a.as_ptr().add(i));
            let hb = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(ha, hb, sum0);
        }
        i += 8;
    }

    // SAFETY: the documented x86-64-v3 (AVX2+FMA) / aarch64 NEON precondition
    // guarantees the ISA; this only adds register values and accesses no memory.
    let sum = unsafe { _mm256_add_ps(sum0, sum1) };
    // SAFETY: same documented ISA precondition; this extracts a register lane
    // and accesses no memory.
    let hi128 = unsafe { _mm256_extractf128_ps(sum, 1) };
    // SAFETY: same documented ISA precondition; this is a register-only cast.
    let lo128 = unsafe { _mm256_castps256_ps128(sum) };
    // SAFETY: same documented ISA precondition; this only adds register values.
    let s128 = unsafe { _mm_add_ps(lo128, hi128) };
    // SAFETY: same documented ISA precondition; this shuffles a register only.
    let shuf = unsafe { _mm_movehdup_ps(s128) };
    // SAFETY: same documented ISA precondition; this only adds register values.
    let sums = unsafe { _mm_add_ps(s128, shuf) };
    // SAFETY: same documented ISA precondition; this shuffles register lanes only.
    let shuf2 = unsafe { _mm_movehl_ps(sums, sums) };
    // SAFETY: same documented ISA precondition; this only adds register values.
    let r = unsafe { _mm_add_ss(sums, shuf2) };
    // SAFETY: same documented ISA precondition; this extracts a scalar from a register.
    let mut out = unsafe { _mm_cvtss_f32(r) };

    while i < len {
        // SAFETY: `i < len == a.len()` and `b.len() == a.len()`, so both
        // unchecked tail reads are in bounds.
        out += unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) };
        i += 1;
    }

    out
}

#[cfg(test)]
#[path = "film_test.rs"]
mod tests;
