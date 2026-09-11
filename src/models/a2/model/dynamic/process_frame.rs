// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 Dynamic model — per-frame inner processing kernel.

use crate::math::common::SimdMath;
use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::{BlendingActivationConfig, GatingActivationConfig};
use crate::models::a2::layer::A2Layer;

use core::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};

/// Per-frame inner core: conv, FiLM, mixin, activation/gating/blending,
/// head accumulation, and l1x1 residual for a single frame in one layer.
///
/// `M` is the ISA monomorphization type propagated from the top-level
/// `dispatch_simd!` in [`WaveNetA2Dyn::process`](super::WaveNetA2Dyn::process).
#[expect(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "Audio DSP kernel with many dimension parameters and explicit SIMD indexing — struct consolidation would add indirection overhead in the hot path"
)]
#[inline(always)]
pub(crate) unsafe fn process_frame_dyn<M: SimdMath>(
    layer: &mut A2Layer,
    history: &[f32],
    f: usize,
    max_lookback_cols: usize,
    head_wp: usize,
    z_out_ch: usize,
    use_gating: bool,
    use_blending: bool,
    is_first: bool,
    is_last: bool,
    channels: usize,
    head_accum_size: usize,
    bottleneck: usize,
    z_scratch: &mut [f32],
    mixin_scratch: &mut [f32],
    l1x1_scratch: &mut [f32],
    head_accum: &mut [f32],
    layer_in: &mut [f32],
    head1x1_scratch: &mut [f32],
    cond_scratch: &mut [f32],
    gating_config: Option<&GatingActivationConfig>,
    blending_config: Option<&mut BlendingActivationConfig>,
    activation: &ActivationType,
    cond_buf: &[f32],
    cond_size: usize,
) {
    let frame_idx = max_lookback_cols + f;
    let cond_slice = &cond_buf[f * cond_size..(f + 1) * cond_size];

    #[expect(
        unused_assignments,
        reason = "Variable assigned for clarity but value consumed by debug_assert only in release builds"
    )]
    let mut z_len = z_out_ch;

    // 1. Dilated conv → z_scratch.
    // SAFETY: `process_single_frame::<M>` is an `unsafe fn`; its preconditions hold:
    // `history` spans `bs - lookback .. bs + nf*channels` (buffer start already
    // advanced with wrap), `z_scratch[..z_out_ch]` has length ≥ the conv's out channels,
    // `frame_idx = max_lookback_cols + f` allows the kernel lookback taps, and `M`
    // matches the CPU ISA (top-level `dispatch_simd!`).
    unsafe {
        layer
            .conv
            .process_single_frame::<M>(history, &mut z_scratch[..z_out_ch], frame_idx, None);
    }

    // FiLM post-conv + pre-mixin.
    if let Some(ref mut film) = layer.conv_post_film {
        // SAFETY: `cond_slice` has length exactly `cond_size` (matching this FiLM
        // layer's `cond_size`) and `z_scratch[..z_out_ch]` is a valid in-bounds
        // sub-slice of length ≤ `channels`; both satisfy `film.process`'s
        // documented preconditions.
        unsafe {
            film.process(&mut z_scratch[..z_out_ch], cond_slice);
        }
    }

    // 2. Input mixin — matrix-vector multiply.
    // When input_mixin_pre_film is active, the condition vector is first
    // modulated by FiLM (self-modulation, C++ model.cpp:188-197), then the
    // modulated condition feeds the mixin instead of the raw condition.
    //
    // Weights are stored col-major [in_pg][out_per_g] per group (col-major
    // transposition in builder.rs). Each condition channel broadcasts
    // into 8-wide SIMD FMA over contiguous output weights.
    {
        let mut cond_is_modulated = false;
        if let Some(ref mut film) = layer.input_mixin_pre_film {
            debug_assert!(
                cond_size <= cond_scratch.len(),
                "cond_size ({cond_size}) exceeds cond_scratch capacity ({})",
                cond_scratch.len()
            );
            cond_scratch[..cond_size].copy_from_slice(cond_slice);
            // SAFETY: `cond_scratch[..cond_size]` is in-bounds by the `debug_assert!`
            // above (`cond_size <= cond_scratch.len()`), `cond_slice` has length
            // `cond_size` matching this FiLM layer's `cond_size`, and the input slice
            // is a valid sub-slice of length ≤ `channels`; all satisfy `film.process`'s
            // documented preconditions.
            unsafe {
                film.process(&mut cond_scratch[..cond_size], cond_slice);
            }
            cond_is_modulated = true;
        }
        let cond_for_mixin: &[f32] = if cond_is_modulated {
            &cond_scratch[..cond_size]
        } else {
            cond_slice
        };

        let in_pg = if layer.mixin_groups <= 1 {
            cond_size
        } else {
            cond_size / layer.mixin_groups as usize
        };
        let out_per_g = if layer.mixin_groups <= 1 {
            z_out_ch
        } else {
            z_out_ch / layer.mixin_groups as usize
        };
        let num_groups = layer.mixin_groups.max(1) as usize;

        if out_per_g >= 8 {
            for g in 0..num_groups {
                let group_base = g * out_per_g * in_pg;
                let in_start = g * in_pg;
                let out_start = g * out_per_g;
                // SAFETY: `while oc + 8 <= out_per_g` keeps the 8-lane `loadu`/`storeu`
                // at `group_base + ic*out_per_g + oc` within `mixin_w` (len
                // `num_groups*out_per_g*in_pg`) and at `out_start + oc` within
                // `mixin_scratch` (len ≥ `z_out_ch`); `loadu`/`storeu` need no alignment.
                unsafe {
                    let mut oc = 0;
                    while oc + 8 <= out_per_g {
                        let mut acc = _mm256_setzero_ps();
                        for ic in 0..in_pg {
                            let cond = _mm256_set1_ps(cond_for_mixin[in_start + ic]);
                            let w = _mm256_loadu_ps(
                                layer.mixin_w.as_ptr().add(group_base + ic * out_per_g + oc),
                            );
                            acc = _mm256_fmadd_ps(cond, w, acc);
                        }
                        _mm256_storeu_ps(mixin_scratch.as_mut_ptr().add(out_start + oc), acc);
                        oc += 8;
                    }
                    // Scalar tail for remaining output channels in this group.
                    for oc in oc..out_per_g {
                        let mut sum = 0.0;
                        for ic in 0..in_pg {
                            sum += layer.mixin_w[group_base + ic * out_per_g + oc]
                                * cond_for_mixin[in_start + ic];
                        }
                        mixin_scratch[out_start + oc] = sum;
                    }
                }
            }
        } else {
            // Scalar fallback for small groups (out_per_g < 8).
            for g in 0..num_groups {
                let group_base = g * out_per_g * in_pg;
                let in_start = g * in_pg;
                let out_start = g * out_per_g;
                for oc in 0..out_per_g {
                    let mut sum = 0.0;
                    for ic in 0..in_pg {
                        sum += layer.mixin_w[group_base + ic * out_per_g + oc]
                            * cond_for_mixin[in_start + ic];
                    }
                    mixin_scratch[out_start + oc] = sum;
                }
            }
        }
    }

    // FiLM post-mixin + pre-activation.
    // Apply FiLM on the isolated mixin buffer before summing.
    if let Some(ref mut film) = layer.input_mixin_post_film {
        // SAFETY: `cond_slice` has length exactly `cond_size` (matching this FiLM
        // layer's `cond_size`) and `mixin_scratch[..z_out_ch]` is a valid in-bounds
        // sub-slice of length ≤ `channels`; both satisfy `film.process`'s
        // documented preconditions.
        unsafe {
            film.process(&mut mixin_scratch[..z_out_ch], cond_slice);
        }
    }

    // Sum mixin output to z_scratch (vectorized 8-wide).
    if z_out_ch >= 8 {
        // SAFETY: `while c + 8 <= z_out_ch` keeps the 8-lane `loadu`/`storeu` at
        // offset `c` within both `mixin_scratch` and `z_scratch` (len ≥ `z_out_ch`);
        // `loadu`/`storeu` need no alignment.
        unsafe {
            let mut c = 0;
            while c + 8 <= z_out_ch {
                let src = _mm256_loadu_ps(mixin_scratch.as_ptr().add(c));
                let dst = _mm256_loadu_ps(z_scratch.as_ptr().add(c));
                _mm256_storeu_ps(z_scratch.as_mut_ptr().add(c), _mm256_add_ps(dst, src));
                c += 8;
            }
            for c in c..z_out_ch {
                z_scratch[c] += mixin_scratch[c];
            }
        }
    } else {
        for c in 0..z_out_ch {
            z_scratch[c] += mixin_scratch[c];
        }
    }

    if let Some(ref mut film) = layer.activation_pre_film {
        // SAFETY: `cond_slice` has length exactly `cond_size` (matching this FiLM
        // layer's `cond_size`) and `z_scratch[..z_out_ch]` is a valid in-bounds
        // sub-slice of length ≤ `channels`; both satisfy `film.process`'s
        // documented preconditions.
        unsafe {
            film.process(&mut z_scratch[..z_out_ch], cond_slice);
        }
    }

    // 3. Activation or Gating/Blending.
    if use_gating {
        if let Some(gc) = gating_config {
            // SAFETY: `M` matches the CPU ISA (top-level `dispatch_simd!`) and
            // `z_scratch[..z_out_ch]` is a valid in-bounds slice; gating's
            // `debug_assert!` requires an even length, which holds because
            // `z_out_ch = bottleneck * 2` when gating is active.
            unsafe {
                gc.apply_gating_simd::<M>(&mut z_scratch[..z_out_ch]);
            }
        }
        z_len = bottleneck;
    } else if use_blending {
        if let Some(bc) = blending_config {
            // SAFETY: `M` matches the CPU ISA (top-level `dispatch_simd!`) and
            // `z_scratch[..z_out_ch]` is a valid in-bounds slice; blending's
            // `debug_assert!`s require an even length and pre-allocated scratch,
            // which hold because `z_out_ch = bottleneck * 2` when blending is active.
            unsafe {
                bc.apply_blending_simd::<M>(&mut z_scratch[..z_out_ch]);
            }
        }
        z_len = bottleneck;
    } else {
        // SAFETY: `M` matches the CPU ISA (top-level `dispatch_simd!`) and
        // `z_scratch[..bottleneck]` is a valid in-bounds slice (documented
        // precondition of `apply_simd`).
        unsafe {
            activation.apply_simd::<M>(&mut z_scratch[..bottleneck]);
        }
        z_len = bottleneck;
    }

    // FiLM post-activation.
    if let Some(ref mut film) = layer.activation_post_film {
        // SAFETY: `cond_slice` has length exactly `cond_size` (matching this FiLM
        // layer's `cond_size`) and `z_scratch[..z_len]` is a valid in-bounds
        // sub-slice of length ≤ `channels`; both satisfy `film.process`'s
        // documented preconditions.
        unsafe {
            film.process(&mut z_scratch[..z_len], cond_slice);
        }
    }

    let head1x1_active = layer.head1x1_active;
    let head1x1_w = &layer.head1x1_w;
    let head1x1_b = &layer.head1x1_b;
    let head_off = (head_wp + f) * head_accum_size;
    if head1x1_active {
        // head1x1_w is [head_accum_size][h1_in] row-major (transposed in build.rs).
        let h1_in = if head1x1_w.is_empty() {
            0
        } else {
            head1x1_w.len() / head_accum_size
        };
        let h1_groups = bottleneck.checked_div(h1_in).unwrap_or(1);
        let ch_per_group = head_accum_size / h1_groups;
        for grp in 0..h1_groups {
            let z_off = grp * h1_in;
            let ch_start = grp * ch_per_group;
            let ch_end = (grp + 1) * ch_per_group;
            // Vectorized inner dot product for each output channel.
            // Processes h1_in in 8-wide SIMD steps, extracting lanes
            // sequentially to preserve exact left-to-right accumulation.
            if h1_in >= 8 {
                for oc in ch_start..ch_end {
                    // SAFETY: `while ic + 8 <= h1_in` keeps the 8-lane loads at
                    // `z_off + ic` within `z_scratch` (`z_off + h1_in <= bottleneck`)
                    // and at `oc*h1_in + ic` within `head1x1_w` (len
                    // `head_accum_size*h1_in`); `loadu`/`storeu` need no alignment.
                    unsafe {
                        let mut acc = _mm256_setzero_ps();
                        let mut ic = 0;
                        while ic + 8 <= h1_in {
                            let inputs = _mm256_loadu_ps(z_scratch.as_ptr().add(z_off + ic));
                            let weights = _mm256_loadu_ps(head1x1_w.as_ptr().add(oc * h1_in + ic));
                            acc = _mm256_fmadd_ps(inputs, weights, acc);
                            ic += 8;
                        }
                        // Extract lanes preserving left-to-right accumulation order.
                        let mut sum = head1x1_b[oc];
                        {
                            let mut lane_buf = [0.0f32; 8];
                            _mm256_storeu_ps(lane_buf.as_mut_ptr(), acc);
                            for v in &lane_buf {
                                sum += *v;
                            }
                        }
                        // Scalar tail for remaining h1_in.
                        for ic in ic..h1_in {
                            sum += head1x1_w[oc * h1_in + ic] * z_scratch[z_off + ic];
                        }
                        head1x1_scratch[oc] = sum;
                    }
                }
            } else {
                for oc in ch_start..ch_end {
                    let mut sum = head1x1_b[oc];
                    let b_start = oc * h1_in;
                    for ic in 0..h1_in {
                        sum += head1x1_w[b_start + ic] * z_scratch[z_off + ic];
                    }
                    head1x1_scratch[oc] = sum;
                }
            }
        }
        // FiLM after head1x1 projection (C++ model.cpp:283-287).
        if let Some(ref mut film) = layer.head1x1_post_film {
            // SAFETY: `cond_slice` has length exactly `cond_size` (matching this FiLM
            // layer's `cond_size`) and `head1x1_scratch[..head_accum_size]` is a valid
            // in-bounds sub-slice of length ≤ `channels`; both satisfy `film.process`'s
            // documented preconditions.
            unsafe {
                film.process(&mut head1x1_scratch[..head_accum_size], cond_slice);
            }
        }
        if is_first {
            head_accum[head_off..head_off + head_accum_size]
                .copy_from_slice(&head1x1_scratch[..head_accum_size]);
        } else {
            // Vectorized accumulation into head ring buffer.
            // SAFETY: `while c + 8 <= head_accum_size` keeps the 8-lane `loadu`/`storeu`
            // at `head_off + c` within `head_accum` — `head_off = (head_wp + f)*head_accum_size`
            // and `advance_head_ring` guarantees `head_wp + nf <= head_cap`, so
            // `head_off + head_accum_size <= head_accum.len()`; `loadu`/`storeu` need no alignment.
            unsafe {
                let mut c = 0;
                while c + 8 <= head_accum_size {
                    let src = _mm256_loadu_ps(head1x1_scratch.as_ptr().add(c));
                    let dst = _mm256_loadu_ps(head_accum.as_ptr().add(head_off + c));
                    _mm256_storeu_ps(
                        head_accum.as_mut_ptr().add(head_off + c),
                        _mm256_add_ps(dst, src),
                    );
                    c += 8;
                }
                for c in c..head_accum_size {
                    head_accum[head_off + c] += head1x1_scratch[c];
                }
            }
        }
    } else {
        debug_assert_eq!(
            bottleneck, head_accum_size,
            "head1x1 must be active when bottleneck != head_accum_size"
        );
        if is_first {
            head_accum[head_off..head_off + bottleneck].copy_from_slice(&z_scratch[..bottleneck]);
        } else {
            // SAFETY: `while c + 8 <= bottleneck` keeps the 8-lane `loadu`/`storeu`
            // at `head_off + c` within `head_accum` — `head_off = (head_wp + f)*head_accum_size`
            // and `advance_head_ring` guarantees `head_wp + nf <= head_cap`; here
            // `bottleneck == head_accum_size` (debug_assert_eq above), so
            // `head_off + bottleneck <= head_accum.len()`; `loadu`/`storeu` need no alignment.
            unsafe {
                let mut c = 0;
                while c + 8 <= bottleneck {
                    let src = _mm256_loadu_ps(z_scratch.as_ptr().add(c));
                    let dst = _mm256_loadu_ps(head_accum.as_ptr().add(head_off + c));
                    _mm256_storeu_ps(
                        head_accum.as_mut_ptr().add(head_off + c),
                        _mm256_add_ps(dst, src),
                    );
                    c += 8;
                }
                for c in c..bottleneck {
                    head_accum[head_off + c] += z_scratch[c];
                }
            }
        }
    }

    // 5. L1x1 residual (skip on last layer).
    if !is_last {
        let base = f * channels;
        let l1x1_w = &layer.l1x1_w;
        let l1x1_b = &layer.l1x1_b;
        if layer.l1x1_groups <= 1 {
            // Dense L1x1: weights are col-major [bottleneck][channels].
            // Each ic row has `channels` contiguous weights, enabling
            // 8-wide SIMD across output channels with broadcast input.
            if channels >= 8 {
                let channels_aligned = channels & !7;
                // SAFETY: `channels_aligned = channels & !7` is a multiple of 8, so
                // each `step_by(8)` iteration keeps `oc + 8 <= channels`; the 8-lane
                // `loadu`/`storeu` at `oc`, `ic*channels + oc`, and `l1x1_scratch`
                // offset `oc` all stay within buffers of length ≥ `channels`;
                // `loadu`/`storeu` need no alignment.
                unsafe {
                    for oc in (0..channels_aligned).step_by(8) {
                        let mut acc = _mm256_loadu_ps(l1x1_b.as_ptr().add(oc));
                        for ic in 0..bottleneck {
                            let z = _mm256_set1_ps(z_scratch[ic]);
                            let w = _mm256_loadu_ps(l1x1_w.as_ptr().add(ic * channels + oc));
                            acc = _mm256_fmadd_ps(z, w, acc);
                        }
                        _mm256_storeu_ps(l1x1_scratch.as_mut_ptr().add(oc), acc);
                    }
                }
                // Scalar tail.
                for oc in channels_aligned..channels {
                    let mut sum = l1x1_b[oc];
                    for ic in 0..bottleneck {
                        sum += l1x1_w[ic * channels + oc] * z_scratch[ic];
                    }
                    l1x1_scratch[oc] = sum;
                }
            } else {
                for oc in 0..channels {
                    let mut sum = l1x1_b[oc];
                    for ic in 0..bottleneck {
                        sum += l1x1_w[ic * channels + oc] * z_scratch[ic];
                    }
                    l1x1_scratch[oc] = sum;
                }
            }
        } else {
            let in_pg = bottleneck / layer.l1x1_groups as usize;
            let out_per_g = channels / layer.l1x1_groups as usize;
            // Grouped L1x1: weights are row-major [channels][in_pg].
            // Vectorize inner dot product over in_pg dimension.
            if in_pg >= 8 {
                for g in 0..layer.l1x1_groups as usize {
                    let in_start = g * in_pg;
                    let out_start = g * out_per_g;
                    for oc in out_start..out_start + out_per_g {
                        // SAFETY: `while ic + 8 <= in_pg` keeps the 8-lane loads at
                        // `in_start + ic` within `z_scratch` (`in_start + in_pg <= bottleneck`)
                        // and at `oc*in_pg + ic` within `l1x1_w` (row-major len
                        // `channels*in_pg`); `loadu`/`storeu` need no alignment.
                        unsafe {
                            let mut acc = _mm256_setzero_ps();
                            let mut ic = 0;
                            while ic + 8 <= in_pg {
                                let inputs = _mm256_loadu_ps(z_scratch.as_ptr().add(in_start + ic));
                                let weights = _mm256_loadu_ps(l1x1_w.as_ptr().add(oc * in_pg + ic));
                                acc = _mm256_fmadd_ps(inputs, weights, acc);
                                ic += 8;
                            }
                            let mut sum = l1x1_b[oc];
                            {
                                let mut lane_buf = [0.0f32; 8];
                                _mm256_storeu_ps(lane_buf.as_mut_ptr(), acc);
                                for v in &lane_buf {
                                    sum += *v;
                                }
                            }
                            for ic in ic..in_pg {
                                sum += l1x1_w[oc * in_pg + ic] * z_scratch[in_start + ic];
                            }
                            l1x1_scratch[oc] = sum;
                        }
                    }
                }
            } else {
                for g in 0..layer.l1x1_groups as usize {
                    let in_start = g * in_pg;
                    let out_start = g * out_per_g;
                    for oc in out_start..out_start + out_per_g {
                        let mut sum = l1x1_b[oc];
                        let w_base = oc * in_pg;
                        for ic in 0..in_pg {
                            sum += l1x1_w[w_base + ic] * z_scratch[in_start + ic];
                        }
                        l1x1_scratch[oc] = sum;
                    }
                }
            }
        }
        if let Some(ref mut film) = layer.layer1x1_post_film.as_mut().filter(|_| use_blending) {
            // SAFETY: `cond_slice` has length exactly `cond_size` (matching this FiLM
            // layer's `cond_size`) and `l1x1_scratch[..channels]` is a valid in-bounds
            // sub-slice of length ≤ `channels`; both satisfy `film.process`'s
            // documented preconditions.
            unsafe {
                film.process(&mut l1x1_scratch[..channels], cond_slice);
            }
        }
        // Vectorized accumulation into layer_in.
        if channels >= 8 {
            // SAFETY: `while oc + 8 <= channels` keeps the 8-lane `loadu`/`storeu` at
            // `base + oc` within `layer_in` (`base = f*channels` with `f < nf`, capacity
            // ≥ `nf*channels`) and at `oc` within `l1x1_scratch` (len ≥ `channels`);
            // `loadu`/`storeu` need no alignment.
            unsafe {
                let mut oc = 0;
                while oc + 8 <= channels {
                    let src = _mm256_loadu_ps(l1x1_scratch.as_ptr().add(oc));
                    let dst = _mm256_loadu_ps(layer_in.as_ptr().add(base + oc));
                    _mm256_storeu_ps(
                        layer_in.as_mut_ptr().add(base + oc),
                        _mm256_add_ps(dst, src),
                    );
                    oc += 8;
                }
                for oc in oc..channels {
                    layer_in[base + oc] += l1x1_scratch[oc];
                }
            }
        } else {
            for oc in 0..channels {
                layer_in[base + oc] += l1x1_scratch[oc];
            }
        }
    }
}
