// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(missing_docs)]

use crate::loader::nam_json::model::NamModelData;

use super::super::*;
use super::static_eval::*;

/// High-precision f64 reference oracle for A2 architecture forward pass.
///
/// Evaluates A2 model arrays sample-by-sample, applying FiLM modulations (Slots 0-7),
/// dilated 1D convolutions, activation/gating functions, and head accumulation.
#[expect(
    clippy::needless_range_loop,
    reason = "Range loop required for explicit SIMD lane indexing not expressible via iterator"
)]
pub(crate) fn oracle_a2_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    let num_frames = input.len();
    if num_frames == 0 {
        return vec![];
    }

    let layers = &model_data.config.layers;
    if layers.is_empty() {
        return vec![0.0; num_frames];
    }

    // Process condition_dsp sub-model to obtain per-frame condition
    // vectors. The sub-model processes the raw input and produces condition_size
    // samples per frame (the head_size of the condition_dsp's last array).
    let cond_output: Option<Vec<f64>> = model_data.config.condition_dsp.as_ref().map(|json| {
        let cond_model: NamModelData =
            serde_json::from_value(json.clone()).expect("Failed to parse condition_dsp JSON");
        oracle_condition_dsp_channels(&cond_model, input, config)
    });

    // Broadcast single-channel condition_dsp output (e.g. LSTM) to condition_size channels.
    let cond_size_oracle = layers.first().and_then(|l| l.condition_size).unwrap_or(1);
    let cond_output: Option<Vec<f64>> = cond_output.map(|raw_out| {
        if cond_size_oracle > 1 && raw_out.len() == num_frames {
            let mut broadcasted = vec![0.0f64; num_frames * cond_size_oracle];
            for f in 0..num_frames {
                let val = raw_out[f];
                for c in 0..cond_size_oracle {
                    broadcasted[f * cond_size_oracle + c] = val;
                }
            }
            broadcasted
        } else {
            raw_out
        }
    });

    let head_scale = model_data.config.head_scale.unwrap_or(1.0) as f64;
    let mut cursor = Cursor::new(&model_data.weights, config.weight_precision);
    let acc_mode = config.accumulation;

    let mut arrays = match build_a2_arrays(model_data, &mut cursor) {
        Some(arrs) => arrs,
        None => return vec![0.0; num_frames],
    };

    let num_arrays = arrays.len();

    // Allocate history buffers per array (largest across arrays).
    let mut max_rf: usize = 0;
    for arr in &arrays {
        let max_dil = arr.lws.iter().map(|lw| lw.dil).max().unwrap_or(1);
        let max_ks_a = arr.lws.iter().map(|lw| lw.ks).max().unwrap_or(6);
        max_rf = max_rf.max((max_ks_a - 1) * max_dil + 64);
    }
    let hist_size = max_rf + num_frames + 64;
    let bs = max_rf;

    for arr in &mut arrays {
        let num_layers = arr.lws.len();
        let ch = arr.ch;
        arr.fwd_bufs = (0..num_layers)
            .map(|_| vec![0.0f64; hist_size * ch])
            .collect();
    }

    // Head accumulator (shared across arrays, per-channel).
    let hr_len = (max_rf + num_frames + 64).next_power_of_two();
    let ring_mask = hr_len - 1;
    let max_ch = arrays.iter().map(|a| a.ch).max().unwrap_or(8);
    let mut head_acc = vec![0.0f64; hr_len * max_ch];
    let mut head_wp = 0usize;

    // Pre-compute channel counts for cascade residual flow.
    let array_channels: Vec<usize> = arrays.iter().map(|a| a.ch).collect();

    // Reserve cascade residual buffer (multi-channel between arrays).
    let mut cascade_residual = vec![0.0f64; hist_size * max_ch];

    let mut output = vec![0.0f64; num_frames];

    #[expect(
        clippy::explicit_counter_loop,
        reason = "Explicit index required to synchronize progress across multiple arrays simultaneously"
    )]
    for (f, out_val) in output.iter_mut().enumerate() {
        let fi = bs + f;
        let x = input[f];
        let head_col = head_wp;
        head_wp += 1;

        // ── Cascade: process each array ──
        for (ai, arr) in arrays.iter_mut().enumerate() {
            let ch = arr.ch;
            let bottleneck = arr.bottleneck;
            let cond_size = arr.cond_size;

            // Condition vector: from condition_dsp or raw input.
            let condition: &[f64] = if cond_size == 1 {
                std::slice::from_ref(&x)
            } else if let Some(ref cond_out) = cond_output {
                let offset = f * cond_size;
                if offset + cond_size <= cond_out.len() {
                    &cond_out[offset..offset + cond_size]
                } else {
                    &[]
                }
            } else {
                &[]
            };

            // Per-array history buffers.
            let num_layers = arr.lws.len();
            let mut head1x1_scratch = if arr.lws.iter().any(|lw| lw.head1x1_active) {
                vec![0.0f64; arr.head_accum_size]
            } else {
                vec![]
            };
            let mut z_scratch = vec![0.0f64; bottleneck * 2];

            // Input to this array: mono for array 0, cascade residual for others.
            let mut layer_in = vec![0.0f64; ch];
            if ai == 0 {
                for c in 0..ch {
                    layer_in[c] = x * arr.rechannel_w[c];
                }
            } else {
                let prev_ch = array_channels[ai - 1];
                let rw = &arr.rechannel_w;
                for nc in 0..ch {
                    let mut sum = 0.0;
                    for ic in 0..prev_ch {
                        sum += cascade_residual[fi * max_ch + ic] * rw[ic * ch + nc];
                    }
                    layer_in[nc] = sum;
                }
            }

            // Per-layer history buffers
            let fwd_bufs = &mut arr.fwd_bufs;

            // Write input to first layer's history
            for c in 0..ch {
                fwd_bufs[0][fi * ch + c] = layer_in[c];
            }

            for (li, lw) in arr.lws.iter_mut().enumerate() {
                let z_out_ch = lw.conv_out;
                let use_gating = lw.gating_mode == GatingModeOracle::Gated;
                let use_blending = lw.gating_mode == GatingModeOracle::Blended;

                // conv_pre_film (slot 0)
                if let Some(ref mut film) = lw.film[0] {
                    film.apply(&mut fwd_bufs[li][fi * ch..fi * ch + ch], condition);
                }

                // Conv1d
                z_scratch.fill(0.0);
                for oc in 0..z_out_ch {
                    let mut sum = lw.conv_b[oc];
                    let wb = oc * ch * lw.ks;
                    for kt in 0..lw.ks {
                        let off = (lw.dil as isize) * ((kt as isize) + 1 - (lw.ks as isize));
                        let ins = ((fi as isize) + off) as usize * ch;
                        for ic in 0..ch {
                            if ins + ic < fwd_bufs[li].len() {
                                sum = mul_add_f64(
                                    fwd_bufs[li][ins + ic],
                                    lw.conv_w[wb + ic * lw.ks + kt],
                                    sum,
                                    acc_mode,
                                );
                            }
                        }
                    }
                    z_scratch[oc] = sum;
                }

                // conv_post_film (slot 1)
                if let Some(ref mut film) = lw.film[1] {
                    film.apply(&mut z_scratch[..z_out_ch], condition);
                }

                // Mixin — input_mixin_pre_film (slot 2) applied to condition
                let condition_mod = if lw.film[2].is_some() {
                    let mut cond_copy = condition.to_vec();
                    lw.film[2]
                        .as_mut()
                        .unwrap()
                        .apply(&mut cond_copy, condition);
                    cond_copy
                } else {
                    condition.to_vec()
                };
                let mut mixin_contrib = vec![0.0f64; z_out_ch];
                if !condition_mod.is_empty() {
                    if lw.mixin_groups <= 1 {
                        for c in 0..z_out_ch {
                            let mut sum = 0.0;
                            for k in 0..cond_size.min(condition_mod.len()) {
                                sum += lw.mixin_w[c * cond_size + k] * condition_mod[k];
                            }
                            mixin_contrib[c] = sum;
                        }
                    } else {
                        let in_pg = cond_size / lw.mixin_groups as usize;
                        let out_per_g = z_out_ch / lw.mixin_groups as usize;
                        for g in 0..lw.mixin_groups as usize {
                            let in_start = g * in_pg;
                            let out_start = g * out_per_g;
                            for oc in out_start..out_start + out_per_g {
                                let mut sum = 0.0;
                                let w_base = oc * in_pg;
                                for ic in 0..in_pg {
                                    if in_start + ic < condition_mod.len() {
                                        sum +=
                                            lw.mixin_w[w_base + ic] * condition_mod[in_start + ic];
                                    }
                                }
                                mixin_contrib[oc] = sum;
                            }
                        }
                    }
                }

                // input_mixin_post_film (slot 3)
                if let Some(ref mut film) = lw.film[3] {
                    film.apply(&mut mixin_contrib[..z_out_ch], condition);
                }

                // Sum mixin output to z_scratch
                for c in 0..z_out_ch {
                    z_scratch[c] += mixin_contrib[c];
                }

                // activation_pre_film (slot 4)
                if let Some(ref mut film) = lw.film[4] {
                    film.apply(&mut z_scratch[..z_out_ch], condition);
                }

                // Activation or Gating/Blending
                let z_len = if use_gating {
                    let half = bottleneck;
                    lw.activation
                        .apply(&mut z_scratch[..half], config.activation);
                    lw.secondary_activation
                        .apply(&mut z_scratch[half..half * 2], config.activation);
                    for i in 0..half {
                        z_scratch[i] *= z_scratch[half + i];
                    }
                    half
                } else if use_blending {
                    let half = bottleneck;
                    let mut original = vec![0.0f64; half];
                    original.copy_from_slice(&z_scratch[..half]);
                    lw.activation
                        .apply(&mut z_scratch[..half], config.activation);
                    lw.secondary_activation
                        .apply(&mut z_scratch[half..half * 2], config.activation);
                    for i in 0..half {
                        let alpha = z_scratch[half + i];
                        z_scratch[i] = original[i] + alpha * (z_scratch[i] - original[i]);
                    }
                    half
                } else {
                    lw.activation
                        .apply(&mut z_scratch[..bottleneck], config.activation);
                    bottleneck
                };

                // activation_post_film (slot 5)
                if let Some(ref mut film) = lw.film[5] {
                    film.apply(&mut z_scratch[..z_len], condition);
                }

                // Head accumulate
                let head_off = head_col * max_ch;
                if lw.head1x1_active {
                    let h1_in = if lw.head1x1_w.is_empty() {
                        0
                    } else {
                        lw.head1x1_w.len() / arr.head_accum_size
                    };
                    let h1_groups = bottleneck.checked_div(h1_in).unwrap_or(1);
                    let ch_per_group = arr.head_accum_size / h1_groups;
                    head1x1_scratch.fill(0.0);
                    for grp in 0..h1_groups {
                        for oc in grp * ch_per_group..(grp + 1) * ch_per_group {
                            let mut sum = lw.head1x1_b[oc];
                            for ic in 0..h1_in {
                                sum = mul_add_f64(
                                    z_scratch[grp * h1_in + ic],
                                    lw.head1x1_w[oc * h1_in + ic],
                                    sum,
                                    acc_mode,
                                );
                            }
                            head1x1_scratch[oc] = sum;
                        }
                    }
                    if let Some(ref mut film) = lw.film[7] {
                        film.apply(&mut head1x1_scratch, condition);
                    }
                    if li == 0 && ai == 0 {
                        head_acc[head_off..head_off + arr.head_accum_size]
                            .copy_from_slice(&head1x1_scratch[..arr.head_accum_size]);
                    } else {
                        for c in 0..arr.head_accum_size {
                            head_acc[head_off + c] =
                                accum_f64(head_acc[head_off + c], head1x1_scratch[c], acc_mode);
                        }
                    }
                } else {
                    if li == 0 && ai == 0 {
                        head_acc[head_off..head_off + z_len].copy_from_slice(&z_scratch[..z_len]);
                    } else {
                        for c in 0..z_len {
                            head_acc[head_off + c] =
                                accum_f64(head_acc[head_off + c], z_scratch[c], acc_mode);
                        }
                    }
                }

                // L1x1 residual
                if li < num_layers - 1 {
                    let mut l1x1_contrib = vec![0.0f64; ch];
                    if lw.l1x1_groups <= 1 {
                        for oc in 0..ch {
                            let mut sum = lw.l1x1_b[oc];
                            for ic in 0..bottleneck {
                                sum = mul_add_f64(
                                    z_scratch[ic],
                                    lw.l1x1_w[oc * bottleneck + ic],
                                    sum,
                                    acc_mode,
                                );
                            }
                            l1x1_contrib[oc] = sum;
                        }
                    } else {
                        let in_pg = bottleneck / lw.l1x1_groups as usize;
                        let out_per_g = ch / lw.l1x1_groups as usize;
                        for g in 0..lw.l1x1_groups as usize {
                            let in_start = g * in_pg;
                            let out_start = g * out_per_g;
                            for oc in out_start..out_start + out_per_g {
                                let mut sum = lw.l1x1_b[oc];
                                let w_base = oc * in_pg;
                                for ic in 0..in_pg {
                                    sum = mul_add_f64(
                                        z_scratch[in_start + ic],
                                        lw.l1x1_w[w_base + ic],
                                        sum,
                                        acc_mode,
                                    );
                                }
                                l1x1_contrib[oc] = sum;
                            }
                        }
                    }
                    if use_blending && lw.film[6].is_some() {
                        let film = lw.film[6].as_mut().unwrap();
                        film.apply(&mut l1x1_contrib, condition);
                    }
                    let mut next = vec![0.0f64; ch];
                    for oc in 0..ch {
                        next[oc] = accum_f64(layer_in[oc], l1x1_contrib[oc], acc_mode);
                    }
                    for c in 0..ch {
                        fwd_bufs[li + 1][fi * ch + c] = next[c];
                    }
                    layer_in = next;
                }
            }

            // Save residual for next array (cascade_input reads from cascade_residual).
            if ai + 1 < num_arrays {
                for c in 0..ch {
                    cascade_residual[fi * max_ch + c] = layer_in[c];
                }
            }
        }

        // ── Head finalize (last array only) ──
        let last_arr = &arrays[num_arrays - 1];
        let lch = last_arr.head_accum_size;
        let k = if last_arr.head_is_rechannel {
            last_arr.head_size
        } else {
            if last_arr.head_size == 1 {
                last_arr.head_kernel_size
            } else {
                last_arr.head_size
            }
        };
        let cb = head_col.wrapping_sub(k - 1);
        let mut y = last_arr.head_b[0];
        for t in 0..k {
            let col = cb.wrapping_add(t) & ring_mask;
            let so = col * max_ch;
            let wo = t * lch;
            for c in 0..last_arr.head_accum_size {
                y = mul_add_f64(last_arr.head_w[wo + c], head_acc[so + c], y, acc_mode);
            }
        }
        *out_val = y * head_scale;
    }

    output
}
