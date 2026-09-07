// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(missing_docs)]

use crate::loader::nam_json::model::NamModelData;

use super::super::*;
use super::static_eval::*;

/// High-precision f64 reference oracle for A2 architecture forward pass (mono output).
///
/// Evaluates A2 model arrays sample-by-sample, applying FiLM modulations (Slots 0-7),
/// dilated 1D convolutions, activation/gating functions, and head accumulation.
pub(crate) fn oracle_a2_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    oracle_a2_forward_internal(model_data, input, config, false)
}

/// High-precision f64 reference oracle for A2 architecture returning all head channels interleaved.
///
/// Returns `[f0_ch0, ..., f0_chN, f1_ch0, ...]` matching the layout used by
/// the Rust production engine (`condition_dsp_output`) and C++ NAMcore (`_condition_dsp_output_buffers`).
pub(crate) fn oracle_a2_all_channels(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    oracle_a2_forward_internal(model_data, input, config, true)
}

#[expect(
    clippy::needless_range_loop,
    reason = "Range loop required for explicit SIMD lane indexing not expressible via iterator"
)]
fn oracle_a2_forward_internal(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
    all_channels: bool,
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

    // Broadcast condition_dsp output if dsp_ch < cond_size_oracle.
    let cond_size_oracle = layers.first().and_then(|l| l.condition_size).unwrap_or(1);
    let cond_output: Option<Vec<f64>> = cond_output.map(|raw_out| {
        let dsp_ch = raw_out.len().checked_div(num_frames).unwrap_or(1);
        if cond_size_oracle > dsp_ch && num_frames > 0 {
            let mut broadcasted = vec![0.0f64; num_frames * cond_size_oracle];
            for f in 0..num_frames {
                for c in 0..cond_size_oracle {
                    broadcasted[f * cond_size_oracle + c] = raw_out[f * dsp_ch + (c % dsp_ch)];
                }
            }
            broadcasted
        } else {
            raw_out
        }
    });

    let mut cursor = Cursor::new(&model_data.weights, config.weight_precision);
    let acc_mode = config.accumulation;

    let mut arrays = match build_a2_arrays(model_data, &mut cursor) {
        Some(arrs) => arrs,
        None => {
            let out_channels = if all_channels {
                layers.last().and_then(|l| l.head_size).unwrap_or(1)
            } else {
                1
            };
            return vec![0.0; num_frames * out_channels];
        }
    };

    let cascade_head_scale = if cursor.remaining() == 1 {
        cursor.read_one_f64()
    } else {
        1.0f64
    };

    let num_arrays = arrays.len();
    if num_arrays == 0 {
        return vec![0.0; if all_channels { 0 } else { num_frames }];
    }

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

    // Head accumulators (dedicated per array, sized to array's head_accum_size).
    let hr_len = (max_rf + num_frames + 64).next_power_of_two();
    let ring_mask = hr_len - 1;
    let mut head_accs: Vec<Vec<f64>> = arrays
        .iter()
        .map(|a| vec![0.0f64; hr_len * a.head_accum_size])
        .collect();

    // Pre-compute channel counts for cascade residual flow.
    let array_channels: Vec<usize> = arrays.iter().map(|a| a.ch).collect();
    let max_ch = array_channels.iter().copied().max().unwrap_or(8);

    // Reserve cascade residual buffer (multi-channel between arrays).
    let mut cascade_residual = vec![0.0f64; hist_size * max_ch];

    let last_head_size = arrays[num_arrays - 1].head_size;
    let out_channels = if all_channels { last_head_size } else { 1 };
    let mut output = vec![0.0f64; num_frames * out_channels];

    for f in 0..num_frames {
        let fi = bs + f;
        let x = input[f];
        let head_col = f;

        let mut prev_head_out: Vec<f64> = Vec::new();

        // ── Cascade: process each array ──
        for ai in 0..num_arrays {
            let arr = &mut arrays[ai];
            let ch = arr.ch;
            let bottleneck = arr.bottleneck;
            let cond_size = arr.cond_size;
            let head_accum_size = arr.head_accum_size;

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
                vec![0.0f64; head_accum_size]
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
                        sum += cascade_residual[fi * max_ch + ic] * rw[nc * prev_ch + ic];
                    }
                    layer_in[nc] = sum;
                }
            }

            // Write input to first layer's history
            let fwd_bufs = &mut arr.fwd_bufs;
            for c in 0..ch {
                fwd_bufs[0][fi * ch + c] = layer_in[c];
            }

            // Seed head accumulator for ai > 0 from prev_head_out
            let head_off = (head_col & ring_mask) * head_accum_size;
            if ai > 0 {
                let copy_ch = prev_head_out.len().min(head_accum_size);
                head_accs[ai][head_off..head_off + copy_ch]
                    .copy_from_slice(&prev_head_out[..copy_ch]);
                head_accs[ai][head_off + copy_ch..head_off + head_accum_size].fill(0.0);
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
                if lw.head1x1_active {
                    let h1_in = if lw.head1x1_w.is_empty() {
                        0
                    } else {
                        lw.head1x1_w.len() / head_accum_size
                    };
                    let h1_groups = bottleneck.checked_div(h1_in).unwrap_or(1);
                    let ch_per_group = head_accum_size / h1_groups;
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
                        head_accs[0][head_off..head_off + head_accum_size]
                            .copy_from_slice(&head1x1_scratch[..head_accum_size]);
                    } else {
                        for c in 0..head_accum_size {
                            head_accs[ai][head_off + c] = accum_f64(
                                head_accs[ai][head_off + c],
                                head1x1_scratch[c],
                                acc_mode,
                            );
                        }
                    }
                } else {
                    if li == 0 && ai == 0 {
                        head_accs[0][head_off..head_off + z_len]
                            .copy_from_slice(&z_scratch[..z_len]);
                    } else {
                        for c in 0..z_len {
                            head_accs[ai][head_off + c] =
                                accum_f64(head_accs[ai][head_off + c], z_scratch[c], acc_mode);
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

            // Finalize head for this array
            let hs = arr.head_size;
            let k = arr.head_kernel_size;
            let hw = &arr.head_w;
            let hb = &arr.head_b;
            let channels = arr.head_accum_size;
            let mut cur_head_out = vec![0.0f64; hs];
            for oc in 0..hs {
                let w_base = oc * k * channels;
                let mut y = hb[oc];
                for t in 0..k {
                    let col = head_col.wrapping_sub(k - 1 - t) & ring_mask;
                    let ha_off = col * channels;
                    let w_off = w_base + t * channels;
                    for ic in 0..channels {
                        y = mul_add_f64(hw[w_off + ic], head_accs[ai][ha_off + ic], y, acc_mode);
                    }
                }
                cur_head_out[oc] = y * arr.head_scale;
            }

            if ai + 1 < num_arrays {
                prev_head_out = cur_head_out;
            } else if all_channels {
                let out_base = f * hs;
                for oc in 0..hs {
                    output[out_base + oc] = cur_head_out[oc] * cascade_head_scale;
                }
            } else {
                output[f] = cur_head_out[0] * cascade_head_scale;
            }
        }
    }

    output
}
