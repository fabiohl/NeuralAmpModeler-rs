// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Scalar reference for `_cond_to_scale_shift` — computes
/// `scale_shift[o] = bias[o] + Σ weight[o*cond_sz + i] * cond[i]`.
fn cond_to_scale_shift_ref(
    weights: &[f32],
    bias: &[f32],
    condition: &[f32],
    channels: usize,
    cond_size: usize,
    shift: bool,
    groups: u32,
) -> Vec<f32> {
    let g = groups as usize;
    let ch_per_group = channels / g;
    let cond_per_group = cond_size / g;
    let out_per_group = if shift {
        ch_per_group * 2
    } else {
        ch_per_group
    };
    let mut out = vec![0.0f32; channels * 2];

    for grp in 0..g {
        let cond_slice = &condition[grp * cond_per_group..(grp + 1) * cond_per_group];
        let w_offset = grp * out_per_group * cond_per_group;

        for row in 0..out_per_group {
            let global_out = if row < ch_per_group {
                grp * ch_per_group + row
            } else {
                channels + grp * ch_per_group + (row - ch_per_group)
            };
            let mut sum = bias[global_out];
            let w_start = w_offset + row * cond_per_group;
            for i in 0..cond_per_group {
                sum += weights[w_start + i] * cond_slice[i];
            }
            out[global_out] = sum;
        }
    }
    out
}

/// Scalar reference for per-channel modulation.
fn apply_modulation_ref(input: &[f32], scale_shift: &[f32]) -> Vec<f32> {
    let ch = input.len();
    let scale = &scale_shift[..ch];
    let shift = &scale_shift[ch..ch * 2];
    input
        .iter()
        .enumerate()
        .map(|(c, &v)| v * scale[c] + shift[c])
        .collect()
}

#[test]
fn test_film_config_default() {
    let config = FiLMConfig::default();
    assert!(!config.active);
    assert!(config.shift);
    assert_eq!(config.groups, 1);
}

#[test]
fn test_film_config_custom() {
    let config = FiLMConfig {
        active: true,
        shift: false,
        groups: 4,
    };
    assert!(config.active);
    assert!(!config.shift);
    assert_eq!(config.groups, 4);
}

/// FiLM with `groups=1, shift=true` — identity scale + zero shift.
#[test]
fn test_film_process_identity_shift() {
    let cond_size = 4;
    let channels = 8;
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };

    // weights: 16 rows (8 scale + 8 shift) × 4 cond
    let mut weights = vec![0.0f32; channels * 2 * cond_size];
    // Set scale weights to produce scale[c] = 1.0 via identity mapping
    for c in 0..channels {
        weights[c * cond_size + (c % cond_size)] = 1.0;
    }
    let bias = vec![0.0f32; channels * 2];

    let mut layer = FiLMLayer::load(config, cond_size, channels, weights, bias).unwrap();

    // All-ones condition so every scale[c] = Σ weight[c, i] * 1.0 = 1.0
    let condition = vec![1.0f32; cond_size];
    let mut input = vec![2.0f32, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0];
    let expected_input = input.clone();

    // SAFETY: `condition.len() == cond_size` and `input.len() <= channels` as required by
    // `FiLMLayer::process`'s Safety docs.
    unsafe { layer.process(&mut input, &condition) };

    // Identity scale (≈1.0) + zero shift → output ≈ input
    for c in 0..channels {
        assert!(
            (input[c] - expected_input[c]).abs() < 1e-5,
            "channel {}: expected {}, got {}",
            c,
            expected_input[c],
            input[c]
        );
    }
}

/// FiLM with `groups=1, shift=false` — scale-only.
#[test]
fn test_film_process_scale_only() {
    let cond_size = 3;
    let channels = 8;
    let config = FiLMConfig {
        active: true,
        shift: false,
        groups: 1,
    };

    // weights: 8 rows (scale only) × 3 cond
    let mut weights = vec![0.0f32; channels * cond_size];
    // Diagonal: scale[c] = condition[c % 3] * 2.0
    for c in 0..channels {
        weights[c * cond_size + (c % cond_size)] = 2.0;
    }
    let bias = vec![0.1f32; channels]; // small bias

    let mut layer =
        FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone()).unwrap();

    let condition = vec![0.5f32, 0.25, 0.125];
    let mut input = vec![1.0f32; channels];

    // Reference: scale = W * cond + b
    let ref_scale_shift =
        cond_to_scale_shift_ref(&weights, &bias, &condition, channels, cond_size, false, 1);
    let expected = apply_modulation_ref(&input, &ref_scale_shift);

    // SAFETY: `condition.len() == cond_size` and `input.len() <= channels` as required by
    // `FiLMLayer::process`'s Safety docs.
    unsafe { layer.process(&mut input, &condition) };

    for c in 0..channels {
        assert!(
            (input[c] - expected[c]).abs() < 1e-5,
            "channel {}: expected {}, got {}",
            c,
            expected[c],
            input[c]
        );
    }
}

/// FiLM with `groups=2, shift=true`.
#[test]
fn test_film_process_groups_shift() {
    let cond_size = 6;
    let channels = 8;
    let groups = 2u32;
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups,
    };

    let g = groups as usize;
    let ch_per_group = channels / g;
    let cond_per_group = cond_size / g;
    let out_per_group = ch_per_group * 2; // shift=true → 2×

    // Build weights: group0 → inputs 0..3, group1 → inputs 3..6
    let total_w = g * out_per_group * cond_per_group;
    let mut weights = vec![0.0f32; total_w];
    let mut bias = vec![0.0f32; channels * 2];

    // Scale each group's output to be 2× its condition slice
    for grp in 0..g {
        let w_offset = grp * out_per_group * cond_per_group;
        for row in 0..out_per_group {
            let w_start = w_offset + row * cond_per_group;
            for ic in 0..cond_per_group {
                weights[w_start + ic] = 2.0;
            }
            let global_out = if row < ch_per_group {
                grp * ch_per_group + row
            } else {
                channels + grp * ch_per_group + (row - ch_per_group)
            };
            bias[global_out] = 0.5;
        }
    }

    let mut layer =
        FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone()).unwrap();

    let condition = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut input = vec![0.5f32; channels];

    let ref_scale_shift = cond_to_scale_shift_ref(
        &weights, &bias, &condition, channels, cond_size, true, groups,
    );
    let expected = apply_modulation_ref(&input, &ref_scale_shift);

    // SAFETY: `condition.len() == cond_size` and `input.len() <= channels` as required by
    // `FiLMLayer::process`'s Safety docs.
    unsafe { layer.process(&mut input, &condition) };

    for c in 0..channels {
        assert!(
            (input[c] - expected[c]).abs() < 1e-4,
            "channel {}: expected {}, got {}",
            c,
            expected[c],
            input[c]
        );
    }
}

/// FiLM with odd channel count (3, A2-nano).
#[test]
fn test_film_process_odd_channels() {
    let cond_size = 3;
    let channels = 3;
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };

    let mut weights = vec![0.0f32; channels * 2 * cond_size];
    // Group 0: scale = cond[0] * 2, shift = cond[0] * 0.1
    for c in 0..channels {
        weights[c * cond_size + (c % cond_size)] = 2.0;
        weights[(channels + c) * cond_size + (c % cond_size)] = 0.1;
    }
    let bias = vec![0.0f32; channels * 2];

    let mut layer =
        FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone()).unwrap();

    let condition = vec![0.5f32, 0.25, 0.125];
    let mut input = vec![0.5f32, 1.0, 1.5];

    let ref_scale_shift =
        cond_to_scale_shift_ref(&weights, &bias, &condition, channels, cond_size, true, 1);
    let expected = apply_modulation_ref(&input, &ref_scale_shift);

    // SAFETY: `condition.len() == cond_size` and `input.len() <= channels` as required by
    // `FiLMLayer::process`'s Safety docs.
    unsafe { layer.process(&mut input, &condition) };

    for c in 0..channels {
        assert!(
            (input[c] - expected[c]).abs() < 1e-5,
            "channel {}: expected {}, got {}",
            c,
            expected[c],
            input[c]
        );
    }
}

/// Verify `scale_shift_buf` is zeroed in the shift region when
/// `shift == false`, so the modulation step produces correct results.
#[test]
fn test_film_shift_buffer_zeroed_when_shift_false() {
    let cond_size = 2;
    let channels = 4;
    let config = FiLMConfig {
        active: true,
        shift: false,
        groups: 1,
    };

    let weights = vec![1.0f32; channels * cond_size];
    let bias = vec![0.0f32; channels];

    let mut layer =
        FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone()).unwrap();

    let condition = vec![2.0f32, 3.0];
    let mut input = vec![1.0f32; channels];

    // Pre-fill shift region with garbage to ensure it gets zeroed
    for c in channels..channels * 2 {
        // SAFETY: buffer is pre-allocated via load()
        layer.scale_shift_buf[c] = 999.0;
    }

    // SAFETY: `condition.len() == cond_size` and `input.len() <= channels` as required by
    // `FiLMLayer::process`'s Safety docs.
    unsafe { layer.process(&mut input, &condition) };

    // Shift region must be zero
    for c in channels..channels * 2 {
        assert_eq!(layer.scale_shift_buf[c], 0.0);
    }
}

/// F10 regression: FiLM with cond_size=2 (> 1) and groups=1, shift=true.
/// Exercises the debug_assert guard added in E1.2 with properly sized condition.
#[test]
fn test_film_process_cond_size_greater_than_1() {
    let cond_size = 2;
    let channels = 4;
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };

    let mut weights = vec![0.0f32; channels * 2 * cond_size];
    for c in 0..channels {
        weights[c * cond_size + (c % cond_size)] = 1.0;
        weights[(channels + c) * cond_size + (c % cond_size)] = 0.5;
    }
    let bias = vec![0.0f32; channels * 2];

    let mut layer =
        FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone()).unwrap();

    let condition = vec![0.25f32, 0.75];
    let mut input = vec![1.0f32, 2.0, 3.0, 4.0];

    let ref_scale_shift =
        cond_to_scale_shift_ref(&weights, &bias, &condition, channels, cond_size, true, 1);
    let expected = apply_modulation_ref(&input, &ref_scale_shift);

    // SAFETY: `condition.len() == cond_size` and `input.len() <= channels` as required by
    // `FiLMLayer::process`'s Safety docs.
    unsafe { layer.process(&mut input, &condition) };

    for c in 0..channels {
        assert!(
            (input[c] - expected[c]).abs() < 1e-5,
            "channel {}: expected {}, got {}",
            c,
            expected[c],
            input[c]
        );
    }
}

/// F10 regression: FiLM with cond_size=2 (> 1) and groups=2, shift=true.
/// Exercises group-based indexing in cond_to_scale_shift with cond_size > 1.
#[test]
fn test_film_process_cond_size_2_groups_2() {
    let cond_size = 2;
    let channels = 4;
    let groups = 2u32;
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups,
    };

    let g = groups as usize;
    let ch_per_group = channels / g;
    let cond_per_group = cond_size / g;
    let out_per_group = ch_per_group * 2;
    let total_w = g * out_per_group * cond_per_group;

    let mut weights = vec![0.0f32; total_w];
    let mut bias = vec![0.0f32; channels * 2];

    for grp in 0..g {
        let w_offset = grp * out_per_group * cond_per_group;
        for row in 0..out_per_group {
            let w_start = w_offset + row * cond_per_group;
            for ic in 0..cond_per_group {
                weights[w_start + ic] = 1.5;
            }
            let global_out = if row < ch_per_group {
                grp * ch_per_group + row
            } else {
                channels + grp * ch_per_group + (row - ch_per_group)
            };
            bias[global_out] = 0.1;
        }
    }

    let mut layer =
        FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone()).unwrap();

    let condition = vec![0.5f32, 0.5];
    let mut input = vec![1.0f32; channels];

    let ref_scale_shift = cond_to_scale_shift_ref(
        &weights, &bias, &condition, channels, cond_size, true, groups,
    );
    let expected = apply_modulation_ref(&input, &ref_scale_shift);

    // SAFETY: `condition.len() == cond_size` and `input.len() <= channels` as required by
    // `FiLMLayer::process`'s Safety docs.
    unsafe { layer.process(&mut input, &condition) };

    for c in 0..channels {
        assert!(
            (input[c] - expected[c]).abs() < 1e-5,
            "channel {}: expected {}, got {}",
            c,
            expected[c],
            input[c]
        );
    }
}

/// F10 regression: FiLM with cond_size=4, groups=4, shift=false (scale-only).
/// Exercises cond_per_group = 1 with cond_size > 1.
#[test]
fn test_film_process_cond_size_4_groups_4_scale_only() {
    let cond_size = 4;
    let channels = 4;
    let groups = 4u32;
    let config = FiLMConfig {
        active: true,
        shift: false,
        groups,
    };

    let g = groups as usize;
    let ch_per_group = channels / g;
    let cond_per_group = cond_size / g;
    let out_per_group = ch_per_group;
    let total_w = g * out_per_group * cond_per_group;

    let mut weights = vec![0.0f32; total_w];
    let mut bias = vec![0.0f32; channels];

    for grp in 0..g {
        let w_offset = grp * out_per_group * cond_per_group;
        for row in 0..out_per_group {
            let w_start = w_offset + row * cond_per_group;
            for ic in 0..cond_per_group {
                weights[w_start + ic] = 2.0;
            }
            let global_out = grp * ch_per_group + row;
            bias[global_out] = 0.5;
        }
    }

    let mut layer =
        FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone()).unwrap();

    let condition = vec![0.1f32, 0.2, 0.3, 0.4];
    let mut input = vec![1.0f32; channels];

    let ref_scale_shift = cond_to_scale_shift_ref(
        &weights, &bias, &condition, channels, cond_size, false, groups,
    );
    let expected = apply_modulation_ref(&input, &ref_scale_shift);

    // SAFETY: `condition.len() == cond_size` and `input.len() <= channels` as required by
    // `FiLMLayer::process`'s Safety docs.
    unsafe { layer.process(&mut input, &condition) };

    // Shift region must be zero when shift=false
    for c in channels..channels * 2 {
        assert_eq!(layer.scale_shift_buf[c], 0.0);
    }

    for c in 0..channels {
        assert!(
            (input[c] - expected[c]).abs() < 1e-5,
            "channel {}: expected {}, got {}",
            c,
            expected[c],
            input[c]
        );
    }
}
