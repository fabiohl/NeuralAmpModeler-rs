// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::loader::dispatcher::wavenet::layout::*;

    #[test]
    fn select_interleave_width_ch12_returns_16() {
        assert_eq!(select_interleave_width(12), 16);
        assert_eq!(select_interleave_width(16), 16);
        assert_eq!(select_interleave_width(8), 8);
        assert_eq!(select_interleave_width(4), 4);
        assert_eq!(select_interleave_width(7), 4);
    }

    #[test]
    fn transpose_16wide_ch12_pads_lanes_12_to_15() {
        let in_ch: usize = 12;
        let out_ch: usize = 12;
        let kernel: usize = 3;
        let raw_len = out_ch * in_ch * kernel;
        let padded_total = out_ch.div_ceil(16) * 16 * in_ch * kernel;

        let mut raw = vec![0.0f32; raw_len];
        let mut weights = vec![1.0f32; padded_total];

        for out_c in 0..out_ch {
            for in_c in 0..in_ch {
                for k in 0..kernel {
                    raw[(out_c * in_ch + in_c) * kernel + k] = (out_c * 100 + in_c * 10 + k) as f32;
                }
            }
        }

        transpose_conv1d_interleaved_16wide(&raw, &mut weights, in_ch, out_ch, kernel);

        for out_c in 0..12 {
            let lane = out_c;
            for k in 0..kernel {
                for in_c in 0..in_ch {
                    let idx = k * (in_ch * 16) + in_c * 16 + lane;
                    let expected = (out_c * 100 + in_c * 10 + k) as f32;
                    assert_eq!(
                        weights[idx], expected,
                        "lane {lane}, k={k}, in_c={in_c}: expected {expected}, got {}",
                        weights[idx]
                    );
                }
            }
        }

        for lane in 12..16 {
            for k in 0..kernel {
                for in_c in 0..in_ch {
                    let idx = k * (in_ch * 16) + in_c * 16 + lane;
                    assert_eq!(
                        weights[idx], 0.0,
                        "lane {lane}, k={k}, in_c={in_c} should be zero, got {}",
                        weights[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn transpose_4wide_to_16wide_ch12_pads_correctly() {
        let in_ch: usize = 12;
        let out_ch: usize = 12;
        let kernel: usize = 3;
        let num_blocks_4 = out_ch.div_ceil(4);
        let src_len = num_blocks_4 * 4 * in_ch * kernel;
        let dst_len = out_ch.div_ceil(16) * 16 * in_ch * kernel;

        let mut src = vec![0.0f32; src_len];
        let mut dst = vec![1.0f32; dst_len];

        for b in 0..num_blocks_4 {
            for k in 0..kernel {
                for in_c in 0..in_ch {
                    for lane in 0..4 {
                        let out_c = b * 4 + lane;
                        let src_idx = b * (kernel * in_ch * 4) + k * (in_ch * 4) + in_c * 4 + lane;
                        if out_c < out_ch {
                            src[src_idx] = (out_c * 100 + in_c * 10 + k) as f32;
                        }
                    }
                }
            }
        }

        transpose_4wide_to_16wide(&src, &mut dst, in_ch, out_ch, kernel);

        for out_c in 0..12 {
            let lane = out_c % 16;
            for k in 0..kernel {
                for in_c in 0..in_ch {
                    let dst_idx = k * (in_ch * 16) + in_c * 16 + lane;
                    let expected = (out_c * 100 + in_c * 10 + k) as f32;
                    assert_eq!(
                        dst[dst_idx], expected,
                        "out_c={out_c}, k={k}, in_c={in_c}: expected {expected}, got {}",
                        dst[dst_idx]
                    );
                }
            }
        }

        for lane in 12..16 {
            for k in 0..kernel {
                for in_c in 0..in_ch {
                    let dst_idx = k * (in_ch * 16) + in_c * 16 + lane;
                    assert_eq!(
                        dst[dst_idx], 0.0,
                        "lane {lane}, k={k}, in_c={in_c} should be zero, got {}",
                        dst[dst_idx]
                    );
                }
            }
        }
    }
}
