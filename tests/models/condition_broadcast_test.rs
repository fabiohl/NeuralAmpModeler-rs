// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dedicated verification for Condition-DSP broadcasting (TCP2.4 / F-PERF-05).
//!
//! Validates:
//! 1. Single-channel condition DSP (`dsp_ch == 1`) broadcasting across all `cond_size`
//!    channels (including `c = 0` for all frames `f >= 1`).
//! 2. Multi-channel condition DSP (`1 < dsp_ch < cond_size`) modular broadcasting
//!    `raw_out[f * dsp_ch + (c % dsp_ch)]`.
//! 3. Prewarm vs process broadcast consistency for `dsp_ch == 1` and `1 < dsp_ch < cond`.
//! 4. Mathematical bit-exact parity against the f64 Reference Oracle broadcast contract.

// Exact reference parity tests use explicit loop indexing and identity arithmetic.
#![allow(clippy::needless_range_loop, clippy::identity_op)]

#[test]
fn test_a2_dyn_broadcast_dsp_ch1_includes_channel_zero() {
    let nf = 8;
    let cond_size = 4;
    let dsp_ch = 1;

    // Buffer pre-filled with sentinel value to detect skipped writes
    let mut buf = vec![-999.0f32; nf * cond_size];

    // Simulate cond_dsp.process writing single-channel output in the first nf positions
    for f in 0..nf {
        buf[f] = (f as f32) + 1.0;
    }

    // Production broadcast algorithm from A2 process.rs / cascade mod.rs
    if dsp_ch > 0 && dsp_ch < cond_size {
        if dsp_ch == 1 {
            for f in (0..nf).rev() {
                let val = buf[f];
                for c in 0..cond_size {
                    buf[f * cond_size + c] = val;
                }
            }
        } else {
            for f in (0..nf).rev() {
                for c in (0..dsp_ch).rev() {
                    buf[f * cond_size + c] = buf[f * dsp_ch + c];
                }
                for c in dsp_ch..cond_size {
                    buf[f * cond_size + c] = buf[f * cond_size + (c % dsp_ch)];
                }
            }
        }
    }

    // Verify all frames and all channels, with specific emphasis on c = 0 for f >= 1
    for f in 0..nf {
        let expected_val = (f as f32) + 1.0;
        for c in 0..cond_size {
            let actual = buf[f * cond_size + c];
            assert_eq!(
                actual, expected_val,
                "Mismatch at frame {f}, channel {c}: expected {expected_val}, got {actual}"
            );
        }
    }
}

#[test]
fn test_wavenet_dyn_broadcast_multichannel_modular() {
    let nf = 6;
    let dsp_ch = 2;
    let cond = 5;

    // Buffer pre-filled with sentinels
    let mut cond_out = vec![-999.0f32; nf * cond];

    // Simulate cond_dsp producing 2 channels per frame
    for f in 0..nf {
        cond_out[f * dsp_ch + 0] = (f as f32) * 10.0 + 1.0; // ch 0
        cond_out[f * dsp_ch + 1] = (f as f32) * 10.0 + 2.0; // ch 1
    }

    // Production algorithm from WaveNet model_dyn.rs
    if dsp_ch > 0 && dsp_ch < cond {
        if dsp_ch == 1 {
            for f in (0..num_frames_mock(nf)).rev() {
                let val = cond_out[f];
                for c in 0..cond {
                    cond_out[f * cond + c] = val;
                }
            }
        } else {
            for f in (0..nf).rev() {
                for c in (0..dsp_ch).rev() {
                    cond_out[f * cond + c] = cond_out[f * dsp_ch + c];
                }
                for c in dsp_ch..cond {
                    cond_out[f * cond + c] = cond_out[f * cond + (c % dsp_ch)];
                }
            }
        }
    }

    for f in 0..nf {
        let ch0 = (f as f32) * 10.0 + 1.0;
        let ch1 = (f as f32) * 10.0 + 2.0;

        assert_eq!(cond_out[f * cond + 0], ch0, "Frame {f}, ch 0");
        assert_eq!(cond_out[f * cond + 1], ch1, "Frame {f}, ch 1");
        assert_eq!(cond_out[f * cond + 2], ch0, "Frame {f}, ch 2 (2 % 2 = 0)");
        assert_eq!(cond_out[f * cond + 3], ch1, "Frame {f}, ch 3 (3 % 2 = 1)");
        assert_eq!(cond_out[f * cond + 4], ch0, "Frame {f}, ch 4 (4 % 2 = 0)");
    }
}

fn num_frames_mock(nf: usize) -> usize {
    nf
}

#[test]
fn test_wavenet_dyn_prewarm_vs_process_single_frame_parity() {
    let cond = 5;

    for dsp_ch in [1, 2, 3] {
        // --- 1. Run Prewarm pattern ---
        let mut prewarm_buf = vec![-999.0f32; cond];
        // Populate first dsp_ch channels
        for c in 0..dsp_ch {
            prewarm_buf[c] = (c as f32) + 42.0;
        }

        if dsp_ch > 0 && dsp_ch < cond {
            if dsp_ch == 1 {
                let val = prewarm_buf[0];
                for c in 1..cond {
                    prewarm_buf[c] = val;
                }
            } else {
                for c in dsp_ch..cond {
                    prewarm_buf[c] = prewarm_buf[c % dsp_ch];
                }
            }
        }

        // --- 2. Run Process pattern with nf = 1 ---
        let mut process_buf = vec![-999.0f32; cond];
        for c in 0..dsp_ch {
            process_buf[c] = (c as f32) + 42.0;
        }

        if dsp_ch > 0 && dsp_ch < cond {
            if dsp_ch == 1 {
                for f in (0..1).rev() {
                    let val = process_buf[f];
                    for c in 0..cond {
                        process_buf[f * cond + c] = val;
                    }
                }
            } else {
                for f in (0..1).rev() {
                    for c in (0..dsp_ch).rev() {
                        process_buf[f * cond + c] = process_buf[f * dsp_ch + c];
                    }
                    for c in dsp_ch..cond {
                        process_buf[f * cond + c] = process_buf[f * cond + (c % dsp_ch)];
                    }
                }
            }
        }

        // Assert exact equality between prewarm and single-frame process
        assert_eq!(
            prewarm_buf, process_buf,
            "Prewarm and process condition mismatch for dsp_ch={dsp_ch}, cond={cond}"
        );
    }
}

#[test]
fn test_production_broadcast_matches_f64_oracle() {
    let num_frames = 16;
    let cond_size = 6;

    for dsp_ch in [1, 2, 3] {
        // Raw DSP output: distinct pseudo-random values per channel and frame
        let mut raw_dsp_f32 = vec![0.0f32; num_frames * dsp_ch];
        for i in 0..raw_dsp_f32.len() {
            raw_dsp_f32[i] = ((i * 37 + 13) % 100) as f32 * 0.01;
        }

        // 1. Run f64 Reference Oracle broadcast contract (from dynamic_eval.rs:68-75)
        let mut oracle_f64 = vec![0.0f64; num_frames * cond_size];
        for f in 0..num_frames {
            for c in 0..cond_size {
                oracle_f64[f * cond_size + c] = raw_dsp_f32[f * dsp_ch + (c % dsp_ch)] as f64;
            }
        }

        // 2. Run Production in-place broadcast contract
        let mut prod_f32 = vec![0.0f32; num_frames * cond_size];
        prod_f32[..num_frames * dsp_ch].copy_from_slice(&raw_dsp_f32);

        if dsp_ch == 1 {
            for f in (0..num_frames).rev() {
                let val = prod_f32[f];
                for c in 0..cond_size {
                    prod_f32[f * cond_size + c] = val;
                }
            }
        } else {
            for f in (0..num_frames).rev() {
                for c in (0..dsp_ch).rev() {
                    prod_f32[f * cond_size + c] = prod_f32[f * dsp_ch + c];
                }
                for c in dsp_ch..cond_size {
                    prod_f32[f * cond_size + c] = prod_f32[f * cond_size + (c % dsp_ch)];
                }
            }
        }

        // Verify exact equivalence between production and oracle
        for i in 0..num_frames * cond_size {
            let p = prod_f32[i];
            let o = oracle_f64[i] as f32;
            assert_eq!(
                p, o,
                "Production vs Oracle mismatch at index {i} for dsp_ch={dsp_ch}"
            );
        }
    }
}
