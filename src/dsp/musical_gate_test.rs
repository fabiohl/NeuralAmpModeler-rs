// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

fn cpp_reference_gate(samples: &[f64], sample_rate: f64, threshold_db: f64) -> Vec<f64> {
    let alpha = 0.5_f64.powf(1.0 / (MUSICAL_GATE_TIME_SEC * sample_rate));
    let beta = 1.0 - alpha;
    let dt = 1.0 / sample_rate;
    let max_gr = gain_reduction_db(MINIMUM_LOUDNESS_DB, threshold_db);
    let d_open = -max_gr / (MUSICAL_GATE_OPEN_TIME_SEC * sample_rate);
    let d_close = max_gr / (MUSICAL_GATE_CLOSE_TIME_SEC * sample_rate);

    let mut level = MINIMUM_LOUDNESS_POWER;
    let mut last_gr = max_gr;
    let mut time_held = 0.0;
    let mut holding = false;
    let mut out = Vec::with_capacity(samples.len());

    for &s in samples {
        level =
            (alpha * level + beta * (s * s)).clamp(MINIMUM_LOUDNESS_POWER, MAXIMUM_LOUDNESS_POWER);
        let level_db = 10.0 * level.log10();

        if holding {
            last_gr = 0.0;
            if level_db < threshold_db {
                time_held += dt;
                if time_held >= MUSICAL_GATE_HOLD_TIME_SEC {
                    holding = false;
                }
            } else {
                time_held = 0.0;
            }
            out.push(1.0);
        } else {
            let target_gr = gain_reduction_db(level_db, threshold_db);
            if target_gr > last_gr {
                let d_gain = (0.5 * (target_gr - last_gr)).clamp(0.0, d_open);
                last_gr += d_gain;
                if last_gr >= 0.0 {
                    last_gr = 0.0;
                    holding = true;
                    time_held = 0.0;
                    out.push(1.0);
                    continue;
                }
            } else if target_gr < last_gr {
                let d_gain = (0.5 * (target_gr - last_gr)).clamp(d_close, 0.0);
                last_gr += d_gain;
                if last_gr < max_gr {
                    last_gr = max_gr;
                }
            }
            out.push(10.0_f64.powf(last_gr / 10.0));
        }
    }
    out
}

#[test]
fn fast_math_accuracy_and_prepare_zero_libm() {
    for exp_db in -120..=30 {
        let x = 10.0_f64.powf((exp_db as f64) / 10.0);
        let err = (fast_log10_pos_f64(x) - x.log10()).abs();
        assert!(err < 1e-8, "log10 err {err} at x={x}");
    }
    for i in 0..=250 {
        let gr = -(i as f64);
        let got = fast_pow10_nonpos_f64(gr * 0.1);
        let expected = 10.0_f64.powf(gr * 0.1);
        let err = (got - expected).abs();
        assert!(err < 1e-9, "pow10 err {err} at gr={gr}");
    }
    for sr in [44_100_u32, 48_000, 96_000] {
        let g = MusicalNoiseGate::new(sr);
        let expected_alpha = 0.5_f64.powf(1.0 / (MUSICAL_GATE_TIME_SEC * (sr as f64)));
        assert!(
            (g.alpha - expected_alpha).abs() < 1e-9,
            "prepare alpha mismatch at {sr}: {} vs {expected_alpha}",
            g.alpha
        );
    }
}

#[test]
fn parity_with_cpp_reference_across_rates() {
    for sr in [44_100_u32, 48_000, 96_000] {
        let n = (sr / 5) as usize;
        let mut input_f64 = Vec::with_capacity(n);
        for i in 0..n {
            let env = if i < n / 4 {
                0.0
            } else if i < n / 2 {
                0.5
            } else {
                1e-4
            };
            let phase = (i as f64) * 2.0 * core::f64::consts::PI * 220.0 / (sr as f64);
            input_f64.push(env * phase.sin());
        }
        let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
        let input_f64_q: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();

        let expected = cpp_reference_gate(&input_f64_q, sr as f64, -55.0);
        let mut gate = MusicalNoiseGate::new(sr);
        gate.set_threshold(-55.0, true);

        let mut got = Vec::with_capacity(n);
        let mut idx = 0;
        while idx < n {
            let chunk = (n - idx).min(MAX_RESAMP_BUF);
            gate.process_trigger(
                &input_f32[idx..idx + chunk],
                &input_f32[idx..idx + chunk],
                chunk,
                true,
            );
            got.extend_from_slice(gate.last_gains_l(chunk));
            idx += chunk;
        }

        let mut max_err = 0.0_f64;
        for (&g, &e) in got.iter().zip(expected.iter()) {
            max_err = max_err.max((g as f64 - e).abs());
        }
        assert!(max_err < 1e-5, "max_err={max_err} at sr={sr}");
    }
}

#[test]
fn disabled_bypass_and_scaled_trigger_work_independently() {
    let mut gate = MusicalNoiseGate::new(48_000);
    gate.set_threshold(-40.0, false);

    let loud = vec![0.4f32; 64];
    let quiet = vec![0.0f32; 64];
    let mut out_l = vec![0.25f32; 64];
    let mut out_r = vec![0.25f32; 64];
    gate.process_trigger(&quiet, &quiet, 64, false);
    gate.apply_gain(&mut out_l, &mut out_r, 64, false);
    assert!(out_l.iter().all(|&x| (x - 0.25).abs() < 1e-7));

    gate.set_threshold(-40.0, true);
    for _ in 0..80 {
        gate.process_trigger_scaled(&loud, &quiet, 64, false, 1.0);
    }
    let mut mod_l = vec![1.0f32; 64];
    let mut mod_r = vec![1.0f32; 64];
    gate.apply_gain(&mut mod_l, &mut mod_r, 64, false);
    assert!(mod_l[63] > 0.99, "L expected open, got {}", mod_l[63]);
    assert!(mod_r[63] < 0.01, "R expected closed, got {}", mod_r[63]);
}

#[test]
fn apply_gain_extends_last_valid_gain_and_reset_clears_stale_buffer() {
    let mut gate = MusicalNoiseGate::new(48_000);
    gate.set_threshold(-40.0, true);

    // First block: 64 silent samples drive gains to heavy attenuation (< 0.1).
    let quiet = vec![0.0f32; 64];
    gate.process_trigger_scaled(&quiet, &quiet, 64, false, 1.0);
    let attenuated_tail = gate.last_gains_l(64)[63];
    assert!(
        attenuated_tail < 0.5,
        "expected attenuated gain after silent block, got {attenuated_tail}"
    );

    // Second block: trigger with loud signal for only n_trig = 16 samples,
    // then apply_gain with n_out = 64 (> n_trig). Samples 16..64 MUST extend
    // the last valid gain of the current block (gains[15]), NOT read stale
    // attenuated gains from block 1's slots [16..64].
    let loud = vec![1.0f32; 16];
    for _ in 0..80 {
        gate.process_trigger_scaled(&loud, &loud, 16, false, 1.0);
    }
    let last_valid_l = gate.last_gains_l(16)[15];
    assert!(
        last_valid_l > 0.99,
        "expected open gain at index 15, got {last_valid_l}"
    );

    let mut out_l = vec![1.0f32; 64];
    let mut out_r = vec![1.0f32; 64];
    gate.apply_gain(&mut out_l, &mut out_r, 64, false);
    for (i, (&yl, &yr)) in out_l.iter().zip(out_r.iter()).enumerate() {
        if i >= 16 {
            assert!(
                (yl - last_valid_l).abs() < 1e-6,
                "out_l[{i}]={yl} leaked stale gain instead of extending last_valid={last_valid_l}"
            );
            assert!(
                (yr - last_valid_l).abs() < 1e-6,
                "out_r[{i}]={yr} leaked stale gain instead of extending last_valid={last_valid_l}"
            );
        }
    }

    // After reset(), valid_gain_l/r == 0, so apply_gain without a new trigger falls back to 1.0.
    gate.reset();
    let mut after_reset_l = vec![0.5f32; 32];
    let mut after_reset_r = vec![0.5f32; 32];
    gate.apply_gain(&mut after_reset_l, &mut after_reset_r, 32, false);
    assert!(after_reset_l.iter().all(|&x| (x - 0.5).abs() < 1e-7));
    assert!(after_reset_r.iter().all(|&x| (x - 0.5).abs() < 1e-7));
}

fn reference_log10_trigger_and_scalar_apply(
    samples: &[f32],
    audio_l: &mut [f32],
    sr: u32,
    threshold_db: f64,
    n_trig: usize,
    n_out: usize,
) -> Vec<f32> {
    let fs = sr as f64;
    let alpha = fast_pow10_nonpos_f64(-core::f64::consts::LOG10_2 / (MUSICAL_GATE_TIME_SEC * fs));
    let beta = 1.0 - alpha;
    let dt = 1.0 / fs;
    let max_gr = gain_reduction_db(MINIMUM_LOUDNESS_DB, threshold_db);
    let d_open = -max_gr / (MUSICAL_GATE_OPEN_TIME_SEC * fs);
    let d_close = max_gr / (MUSICAL_GATE_CLOSE_TIME_SEC * fs);

    let mut level = MINIMUM_LOUDNESS_POWER;
    let mut last_gr = max_gr;
    let mut time_held = 0.0;
    let mut holding = false;
    let mut gains = Vec::with_capacity(n_trig);

    for &s in &samples[..n_trig] {
        let x = if s.is_finite() { s as f64 } else { 0.0 };
        level =
            (alpha * level + beta * (x * x)).clamp(MINIMUM_LOUDNESS_POWER, MAXIMUM_LOUDNESS_POWER);
        let level_db = 10.0 * fast_log10_pos_f64(level);

        if holding {
            last_gr = 0.0;
            if level_db < threshold_db {
                time_held += dt;
                if time_held >= MUSICAL_GATE_HOLD_TIME_SEC {
                    holding = false;
                }
            } else {
                time_held = 0.0;
            }
            gains.push(1.0);
        } else {
            let target_gr = gain_reduction_db(level_db, threshold_db);
            if target_gr > last_gr {
                let d_gain = (0.5 * (target_gr - last_gr)).clamp(0.0, d_open);
                last_gr += d_gain;
                if last_gr >= 0.0 {
                    last_gr = 0.0;
                    holding = true;
                    time_held = 0.0;
                    gains.push(1.0);
                    continue;
                }
            } else if target_gr < last_gr {
                let d_gain = (0.5 * (target_gr - last_gr)).clamp(d_close, 0.0);
                last_gr += d_gain;
                if last_gr < max_gr {
                    last_gr = max_gr;
                }
            }
            gains.push(fast_pow10_nonpos_f64(last_gr * 0.1) as f32);
        }
    }

    let valid = n_out.min(n_trig).min(audio_l.len());
    for (y, &g) in audio_l[..valid].iter_mut().zip(gains[..valid].iter()) {
        *y *= g;
    }
    if valid < n_out {
        let tail_gain = *gains.last().unwrap_or(&1.0);
        for y in &mut audio_l[valid..n_out] {
            *y *= tail_gain;
        }
    }
    gains
}

#[test]
fn fast_path_linear_power_and_avx2_apply_gain_parity() {
    let sr = 48_000_u32;
    let threshold_db = -48.0;
    // Multi-regime signal: silence -> loud attack (above threshold) -> decay -> re-trigger.
    // Non-multiple-of-8 length (131 samples) exercises both AVX2 8-wide chunks and scalar tail,
    // plus n_out > n_trig tail-gain vectorization.
    let n_trig = 131usize;
    let n_out = 163usize;
    let mut trig = Vec::with_capacity(n_trig);
    for i in 0..n_trig {
        let amp = if i < 20 {
            0.0f32
        } else if i < 85 {
            0.25f32
        } else {
            1e-5f32
        };
        trig.push(amp * ((i as f32) * 0.21).sin());
    }

    let mut gate = MusicalNoiseGate::new(sr);
    gate.set_threshold(threshold_db, true);

    let audio_init: Vec<f32> = (0..n_out).map(|i| ((i as f32) * 0.13).cos()).collect();
    let mut out_fast = audio_init.clone();
    let mut out_ref = audio_init;

    gate.process_trigger(&trig, &trig, n_trig, true);
    gate.apply_gain(&mut out_fast, &mut [], n_out, true);
    let gains_fast = gate.last_gains_l(n_trig);

    let gains_ref = reference_log10_trigger_and_scalar_apply(
        &trig,
        &mut out_ref,
        sr,
        threshold_db,
        n_trig,
        n_out,
    );

    let max_gain_diff = gains_fast
        .iter()
        .zip(gains_ref.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Measured: max_gain_diff = 0.0 (exact match across silence, attack, hold, and release).
    assert!(
        max_gain_diff <= 1e-7,
        "gain fast-path diverged: max_gain_diff={max_gain_diff}"
    );

    let max_out_diff = out_fast
        .iter()
        .zip(out_ref.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Measured: max_out_diff = 0.0 (AVX2 apply_gain matches scalar reference bit-for-bit).
    assert!(
        max_out_diff <= 1e-7,
        "apply_gain AVX2 diverged: max_out_diff={max_out_diff}"
    );
}
