// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::dsp::sinc_kernel::NUM_PHASES;

#[test]
fn test_bypass_asymmetric_buffers() {
    let mut rs = NamResampler::new(48_000, 48_000, 256).expect("new failed");
    assert!(rs.is_bypass());

    let in_full = [1.0f32; 8];
    let in_short = [2.0f32; 3];
    let mut out_full = [0.0f32; 8];
    let mut out_short = [0.0f32; 3];

    let n = rs
        .process_input(&in_full, &in_short, &mut out_full, &mut out_short)
        .samples_written;
    assert_eq!(n, 3);
    assert_eq!(&out_full[..3], &in_full[..3]);
    assert_eq!(&out_short[..3], &in_short[..3]);

    let n = rs
        .process_input(&in_short, &in_full, &mut out_full, &mut out_short)
        .samples_written;
    assert_eq!(n, 3);
    assert_eq!(&out_full[..3], &in_short[..3]);
    assert_eq!(&out_short[..3], &in_full[..3]);

    let n = rs
        .process_output(&in_full, &in_short, &mut out_full, &mut out_short)
        .samples_written;
    assert_eq!(n, 3);
    assert_eq!(&out_full[..3], &in_full[..3]);
    assert_eq!(&out_short[..3], &in_short[..3]);

    let n = rs
        .process_output(&in_short, &in_full, &mut out_full, &mut out_short)
        .samples_written;
    assert_eq!(n, 3);
    assert_eq!(&out_full[..3], &in_short[..3]);
    assert_eq!(&out_short[..3], &in_full[..3]);
}

#[test]
fn test_bypass_48k() {
    let mut rs = NamResampler::new(48_000, 48_000, 256).expect("new failed");
    assert!(rs.is_bypass(), "48k should be bypass");

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input_r = [5.0f32, 4.0, 3.0, 2.0, 1.0];
    let mut output = [0.0f32; 5];
    let mut output_r = [0.0f32; 5];

    let n = rs
        .process_input(&input, &input_r, &mut output, &mut output_r)
        .samples_written;
    assert_eq!(n, 5);
    assert_eq!(output, input, "bypass must copy exactly L");
    assert_eq!(output_r, input_r, "bypass must copy exactly R");

    let n2 = rs
        .process_output(&input, &input_r, &mut output, &mut output_r)
        .samples_written;
    assert_eq!(n2, 5);
    assert_eq!(output, input);
    assert_eq!(output_r, input_r);
}

#[test]
fn test_downsample_96k_to_48k() {
    let chunk = 512usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new failed");
    assert!(!rs.is_bypass());

    let input = vec![0.5f32; chunk];
    let input_r = vec![0.5f32; chunk];
    let mut output = vec![0.0f32; chunk * 2];
    let mut output_r = vec![0.0f32; chunk * 2];
    let n = rs
        .process_input(&input, &input_r, &mut output, &mut output_r)
        .samples_written;

    let expected_approx = chunk / 2;
    assert!(
        n >= expected_approx.saturating_sub(64) && n <= expected_approx + 64,
        "96k→48k: expected ~{expected_approx} samples, got {n}"
    );
}

#[test]
fn test_upsample_44k_to_48k() {
    let chunk = 441usize;
    let mut rs = NamResampler::new(44_100, 48_000, chunk).expect("new failed");
    assert!(!rs.is_bypass());

    let input = vec![0.3f32; chunk];
    let input_r = vec![0.3f32; chunk];
    let mut output = vec![0.0f32; chunk * 2];
    let mut output_r = vec![0.0f32; chunk * 2];
    let n = rs
        .process_input(&input, &input_r, &mut output, &mut output_r)
        .samples_written;

    let expected_approx = (chunk as f64 * 48_000.0 / 44_100.0) as usize;
    assert!(
        n >= expected_approx.saturating_sub(64) && n <= expected_approx + 64,
        "44.1k→48k: expected ~{expected_approx} samples, got {n}"
    );
}

#[test]
fn test_output_upsample_48k_to_96k() {
    let chunk = 256usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new failed");

    let inner_out_size = chunk / 2;
    let input = vec![0.4f32; inner_out_size];
    let input_r = vec![0.4f32; inner_out_size];
    let mut output = vec![0.0f32; chunk * 2];
    let mut output_r = vec![0.0f32; chunk * 2];
    let n = rs
        .process_output(&input, &input_r, &mut output, &mut output_r)
        .samples_written;

    let expected_approx = inner_out_size * 2;
    assert!(
        n >= expected_approx.saturating_sub(64) && n <= expected_approx + 64,
        "48k→96k (output): expected ~{expected_approx} samples, got {n}"
    );
}

#[test]
fn test_roundtrip_96k() {
    let chunk = 1024usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new failed");

    let n_total = chunk * 4;
    let input: Vec<f32> = (0..n_total)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 96_000.0).sin())
        .collect();
    let input_r = input.clone();

    let mut total_mid = 0usize;
    let mut total_out = 0usize;
    let mut mid_energy_sum = 0.0f32;
    let mut out_energy_sum = 0.0f32;

    for start in (0..n_total).step_by(chunk) {
        let end = (start + chunk).min(n_total);
        let blk_l = &input[start..end];
        let blk_r = &input_r[start..end];

        let mut mid = vec![0.0f32; chunk];
        let mut mid_r = vec![0.0f32; chunk];
        let n_mid = rs
            .process_input(blk_l, blk_r, &mut mid, &mut mid_r)
            .samples_written;
        total_mid += n_mid;
        mid_energy_sum += mid[..n_mid].iter().map(|x| x * x).sum::<f32>();

        if n_mid > 0 {
            let mut out = vec![0.0f32; chunk * 2];
            let mut out_r = vec![0.0f32; chunk * 2];
            let n_out = rs
                .process_output(&mid[..n_mid], &mid_r[..n_mid], &mut out, &mut out_r)
                .samples_written;
            total_out += n_out;
            out_energy_sum += out[..n_out].iter().map(|x| x * x).sum::<f32>();
        }
    }

    assert!(
        total_mid > 0,
        "process_input produced no samples across {n_total} frames"
    );
    assert!(
        total_out > 0,
        "process_output produced no samples (mid_total={total_mid})"
    );
    assert!(
        mid_energy_sum > 0.0,
        "Intermediate energy (96→48) is zero (mid_total={total_mid})"
    );

    let energy_in = input.iter().map(|x| x * x).sum::<f32>() / n_total as f32;
    let energy_out = out_energy_sum / total_out.max(1) as f32;

    assert!(
        energy_out > energy_in * 0.05,
        "Roundtrip energy collapsed: in={energy_in:.4}, out={energy_out:.4}, \
         mid_samples={total_mid}, out_samples={total_out}, mid_energy={mid_energy_sum:.4}"
    );
}

#[test]
fn test_impulse_response_input() {
    let chunk = 512usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new failed");

    let mut input = vec![0.0f32; chunk];
    input[0] = 1.0;
    let mut input_r = vec![0.0f32; chunk];
    input_r[0] = 1.0;

    let mut output = vec![0.0f32; chunk];
    let mut output_r = vec![0.0f32; chunk];
    let n = rs
        .process_input(&input, &input_r, &mut output, &mut output_r)
        .samples_written;
    assert!(n > 0);

    let energy: f32 = output[..n].iter().map(|x| x * x).sum();
    assert!(
        energy > 0.0 && energy.is_finite(),
        "Invalid impulse response: energy={energy}"
    );
    let peak = output[..n].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(peak <= 1.5, "Excessive peak: {peak:.4}");
}

#[test]
fn test_impulse_response_output() {
    let chunk = 256usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new failed");

    let inner_out_approx = chunk / 2;
    let mut input = vec![0.0f32; inner_out_approx];
    input[0] = 1.0;
    let mut input_r = vec![0.0f32; inner_out_approx];
    input_r[0] = 1.0;

    let mut output = vec![0.0f32; chunk];
    let mut output_r = vec![0.0f32; chunk];
    let n = rs
        .process_output(&input, &input_r, &mut output, &mut output_r)
        .samples_written;
    assert!(n > 0);

    let energy: f32 = output[..n].iter().map(|x| x * x).sum();
    assert!(
        energy > 0.0 && energy.is_finite(),
        "Invalid impulse response (output): energy={energy}"
    );
    let peak = output[..n].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(peak <= 4.0, "Excessive peak (output): {peak:.4}");
}

#[test]
fn test_phase_accum_underflow_guard() {
    let bank = crate::dsp::sinc_kernel::generate_polyphase_bank(44100, 48000)
        .expect("construction should succeed for test-sized buffers");
    let mut core = ResamplerCore::new(44100, 48000, bank).unwrap();

    core.phase_accum = 0;

    let in_l = [0.0f32; 64];
    let in_r = [0.0f32; 64];
    let mut out_l = [0.0f32; 64];
    let mut out_r = [0.0f32; 64];

    let n = core
        .process_static_stereo(&in_l, &in_r, &mut out_l, &mut out_r)
        .samples_written;
    assert!(n > 0);
}

#[test]
fn test_resampler_micro_soak() {
    let rate_pairs = [
        (44100, 48000),
        (48000, 44100),
        (96000, 48000),
        (22050, 48000),
        (88200, 48000),
    ];

    let chunk_size = 512;
    let n_iterations = 5_000;

    let in_l = vec![0.1f32; chunk_size];
    let in_r = vec![0.1f32; chunk_size];
    let mut out_l = vec![0.0f32; chunk_size * 4];
    let mut out_r = vec![0.0f32; chunk_size * 4];

    for (from, to) in rate_pairs {
        let mut rs = NamResampler::new(from, to, chunk_size).unwrap();

        for _ in 0..n_iterations {
            let n = rs
                .process_input(&in_l, &in_r, &mut out_l, &mut out_r)
                .samples_written;

            for i in 0..n {
                assert!(out_l[i].is_finite());
                assert!(out_r[i].is_finite());
            }

            if let Some(ref core) = rs.inner {
                let num_phases_fp = (NUM_PHASES as u64) << 40;
                assert!(
                    core.phase_accum < num_phases_fp + core.phase_step * 2,
                    "Overflow detected in {}->{}",
                    from,
                    to
                );
            }
        }
    }
}

#[test]
fn test_resampler_snr_against_reference() {
    // SNR of minimum-phase polyphase resampler against libsoxr reference.
    //
    // With TAPS_PER_PHASE=64: ~31 dB SNR in the passband.
    // The minimum-phase polyphase architecture with per-phase normalization
    // has inherent ripple (~0.06 dB cepstrum + per-phase gain dispersion)
    // that limits SNR. The linear-phase variant (test_resampler_linear_snr)
    // achieves significantly better performance.
    //
    // Gate elevated from 20 dB to 25 dB (with margin).

    let rate_pairs: &[(u32, u32)] = &[
        (44100, 48000),
        (48000, 44100),
        (48000, 96000),
        (96000, 48000),
    ];

    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture_dir = manifest_dir.join("tests").join("fixtures");
    let num_tones = 10usize;

    for &(from_rate, to_rate) in rate_pairs {
        let input_path = fixture_dir.join(format!("resampler_input_{}.f32", from_rate));
        let ref_path = fixture_dir.join(format!("resampler_ref_{}_to_{}.f32", from_rate, to_rate));
        assert!(
            input_path.exists(),
            "Missing: {}. Run generate_resampler_reference.py",
            input_path.display()
        );
        assert!(
            ref_path.exists(),
            "Missing: {}. Run generate_resampler_reference.py",
            ref_path.display()
        );

        let input = read_raw_f32(&input_path);
        let reference = read_raw_f32(&ref_path);

        let chunk_size = input.len().max(256);
        let mut resampler = NamResampler::new(from_rate, to_rate, chunk_size)
            .expect("Failed to create NamResampler");

        let output_capacity =
            ((input.len() as f64 * to_rate as f64 / from_rate as f64).ceil() as usize) + 128;
        let mut out_l = vec![0.0f32; output_capacity];
        let mut out_r = vec![0.0f32; output_capacity];
        let produced = resampler
            .process_input_mono(&input, &mut out_l, &mut out_r)
            .samples_written;
        assert!(produced > 0, "No output for {}->{}", from_rate, to_rate);
        let output = &out_l[..produced];

        let trim = 4096usize;
        let ref_trim = &reference[trim..reference.len().saturating_sub(trim)];
        let out_trim = &output[trim..output.len().saturating_sub(trim)];

        let freqs = log_spaced_tones(from_rate, num_tones);
        let passband_tones = &freqs[..freqs.len() - 1];

        let mut sig_sum_sq = 0.0f64;
        let mut err_sum_sq = 0.0f64;

        for &f in passband_tones {
            let ref_mag = goertzel_magnitude(ref_trim, f, to_rate);
            let out_mag = goertzel_magnitude(out_trim, f, to_rate);

            sig_sum_sq += (ref_mag as f64).powi(2);
            let diff = ref_mag as f64 - out_mag as f64;
            err_sum_sq += diff * diff;
        }

        let snr = if err_sum_sq > 0.0 {
            10.0 * (sig_sum_sq / err_sum_sq).log10()
        } else {
            f64::INFINITY
        };

        assert!(
            snr >= 25.0,
            "{}->{}: multitone SNR {:.1} dB (sig={:.3e}, err={:.3e}), expected >= 25 dB",
            from_rate,
            to_rate,
            snr,
            sig_sum_sq,
            err_sum_sq
        );
    }
}

#[test]
fn test_resampler_linear_snr() {
    // SNR of linear-phase polyphase resampler.
    // Linear-phase: symmetric impulse, uniform per-phase gains,
    // linear interpolation between adjacent phases is accurate.

    let rate_pairs: &[(u32, u32)] = &[
        (44100, 48000),
        (48000, 44100),
        (48000, 96000),
        (96000, 48000),
    ];

    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture_dir = manifest_dir.join("tests").join("fixtures");
    let num_tones = 10usize;

    for &(from_rate, to_rate) in rate_pairs {
        let input_path = fixture_dir.join(format!("resampler_input_{}.f32", from_rate));
        let ref_path = fixture_dir.join(format!("resampler_ref_{}_to_{}.f32", from_rate, to_rate));
        assert!(
            input_path.exists(),
            "Missing fixture for {}->{}",
            from_rate,
            to_rate
        );
        assert!(
            ref_path.exists(),
            "Missing fixture for {}->{}",
            from_rate,
            to_rate
        );

        let input = read_raw_f32(&input_path);
        let reference = read_raw_f32(&ref_path);

        let chunk_size = input.len().max(256);
        let mut resampler = NamResampler::new_linear(from_rate, to_rate, chunk_size)
            .expect("Failed to create linear-phase NamResampler");

        let output_capacity =
            ((input.len() as f64 * to_rate as f64 / from_rate as f64).ceil() as usize) + 128;
        let mut out_l = vec![0.0f32; output_capacity];
        let mut out_r = vec![0.0f32; output_capacity];
        let produced = resampler
            .process_input_mono(&input, &mut out_l, &mut out_r)
            .samples_written;
        assert!(
            produced > 0,
            "No output for {}->{} (linear)",
            from_rate,
            to_rate
        );
        let output = &out_l[..produced];

        let trim = 4096usize;
        let ref_trim = &reference[trim..reference.len().saturating_sub(trim)];
        let out_trim = &output[trim..output.len().saturating_sub(trim)];

        let freqs = log_spaced_tones(from_rate, num_tones);
        let passband_tones = &freqs[..freqs.len() - 1];

        let mut sig_sum_sq = 0.0f64;
        let mut err_sum_sq = 0.0f64;

        for &f in passband_tones {
            let ref_mag = goertzel_magnitude(ref_trim, f, to_rate);
            let out_mag = goertzel_magnitude(out_trim, f, to_rate);

            sig_sum_sq += (ref_mag as f64).powi(2);
            let diff = ref_mag as f64 - out_mag as f64;
            err_sum_sq += diff * diff;
        }

        let snr = if err_sum_sq > 0.0 {
            10.0 * (sig_sum_sq / err_sum_sq).log10()
        } else {
            f64::INFINITY
        };

        assert!(
            snr >= 25.0,
            "{}->{} (linear): multitone SNR {:.1} dB, expected >= 25 dB",
            from_rate,
            to_rate,
            snr
        );
    }
}

#[test]
fn test_linear_phase_roundtrip() {
    let chunk = 512usize;
    let mut rs = NamResampler::new_linear(96_000, 48_000, chunk).expect("new_linear failed");
    assert!(!rs.is_bypass());

    let input: Vec<f32> = (0..chunk)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 96_000.0).sin())
        .collect();
    let input_r = input.clone();

    let mut mid = vec![0.0f32; chunk];
    let mut mid_r = vec![0.0f32; chunk];
    let n_mid = rs
        .process_input(&input, &input_r, &mut mid, &mut mid_r)
        .samples_written;
    assert!(n_mid > 0);

    let mut out = vec![0.0f32; chunk];
    let mut out_r = vec![0.0f32; chunk];
    let n_out = rs
        .process_output(&mid[..n_mid], &mid_r[..n_mid], &mut out, &mut out_r)
        .samples_written;
    assert!(n_out > 0);

    let energy_in: f32 = input.iter().map(|x| x * x).sum();
    let energy_out: f32 = out[..n_out].iter().map(|x| x * x).sum();
    assert!(
        energy_out > energy_in * 0.05,
        "Linear-phase roundtrip energy collapsed: in={energy_in:.4}, out={energy_out:.4}"
    );
}

#[test]
fn test_latency_calculation() {
    let rs_bypass = NamResampler::new(48_000, 48_000, 256).unwrap();
    assert_eq!(rs_bypass.latency_samples(48_000), 0);

    let rate_pairs = [
        (44_100, 48_000),
        (48_000, 44_100),
        (96_000, 48_000),
        (48_000, 96_000),
    ];

    for &(pw, nam) in &rate_pairs {
        let rs_min = NamResampler::new(pw, nam, 256).unwrap();
        let rs_lin = NamResampler::new_linear(pw, nam, 256).unwrap();

        let lat_min = rs_min.latency_samples(pw);
        let lat_lin = rs_lin.latency_samples(pw);

        assert!(
            lat_min > 0,
            "{pw}->{nam}: min-phase latency must be positive, got {lat_min}"
        );

        // Linear-phase: each stage adds TAPS_PER_PHASE/2 = 32 samples,
        // outer stage rate-converted: 32 * host_rate / nam_rate
        let expected_lin = (32.0 + 32.0 * pw as f64 / nam as f64).round() as u32;
        assert_eq!(
            lat_lin, expected_lin,
            "{pw}->{nam}: linear-phase latency mismatch: got {lat_lin}, expected {expected_lin}"
        );

        assert!(
            lat_min < lat_lin,
            "{pw}->{nam}: min-phase latency ({lat_min}) must be less than linear-phase ({lat_lin})"
        );
    }

    // Verify host_rate parameter is ignored (now deprecated, must not panic)
    assert_eq!(rs_bypass.latency_samples(0), 0);
}

#[test]
fn test_resampler_mono_equivalence() {
    {
        let mut rs = NamResampler::new(48_000, 48_000, 256).unwrap();
        assert!(rs.is_bypass());

        let in_l = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out_l_stereo = [0.0f32; 5];
        let mut out_r_stereo = [0.0f32; 5];
        let mut out_l_mono = [0.0f32; 5];
        let mut out_r_mono = [0.0f32; 5];

        let n_stereo = rs
            .process_input(&in_l, &in_l, &mut out_l_stereo, &mut out_r_stereo)
            .samples_written;
        let n_mono = rs
            .process_input_mono(&in_l, &mut out_l_mono, &mut out_r_mono)
            .samples_written;

        assert_eq!(n_stereo, n_mono);
        assert_eq!(out_l_stereo, out_l_mono);
        assert_eq!(out_r_stereo, out_r_mono);
        assert_eq!(out_l_mono, out_r_mono);

        let n_out_stereo = rs
            .process_output(&in_l, &in_l, &mut out_l_stereo, &mut out_r_stereo)
            .samples_written;
        let n_out_mono = rs
            .process_output_mono(&in_l, &mut out_l_mono, &mut out_r_mono)
            .samples_written;

        assert_eq!(n_out_stereo, n_out_mono);
        assert_eq!(out_l_stereo, out_l_mono);
        assert_eq!(out_r_stereo, out_r_mono);
    }

    {
        let chunk = 256;
        let mut rs_stereo = NamResampler::new(44_100, 48_000, chunk).unwrap();
        let mut rs_mono = NamResampler::new(44_100, 48_000, chunk).unwrap();

        let in_l: Vec<f32> = (0..chunk).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut out_l_stereo = vec![0.0f32; chunk * 2];
        let mut out_r_stereo = vec![0.0f32; chunk * 2];
        let mut out_l_mono = vec![0.0f32; chunk * 2];
        let mut out_r_mono = vec![0.0f32; chunk * 2];

        let n_stereo = rs_stereo
            .process_input(&in_l, &in_l, &mut out_l_stereo, &mut out_r_stereo)
            .samples_written;
        let n_mono = rs_mono
            .process_input_mono(&in_l, &mut out_l_mono, &mut out_r_mono)
            .samples_written;

        assert_eq!(n_stereo, n_mono);
        for i in 0..n_stereo {
            assert!(
                (out_l_stereo[i] - out_l_mono[i]).abs() < 1e-4,
                "Mismatch L at index {}",
                i
            );
            assert!(
                (out_r_stereo[i] - out_r_mono[i]).abs() < 1e-4,
                "Mismatch R at index {}",
                i
            );
            assert_eq!(out_l_mono[i], out_r_mono[i]);
        }

        let mut rs_out_stereo = NamResampler::new(48_000, 44_100, chunk).unwrap();
        let mut rs_out_mono = NamResampler::new(48_000, 44_100, chunk).unwrap();

        let in_mid = &out_l_stereo[..n_stereo];
        let mut out_final_l_stereo = vec![0.0f32; chunk * 2];
        let mut out_final_r_stereo = vec![0.0f32; chunk * 2];
        let mut out_final_l_mono = vec![0.0f32; chunk * 2];
        let mut out_final_r_mono = vec![0.0f32; chunk * 2];

        let n_final_stereo = rs_out_stereo
            .process_output(
                in_mid,
                in_mid,
                &mut out_final_l_stereo,
                &mut out_final_r_stereo,
            )
            .samples_written;
        let n_final_mono = rs_out_mono
            .process_output_mono(in_mid, &mut out_final_l_mono, &mut out_final_r_mono)
            .samples_written;

        assert_eq!(n_final_stereo, n_final_mono);
        for i in 0..n_final_stereo {
            assert!(
                (out_final_l_stereo[i] - out_final_l_mono[i]).abs() < 1e-4,
                "Mismatch final L at index {}",
                i
            );
            assert!(
                (out_final_r_stereo[i] - out_final_r_mono[i]).abs() < 1e-4,
                "Mismatch final R at index {}",
                i
            );
            assert_eq!(out_final_l_mono[i], out_final_r_mono[i]);
        }
    }
}

#[test]
fn test_fixed_point_drift_random_ratios() {
    let ratios = [
        (44100.0, 48000.0),
        (48000.0, 44100.0),
        (96000.0, 48000.0),
        (88200.0, 48000.0),
    ];

    for &(from, to) in &ratios {
        let phase_step_f64 = (from / to) * NUM_PHASES as f64;
        let phase_step_u64 = (phase_step_f64 * ((1u64 << 40) as f64)).round() as u64;

        let mut accum_f64 = NUM_PHASES as f64;
        let mut accum_u64 = (NUM_PHASES as u64) << 40;

        let num_phases_fp = (NUM_PHASES as u64) << 40;

        for _ in 0..100_000 {
            while accum_f64 >= NUM_PHASES as f64 {
                accum_f64 -= NUM_PHASES as f64;
            }
            while accum_u64 >= num_phases_fp {
                accum_u64 -= num_phases_fp;
            }

            let phase_idx_f64 = accum_f64 as usize;
            let frac_f64 = accum_f64 - phase_idx_f64 as f64;

            let frac_bits = accum_u64 & ((1u64 << 40) - 1);
            let frac_u64 = frac_bits as f64 * (1.0 / (1u64 << 40) as f64);

            let diff = (frac_f64 - frac_u64).abs();
            assert!(
                diff < 1e-7,
                "Drift exceeds 1e-7: from={}, to={}, diff={}",
                from,
                to,
                diff
            );

            accum_f64 += phase_step_f64;
            accum_u64 += phase_step_u64;
        }
    }
}

// =============================================================================
// Edge-Case Tests — T2.1.3
// =============================================================================

#[test]
fn test_exact_output_buffer_size() {
    for &in_rate in &[22_050, 44_100, 48_000, 88_200, 96_000] {
        for &out_rate in &[22_050, 44_100, 48_000, 88_200, 96_000] {
            if in_rate == out_rate {
                continue;
            }
            let n_in: usize = 128;
            let exact = NamResampler::min_output_samples(n_in, in_rate, out_rate);
            let mut rs = NamResampler::new(in_rate, out_rate, 64).unwrap();

            let in_l = vec![0.5f32; n_in];
            let in_r = vec![0.5f32; n_in];
            let mut out_l = vec![0.0f32; exact];
            let mut out_r = vec![0.0f32; exact];

            let progress = rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
            assert_eq!(
                progress.samples_written, exact,
                "exact buffer ({in_rate}→{out_rate}): expected {exact} written, got {}",
                progress.samples_written
            );
            assert!(
                progress.samples_read <= n_in,
                "exact buffer: consumed {}/{} inputs",
                progress.samples_read,
                n_in
            );
            for i in 0..exact {
                assert!(out_l[i].is_finite(), "non-finite output at idx {i}");
                assert!(out_r[i].is_finite(), "non-finite output at idx {i}");
            }
        }
    }
}

#[test]
fn test_output_buffer_one_short() {
    for &in_rate in &[22_050, 44_100, 48_000, 88_200, 96_000] {
        for &out_rate in &[22_050, 44_100, 48_000, 88_200, 96_000] {
            if in_rate == out_rate {
                continue;
            }
            let n_in: usize = 256;
            let required = NamResampler::min_output_samples(n_in, in_rate, out_rate);
            if required < 2 {
                continue;
            }
            let capped = required - 1;

            let mut rs = NamResampler::new(in_rate, out_rate, 64).unwrap();

            let in_l: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.3).sin()).collect();
            let in_r: Vec<f32> = (0..n_in).map(|i| ((i + 1000) as f32 * 0.3).sin()).collect();

            let mut out_l = vec![0.0f32; capped];
            let mut out_r = vec![0.0f32; capped];

            let progress = rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);

            assert_eq!(
                progress.samples_written, capped,
                "one-short ({in_rate}→{out_rate}): expected {capped} written, got {}",
                progress.samples_written
            );
            assert!(
                progress.samples_read <= n_in,
                "one-short ({in_rate}→{out_rate}): consumed {} > available {n_in}",
                progress.samples_read
            );
            assert!(
                progress.samples_read > 0,
                "one-short ({in_rate}→{out_rate}): must consume at least 1 input"
            );

            for i in 0..capped {
                assert!(out_l[i].is_finite(), "non-finite output at idx {i}");
                assert!(out_r[i].is_finite(), "non-finite output at idx {i}");
            }
        }
    }
}

#[test]
fn test_output_buffer_zero_capacity_no_state_mutation() {
    for &in_rate in &[22_050, 44_100, 48_000, 88_200, 96_000] {
        for &out_rate in &[22_050, 44_100, 48_000, 88_200, 96_000] {
            if in_rate == out_rate {
                continue;
            }
            let mut rs = NamResampler::new(in_rate, out_rate, 64).unwrap();
            let phase_before = rs.inner.as_ref().map(|c| c.phase_accum);

            let in_l = vec![0.5f32; 128];
            let in_r = vec![0.5f32; 128];
            let mut out_l: [f32; 0] = [];
            let mut out_r: [f32; 0] = [];

            let progress = rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
            assert_eq!(
                progress.samples_read, 0,
                "zero-capacity ({in_rate}→{out_rate}): must not consume any input"
            );
            assert_eq!(
                progress.samples_written, 0,
                "zero-capacity ({in_rate}→{out_rate}): must not produce any output"
            );

            let phase_after = rs.inner.as_ref().map(|c| c.phase_accum);
            assert_eq!(
                phase_before, phase_after,
                "zero-capacity ({in_rate}→{out_rate}): state must be unchanged"
            );
        }
    }
}

#[test]
fn test_extreme_upsampling() {
    let cases = [(22_050, 96_000), (44_100, 192_000)];
    let n_in: usize = 512;

    for &(in_rate, out_rate) in &cases {
        let mut rs = NamResampler::new(in_rate, out_rate, 64).unwrap();
        let required = NamResampler::min_output_samples(n_in, in_rate, out_rate);

        let in_l: Vec<f32> = (0..n_in)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / in_rate as f32).sin())
            .collect();
        let in_r = in_l.clone();
        let mut out_l = vec![0.0f32; required];
        let mut out_r = vec![0.0f32; required];

        let progress = rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        assert!(progress.samples_written > 0);
        assert!(progress.samples_read > 0);

        let energy: f32 = out_l[..progress.samples_written]
            .iter()
            .map(|x| x * x)
            .sum();
        assert!(
            energy > 0.0,
            "extreme upsample {in_rate}→{out_rate}: no energy"
        );

        for i in 0..progress.samples_written {
            assert!(out_l[i].is_finite(), "non-finite sample at idx {i}");
            assert!(out_r[i].is_finite(), "non-finite sample at idx {i}");
        }
    }
}

#[test]
fn test_extreme_downsampling() {
    let cases = [(96_000, 22_050)];
    let n_in: usize = 512;

    for &(in_rate, out_rate) in &cases {
        let mut rs = NamResampler::new(in_rate, out_rate, 64).unwrap();
        let required = NamResampler::min_output_samples(n_in, in_rate, out_rate);

        let in_l: Vec<f32> = (0..n_in)
            .map(|i| (2.0 * std::f32::consts::PI * 500.0 * i as f32 / in_rate as f32).sin())
            .collect();
        let in_r = in_l.clone();
        let mut out_l = vec![0.0f32; required];
        let mut out_r = vec![0.0f32; required];

        let progress = rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        assert!(progress.samples_written > 0);
        assert!(progress.samples_read > 0);

        let energy: f32 = out_l[..progress.samples_written]
            .iter()
            .map(|x| x * x)
            .sum();
        assert!(
            energy > 0.0,
            "extreme downsample {in_rate}→{out_rate}: no energy"
        );

        for i in 0..progress.samples_written {
            assert!(out_l[i].is_finite(), "non-finite sample at idx {i}");
            assert!(out_r[i].is_finite(), "non-finite sample at idx {i}");
        }
    }
}

#[test]
fn test_phase_continuity_fragmented_buffers() {
    let rate_pairs = [
        (44_100, 48_000),
        (48_000, 44_100),
        (22_050, 48_000),
        (96_000, 48_000),
    ];

    for &(in_rate, out_rate) in &rate_pairs {
        let n_total = 4096;
        let freq = 440.0f32;
        let in_full: Vec<f32> = (0..n_total)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / in_rate as f32).sin())
            .collect();
        let in_full_r = in_full.clone();

        let required_single = NamResampler::min_output_samples(n_total, in_rate, out_rate);

        let mut rs_one = NamResampler::new(in_rate, out_rate, 64).unwrap();
        let mut out_single_l = vec![0.0f32; required_single];
        let mut out_single_r = vec![0.0f32; required_single];
        let progress_single =
            rs_one.process_input(&in_full, &in_full_r, &mut out_single_l, &mut out_single_r);
        let n_single = progress_single.samples_written;

        let mut rs_frag = NamResampler::new(in_rate, out_rate, 64).unwrap();
        let frag_size = 64;
        let max_frag_out =
            NamResampler::max_input_samples(required_single.max(8192), in_rate, out_rate)
                .clamp(256, 2048);
        let mut out_total = Vec::with_capacity(required_single);
        let mut out_total_r = Vec::with_capacity(required_single);
        let mut buf_l = vec![0.0f32; max_frag_out];
        let mut buf_r = vec![0.0f32; max_frag_out];

        let mut pos = 0;
        while pos < n_total {
            let chunk = frag_size.min(n_total - pos);
            let progress = rs_frag.process_input(
                &in_full[pos..pos + chunk],
                &in_full_r[pos..pos + chunk],
                &mut buf_l,
                &mut buf_r,
            );
            out_total.extend_from_slice(&buf_l[..progress.samples_written]);
            out_total_r.extend_from_slice(&buf_r[..progress.samples_written]);
            pos += chunk;
        }

        let common = n_single.min(out_total.len());
        assert!(
            common >= 16,
            "{in_rate}→{out_rate}: too few common samples (single={n_single}, frag={})",
            out_total.len()
        );

        let rmse: f32 = ((0..common)
            .map(|i| (out_single_l[i] - out_total[i]).powi(2))
            .sum::<f32>()
            / common as f32)
            .sqrt();
        let rms_ref: f32 = (out_single_l[..common]
            .iter()
            .map(|x| x.powi(2))
            .sum::<f32>()
            / common as f32)
            .sqrt();

        assert!(
            rmse < rms_ref * 0.15,
            "{in_rate}→{out_rate}: phase discontinuity detected (RMSE={rmse:.6}, ref_RMS={rms_ref:.6}, common={common})"
        );
    }
}

// =============================================================================
// Helper functions
// =============================================================================

fn log_spaced_tones(sample_rate: u32, num_tones: usize) -> Vec<f32> {
    let nyquist = sample_rate as f64 / 2.0;
    let f_start = 100.0f64;
    let f_end = 0.45 * nyquist;
    let log_start = f_start.log10();
    let log_end = f_end.log10();
    (0..num_tones)
        .map(|i| {
            10.0f64.powf(log_start + (log_end - log_start) * i as f64 / (num_tones - 1) as f64)
                as f32
        })
        .collect()
}

fn goertzel_magnitude(signal: &[f32], freq: f32, sample_rate: u32) -> f32 {
    let omega = 2.0 * std::f32::consts::PI * freq / sample_rate as f32;
    let coeff = 2.0 * omega.cos();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;

    for &sample in signal {
        let s2 = s1;
        s1 = s0;
        s0 = sample + coeff * s1 - s2;
    }

    let mag_sq = s1.powi(2) + s0.powi(2) - coeff * s1 * s0;
    if mag_sq < 0.0 { 0.0 } else { mag_sq.sqrt() }
}

const TEST_RATES: &[u32] = &[22_050, 44_100, 48_000, 88_200, 96_000, 192_000];
const TEST_BUFFER_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

#[test]
fn test_min_output_samples_empirical_sweep() {
    for &in_rate in TEST_RATES {
        for &out_rate in TEST_RATES {
            if in_rate == out_rate {
                continue;
            }
            let mut rs = NamResampler::new(in_rate, out_rate, 128).expect("new failed");
            for &n_in in TEST_BUFFER_SIZES {
                let min_out = NamResampler::min_output_samples(n_in, in_rate, out_rate);
                let in_l = vec![0.0f32; n_in];
                let in_r = vec![0.0f32; n_in];
                let mut out_l = vec![0.0f32; min_out];
                let mut out_r = vec![0.0f32; min_out];
                let written = rs
                    .process_input(&in_l, &in_r, &mut out_l, &mut out_r)
                    .samples_written;
                assert!(
                    written <= min_out,
                    "in_rate={} out_rate={} n_in={} min_out={} written={}",
                    in_rate,
                    out_rate,
                    n_in,
                    min_out,
                    written
                );
            }
        }
    }
}

#[test]
fn test_max_input_samples_empirical_sweep() {
    for &in_rate in TEST_RATES {
        for &out_rate in TEST_RATES {
            if in_rate == out_rate {
                continue;
            }
            let mut rs = NamResampler::new(in_rate, out_rate, 128).expect("new failed");
            for &cap in TEST_BUFFER_SIZES {
                let max_in = NamResampler::max_input_samples(cap, in_rate, out_rate);
                if max_in == 0 {
                    continue;
                }
                let in_l = vec![0.0f32; max_in];
                let in_r = vec![0.0f32; max_in];
                let mut out_l = vec![0.0f32; cap];
                let mut out_r = vec![0.0f32; cap];
                let written = rs
                    .process_input(&in_l, &in_r, &mut out_l, &mut out_r)
                    .samples_written;
                assert!(
                    written <= cap,
                    "in_rate={} out_rate={} cap={} max_in={} written={}",
                    in_rate,
                    out_rate,
                    cap,
                    max_in,
                    written
                );
            }
        }
    }
}

#[test]
fn test_helpers_reciprocal_consistency() {
    for &in_rate in TEST_RATES {
        for &out_rate in TEST_RATES {
            if in_rate == out_rate {
                continue;
            }
            for &n_in in TEST_BUFFER_SIZES {
                let min_out = NamResampler::min_output_samples(n_in, in_rate, out_rate);
                let max_from_out = NamResampler::max_input_samples(min_out, in_rate, out_rate);
                assert!(
                    max_from_out >= n_in,
                    "reciprocal fail: in_rate={} out_rate={} n_in={} min_out={} max_in={}",
                    in_rate,
                    out_rate,
                    n_in,
                    min_out,
                    max_from_out
                );
            }
        }
    }
}

#[test]
fn test_min_output_samples_zero_input() {
    for &in_rate in TEST_RATES {
        for &out_rate in TEST_RATES {
            let m = NamResampler::min_output_samples(0, in_rate, out_rate);
            assert_eq!(m, 0, "zero input should require zero output");
        }
    }
}

#[test]
fn test_max_input_samples_zero_capacity() {
    for &in_rate in TEST_RATES {
        for &out_rate in TEST_RATES {
            let m = NamResampler::max_input_samples(0, in_rate, out_rate);
            assert_eq!(m, 0, "zero capacity should yield zero max input");
        }
    }
}

#[test]
fn test_helpers_overflow_protection() {
    let huge = NamResampler::min_output_samples(usize::MAX, 192_000, 4_000);
    assert_eq!(huge, usize::MAX, "overflow must saturate at usize::MAX");

    let huge_inv = NamResampler::max_input_samples(usize::MAX, 192_000, 4_000);
    assert_eq!(huge_inv, usize::MAX, "overflow must saturate at usize::MAX");
}

fn read_raw_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("Failed to read fixture file");
    assert!(
        bytes.len().is_multiple_of(4),
        "Fixture {} has invalid size ({} bytes, not multiple of 4)",
        path.display(),
        bytes.len()
    );
    let n = bytes.len() / 4;
    let mut samples = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    samples
}
