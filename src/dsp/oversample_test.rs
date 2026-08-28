// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::common::spsc::{RT_STATUS_HOST_CONTRACT_VIOLATION, RtStatusFlags};
use crate::dsp::stage::HalfBandFilter;

#[test]
fn test_oversample_off_bypass() {
    let mut engine = OversampleEngine::new(OversampleFactor::Off, 256).unwrap();
    assert!(engine.is_bypass());

    let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    let mut up = vec![0.0f32; 64];
    let n_up = engine.upsample(&input, &mut up, None);
    assert_eq!(n_up, 64);
    assert_eq!(&up[..64], &input[..64]);

    let mut down = vec![0.0f32; 64];
    let n_down = engine.downsample(&up, &mut down, None);
    assert_eq!(n_down, 64);
    assert_eq!(&down[..64], &input[..64]);
}

#[test]
fn test_x2_upsample_dc() {
    let mut engine = OversampleEngine::new(OversampleFactor::X2, 256).unwrap();
    assert!(!engine.is_bypass());
    assert_eq!(engine.factor(), OversampleFactor::X2);

    let input = vec![1.0f32; 64];
    let mut up = vec![0.0f32; 128];
    let n_up = engine.upsample(&input, &mut up, None);
    assert_eq!(n_up, 128);

    // DC gain should be ~1.0 (half-band has flat passband at DC)
    // Check the middle of the buffer (after filter warm-up)
    let mid: Vec<f32> = up[50..78].to_vec();
    let avg: f32 = mid.iter().sum::<f32>() / mid.len() as f32;
    assert!(
        (avg - 1.0).abs() < 0.01,
        "DC gain after upsampling should be ~1.0, got {avg}"
    );
}

#[test]
fn test_x2_roundtrip_dc() {
    let mut engine = OversampleEngine::new(OversampleFactor::X2, 256).unwrap();
    let input = vec![1.0f32; 256];
    let mut up = vec![0.0f32; 512];
    let mut down = vec![0.0f32; 256];

    let n_up = engine.upsample(&input, &mut up, None);
    assert_eq!(n_up, 512);
    let mut model_out = vec![0.0f32; 512];
    model_out[..n_up].copy_from_slice(&up[..n_up]);
    let n_down = engine.downsample(&model_out[..n_up], &mut down, None);
    // Filter delay: (512 - 24) / 2 = 244
    assert!(
        (240..=256).contains(&n_down),
        "Expected ~244-256 samples, got {n_down}"
    );

    // Round-trip DC gain should be ~1.0 (half-band interp + decim)
    let valid = &down[50..200];
    let avg: f32 = valid.iter().sum::<f32>() / valid.len() as f32;
    assert!(
        (avg - 1.0).abs() < 0.02,
        "Round-trip DC gain should be ~1.0, got {avg}"
    );
}

#[test]
fn test_x2_aliasing_rejection() {
    let mut engine = OversampleEngine::new(OversampleFactor::X2, 512).unwrap();
    let sr = 48000.0f32;
    let freq = 23000.0f32;
    let n = 128;
    let input: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
        .collect();

    let mut up = vec![0.0f32; n * 2];
    let mut down = vec![0.0f32; n];

    let n_up = engine.upsample(&input, &mut up, None);
    // Pass-through model
    let mut model_out = vec![0.0f32; n * 2];
    model_out[..n_up].copy_from_slice(&up[..n_up]);

    let n_down = engine.downsample(&model_out[..n_up], &mut down, None);

    // Energy in the decimated output (steady state) should be near zero
    // because the half-band attenuates >100 dB in stopband.
    // But we're near Nyquist (23k/24k = 0.958), which is in the transition band.
    // The filter start-stop band is at 0.45*fs_out relative to the oversampled rate.
    // At 2× rate: Nyquist = 48 kHz. Half-band cutoff = 24 kHz (at 2× rate).
    // Wait — half-band has cutoff at 0.25*fs_in (input is at fs → 2× output fs's Nyquist is fs.
    // The half-band passes [0, 0.225*2fs] ≈ [0, 0.45fs_out] relative to input.
    // For 48 kHz input: 23 kHz is 23/48 = 0.479 → in the transition band.
    // Let's check attenuation: it should be significant but not >100 dB at this point.

    let start = 50;
    let end = n_down.min(n) - 10;
    let amp_in: f32 = input[start..end]
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    let amp_out: f32 = down[start..end]
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);

    // At 23 kHz (0.479 * fs_in), the half-band transition band provides
    // some (not full-stopband) attenuation — this is the transition band,
    // not the stopband, so a >10 dB expectation was never physically
    // achievable and this test hung/failed for unrelated reasons for a
    // long time before that was noticed (see
    // docs/postmortem-libm-symbol-interposition.md — the hang was an ELF
    // symbol-interposition bug in `f32::log10()`'s linkage, unrelated to
    // this algorithm).
    // Measured: -5.751028 dB (2026-07-04, X2, 23 kHz/48 kHz, deterministic
    // FIR — no noise/randomness in this computation). Margin below the
    // measured value only, to still catch a genuine regression in the
    // half-band coefficients or convolution.
    let ratio = if amp_in > 1e-6 {
        20.0 * (amp_out / amp_in).log10()
    } else {
        -200.0
    };
    assert!(
        ratio < -5.5,
        "23 kHz tone should be attenuated by half-band transition band, got {ratio:.3} dB (expected < -5.5 dB, measured baseline: -5.751028 dB)"
    );
}

#[test]
fn test_x4_upsample_dc() {
    let mut engine = OversampleEngine::new(OversampleFactor::X4, 256).unwrap();
    assert!(!engine.is_bypass());
    assert_eq!(engine.factor(), OversampleFactor::X4);

    let input = vec![1.0f32; 64];
    let mut up = vec![0.0f32; 256];
    let n_up = engine.upsample(&input, &mut up, None);
    assert_eq!(n_up, 256);

    let mid: Vec<f32> = up[120..136].to_vec();
    let avg: f32 = mid.iter().sum::<f32>() / mid.len() as f32;
    assert!(
        (avg - 1.0).abs() < 0.05,
        "4× DC gain should be ~1.0, got {avg}"
    );
}

#[test]
fn test_x4_roundtrip_dc() {
    let mut engine = OversampleEngine::new(OversampleFactor::X4, 512).unwrap();
    let input = vec![1.0f32; 512];
    let mut up = vec![0.0f32; 2048];
    let mut down = vec![0.0f32; 512];

    let n_up = engine.upsample(&input, &mut up, None);
    assert_eq!(n_up, 2048);

    let mut model_out = vec![0.0f32; 2048];
    model_out[..n_up].copy_from_slice(&up[..n_up]);

    let n_down = engine.downsample(&model_out[..n_up], &mut down, None);
    assert!(
        n_down >= 400,
        "Expected at least 400 output samples, got {n_down}"
    );

    let valid = &down[150..400];
    let avg: f32 = valid.iter().sum::<f32>() / valid.len() as f32;
    assert!(
        (avg - 1.0).abs() < 0.05,
        "4× round-trip DC gain should be ~1.0, got {avg}"
    );
}

#[test]
fn test_latency_samples() {
    let off = OversampleEngine::new(OversampleFactor::Off, 256).unwrap();
    assert_eq!(off.latency_samples(), 0);
    assert!(off.is_bypass());

    let x2 = OversampleEngine::new(OversampleFactor::X2, 256).unwrap();
    assert_eq!(x2.latency_samples(), 12);
    assert!(!x2.is_bypass());

    let x4 = OversampleEngine::new(OversampleFactor::X4, 256).unwrap();
    assert_eq!(x4.latency_samples(), 24);
    assert!(!x4.is_bypass());
}

#[test]
fn test_factor_multiplier() {
    assert_eq!(OversampleFactor::Off.multiplier(), 1);
    assert_eq!(OversampleFactor::X2.multiplier(), 2);
    assert_eq!(OversampleFactor::X4.multiplier(), 4);
}

#[test]
fn test_factor_stage_count() {
    assert_eq!(OversampleFactor::Off.stage_count(), 0);
    assert_eq!(OversampleFactor::X2.stage_count(), 1);
    assert_eq!(OversampleFactor::X4.stage_count(), 2);
}

#[test]
fn test_x2_oversampling_factor_matches() {
    let engine = OversampleEngine::new(OversampleFactor::X2, 256).unwrap();
    assert_eq!(engine.factor(), OversampleFactor::X2);
    assert!(!engine.is_bypass());
}

#[test]
fn test_reset_clears_stage_state_and_matches_fresh() {
    // After driving the engine with a non-trivial signal, `reset()` must make
    // a subsequent round-trip bit-identical to a freshly constructed engine
    // (same factor/size) — the delay lines, phase counters and inter-stage
    // scratch are all cleared in-place (T4.3 / F-CLAP-010).
    for factor in [OversampleFactor::X2, OversampleFactor::X4] {
        let mut dirty = OversampleEngine::new(factor, 256).unwrap();
        let mut fresh = OversampleEngine::new(factor, 256).unwrap();

        // Pollute the dirty engine with a few round-trips of a ramp.
        let noise: Vec<f32> = (0..64).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
        let mut up = vec![0.0f32; 256];
        let mut down = vec![0.0f32; 64];
        for _ in 0..4 {
            let n_up = dirty.upsample(&noise, &mut up, None);
            dirty.downsample(&up[..n_up], &mut down, None);
        }
        // Odd-length input desynchronizes the downsampler phase — good coverage.
        let odd: Vec<f32> = (0..37).map(|i| (i as f32 * 0.03).cos()).collect();
        let n_up = dirty.upsample(&odd, &mut up, None);
        dirty.downsample(&up[..n_up], &mut down, None);

        dirty.reset();

        // Reference: a fresh engine processing the same odd-length input.
        let n_up_ref = fresh.upsample(&odd, &mut up, None);
        let mut ref_out = vec![0.0f32; 64];
        fresh.downsample(&up[..n_up_ref], &mut ref_out, None);

        let n_up = dirty.upsample(&odd, &mut up, None);
        let mut out = vec![0.0f32; 64];
        dirty.downsample(&up[..n_up], &mut out, None);

        assert_eq!(n_up, n_up_ref, "factor {factor:?}: upsample count mismatch");
        assert_eq!(
            out, ref_out,
            "factor {factor:?}: reset not bit-identical to fresh"
        );
    }
}

#[test]
fn test_back_to_back_roundtrips_x2() {
    // Multiple round-trips through the same engine should remain stable
    // (memoryless nonlinearity check).
    let mut engine = OversampleEngine::new(OversampleFactor::X2, 256).unwrap();
    let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.02).sin()).collect();

    let mut up = vec![0.0f32; 256];
    let mut down1 = vec![0.0f32; 128];
    let mut down2 = vec![0.0f32; 128];

    // First round-trip
    let n_up = engine.upsample(&input, &mut up, None);
    let mut model_out1 = vec![0.0f32; 256];
    model_out1[..n_up].copy_from_slice(&up[..n_up]);
    let n1 = engine.downsample(&model_out1[..n_up], &mut down1, None);

    // Second round-trip
    let n_up2 = engine.upsample(&down1[..n1], &mut up, None);
    let mut model_out2 = vec![0.0f32; 256];
    model_out2[..n_up2].copy_from_slice(&up[..n_up2]);
    let n2 = engine.downsample(&model_out2[..n_up2], &mut down2, None);

    assert!(n1 >= 100);
    assert!(n2 >= 100);

    // Both round-trips should produce similar outputs (no drift)
    let corr: f32 = down1[50..n1.min(n2) - 5]
        .iter()
        .zip(down2[50..n1.min(n2) - 5].iter())
        .map(|(a, b)| a * b)
        .sum();
    let e1: f32 = down1[50..n1 - 5].iter().map(|a| a * a).sum();
    assert!(
        corr / (e1 + 1e-10) > 0.95,
        "Multiple round-trips should be stable"
    );
}

#[test]
fn test_halfband_filter_coefficients() {
    let filter = HalfBandFilter::design(12.0, 2.0);

    // Center coefficient (h[12]) is 0.5 — we don't store it in coeffs[].
    // The odd coefficients (coeffs[0..11]) should all be non-zero.
    for (i, &c) in filter.coeffs.iter().enumerate() {
        assert!(
            c.abs() > 1e-6,
            "Odd coeff [{i}] should be non-zero, got {c}"
        );
    }

    // Coefficients should decay toward edges (low-pass filter)
    let max_coeff = filter.coeffs.iter().map(|c| c.abs()).fold(0.0f32, f32::max);
    assert!(
        max_coeff > 0.05,
        "Odd coefficients should have significant magnitude"
    );

    // Sum of odd coefficients: for a half-band interpolator (L=2),
    // the DC gain condition is: h[D] + Σ(h[odd]) = 2.0.
    // With h[D] = 1.0 (hardcoded in convolution), we expect Σ(h[odd]) = 1.0.
    let odd_sum: f32 = filter.coeffs.iter().sum();
    let dc_gain = 1.0 + odd_sum;
    assert!(
        (dc_gain - 2.0).abs() < 0.002,
        "Half-band DC gain should be ~2.0, got {dc_gain} (odd_sum={odd_sum})"
    );
}

#[test]
fn test_oversized_input_clamped_off() {
    let mut engine = OversampleEngine::new(OversampleFactor::Off, 64).unwrap();
    let input = vec![1.0f32; 96];
    let mut up = vec![0.0f32; 96];
    let n_up = engine.upsample(&input, &mut up, None);
    assert!(n_up <= 64);

    let mut down = vec![0.0f32; 96];
    let n_down = engine.downsample(&input, &mut down, None);
    assert!(n_down <= 64);
}

#[test]
fn test_oversized_input_clamped_x2() {
    let mut engine = OversampleEngine::new(OversampleFactor::X2, 64).unwrap();
    let input = vec![0.5f32; 96];
    let mut up = vec![0.0f32; 192];
    let n_up = engine.upsample(&input, &mut up, None);
    assert!(n_up <= 128, "expected n_up <= 128, got {n_up}");
}

#[test]
fn test_oversized_input_clamped_x4() {
    let mut engine = OversampleEngine::new(OversampleFactor::X4, 64).unwrap();
    let input = vec![0.5f32; 96];
    let mut up = vec![0.0f32; 384];
    let n_up = engine.upsample(&input, &mut up, None);
    assert!(n_up <= 256, "expected n_up <= 256, got {n_up}");
}

#[test]
fn test_oversized_downsample_clamped_x2() {
    let mut engine = OversampleEngine::new(OversampleFactor::X2, 64).unwrap();
    let input = vec![0.25f32; 96];
    let mut down = vec![0.0f32; 48];
    let n_down = engine.downsample(&input, &mut down, None);
    assert!(n_down <= 64, "expected n_down <= 64, got {n_down}");
}

#[test]
fn truncation_raises_host_contract_flag_upsample() {
    // F-12 / T2.4: input blocks larger than `max_samples` are clamped — the
    // truncation must raise RT_STATUS_HOST_CONTRACT_VIOLATION (no more silent
    // truncation).
    let mut engine = OversampleEngine::new(OversampleFactor::X2, 64).unwrap();
    let rt = RtStatusFlags::new();

    let input = vec![0.5f32; 128];
    let mut up = vec![0.0f32; 256];
    let n_up = engine.upsample(&input, &mut up, Some(&rt));

    assert_eq!(n_up, 128, "clamped to max_samples=64 × 2");
    assert!(rt.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));
}

#[test]
fn truncation_raises_host_contract_flag_downsample() {
    // F-12 / T2.4: input blocks larger than max_samples × multiplier are
    // clamped — the truncation must raise RT_STATUS_HOST_CONTRACT_VIOLATION.
    let mut engine = OversampleEngine::new(OversampleFactor::X2, 64).unwrap();
    let rt = RtStatusFlags::new();

    let input = vec![0.25f32; 256];
    let mut down = vec![0.0f32; 128];
    let n_down = engine.downsample(&input, &mut down, Some(&rt));

    // The half-band stage holds back HB_DELAY samples (group delay), so the
    // exact return count is stage-dependent; the contract is the clamp bound.
    assert!(n_down <= 64, "clamped to max_os=128 → ≤64 native samples");
    assert!(rt.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));
}

#[test]
fn contract_compliant_blocks_do_not_raise_flag() {
    // F-12 / T2.4: blocks within the negotiated maxima must not raise the flag.
    let mut engine = OversampleEngine::new(OversampleFactor::X2, 64).unwrap();
    let rt = RtStatusFlags::new();

    let input = vec![0.5f32; 64];
    let mut up = vec![0.0f32; 128];
    let _n_up = engine.upsample(&input, &mut up, Some(&rt));
    let mut down = vec![0.0f32; 64];
    let _n_down = engine.downsample(&up, &mut down, Some(&rt));

    assert!(!rt.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));
}

#[test]
fn undersized_output_upsample_clamps_and_flags_x2() {
    // F-05: an output buffer smaller than `input.len() * multiplier` must be
    // clamped (never written past) and raise RT_STATUS_HOST_CONTRACT_VIOLATION.
    // Verified in release builds too (`cargo test --release dsp::oversample`).
    let mut engine = OversampleEngine::new(OversampleFactor::X2, 256).unwrap();
    let rt = RtStatusFlags::new();

    let input = vec![0.5f32; 64];
    let mut up = vec![0.0f32; 32]; // needs 128 for a full 64-sample upsample
    let n_up = engine.upsample(&input, &mut up, Some(&rt));

    assert!(n_up <= 32, "clamped to output capacity, got {n_up}");
    assert!(rt.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));
}

#[test]
fn undersized_output_upsample_clamps_and_flags_x4() {
    let mut engine = OversampleEngine::new(OversampleFactor::X4, 256).unwrap();
    let rt = RtStatusFlags::new();

    let input = vec![0.5f32; 64];
    let mut up = vec![0.0f32; 64]; // needs 256 for a full 64-sample 4× upsample
    let n_up = engine.upsample(&input, &mut up, Some(&rt));

    assert!(n_up <= 64, "clamped to output capacity, got {n_up}");
    assert!(rt.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));
}

#[test]
fn undersized_output_downsample_clamps_and_flags_x2() {
    // F-05: output smaller than `input.len() / multiplier` must be clamped and
    // raise RT_STATUS_HOST_CONTRACT_VIOLATION (no silent truncation).
    let mut engine = OversampleEngine::new(OversampleFactor::X2, 256).unwrap();
    let rt = RtStatusFlags::new();

    let input = vec![0.25f32; 128];
    let mut down = vec![0.0f32; 8]; // needs 64 for a full 128-sample downsample
    let n_down = engine.downsample(&input, &mut down, Some(&rt));

    assert!(n_down <= 8, "clamped to output capacity, got {n_down}");
    assert!(rt.check_flag(RT_STATUS_HOST_CONTRACT_VIOLATION));
}

#[test]
fn test_x2_stage_phase_batched_reference_parity() {
    // T5.3: the phase-batched upsample/downsample must match a naive reference
    // implementation of the half-band delay-line algorithm across a sweep of
    // block sizes (even and odd), including the downsample leftover-tail path.
    //
    // The comparison is ULP-tolerant: the fused AVX2 8-lane + 4-lane FMA with a
    // single horizontal reduction reassociates the summation tree, so results
    // differ from the old sequential `hsum + scalar tail` by at most a few ULP.
    // The spectral/phase invariant is validated by the quality-dashboard
    // fidelity phases (aliasing rejection, DC and roundtrip tests below).
    use crate::dsp::oversample::HB_DELAY;
    use crate::dsp::stage::X2Stage;

    let n = HB_DELAY;
    let stage = X2Stage::new().unwrap();
    let up_coeffs = stage.up_filter.coeffs;
    let down_coeffs = stage.down_filter.coeffs;
    let up_center = stage.up_center;
    let down_center = stage.down_center;

    // ── Reference upsample (original write-then-advance algorithm) ──
    let ref_upsample = |input: &[f32], ring: &mut Vec<f32>, pos: &mut usize, out: &mut Vec<f32>| {
        for &x in input {
            let p = *pos;
            ring[p] = x;
            ring[p + n] = x;
            *pos = (p + 1) % n;
            let w = *pos;
            let even = ring[w + 5] * up_center;
            let mut odd = 0.0f32;
            for k in 0..up_coeffs.len() {
                odd += up_coeffs[k] * ring[w + k];
            }
            out.push(even);
            out.push(odd);
        }
    };

    // ── Reference downsample (original per-sample parity algorithm) ──
    let ref_downsample = |input: &[f32],
                          ring_e: &mut Vec<f32>,
                          ring_o: &mut Vec<f32>,
                          pos_e: &mut usize,
                          pos_o: &mut usize,
                          total: &mut u64,
                          out: &mut Vec<f32>| {
        for &x in input {
            if (*total & 1) == 0 {
                ring_e[*pos_e] = x;
                ring_e[*pos_e + HB_EVEN_LEN] = x;
                *pos_e = (*pos_e + 1) % HB_EVEN_LEN;
            } else {
                ring_o[*pos_o] = x;
                ring_o[*pos_o + HB_ODD_LEN] = x;
                *pos_o = (*pos_o + 1) % HB_ODD_LEN;
            }
            *total += 1;
            if *total >= HB_TAPS as u64 && (*total & 1) == 1 {
                let mut sum = ring_e[*pos_e + 6] * down_center;
                for k in 0..down_coeffs.len() {
                    sum += down_coeffs[k] * ring_o[*pos_o + k];
                }
                out.push(sum);
            }
        }
    };

    const HB_TAPS: usize = 25;
    const HB_EVEN_LEN: usize = HB_TAPS.div_ceil(2);
    const HB_ODD_LEN: usize = HB_TAPS / 2;

    let signal: Vec<f32> = (0..1024)
        .map(|i| (i as f32 * 0.003).sin() * 0.7 + (i as f32 * 0.0007).cos())
        .collect();

    // ── Upsample parity across block sizes 1..=37 ──
    for block in 1..=37usize {
        let mut s = X2Stage::new().unwrap();
        let mut up_out = Vec::new();
        let mut r_ring = vec![0.0f32; 2 * n];
        let mut r_pos = 0usize;
        let mut r_out = Vec::new();

        let mut start = 0;
        while start < signal.len() {
            let end = (start + block).min(signal.len());
            let chunk = &signal[start..end];
            let mut buf = vec![0.0f32; chunk.len() * 2];
            s.upsample(chunk, &mut buf);
            up_out.extend_from_slice(&buf[..chunk.len() * 2]);
            ref_upsample(chunk, &mut r_ring, &mut r_pos, &mut r_out);
            start = end;
        }

        assert_eq!(up_out.len(), r_out.len(), "upsample block={block}");
        for (i, (a, b)) in up_out.iter().zip(r_out.iter()).enumerate() {
            let err = (a - b).abs();
            assert!(
                err < 1e-5,
                "upsample mismatch at {i} block={block}: {a} vs {b} (err={err})"
            );
        }
    }

    // ── Downsample parity across block sizes 1..=37 ──
    for block in 1..=37usize {
        let mut s = X2Stage::new().unwrap();
        let mut down_out = Vec::new();
        let mut r_e = vec![0.0f32; HB_EVEN_LEN * 2];
        let mut r_o = vec![0.0f32; HB_ODD_LEN * 2];
        let mut r_pe = 0usize;
        let mut r_po = 0usize;
        let mut r_total = 0u64;
        let mut r_out = Vec::new();

        let mut start = 0;
        while start < signal.len() {
            let end = (start + block).min(signal.len());
            let chunk = &signal[start..end];
            let mut buf = vec![0.0f32; chunk.len()];
            let written = s.downsample(chunk, &mut buf);
            down_out.extend_from_slice(&buf[..written]);
            ref_downsample(
                chunk,
                &mut r_e,
                &mut r_o,
                &mut r_pe,
                &mut r_po,
                &mut r_total,
                &mut r_out,
            );
            start = end;
        }

        assert_eq!(down_out.len(), r_out.len(), "downsample block={block}");
        for (i, (a, b)) in down_out.iter().zip(r_out.iter()).enumerate() {
            let err = (a - b).abs();
            assert!(
                err < 1e-5,
                "downsample mismatch at {i} block={block}: {a} vs {b} (err={err})"
            );
        }
    }
}
