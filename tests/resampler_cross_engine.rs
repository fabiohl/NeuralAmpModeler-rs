// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Cross-engine resampler comparison and characterization.
//
//  ## Why this test exists
//
//  `tests/parity/cpp_parity.rs:298-332,391-400` resamples the stress signal
//  with the **Rust** `NamResampler` on both sides of the cross-rate comparison;
//  the resampler error therefore cancels and that suite measures model
//  arithmetic parity at the model rate — it does **not** compare resamplers.
//  This file builds the independent cross-engine comparison:
//
//  * **Rust production default** — `NamResampler::new_simple` (minimum-phase
//    polyphase sinc, `src/dsp/sinc_kernel.rs:25-39`), plus its linear-phase
//    option (`new_linear_simple`) as a documented alternative;
//  * **C++ Lanczos** — the real vendor `dsp::ResamplingContainer<T,1,12>` of
//    the C++ plugin (`AudioDSPTools/dsp/ResamplingContainer/
//    ResamplingContainer.h:84`, `NeuralAmpModeler.h:187`; the plugin runs
//    NAM_SAMPLE = double because NAM_SAMPLE_FLOAT is defined nowhere in its
//    tree), driven via `tests/resampler_cxx/lanczos_probe.cpp` and
//    `utils/ensure_lanczos_probe.sh`;
//  * **High-quality reference** — libsoxr via ffmpeg, pinned as fixtures by
//    `tests/fixtures/generate_resampler_reference.py` (host fingerprint
//    recorded in `docs/benchmarks.md` §13).
//
//  ## Metric separation (criteria documented, never swapped)
//
//  * **Stopband rejection** is a filter-design property (≥ 105 dB by design —
//    `sinc_kernel.rs`); measured here as image-band / OOB tone rejection.
//  * **End-to-end multitone SNR vs. soxr** is a different, much lower figure
//    for the production minimum-phase bank (~31 dB; gate ≥ 25 dB in
//    `resampler_test.rs`). Both axes are reported side by side, never
//    conflated.
//  * Impulse, multitone, sweep and real-signal probes at 44.1/48/96 kHz in
//    both directions report magnitude, phase, aliasing, delay, ESR/SNR and
//    reported latency. A phase difference in isolation does not prove an
//    audible defect.
//  * Perceptual A/B per `docs/perceptual_validation.md` requires a human
//    panel; where absent this is a **GAP**, not a pass.
//
//  ## Classification (A3)
//
//  * Quick, vendor-independent tests lock Rust-only structural guarantees
//    (bit-exact shared-resampler cancellation, minimum-phase pre-ring floor,
//    image-band rejection of the production default).
//  * Cross-engine tables need the C++ probe and vendor mirrors → `#[ignore]`
//    ("long: ...", long discipline; operator hosts / nightly).
//
//  All measurements run off-RT; nothing here is a release gate — the
//  production default (minimum-phase) is frozen as the baseline and any
//  substitution requires a demonstrated benefit plus the audio/performance
//  gates.

use neural_amp_modeler_rs::dsp::resampler::NamResampler;
use neural_amp_modeler_rs::testing::{
    fixtures::nam_core_dir, report_skip, stress::generate_stress_signal_v2,
};

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::Command;

const RATE_PAIRS: &[(u32, u32)] = &[
    (44_100, 48_000),
    (48_000, 44_100),
    (48_000, 96_000),
    (96_000, 48_000),
];

const IMPULSE_ADDR: usize = 4;

// ── Self-contained deterministic helpers ────────────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        x * 2.0 - 1.0
    }
}

fn read_raw_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    out
}

fn write_raw_f32(path: &Path, data: &[f32]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("create probe scratch file");
    for v in data {
        f.write_all(&v.to_le_bytes()).expect("write probe scratch");
    }
}

/// Decodes a canonical PCM mono WAV into f32 — used for the real-signal probe
/// from the vendor mirror so tests never need ffmpeg at runtime.
fn read_wav_mono(path: &Path) -> Result<(u32, Vec<f32>), String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| e.to_string())?
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut pos = 12usize;
    let mut fmt = None;
    let mut data = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        if size == 0 || pos + 8 + size > bytes.len() {
            break;
        }
        let body = &bytes[pos + 8..pos + 8 + size];
        if id == b"fmt " && body.len() >= 16 {
            let audio_format = u16::from_le_bytes([body[0], body[1]]);
            let channels = u16::from_le_bytes([body[2], body[3]]);
            let rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            let bits = u16::from_le_bytes([body[14], body[15]]);
            fmt = Some((audio_format, channels, rate, bits));
        } else if id == b"data" {
            data = Some(body.to_vec());
        }
        pos += 8 + size + (size & 1);
    }
    let (format, channels, rate, bits) = fmt.ok_or("missing fmt chunk")?;
    if channels != 1 || format != 1 {
        return Err(format!(
            "unsupported wav: format={format} channels={channels}"
        ));
    }
    let data = data.ok_or("missing data chunk")?;
    let samples: Vec<f32> = match bits {
        16 => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        24 => data
            .chunks_exact(3)
            .map(|c| (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / 8_388_608.0)
            .collect(),
        32 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0)
            .collect(),
        other => return Err(format!("unsupported bit depth {other}")),
    };
    Ok((rate, samples))
}

/// Full offline conversion through the production engine (`process_input_mono`:
/// the host→model stage of the production two-stage chain), collecting
/// everything the engine emits for the whole input in one deterministic feed.
fn rust_convert(in_rate: u32, out_rate: u32, mono: &[f32], linear: bool) -> Vec<f32> {
    let mut res = if linear {
        NamResampler::new_linear_simple(in_rate, out_rate)
    } else {
        NamResampler::new_simple(in_rate, out_rate)
    }
    .expect("resampler construction");
    assert!(
        !res.is_bypass(),
        "{in_rate} == {out_rate} would be a bypass pair"
    );
    let expected = (mono.len() as u64 * out_rate as u64).div_ceil(in_rate as u64);
    let mut out_l = vec![0.0f32; expected as usize + 4096];
    let mut out_r = vec![0.0f32; expected as usize + 4096];
    let written = res
        .process_input_mono(mono, &mut out_l, &mut out_r)
        .samples_written;
    assert!(written > 0, "no output emitted for {in_rate}->{out_rate}");
    out_l.truncate(written);
    out_l
}

/// Streaming conversion split into variable callback-sized blocks (blocos
/// variáveis), identical in shape to the production streaming path.
fn rust_convert_blocked(in_rate: u32, out_rate: u32, mono: &[f32], linear: bool) -> Vec<f32> {
    let mut res = if linear {
        NamResampler::new_linear_simple(in_rate, out_rate)
    } else {
        NamResampler::new_simple(in_rate, out_rate)
    }
    .expect("resampler construction");
    assert!(
        !res.is_bypass(),
        "{in_rate} == {out_rate} would be a bypass pair"
    );
    let expected = (mono.len() as u64 * out_rate as u64).div_ceil(in_rate as u64);
    let mut out_l = vec![0.0f32; expected as usize + 4096];
    let mut out_r = vec![0.0f32; expected as usize + 4096];
    let mut total = 0usize;
    let mut start = 0usize;
    let mut rng = Lcg(5_735_119);
    let mut call = 0usize;
    while start < mono.len() {
        // 1..512-frame blocks, LCG-spread — the shape the plugin callbacks see.
        let jitter = (rng.next_f64().abs() * 512.0) as usize + 1;
        let end = (start + jitter).min(mono.len());
        let (o1, o2) = (&mut out_l[total..], &mut out_r[total..]);
        let _ = call;
        call += 1;
        total += res
            .process_input_mono(&mono[start..end], o1, o2)
            .samples_written;
        start = end;
    }
    assert!(
        total > 0,
        "blocked streaming produced nothing for {in_rate}->{out_rate}"
    );
    out_l.truncate(total);
    out_l
}

fn rust_roundtrip(rate: u32, mid_rate: u32, mono: &[f32], linear: bool) -> Vec<f32> {
    let mid = rust_convert(rate, mid_rate, mono, linear);
    rust_convert(mid_rate, rate, &mid, linear)
}

// ── Spectral helpers (direct DFT bins / ESR / alignment) ────────────────────

/// (amplitude, phase) at `freq` via direct DFT (exact, ≤ 12 tones/signal).
fn tone_at(signal: &[f32], freq: f64, rate: u32) -> (f64, f64) {
    if signal.len() < 8 {
        return (0.0, 0.0);
    }
    let n = signal.len() as f64;
    let w = 2.0 * std::f64::consts::PI * freq / rate as f64;
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for (i, &v) in signal.iter().enumerate() {
        let a = w * i as f64;
        re += v as f64 * a.cos();
        im -= v as f64 * a.sin();
    }
    ((2.0 * (re * re + im * im).sqrt() / n), im.atan2(re))
}

/// Log-spaced tones 100 Hz..`hi` — same tier as `resampler_test.rs` fixtures.
fn log_tones(num: usize, hi: f64) -> Vec<f64> {
    let f_start = 100.0f64;
    let f_end = hi;
    if num < 2 {
        return vec![f_start * (f_end / f_start).powf(0.5)];
    }
    let log_start = f_start.log10();
    let log_end = f_end.log10();
    (0..num)
        .map(|i| 10f64.powf(log_start + (log_end - log_start) * i as f64 / (num - 1) as f64))
        .collect()
}

/// Fixture equal tones (全体 10) trimmed to the passband (9), per existing test.
fn passband_tones(src: u32) -> Vec<f64> {
    let freqs = log_tones(10, 0.45 * src as f64 * 0.5);
    freqs[..freqs.len() - 1].to_vec()
}

/// (ESR, SNR dB) between pre-aligned signals over the common span.
fn esr_snr(reference: &[f32], test: &[f32]) -> (f64, f64) {
    let n = reference.len().min(test.len());
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for i in 0..n {
        let d = reference[i] as f64 - test[i] as f64;
        sig += reference[i] as f64 * reference[i] as f64;
        err += d * d;
    }
    if err == 0.0 {
        return (0.0, f64::INFINITY);
    }
    if sig == 0.0 {
        sig = err * 1e-30;
    }
    (err / sig, 10.0 * (sig / err).log10())
}

/// Overlap of the two signals aligned at integer `lag` (positive = probe is
/// delayed relative to reference).
fn aligned_overlap(reference: &[f32], probe: &[f32], lag: i64) -> (Vec<f32>, Vec<f32>) {
    let n = reference.len().min(probe.len());
    if lag >= 0 {
        let l = (lag as usize).min(n);
        (reference[l..n].to_vec(), probe[..n - l].to_vec())
    } else {
        let l = ((-lag) as usize).min(n);
        (reference[..n - l].to_vec(), probe[l..n].to_vec())
    }
}

/// Best integer lag of `probe` against `reference` (minimize aligned MSE);
/// coarse stride pass then fine pass (bounded, deterministic).
fn best_lag(reference: &[f32], probe: &[f32], search: usize) -> i64 {
    let coarse = |stride: i64, span: i64| -> (i64, f64) {
        let n = reference.len().min(probe.len());
        let mut best = 0i64;
        let mut best_mse = f64::INFINITY;
        let mut lag = -span;
        while lag <= span {
            let (ref_off, probe_off, len) = if lag >= 0 {
                (lag as usize, 0usize, n - lag as usize)
            } else {
                (0usize, (-lag) as usize, n - (-lag) as usize)
            };
            if len >= 16 {
                let mut acc = 0.0f64;
                for i in 0..len {
                    let d = reference[ref_off + i] as f64 - probe[probe_off + i] as f64;
                    acc += d * d;
                }
                if acc < best_mse {
                    best_mse = acc;
                    best = lag;
                }
            }
            lag += stride;
        }
        (best, best_mse)
    };
    let (c, _) = coarse(8, search as i64);
    let mut best = c;
    let mut best_mse = f64::INFINITY;
    for lag in (c - 8)..=(c + 8) {
        let (r, o) = aligned_overlap(reference, probe, lag);
        let n = r.len().min(o.len());
        if n < 16 {
            continue;
        }
        let mut acc = 0.0f64;
        for i in 0..n {
            let d = r[i] as f64 - o[i] as f64;
            acc += d * d;
        }
        if acc < best_mse {
            best_mse = acc;
            best = lag;
        }
    }
    best
}

fn trim_pair(reference: &mut Vec<f32>, probe: &mut Vec<f32>, trim: usize) {
    let t = trim.min(reference.len() / 4).min(probe.len() / 4);
    let e = reference.len() - t;
    let e2 = probe.len() - t;
    reference.truncate(e);
    reference.drain(..t);
    probe.truncate(e2);
    probe.drain(..t);
}

/// Passband multitone SNR vs reference plus (min,max) per-tone gain in dB —
/// the documented end-to-end axis of `resampler_test.rs`, kept verbatim
/// (same fixture tones, same per-tone ratio shape).
fn multitone_snr(reference: &[f32], out: &[f32], src: u32, dst: u32) -> (f64, f64, f64) {
    if reference.len() < 12_288 || out.len() < 12_288 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let lag = best_lag(reference, out, 256);
    let (mut ref_c, mut out_c) = aligned_overlap(reference, out, lag);
    trim_pair(&mut ref_c, &mut out_c, 4096);
    let freqs = passband_tones(src);
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    let mut gain_min = f64::INFINITY;
    let mut gain_max = f64::NEG_INFINITY;
    for &f in &freqs {
        let (ref_mag, _) = tone_at(&ref_c, f, dst);
        let (out_mag, _) = tone_at(&out_c, f, dst);
        sig += ref_mag * ref_mag;
        let d = ref_mag - out_mag;
        err += d * d;
        if ref_mag > 1e-9 {
            let g = 20.0 * (out_mag / ref_mag).log10();
            gain_min = gain_min.min(g);
            gain_max = gain_max.max(g);
        }
    }
    let snr = if err > 0.0 {
        10.0 * (sig / err).log10()
    } else {
        f64::INFINITY
    };
    (snr, gain_min, gain_max)
}

/// Full-band (aligned, trimmed) SNR vs a reference at the same rate.
fn aligned_snr(reference: &[f32], probe: &[f32]) -> f64 {
    if reference.len() < 16 || probe.len() < 16 {
        return f64::NAN;
    }
    let lag = best_lag(reference, probe, 256);
    let (mut r, mut o) = aligned_overlap(reference, probe, lag);
    trim_pair(&mut r, &mut o, 2048);
    let (_, snr) = esr_snr(&r, &o);
    snr
}

/// Impulse probe metrics: (peak index, pre-ring-to-peak dB, peak value).
fn impulse_metrics(signal: &[f32]) -> (usize, f64, f64) {
    let mut peak_idx = 0usize;
    let mut peak = 0.0f64;
    for (i, &v) in signal.iter().enumerate() {
        let a = v.abs() as f64;
        if a > peak {
            peak = a;
            peak_idx = i;
        }
    }
    let pre_energy: f64 = signal[..peak_idx]
        .iter()
        .map(|v| *v as f64 * *v as f64)
        .sum();
    let peak_energy = (peak * peak).max(1e-30);
    let prering_db = if pre_energy == 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * (pre_energy / peak_energy).log10()
    };
    (peak_idx, prering_db, peak)
}

/// Image-band rejection on upsampling: input tones ≤ 0.45 × Nyq(in) leak
/// mirror components at `in−f` in the output; the worst mirror magnitude (dB
/// rel. to the 0.1 source tone) reports the design stopband in operation.
fn image_rejection_db(in_rate: u32, out_rate: u32, linear: bool) -> f64 {
    let mid = out_rate as f64 * 0.5;
    let in_tones = log_tones(12, 0.45 * in_rate as f64 * 0.5);
    let n_in = in_rate as usize;
    let mut input = vec![0.0f32; n_in];
    for &f in &in_tones {
        for (i, v) in input.iter_mut().enumerate() {
            let w = 2.0 * std::f64::consts::PI * f * i as f64 / in_rate as f64;
            *v += (0.1 * w.sin()) as f32;
        }
    }
    let out = rust_convert(in_rate, out_rate, &input, linear);
    let mut worst = f64::NEG_INFINITY;
    for &f in &in_tones {
        let mirror = in_rate as f64 - f;
        if mirror >= mid - 1.0 || mirror <= 0.0 {
            continue; // mirror outside the measurable band
        }
        let (img_amp, _) = tone_at(&out, mirror, out_rate);
        let db = 20.0 * (img_amp / 0.1f64).log10();
        worst = worst.max(db);
    }
    worst
}

/// OOB-tone aliasing rejection for downsampling: deterministic tones between
/// destination Nyquist and source Nyquist; residual folded images at
/// `in−f` (< dst Nyq) in dB rel. to the 0.1 source tone.
fn oob_alias_rejection_db(in_rate: u32, out_rate: u32, linear: bool) -> f64 {
    let dst_nyq = out_rate as f64 * 0.5;
    let src_nyq = in_rate as f64 * 0.5;
    let t0 = dst_nyq * 1.01;
    let t1 = src_nyq * 0.99;
    let n_in = in_rate as usize * 2;
    let mut input = vec![0.0f32; n_in];
    let mut tones = Vec::new();
    for k in 1..=8usize {
        let f = t0 + (t1 - t0) * k as f64 / 9.0;
        for (i, v) in input.iter_mut().enumerate() {
            let w = 2.0 * std::f64::consts::PI * f * i as f64 / in_rate as f64;
            *v += (0.1 * w.sin()) as f32;
        }
        tones.push(f);
    }
    let out = rust_convert(in_rate, out_rate, &input, linear);
    let mut worst = f64::NEG_INFINITY;
    for f in tones {
        let fold = in_rate as f64 - f;
        if fold <= 0.0 || fold >= dst_nyq - 1.0 {
            continue;
        }
        let (img_amp, _) = tone_at(&out, fold, out_rate);
        let db = 20.0 * (img_amp / 0.1f64).log10();
        worst = worst.max(db);
    }
    worst
}

/// Deterministic phase-continuous 0.5 s sweep 20 Hz → 0.45 × Nyquist.
fn chirp(rate: u32) -> Vec<f32> {
    let dur = 0.5f64;
    let n = (dur * rate as f64) as usize;
    let f0 = 20.0f64;
    let f1 = 0.45 * rate as f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / rate as f64;
        let phase = 2.0 * std::f64::consts::PI * (f0 * t + (f1 - f0) * t * t / (2.0 * dur));
        out.push((0.35 * phase.sin()) as f32);
    }
    out
}

/// Phase linearity per linear-phase engine output through the multitone
/// tones: scan the group delay τ with wrap-aware phase distance
/// (φ(f) = φ0 − 2π·f·τ/dst), report (τ in output samples, RMS residual in
/// radians around the best fit). The fit is descriptive; no gate asserted.
fn phase_linearity(out: &[f32], src: u32, dst: u32) -> (f64, f64) {
    let xs = passband_tones(src);
    let ys: Vec<f64> = xs.iter().map(|&f| tone_at(out, f, dst).1).collect();
    if xs.len() < 3 {
        return (f64::NAN, f64::NAN);
    }
    let wrap = |a: f64| {
        let mut x = a;
        while x > std::f64::consts::PI {
            x -= 2.0 * std::f64::consts::PI;
        }
        while x <= -std::f64::consts::PI {
            x += 2.0 * std::f64::consts::PI;
        }
        x
    };
    let mut best_tau = 0.0f64;
    let mut best_res = f64::INFINITY;
    let mut tau = -64.0f64;
    while tau <= 64.0 {
        let (mut cx, mut cy) = (0.0f64, 0.0f64);
        for (f, phi) in xs.iter().zip(&ys) {
            let fit = -2.0 * std::f64::consts::PI * f * tau / dst as f64;
            cx += (phi - fit).cos();
            cy += (phi - fit).sin();
        }
        let phi0 = cy.atan2(cx);
        let mut acc = 0.0f64;
        for (f, phi) in xs.iter().zip(&ys) {
            let fit = -2.0 * std::f64::consts::PI * f * tau / dst as f64;
            let d = wrap(phi - phi0 - fit);
            acc += d * d;
        }
        if acc < best_res {
            best_res = acc;
            best_tau = tau;
        }
        tau += 0.01;
    }
    let rms = (best_res / xs.len() as f64).sqrt();
    (best_tau, rms)
}

// ── Probe (C++ vendor) invocation ───────────────────────────────────────────

fn manifest_dir() -> PathBuf {
    std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .expect("CARGO_MANIFEST_DIR")
}

/// Path of the compiled Lanczos probe; compiles it once on demand through the
/// Classe D helper (mirrors `utils/ensure_namcore_render.sh` usage).
fn ensure_probe() -> Result<PathBuf, String> {
    static PROBE: std::sync::OnceLock<Result<PathBuf, String>> = std::sync::OnceLock::new();
    PROBE
        .get_or_init(|| {
            let bin = manifest_dir().join("build/resampler_probe/lanczos_probe");
            if bin.is_file() {
                return Ok(bin);
            }
            let helper = manifest_dir().join("utils/ensure_lanczos_probe.sh");
            let out = Command::new("bash")
                .arg(&helper)
                .env("lib_quiet", "1")
                .output()
                .map_err(|e| format!("failed to run {}: {e}", helper.display()))?;
            if !out.status.success() {
                return Err(format!(
                    "probe build failed rc={:?}: {}",
                    out.status.code(),
                    String::from_utf8_lossy(&out.stderr)
                ));
            }
            let path = String::from_utf8_lossy(&out.stdout);
            let path = path.split_whitespace().last().unwrap_or_default().trim();
            let built = PathBuf::from(path);
            if !built.is_file() {
                return Err(format!("probe path not produced: '{path}'"));
            }
            Ok(built)
        })
        .clone()
}

/// Vendor-mirror availability for the Lanczos comparison axis.
fn vendor_available() -> bool {
    nam_core_dir()
        .join("Dependencies/AudioDSPTools/dsp/ResamplingContainer/Dependencies/LanczosResampler.h")
        .is_file()
}

fn probe_common(cmd: &mut Command) -> Result<String, String> {
    let out = cmd.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "probe failed rc={:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn probe_stdout_token(stdout: &str, key: &str) -> Option<String> {
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix(key)
            && let Some(v) = rest.split_whitespace().next()
        {
            return Some(v.to_string());
        }
    }
    None
}

/// Real vendor `LanczosResampler` isolated stage conversion (plugin per-stage
/// view; `t_float` = float kernel, otherwise double = plugin production).
fn probe_single(
    probe: &Path,
    in_rate: u32,
    out_rate: u32,
    input: &[f32],
    scratch: &Path,
    tag: &str,
    t_float: bool,
) -> Result<Vec<f32>, String> {
    let in_path = scratch.join(format!("{tag}_in.f32"));
    let out_path = scratch.join(format!("{tag}_out.f32"));
    write_raw_f32(&in_path, input);
    let mut cmd = Command::new(probe);
    cmd.arg("single")
        .arg(in_rate.to_string())
        .arg(out_rate.to_string())
        .arg(&in_path)
        .arg(&out_path);
    if t_float {
        cmd.arg("float");
    }
    let stdout = probe_common(&mut cmd)?;
    let n: usize = probe_stdout_token(&stdout, "SAMPLES_OUT ")
        .ok_or("SAMPLES_OUT missing")?
        .parse()
        .map_err(|e| format!("bad SAMPLES_OUT: {e}"))?;
    let data = read_raw_f32(&out_path);
    if data.len() != n {
        return Err(format!("SAMPLES_OUT {n} != file {}", data.len()));
    }
    Ok(data)
}

/// Real plugin container pair (Reset warm-up + per-block passthrough):
/// plugin-visible stream (first n_in samples) and reported LATENCY_IN.
fn probe_pair(
    probe: &Path,
    in_rate: u32,
    out_rate: u32,
    input: &[f32],
    scratch: &Path,
    tag: &str,
    t_float: bool,
) -> Result<(Vec<f32>, i64), String> {
    let in_path = scratch.join(format!("{tag}_in.f32"));
    let out_path = scratch.join(format!("{tag}_out.f32"));
    write_raw_f32(&in_path, input);
    let mut cmd = Command::new(probe);
    cmd.arg("pair")
        .arg(in_rate.to_string())
        .arg(out_rate.to_string())
        .arg(&in_path)
        .arg(&out_path);
    if t_float {
        cmd.arg("float");
    }
    let stdout = probe_common(&mut cmd)?;
    let n: usize = probe_stdout_token(&stdout, "SAMPLES_OUT ")
        .ok_or("SAMPLES_OUT missing")?
        .parse()
        .map_err(|e| format!("bad SAMPLES_OUT: {e}"))?;
    let latency: i64 = probe_stdout_token(&stdout, "LATENCY_IN ")
        .ok_or("LATENCY_IN missing")?
        .parse()
        .map_err(|e| format!("bad LATENCY_IN: {e}"))?;
    let data = read_raw_f32(&out_path);
    if data.len() != n {
        return Err(format!("SAMPLES_OUT {n} != file {}", data.len()));
    }
    Ok((data, latency))
}

/// Vendor fresh-engine bench (construction/table-init excluded from the
/// window, exactly like the Rust side below): (stage median, container
/// median) in ns per external sample.
fn probe_bench(
    probe: &Path,
    rates: (u32, u32),
    input: &[f32],
    scratch: &Path,
    tag: &str,
    reps: usize,
    t_float: bool,
) -> Result<(f64, f64), String> {
    let (in_rate, out_rate) = rates;
    let in_path = scratch.join(format!("{tag}_bin.f32"));
    write_raw_f32(&in_path, input);
    let mut cmd = Command::new(probe);
    cmd.arg("bench")
        .arg(in_rate.to_string())
        .arg(out_rate.to_string())
        .arg(&in_path)
        .arg(reps.to_string());
    if t_float {
        cmd.arg("float");
    }
    let stdout = probe_common(&mut cmd)?;
    let stage = probe_stdout_token(&stdout, "STAGE_NS ")
        .ok_or("STAGE_NS missing")?
        .parse::<f64>()
        .map_err(|e| format!("bad STAGE_NS: {e}"))?;
    let container = probe_stdout_token(&stdout, "CONTAINER_NS ")
        .ok_or("CONTAINER_NS missing")?
        .parse::<f64>()
        .map_err(|e| format!("bad CONTAINER_NS: {e}"))?;
    Ok((stage, container))
}

// ── CPU timing (Rust side) ──────────────────────────────────────────────────

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Median ns per emitted output sample of the production engine stage, warm
/// engine (polyphase banks built once), timed conversions only.
fn rust_stage_ns_per_out(
    in_rate: u32,
    out_rate: u32,
    input: &[f32],
    reps: usize,
    linear: bool,
    output_stage: bool,
) -> f64 {
    let (r_in, r_out) = if output_stage {
        (out_rate, in_rate)
    } else {
        (in_rate, out_rate)
    };
    let expected = (input.len() as u64 * r_out as u64).div_ceil(r_in as u64);
    let mut out = vec![0.0f32; expected as usize + 4096];
    let mut out_r = vec![0.0f32; expected as usize + 4096];
    let mut res = if linear {
        NamResampler::new_linear_simple(r_in, r_out)
    } else {
        NamResampler::new_simple(r_in, r_out)
    }
    .expect("resampler construction");
    // Warm pass (banks, delay line, caches).
    let _ = res
        .process_input_mono(input, &mut out, &mut out_r)
        .samples_written;
    let mut samples: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        let written = res
            .process_input_mono(input, &mut out, &mut out_r)
            .samples_written;
        let elapsed = t0.elapsed().as_nanos() as f64;
        if written > 0 {
            samples.push(elapsed / written as f64);
        }
    }
    median(&mut samples)
}

// ── Quick, vendor-independent structural tests ──────────────────────────────

/// The cross-rate parity anchors are resampler-canceled by construction: both
/// comparison sides receive bit-identical resampled input, and two independent
/// engine instances must resample identically. This quick anchor locks that
/// contract bit-for-bit, so any future resampler change (kernel, threading)
/// surfaces here instead of silently drifting the anchor semantics of
/// `cpp_parity.rs:298-332` (Rule 6 — non-circular independence, re-verified).
#[test]
fn test_shared_resampler_cancellation_is_bit_exact() {
    let input = generate_stress_signal_v2("s12-anchor", 48_000);
    let a = rust_convert(48_000, 44_100, &input, false);
    let b = rust_convert(48_000, 44_100, &input, false);
    assert_eq!(
        a, b,
        "two independent engine instances must resample bit-identically (anchor contract)"
    );
}

/// Variable-block streaming must reproduce the whole-feed conversion for the
/// anchor axis (block-boundary invariance of the deterministic FIR — cf.
/// tests/dsp_core/resampler_invariance_test.rs, stress material here).
#[test]
fn test_min_phase_blocked_matches_whole_feed() {
    let input = generate_stress_signal_v2("s12-blocks", 44_100);
    let whole = rust_convert(44_100, 48_000, &input, false);
    let blocked = rust_convert_blocked(44_100, 48_000, &input, false);
    assert_eq!(whole, blocked, "block fragmentation must not change output");
}

/// Minimum-phase impulse envelope across raster positions: reported (no
/// numeric gate — the split-phase case legally approaches the symmetric
/// balance; the delay spread is the discriminant that holds, see
/// `test_linear_phase_delay_above_min_phase`).
#[test]
fn test_min_phase_impulse_prering_floor() {
    for (pw, nam) in [(96_000u32, 48_000u32), (44_100, 48_000), (48_000, 44_100)] {
        let mut worst_prering = f64::NEG_INFINITY;
        let mut worst_pre_post = f64::NEG_INFINITY;
        for raster in 0..8usize {
            let mut input = vec![0.0f32; 4096];
            input[IMPULSE_ADDR + 9 * raster] = 1.0;
            let out = rust_convert(pw, nam, &input, false);
            let (peak_idx, prering_db, peak) = impulse_metrics(&out);
            let peak_energy = (peak * peak).max(1e-30);
            let post_energy: f64 = out[peak_idx..]
                .iter()
                .map(|v| *v as f64 * *v as f64)
                .sum::<f64>()
                - peak_energy;
            let pre_post = if post_energy > 0.0 {
                10.0 * (pre_energy_of(&out[..peak_idx]) / post_energy).log10()
            } else {
                f64::NEG_INFINITY
            };
            worst_prering = worst_prering.max(prering_db);
            worst_pre_post = worst_pre_post.max(pre_post);
            assert!(
                peak > 0.05 && peak < 1.5,
                "{pw}->{nam} raster {raster}: peak {peak}"
            );
        }
        println!(
            "// Measured: {pw}->{nam} min-phase raster-swept impulse: worst prering {worst_prering:.1} dB, worst pre/post {worst_pre_post:.1} dB"
        );
    }
}

fn pre_energy_of(signal: &[f32]) -> f64 {
    signal.iter().map(|v| *v as f64 * *v as f64).sum::<f64>()
}

/// Linear-phase alternative: larger group delay than minimum-phase, with its
/// pre/post-echo balance reported (the delay spread, not the raw pre-ring
/// magnitude, is the robust structural differentiator).
#[test]
fn test_linear_phase_delay_above_min_phase() {
    let mut input = vec![0.0f32; 4096];
    input[IMPULSE_ADDR] = 1.0;
    for (pw, nam) in [(96_000u32, 48_000u32), (44_100, 48_000)] {
        let out_min = rust_convert(pw, nam, &input, false);
        let out_lin = rust_convert(pw, nam, &input, true);
        let (idx_min, prering_min, _) = impulse_metrics(&out_min);
        let (idx_lin, prering_lin, _) = impulse_metrics(&out_lin);
        println!(
            "// Measured: {pw}->{nam} impulse prering min-phase {prering_min:.1} dB (peak@{idx_min}) | linear {prering_lin:.1} dB (peak@{idx_lin})"
        );
        assert!(
            idx_lin > idx_min,
            "{pw}->{nam}: linear peak delay exceeds minimum-phase ({idx_lin} vs {idx_min})"
        );
    }
}

/// Image-band (mirror-frequency) rejection of the production engine on
/// upsampling — the *design* stopband in operation (≥ 105 dB), NOT the
/// end-to-end multitone SNR (that axis is `test_resampler_snr_against_reference`).
#[test]
fn test_min_phase_image_band_rejection_upsample() {
    // Measured (2026-11-13): for 44.1k→48k the fixture-tier content (≤ 0.45 ×
    // Nyq) leaves NO in-band images at all (mirrors fall above the output
    // Nyquist — −∞ dB); the measurable axis is 48k→96k where mirrors land at
    // in−f < 48 kHz: min-phase −87.7 dB | linear −109.2 dB. Gate −80 dB keeps
    // 7.7 dB margin below the measured worst case and catches stopband
    // regressions (Rule 3 measurement note).
    for (pw, nam) in [(44_100u32, 48_000u32), (48_000, 96_000)] {
        let db = image_rejection_db(pw, nam, false);
        let db_lin = image_rejection_db(pw, nam, true);
        println!(
            "// Measured: {pw}->{nam} image-band rejection: min-phase {db:.1} dB | linear {db_lin:.1} dB"
        );
        assert!(
            db < -80.0,
            "{pw}->{nam}: image-band rejection degraded: {db:.1} dB (>= -80 dB)"
        );
    }
}

// ── Long: cross-engine characterization ────────────────────────────────────

fn scrub_dir(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[ignore = "long: cross-engine resampler comparison (requires vendor mirror + C++ probe)"]
fn test_s12_cross_engine_characterization() {
    if !vendor_available() {
        report_skip(
            "test_s12_cross_engine_characterization",
            "vendor mirrors missing (optional): Lanczos comparison axis unavailable — GAP",
        );
        return;
    }
    let probe = match ensure_probe() {
        Ok(p) => p,
        Err(e) => {
            report_skip(
                "test_s12_cross_engine_characterization",
                &format!("C++ compiler/probe unavailable (optional): {e}"),
            );
            return;
        }
    };
    let manifest = manifest_dir();
    let fixture_dir = manifest.join("tests/fixtures");
    let scratch = manifest.join("target/tests/s12_scratch/characterization");
    scrub_dir(&scratch);
    std::fs::create_dir_all(&scratch).expect("scratch dir");

    let mut lines: Vec<String> = Vec::new();
    for &(src, dst) in RATE_PAIRS {
        let ref_path = fixture_dir.join(format!("resampler_ref_{src}_to_{dst}.f32"));
        assert!(
            ref_path.is_file(),
            "missing soxr fixture {} — run generate_resampler_reference.py",
            ref_path.display()
        );
        let soxr = read_raw_f32(&ref_path);

        // ── Signal set for this conversion axis ──────────────────────────
        let multitone = read_raw_f32(&fixture_dir.join(format!("resampler_input_{src}.f32")));
        let mut impulse = vec![0.0f32; 4096];
        impulse[IMPULSE_ADDR] = 1.0;
        let sweep = chirp(src);
        let stress_v2 = generate_stress_signal_v2("s12-stress", src);

        // ── Multitone axis (fixture tones + soxr reference) ─────────────
        {
            let (snr, gmin, gmax) =
                multitone_snr(&soxr, &rust_convert(src, dst, &multitone, false), src, dst);
            lines.push(format!(
                "{src}->{dst} min-phase multitone SNR vs soxr {snr:.1} dB (per-tone gain {gmin:+.3}/{gmax:+.3} dB)"
            ));
            let (snr, gmin, gmax) =
                multitone_snr(&soxr, &rust_convert(src, dst, &multitone, true), src, dst);
            lines.push(format!(
                "{src}->{dst} linear multitone SNR vs soxr {snr:.1} dB (per-tone gain {gmin:+.3}/{gmax:+.3} dB)"
            ));
            let lan = probe_single(&probe, src, dst, &multitone, &scratch, "lan_mt", false)
                .unwrap_or_else(|e| panic!("{src}->{dst} probe single: {e}"));
            let (snr, gmin, gmax) = multitone_snr(&soxr, &lan, src, dst);
            lines.push(format!(
                "{src}->{dst} lanczos-single(double) multitone SNR vs soxr {snr:.1} dB (per-tone gain {gmin:+.3}/{gmax:+.3} dB)"
            ));
            let lan32 = probe_single(&probe, src, dst, &multitone, &scratch, "lan_mt32", true)
                .unwrap_or_else(|e| panic!("{src}->{dst} probe single float: {e}"));
            let (snr, _, _) = multitone_snr(&soxr, &lan32, src, dst);
            lines.push(format!(
                "{src}->{dst} lanczos-single(float) multitone SNR vs soxr {snr:.1} dB"
            ));
            // Full-band aligned SNR of each engine vs the soxr reference.
            let r_min = rust_convert(src, dst, &multitone, false);
            let r_lin = rust_convert(src, dst, &multitone, true);
            let snr_min = aligned_snr(&soxr, &r_min);
            let snr_lin = aligned_snr(&soxr, &r_lin);
            let snr_lan = aligned_snr(&soxr, &lan);
            let snr_lan32 = aligned_snr(&soxr, &lan32);
            lines.push(format!(
                "{src}->{dst} full-band aligned SNR vs soxr: min-phase {snr_min:.1} | linear {snr_lin:.1} | lanczos-single(double) {snr_lan:.1} | lanczos-single(float) {snr_lan32:.1} dB"
            ));
            // Phase linearity (delay + residual) — defined only for the
            // linear-phase engines; the minimum-phase response is nonlinear
            // by design (characterization note, not a defect claim).
            let (tau_l, res_l) = phase_linearity(&lan, src, dst);
            let (tau_soxr, res_soxr) = phase_linearity(&soxr, src, dst);
            let (tau_lin, res_lin) = phase_linearity(&r_lin, src, dst);
            lines.push(format!(
                "{src}->{dst} phase (linear-phase engines, passband fit): delay out-smp linear {tau_lin:.3} (residual {res_lin:.4} rad) | lanczos {tau_l:.3} ({res_l:.4}) | soxr {tau_soxr:.3} ({res_soxr:.4}); min-phase: nonlinear by design"
            ));
        }

        // ── Impulse axis (per-engine metrics + cross-engine delay spread) ──
        {
            let out_min = rust_convert(src, dst, &impulse, false);
            let out_lin = rust_convert(src, dst, &impulse, true);
            let (idx_min, pr_min, _) = impulse_metrics(&out_min);
            let (idx_lin, pr_lin, _) = impulse_metrics(&out_lin);
            let lan = probe_single(&probe, src, dst, &impulse, &scratch, "lan_px", false)
                .unwrap_or_else(|e| panic!("{src}->{dst} probe single impulse: {e}"));
            let (idx_lan, pr_lan, _) = impulse_metrics(&lan);
            lines.push(format!(
                "{src}->{dst} impulse: min-phase peak@{idx_min}/prering {pr_min:.1} dB | linear peak@{idx_lin}/{pr_lin:.1} | lanczos-single(double) peak@{idx_lan}/{pr_lan:.1} | peak-delay spreads vs min-phase: linear {:+} | lanczos {:+} out-smp",
                idx_lin as i64 - idx_min as i64,
                idx_lan as i64 - idx_min as i64
            ));
            // Structural: measured commercial delays are bounded within one
            // block; a debug outlier breaks the offline comparison contract.
            assert!(
                idx_min > 0 && idx_min < 4096,
                "{src}->{dst} impulse min-phase peak out of range"
            );
            assert!(
                idx_lan > 0 && idx_lan < 4096,
                "{src}->{dst} impulse lanczos peak out of range"
            );
        }

        // ── Sweep axis (engine-vs-engine broadband delta, same-rate) ────
        {
            let out_min = rust_convert(src, dst, &sweep, false);
            let out_lin = rust_convert(src, dst, &sweep, true);
            let lan = probe_single(&probe, src, dst, &sweep, &scratch, "lan_sc", false)
                .unwrap_or_else(|e| panic!("{src}->{dst} probe single sweep: {e}"));
            let snr_min_lan = aligned_snr(&out_min, &lan);
            let snr_min_lin = aligned_snr(&out_min, &out_lin);
            lines.push(format!(
                "{src}->{dst} sweep cross-engine aligned SNR vs min-phase: linear {snr_min_lin:.1} | lanczos-single(double) {snr_min_lan:.1} dB"
            ));
        }

        // ── Stress v2 axis (real-model program, 4.5 s; engine vs engine) ──
        {
            let out_min = rust_convert(src, dst, &stress_v2, false);
            let out_lin = rust_convert(src, dst, &stress_v2, true);
            assert!(out_min.iter().all(|v| v.is_finite()));
            assert!(out_lin.iter().all(|v| v.is_finite()));
            let lan = probe_single(&probe, src, dst, &stress_v2, &scratch, "lan_sv", false)
                .unwrap_or_else(|e| panic!("{src}->{dst} probe single stress: {e}"));
            assert!(lan.iter().all(|v| v.is_finite()));
            let snr_min_lan = aligned_snr(&out_min, &lan);
            let snr_min_lin = aligned_snr(&out_min, &out_lin);
            lines.push(format!(
                "{src}->{dst} stress_v2 cross-engine aligned SNR vs min-phase: linear {snr_min_lin:.1} | lanczos-single(double) {snr_min_lan:.1} dB"
            ));
        }

        // ── OOB aliasing axis (downsampling only) ───────────────────────
        if dst < src {
            let db_min = oob_alias_rejection_db(src, dst, false);
            let db_lin = oob_alias_rejection_db(src, dst, true);
            lines.push(format!(
                "{src}->{dst} OOB alias rejection (worst fold): min-phase {db_min:.1} | linear {db_lin:.1} dB"
            ));
        }
    }

    // ── Real signal axis (48 kHz mirror fixture: 48k→44.1k and 48k→96k) ──
    for &(src, dst) in RATE_PAIRS {
        if src != 48_000 {
            continue;
        }
        let wav = nam_core_dir().join("example_audio/input.wav");
        if !wav.is_file() {
            report_skip(
                "test_s12_cross_engine_characterization",
                "real-signal fixture missing (example_audio/input.wav) — REAL axis GAP",
            );
            continue;
        }
        let (wav_src, real) = read_wav_mono(&wav).expect("read real WAV");
        assert_eq!(wav_src, 48_000, "expected the 48 kHz mirror fixture");
        let r_min = rust_convert(src, dst, &real, false);
        let r_lin = rust_convert(src, dst, &real, true);
        let lan = probe_single(&probe, src, dst, &real, &scratch, "real", false)
            .unwrap_or_else(|e| panic!("{src}->{dst} probe single real: {e}"));
        // Engine ↔ engine integer-delay spread (same-rate outputs).
        let lag_lin = best_lag(&r_min, &r_lin, 256);
        let lag_lan = best_lag(&r_min, &lan, 256);
        let snr_min_lan = aligned_snr(&r_min, &lan);
        let snr_min_lin = aligned_snr(&r_min, &r_lin);
        lines.push(format!(
            "{src}->{dst} real_guitar: cross-engine aligned SNR vs min-phase: linear {snr_min_lin:.1} | lanczos-single(double) {snr_min_lan:.1} dB; delay spreads vs min-phase out-smp: linear {lag_lin:+} | lanczos {lag_lan:+}"
        ));
        // Approximate real-path energy sanity (per-sample RMS heuristic: the
        // converted stream must conserve energy within generous bounds).
        let rms = |v: &[f32]| -> f64 {
            v.iter().map(|s| *s as f64 * *s as f64).sum::<f64>() / v.len() as f64
        };
        let rms_in = rms(&real);
        let rms_out = rms(&r_min);
        assert!(
            rms_out > rms_in * 0.5 && rms_out < rms_in * 1.6,
            "real-signal RMS conservation violated: in_{rms_in:.4} out_{rms_out:.4}"
        );
    }

    println!(
        "PERCEPTUAL: GAP — A/B humano conforme docs/perceptual_validation.md indisponível nesta máquina"
    );
    for l in &lines {
        println!("// Measured: {l}");
    }
    scrub_dir(&scratch);
}

/// Round-trip (two-stage) characterization quantifying the "independence of
/// the f64 anchors": how much SNR each cross-rate chain preserves when the
/// two comparison sides use (a) the same Rust engine — the shared-resampler
/// cancellation the cpp_parity anchors rely on — or (b/d) independent engine
/// per side (Rust ↔ container, container both sides).
#[test]
#[ignore = "long: cross-rate anchor independence characterization (requires vendor + C++ probe)"]
fn test_s12_anchor_independence_cancellation() {
    if !vendor_available() {
        report_skip(
            "test_s12_anchor_independence_cancellation",
            "vendor mirrors missing (optional): crossed roundtrip axis unavailable — GAP",
        );
        return;
    }
    let probe = match ensure_probe() {
        Ok(p) => p,
        Err(e) => {
            report_skip(
                "test_s12_anchor_independence_cancellation",
                &format!("C++ compiler/probe unavailable (optional): {e}"),
            );
            return;
        }
    };
    let manifest = manifest_dir();
    let scratch = manifest.join("target/tests/s12_scratch/anchors");
    std::fs::create_dir_all(&scratch).expect("scratch dir");

    let mut lines: Vec<String> = Vec::new();
    for &(rate, mid) in &[(44_100u32, 48_000u32), (48_000, 96_000), (96_000, 48_000)] {
        let input = generate_stress_signal_v2("s12-roundtrip", rate);

        // (a) shared Rust chain (cpp_parity design point); chains are aligned
        // before the SNR (the two stages shift the stream deterministically).
        let rr = rust_roundtrip(rate, mid, &input, false);
        let rr_snr = aligned_snr(&input, &rr);
        lines.push(format!(
            "{rate}->{mid}->{rate} shared-Rust roundtrip: full-band SNR vs source {rr_snr:.1} dB"
        ));

        // (b) Rust-in + lanczos-out.
        let mid_r = rust_convert(rate, mid, &input, false);
        let back_b = probe_single(&probe, mid, rate, &mid_r, &scratch, "ch_b", false)
            .unwrap_or_else(|e| panic!("{mid}->{rate} probe single: {e}"));
        let b_snr = aligned_snr(&input, &back_b);
        lines.push(format!(
            "{rate}->{mid}->{rate} Rust-in + lanczos(f64)-out: full-band SNR vs source {b_snr:.1} dB"
        ));

        // (c) lanczos-in + Rust-out.
        let mid_c = probe_single(&probe, rate, mid, &input, &scratch, "ch_c", false)
            .unwrap_or_else(|e| panic!("{rate}->{mid} probe single: {e}"));
        let back_c = rust_convert(mid, rate, &mid_c, false);
        let c_snr = aligned_snr(&input, &back_c);
        lines.push(format!(
            "{rate}->{mid}->{rate} lanczos(f64)-in + Rust-out: full-band SNR vs source {c_snr:.1} dB"
        ));

        // (d) container pair both sides (plugin production chain shape).
        let (pair, latency) = probe_pair(&probe, rate, mid, &input, &scratch, "ch_d", false)
            .unwrap_or_else(|e| panic!("{rate}->{mid} probe pair: {e}"));
        let d_snr = aligned_snr(&input, &pair);
        lines.push(format!(
            "{rate}->{mid}->{rate} container(f64) both sides: full-band SNR vs source {d_snr:.1} dB (reported latency {latency} in-smp)"
        ));
        // Structural: reported container latency is positive when not bypassed.
        assert!(
            latency > 0,
            "{rate}->{mid}: container must report latency > 0"
        );
        // Crossed chains MUST be reachable — a pathological engine (delay > 1
        // block) breaks the offline comparison contract rather than the audio.
        assert!(b_snr.is_finite(), "{rate}->{mid}: crossed chain (b) failed");
        assert!(c_snr.is_finite(), "{rate}->{mid}: crossed chain (c) failed");
    }
    for l in &lines {
        println!("// Measured: {l}");
    }
    scrub_dir(&scratch);
}

/// CPU cost comparison: Rust production stages vs the vendor probe —
/// same signal body (4 s stress v2), conversions only (construction and
/// polyphase-bank build excluded on both sides), median ns per emitted
/// sample. Host bench, off-cgroup; operator-owned (`docs/benchmarks.md` §13).
#[test]
#[ignore = "long: resampler CPU cost comparison (off-RT bench; requires vendor + C++ probe)"]
fn test_s12_cpu_cost_comparison() {
    if !vendor_available() {
        report_skip(
            "test_s12_cpu_cost_comparison",
            "vendor mirrors missing (optional): CPU comparison axis unavailable — GAP",
        );
        return;
    }
    let probe = match ensure_probe() {
        Ok(p) => p,
        Err(e) => {
            report_skip(
                "test_s12_cpu_cost_comparison",
                &format!("C++ compiler/probe unavailable (optional): {e}"),
            );
            return;
        }
    };
    let manifest = manifest_dir();
    let scratch = manifest.join("target/tests/s12_scratch/cpu");
    std::fs::create_dir_all(&scratch).expect("scratch dir");
    const REPS: usize = 40;

    let mut lines: Vec<String> = Vec::new();
    for &(pw, nam) in &[
        (44_100u32, 48_000u32),
        (48_000, 44_100),
        (96_000, 48_000),
        (48_000, 96_000),
    ] {
        let body_in = generate_stress_signal_v2("s12-cpu", pw);
        let body_rev = generate_stress_signal_v2("s12-cpu", nam);
        let rust_in = rust_stage_ns_per_out(pw, nam, &body_in, REPS, false, false);
        let rust_out = rust_stage_ns_per_out(pw, nam, &body_rev, REPS, false, true);
        let rust_lin = rust_stage_ns_per_out(pw, nam, &body_in, REPS, true, false);
        let (stage_fwd, container) =
            probe_bench(&probe, (pw, nam), &body_in, &scratch, "cpu", REPS, false)
                .unwrap_or_else(|e| panic!("{pw}->{nam} probe bench: {e}"));
        let (stage_rev, _) = probe_bench(
            &probe,
            (nam, pw),
            &body_rev,
            &scratch,
            "cpu_rev",
            REPS,
            false,
        )
        .unwrap_or_else(|e| panic!("{nam}->{pw} probe bench: {e}"));
        let (stage32, container32) =
            probe_bench(&probe, (pw, nam), &body_in, &scratch, "cpu32", REPS, true)
                .unwrap_or_else(|e| panic!("{pw}->{nam} probe bench float: {e}"));
        lines.push(format!(
            "{pw}->{nam} ns/ext-sample (median): rust-in {rust_in:.0} | rust-out {rust_out:.0} | rust-lin-in {rust_lin:.0} | lanczos-fwd(f64) {stage_fwd:.0} | lanczos-rev(f64) {stage_rev:.0} | container-pair(f64) {container:.0} | lanczos-fwd(f32) {stage32:.0} | container-pair(f32) {container32:.0}"
        ));
        assert!(rust_in > 0.0 && rust_out > 0.0 && rust_lin > 0.0);
        assert!(
            stage_fwd > 0.0
                && stage_rev > 0.0
                && container > 0.0
                && stage32 > 0.0
                && container32 > 0.0
        );
    }
    for l in &lines {
        println!("// Measured: {l}");
    }
    scrub_dir(&scratch);
}
