// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Independent f64 streaming-resampling oracle (T5.4 / F9).
//!
//! This test never calls [`NamResampler`] or [`StreamingResampleBuffer`] inside
//! the oracle: it re-derives the conversion from the *mathematical
//! specification* documented below and compares it against the production
//! streaming adapter driven with variable block sizes.
//!
//! ## Filter specification (derived, not copied)
//!
//! The production bank is a polyphase decomposition of a minimum-phase FIR
//! prototype:
//!
//! - **Prototype**: ideal low-pass sinc windowed by a Kaiser window with
//!   `β = 12`, length `NUM_PHASES × TAPS = 256 × 64 = 16384`. The prototype
//!   lives at the conceptual rate `from_rate × NUM_PHASES`, so its normalized
//!   cutoff is `0.95 × min(from, to) / (from × NUM_PHASES)` (95 % of the
//!   lower Nyquist). The prototype is normalized to unit DC gain.
//! - **Phase**: minimum phase, obtained with the real-cepstrum method
//!   (Oppenheim & Schafer): zero-pad to a power of two `≥ 4 ×` the prototype
//!   length, take `ln(|H| + 1e-10)`, fold the causal part of the cepstrum
//!   (`c[1..N/2)` doubled, `c[N/2+1..)` zeroed), exponentiate and inverse
//!   transform. This is a magnitude-preserving transform, so the stopband of
//!   the linear-phase prototype carries over.
//! - **Delay**: the minimum-phase kernel has no closed-form group delay; the
//!   implementation publishes the energy centroid of the impulse response
//!   (`Σ n·h² / Σ h²`) divided by `NUM_PHASES`, i.e. in output-rate samples.
//!   The declared latency of the two-stage chain is
//!   `round(gd_in + gd_out × host/model)`.
//! - **Priming**: the streaming adapter declares that latency as leading zeros
//!   and discards the first `latency` real outputs (the startup transient), so
//!   the delivered waveform is delayed by exactly `latency` samples rather
//!   than `2 × latency`.
//! - **Phase tracking**: 24.40 fixed point, `phase_step =
//!   round(from/to × NUM_PHASES × 2^40)`, starting at `NUM_PHASES` (one whole
//!   input sample). An output consumes input samples while `phase ≥ 1`, then
//!   evaluates two adjacent polyphase sub-filters and linearly interpolates
//!   them with the 31-bit truncated sub-phase fraction. The sub-filter tap
//!   layout is `c[phase][k] = proto[phase + k × NUM_PHASES] × NUM_PHASES`,
//!   applied with `x[n-k]` (newest sample against `k = 0`).
//!
//! The oracle is evaluated in f64 with f64 coefficients throughout; the
//! production path quantizes the bank to f32 and accumulates in f32/SIMD. The
//! residual is therefore the f32-vs-f64 floor, measured below.

use neural_amp_modeler_rs::dsp::resampling::StreamingResampleBuffer;

const NUM_PHASES: usize = 256;
const TAPS: usize = 64;
const PROTO_LEN: usize = NUM_PHASES * TAPS;
const PHASE_UNIT: u64 = 1 << 40;
const ONE: u64 = NUM_PHASES as u64 * PHASE_UNIT;

/// Gate for the production-vs-oracle energy signal-to-error ratio.
// `// Measured (2026-11-17, debug `test` profile, 48 000 host samples + 4 096
// tail, L/R-distinct sweep+multitone, f32 production bank vs f64 oracle):
//   ESR 3.44e-14/3.50e-14 (48->44.1), 4.61e-14/4.58e-14 (44.1->48),
//       5.86e-15/5.80e-15 (48->96), 1.11e-14/1.11e-14 (96->48)
//   SNR 134.6/134.6, 133.4/133.4, 142.3/142.4, 139.5/139.6 dB
// Gates keep ~20x margin over the worst measured ESR and ~30 dB over the
// worst measured SNR while staying far below the f32 bank quantization floor.`
const ESR_GATE: f64 = 1.0e-12;
/// Gate for the production-vs-oracle SNR (dB).
const SNR_GATE_DB: f64 = 100.0;

/// `(host_rate, model_rate)` cells required by T5.4.
const RATE_PAIRS: [(u32, u32); 4] = [
    (48_000, 44_100),
    (44_100, 48_000),
    (48_000, 96_000),
    (96_000, 48_000),
];

/// Kaiser `I₀(x)` (Taylor series, 20 terms; |error| < 1e-12 for β ≤ 25).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half = x / 2.0;
    for k in 1..=20 {
        term *= (half / k as f64) * (half / k as f64);
        sum += term;
        if term < 1e-15 * sum {
            break;
        }
    }
    sum
}

/// In-place iterative radix-2 FFT (forward). `re`/`im` must have equal
/// power-of-two lengths.
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    assert!(n.is_power_of_two());
    assert_eq!(n, im.len());

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2usize;
    while len <= n {
        let angle = 2.0 * std::f64::consts::PI / len as f64 * if inverse { 1.0 } else { -1.0 };
        let (wr, wi) = (angle.cos(), angle.sin());
        let mut i = 0usize;
        while i < n {
            let (mut cr, mut ci) = (1.0_f64, 0.0_f64);
            for k in 0..len / 2 {
                let ur = re[i + k];
                let ui = im[i + k];
                let vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                let vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let next = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = next;
            }
            i += len;
        }
        len *= 2;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for v in re.iter_mut() {
            *v *= scale;
        }
        for v in im.iter_mut() {
            *v *= scale;
        }
    }
}

/// Minimum-phase polyphase bank derived from the specification above.
#[derive(Clone)]
struct OracleBank {
    /// `coeffs[phase * TAPS + k]` (f64, already scaled by `NUM_PHASES`).
    coeffs: Vec<f64>,
    /// Energy-centroid group delay in output-rate samples.
    group_delay: f64,
}

fn design_bank(from_rate: u32, to_rate: u32) -> OracleBank {
    // Linear-phase keynote: sinc × Kaiser(β = 12), unit DC gain.
    let half = (PROTO_LEN - 1) as f64 / 2.0;
    let i0_beta = bessel_i0(12.0);
    let cutoff = 0.95 * from_rate.min(to_rate) as f64 / (from_rate as f64 * NUM_PHASES as f64);
    let mut proto = Vec::with_capacity(PROTO_LEN);
    for i in 0..PROTO_LEN {
        let n = i as f64 - half;
        let sinc = if n.abs() < 1e-12 {
            cutoff
        } else {
            let x = std::f64::consts::PI * n * cutoff;
            x.sin() / (std::f64::consts::PI * n)
        };
        let ratio = n / half;
        let arg = 12.0 * (1.0 - ratio * ratio).max(0.0).sqrt();
        proto.push(sinc * (bessel_i0(arg) / i0_beta));
    }
    let dc: f64 = proto.iter().sum();
    if dc.abs() > 1e-15 {
        for k in &mut proto {
            *k /= dc;
        }
    }

    // Real-cepstrum minimum-phase transform.
    let n_fft = (4 * PROTO_LEN).next_power_of_two();
    let mut re = vec![0.0_f64; n_fft];
    let mut im = vec![0.0_f64; n_fft];
    re[..PROTO_LEN].copy_from_slice(&proto);
    fft(&mut re, &mut im, false);
    for i in 0..n_fft {
        re[i] = (re[i].hypot(im[i]) + 1e-10).ln();
        im[i] = 0.0;
    }
    fft(&mut re, &mut im, true);
    let half_fft = n_fft / 2;
    for item in re.iter_mut().take(half_fft).skip(1) {
        *item *= 2.0;
    }
    for item in re.iter_mut().skip(half_fft + 1) {
        *item = 0.0;
    }
    im.fill(0.0);
    fft(&mut re, &mut im, false);
    for i in 0..n_fft {
        let exp_a = re[i].exp();
        let (sin_b, cos_b) = im[i].sin_cos();
        re[i] = exp_a * cos_b;
        im[i] = exp_a * sin_b;
    }
    fft(&mut re, &mut im, true);
    let min_phase = &re[..PROTO_LEN];

    // Energy centroid → published group delay (output-rate samples).
    let mut num = 0.0;
    let mut den = 0.0;
    for (n, &v) in min_phase.iter().enumerate() {
        let energy = v * v;
        num += n as f64 * energy;
        den += energy;
    }
    let group_delay = if den > 1e-30 { num / den } else { 0.0 } / NUM_PHASES as f64;

    // Polyphase partition, reversed taps exactly like the specification.
    let gain = NUM_PHASES as f64;
    let mut coeffs = vec![0.0_f64; NUM_PHASES * TAPS];
    for (n, &c) in min_phase.iter().enumerate() {
        let phase = n % NUM_PHASES;
        let tap = n / NUM_PHASES;
        if tap < TAPS {
            coeffs[phase * TAPS + (TAPS - 1 - tap)] = c * gain;
        }
    }

    OracleBank {
        coeffs,
        group_delay,
    }
}

/// One direction of the streaming converter, implemented from the phase
/// tracking specification (not from the production source).
struct OracleCore {
    bank: OracleBank,
    /// Ring of the last `TAPS` inputs; `pos` is the next write slot.
    ring: [f64; TAPS],
    pos: usize,
    phase_accum: u64,
    phase_step: u64,
}

impl OracleCore {
    fn new(bank: OracleBank, from_rate: u32, to_rate: u32) -> Self {
        let phase_step =
            ((from_rate as f64 / to_rate as f64) * NUM_PHASES as f64 * PHASE_UNIT as f64).round()
                as u64;
        Self {
            bank,
            ring: [0.0; TAPS],
            pos: 0,
            phase_accum: NUM_PHASES as u64 * PHASE_UNIT,
            phase_step,
        }
    }

    fn push(&mut self, sample: f64) {
        self.ring[self.pos] = sample;
        self.pos = (self.pos + 1) % TAPS;
    }

    /// `x[n - k]`, zero before any sample was pushed.
    fn x(&self, k: usize) -> f64 {
        self.ring[(self.pos + TAPS - 1 - k) % TAPS]
    }

    fn convolve(&self, phase: usize) -> f64 {
        // `x(k)` is the input n-k (newest at k = 0). The stored bank is
        // reversed inside each phase so the SIMD window (oldest first) lines
        // up; the mathematical coefficient for `x[n-k]` is `proto[phase +
        // k*NUM_PHASES]`, stored at index `TAPS - 1 - k`.
        let mut acc = 0.0;
        for k in 0..TAPS {
            acc += self.x(k) * self.bank.coeffs[phase * TAPS + (TAPS - 1 - k)];
        }
        acc
    }

    /// Consumes as many inputs as needed to produce outputs, stopping exactly
    /// when the next output would need an input sample that is not available.
    fn process(&mut self, input: &[f64], out: &mut Vec<f64>) {
        let mut idx = 0usize;
        loop {
            while self.phase_accum >= ONE {
                if idx >= input.len() {
                    return;
                }
                self.push(input[idx]);
                idx += 1;
                self.phase_accum -= ONE;
            }
            let phase = (self.phase_accum >> 40) as usize;
            debug_assert!(phase < NUM_PHASES);
            let phase_next = if phase + 1 >= NUM_PHASES {
                0
            } else {
                phase + 1
            };
            let frac = ((self.phase_accum & (PHASE_UNIT - 1)) >> 9) as f64 / (1u64 << 31) as f64;
            let y0 = self.convolve(phase);
            let y1 = self.convolve(phase_next);
            out.push(y0 + frac * (y1 - y0));
            self.phase_accum += self.phase_step;
        }
    }
}

/// Two-stage streaming oracle (host → model → host), one independent chain per
/// channel (L/R carry distinct signals).
struct OracleConverter {
    inner_l: OracleCore,
    outer_l: OracleCore,
    inner_r: OracleCore,
    outer_r: OracleCore,
    latency: u32,
}

impl OracleConverter {
    fn new(host_rate: u32, model_rate: u32) -> Self {
        let bank_in = design_bank(host_rate, model_rate);
        let bank_out = design_bank(model_rate, host_rate);
        let delay_in = bank_in.group_delay;
        let delay_out = bank_out.group_delay * (host_rate as f64 / model_rate as f64);
        let latency = (delay_in + delay_out).round() as u32;
        Self {
            inner_l: OracleCore::new(bank_in.clone(), host_rate, model_rate),
            outer_l: OracleCore::new(bank_out.clone(), model_rate, host_rate),
            inner_r: OracleCore::new(bank_in, host_rate, model_rate),
            outer_r: OracleCore::new(bank_out, model_rate, host_rate),
            latency,
        }
    }

    fn process_l(&mut self, host_in: &[f64], out: &mut Vec<f64>) {
        let mut mid = Vec::new();
        self.inner_l.process(host_in, &mut mid);
        self.outer_l.process(&mid, out);
    }

    fn process_r(&mut self, host_in: &[f64], out: &mut Vec<f64>) {
        let mut mid = Vec::new();
        self.inner_r.process(host_in, &mut mid);
        self.outer_r.process(&mid, out);
    }
}

fn esr(reference: &[f64], test: &[f64]) -> f64 {
    let len = reference.len().min(test.len());
    assert!(len > 0);
    let mut signal = 0.0;
    let mut noise = 0.0;
    for i in 0..len {
        signal += reference[i] * reference[i];
        let d = reference[i] - test[i];
        noise += d * d;
    }
    if signal <= 0.0 {
        return if noise <= 0.0 { 0.0 } else { f64::INFINITY };
    }
    noise / signal
}

fn snr_db(reference: &[f64], test: &[f64]) -> f64 {
    let e = esr(reference, test);
    if e <= 0.0 { 200.0 } else { -10.0 * e.log10() }
}

/// Best integer lag of `test` against `reference` (positive lag = `test` is
/// delayed), plus a parabolic fractional estimate before any alignment.
fn estimate_delay(reference: &[f64], test: &[f64], max_lag: isize) -> (isize, f64) {
    let window = reference.len().min(test.len()).min(16_384);
    let corr = |lag: isize| -> f64 {
        let mut acc = 0.0;
        for (i, &r) in reference.iter().enumerate().take(window) {
            let j = i as isize + lag;
            if j >= 0 && (j as usize) < test.len() {
                acc += r * test[j as usize];
            }
        }
        acc
    };
    let mut best_lag = 0isize;
    let mut best = f64::NEG_INFINITY;
    for lag in -max_lag..=max_lag {
        let c = corr(lag);
        if c > best {
            best = c;
            best_lag = lag;
        }
    }
    let left = corr(best_lag - 1);
    let center = corr(best_lag);
    let right = corr(best_lag + 1);
    let denom = left - 2.0 * center + right;
    let frac = if denom.abs() > 1e-30 {
        0.5 * (left - right) / denom
    } else {
        0.0
    };
    (best_lag, frac)
}

/// Deterministic L/R-distinct stimulus: log-free sweep + two tones.
fn stimulus(len: usize, rate: u32, phase_offset: f64) -> Vec<f64> {
    let duration = len as f64 / rate as f64;
    (0..len)
        .map(|i| {
            let t = i as f64 / rate as f64;
            let sweep_phase =
                2.0 * std::f64::consts::PI * (220.0 * t + 0.5 * (880.0 - 220.0) * t * t / duration);
            0.25 * sweep_phase.sin()
                + 0.1 * (2.0 * std::f64::consts::PI * 1000.0 * t + phase_offset).sin()
                + 0.05 * (2.0 * std::f64::consts::PI * 3000.0 * t).cos()
        })
        .collect()
}

/// Variable block schedule (all ≤ the adapter's `max_block`).
const BLOCKS: [usize; 6] = [64, 127, 256, 33, 512, 128];

#[test]
fn streaming_resampler_matches_independent_f64_oracle() {
    const HOST_SAMPLES: usize = 48_000;
    // Extra zeros flush every internal FIFO/phase so both streams include the
    // same amount of tail energy instead of comparing truncated outputs.
    const TAIL: usize = 4_096;

    for (host_rate, model_rate) in RATE_PAIRS {
        let mut left = stimulus(HOST_SAMPLES, host_rate, 0.0);
        let mut right = stimulus(HOST_SAMPLES, host_rate, 0.7);
        left.extend(std::iter::repeat_n(0.0, TAIL));
        right.extend(std::iter::repeat_n(0.0, TAIL));

        // ── Production streaming adapter ────────────────────────────────────
        let max_block = *BLOCKS.iter().max().unwrap();
        let mut stream = StreamingResampleBuffer::new(host_rate, model_rate, max_block)
            .expect("streaming adapter construction");
        let mut produced_l: Vec<f32> = Vec::with_capacity(left.len() + 2 * max_block);
        let mut produced_r: Vec<f32> = Vec::with_capacity(right.len() + 2 * max_block);
        let mut out_l = vec![0.0f32; max_block];
        let mut out_r = vec![0.0f32; max_block];
        let mut offset = 0usize;
        let mut block_iter = BLOCKS.iter().copied().cycle();
        while offset < left.len() {
            let n = block_iter.next().unwrap().min(left.len() - offset);
            let in_l = &left[offset..offset + n];
            let in_r = &right[offset..offset + n];
            let result = stream.process(
                &in_l.iter().map(|&s| s as f32).collect::<Vec<_>>(),
                &in_r.iter().map(|&s| s as f32).collect::<Vec<_>>(),
                &mut out_l[..n],
                &mut out_r[..n],
                |il, ir, ol, or| {
                    ol[..il.len()].copy_from_slice(il);
                    or[..ir.len()].copy_from_slice(ir);
                    il.len()
                },
            );
            assert_eq!(
                result.consumed, n,
                "{host_rate}->{model_rate}: every host sample must be accepted"
            );
            assert_eq!(result.written, n);
            assert!(
                !result.underflow,
                "{host_rate}->{model_rate}: no underflow expected with a synchronous passthrough model"
            );
            produced_l.extend_from_slice(&out_l[..n]);
            produced_r.extend_from_slice(&out_r[..n]);
            offset += n;
        }

        let latency = stream.latency_samples() as usize;

        // ── Independent f64 oracle over the full stimulus ───────────────────
        let mut oracle = OracleConverter::new(host_rate, model_rate);
        assert_eq!(
            latency, oracle.latency as usize,
            "{host_rate}->{model_rate}: declared latency must equal the oracle group delay"
        );
        let mut oracle_l = Vec::with_capacity(left.len() + 2 * max_block);
        let mut oracle_r = Vec::with_capacity(right.len() + 2 * max_block);
        let mut o_offset = 0usize;
        let mut o_blocks = BLOCKS.iter().copied().cycle();
        while o_offset < left.len() {
            let n = o_blocks.next().unwrap().min(left.len() - o_offset);
            oracle.process_l(&left[o_offset..o_offset + n], &mut oracle_l);
            oracle.process_r(&right[o_offset..o_offset + n], &mut oracle_r);
            o_offset += n;
        }

        // ── Priming and counters ────────────────────────────────────────────
        assert_eq!(
            stream.input_total(),
            left.len() as u64,
            "{host_rate}->{model_rate}: accepted-sample ledger"
        );
        assert_eq!(
            stream.output_padded_total(),
            latency as u64,
            "{host_rate}->{model_rate}: the declared priming must be the only padding"
        );
        assert_eq!(
            stream.underflow_total(),
            0,
            "{host_rate}->{model_rate}: no fabricated sample beyond priming"
        );
        assert_eq!(
            stream.output_real_total(),
            (produced_l.len() - latency) as u64,
            "{host_rate}->{model_rate}: produced-sample ledger"
        );
        // Output cardinality is exactly one host sample per accepted host sample.
        assert_eq!(
            produced_l.len(),
            left.len(),
            "{host_rate}->{model_rate}: streaming cardinality"
        );
        for &s in &produced_l[..latency] {
            assert_eq!(
                s, 0.0,
                "{host_rate}->{model_rate}: priming must be contiguous zeros"
            );
        }

        // Delivered real region = discarded transient shift of the full chain.
        let dut_l: Vec<f64> = produced_l[latency..].iter().map(|&s| s as f64).collect();
        let dut_r: Vec<f64> = produced_r[latency..].iter().map(|&s| s as f64).collect();
        let ref_l = &oracle_l[latency.min(oracle_l.len())..];
        let ref_r = &oracle_r[latency.min(oracle_r.len())..];
        let len = dut_l.len().min(ref_l.len());
        assert!(
            len > HOST_SAMPLES / 2,
            "{host_rate}->{model_rate}: oracle produced too few samples ({len})"
        );

        // Delay before alignment (integer + fractional estimate).
        let (lag_l, frac_l) = estimate_delay(&ref_l[..len], &dut_l[..len], 64);
        let (lag_r, frac_r) = estimate_delay(&ref_r[..len], &dut_r[..len], 64);
        println!(
            "// Measured: {host_rate}->{model_rate} delay L=({lag_l},{frac_l:+.3}) R=({lag_r},{frac_r:+.3}) latency={latency}"
        );
        assert_eq!(
            lag_l, 0,
            "{host_rate}->{model_rate}: left channel integer delay"
        );
        assert_eq!(
            lag_r, 0,
            "{host_rate}->{model_rate}: right channel integer delay"
        );
        assert!(
            frac_l.abs() <= 0.5 && frac_r.abs() <= 0.5,
            "{host_rate}->{model_rate}: fractional delay must stay within one sample"
        );

        // Energy/SNR over the delayed real region (tail included).
        let e_l = esr(&ref_l[..len], &dut_l[..len]);
        let e_r = esr(&ref_r[..len], &dut_r[..len]);
        let s_l = snr_db(&ref_l[..len], &dut_l[..len]);
        let s_r = snr_db(&ref_r[..len], &dut_r[..len]);
        println!(
            "// Measured: {host_rate}->{model_rate} ESR L={e_l:.3e} R={e_r:.3e} SNR L={s_l:.2} dB R={s_r:.2} dB n={len}"
        );
        assert!(
            e_l <= ESR_GATE && e_r <= ESR_GATE,
            "{host_rate}->{model_rate}: ESR L={e_l:.3e} R={e_r:.3e} exceeds gate {ESR_GATE:.1e}"
        );
        assert!(
            s_l >= SNR_GATE_DB && s_r >= SNR_GATE_DB,
            "{host_rate}->{model_rate}: SNR L={s_l:.2} dB R={s_r:.2} dB below gate {SNR_GATE_DB:.1} dB"
        );
    }
}

#[test]
fn intentional_underflow_is_flagged_and_counted() {
    let mut stream = StreamingResampleBuffer::new(48_000, 44_100, 512).expect("adapter");
    let mut out_l = [0.0f32; 64];
    let mut out_r = [0.0f32; 64];

    // Pull before any input: the adapter must fabricate zeros (cardinality),
    // flag the underflow and never claim them as real samples.
    let pull = stream.pull_output(&mut out_l, &mut out_r, 64);
    let latency = stream.latency_samples() as usize;
    assert_eq!(pull.written, 64);
    assert!(pull.underflow, "pulling with no input must flag underflow");
    assert_eq!(pull.real, 0);
    assert_eq!(
        pull.padded, 64,
        "the whole pull is zeros: declared priming plus the underflow deficit"
    );
    assert!(out_l.iter().all(|&s| s == 0.0));
    assert!(out_r.iter().all(|&s| s == 0.0));
    let priming = latency.min(64) as u64;
    assert_eq!(
        stream.underflow_total(),
        64 - priming,
        "only the deficit beyond the declared priming budget is underflow"
    );
    assert_eq!(stream.output_padded_total(), 64);
    assert_eq!(stream.output_real_total(), 0);
}
