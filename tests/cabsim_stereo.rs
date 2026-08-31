// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(dead_code)]

//! ER-2 stereo cab-sim validation harness.
//!
//! Engine-level gates for the stereo-decoupled `CabSimPair`:
//!
//! 1. **Full-pipeline stereo fidelity vs dual mono** (`feature = "stereo"`):
//!    a unified stereo pipeline (independent L/R models + `CabSimPair`) must be
//!    bit-exact against two 100% isolated mono pipelines (single adapter each)
//!    — `MSE == 0.0`, `SNR > 120 dB`.
//! 2. **IR multirate consistency**: resampling an IR from its native rate to
//!    the applied host rate yields a valid impulse response — the resampled IR
//!    convolved through `ConvEngine` matches direct convolution, and the rate
//!    metadata scales correctly.
//! 3. **Zero allocations** (`feature = "heap-audit"`): the stereo pipeline with
//!    an active `CabSimPair` performs zero heap allocations on the processing
//!    thread.

mod common;

use common::alloc_audit::CountingAllocator;
use common::conv_helpers::direct_convolve;
use common::io_helpers::model_path;
use common::metrics::compute_mse;

// The lib only registers its own allocator under `#[cfg(test)]`; integration
// binaries register the shared test-local one so heap audits count in every
// build (including `feature = "heap-audit"`).
#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

use neural_amp_modeler_rs::dsp::adaptive::{AdaptiveCompute, AdaptiveComputeMode};
use neural_amp_modeler_rs::dsp::cabsim::adapter::{CabSimAdapter, CabSimPair};
use neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine;
use neural_amp_modeler_rs::dsp::cabsim::loader::CabSimIr;
use neural_amp_modeler_rs::dsp::gate::{DynamicHysteresis, GateParams};
use neural_amp_modeler_rs::dsp::oversample::{OversampleEngine, OversampleFactor};
use neural_amp_modeler_rs::dsp::pipeline::{
    BridgeBuffer, BridgeRef, DspBridge, DspBridgeWriter, DspBuffers, DspPipelineContext,
    MAX_RESAMP_BUF, capture_dsp_pipeline,
};
use neural_amp_modeler_rs::dsp::resampler::NamResampler;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::{NamModel, StaticModel};

use std::fs;

const BLOCK: usize = 64;
const PARTITION: usize = 512;
const IR_LEN: usize = 4096;
const SAMPLE_RATE: u32 = 48_000;

// ── Full-pipeline driver ────────────────────────────────────────────────────

/// Owning stereo/mono pipeline driver for the engine's public
/// `capture_dsp_pipeline`, mirroring how an audio host pipeline wires it.
struct CabsimPipeline {
    resampler: NamResampler,
    os_l: OversampleEngine,
    os_r: OversampleEngine,
    model_l: Option<Box<StaticModel>>,
    model_r: Option<Box<StaticModel>>,
    /// Stereo-decoupled pair (L/R adapters) — used when `Some`.
    pair: Option<Box<CabSimPair>>,
    input_gain_mult: f32,
    output_gain_mult: f32,
    gate_params: GateParams,
    silence_hysteresis: DynamicHysteresis,
    mono_hysteresis: DynamicHysteresis,
    threshold_open_sq: f32,
    threshold_close_sq: f32,
    process_mono: bool,
    rt_status: neural_amp_modeler_rs::common::spsc::RtStatusFlags,
    adaptive: AdaptiveCompute,
    bridge: Box<DspBridge>,
    resamp_mid_l: Box<[f32; MAX_RESAMP_BUF]>,
    resamp_mid_r: Box<[f32; MAX_RESAMP_BUF]>,
    resamp_out_l: Box<[f32; MAX_RESAMP_BUF]>,
    resamp_out_r: Box<[f32; MAX_RESAMP_BUF]>,
    model_out_l: Box<[f32; MAX_RESAMP_BUF]>,
    model_out_r: Box<[f32; MAX_RESAMP_BUF]>,
    os_in_l: Box<[f32; MAX_RESAMP_BUF * 4]>,
    os_in_r: Box<[f32; MAX_RESAMP_BUF * 4]>,
    os_model_l: Box<[f32; MAX_RESAMP_BUF * 4]>,
    os_model_r: Box<[f32; MAX_RESAMP_BUF * 4]>,
    xfd_l: Box<[f32; MAX_RESAMP_BUF]>,
    xfd_r: Box<[f32; MAX_RESAMP_BUF]>,
    last_n_pw: usize,
}

impl CabsimPipeline {
    fn new(host_rate: u32, nam_rate: u32) -> Self {
        let gate_params = GateParams::default();
        let lut = neural_amp_modeler_rs::math::dsp::gain_lut::get_gain_lut();
        let open_lin = lut.db_to_linear(gate_params.threshold_open_db);
        let close_lin = lut.db_to_linear(gate_params.threshold_close_db);
        Self {
            resampler: NamResampler::new(host_rate, nam_rate, BLOCK).expect("resampler"),
            os_l: OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).expect("os"),
            os_r: OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).expect("os"),
            model_l: None,
            model_r: None,
            pair: None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params,
            silence_hysteresis: DynamicHysteresis::new(),
            mono_hysteresis: DynamicHysteresis::new(),
            threshold_open_sq: open_lin * open_lin,
            threshold_close_sq: close_lin * close_lin,
            process_mono: false,
            rt_status: neural_amp_modeler_rs::common::spsc::RtStatusFlags::new(),
            adaptive: AdaptiveCompute::new(AdaptiveComputeMode::Off),
            bridge: Box::new(DspBridge {
                buffers: [BridgeBuffer::new(), BridgeBuffer::new()],
                active_read_idx: Default::default(),
                generation: Default::default(),
                consumed_gen: Default::default(),
                dropped_frames: Default::default(),
            }),
            resamp_mid_l: Box::new([0.0; MAX_RESAMP_BUF]),
            resamp_mid_r: Box::new([0.0; MAX_RESAMP_BUF]),
            resamp_out_l: Box::new([0.0; MAX_RESAMP_BUF]),
            resamp_out_r: Box::new([0.0; MAX_RESAMP_BUF]),
            model_out_l: Box::new([0.0; MAX_RESAMP_BUF]),
            model_out_r: Box::new([0.0; MAX_RESAMP_BUF]),
            os_in_l: Box::new([0.0; MAX_RESAMP_BUF * 4]),
            os_in_r: Box::new([0.0; MAX_RESAMP_BUF * 4]),
            os_model_l: Box::new([0.0; MAX_RESAMP_BUF * 4]),
            os_model_r: Box::new([0.0; MAX_RESAMP_BUF * 4]),
            xfd_l: Box::new([0.0; MAX_RESAMP_BUF]),
            xfd_r: Box::new([0.0; MAX_RESAMP_BUF]),
            last_n_pw: 0,
        }
    }

    fn process(&mut self, in_l: &mut [f32], in_r: &mut [f32], n: usize, rate: u32) -> usize {
        // SAFETY: `self.bridge` is owned and outlives the writer's use.
        let bridge_ref = unsafe { BridgeRef::new(&mut *self.bridge as *mut DspBridge) };
        let writer = DspBridgeWriter::from_ref(bridge_ref).expect("bridge non-null");
        let ctx = DspPipelineContext {
            resampler: &mut self.resampler,
            os_l: &mut self.os_l,
            os_r: &mut self.os_r,
            active_model_l: &mut self.model_l,
            active_model_r: &mut self.model_r,
            input_gain_mult: self.input_gain_mult,
            output_gain_mult: self.output_gain_mult,
            gate_params: &self.gate_params,
            silence_hysteresis: &mut self.silence_hysteresis,
            mono_hysteresis: &mut self.mono_hysteresis,
            threshold_open_sq: self.threshold_open_sq,
            threshold_close_sq: self.threshold_close_sq,
            process_mono: &mut self.process_mono,
            rt_status: &self.rt_status,
            adaptive: &mut self.adaptive,
            bridge_writer: Some(writer),
            conv: None,
            conv_pair: self.pair.as_deref_mut(),
        };
        let bufs = DspBuffers {
            resamp_mid_l: &mut *self.resamp_mid_l,
            resamp_mid_r: &mut *self.resamp_mid_r,
            resamp_out_l: &mut *self.resamp_out_l,
            resamp_out_r: &mut *self.resamp_out_r,
            model_out_l: &mut *self.model_out_l,
            model_out_r: &mut *self.model_out_r,
            os_in_l: &mut *self.os_in_l,
            os_in_r: &mut *self.os_in_r,
            os_model_l: &mut *self.os_model_l,
            os_model_r: &mut *self.os_model_r,
            crossfade_scratch_l: &mut *self.xfd_l,
            crossfade_scratch_r: &mut *self.xfd_r,
        };
        let n_pw = capture_dsp_pipeline(in_l, in_r, n, ctx, bufs, rate);
        self.last_n_pw = n_pw;
        n_pw
    }

    fn out_l(&self) -> &[f32] {
        &self.resamp_out_l[..self.last_n_pw.min(MAX_RESAMP_BUF)]
    }

    fn out_r(&self) -> &[f32] {
        &self.resamp_out_r[..self.last_n_pw.min(MAX_RESAMP_BUF)]
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn load_model(name: &str) -> Box<StaticModel> {
    let path = model_path(name);
    assert!(path.exists(), "model fixture not found: {path:?}");
    let json = fs::read_to_string(&path).expect("read model json");
    let data = parse_nam_json(&json).expect("parse model json");
    let mut model = build_model(&data).expect("build model");
    model.prewarm(2048);
    model
}

fn synth_ir(len: usize, freq: f32, decay: f32) -> Vec<f32> {
    let sr = SAMPLE_RATE as f32;
    (0..len)
        .map(|i| {
            let t = i as f32 / sr;
            (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
        })
        .collect()
}

fn adapter_from_ir(ir: &[f32]) -> CabSimAdapter {
    CabSimAdapter::new(Box::new(
        ConvEngine::new(ir, PARTITION).expect("conv engine"),
    ))
    .expect("cab-sim adapter")
}

fn pair_from_ir(ir_l: &[f32], ir_r: &[f32]) -> Box<CabSimPair> {
    Box::new(CabSimPair {
        l: Box::new(adapter_from_ir(ir_l)),
        r: Box::new(adapter_from_ir(ir_r)),
        sample_rate: SAMPLE_RATE,
    })
}

/// Deterministic stereo signal pair (L != R).
fn signal_pair(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut seed_l: u64 = 0x1234_5678_9ABC_DEF0;
    let mut seed_r: u64 = 0xFEDC_BA98_7654_3210;
    let next = |seed: &mut u64| -> f32 {
        let mut x = *seed;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *seed = x;
        let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (r >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let l: Vec<f32> = (0..n).map(|_| 0.35 * next(&mut seed_l)).collect();
    let r: Vec<f32> = (0..n).map(|_| 0.35 * next(&mut seed_r)).collect();
    (l, r)
}

// ── 1. Stereo cabsim fidelity vs dual mono (feature = "stereo") ─────────────

/// A unified stereo pipeline (independent L/R models + `CabSimPair`) is
/// bit-exact against two isolated mono pipelines (single adapter each) across
/// multiple blocks: `MSE == 0.0`, `SNR > 120 dB`.
#[cfg(feature = "stereo")]
#[test]
fn cabsim_stereo_pipeline_bit_exact_vs_dual_mono() {
    const N_BLOCKS: usize = 96;
    const WARMUP: usize = 8;

    let ir_l = synth_ir(IR_LEN, 880.0, 14.0);
    let ir_r = synth_ir(IR_LEN, 1760.0, 22.0);

    // Unified stereo run: independent L/R model instances + CabSimPair.
    let mut stereo = CabsimPipeline::new(SAMPLE_RATE, SAMPLE_RATE);
    stereo.model_l = Some(load_model("BossWN-nano.nam"));
    stereo.model_r = Some(load_model("BossWN-nano.nam"));
    stereo.pair = Some(pair_from_ir(&ir_l, &ir_r));

    // Mono-L reference: single model + a pair whose `l` adapter carries the
    // L-channel IR (a dummy `r` keeps the same `conv_pair` stage shape as the
    // stereo run — a single shared adapter would double-advance while the
    // mono detector is still opening).
    let mut mono_l = CabsimPipeline::new(SAMPLE_RATE, SAMPLE_RATE);
    mono_l.model_l = Some(load_model("BossWN-nano.nam"));
    mono_l.pair = Some(pair_from_ir(&ir_l, &ir_l));

    // Mono-R reference.
    let mut mono_r = CabsimPipeline::new(SAMPLE_RATE, SAMPLE_RATE);
    mono_r.model_l = Some(load_model("BossWN-nano.nam"));
    mono_r.pair = Some(pair_from_ir(&ir_r, &ir_r));

    let (sig_l, sig_r) = signal_pair(N_BLOCKS * BLOCK);

    let mut sum_sq_err: f64 = 0.0;
    let mut sum_sq_signal: f64 = 0.0;
    let mut compared: usize = 0;
    let mut first_mismatch: Option<usize> = None;

    for block in 0..N_BLOCKS {
        let measuring = block >= WARMUP;
        let sl = &sig_l[block * BLOCK..(block + 1) * BLOCK];
        let sr = &sig_r[block * BLOCK..(block + 1) * BLOCK];

        let mut stereo_in_l = sl.to_vec();
        let mut stereo_in_r = sr.to_vec();
        let n_stereo = stereo.process(&mut stereo_in_l, &mut stereo_in_r, BLOCK, SAMPLE_RATE);
        let out_s_l = stereo.out_l().to_vec();
        let out_s_r = stereo.out_r().to_vec();

        let mut mono_l_in = sl.to_vec();
        let mut mono_l_in_r = sl.to_vec();
        let n_l = mono_l.process(&mut mono_l_in, &mut mono_l_in_r, BLOCK, SAMPLE_RATE);
        let out_m_l = mono_l.out_l().to_vec();

        let mut mono_r_in = sr.to_vec();
        let mut mono_r_in_r = sr.to_vec();
        let n_r = mono_r.process(&mut mono_r_in, &mut mono_r_in_r, BLOCK, SAMPLE_RATE);
        let out_m_r = mono_r.out_l().to_vec();

        assert_eq!(n_stereo, n_l, "stereo and mono-L n_pw must match");
        assert_eq!(n_stereo, n_r, "stereo and mono-R n_pw must match");
        let n = n_stereo.min(BLOCK);

        if measuring {
            for i in 0..n {
                if out_s_l[i].to_bits() != out_m_l[i].to_bits() && first_mismatch.is_none() {
                    first_mismatch = Some(block * BLOCK + i);
                }
                let el = (out_s_l[i] as f64 - out_m_l[i] as f64).abs();
                sum_sq_err += el * el;
                sum_sq_signal += out_s_l[i] as f64 * out_s_l[i] as f64;

                if out_s_r[i].to_bits() != out_m_r[i].to_bits() && first_mismatch.is_none() {
                    first_mismatch = Some(block * BLOCK + i);
                }
                let er = (out_s_r[i] as f64 - out_m_r[i] as f64).abs();
                sum_sq_err += er * er;
                sum_sq_signal += out_s_r[i] as f64 * out_s_r[i] as f64;

                compared += 2;
            }
        }
    }

    let mse = sum_sq_err / compared.max(1) as f64;
    let snr_db = if sum_sq_err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sum_sq_signal / sum_sq_err).log10()
    };

    assert!(compared > 0, "no samples compared");
    assert_eq!(
        mse, 0.0,
        "stereo vs dual-mono must be bit-exact (MSE==0), got {mse} (first mismatch {:?})",
        first_mismatch
    );
    assert!(
        snr_db > 120.0,
        "SNR must exceed 120 dB, got {snr_db:.2} dB (first mismatch {:?})",
        first_mismatch
    );
}

// ── 2. IR multirate consistency ─────────────────────────────────────────────

/// Resampling an IR from its native rate to the applied host rate must yield a
/// valid impulse response: the meaningful (energy-bearing) head scales by the
/// rate ratio and the resampled IR convolved through `ConvEngine` matches
/// direct convolution (reference).
#[test]
fn cabsim_ir_multirate_resample_is_consistent() {
    let ir_48k = synth_ir(4096, 440.0, 10.0);

    // Resample 48 kHz → 44.1 kHz (common host rate).
    let ir_44100 = CabSimIr::resample(&ir_48k, 48_000, 44_100).expect("resample IR");

    assert!(!ir_44100.is_empty(), "resampled IR must not be empty");
    assert!(
        ir_44100.iter().all(|s| s.is_finite()),
        "resampled IR must be finite"
    );

    // The impulse response's energy must be concentrated in the head, whose
    // length scales by the rate ratio (the resampler may append a zero-padded
    // flush tail beyond it — its exact length is not part of the contract).
    let expected_len = (4096.0_f64 * 44_100.0_f64 / 48_000.0_f64).round() as usize;
    let head_window = ir_44100.len().min(expected_len * 3);
    let head_energy: f32 = ir_44100[..head_window].iter().map(|s| s * s).sum();
    let total_energy: f32 = ir_44100.iter().map(|s| s * s).sum();
    assert!(
        total_energy > 0.0 && head_energy > 0.99 * total_energy,
        "resampled IR energy must be concentrated in the ~{expected_len}-sample head \
         (head {head_energy} vs total {total_energy})"
    );

    // The resampled IR is a valid impulse response: ConvEngine (partitioned,
    // frequency-domain) output matches direct convolution of the same IR
    // (aligned from index 0, as in `cabsim_golden`).
    let engine_ir: Vec<f32> = ir_44100.iter().take(512).copied().collect();
    let mut engine = ConvEngine::new(&engine_ir, 128).expect("conv engine");
    let input: Vec<f32> = signal_pair(256).0;

    let conv_out = common::conv_helpers::process_full_signal(&mut engine, &input);
    let reference = direct_convolve(&engine_ir, &input);

    assert!(
        conv_out.len() >= reference.len(),
        "conv engine output {} shorter than reference {}",
        conv_out.len(),
        reference.len()
    );
    let min_len = reference.len();
    let mse = compute_mse(&conv_out[..min_len], &reference[..min_len]);
    assert!(
        mse < 1e-4,
        "ConvEngine vs direct convolution on the resampled IR diverges: MSE {mse}"
    );
}

/// The IR loader reports the calibrated target rate and normalizes correctly.
#[test]
fn cabsim_ir_multirate_load_metadata() {
    // Write a synthetic 44.1 kHz WAV and load it targeted at 48 kHz.
    let dir = std::env::temp_dir().join("nam-cabsim-stereo-test");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("ir_44100.wav");

    let sr = 44_100u32;
    let samples: Vec<f32> = (0..2048)
        .map(|i| {
            let t = i as f32 / sr as f32;
            (std::f32::consts::TAU * 440.0 * t).sin() * (-30.0 * t).exp()
        })
        .collect();
    write_wav_f32(&path, sr, &samples).expect("write IR wav");

    let ir = CabSimIr::load(&path, 48_000, true).expect("load IR at target rate");
    assert_eq!(
        ir.sample_rate, 48_000,
        "IR must be recalibrated to the target rate (multirate)"
    );
    assert_eq!(ir.original_rate, 44_100);
    assert!(ir.normalized, "IR normalization must be requested");
    assert!(ir.samples.iter().all(|s| s.is_finite()));

    let peak = ir.samples.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
    assert!(
        (peak - 1.0).abs() < 1e-3,
        "normalized IR peak must be ~1.0, got {peak}"
    );
    let _ = std::fs::remove_file(&path);
}

/// Minimal mono f32 WAV writer for the IR loader.
fn write_wav_f32(path: &std::path::Path, sample_rate: u32, samples: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    let data_len = (samples.len() * 4) as u32;
    out.write_all(b"RIFF")?;
    out.write_all(&(36 + data_len).to_le_bytes())?;
    out.write_all(b"WAVE")?;
    out.write_all(b"fmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&1u16.to_le_bytes())?; // PCM
    out.write_all(&1u16.to_le_bytes())?; // mono
    out.write_all(&sample_rate.to_le_bytes())?;
    out.write_all(&(sample_rate * 4).to_le_bytes())?; // byte rate
    out.write_all(&4u16.to_le_bytes())?; // block align
    out.write_all(&32u16.to_le_bytes())?; // bits
    out.write_all(b"data")?;
    out.write_all(&data_len.to_le_bytes())?;
    for &s in samples {
        out.write_all(&s.to_le_bytes())?;
    }
    out.flush()
}

// ── 3. Zero allocations on the stereo cabsim path (feature = "heap-audit") ──

/// The full stereo pipeline with an active `CabSimPair` performs zero heap
/// allocations on the processing thread.
#[cfg(feature = "heap-audit")]
#[test]
fn cabsim_stereo_pipeline_heap_audit_zero_alloc() {
    use common::alloc_audit::{TrackingGuard, get_alloc_count};

    let ir_l = synth_ir(IR_LEN, 880.0, 14.0);
    let ir_r = synth_ir(IR_LEN, 1760.0, 22.0);

    let mut pipeline = CabsimPipeline::new(SAMPLE_RATE, SAMPLE_RATE);
    pipeline.model_l = Some(load_model("BossWN-nano.nam"));
    pipeline.model_r = Some(load_model("BossWN-nano.nam"));
    pipeline.pair = Some(pair_from_ir(&ir_l, &ir_r));

    // Warm-up outside the guard (fills convolution FDLs, model prewarm state).
    let (sig_l, sig_r) = signal_pair(160 * BLOCK);
    for block in 0..8 {
        let mut l = sig_l[block * BLOCK..(block + 1) * BLOCK].to_vec();
        let mut r = sig_r[block * BLOCK..(block + 1) * BLOCK].to_vec();
        pipeline.process(&mut l, &mut r, BLOCK, SAMPLE_RATE);
    }

    let allocs = {
        let _guard = TrackingGuard::new();
        let mut in_l = [0.0f32; BLOCK];
        let mut in_r = [0.0f32; BLOCK];
        for block in 0..128 {
            in_l.copy_from_slice(&sig_l[block * BLOCK..(block + 1) * BLOCK]);
            in_r.copy_from_slice(&sig_r[block * BLOCK..(block + 1) * BLOCK]);
            pipeline.process(&mut in_l, &mut in_r, BLOCK, SAMPLE_RATE);
        }
        get_alloc_count()
    };
    assert_eq!(
        allocs, 0,
        "stereo cabsim pipeline allocated on the processing thread: {allocs}"
    );
}
