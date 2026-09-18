// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration tests for the engine-level capture pipeline entry points:
//!
//! 1. **Headless mode (F5)** — `bridge_writer: None` is a first-class mode:
//!    `capture_dsp_pipeline_streaming` runs the full stage orchestration and
//!    returns `> 0` for non-silent input instead of bailing out early.
//! 2. **Cab-sim IR tail drain (F8)** — after the noise gate closes, the
//!    engine keeps emitting the decaying convolution tail (bounded by the
//!    armed ring-out budget) before switching to true silence.

use neural_amp_modeler_rs::common::spsc::RtStatusFlags;
use neural_amp_modeler_rs::dsp::adaptive::{AdaptiveCompute, AdaptiveComputeMode};
use neural_amp_modeler_rs::dsp::cabsim::adapter::{CabSimAdapter, CabSimPair};
use neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine;
use neural_amp_modeler_rs::dsp::gate::{DynamicHysteresis, GateParams};
use neural_amp_modeler_rs::dsp::oversample::{OversampleEngine, OversampleFactor};
use neural_amp_modeler_rs::dsp::pipeline::{
    DspBuffers, DspPipelineContext, MAX_RESAMP_BUF, capture_dsp_pipeline_streaming,
};
use neural_amp_modeler_rs::dsp::resampler::NamResampler;
use neural_amp_modeler_rs::dsp::resampling::StreamingResampleBuffer;

const BLOCK: usize = 64;
const PARTITION: usize = 256;
const IR_LEN: usize = 2048;
const HOST_RATE: u32 = 44_100;
const NAM_RATE: u32 = 48_000;

/// Deterministic stereo noise pair (L != R, both well above the gate's open
/// threshold at 0.5 amplitude).
fn signal_pair(n: usize, seed_l: u64, seed_r: u64) -> (Vec<f32>, Vec<f32>) {
    let next = |seed: &mut u64| -> f32 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let mut seed_l = seed_l;
    let mut seed_r = seed_r;
    let l: Vec<f32> = (0..n).map(|_| 0.5 * next(&mut seed_l)).collect();
    let r: Vec<f32> = (0..n).map(|_| 0.5 * next(&mut seed_r)).collect();
    (l, r)
}

fn synth_ir(len: usize, freq: f32, decay: f32, sample_rate: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
        })
        .collect()
}

/// Owning streaming pipeline driver mirroring how a host wires
/// `capture_dsp_pipeline_streaming` — deliberately headless
/// (`bridge_writer: None`) to exercise the no-bridge consumer mode.
struct StreamPipeline {
    resampler: NamResampler,
    stream: StreamingResampleBuffer,
    os_l: OversampleEngine,
    os_r: OversampleEngine,
    pair: Option<Box<CabSimPair>>,
    input_gain_mult: f32,
    output_gain_mult: f32,
    gate_params: GateParams,
    silence_hysteresis: DynamicHysteresis,
    mono_hysteresis: DynamicHysteresis,
    threshold_open_sq: f32,
    threshold_close_sq: f32,
    process_mono: bool,
    rt_status: RtStatusFlags,
    adaptive: AdaptiveCompute,
    resamp_mid_l: Box<[f32; MAX_RESAMP_BUF]>,
    resamp_mid_r: Box<[f32; MAX_RESAMP_BUF]>,
    resamp_out_l: Box<[f32; MAX_RESAMP_BUF]>,
    resamp_out_r: Box<[f32; MAX_RESAMP_BUF]>,
    model_out_l: Box<[f32; MAX_RESAMP_BUF]>,
    model_out_r: Box<[f32; MAX_RESAMP_BUF]>,
    last_n_pw: usize,
}

impl StreamPipeline {
    fn new(gate_params: GateParams) -> Self {
        use neural_amp_modeler_rs::math::dsp::gain_lut::get_gain_lut;
        let lut = get_gain_lut();
        let open_lin = lut.db_to_linear(gate_params.threshold_open_db);
        let close_lin = lut.db_to_linear(gate_params.threshold_close_db);
        Self {
            resampler: NamResampler::new(HOST_RATE, NAM_RATE, BLOCK).expect("resampler"),
            stream: StreamingResampleBuffer::new(HOST_RATE, NAM_RATE, MAX_RESAMP_BUF)
                .expect("stream"),
            os_l: OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).expect("os"),
            os_r: OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).expect("os"),
            pair: None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params,
            silence_hysteresis: DynamicHysteresis::new(),
            mono_hysteresis: DynamicHysteresis::new(),
            threshold_open_sq: open_lin * open_lin,
            threshold_close_sq: close_lin * close_lin,
            process_mono: false,
            rt_status: RtStatusFlags::new(),
            adaptive: AdaptiveCompute::new(AdaptiveComputeMode::Off),
            resamp_mid_l: Box::new([0.0; MAX_RESAMP_BUF]),
            resamp_mid_r: Box::new([0.0; MAX_RESAMP_BUF]),
            resamp_out_l: Box::new([0.0; MAX_RESAMP_BUF]),
            resamp_out_r: Box::new([0.0; MAX_RESAMP_BUF]),
            model_out_l: Box::new([0.0; MAX_RESAMP_BUF]),
            model_out_r: Box::new([0.0; MAX_RESAMP_BUF]),
            last_n_pw: 0,
        }
    }

    /// Processes one block with `bridge_writer: None`; returns the pipeline
    /// return value and the peak magnitude of the produced left output.
    fn process(&mut self, in_l: &mut [f32], in_r: &mut [f32], n: usize) -> (usize, f32) {
        // Disjoint mutable field borrows through `self` (same pattern as the
        // engine's own stereo harness).
        let ctx = DspPipelineContext {
            resampler: &mut self.resampler,
            os_l: &mut self.os_l,
            os_r: &mut self.os_r,
            active_model_l: &mut None,
            active_model_r: &mut None,
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
            bridge_writer: None,
            conv: None,
            conv_pair: self.pair.as_deref_mut(),
        };
        let mut os_in_l = [0.0f32; MAX_RESAMP_BUF];
        let mut os_in_r = [0.0f32; MAX_RESAMP_BUF];
        let mut os_model_l = [0.0f32; MAX_RESAMP_BUF];
        let mut os_model_r = [0.0f32; MAX_RESAMP_BUF];
        let bufs = DspBuffers {
            resamp_mid_l: &mut *self.resamp_mid_l,
            resamp_mid_r: &mut *self.resamp_mid_r,
            resamp_out_l: &mut *self.resamp_out_l,
            resamp_out_r: &mut *self.resamp_out_r,
            model_out_l: &mut *self.model_out_l,
            model_out_r: &mut *self.model_out_r,
            os_in_l: &mut os_in_l,
            os_in_r: &mut os_in_r,
            os_model_l: &mut os_model_l,
            os_model_r: &mut os_model_r,
            crossfade_scratch_l: &mut [],
            crossfade_scratch_r: &mut [],
        };
        let n_pw =
            capture_dsp_pipeline_streaming(in_l, in_r, n, ctx, &mut self.stream, bufs, HOST_RATE);
        self.last_n_pw = n_pw;
        let peak = self.resamp_out_l[..n_pw.min(MAX_RESAMP_BUF)]
            .iter()
            .fold(0.0f32, |a, &s| a.max(s.abs()));
        (n_pw, peak)
    }
}

fn pair_from_ir(ir: &[f32]) -> Box<CabSimPair> {
    let make = || {
        CabSimAdapter::new(Box::new(
            ConvEngine::new(ir, PARTITION).expect("conv engine"),
        ))
        .expect("cab-sim adapter")
    };
    Box::new(CabSimPair {
        l: Box::new(make()),
        r: Box::new(make()),
        sample_rate: NAM_RATE,
    })
}

// ── 1. Headless mode: bridge_writer = None (F5) ─────────────────────────────

/// With `bridge_writer: None` the streaming entry must run the full pipeline
/// and return `> 0` for non-silent input — the early return on
/// `bridge_writer.is_none()` is gone, so bridge-less consumers get the whole
/// orchestrated pipeline instead of a silent stub.
#[test]
fn streaming_pipeline_processes_without_bridge_writer() {
    let mut pipeline = StreamPipeline::new(GateParams::default());
    let (sig_l, sig_r) = signal_pair(8 * BLOCK, 0xA11CE, 0xB0B);

    for block in 0..8 {
        let mut in_l = sig_l[block * BLOCK..(block + 1) * BLOCK].to_vec();
        let mut in_r = sig_r[block * BLOCK..(block + 1) * BLOCK].to_vec();
        let (n_pw, _) = pipeline.process(&mut in_l, &mut in_r, BLOCK);
        assert_eq!(
            n_pw, BLOCK,
            "headless streaming pipeline must honor strict cardinality (block {block})"
        );
    }
    // The last block's output must carry real signal — the pipeline actually
    // processed audio instead of returning a silent stub.
    assert!(
        pipeline.last_n_pw > 0,
        "headless pipeline must report processed samples"
    );
    assert!(
        pipeline.resamp_out_l[..pipeline.last_n_pw]
            .iter()
            .any(|&s| s.abs() > 1e-3),
        "headless pipeline must produce non-silent output for non-silent input"
    );
}

// ── 2. Cab-sim IR tail drain on gate closure (F8) ───────────────────────────

/// After the gate closes on silent input, the engine must keep emitting the
/// decaying cab-sim IR tail for the armed ring-out budget before switching to
/// true silence — the reverb tail must not be truncated by instant silence.
#[test]
fn streaming_pipeline_drains_cabsim_tail_after_gate_close() {
    // hold = 0 (close immediately on silence), fade = 256 frames (~4 blocks).
    let gate = GateParams::new(-70.0, -80.0, 0, 256, 1e-4);
    let mut pipeline = StreamPipeline::new(gate);
    pipeline.pair = Some(pair_from_ir(&synth_ir(IR_LEN, 880.0, 3.0, NAM_RATE)));

    const LOUD_BLOCKS: usize = 32; // fills the FDL (8 partitions of 256)
    const SILENCE_BLOCKS: usize = 56;

    let (sig_l, sig_r) = signal_pair(LOUD_BLOCKS * BLOCK, 0xC0FFEE, 0xF00D);
    for block in 0..LOUD_BLOCKS {
        let mut l = sig_l[block * BLOCK..(block + 1) * BLOCK].to_vec();
        let mut r = sig_r[block * BLOCK..(block + 1) * BLOCK].to_vec();
        let (n_pw, _) = pipeline.process(&mut l, &mut r, BLOCK);
        assert_eq!(n_pw, BLOCK, "loud block {block} must be fully processed");
    }

    // Silence onset: record per-block peak + return value while the gate
    // closes (fade), drains the IR tail, and finally rests in true silence.
    // Budget: (num_partitions + 1) * PARTITION = 9 * 256 = 2304 samples
    // = 36 blocks of 64. Fade adds ~4 more blocks before the drain starts.
    let mut peaks = Vec::with_capacity(SILENCE_BLOCKS);
    let mut returns = Vec::with_capacity(SILENCE_BLOCKS);
    for _ in 0..SILENCE_BLOCKS {
        let mut l = vec![0.0f32; BLOCK];
        let mut r = vec![0.0f32; BLOCK];
        let (n_pw, peak) = pipeline.process(&mut l, &mut r, BLOCK);
        peaks.push(peak);
        returns.push(n_pw);
    }

    // The IR tail must survive far beyond the gate fade (~4 blocks): with the
    // drain, non-silent output reaches deep into the ring-out window
    // (≥ num_partitions * PARTITION / BLOCK = 32 blocks after silence onset).
    // Without the drain, output would cut to zero at the end of the fade.
    let last_nonzero = peaks
        .iter()
        .rposition(|&p| p > 1e-4)
        .expect("silence phase must contain signal (fade or ring-out)");
    let min_drain_reach = (8 * PARTITION) / BLOCK;
    assert!(
        last_nonzero >= min_drain_reach,
        "IR tail must ring out well past the gate fade: last non-silent block {last_nonzero}, \
         expected ≥ {min_drain_reach}"
    );

    // The drained window must actually emit audible tail samples (not just
    // the input-stage fade), and the drain must deliver the strict-cardinality
    // block size while it runs.
    let drain_window = &peaks[min_drain_reach..=last_nonzero];
    assert!(
        drain_window.iter().any(|&p| p > 1e-4),
        "drain window must emit non-zero IR ring-out samples"
    );
    assert!(
        returns[..=last_nonzero].iter().all(|&n| n == BLOCK),
        "drained blocks must still consume/produce exactly {BLOCK} samples"
    );

    // Termination: after the armed budget is consumed, the pipeline rests in
    // true silence (return 0, all-zero output) and stays there. Worst-case
    // onset: fade (4 blocks) + budget (36) = 41; start the rest-window at 44.
    let rest_start = min_drain_reach + (PARTITION / BLOCK) + 8;
    for i in rest_start..SILENCE_BLOCKS {
        assert_eq!(
            returns[i], 0,
            "block {i}: tail exhausted — expected true silence"
        );
        assert_eq!(
            peaks[i], 0.0,
            "block {i}: output must be exactly zero after the drain budget"
        );
    }
}
