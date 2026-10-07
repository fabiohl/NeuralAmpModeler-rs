// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `SequentialModel` — serial composition ("chain") of independent DSP models.
//!
//! Mirrors the C++ NAMcore `sequential.cpp` model entity (`SequentialModel`,
//! v0.6.0):
//!
//! - **Construction** (`SequentialModel` ctor, `sequential.cpp:123-129`):
//!   stages arrive already built; the constructor derives
//!   `get_input_channels` / `get_output_channels` (first/last stage, L64-80),
//!   resolves the expected sample rate (`resolve_expected_sample_rate`,
//!   L36-62, DEC-01 in [`docs/cpp_parity_map.md`](docs/cpp_parity_map.md) §5.1)
//!   and validates the stage channel links (`validate_channel_links`,
//!   L82-93: `output_channels(i) == input_channels(i+1)`).
//! - **Reset** (`SequentialModel::Reset`, L152-180): the child
//!   `PrewarmOnReset` flags are saved, disabled, children are reset, the
//!   flags are restored, and — gated by the chain's own flag — a **single
//!   chain-wide stabilization pass** runs. Children never stabilize
//!   individually inside the chain, avoiding double stabilization and the
//!   numeric divergence it would cause.
//! - **Prewarm** (`GetPrewarmSamples`, L189-199): the chain's stabilization
//!   count is the saturating sum of the child counts.
//! - **Buffers** (`SetMaxBufferSize`, L202-222): `-stages-1` intermediate
//!   buffers sized per stage output channels, allocated off-RT and reused
//!   zero-alloc on the hot path; the last stage writes directly into the
//!   caller's output buffer.
//!
//! # Declared divergences (hardening)
//!
//! - Blocks larger than the negotiated maximum (`sequential.cpp:133-136`,
//!   where the C++ throws `"maximum buffer size"`): the RT hot path cannot
//!   unwind, so the decided contract is **controlled truncation in release** —
//!   exactly the first `max_buffer_size` frames are processed and the
//!   caller's output tail is left byte-untouched, symmetric for input and
//!   output — with a `debug_assert!` trap that dev/CI builds exercise (the
//!   `#[should_panic]` mirror of upstream
//!   `test_sequential_rejects_blocks_larger_than_reset_maximum`). Never a
//!   panic on the audio thread.
//! - The all-unknown sample-rate case resolves to the engine's global
//!   48 kHz standalone default instead of the C++ `-1.0` sentinel (DEC-01,
//!   option A — see `resolve_expected_sample_rate`).
//! - The chain stabilization count is fed at exact sample granularity;
//!   the C++ `DSP::prewarm` rounds up to whole `max_buffer_size` chunks,
//!   which can overshoot by `chunk - 1` zeros when a child is LSTM-driven
//!   (asymptotic recurrent state). Feed-exactness keeps the split and
//!   integral flows bit-identical regardless of caller chunking.

use super::{NamModel, StaticModel};
use crate::common::diagnostics::NamErrorCode;
use crate::loader::loaded_model_pair::DEFAULT_SAMPLE_RATE;
use crate::math::common::AlignedVec;
use log::error;

/// Fallback chunk length for the chain stabilization pass when no block size
/// has been negotiated yet (C++ `NAM_DEFAULT_MAX_BUFFER_SIZE`, `dsp.h:25-26`).
const DEFAULT_MAX_BUFFER_SIZE: usize = 4096;

/// Resolves the chain expected sample rate (C++
/// `resolve_expected_sample_rate`, `sequential.cpp:36-62`, DEC-01 option A).
///
/// `top_level_rate` is the rate declared in the root envelope (`None` =
/// unspecified); `child_rates[i]` is the rate declared by the `i`-th child
/// envelope (`None` = unspecified, including the C++ `-1.0` unknown marker
/// already normalized to `None` by the strict deserializer).
///
/// Resolution rules:
///
/// 1. The top-level rate, when known, seeds the expectation; children with
///    unknown rates are ignored and children with known rates must match it.
/// 2. Without a top-level rate, the first known child dictates the chain
///    rate; every subsequent known child must match it.
/// 3. Any conflict between two known declarations is rejected fail-closed
///    with [`NamErrorCode::SequentialSampleRateMismatch`].
/// 4. When nothing is declared (top level and all children unknown), the
///    resolved rate falls back to the engine global default (48 kHz) so a
///    chain always carries a usable expected rate.
///
/// Comparisons are exact `f32` equality, mirroring the C++ `double` compare:
/// declared sample rates are integral (< 2^24 values), so their binary
/// representation is exact.
pub fn resolve_expected_sample_rate(
    child_rates: &[Option<f32>],
    top_level_rate: Option<f32>,
) -> Result<f32, NamErrorCode> {
    let mut resolved = top_level_rate;
    for (i, child_rate) in child_rates.iter().enumerate() {
        let child_rate = match child_rate {
            None => continue,
            Some(rate) => *rate,
        };
        match resolved {
            None => resolved = Some(child_rate),
            Some(expected) => {
                if child_rate != expected {
                    error!(
                        "[Dispatcher] Sequential validation failed: submodel[{}].sample_rate={} \
                         conflicts with governing rate={} (E1309 SequentialSampleRateMismatch)",
                        i, child_rate, expected
                    );
                    return Err(NamErrorCode::SequentialSampleRateMismatch);
                }
            }
        }
    }
    Ok(resolved.unwrap_or(DEFAULT_SAMPLE_RATE))
}

/// Saturating sum of per-child stabilization counts (C++
/// `SequentialModel::GetPrewarmSamples` loops, `sequential.cpp:189-199`,
/// saturating at `i32::MAX`; the Rust engine saturates at `usize::MAX`).
pub(crate) fn saturating_prewarm_sum(child_samples: impl Iterator<Item = usize>) -> usize {
    child_samples.fold(0usize, usize::saturating_add)
}

/// Serial chain of already-constructed model stages.
///
/// Geometries are fixed at construction: the chain input channels are the
/// first child's input channels, the chain output channels the last child's
/// output channels, and every interior link is validated by
/// [`new`](Self::new). Stage processing order is fixed (stage 0 first).
pub struct SequentialModel {
    /// Chain stages in processing order (never empty; enforced in
    /// [`new`](Self::new)).
    models: Vec<StaticModel>,
    /// First stage input channels (C++ `NumInputChannels`).
    in_channels: usize,
    /// Last stage output channels (C++ `NumOutputChannels`).
    out_channels: usize,
    /// Resolved expected sample rate (DEC-01; always concrete).
    expected_sample_rate: f32,
    /// Whether [`reset`](Self::reset) runs the chain stabilization pass
    /// (C++ `mPrewarmOnReset`; propagated by `SetPrewarmOnReset`).
    prewarm_on_reset: bool,
    /// Outstanding split-stabilization unit count (armed exclusively by
    /// [`prewarm_reset`](Self::prewarm_reset)).
    prewarm_pending: usize,
    /// Negotiated maximum block size in frames (C++ `mMaxBufferSize`).
    max_buffer_size: usize,
    /// Frames currently reserved per stage-buffer channel plane.
    stage_frames: usize,
    /// One intermediate plane per boundary `b` (output of stage `b`,
    /// `b < stages - 1`), laid out as `out_channels(b) x stage_frames`.
    stage_buffers: Vec<AlignedVec<f32>>,
    /// Channel pointer tables over [stage_buffers]: boundary `b` output
    /// pointers (rebuilt with the buffers, off-RT). Channels are stored as
    /// `usize` addresses to keep the chain `Send + Sync` (the engine is
    /// x86-64-v3 only, where raw pointers and `usize` share layout) and are
    /// cast back per call with zero allocation.
    stage_out_ptrs: Vec<AlignedVec<usize>>,
    /// Channel pointer tables over [stage_buffers]: stage `b + 1` input
    /// pointers (rebuilt with the buffers, off-RT).
    stage_in_ptrs: Vec<AlignedVec<usize>>,
    /// Owned zero source plane (`in_channels x stage_frames`), zero-filled
    /// once at allocation and never written again; backs the chain
    /// stabilization zero-feed on every channel count with zero allocs.
    zero_plane: AlignedVec<f32>,
    /// Channel pointers into [zero_plane].
    zero_in_ptrs: AlignedVec<usize>,
    /// Owned discard plane (`out_channels x stage_frames`) receiving the
    /// last stage output during chain stabilization feeds.
    sink_plane: AlignedVec<f32>,
    /// Channel pointers into [sink_plane].
    sink_out_ptrs: AlignedVec<usize>,
}

impl SequentialModel {
    /// Validates and assembles the chain (C++ `SequentialModel` ctor,
    /// `sequential.cpp:123-129`; presence/shape of the child array is already
    /// fail-closed by the loader topology scan and re-checked here).
    ///
    /// `declared_child_rates[i]` must be the envelope-declared rate of
    /// `models[i]` (aligned zip; `None` = unspecified). `top_level_rate` is
    /// the root envelope's declared rate (`None` = unspecified).
    ///
    /// Stage intermediate buffers are sized eagerly for the engine's
    /// documented maximum block (`MAX_RESAMP_BUF`), so the chain is process-
    /// ready even before an explicit
    /// [`set_max_buffer_size`](NamModel::set_max_buffer_size) call.
    ///
    /// # Errors
    /// - [`NamErrorCode::SequentialEmptyModels`] — no stages provided.
    /// - [`NamErrorCode::SequentialSampleRateMismatch`] — two known
    ///   declarations disagree (DEC-01).
    /// - [`NamErrorCode::SequentialChannelMismatch`] — a link violates
    ///   `output_channels(i) == input_channels(i+1)`.
    /// - [`NamErrorCode::OutOfMemory`] — stage buffer allocation failed.
    pub fn new(
        models: Vec<StaticModel>,
        declared_child_rates: Vec<Option<f32>>,
        top_level_rate: Option<f32>,
    ) -> Result<Self, NamErrorCode> {
        if models.is_empty() {
            error!(
                "[Dispatcher] Sequential validation failed: legal chains require a \
                 non-empty 'config.models' array (E1306 SequentialEmptyModels)"
            );
            return Err(NamErrorCode::SequentialEmptyModels);
        }
        debug_assert_eq!(
            models.len(),
            declared_child_rates.len(),
            "declared_child_rates must be aligned with models"
        );
        let in_channels = models[0].in_channels();
        let out_channels = models[models.len() - 1].num_output_channels();
        let expected_sample_rate = resolve_expected_sample_rate(
            &declared_child_rates[..declared_child_rates.len().min(models.len())],
            top_level_rate,
        )?;
        for link in 1..models.len() {
            let left = models[link - 1].num_output_channels();
            let right = models[link].in_channels();
            if left != right {
                error!(
                    "[Dispatcher] Sequential validation failed: submodel[{}] outputs {} channels \
                     but submodel[{}] consumes {} (E1308 SequentialChannelMismatch)",
                    link - 1,
                    left,
                    link,
                    right
                );
                return Err(NamErrorCode::SequentialChannelMismatch);
            }
        }

        let mut model = Self {
            models,
            in_channels,
            out_channels,
            expected_sample_rate,
            prewarm_on_reset: true,
            prewarm_pending: 0,
            max_buffer_size: 0,
            stage_frames: 0,
            stage_buffers: Vec::new(),
            stage_out_ptrs: Vec::new(),
            stage_in_ptrs: Vec::new(),
            zero_plane: AlignedVec::empty(),
            zero_in_ptrs: AlignedVec::empty(),
            sink_plane: AlignedVec::empty(),
            sink_out_ptrs: AlignedVec::empty(),
        };
        model
            .resize_stage_buffers(crate::dsp::pipeline::MAX_RESAMP_BUF)
            .map_err(|_| NamErrorCode::OutOfMemory)?;
        Ok(model)
    }

    /// Number of stages in the chain.
    pub fn num_stages(&self) -> usize {
        self.models.len()
    }

    /// Stage references (in processing order) for diagnostics and tests.
    #[cfg(test)]
    pub(crate) fn stages(&mut self) -> &mut [StaticModel] {
        &mut self.models
    }

    /// First stage input channels (chain input channels).
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Last stage output channels (chain output channels).
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Resolved chain expected sample rate (DEC-01; the C++
    /// `GetExpectedSampleRate()` counterpart with the all-unknown cases
    /// mapped to the engine's 48 kHz global default).
    pub fn expected_sample_rate(&self) -> f32 {
        self.expected_sample_rate
    }

    /// Output channel width of stage `stage` (`None` when out of range).
    ///
    /// Diagnostics helper describing the interior chain geometry.
    pub fn stage_output_channels(&self, stage: usize) -> Option<usize> {
        self.models.get(stage).map(|m| m.num_output_channels())
    }

    /// (Re)sizes every intermediate stage plane and requires the pointer
    /// tables, zero source, and sink planes (C++ `SetMaxBufferSize`
    /// body, `sequential.cpp:202-222`). Off-RT only: allocates.
    fn resize_stage_buffers(&mut self, frames: usize) -> Result<(), NamErrorCode> {
        self.max_buffer_size = frames;
        let intermediate = self.models.len().saturating_sub(1);
        if self.stage_buffers.len() != intermediate {
            self.stage_buffers = (0..intermediate).map(|_| AlignedVec::empty()).collect();
            self.stage_in_ptrs = (0..intermediate).map(|_| AlignedVec::empty()).collect();
            self.stage_out_ptrs = (0..intermediate).map(|_| AlignedVec::empty()).collect();
        }

        for boundary in 0..intermediate {
            let channels = self.models[boundary].num_output_channels();
            let plane = channels
                .checked_mul(frames)
                .ok_or(NamErrorCode::OutOfMemory)?;
            self.stage_buffers[boundary].resize(plane, 0.0f32)?;
            let base = self.stage_buffers[boundary].as_mut_ptr();
            let mut out_ptrs = Vec::with_capacity(channels);
            let mut in_ptrs = Vec::with_capacity(channels);
            for ch in 0..channels {
                // SAFETY: `base` covers `channels * frames` elements and the
                // channel sub-planes stay inside that capacity by stride.
                let channel_ptr = unsafe { base.add(ch * frames) };
                out_ptrs.push(channel_ptr as usize);
                in_ptrs.push(channel_ptr as usize);
            }
            self.stage_out_ptrs[boundary] = AlignedVec::from_vec(out_ptrs)?;
            self.stage_in_ptrs[boundary] = AlignedVec::from_vec(in_ptrs)?;
        }
        self.stage_frames = frames;

        // Stabilization planes: zero source (never written afterwards) and
        // the last-stage discard sink.
        let zero_len = self
            .in_channels
            .checked_mul(frames)
            .ok_or(NamErrorCode::OutOfMemory)?;
        self.zero_plane.resize(zero_len, 0.0f32)?;
        let mut zero_ptrs = Vec::with_capacity(self.in_channels);
        for ch in 0..self.in_channels {
            // SAFETY: zero plane holds `in_channels * frames` elements; the
            // channel sub-planes stay inside that capacity by stride.
            zero_ptrs.push(unsafe { self.zero_plane.as_mut_ptr().add(ch * frames) } as usize);
        }
        self.zero_in_ptrs = AlignedVec::from_vec(zero_ptrs)?;

        let sink_len = self
            .out_channels
            .checked_mul(frames)
            .ok_or(NamErrorCode::OutOfMemory)?;
        self.sink_plane.resize(sink_len, 0.0f32)?;
        let mut sink_ptrs = Vec::with_capacity(self.out_channels);
        for ch in 0..self.out_channels {
            // SAFETY: sink plane holds `out_channels * frames` elements; the
            // channel sub-planes stay inside that capacity by stride.
            sink_ptrs.push(unsafe { self.sink_plane.as_mut_ptr().add(ch * frames) } as usize);
        }
        self.sink_out_ptrs = AlignedVec::from_vec(sink_ptrs)?;
        Ok(())
    }
}

impl super::sealed::Sealed for SequentialModel {}

impl NamModel for SequentialModel {
    /// Processes a mono block through the whole chain.
    ///
    /// The chain must be mono (`in_channels == out_channels == 1`); other
    /// geometries must use [`process_multichannel`](Self::process_multichannel)
    /// or [`process_raw`](Self::process_raw). On violation, this no-ops in
    /// release (documented block-contract degradation) and traps in debug.
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert!(
            self.in_channels == 1 && self.out_channels == 1,
            "Sequential chain with channels {}->{} must use process_multichannel",
            self.in_channels,
            self.out_channels
        );
        if self.in_channels != 1 || self.out_channels != 1 {
            return;
        }
        let n = input.len().min(output.len());
        debug_assert!(
            n <= self.max_buffer_size,
            "block of {} frames exceeds the negotiated maximum {} \
             (upstream SequentialModel::process throws; release truncates \
             to the first `max_buffer_size` frames)",
            n,
            self.max_buffer_size
        );
        let in_arr = [input.as_ptr()];
        let out_arr = [output.as_mut_ptr()];
        // SAFETY: `in_arr`/`out_arr` hold one valid borrow of `num_frames`
        // elements each (see `NamModel::process_raw` safety contract).
        unsafe { self.process_chain(in_arr.as_ptr(), out_arr.as_ptr(), n) };
    }

    /// Processes a multichannel block through the whole chain (C++
    /// `SequentialModel::process`, `sequential.cpp:131-147`): stage 0
    /// consumes the caller's input pointers and the last stage writes
    /// directly into the caller's output pointers; interior stages run over
    /// the pre-allocated boundary planes.
    ///
    /// # Safety
    /// See [`NamModel::process_raw`]: `input` must provide `in_channels`
    /// valid pointers and `output` `out_channels` valid writable pointers,
    /// each covering `num_frames` samples. In-place processing (aliasing
    /// caller input and output) is safe by construction: interior writing
    /// stages read their (distinct) boundary planes before the last stage
    /// writes into the caller's output. `num_frames > max_buffer_size`
    /// follows the module truncation contract (debug trap, release
    /// truncation) and never panics.
    #[inline(always)]
    unsafe fn process_raw(
        &mut self,
        input: *const *const f32,
        output: *const *mut f32,
        num_frames: usize,
    ) {
        // SAFETY: pointers and lengths satisfy the `process_raw` contract
        // documented on `NamModel::process_raw`.
        unsafe { self.process_chain(input, output, num_frames) }
    }

    /// Chain-wide stabilization (C++ `DSP::prewarm` body, `dsp.cpp:66-101`):
    /// feeds zeroed samples through the whole chain until `num_samples`
    /// stabilized frames are consumed (C++ rounds the total up to whole
    /// `max_buffer_size` chunks; this implementation feeds the exact count —
    /// see the module divergence note).
    ///
    /// When no memory size was ever negotiated (`stage_frames` == 0), the
    /// chain first sizes its planes to the C++ fallback default
    /// (`NAM_DEFAULT_MAX_BUFFER_SIZE`).
    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        if self.stage_frames == 0 {
            // C++ `DSP::prewarm` (`dsp.cpp:67-70`): never-size models first
            // size themselves to `NAM_DEFAULT_MAX_BUFFER_SIZE`. Off-RT.
            self.resize_stage_buffers(DEFAULT_MAX_BUFFER_SIZE)
                .unwrap_or_else(|e| {
                    error!(
                        "[Sequential] stabilization skipped: stage buffer \
                            allocation failed ({e:?}) — cold-start transients will be audible"
                    );
                });
        }
        if num_samples == 0 {
            return;
        }
        // Buffer-sized chunks like C++ `dsp.cpp:96`, clamped to the reserved
        // plane length; chunk granularity does not change the stabilized
        // state (all feed sources are zeros, and every stage family is
        // causal state-equal under block splitting).
        let chunk = self.max_buffer_size.max(1).min(self.stage_frames.max(1));
        let mut fed = 0usize;
        while fed < num_samples {
            if self.stage_frames == 0 {
                break;
            }
            let take = num_samples.saturating_sub(fed).min(chunk);
            if take == 0 {
                break;
            }
            // SAFETY: stabilization planes sized for `take <= stage_frames`
            // elements per channel; internal feeds never alias user data.
            // Table casts are layout-identical (`usize` <-> raw pointer on
            // this engine's x86-64-v3 target).
            unsafe {
                self.process_chain(
                    self.zero_in_ptrs.as_ptr().cast::<*const f32>(),
                    self.sink_out_ptrs.as_ptr().cast::<*mut f32>(),
                    take,
                );
            }
            fed += take;
        }
    }

    /// Zero phase of the deferred split-stabilization flow for the chain:
    /// each stage clears its own temporal state (its family contracts), and
    /// the chain arms a single pending feed of
    /// [`prewarm_samples`](Self::prewarm_samples) frames that
    /// [`prewarm_step`](Self::prewarm_step) drains. Real-time safe.
    fn prewarm_reset(&mut self) {
        for model in &mut self.models {
            model.prewarm_reset();
        }
        self.prewarm_pending = self.prewarm_samples();
    }

    /// Advances the outstanding chain feed by at most `samples` zeroed
    /// frames through the whole chain (stage 0 zeros, last stage into the
    /// owned sink plane). Real-time safe.
    fn prewarm_step(&mut self, samples: usize) -> usize {
        let take = samples.min(self.prewarm_pending);
        if take == 0 {
            return self.prewarm_pending;
        }
        if self.stage_frames == 0 {
            // Pathological negotiation (`set_max_buffer_size(0)` without a
            // later resize): no capacity to feed; drop the accounting.
            self.prewarm_pending = 0;
            return 0;
        }
        // SAFETY: stabilization planes sized for `take <= stage_frames`
        // per channel; internal feeds never alias user data. Table casts are
        // layout-identical (`usize` <-> raw pointer on this engine's
        // x86-64-v3 target).
        unsafe {
            self.process_chain(
                self.zero_in_ptrs.as_ptr().cast::<*const f32>(),
                self.sink_out_ptrs.as_ptr().cast::<*mut f32>(),
                take,
            );
        }
        self.prewarm_pending -= take;
        self.prewarm_pending
    }

    /// Split-stabilization outstanding for the chain. `true` for a just-
    /// built or just-stabilized chain.
    fn prewarm_complete(&self) -> bool {
        self.prewarm_pending == 0
    }

    /// Whether the chain stabilization runs on reset (C++
    /// `GetPrewarmOnReset`).
    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    /// Sets the chain prewarm-on-reset flag and propagates to every stage
    /// (C++ `SequentialModel::SetPrewarmOnReset`, `sequential.cpp:182-187`).
    fn set_prewarm_on_reset(&mut self, val: bool) {
        self.prewarm_on_reset = val;
        for model in &mut self.models {
            model.set_prewarm_on_reset(val);
        }
    }

    /// Resets the chain (C++ `SequentialModel::Reset`, `sequential.cpp:152-180`):
    /// negotiate the maximum, disable every child prewarm flag, reset the
    /// stages in order, restore the flags exactly as saved, and — gated by
    /// the chain's own flag — run a single chain-wide stabilization pass
    /// (the stages receive their stabilization through the chain feed; child
    /// flags are restored after the reset and never consulted here).
    ///
    /// On stage reset failure the flags are restored before returning the
    /// error, mirroring the C++ try/catch restart guarantees.
    fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.set_max_buffer_size(max_buffer_size)?;

        let saved: Vec<bool> = self.models.iter().map(|m| m.prewarm_on_reset()).collect();
        for model in &mut self.models {
            model.set_prewarm_on_reset(false);
        }
        let mut child_failure: Option<anyhow::Error> = None;
        for model in &mut self.models {
            // C++ try/catch: the first failing child aborts the reset chain;
            // later stages are left untouched.
            let result = if child_failure.is_none() {
                model.reset(sample_rate, max_buffer_size)
            } else {
                Ok(())
            };
            if let Err(e) = result {
                error!(
                    "[Sequential] child reset failed: {e} — restoring \
                     child prewarm flags and aborting the chain reset"
                );
                child_failure = Some(e);
            }
        }
        for (model, state) in self.models.iter_mut().zip(saved) {
            model.set_prewarm_on_reset(state);
        }
        if let Some(e) = child_failure {
            return Err(e);
        }

        if self.prewarm_on_reset {
            self.prewarm(self.prewarm_samples());
        }
        Ok(())
    }

    /// Reallocates the chain stage planes for the new maximum block size
    /// (C++ `SequentialModel::SetMaxBufferSize`, `sequential.cpp:202-222`).
    /// Stage children are NOT resized — the C++ chain resizes them inside
    /// their own `Reset` calls only.
    fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        self.resize_stage_buffers(max_buf)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Saturating sum of the stage stabilization counts (C++
    /// `GetPrewarmSamples`, `sequential.cpp:189-199`).
    fn prewarm_samples(&self) -> usize {
        saturating_prewarm_sum(self.models.iter().map(|m| m.prewarm_samples()))
    }
}

impl SequentialModel {
    /// The chain hot path (C++ `SequentialModel::process`,
    /// `sequential.cpp:131-147`).
    ///
    /// `stage_in` provides `in_channels` pointers for stage 0 and `stage_out`
    /// receives the last stage's `out_channels` pointers. Number of frames is
    /// clamped to the negotiated maximum and the reserved plane length.
    ///
    /// # Safety
    /// Same contract as [`NamModel::process_raw`].
    unsafe fn process_chain(
        &mut self,
        stage_in: *const *const f32,
        stage_out: *const *mut f32,
        num_frames: usize,
    ) {
        debug_assert!(
            num_frames <= self.max_buffer_size,
            "block of {} frames exceeds the negotiated maximum {} \
             (upstream SequentialModel::process throws; release truncates \
             to the first `max_buffer_size` frames)",
            num_frames,
            self.max_buffer_size
        );
        let has_intermediate = self.models.len() > 1;
        let frame_limit = if has_intermediate {
            self.stage_frames
        } else {
            self.max_buffer_size
        };
        let n = num_frames.min(frame_limit);
        if n == 0 {
            return;
        }
        let stages = self.models.len();
        let mut stage_in = stage_in;
        for i in 0..stages {
            let stage_out = if i + 1 == stages {
                stage_out
            } else {
                // SAFETY: `stage_out_ptrs[i]` was rebuilt off-RT after the
                // last stage-buffer (re)allocation and holds
                // `num_output_channels(i)` valid writable pointers, each
                // covering `stage_frames >= n` samples. The `usize` table is
                // cast to a raw-pointer array view (layout-identical on this
                // engine's x86-64-v3 target).
                self.stage_out_ptrs[i].as_ptr().cast::<*mut f32>()
            };
            // SAFETY: `stage_in` either points into the caller's pointer
            // array (`i == 0`) or into the prebuilt `stage_in_ptrs[i - 1]`
            // const tables, both satisfying the `process_raw` safety contract
            // for `n` frames over the stage's declared channel geometry.
            unsafe { self.models[i].process_raw(stage_in, stage_out, n) };
            if i + 1 < stages {
                stage_in = self.stage_in_ptrs[i].as_ptr().cast::<*const f32>();
            }
        }
    }
}

#[cfg(test)]
#[path = "sequential_test.rs"]
mod tests;
