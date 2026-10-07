// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Linear Model — Finite Impulse Response (FIR) network architecture for NAM.
//!
//! The Linear architecture implements a simple linear filter: the output at each
//! time step is obtained by the dot product of the model weights with a window of
//! input history (receptive field), plus a scalar bias:
//!
//! `output = bias + dot(weights, history_window)`
//!
//! Weights are stored in **reversed** order (matching C++ `nam::Linear` internal
//! layout) so that a dot product with the oldest-to-newest history window yields
//! the FIR convolution directly. The input history is stored in a
//! `MirroredBuffer<f32>`, which provides branch-free, contiguous access via
//! mirrored memory mapping — eliminating ring-buffer wrap-around logic in the
//! audio hot-path.
//!
//! # C++ Parity
//! This implementation matches `NeuralAmpModelerCore/NAM/dsp.cpp:255-301`
//! exactly: JSON weights are reversed on construction, and the dot product is
//! computed with the oldest-to-newest history window plus the scalar bias,
//! without tanh or head_scale (those are exclusive to WaveNet).

use super::NamModel;
use super::linear_fft::LinearFftState;
use super::sealed;
use crate::common::diagnostics::NamErrorCode;
use crate::dsp::mirror_buf::MirroredBuffer;
use crate::loader::nam_json::{LinearImplementation, LinearTopology};
use crate::math::common::AlignedVec;
use log::warn;

pub(crate) mod multichannel;
pub use multichannel::{LinearMultichannel, MultichannelMode};

/// Runtime convolution mode for the Linear model.
///
/// Controls whether the model uses direct time-domain convolution or
/// zero-latency partitioned FFT (hybrid: direct head + FFT tail).
#[derive(Debug)]
pub enum LinearMode {
    /// Direct time-domain convolution — dot product over the full receptive field.
    Direct,
    /// FFT partitioned convolution with `LinearFftState` for the tail.
    Fft(Box<LinearFftState>),
}

/// Linear Model — lightweight FIR-based neural model.
///
/// This is the simplest NAM architecture: a single linear layer (dot product)
/// applied over the recent sample history with an optional scalar bias.
///
/// # RT-Safety
/// - Zero allocation on the hot-path (`process`).
/// - Uses `MirroredBuffer` for branch-free ring buffer access.
/// - No locks, no `unwrap()`, no I/O.
pub struct LinearModel {
    /// FIR filter weights stored in **reversed** order (matching C++ internal
    /// layout). JSON weights are reversed on construction, so that
    /// `dot(weights, oldest_to_newest_window)` produces the FIR convolution.
    /// 64-byte aligned for AVX2/AVX-512 SIMD loads.
    pub weights: AlignedVec<f32>,
    /// Scalar bias added after the dot product.
    pub bias: f32,
    /// Circular buffer of past input samples, backed by mirrored memory mapping
    /// for branch-free contiguous access across the wrap boundary.
    pub history: MirroredBuffer<f32>,
    /// Current write position in the `history` ring buffer (0..receptive_field-1).
    pub write_pos: usize,
    /// Number of input samples in the receptive field (= `weights.len()`).
    pub receptive_field: usize,
    /// Precalculated limit * 2 to avoid runtime multiplication overflow checks.
    double_limit: usize,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,
    /// Zeroed-sample stabilization work still pending for the deferred split
    /// pass armed by [`Self::prewarm_reset`](NamModel::prewarm_reset).
    /// Always `0` for freshly built models; the integral
    /// [`Self::prewarm`](NamModel::prewarm) / [`Self::reset`](NamModel::reset)
    /// paths neither consult nor alter it.
    pub prewarm_pending: usize,
    /// Convolution implementation mode as configured in the JSON.
    pub implementation: LinearImplementation,
    /// Runtime convolution mode — `Direct` or `Fft` with partitioned FFT state.
    pub mode: LinearMode,
    /// Number of input channels (1 for legacy mono).
    pub in_channels: usize,
    /// Number of output channels (1 for legacy mono).
    pub out_channels: usize,
    /// Output biases per channel.
    pub biases: Vec<f32>,
    /// Specialized multichannel processing engine (Some for multichannel, None for mono).
    pub multichannel: Option<Box<LinearMultichannel>>,
}

/// Minimum receptive field (taps) for auto-selecting FFT partitioned convolution.
///
/// Below this threshold, time-domain direct convolution is more efficient
/// due to FFT overhead.
pub(crate) const FFT_AUTO_THRESHOLD: usize = 256;

/// Largest power of two ≤ `n`.
pub(crate) const fn largest_power_of_two_le(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut v = n;
    let mut r = 1;
    while v > 1 {
        r <<= 1;
        v >>= 1;
    }
    r
}

/// Selects the partition size `P` for FFT hybrid convolution.
///
/// Returns the largest power of two ≤ `receptive_field / 2`, guaranteeing
/// that `2 * P ≤ receptive_field` — which ensures the `block_start`
/// subtraction never underflows in the hot-path.
pub(crate) fn select_partition_size(receptive_field: usize) -> usize {
    let max_p = receptive_field / 2;
    largest_power_of_two_le(max_p.max(1))
}

impl LinearModel {
    /// Creates a new LinearModel with the given weights, bias, and implementation.
    ///
    /// Weights are expected in **forward-time order** as stored in the `.nam`
    /// JSON (`w[0]` is the response at the current sample). They are reversed
    /// internally to match the C++ `nam::Linear` layout.
    ///
    /// `implementation` controls the convolution strategy (`Auto`, `Direct`, `Fft`)
    /// as configured in the model's JSON:
    /// - `Direct`: always uses time-domain dot product.
    /// - `Auto`: uses FFT when `receptive_field >= 256`, otherwise Direct.
    /// - `Fft`: uses FFT partitioned convolution; falls back to Direct with a
    ///   warning if the receptive field is too small (< 256).
    ///
    /// Allocates the `MirroredBuffer` for the input history. The buffer is
    /// initialized to zero (silence) by the operating system via `mmap`.
    ///
    /// # Errors
    /// Returns `std::io::Error` if the `MirroredBuffer` allocation fails
    /// (e.g., out of memory or virtual address space).
    pub fn new(
        weights: Vec<f32>,
        bias: f32,
        implementation: LinearImplementation,
    ) -> std::io::Result<Self> {
        let topo = LinearTopology {
            in_channels: 1,
            out_channels: 1,
            receptive_field: weights.len(),
            has_bias: bias != 0.0,
            implementation,
        };
        Self::new_with_topology(topo, weights, vec![bias])
    }

    /// Creates a new LinearModel configured with a given [`LinearTopology`],
    /// impulse response weights, and channel biases.
    ///
    /// Weights are expected in **forward-time order** per kernel as stored
    /// in the `.nam` JSON. Each kernel of length `receptive_field` is reversed
    /// internally to match the C++ `nam::Linear` layout (`linear.cpp:136-140, 201-210`).
    ///
    /// # Errors
    /// Returns `std::io::Error` if memory allocation fails for aligned weights
    /// or `MirroredBuffer`.
    pub fn new_with_topology(
        topo: LinearTopology,
        weights: Vec<f32>,
        biases: Vec<f32>,
    ) -> std::io::Result<Self> {
        let receptive_field = topo.receptive_field;
        let mode = if topo.in_channels != 1 || topo.out_channels != 1 {
            // Multichannel FFT state will be wired in NC-2.2; default to Direct.
            LinearMode::Direct
        } else {
            let kernel_slice = if weights.len() >= receptive_field {
                &weights[..receptive_field]
            } else {
                &weights[..]
            };
            Self::resolve_mode(topo.implementation, receptive_field, kernel_slice)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?
        };

        let multichannel = if topo.in_channels != 1 || topo.out_channels != 1 {
            Some(Box::new(LinearMultichannel::new(topo, &weights, &biases)?))
        } else {
            None
        };

        let mut aligned = AlignedVec::from_vec(weights)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?;

        if receptive_field > 0 {
            for chunk in aligned.chunks_exact_mut(receptive_field) {
                chunk.reverse();
            }
        }

        let history = MirroredBuffer::<f32>::new(receptive_field)?;
        let limit = history.size();
        let double_limit = limit.checked_mul(2).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Limit overflow")
        })?;

        let bias = biases.first().copied().unwrap_or(0.0);

        Ok(Self {
            weights: aligned,
            bias,
            history,
            write_pos: limit,
            receptive_field,
            double_limit,
            prewarm_on_reset: true,
            prewarm_pending: 0,
            implementation: topo.implementation,
            mode,
            in_channels: topo.in_channels,
            out_channels: topo.out_channels,
            biases,
            multichannel,
        })
    }

    /// Validates Linear topology channel and parameter counts against C++ NAMCore invariants.
    pub fn validate_parameters(
        in_channels: usize,
        out_channels: usize,
        receptive_field: usize,
        has_bias: bool,
        weights_len: usize,
    ) -> Result<LinearTopology, NamErrorCode> {
        let topo = LinearTopology {
            in_channels,
            out_channels,
            receptive_field,
            has_bias,
            implementation: LinearImplementation::default(),
        };
        topo.validate_channels()?;
        topo.validate_weights_count(weights_len)?;
        Ok(topo)
    }

    /// Resolves which convolution mode to use based on the requested
    /// implementation and the receptive field size.
    fn resolve_mode(
        implementation: LinearImplementation,
        receptive_field: usize,
        weights: &[f32],
    ) -> Result<LinearMode, NamErrorCode> {
        match implementation {
            LinearImplementation::Direct => Ok(LinearMode::Direct),
            LinearImplementation::Auto => {
                if receptive_field >= FFT_AUTO_THRESHOLD {
                    let p = select_partition_size(receptive_field);
                    if p < receptive_field {
                        return Ok(LinearMode::Fft(Box::new(LinearFftState::new(p, weights)?)));
                    }
                }
                Ok(LinearMode::Direct)
            }
            LinearImplementation::Fft => {
                if receptive_field < FFT_AUTO_THRESHOLD {
                    warn!(
                        "[Linear] Fft requested but receptive_field={receptive_field} < {FFT_AUTO_THRESHOLD} \
                         — falling back to Direct"
                    );
                    return Ok(LinearMode::Direct);
                }
                let p = select_partition_size(receptive_field);
                Ok(LinearMode::Fft(Box::new(LinearFftState::new(p, weights)?)))
            }
        }
    }
}

mod process;

impl sealed::Sealed for LinearModel {}

impl NamModel for LinearModel {
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // SAFETY: weights are 64-byte aligned (AlignedVec).
        unsafe { self.process(input, output) };
    }

    #[inline(always)]
    unsafe fn process_raw(
        &mut self,
        input: *const *const f32,
        output: *const *mut f32,
        num_frames: usize,
    ) {
        // SAFETY: Caller guarantees input and output pointers satisfy safety contract.
        unsafe { self.process_raw(input, output, num_frames) };
    }

    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        self.prewarm(num_samples);
    }

    /// Zero phase of the deferred split flow: zeroes history/FFT state in
    /// place and arms the pending unit. Real-time safe.
    ///
    /// Design note: the family is stateless-by-construction after the zeroing
    /// (FIR history + FFT state are deterministic functions of the input
    /// stream alone), so the zeroed-model state already equals the integral
    /// post-`prewarm` state — the integral `prewarm` is itself a pure zeroing
    /// with no feed (`prewarm_samples() == 0`). The pending unit exists only
    /// to satisfy the split-flow accounting (`prewarm_complete() == false`
    /// until stepped), and stepping it feeds zeros through the family's own
    /// `process` path exactly like any other input: subsequent live samples
    /// then behave as if the stream had those leading zeros. Callers that
    /// need integral-equivalent output must drain the feed before live audio
    /// (same rule as every other family); the zeroed state's feed-sensitivity
    /// is inherent to the convolution (verified: any nonzero feed length
    /// advances the delay line and changes subsequent output).
    fn prewarm_reset(&mut self) {
        // Zero phase: bit-exact with the integral `reset()` zeroing.
        self.reset(0, 0);
        // Arm exactly one stabilization unit; stepping it feeds a single zero
        // through the family's `process` path (accounting only).
        self.prewarm_pending = 1;
    }

    /// Advances the armed feed by up to `samples` zeros through the family's
    /// own `process` path; returns the work still pending.
    /// Real-time safe.
    fn prewarm_step(&mut self, samples: usize) -> usize {
        let n = samples.min(self.prewarm_pending);
        if n > 0 {
            const CHUNK: usize = 512;
            let zeros = [0.0f32; CHUNK];
            let mut sink = [0.0f32; CHUNK];
            let mut fed = 0usize;
            while fed < n {
                let take = (n - fed).min(CHUNK);
                // SAFETY: weights are 64-byte aligned (AlignedVec), same
                // precondition as the `process` trait entry above.
                unsafe { self.process(&zeros[..take], &mut sink[..take]) };
                fed += take;
            }
            self.prewarm_pending -= n;
        }
        self.prewarm_pending
    }

    /// Deferred pass pending? (`true` when nothing is armed/left.)
    fn prewarm_complete(&self) -> bool {
        self.prewarm_pending == 0
    }

    fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        // State cleanup is unconditional: the FIR history and the FFT tail must
        // always be silenced, regardless of `prewarm_on_reset`. The flag only
        // gates the optional priming pass afterwards.
        self.reset(sample_rate, max_buffer_size);
        if self.prewarm_on_reset {
            self.prewarm(max_buffer_size);
        }
        Ok(())
    }

    fn prewarm_samples(&self) -> usize {
        0
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    fn set_prewarm_on_reset(&mut self, val: bool) {
        self.prewarm_on_reset = val;
    }
}

#[cfg(test)]
#[path = "linear_test.rs"]
mod tests;
