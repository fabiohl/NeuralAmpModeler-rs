// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Specialized multichannel processing architectures for the Linear FIR/FFT model.
//!
//! Provides dedicated zero-allocation paths for non-mono Linear topologies:
//! - [`LinearOneToMany`]: 1 input channel convolved with $N$ independent kernels producing
//!   $N$ output channels ($1 \to N$).
//! - [`LinearManyToOne`]: $N$ input channels convolved with $N$ independent kernels and
//!   summed in strictly ascending channel index order into 1 output channel ($N \to 1$).
//! - [`LinearManyToManyShared`]: $N$ input channels convolved with a single shared kernel
//!   producing $N$ output channels ($N \to N$).
//!
//! # In-Place Processing Guarantee
//!
//! In digital audio workstations and low-latency hosts, input and output buffer pointers
//! frequently alias (`input == output`). All variants in this module guarantee complete
//! in-place safety without temporary heap allocations:
//!
//! For each sample frame $i \in 0..\text{num\_frames}$:
//! 1. All input samples `*input[ch].add(i)` are read and copied into their corresponding
//!    `MirroredBuffer` ring buffer(s) *prior* to calculating or writing sample $i$ to any
//!    output channel pointer.
//! 2. Future samples ($i+1, \dots$) remain untouched in host buffers, and past history is
//!    accessed solely through the internal `MirroredBuffer` memory mapping.
//! 3. This eliminates read-after-write corruption completely across all channel configurations.
//!
//! # Summation Order for $N \to 1$
//!
//! In $N \to 1$ topologies, channel outputs are accumulated into the single output channel.
//! Due to the non-associativity of floating-point arithmetic, accumulation order directly
//! impacts numerical output and ESR:
//!
//! `acc = bias[0] + (w_0 * x_0) + (w_1 * x_1) + ... + (w_{N-1} * x_{N-1})`
//!
//! The accumulator is initialized with `bias[0]`, then channels are accumulated in strictly
//! ascending order ($ch = 0, 1, \dots, N-1$). This matches C++ NAMCore `linear.cpp:328-342`
//! bit-identically.

pub mod many_to_many;
pub mod many_to_one;
pub mod mode;
pub mod one_to_many;

pub use many_to_many::LinearManyToManyShared;
pub use many_to_one::LinearManyToOne;
pub use mode::MultichannelMode;
pub use one_to_many::LinearOneToMany;

use crate::dsp::mirror_buf::MirroredBuffer;
use crate::loader::nam_json::LinearTopology;

/// Shared delay-line bound used by the multichannel safe boundary.
///
/// Returns whether `write_pos` can accept a sample and still expose the full
/// `receptive_field` history window without leaving the mirrored view.
pub(crate) fn history_is_sound(
    history: &MirroredBuffer<f32>,
    write_pos: usize,
    receptive_field: usize,
    double_limit: usize,
) -> bool {
    let size = history.size();
    if receptive_field == 0 || size < receptive_field {
        return false;
    }
    let Some(total) = size.checked_mul(2) else {
        return false;
    };
    double_limit == total && write_pos >= receptive_field && write_pos < total
}

/// Unified multichannel dispatcher enum for the Linear model.
pub enum LinearMultichannel {
    /// 1 -> N multichannel model.
    OneToMany(LinearOneToMany),
    /// N -> 1 multichannel model.
    ManyToOne(LinearManyToOne),
    /// N -> N multichannel model with shared kernel.
    ManyToManyShared(LinearManyToManyShared),
}

impl LinearMultichannel {
    /// Number of input channels declared by the selected engine.
    #[inline]
    pub fn in_channels(&self) -> usize {
        match self {
            Self::OneToMany(_) => 1,
            Self::ManyToOne(m) => m.in_channels,
            Self::ManyToManyShared(m) => m.channels,
        }
    }

    /// Number of output channels declared by the selected engine.
    #[inline]
    pub fn out_channels(&self) -> usize {
        match self {
            Self::OneToMany(m) => m.out_channels,
            Self::ManyToOne(_) => 1,
            Self::ManyToManyShared(m) => m.channels,
        }
    }

    /// Returns whether the selected variant can safely reach its unchecked
    /// kernels.
    pub fn is_sound(&self) -> bool {
        match self {
            Self::OneToMany(m) => m.is_sound(),
            Self::ManyToOne(m) => m.is_sound(),
            Self::ManyToManyShared(m) => m.is_sound(),
        }
    }

    /// Constructs a specialized multichannel Linear model according to `topo`.
    pub fn new(topo: LinearTopology, weights: &[f32], biases: &[f32]) -> std::io::Result<Self> {
        let in_ch = topo.in_channels;
        let out_ch = topo.out_channels;

        if in_ch == 1 && out_ch > 1 {
            Ok(Self::OneToMany(LinearOneToMany::new(
                topo, weights, biases,
            )?))
        } else if in_ch > 1 && out_ch == 1 {
            Ok(Self::ManyToOne(LinearManyToOne::new(
                topo, weights, biases,
            )?))
        } else if in_ch > 1 && in_ch == out_ch {
            Ok(Self::ManyToManyShared(LinearManyToManyShared::new(
                topo, weights, biases,
            )?))
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Unsupported Linear multichannel geometry",
            ))
        }
    }

    /// Resets internal history and FFT tail state.
    pub fn reset(&mut self) {
        match self {
            Self::OneToMany(m) => m.reset(),
            Self::ManyToOne(m) => m.reset(),
            Self::ManyToManyShared(m) => m.reset(),
        }
    }

    /// Prewarms internal state.
    pub fn prewarm(&mut self, _num_samples: usize) {
        self.reset();
    }

    /// Processes audio samples via raw channel pointers.
    ///
    /// # Safety
    /// `input` and `output` must point to valid arrays of channel pointers with at least
    /// `num_frames` valid samples per channel. In-place processing is fully supported.
    #[inline(always)]
    pub unsafe fn process_raw(
        &mut self,
        input: *const *const f32,
        output: *const *mut f32,
        num_frames: usize,
    ) {
        match self {
            // SAFETY: Caller guarantees input and output arrays and buffers are valid for num_frames.
            Self::OneToMany(m) => unsafe { m.process_raw(input, output, num_frames) },
            // SAFETY: Caller guarantees input and output arrays and buffers are valid for num_frames.
            Self::ManyToOne(m) => unsafe { m.process_raw(input, output, num_frames) },
            // SAFETY: Caller guarantees input and output arrays and buffers are valid for num_frames.
            Self::ManyToManyShared(m) => unsafe { m.process_raw(input, output, num_frames) },
        }
    }
}
