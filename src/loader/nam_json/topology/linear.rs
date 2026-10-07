// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of Linear topologies from model data.

use super::super::data::{LinearImplementation, NamModelData};
use super::super::validation::{MAX_LINEAR_CHANNELS, MAX_RECEPTIVE_FIELD};
use crate::common::diagnostics::NamErrorCode;

/// Detected Linear topology.
///
/// Contains channel configuration, receptive field, bias flag, and convolution
/// implementation mode for the FIR-based Linear architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearTopology {
    /// Number of input channels (defaults to 1 for legacy models).
    pub in_channels: usize,
    /// Number of output channels (defaults to 1 for legacy models).
    pub out_channels: usize,
    /// Number of FIR filter taps per kernel.
    pub receptive_field: usize,
    /// Whether bias term(s) are present.
    pub has_bias: bool,
    /// Convolution implementation mode (`Auto`, `Direct`, `Fft`).
    pub implementation: LinearImplementation,
}

impl LinearTopology {
    /// Number of impulse response kernels.
    ///
    /// For equal channel counts (`in == out`, including 1 -> 1 mono and N -> N),
    /// a single shared IR kernel is used.
    /// For 1 -> N or N -> 1, `max(in, out)` separate IR kernels are required.
    #[inline]
    pub fn num_kernels(&self) -> usize {
        if self.in_channels == self.out_channels {
            1
        } else {
            self.in_channels.max(self.out_channels)
        }
    }

    /// Number of bias parameters in the weight layout if bias is enabled.
    ///
    /// For equal channel counts (`in == out`), a single shared bias is used.
    /// Otherwise, `out_channels` separate bias scalars are present.
    #[inline]
    pub fn num_biases(&self) -> usize {
        if self.in_channels == self.out_channels {
            1
        } else {
            self.out_channels
        }
    }

    /// Validates the channel geometry according to C++ NAMCore invariants:
    /// - Channels must be positive (> 0) and <= [`MAX_LINEAR_CHANNELS`].
    /// - `in_channels == out_channels`, `in_channels == 1`, or `out_channels == 1`.
    pub fn validate_channels(&self) -> Result<(), NamErrorCode> {
        if self.in_channels == 0
            || self.out_channels == 0
            || self.in_channels > MAX_LINEAR_CHANNELS
            || self.out_channels > MAX_LINEAR_CHANNELS
            || (self.in_channels != self.out_channels
                && self.in_channels != 1
                && self.out_channels != 1)
        {
            return Err(NamErrorCode::LinearInvalidChannels);
        }
        Ok(())
    }

    /// Calculates expected total weights count:
    /// `receptive_field * num_kernels + (if has_bias { num_biases } else { 0 })`
    /// using checked arithmetic against overflow.
    pub fn expected_weights(&self) -> Result<usize, NamErrorCode> {
        let kernel_weights = self
            .receptive_field
            .checked_mul(self.num_kernels())
            .ok_or(NamErrorCode::LinearWeightCountMismatch)?;
        if self.has_bias {
            kernel_weights
                .checked_add(self.num_biases())
                .ok_or(NamErrorCode::LinearWeightCountMismatch)
        } else {
            Ok(kernel_weights)
        }
    }

    /// Validates the provided weight count against the expected count.
    pub fn validate_weights_count(&self, actual_count: usize) -> Result<(), NamErrorCode> {
        let expected = self.expected_weights()?;
        if actual_count != expected {
            return Err(NamErrorCode::LinearWeightCountMismatch);
        }
        Ok(())
    }
}

/// Checks and returns the Linear topology from model data.
///
/// Returns `None` if the architecture is not `"Linear"`, if `receptive_field`
/// is missing or exceeds [`MAX_RECEPTIVE_FIELD`].
pub fn get_linear_topology(data: &NamModelData) -> Option<LinearTopology> {
    if data.architecture != "Linear" {
        return None;
    }

    let receptive_field = data.config.receptive_field?;
    if receptive_field > MAX_RECEPTIVE_FIELD {
        log::warn!(
            "Linear receptive_field ({receptive_field}) exceeds maximum \
             {MAX_RECEPTIVE_FIELD} — OOM/DoS protection"
        );
        return None;
    }
    let in_channels = data.config.in_channels.unwrap_or(1);
    let out_channels = data.config.out_channels.unwrap_or(1);
    let has_bias = data.config.bias.unwrap_or(false);
    let implementation = data
        .config
        .implementation
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();

    Some(LinearTopology {
        in_channels,
        out_channels,
        receptive_field,
        has_bias,
        implementation,
    })
}
