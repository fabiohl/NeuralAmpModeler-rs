// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Multichannel convolution implementation mode.

use crate::models::linear_fft::LinearFftState;

/// Multichannel convolution implementation mode.
#[derive(Debug)]
pub enum MultichannelMode {
    /// Time-domain direct convolution (dot product over full receptive field).
    Direct,
    /// Frequency-domain partitioned convolution with per-path `LinearFftState`.
    Fft(Vec<LinearFftState>),
}
