// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic LSTM layer with heap-allocated buffers for arbitrary dimensions.

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;

/// An individual LSTM model layer with runtime-sized buffers.
///
/// This is the dynamic counterpart to [`super::layer::LstmLayer`],
/// using [`AlignedVec`] instead of const-generic stack arrays so it
/// can accommodate any `input_size` / `hidden_size` combination.
pub struct LstmLayerDyn {
    /// Input dimension (typically 1 for NAM first layer,
    /// or `hidden_size` for subsequent stacked layers).
    pub input_size: usize,
    /// Number of LSTM hidden units.
    pub hidden_size: usize,
    /// Layer weights in Gate-Major layout.
    /// Flat buffer of `4 * (input_size + hidden_size) * hidden_size` elements.
    pub input_hidden_weights: AlignedVec<f32>,
    /// Layer linear biases (4 × hidden_size).
    pub bias: AlignedVec<f32>,
    /// Consolidated state buffer [Input | Hidden].
    /// Length = input_size + hidden_size.
    pub state: AlignedVec<f32>,
    /// LSTM cell state (C). Length = hidden_size.
    pub cell_state: AlignedVec<f32>,
    /// Kahan compensation shadow state for cell state accumulation.
    pub cell_error: AlignedVec<f32>,
    /// Intermediate gate activations (4 × hidden_size).
    pub gates: AlignedVec<f32>,
}

impl LstmLayerDyn {
    /// Creates a new zero-initialized dynamic LSTM layer.
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, NamErrorCode> {
        let ih = input_size + hidden_size;
        let h4 = 4 * hidden_size;
        let weights_len = 4 * ih * hidden_size;

        Ok(Self {
            input_size,
            hidden_size,
            input_hidden_weights: AlignedVec::new(weights_len, 0.0f32)?,
            bias: AlignedVec::new(h4, 0.0f32)?,
            state: AlignedVec::new(ih, 0.0f32)?,
            cell_state: AlignedVec::new(hidden_size, 0.0f32)?,
            cell_error: AlignedVec::new(hidden_size, 0.0f32)?,
            gates: AlignedVec::new(h4, 0.0f32)?,
        })
    }

    /// Returns a reference to the current hidden state.
    #[inline(always)]
    pub fn get_hidden_state(&self) -> &[f32] {
        &self.state[self.input_size..]
    }

    /// Resets only the input slot, preserving the hidden state and cell state.
    ///
    /// Used during prewarm to avoid discarding the initial `_xh` and `_c` states
    /// loaded from the NAM file.
    pub fn reset_input_slot(&mut self) {
        self.state[..self.input_size].fill(0.0);
    }

    /// Resets the internal states to zero.
    ///
    /// Important for 'clearing the model's memory' between different runs.
    pub fn reset_states(&mut self) {
        self.state.fill(0.0);
        self.cell_state.fill(0.0);
        self.cell_error.fill(0.0);
        self.gates.fill(0.0);
    }
}
