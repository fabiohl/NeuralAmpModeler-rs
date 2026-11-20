// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic LSTM layer with heap-allocated buffers for arbitrary dimensions.

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;

use crate::loader::nam_json::validation::{MAX_LSTM_HIDDEN_SIZE, MAX_LSTM_IO_CHANNELS};

/// An individual LSTM model layer with runtime-sized buffers.
///
/// This is the dynamic counterpart to [`super::layer::LstmLayer`],
/// using [`AlignedVec`] instead of const-generic stack arrays so it
/// can accommodate any `input_size` / `hidden_size` combination.
pub struct LstmLayerDyn {
    /// Input dimension (typically 1 for NAM first layer,
    /// or `hidden_size` for subsequent stacked layers).
    pub(crate) input_size: usize,
    /// Number of LSTM hidden units.
    pub(crate) hidden_size: usize,
    /// Layer weights in Gate-Major layout.
    /// Flat buffer of `4 * (input_size + hidden_size) * hidden_size` elements.
    pub(crate) input_hidden_weights: AlignedVec<f32>,
    /// Layer linear biases (4 × hidden_size).
    pub(crate) bias: AlignedVec<f32>,
    /// Consolidated state buffer [Input | Hidden].
    /// Length = input_size + hidden_size.
    pub(crate) state: AlignedVec<f32>,
    /// LSTM cell state (C). Length = hidden_size.
    pub(crate) cell_state: AlignedVec<f32>,
    /// Kahan compensation shadow state for cell state accumulation.
    pub(crate) cell_error: AlignedVec<f32>,
    /// Intermediate gate activations (4 × hidden_size).
    pub(crate) gates: AlignedVec<f32>,
}

impl LstmLayerDyn {
    /// Creates a new zero-initialized dynamic LSTM layer.
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self, NamErrorCode> {
        let max_input_size = MAX_LSTM_IO_CHANNELS.max(MAX_LSTM_HIDDEN_SIZE);
        if input_size == 0
            || input_size > max_input_size
            || hidden_size == 0
            || hidden_size > MAX_LSTM_HIDDEN_SIZE
        {
            return Err(NamErrorCode::InvalidFeatureTopology);
        }

        let ih = input_size
            .checked_add(hidden_size)
            .ok_or(NamErrorCode::InvalidFeatureTopology)?;
        let h4 = hidden_size
            .checked_mul(4)
            .ok_or(NamErrorCode::InvalidFeatureTopology)?;
        let weights_len = ih
            .checked_mul(h4)
            .ok_or(NamErrorCode::InvalidFeatureTopology)?;

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

    /// Returns the input dimension.
    #[inline(always)]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the number of hidden units.
    #[inline(always)]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the gate-major weight matrix (`4 * (input_size + hidden_size) * hidden_size`).
    #[inline(always)]
    pub fn input_hidden_weights(&self) -> &[f32] {
        &self.input_hidden_weights
    }

    /// Returns the gate bias vector (`4 * hidden_size`).
    #[inline(always)]
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Returns the consolidated `[input | hidden]` state buffer.
    #[inline(always)]
    pub fn state(&self) -> &[f32] {
        &self.state
    }

    /// Returns the LSTM cell state vector (`hidden_size`).
    #[inline(always)]
    pub fn cell_state(&self) -> &[f32] {
        &self.cell_state
    }

    /// Returns the Kahan compensation shadow error vector (`hidden_size`).
    #[inline(always)]
    pub fn cell_error(&self) -> &[f32] {
        &self.cell_error
    }

    /// Returns the intermediate gate activation buffer (`4 * hidden_size`).
    #[inline(always)]
    pub fn gates(&self) -> &[f32] {
        &self.gates
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

    /// Replaces layer weights and biases after validating exact buffer lengths.
    pub fn set_weights(&mut self, weights: &[f32], bias: &[f32]) -> Result<(), NamErrorCode> {
        if weights.len() != self.input_hidden_weights.len() || bias.len() != self.bias.len() {
            return Err(NamErrorCode::WeightCountMismatch);
        }
        self.input_hidden_weights.copy_from_slice(weights);
        self.bias.copy_from_slice(bias);
        Ok(())
    }

    /// Replaces `[input | hidden]` and cell state buffers after validating exact lengths.
    pub fn set_states(&mut self, state: &[f32], cell_state: &[f32]) -> Result<(), NamErrorCode> {
        if state.len() != self.state.len() || cell_state.len() != self.cell_state.len() {
            return Err(NamErrorCode::WeightCountMismatch);
        }
        self.state.copy_from_slice(state);
        self.cell_state.copy_from_slice(cell_state);
        Ok(())
    }
}
