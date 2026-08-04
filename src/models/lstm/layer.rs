// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized Recurrent Cell Mesh (LSTM) for NAM inference.

use crate::math::common::Aligned64;

/// An individual LSTM model layer optimized for SIMD.
pub struct LstmLayer<const I: usize, const H: usize, const IH: usize, const H4: usize> {
    /// Layer weights in Gate-Major layout.
    pub input_hidden_weights: Aligned64<[[[f32; H]; IH]; 4]>,
    /// Layer linear biases.
    pub bias: Aligned64<[f32; H4]>,
    /// Consolidated state buffer [Input | Hidden].
    pub state: Aligned64<[f32; IH]>,
    /// LSTM cell state (C).
    pub cell_state: Aligned64<[f32; H]>,
    /// Kahan compensation shadow state for cell state accumulation.
    pub cell_error: Aligned64<[f32; H]>,
    /// Intermediate gate activations.
    pub gates: Aligned64<[f32; H4]>,
}

impl<const I: usize, const H: usize, const IH: usize, const H4: usize> LstmLayer<I, H, IH, H4> {
    /// Creates a new zero-initialized LSTM layer.
    pub fn new() -> Self {
        Self {
            input_hidden_weights: Aligned64([[[0.0f32; H]; IH]; 4]),
            bias: Aligned64([0.0; H4]),
            state: Aligned64([0.0; IH]),
            cell_state: Aligned64([0.0; H]),
            cell_error: Aligned64([0.0; H]),
            gates: Aligned64([0.0; H4]),
        }
    }
    /// Returns a reference to the current hidden state.
    #[inline(always)]
    pub fn get_hidden_state(&self) -> &[f32] {
        &self.state[I..]
    }
    /// Resets only the input slot, preserving the hidden state and cell state.
    ///
    /// Used during prewarm to avoid discarding the initial `_xh` and `_c` states
    /// loaded from the NAM file.
    pub fn reset_input_slot(&mut self) {
        self.state[..I].fill(0.0);
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

impl<const I: usize, const H: usize, const IH: usize, const H4: usize> Default
    for LstmLayer<I, H, IH, H4>
{
    fn default() -> Self {
        Self::new()
    }
}
