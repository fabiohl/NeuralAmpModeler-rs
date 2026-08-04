// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Prewarm logic for LSTM models — trait + common implementation.

use super::LstmModel1;
use super::LstmModel2;
use super::LstmModelDyn;
use super::NamModel;

/// Internal trait to unify models that have resettable LSTM state.
pub(super) trait LstmLike: NamModel {
    fn reset_input_slots(&mut self);
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> LstmLike
    for LstmModel1<H, H1_IH, H_H4>
{
    fn reset_input_slots(&mut self) {
        self.layer.reset_input_slot();
    }
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> LstmLike
    for LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    fn reset_input_slots(&mut self) {
        self.layer1.reset_input_slot();
        self.layer2.reset_input_slot();
    }
}

impl LstmLike for LstmModelDyn {
    fn reset_input_slots(&mut self) {
        self.reset_input_slots();
    }
}

/// Generic prewarm implementation for LSTM-based models.
/// Zeros only the input slots, preserving the hidden and cell states
/// loaded from the NAM file (`_xh` and `_c`), and processes silence for stabilization.
pub(super) fn lstm_prewarm_common(model: &mut impl LstmLike, num_samples: usize) {
    // 1. Zero only each layer's input slot, preserving _xh and _c from the file.
    model.reset_input_slots();

    // 2. Process zero-value samples.
    const CHUNK: usize = 512;
    let zero_in = [0.0f32; CHUNK];
    let mut zero_out = [0.0f32; CHUNK];
    let mut rem = num_samples;

    while rem > 0 {
        let n = rem.min(CHUNK);
        model.process(&zero_in[..n], &mut zero_out[..n]);
        rem -= n;
    }
}
