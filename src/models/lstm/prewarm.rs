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
    /// Full recurrent-state clearing (hidden, cell, Kahan shadow, gates).
    fn reset_states_full(&mut self);
    /// Accessor for the deferred-stabilization pending counter, so the
    /// common split-pass helpers can arm and drain it in place.
    fn prewarm_pending_slot(&mut self) -> &mut usize;
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> LstmLike
    for LstmModel1<H, H1_IH, H_H4>
{
    fn reset_input_slots(&mut self) {
        self.layer.reset_input_slot();
    }

    fn reset_states_full(&mut self) {
        self.layer.reset_states();
    }

    fn prewarm_pending_slot(&mut self) -> &mut usize {
        &mut self.prewarm_pending
    }
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> LstmLike
    for LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    fn reset_input_slots(&mut self) {
        self.layer1.reset_input_slot();
        self.layer2.reset_input_slot();
    }

    fn reset_states_full(&mut self) {
        self.layer1.reset_states();
        self.layer2.reset_states();
    }

    fn prewarm_pending_slot(&mut self) -> &mut usize {
        &mut self.prewarm_pending
    }
}

impl LstmLike for LstmModelDyn {
    fn reset_input_slots(&mut self) {
        self.reset_input_slots();
    }

    fn reset_states_full(&mut self) {
        self.reset_states();
    }

    fn prewarm_pending_slot(&mut self) -> &mut usize {
        &mut self.prewarm_pending
    }
}

// Generic prewarm implementation for LSTM-based models.
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

/// Zero phase of the deferred split stabilization pass for LSTM models.
///
/// Clears the full recurrent state (the same zero phase the integral
/// [`NamModel::reset`](NamModel::reset) applies) and arms the pending
/// zeroed-sample budget; the silence run then proceeds through
/// [`NamModel::prewarm_step`](NamModel::prewarm_step) in caller-chosen
/// chunks. Total work equals the integral reset's zero phase plus its
/// stabilization feed, reproduced bit-exactly and amortizable.
pub(super) fn lstm_prewarm_split_reset(model: &mut impl LstmLike) {
    // Zero phase 1: full recurrent-state clearing (hidden, cell, Kahan
    // shadow, gates) — identical to the integral reset.
    model.reset_states_full();

    // Zero phase 2 (integral reset continues into its stabilization prime):
    // zero the input slots, then arm the pending budget.
    model.reset_input_slots();
    *model.prewarm_pending_slot() = model.prewarm_samples();
}

/// Advances the deferred split stabilization for LSTM models by at most
/// `samples` zeroed samples, returning the work still pending.
pub(super) fn lstm_prewarm_split_step(model: &mut impl LstmLike, samples: usize) -> usize {
    let pending = *model.prewarm_pending_slot();
    let n = samples.min(pending);
    if n > 0 {
        const CHUNK: usize = 512;
        let zero_in = [0.0f32; CHUNK];
        let mut zero_out = [0.0f32; CHUNK];
        let mut rem = n;

        while rem > 0 {
            let take = rem.min(CHUNK);
            model.process(&zero_in[..take], &mut zero_out[..take]);
            rem -= take;
        }
        *model.prewarm_pending_slot() = pending - n;
    }
    *model.prewarm_pending_slot()
}
