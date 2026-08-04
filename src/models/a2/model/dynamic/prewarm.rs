// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Pre-warm implementation for the dynamic A2 model.
//!
//! Zeroes internal buffers and feeds `receptive_field_size` frames
//! of silence through `process()` to reach the proper steady state.
//! When a condition_dsp sub-model is present, it is also pre-warmed.

use crate::models::NamModel;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

use super::super::a2_prewarm_common;
use super::WaveNetA2Dyn;

impl WaveNetA2Dyn {
    /// Pre-warms the model by filling the receptive field with silence.
    #[cold]
    pub fn prewarm(&mut self) {
        a2_prewarm_common(
            self.num_layers,
            self.receptive_field_size,
            &mut self.layer_buffers,
            &self.layer_ring_sizes,
            &mut self.layer_buffer_starts,
            &mut self.layer_in,
            &mut self.head_accum,
            &mut self.head_write_pos,
        );

        if self.has_weights() {
            // Prewarm the condition_dsp sub-model first (if present).
            if let Some(ref mut cond_dsp) = self.condition_dsp {
                cond_dsp.prewarm(0);
            }

            let prewarm_samples = self.receptive_field_size;
            let block = WAVENET_MAX_NUM_FRAMES;
            let zeros = [0.0f32; WAVENET_MAX_NUM_FRAMES];
            let mut discard = [0.0f32; WAVENET_MAX_NUM_FRAMES];
            let mut remaining = prewarm_samples;
            while remaining > 0 {
                let nf = remaining.min(block);
                self.process(&zeros[..nf], &mut discard[..nf]);
                remaining -= nf;
            }
        }
    }
}
